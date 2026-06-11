from pathlib import Path
import logging
import os
import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt
from robotathome import RobotAtHome
from tqdm import tqdm
import argparse

from shared.utils import ensure_dir, plot_image, plot_mask_overlay, plot_yolo_bboxes, align_all_masks_image, \
    save_json, load_json


LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)
if not logger.handlers:
    file_handler = logging.FileHandler(LOG_DIR / "annotation_convert.log", encoding="utf-8")
    file_handler.setLevel(logging.WARNING)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(file_handler)

logger.setLevel(logging.WARNING)
logger.propagate = False


def prepare_binary_mask(mask):
    """Convert mask to 2D binary uint8."""
    if mask is None:
        return None

    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[..., 0]

    m = (m > 0).astype(np.uint8)
    return m if m.sum() > 0 else None


def mask_to_yolo_polygon(mask, class_id, img_w, img_h, epsilon_ratio=0.002):
    """Convert one aligned object mask to one YOLO segmentation line."""
    m = prepare_binary_mask(mask)
    if m is None:
        return None

    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 1:
        return None

    epsilon = epsilon_ratio * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    points = approx.reshape(-1, 2)

    if len(points) < 3:
        return None

    coords = []
    for x, y in points:
        coords.append(x / float(img_w))
        coords.append(y / float(img_h))

    return f"{int(class_id)} " + " ".join(f"{v:.6f}" for v in coords)


def get_yolo_lines_for_observation(rh_db, obs_id, epsilon_ratio):
    """
    Fetches labels, aligns masks to image, and returns YOLO lines for an obs_id.
    """
    # 1. Fetch files and image
    try:
        rgb_path, _ = rh_db.get_RGBD_files(obs_id)
        image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if image is None:
            return None, None, []
        # Check alignment: mask is 320x240, so we expect h=320, w=240
        # If your image is 240 height and 320 width , you MUST rotate
        img_h, img_w = image.shape[:2]
        if img_h < img_w:
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            img_h, img_w= image.shape[:2]  # Now h=240, w=320

        # 2. Get labels and align masks
        labels_with_masks = rh_db.get_RGBD_labels(obs_id)
        if labels_with_masks.empty:
            return image, rgb_path, []

        aligned_masks = align_all_masks_image(labels_with_masks["mask"], image)

        # 3. Convert aligned masks to YOLO format
        label_lines = []
        for (_, row), aligned_mask in zip(labels_with_masks.iterrows(), aligned_masks):
            line = mask_to_yolo_polygon(
                mask=aligned_mask,
                class_id=row["object_type_id"],
                img_w=img_w,
                img_h=img_h,
                epsilon_ratio=epsilon_ratio
            )
            if line:
                label_lines.append(line)
    except Exception as e:
        logger.warning("Failed to get RGBD files for obs_id=%s: %s", obs_id, e)
        rgb_path = None
        image = None
        label_lines = []

    return image, rgb_path, label_lines


def get_expected_output_paths(rh_db, output_root, rgbd_root):
    """Return expected YOLO image/label output paths via a single bulk DB query.

    Replaces the previous per-observation get_RGBD_files() loop (one SQL query
    per obs_id) with one bulk query over all labeled observations in the DB,
    which is orders of magnitude faster.

    Bulk query data flow:
    - SQL returns one row per labeled RGBD observation with:
      id, local_path, rgb_file
    - local_path + rgb_file are expanded into an absolute RGB source path
    - each source path is converted into one expected YOLO output pair:
      (images/.../<obs_id>.jpg, labels/.../<obs_id>.txt)
    - the returned list is consumed by conversion_outputs_exist(), which only
      checks whether those expected output files already exist on disk.
    """
    output_root = Path(output_root)
    images_dir = output_root / "images"
    labels_dir = output_root / "labels"
    rgbd_root = Path(rgbd_root)

    # Single bulk query: join observations with their file-path records.
    # Query output columns are used as follows:
    # - id -> output filenames (<id>.jpg / <id>.txt)
    # - local_path + rgb_file -> absolute RGB source path under rh_db rgbd root
    # The loop below converts those source paths into relative output folders.
    sql = """
        SELECT
            f.id AS id,
            f.new_path  AS local_path,
            f.new_file_2 AS rgb_file
        FROM (
            SELECT DISTINCT sensor_observation_id AS id
            FROM rh_lblrgbd_labels
        ) AS l
        JOIN rh2_old2new_rgbd_files AS f ON f.id = l.id
        ORDER BY l.id
    """
    try:
        files_df = rh_db.query(sql)
    except Exception as e:
        logger.warning("Bulk file-path query failed: %s", e)
        return []

    rgbd_path_str = str(rh_db._RobotAtHome__rgbd_path)

    expected_paths = []
    skipped = 0
    for row in files_df.itertuples(index=False):
        rgb_full = os.path.join(rgbd_path_str, row.local_path, row.rgb_file)
        try:
            relative_dir = Path(rgb_full).parent.relative_to(rgbd_root)
        except ValueError as e:
            logger.warning("Skipping obs_id=%s — path not under rgbd_root: %s", row.id, e)
            skipped += 1
            continue
        expected_paths.append(
            (
                images_dir / relative_dir / f"{row.id}.jpg",
                labels_dir / relative_dir / f"{row.id}.txt",
            )
        )

    if skipped:
        logger.warning("%d observations skipped (path not under rgbd_root).", skipped)
    print(f"Resolved {len(expected_paths)} expected output paths "
          f"({skipped} skipped) from {len(files_df)} DB records.")
    return expected_paths


def conversion_outputs_exist(rh_db, output_root, rgbd_root):
    """Check YOLO image/label outputs and return a summary for conversion decisions."""
    print(f"Checking existing YOLO conversion outputs under {Path(output_root).resolve()}...")
    print("Resolving expected output paths...")
    expected_paths = get_expected_output_paths(rh_db, output_root, rgbd_root)
    # If we cannot resolve any expected outputs, treat this as a fresh run candidate.
    if not expected_paths:
        print("No expected YOLO outputs could be determined. Conversion will run.")
        return {
            "total_pairs": 0,
            "existing_pairs": 0,
            "missing_pairs": 0,
            "all_exist": False,
            "none_exist": True,
        }

    missing_count = 0
    print("Checking converted image/label file existence...")
    for image_path, label_path in tqdm(
        expected_paths,
        total=len(expected_paths),
        desc="Checking converted outputs",
        unit="obs",
        dynamic_ncols=True,
        file=sys.stdout,
    ):
        if not image_path.exists() or not label_path.exists():
            missing_count += 1

    total_pairs = len(expected_paths)
    existing_pairs = total_pairs - missing_count
    all_exist = missing_count == 0
    print(
        f"Finished checking converted outputs. Existing pairs: {existing_pairs}, "
        f"Missing pairs: {missing_count}, Total pairs: {total_pairs}"
    )
    return {
        "total_pairs": total_pairs,
        "existing_pairs": existing_pairs,
        "missing_pairs": missing_count,
        "all_exist": all_exist,
        "none_exist": existing_pairs == 0,
    }


def mapping_json_exists(mapping_json):
    """Log mapping JSON existence check and return whether it exists."""
    mapping_json = Path(mapping_json)
    print(f"Checking mapping JSON at {mapping_json}...")
    exists = mapping_json.exists()
    if exists:
        print(f"Mapping JSON exists at {mapping_json}.")
    else:
        print(f"Mapping JSON does not exist at {mapping_json}.")
    return exists

def _bulk_fetch_file_paths(rh_db):
    """Fetch all RGB and labels.txt source paths for labeled RGBD observations.

    Bulk query output:
    - one row per observation with columns:
      id, local_path, rgb_file, labels_file

    Returned structure:
    - DataFrame indexed by observation id with columns:
      rgb_full, labels_full

    Downstream usage:
    - convert_df_to_yolo_seg() uses this DataFrame for O(1) lookup of the two
      filesystem paths needed for each obs_id, avoiding repeated DB calls to
      get_RGBD_files() and __get_Labels_file().
    """
    # Query result contains path fragments from rh2_old2new_rgbd_files.
    # Those fragments are immediately expanded into absolute paths under the
    # RobotAtHome RGBD root before returning the indexed DataFrame.
    sql = """
        SELECT
            f.id AS id,
            new_path     AS local_path,
            new_file_2   AS rgb_file,
            new_file_3   AS labels_file
        FROM (
            SELECT DISTINCT sensor_observation_id AS id
            FROM rh_lblrgbd_labels
        ) AS l
        JOIN rh2_old2new_rgbd_files AS f ON f.id = l.id
        ORDER BY l.id
    """
    files_df = rh_db.query(sql)
    rgbd_base = str(rh_db._RobotAtHome__rgbd_path)
    files_df["rgb_full"]    = files_df.apply(lambda r: os.path.join(rgbd_base, r.local_path, r.rgb_file),    axis=1)
    files_df["labels_full"] = files_df.apply(lambda r: os.path.join(rgbd_base, r.local_path, r.labels_file), axis=1)
    return files_df.set_index("id")[["rgb_full", "labels_full"]]


def _bulk_fetch_labels(rh_db):
    """Fetch all DB label rows for labeled RGBD observations in one query.

    Bulk query output:
    - one row per object label with columns:
      id, local_id, object_type_id, sensor_observation_id

    Returned structure:
    - dict[int, DataFrame] grouped by sensor_observation_id

    Downstream usage:
    - convert_df_to_yolo_seg() retrieves labels_by_obs[obs_id], uses local_id
      to reconstruct per-label masks from the labels.txt mask file, and uses
      object_type_id as the initial YOLO class id written to label files.
    """
    # Query all labels once, then group in memory by sensor_observation_id so
    # the conversion loop can do observation-level lookups without more SQL.
    sql = """
        SELECT id, local_id, object_type_id, sensor_observation_id
        FROM rh_lblrgbd_labels
    """
    labels_df = rh_db.query(sql)
    return {
        obs_id: group.reset_index(drop=True)
        for obs_id, group in labels_df.groupby("sensor_observation_id")
    }


def convert_df_to_yolo_seg(rh_db, output_root, rgbd_root, epsilon_ratio=0.002):
    """
    Export Robot@Home annotations to YOLO segmentation dataset.

    Optimized: replaces per-observation SQL calls (3 queries/obs) with two
    bulk queries upfront. Only unavoidable per-observation disk I/O (image +
    mask file reads) and CPU work remain in the loop.

    Args:
        rh_db:        Loaded Robot@Home database object.
        output_root:  Output directory where images/ and labels/ will be created.
        rgbd_root:    Root path used to compute relative output sub-directories.
        epsilon_ratio: Polygon simplification factor for cv2.approxPolyDP().
    """
    output_root = Path(output_root)
    images_dir  = output_root / "images"
    labels_dir  = output_root / "labels"

    rgbd_root   = Path(rgbd_root)

    print("Pre-fetching file paths from DB (bulk)...")
    files_index = _bulk_fetch_file_paths(rh_db)      # id -> (rgb_full, labels_full)

    print("Pre-fetching label rows from DB (bulk)...")
    labels_by_obs = _bulk_fetch_labels(rh_db)         # obs_id -> DataFrame

    # These two bulk query results are joined in memory by obs_id:
    # - files_index provides absolute RGB image and labels.txt paths
    # - labels_by_obs provides the DB rows for all labels of that observation
    # The per-observation loop then performs only unavoidable file I/O and mask
    # processing to emit final YOLO image/label outputs.

    obs_ids = files_index.index.tolist()
    skipped_count   = 0
    converted_count = 0

    progress_bar = tqdm(obs_ids, total=len(obs_ids),
                        desc="Converting observations", unit="obs", file=sys.stdout)
    progress_bar.set_postfix(converted=converted_count, skipped=skipped_count)

    for obs_id in progress_bar:
        # ── 1. Resolve paths from pre-fetched index ────────────────────────────
        try:
            rgb_path    = files_index.at[obs_id, "rgb_full"]
            labels_file = files_index.at[obs_id, "labels_full"]
            relative_dir = Path(rgb_path).parent.relative_to(rgbd_root)
        except (KeyError, ValueError) as e:
            logger.warning("obs_id=%s: cannot resolve paths — %s", obs_id, e)
            skipped_count += 1
            progress_bar.set_postfix(converted=converted_count, skipped=skipped_count)
            continue

        # ── 2. Load image (disk I/O — unavoidable per-obs) ────────────────────
        image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if image is None:
            skipped_count += 1
            progress_bar.set_postfix(converted=converted_count, skipped=skipped_count)
            continue

        img_h, img_w = image.shape[:2]
        if img_h < img_w:
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            img_h, img_w = image.shape[:2]

        # ── 3. Build label mask array from pre-fetched labels + mask file ─────
        labels_df = labels_by_obs.get(obs_id)
        if labels_df is None or labels_df.empty:
            # No labels for this obs; write empty label file and save image.
            label_lines = []
        else:
            try:
                label_mask_array = rh_db._RobotAtHome__get_label_mask_array(labels_file)
                mask_col         = rh_db._RobotAtHome__decompose_label_mask_array(
                                       label_mask_array, labels_df["local_id"])
                labels_with_masks = labels_df.copy()
                labels_with_masks["mask"] = mask_col["mask"].values

                aligned_masks = align_all_masks_image(labels_with_masks["mask"], image)

                label_lines = []
                for (_, row), aligned_mask in zip(labels_with_masks.iterrows(), aligned_masks):
                    line = mask_to_yolo_polygon(
                        mask=aligned_mask,
                        class_id=row["object_type_id"],
                        img_w=img_w,
                        img_h=img_h,
                        epsilon_ratio=epsilon_ratio,
                    )
                    if line:
                        label_lines.append(line)
            except Exception as e:
                logger.warning("obs_id=%s: mask/polygon error — %s", obs_id, e)
                skipped_count += 1
                progress_bar.set_postfix(converted=converted_count, skipped=skipped_count)
                continue

        # ── 4. Write outputs ───────────────────────────────────────────────────
        final_img_dir   = images_dir / relative_dir
        final_label_dir = labels_dir  / relative_dir
        ensure_dir(final_img_dir)
        ensure_dir(final_label_dir)

        cv2.imwrite(str(final_img_dir / f"{obs_id}.jpg"), image)
        with open(final_label_dir / f"{obs_id}.txt", "w", encoding="utf-8") as f:
            if label_lines:
                f.write("\n".join(label_lines) + "\n")

        converted_count += 1
        progress_bar.set_postfix(converted=converted_count, skipped=skipped_count)

    print(
        f"Finished converting observations. "
        f"Converted: {converted_count}, Skipped: {skipped_count}, Total: {len(obs_ids)}"
    )

def replace_class_ids_with_names(label_lines, rh_db):
    """
    Replace first token in each YOLO label line from class id to class name.

    Args:
        label_lines: list[str]
            Example:
            ["5 0.12 0.34 0.56 0.78", "7 0.11 0.22 0.33 0.44"]

        class_id_to_name: dict
            Example:
            {5: "toilet", 7: "window"}

    Returns:
        list[str]
            Example:
            ["toilet 0.12 0.34 0.56 0.78", "window 0.11 0.22 0.33 0.44"]
    """
    replaced_lines = []

    for line in label_lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        class_id = int(float(parts[0]))
        class_name = rh_db.id2name(class_id, 'ot')
        parts[0] = class_name

        replaced_lines.append(" ".join(parts))

    return replaced_lines

def _bulk_fetch_object_type_names(rh_db, name_mode):
    """Fetch the full id→name mapping for the given name_mode table in one query.

    Bulk query output:
    - rows with columns: id, name from the table selected by name_mode
      (for example rh_object_types when name_mode == "ot")

    Returned structure:
    - dict[int, str] mapping the source class id stored in YOLO label files to
      its human-readable class name

    Downstream usage:
    - remap_labels_to_semantic_ids() uses this dict for O(1) name lookup while
      rewriting each label line, avoiding one id2name() SQL query per label.
    """
    table_map = {
        "h":            "rh_homes",
        "home":         "rh_homes",
        "hs":           "rh_home_sessions",
        "home_session": "rh_home_sessions",
        "r":            "rh_rooms",
        "room":         "rh_rooms",
        "rt":           "rh_room_types",
        "room_type":    "rh_room_types",
        "s":            "rh_sensors",
        "sensor":       "rh_sensors",
        "st":           "rh_sensor_types",
        "sensor_type":  "rh_sensor_types",
        "o":            "rh_objects",
        "object":       "rh_objects",
        "ot":           "rh_object_types",
        "object_type":  "rh_object_types",
    }
    table = table_map.get(name_mode)
    if table is None:
        raise ValueError(f"Unknown name_mode '{name_mode}'. Valid values: {list(table_map)}")
    # Load the full lookup table once. The returned rows are converted into an
    # in-memory id->name dict reused for every label line during remapping.
    df = rh_db.query(f"SELECT id, name FROM {table}")
    return dict(zip(df["id"], df["name"]))


def remap_labels_to_semantic_ids(labels_root, rh_db, mapping_json, name_mode="ot", backup=True):
    """
    Replace instance ids in YOLO label files with semantic class ids using a JSON mapping as reference.

    JSON format:
    {
      "0": "chair",
      "1": "table"
    }

    Behavior:
    - Reuse semantic ids already present in mapping_json.
    - If a class name is missing, assign the next semantic id.
    - Overwrite the same JSON file with the updated mapping.

    Optimized: replaces per-label id2name() SQL queries with a single bulk
    lookup table fetched once upfront.

    Example:
    old line:  17 0.12 0.35 0.18 0.40 0.20 0.44
    17 -> "chair" via id2name_map[17]
    new line:   0 0.12 0.35 0.18 0.40 0.20 0.44
    """
    labels_root = Path(labels_root)
    mapping_json = Path(mapping_json)
    mapping_json_preexists = mapping_json.exists()
    txt_files = sorted(labels_root.rglob("*.txt"))
    json_update = False

    if mapping_json_preexists:
        id_to_name = {int(k): v for k, v in load_json(mapping_json).items()}
    else:
        id_to_name = {}

    name_to_id = {v: k for k, v in id_to_name.items()}
    next_id = max(id_to_name, default=-1) + 1

    # Pre-fetch the full instance_id → class_name table in one query.
    print(f"Pre-fetching object type names for name_mode='{name_mode}' (bulk)...")
    id2name_map = _bulk_fetch_object_type_names(rh_db, name_mode)

    for txt_path in tqdm(txt_files, desc="Remapping labels", unit="file", file=sys.stdout):
        old_lines = txt_path.read_text(encoding="utf-8").splitlines()
        new_lines, changed = [], False

        for line in old_lines:
            if not line.strip():
                new_lines.append(line)
                continue

            parts = line.split()
            try:
                instance_id = int(float(parts[0]))
                class_name = id2name_map[instance_id]   # O(1) dict lookup — no SQL
            except (ValueError, KeyError):
                new_lines.append(line)
                continue

            if class_name not in name_to_id:
                name_to_id[class_name] = next_id
                id_to_name[next_id] = class_name
                next_id += 1
                json_update = True

            new_line = " ".join([str(name_to_id[class_name]), *parts[1:]])
            new_lines.append(new_line)
            changed |= (new_line != line)

        if changed:
            if backup:
                txt_path.with_suffix(txt_path.suffix + ".bak").write_text(
                    "\n".join(old_lines) + "\n", encoding="utf-8"
                )
            txt_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    if json_update or not mapping_json_preexists:
        save_json(id_to_name, mapping_json)

    print(f"Done. Semantic classes in mapping: {len(id_to_name)}")
    return id_to_name


data_path = "data"
local_files_path = data_path + "/files"
rgbd = "rgbd"
scene = "scene"


def test_observation_visualization(rh_db, obs_id, epsilon_ratio=0.002):
    # 1. Get raw data and labels
    image, _, label_lines = get_yolo_lines_for_observation(rh_db, obs_id, epsilon_ratio)
    if image is None: return

    # 2. Prepare masks for overlay
    labels_with_masks = rh_db.get_RGBD_labels(obs_id)
    aligned_masks = align_all_masks_image(labels_with_masks["mask"], image)

    # 3. Create 3-panel plot and call separate plotting functions
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    plot_image(image, title="1. Original RGB", ax=axes[0])
    plot_mask_overlay(image, aligned_masks, title="2. DB Mask Overlay", ax=axes[1])

    label_lines = replace_class_ids_with_names(label_lines, rh_db)
    plot_yolo_bboxes(image, label_lines, title="3. YOLO BBoxes", ax=axes[2])

    plt.tight_layout()
    plt.show()

def go(args):

    output_root = Path(args.output_root).resolve()
    rgbd_root = Path(args.rgbd_path).resolve()
    labels_root = Path(args.labels_root).resolve()
    mapping_json = Path(args.mapping_json).resolve()
    should_force_convert = args.force_convert if args.force_convert is not None else False

    try:
        db = RobotAtHome(
            rh_path=Path(args.rh_path).resolve(),
            rgbd_path=Path(args.rgbd_path).resolve(),
            scene_path=Path(args.scene_path).resolve(),
        )
    except Exception as e:
        print(f"Error initializing RobotAtHome: {e}")
        raise SystemExit(1)

    # Force mode bypasses skip logic and rebuilds both dataset outputs and mapping.
    if should_force_convert:
        print("force_convert=True: forcing conversion and mapping generation.")
        convert_df_to_yolo_seg(
            rh_db=db,
            output_root=output_root,
            rgbd_root=rgbd_root,
            epsilon_ratio=args.epsilon_ratio,
        )
        remap_labels_to_semantic_ids(
            labels_root=labels_root,
            rh_db=db,
            name_mode=args.name_mode,
            mapping_json=mapping_json,
            backup=args.backup,
        )
        return

    conversion_status = conversion_outputs_exist(db, output_root, rgbd_root)

    # If nothing exists yet, run the full conversion/remap pipeline normally.
    if conversion_status["none_exist"]:
        print("No converted outputs exist yet. Running conversion and mapping generation.")
        convert_df_to_yolo_seg(
            rh_db=db,
            output_root=output_root,
            rgbd_root=rgbd_root,
            epsilon_ratio=args.epsilon_ratio,
        )
        remap_labels_to_semantic_ids(
            labels_root=labels_root,
            rh_db=db,
            name_mode=args.name_mode,
            mapping_json=mapping_json,
            backup=args.backup,
        )
        return

    # If outputs are only partially present, do not reconvert automatically; warn and only build missing mapping if needed.
    if conversion_status["missing_pairs"] > 0:
        logger.warning(
            "Detected %s missing converted output pairs out of %s expected pairs under %s. "
            "Some observations may be unavailable. Skipping conversion and mapping generation because partial outputs already exist and force_convert is false.",
            conversion_status["missing_pairs"],
            conversion_status["total_pairs"],
            output_root,
        )
        print(
            "Partial converted outputs detected. Logged a warning and skipping conversion."
        )
        if mapping_json_exists(mapping_json):
            print(f"Mapping JSON already exists at {mapping_json}. Skipping remap.")
        else:
            print(
                "Mapping JSON is missing. Generating it from currently available label files."
            )
            remap_labels_to_semantic_ids(
                labels_root=labels_root,
                rh_db=db,
                name_mode=args.name_mode,
                mapping_json=mapping_json,
                backup=args.backup,
            )
        return

    print(f"YOLO conversion outputs already exist under {output_root}. Skipping conversion.")

    # When converted outputs are complete, skip conversion and only build mapping if it is still missing.
    if mapping_json_exists(mapping_json):
        print(f"Mapping JSON already exists at {mapping_json}. Skipping remap.")
    else:
        print(
            "Mapping JSON is missing. Generating it from existing converted label files."
        )
        remap_labels_to_semantic_ids(
            labels_root=labels_root,
            rh_db=db,
            name_mode=args.name_mode,
            mapping_json=mapping_json,
            backup=args.backup,
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Convert Robot@Home annotations to YOLO and remap labels"
    )
    parser.add_argument("--rh_path", type=str, required=True, help="Root Robot@Home path")
    parser.add_argument("--rgbd_path", type=str, required=True, help="Path to RGBD files directory for Loading RobotAtHome")
    parser.add_argument("--scene_path", type=str, required=True, help="Path to scene files directory")
    parser.add_argument("--output_root", type=str, default="yolo", help="Output YOLO dataset root")
    parser.add_argument("--epsilon_ratio", type=float, default=0.002, help="Polygon simplification factor")
    parser.add_argument("--labels_root", type=str, default="yolo/labels", help="Label directory to remap")
    parser.add_argument("--mapping_json", type=str, default="yolo/class_id_to_name.json", help="Class mapping JSON")
    parser.add_argument("--name_mode", type=str, default="ot", help="Robot@Home returns Object Type (ot) as Label")
    parser.add_argument(
        "--force_convert", type=lambda x: x.lower() == "true",
        required=False,
        help="Force conversion and mapping generation even if outputs already exist (true/false)",
    )
    parser.add_argument(
        "--backup", type=lambda x: x.lower() == "true",
        default=False,
        help="Whether to save .bak files (true/false)",
    )

    args = parser.parse_args()

    go(args)
