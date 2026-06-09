from pathlib import Path
import logging
import cv2
import numpy as np
from matplotlib import pyplot as plt
from robotathome import RobotAtHome
from tqdm import tqdm
import argparse

from shared.utils import ensure_dir, align_all_masks, plot_image, plot_mask_overlay, plot_yolo_bboxes, align_all_masks_image, \
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

def convert_df_to_yolo_seg(rh_db, output_root,rgbd_root, epsilon_ratio=0.002):
    """
    Export Robot@Home annotations to YOLO segmentation dataset.

    Args:
        db:
            Loaded Robot@Home database object.
            Must provide:
            - db.get_RGBD_annotations()
            - db.get_RGBD_files(obs_id)

        output_root:
            Output directory where images/ and labels/ will be created.

        epsilon_ratio:
            Polygon simplification factor for cv2.approxPolyDP().

    Returns:
        None
    """
    output_root = Path(output_root)
    images_dir = output_root / "images"
    labels_dir = output_root / "labels"


    observtions_df = rh_db.get_sensor_observations('lblrgbd')

    grouped = observtions_df.groupby("id")
    skipped_count = 0
    converted_count = 0

    progress_bar = tqdm(grouped, total=grouped.ngroups, desc="Converting observations", unit="obs")
    progress_bar.set_postfix(converted=converted_count, skipped=skipped_count)

    for obs_id, _ in progress_bar:

        image, rgb_path, label_lines = get_yolo_lines_for_observation(rh_db, obs_id, epsilon_ratio)
        if image is None:
            skipped_count += 1
            progress_bar.set_postfix(converted=converted_count, skipped=skipped_count)
            continue
        relative_dir = Path(rgb_path).parent.relative_to(rgbd_root)
        final_img_dir = images_dir / relative_dir
        final_label_dir = labels_dir / relative_dir

        ensure_dir(final_img_dir)
        ensure_dir(final_label_dir)

        image_out = final_img_dir /  f"{obs_id}.jpg"
        label_out = final_label_dir / f"{obs_id}.txt"

        cv2.imwrite(str(image_out), image)

        with open(label_out, "w", encoding="utf-8") as f:
            if label_lines:
                f.write("\n".join(label_lines) + "\n")

        converted_count += 1
        progress_bar.set_postfix(converted=converted_count, skipped=skipped_count)

    print(
        f"Finished converting observations. Converted: {converted_count}, "
        f"Skipped: {skipped_count}, Total: {grouped.ngroups}"
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

    Example:
    old line:  17 0.12 0.35 0.18 0.40 0.20 0.44
    17 -> "chair" via rh_db.id2name(17, "ot")
    new line:   0 0.12 0.35 0.18 0.40 0.20 0.44
    """
    labels_root = Path(labels_root)
    mapping_json = Path(mapping_json)
    txt_files = sorted(labels_root.rglob("*.txt"))
    json_update = False

    if mapping_json.exists():
        id_to_name = {int(k): v for k, v in load_json(mapping_json).items()}
    else:
        id_to_name = {}

    name_to_id = {v: k for k, v in id_to_name.items()}
    next_id = max(id_to_name, default=-1) + 1

    for txt_path in tqdm(txt_files, desc="Remapping labels", unit="file"):
        old_lines = txt_path.read_text(encoding="utf-8").splitlines()
        new_lines, changed = [], False

        for line in old_lines:
            if not line.strip():
                new_lines.append(line)
                continue

            parts = line.split()
            try:
                instance_id = int(float(parts[0]))
                class_name = rh_db.id2name(instance_id, name_mode)
            except ValueError:
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

    if json_update:
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

    try:
        db = RobotAtHome(
            rh_path=Path(args.rh_path).resolve(),
            rgbd_path=Path(args.rgbd_path).resolve(),
            scene_path=Path(args.scene_path).resolve(),
        )
    except Exception as e:
        print(f"Error initializing RobotAtHome: {e}")
        raise SystemExit(1)

    convert_df_to_yolo_seg(
        rh_db=db,
        output_root=args.output_root,
        rgbd_root=Path(args.yolo_rgbd_root).resolve(),
        epsilon_ratio=args.epsilon_ratio,
    )

    semantic_id_to_name = remap_labels_to_semantic_ids(
        labels_root=args.labels_root,
        rh_db=db,
        name_mode=args.name_mode,
        mapping_json=args.mapping_json,
        backup=args.backup,
    )

    save_json(semantic_id_to_name, args.mapping_json)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Convert Robot@Home annotations to YOLO and remap labels"
    )
    parser.add_argument("--rh_path", type=str, required=True, help="Root Robot@Home path")
    parser.add_argument("--rgbd_path", type=str, required=True, help="Path to RGBD files directory for Loading RobotAtHome")
    parser.add_argument("--scene_path", type=str, required=True, help="Path to scene files directory")
    parser.add_argument("--output_root", type=str, default="yolo", help="Output YOLO dataset root")
    parser.add_argument("--yolo_rgbd_root", type=str, required=True, help="RGBD root used for relative folder structure in YOLO Dataset")
    parser.add_argument("--epsilon_ratio", type=float, default=0.002, help="Polygon simplification factor")
    parser.add_argument("--labels_root", type=str, default="yolo/labels", help="Label directory to remap")
    parser.add_argument("--mapping_json", type=str, default="yolo/class_id_to_name.json", help="Class mapping JSON")
    parser.add_argument("--name_mode", type=str, default="ot", help="Robot@Home returns Object Type (ot) as Label")
    parser.add_argument("--backup", type=bool, default=False, help="Whether to save .bak files")

    args = parser.parse_args()

    go(args)
