from pathlib import Path
import argparse

import cv2
import numpy as np
import robotathome as rh
from robotathome import RobotAtHome
from robotathome import get_labeled_img
from robotathome import log, logger

from src.shared import align_all_masks

import matplotlib.pyplot as plt

log.set_log_level("INFO")

data_path = "data"
local_files_path = data_path + "/files"
rgbd = "rgbd"
scene = "scene"

# Built-in Robot@Home sources used when no custom source is passed via CLI.
DEFAULT_ROBOTATHOME_SOURCES = [
    {
        "url": "https://zenodo.org/record/7811795/files/Robot@Home2_db.tgz",
        "filename": "Robot@Home2_db.tgz",
        "md5": "d34fb44c01f31c87be8ab14e5ecd0767",
        "extract_to": data_path,
    },
    {
        "url": "https://zenodo.org/record/7811795/files/Robot@Home2_files.tgz",
        "filename": "Robot@Home2_files.tgz",
        "md5": "36faa2ffdd936a14455b2d1f3075e6ca",
        "extract_to": local_files_path,
    },
]


def is_archive_file(filename: str) -> bool:
    """Return True when filename looks like a compressed archive."""
    archive_suffixes = (".zip", ".tar", ".tar.gz", ".tgz", ".gz", ".bz2", ".xz")
    lower = filename.lower()
    return any(lower.endswith(sfx) for sfx in archive_suffixes)


def ask_yes_no(question: str, default: bool = False) -> bool:
    prompt = " [Y/n]: " if default else " [y/N]: "
    reply = input(question + prompt).strip().lower()

    if not reply:
        return default

    return reply in {"y", "yes"}


def download_rh(out_dir, extract_root=None, force_download=None, source_specs=None):
    """
    Download and optionally extract one or more sources.

    Args:
        out_dir: Directory where files are downloaded.
        extract_root: Root directory for extraction (default: home directory).
        force_download: True to force re-download, False to reuse, None to prompt.
        source_specs: List[dict] with keys: url, filename, md5, optional extract_to.
                     If None, uses DEFAULT_ROBOTATHOME_SOURCES.

    Returns:
        List of processed filenames.
    """
    out_dir = Path(out_dir).expanduser()
    extract_root = Path(extract_root).expanduser() if extract_root else Path.home()
    source_specs = source_specs or DEFAULT_ROBOTATHOME_SOURCES

    out_dir.mkdir(parents=True, exist_ok=True)
    processed = []

    for item in source_specs:
        archive_path = out_dir / item["filename"]
        extract_to = item.get("extract_to")
        should_extract = is_archive_file(item["filename"]) and extract_to is not None

        should_download = True

        if archive_path.exists():
            print(f"Found existing file: {archive_path}")

            if force_download is None:
                should_download = ask_yes_no(
                    f"Do you want to force re-download {item['filename']}?",
                    default=False,
                )
            else:
                should_download = force_download

            if should_download:
                print(f"Re-downloading {item['filename']}...")
            else:
                print(f"Reusing existing file: {item['filename']}")

        if should_download:
            rh.download(item["url"], str(out_dir))

        print(f"Verifying {item['filename']}...")
        md5_actual = rh.get_md5(str(archive_path))

        if md5_actual != item["md5"]:
            print(
                f"MD5 mismatch for {item['filename']} "
                f"(expected {item['md5']}, got {md5_actual})."
            )

            re_download = ask_yes_no(
                f"Integrity check failed. Re-download {item['filename']} now?",
                default=True,
            )

            if re_download:
                rh.download(item["url"], str(out_dir))
                md5_actual = rh.get_md5(str(archive_path))

                if md5_actual != item["md5"]:
                    raise ValueError(
                        f"MD5 mismatch persists for {item['filename']} after re-download."
                    )
            else:
                raise ValueError(f"Cannot continue with corrupted file: {item['filename']}")

        if should_extract:
            extract_path = extract_root / extract_to
            extract_path.mkdir(parents=True, exist_ok=True)
            print(f"Extracting {item['filename']} to {extract_path}...")
            rh.uncompress(str(archive_path), str(extract_path))
        elif is_archive_file(item["filename"]):
            print(
                f"Skipping extraction for archive {item['filename']} "
                "(no extract_to provided)."
            )

        processed.append(item["filename"])

    print("Done.")
    return processed


def query_sample_annotation():
    rgbd_path = Path(local_files_path).joinpath(rgbd).resolve()
    scene_path = Path(local_files_path).joinpath(scene).resolve()

    try:
        rh_db = RobotAtHome(
            rh_path=Path(data_path).resolve(),
            rgbd_path=rgbd_path,
            scene_path=scene_path,
        )
    except Exception as e:
        print(f"Error initializing RobotAtHome: {e}")
        return

    lblrgbd = rh_db.get_sensor_observations("lblrgbd")
    print(lblrgbd.head())
    print(lblrgbd.columns)

    rng = np.random.default_rng()
    sample_index = rng.integers(0, len(lblrgbd))
    sample_id = int(lblrgbd.iloc[sample_index]["id"]) if "id" in lblrgbd.columns else int(lblrgbd.index[0])

    print("sample_id:", sample_id)
    print(f"# Labeled RGBD set: {len(lblrgbd)} observations with {len(lblrgbd.columns)} fields")

    rgb_img, depth_img = rh_db.get_RGBD_files(sample_id)
    logger.info("Sensor observation {} files\n RGB file   : {}\n Depth file : {}", sample_id, rgb_img, depth_img)

    annotation = rh_db.get_RGBD_labels(id=sample_id)
    logger.info("\nlabels: \n{}", annotation.columns)

    rgb_img_loaded = cv2.imread(rgb_img)
    depth_img_loaded = cv2.imread(depth_img, cv2.IMREAD_UNCHANGED)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(np.rot90(cv2.cvtColor(rgb_img_loaded, cv2.COLOR_BGR2RGB)))
    ax[0].set_title("RGB Image")
    ax[1].imshow(np.rot90(depth_img_loaded), cmap="gray")
    ax[1].set_title("Depth Image")

    aligned_masks = align_all_masks(annotation["mask"], rgb_img)
    annotation["mask"] = aligned_masks
    labeled_img, _ = get_labeled_img(annotation, rgb_img)

    print(labeled_img.shape)
    ax[2].imshow(np.rot90(labeled_img))
    ax[2].set_title("Per-pixel Annotation")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and extract Robot@Home or one custom source")
    parser.add_argument("--out-dir", type=str, default="data", help="Directory where files are downloaded")
    parser.add_argument(
        "--extract-root",
        type=str,
        default=None,
        help="Root directory where archives are extracted (default: home directory)",
    )
    parser.add_argument("--force-download", action="store_true", help="Force re-download if file exists")
    parser.add_argument("--no-force-download", action="store_true", help="Never re-download; reuse existing files")

    # Custom source arguments (all 3 required together)
    parser.add_argument("--dataset-url", type=str, default=None, help="Dataset URL")
    parser.add_argument("--dataset-filename", type=str, default=None, help="Downloaded filename")
    parser.add_argument("--dataset-md5", type=str, default=None, help="Expected MD5 checksum")
    parser.add_argument(
        "--dataset-extract-to",
        type=str,
        default=None,
        help="Optional extract path relative to --extract-root (archive files only)",
    )

    parser.add_argument("--query-sample", action="store_true", help="Query and visualize a sample annotation")
    args = parser.parse_args()

    force_download_flag = None
    if args.force_download:
        force_download_flag = True
    elif args.no_force_download:
        force_download_flag = False

    custom_fields = [args.dataset_url, args.dataset_filename, args.dataset_md5]
    has_any_custom = any(v is not None for v in custom_fields)
    has_all_required_custom = all(v is not None for v in custom_fields)

    if has_any_custom and not has_all_required_custom:
        parser.error("--dataset-url, --dataset-filename, and --dataset-md5 must be provided together.")

    source_specs = None
    if has_all_required_custom:
        source_specs = [
            {
                "url": args.dataset_url,
                "filename": args.dataset_filename,
                "md5": args.dataset_md5,
                **({"extract_to": args.dataset_extract_to} if args.dataset_extract_to else {}),
            }
        ]
        print("Using one custom source from CLI arguments.")
    else:
        print("Using built-in Robot@Home default sources.")

    try:
        processed = download_rh(
            out_dir=args.out_dir,
            extract_root=args.extract_root,
            force_download=force_download_flag,
            source_specs=source_specs,
        )
        print(f"Successfully processed: {', '.join(processed)}")
    except Exception as e:
        print(f"Download failed: {e}")
        raise SystemExit(1)

    if args.query_sample:
        print("\n" + "=" * 60)
        print("Querying sample annotation...")
        print("=" * 60)
        query_sample_annotation()

