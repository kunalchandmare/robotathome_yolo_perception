from pathlib import Path
import argparse
import random
import shutil
import sys
from collections import Counter

import pandas as pd
from tqdm import tqdm

from shared.utils import save_json


def read_yolo_classes(label_path):
    """
    Read one YOLO label file and return the set of class IDs present in that image.
    """
    classes = set()

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            try:
                cls_id = int(float(parts[0]))
                classes.add(cls_id)
            except ValueError:
                continue

    return classes


def collect_image_label_pairs(images_root, labels_root, image_exts=(".jpg", ".jpeg", ".png")):
    """
    Collect valid (image, label) pairs by matching relative paths with progress tracking.
    Images without labels are skipped with a warning.

    Args:
        images_root: Root directory containing images
        labels_root: Root directory containing label files
        image_exts: Tuple of image file extensions to search for

    Returns:
        pairs: List of (image_path, label_path) tuples for valid pairs
        missing_labels: List of image_paths without corresponding labels
    """
    images_root = Path(images_root)
    labels_root = Path(labels_root)

    # Collect all image files
    print("Scanning for images...")
    image_files = []
    for ext in image_exts:
        image_files.extend(images_root.rglob(f"*{ext}"))

    image_files = sorted(image_files)
    print(f"Found {len(image_files)} images")

    pairs = []
    missing_labels = []

    # Match images with labels and track progress
    for img_path in tqdm(image_files, desc="Matching labels", unit="file", file=sys.stdout):
        rel_path = img_path.relative_to(images_root)
        label_path = labels_root / rel_path.with_suffix(".txt")

        if label_path.exists():
            pairs.append((img_path, label_path))
        else:
            missing_labels.append(img_path)

    # Print warnings for missing labels
    if missing_labels:
        print(f"\n⚠ Missing labels for {len(missing_labels)} images:")
        for img_path in missing_labels[:10]:  # Show first 10
            print(f"  {img_path}")
        if len(missing_labels) > 10:
            print(f"  ... and {len(missing_labels) - 10} more")

    return pairs, missing_labels


def random_split_pairs(pairs, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Randomly shuffle valid image-label pairs and split into train/val/test.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9, "Ratios must sum to 1."

    pairs = list(pairs)
    random.Random(seed).shuffle(pairs)

    total = len(pairs)
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)
    n_test = total - n_train - n_val

    splits = {
        "train": pairs[:n_train],
        "val": pairs[n_train:n_train + n_val],
        "test": pairs[n_train + n_val:]
    }

    return splits


def build_image_class_map(pairs):
    """
    Build image-to-classes and class-to-count mappings with progress tracking.

    Args:
        pairs: List of (image_path, label_path) tuples

    Returns:
        image_to_classes: Dict mapping image_path -> set of class IDs
        class_image_counts: Counter of class_id -> number of images containing it
    """
    image_to_classes = {}
    class_image_counts = Counter()

    for img_path, label_path in tqdm(pairs, desc="Building class map", unit="file", file=sys.stdout):
        classes = read_yolo_classes(label_path)
        image_to_classes[img_path] = classes

        for cls_id in classes:
            class_image_counts[cls_id] += 1

    return image_to_classes, class_image_counts


def identify_rare_classes(class_image_counts, rare_threshold=5):
    """
    Find classes that appear in very few images.
    """
    return {cls_id for cls_id, count in class_image_counts.items() if count <= rare_threshold}


def score_image_rarity(image_classes, class_image_counts):
    """
    Give higher score to images containing rare classes.
    """
    if not image_classes:
        return 0.0

    return sum(1.0 / class_image_counts[c] for c in image_classes)


def choose_best_split(
    image_classes,
    split_sizes,
    target_sizes,
    split_class_counts,
    preferred_splits=("train", "val", "test"),
    rare_bonus_classes=None,
):
    """
    Choose the best split for one image based on:
    - free capacity in the split
    - whether the split is missing those classes
    - extra bonus for rare classes in val/test
    """
    best_split = None
    best_score = -1e9

    for split in preferred_splits:
        if split_sizes[split] >= target_sizes[split]:
            continue

        score = 0.0

        for cls_id in image_classes:
            if split_class_counts[split][cls_id] == 0:
                score += 2.0
            else:
                score += 1.0 / (1.0 + split_class_counts[split][cls_id])

        if rare_bonus_classes and split in {"val", "test"}:
            for cls_id in image_classes:
                if cls_id in rare_bonus_classes:
                    score += 5.0

        score += 0.01 * (target_sizes[split] - split_sizes[split])

        if score > best_score:
            best_score = score
            best_split = split

    if best_split is None:
        best_split = min(target_sizes.keys(), key=lambda s: split_sizes[s])

    return best_split


def stratified_split_pairs(
    pairs,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42,
    rare_threshold=5,
):
    """
    Stratified image-level split for multi-label YOLO datasets.

    Rule:
    1. Try to ensure very rare classes appear at least once in val or test.
    2. Then fill remaining images normally using greedy class-aware assignment.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9, "Ratios must sum to 1."
    print("Using stratified split...", flush=True)

    image_to_classes, class_image_counts = build_image_class_map(pairs)
    rare_classes = identify_rare_classes(class_image_counts, rare_threshold=rare_threshold)

    img_to_pair = {img_path: (img_path, label_path) for img_path, label_path in pairs}
    images = list(image_to_classes.keys())

    rng = random.Random(seed)
    rng.shuffle(images)

    #That line sorts the images list from the most rare-class-heavy images first to the least rare-class-heavy images
    #last. The key= part tells Python what value to sort by, and reverse=True makes it sort in
    # descending order
    images = sorted(
        images,
        key=lambda img: score_image_rarity(image_to_classes[img], class_image_counts),
        reverse=True,
    )

    total = len(images)
    target_sizes = {
        "train": int(round(total * train_ratio)),
        "val": int(round(total * val_ratio)),
        "test": total - int(round(total * train_ratio)) - int(round(total * val_ratio)),
    }

    while sum(target_sizes.values()) != total:
        target_sizes["train"] += total - sum(target_sizes.values())

    split_images = {"train": [], "val": [], "test": []}
    split_sizes = Counter()
    split_class_counts = {"train": Counter(), "val": Counter(), "test": Counter()}
    assigned = set()
    covered_rare_classes = set()

    # Pass 1: reserve rare-class images for val/test if possible
    # It forces images with rare objects into your evaluation sets (val/test) first,
    # ensuring they aren't accidentally hidden away in training only.
    for img_path in images:
        classes = image_to_classes[img_path]

        uncovered_rare = [c for c in classes if c in rare_classes and c not in covered_rare_classes]
        if not uncovered_rare:
            continue

        split = choose_best_split(
            image_classes=classes,
            split_sizes=split_sizes,
            target_sizes=target_sizes,
            split_class_counts=split_class_counts,
            preferred_splits=("val", "test"),
            rare_bonus_classes=rare_classes,
        )

        if split in {"val", "test"}:
            split_images[split].append(img_path)
            split_sizes[split] += 1
            assigned.add(img_path)

            for cls_id in classes:
                split_class_counts[split][cls_id] += 1
                if cls_id in rare_classes:
                    covered_rare_classes.add(cls_id)

    # Pass 2: fill remaining images normally
    for img_path in images:
        if img_path in assigned:
            continue

        classes = image_to_classes[img_path]

        split = choose_best_split(
            image_classes=classes,
            split_sizes=split_sizes,
            target_sizes=target_sizes,
            split_class_counts=split_class_counts,
            preferred_splits=("train", "val", "test"),
            rare_bonus_classes=None,
        )

        split_images[split].append(img_path)
        split_sizes[split] += 1

        for cls_id in classes:
            split_class_counts[split][cls_id] += 1

    splits = {
        split_name: [img_to_pair[img_path] for img_path in img_list]
        for split_name, img_list in split_images.items()
    }

    print(f"Rare classes (<= {rare_threshold} images): {sorted(rare_classes)}")
    print(f"Rare classes covered in val/test: {sorted(covered_rare_classes)}")

    return splits


def copy_split_files(splits, images_root, labels_root, output_root):
    """
    Copy split files into YOLO folder structure with progress tracking.

    Pre-creates all directories once, then copies files with progress bar.

    Args:
        splits: Dict of split_name -> list of (image_path, label_path) tuples
        images_root: Root directory of source images
        labels_root: Root directory of source labels
        output_root: Root output directory
    """
    images_root = Path(images_root)
    labels_root = Path(labels_root)
    output_root = Path(output_root)

    # Pre-create all split directories once
    for split_name in ["train", "val", "test"]:
        (output_root / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split_name).mkdir(parents=True, exist_ok=True)

    # Count total files for progress bar
    total_files = sum(len(pairs) for pairs in splits.values())

    # Copy files with progress tracking
    with tqdm(total=total_files, desc="Copying files", unit="file", file=sys.stdout) as pbar:
        for split_name, pairs in splits.items():
            for img_path, label_path in pairs:
                rel_path = img_path.relative_to(images_root)

                out_img = output_root / "images" / split_name / rel_path
                out_lbl = output_root / "labels" / split_name / rel_path.with_suffix(".txt")

                # Create subdirectories only if needed (rare case)
                out_img.parent.mkdir(parents=True, exist_ok=True)
                out_lbl.parent.mkdir(parents=True, exist_ok=True)

                shutil.copy2(img_path, out_img)
                shutil.copy2(label_path, out_lbl)
                pbar.update(1)


def split_yolo_dataset(
    source_root,
    output_root,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42,
    image_exts=(".jpg", ".jpeg", ".png"),
    use_stratified=False,
    rare_threshold=5,
):
    """
       Split a YOLO-format dataset into train/val/test sets and copy files into
       the standard YOLO directory structure.

       Expected input structure:
           source_root/
               images/
               labels/

       Output structure:
           output_root/
               images/train/
               images/val/
               images/test/
               labels/train/
               labels/val/
               labels/test/

       Args:
           source_root:
               Root folder of the source YOLO dataset. It must contain `images/`
               and `labels/` subfolders with matching relative paths.

           output_root:
               Destination folder where the split dataset will be written.

           train_ratio:
               Fraction of valid image-label pairs assigned to the training set.

           val_ratio:
               Fraction of valid image-label pairs assigned to the validation set.

           test_ratio:
               Fraction of valid image-label pairs assigned to the test set.

           seed:
               Random seed used to keep the split reproducible.

           image_exts:
               Image file extensions to scan when collecting dataset images.

           use_stratified:
               If False, perform a simple random split.
               If True, perform a class-aware stratified split that tries to
               preserve class coverage across splits, especially for rare classes.

           rare_threshold:
               Maximum number of images a class may appear in to be considered
               "rare" during stratified splitting.

               This parameter is only used when `use_stratified=True`.

               Example:
               - If `rare_threshold=5`, any class appearing in 5 or fewer images
                 is treated as rare.
               - The stratified split will try to place such rare classes into
                 validation or test at least once when possible, so they are not
                 seen only in training.

               If `use_stratified=False`, this value has no effect.

       Returns:
           dict:
               A dictionary with keys `train`, `val`, and `test`, where each value
               is a list of `(image_path, label_path)` pairs assigned to that split.

       Notes:
           - Only images with matching label files are included.
           - Images without labels are skipped and reported.
           - Files are copied into the output directory after the split is created.
       """
    source_root = Path(source_root).resolve()
    output_root = Path(output_root).resolve()

    images_root = source_root / "images"
    labels_root = source_root / "labels"

    pairs, missing_labels = collect_image_label_pairs(
        images_root=images_root,
        labels_root=labels_root,
        image_exts=image_exts,
    )

    if use_stratified:
        splits = stratified_split_pairs(
            pairs=pairs,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            rare_threshold=rare_threshold,
        )
    else:
        print("Using random split...")
        splits = random_split_pairs(
            pairs=pairs,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )

    copy_split_files(
        splits=splits,
        images_root=images_root,
        labels_root=labels_root,
        output_root=output_root,
    )

    total = sum(len(v) for v in splits.values())

    print(f"Total valid image-label pairs: {total}")
    print(f"Missing labels skipped: {len(missing_labels)}")
    print(f"Train: {len(splits['train'])}")
    print(f"Val:   {len(splits['val'])}")
    print(f"Test:  {len(splits['test'])}")

    return splits


def export_split_review(
    splits,
    output_dir,
    rare_threshold=5,
    normal_example_count=10,
):
    """
    Export a simple CSV + JSON review of current split.

    Includes:
    - all rare classes (count <= rare_threshold)
    - a few example non-rare classes with the lowest counts
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_class_counts = {}
    global_counts = Counter()

    print("Reviewing split class distributions...")

    for split_name, pairs in splits.items():
        counter = Counter()

        for img_path, label_path in tqdm(
            pairs,
            desc=f"Reviewing {split_name}",
            unit="file",
            file=sys.stdout,
        ):
            classes = read_yolo_classes(label_path)
            for cls_id in classes:
                counter[cls_id] += 1
                global_counts[cls_id] += 1

        split_class_counts[split_name] = counter

    all_classes_sorted = sorted(global_counts.items(), key=lambda x: x[1])

    rare_classes = [cls_id for cls_id, cnt in all_classes_sorted if cnt <= rare_threshold]
    normal_examples = [
        cls_id for cls_id, cnt in all_classes_sorted if cnt > rare_threshold
    ][:normal_example_count]

    selected_classes = rare_classes + normal_examples

    rows = []
    for cls_id in selected_classes:
        train_cnt = split_class_counts.get("train", Counter()).get(cls_id, 0)
        val_cnt = split_class_counts.get("val", Counter()).get(cls_id, 0)
        test_cnt = split_class_counts.get("test", Counter()).get(cls_id, 0)
        total_cnt = global_counts[cls_id]

        rows.append({
            "class_id": cls_id,
            "category": "rare" if cls_id in rare_classes else "normal_example",
            "total_images": total_cnt,
            "train_images": train_cnt,
            "val_images": val_cnt,
            "test_images": test_cnt,
        })

    df = pd.DataFrame(rows).sort_values(["category", "total_images", "class_id"])

    csv_path = output_dir / "split_review.csv"
    json_path = output_dir / "split_review.json"

    df.to_csv(csv_path, index=False)

    report = {
        "rare_threshold": rare_threshold,
        "num_rare_classes": len(rare_classes),
        "num_normal_examples": len(normal_examples),
        "rare_classes": rare_classes,
        "normal_example_classes": normal_examples,
        "rows": df.to_dict(orient="records"),
    }

    save_json(report, json_path)

    print(f"Saved CSV:  {csv_path}")
    print(f"Saved JSON: {json_path}")

    return {
        "csv_path": csv_path,
        "json_path": json_path,
        "df": df,
    }

def _str_to_bool(value):
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected one of: true/false, 1/0, yes/no")


def _parse_image_exts(value):
    if isinstance(value, (tuple, list)):
        return tuple(value)
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    normalized = [part if part.startswith(".") else f".{part}" for part in parts]
    return tuple(normalized)


def go(args) -> int:
    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_root).resolve()
    train_ratio = args.train_ratio
    val_ratio = args.val_ratio
    test_ratio = args.test_ratio
    seed = args.seed
    image_exts = _parse_image_exts(args.image_exts)
    use_stratified = args.use_stratified
    rare_threshold = args.rare_threshold

    split_yolo_dataset(
        source_root=source_root,
        output_root=output_root,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        image_exts=image_exts,
        use_stratified=use_stratified,
        rare_threshold=rare_threshold,
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split YOLO dataset into train/val/test folders")
    parser.add_argument("--source_root", type=str, required=True, help="Source YOLO dataset root")
    parser.add_argument("--output_root", type=str, default="yolo_split_stratified", help="Output split dataset root")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Training split ratio")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Test split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--image_exts", type=str, default=".jpg,.jpeg,.png", help="Comma-separated image extensions")
    parser.add_argument("--use_stratified", type=_str_to_bool, default=True, help="Use stratified split")
    parser.add_argument("--rare_threshold", type=int, default=20, help="Rare class threshold for stratified split")

    args = parser.parse_args()

    go(args)


