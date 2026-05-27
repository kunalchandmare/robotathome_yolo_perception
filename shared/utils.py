import json
from pathlib import Path

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def ask_yes_no(question: str, default: bool = False) -> bool:
    prompt = " [Y/n]: " if default else " [y/N]: "
    reply = input(question + prompt).strip().lower()

    if not reply:
        return default

    return reply in {"y", "yes"}

def align_all_masks_image(mask_list,image:np.ndarray):
    h, w = image.shape[:2]
    aligned_masks = []
    for mask in mask_list:
        # Check if shape already matches
        if (mask.shape[0], mask.shape[1]) == (h, w):
            aligned_masks.append(mask)
            continue

        # Rotate if dimensions are swapped
        if (h, w) == (mask.shape[1], mask.shape[0]):
            mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)

        # Resize if still mismatching
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        aligned_masks.append(mask)
    return aligned_masks

def align_all_masks(mask_list, img_path:str):
    img = cv2.imread(img_path)
    return align_all_masks_image(mask_list, img)


def plot_image(image, title="Image", ax=None, rotate_90_ccw=False):
    """Plots a single BGR image as RGB with optional rotation."""
    if rotate_90_ccw:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(img_rgb)
    ax.set_title(title)
    ax.axis("off")
    return ax


def plot_mask_overlay(image, masks, title="Mask Overlay", ax=None, alpha=0.45, rotate_90_ccw=False):
    """Plots rotated/aligned masks overlaid on an image."""
    if rotate_90_ccw:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    overlay = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).copy()
    rng = np.random.default_rng(0)

    for mask in masks:
        # Rotate mask if needed
        m = (np.asarray(mask) > 0)
        if rotate_90_ccw:
            m = cv2.rotate(m.astype(np.uint8), cv2.ROTATE_90_COUNTERCLOCKWISE) > 0

        if np.any(m):
            color = rng.integers(0, 255, size=3, dtype=np.uint8)
            overlay[m] = ((1 - alpha) * overlay[m] + alpha * color).astype(np.uint8)

    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(overlay)
    ax.set_title(title)
    ax.axis("off")
    return ax


def plot_yolo_bboxes(image, label_lines, title="YOLO BBoxes", ax=None, rotate_90_ccw=False):
    """
    Draw bounding boxes derived from YOLO segmentation polygons.

    label_lines format:
        class_id x1 y1 x2 y2 x3 y3 ... xn yn
    """
    if rotate_90_ccw:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    h, w = image.shape[:2]

    if ax is None:
        fig, ax = plt.subplots()

    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.set_title(title)
    ax.axis("off")

    for line in label_lines:
        parts = line.strip().split()
        if len(parts) < 7:
            continue  # need at least class + 3 points

        class_id = parts[0]

        coords = np.array(list(map(float, parts[1:])), dtype=np.float32).reshape(-1, 2)

        xs = coords[:, 0] * w
        ys = coords[:, 1] * h

        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        rect = Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            linewidth=2,
            edgecolor="lime",
            facecolor="none"
        )
        ax.add_patch(rect)

        ax.text(
            x_min,
            max(0, y_min - 5),
            f"cls {class_id}",
            color="yellow",
            fontsize=9,
            bbox=dict(facecolor="black", alpha=0.5)
        )

    return ax

def save_json(data, path):
    """Save dictionary as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_json(path):
    """Load JSON file and return Python object."""
    with open(Path(path), "r", encoding="utf-8") as f:
        return json.load(f)

import torch

def print_gpu_info(device_id):
    """Print basic CUDA GPU info for the selected device."""
    if not torch.cuda.is_available():
        print("CUDA GPU not available. Using CPU.")

    print("CUDA available:", torch.cuda.is_available())
    print("GPU count:", torch.cuda.device_count())
    print("Current device:", device_id)
    print("GPU name:", torch.cuda.get_device_name(device_id))
    print("Allocated memory (GB):", round(torch.cuda.memory_allocated(device_id) / 1024**3, 3))
    print("Reserved memory (GB):", round(torch.cuda.memory_reserved(device_id) / 1024**3, 3))