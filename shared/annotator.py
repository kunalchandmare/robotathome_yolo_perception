"""
Segmentation annotation utilities: overlay masks and place class labels near mask regions.
"""
import cv2
import numpy as np


# Color palette (BGR)
_PALETTE = [
    (56, 168, 0), (255, 64, 64), (0, 140, 255), (255, 200, 0),
    (180, 0, 180), (0, 220, 220), (255, 128, 0), (100, 100, 255),
    (0, 200, 100), (200, 50, 50), (50, 200, 200), (200, 200, 50),
]


def colour_for(class_id: int) -> tuple[int, int, int]:
    """Return BGR color for a class ID from palette."""
    return _PALETTE[class_id % len(_PALETTE)]


def overlay_mask(img: np.ndarray, mask: np.ndarray, color: tuple, alpha: float = 0.35) -> np.ndarray:
    """Overlay a segmentation mask on image with given color and alpha."""
    overlay = img.copy()
    mask_resized = cv2.resize(
        mask,
        (img.shape[1], img.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    binary = (mask_resized > 0.5).astype(np.uint8)
    overlay[binary == 1] = color

    blended = img.copy()
    cv2.addWeighted(overlay, alpha, blended, 1 - alpha, 0, blended)
    return blended


def _mask_to_binary(mask: np.ndarray, h: int, w: int) -> np.ndarray:
    """Resize mask to frame size and convert to uint8 binary map."""
    resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return (resized > 0.5).astype(np.uint8)


def _label_anchor_from_binary(binary: np.ndarray) -> tuple[int, int] | None:
    """Return anchor near densest region of mask (largest contour centroid)."""
    if binary is None or binary.sum() == 0:
        return None

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        ys, xs = np.where(binary > 0)
        if len(xs) == 0:
            return None
        return int(xs.mean()), int(ys.mean())

    cnt = max(contours, key=cv2.contourArea)
    m = cv2.moments(cnt)
    if m.get("m00", 0) > 0:
        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"])
        return cx, cy

    ys, xs = np.where(binary > 0)
    if len(xs) == 0:
        return None
    return int(xs.mean()), int(ys.mean())


def _draw_label_at(img: np.ndarray, anchor: tuple[int, int], label: str, color: tuple) -> np.ndarray:
    """Draw text label centered near anchor and clamp inside image bounds."""
    h, w = img.shape[:2]
    ax, ay = anchor

    font = cv2.FONT_HERSHEY_SIMPLEX
    # 2x previous size for stronger readability in videos.
    font_scale = 1.80
    text_thickness = 3
    outline_thickness = 6
    pad = 8

    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)

    # Prefer placing label above anchor; if no room, place below.
    x1 = ax - (tw // 2) - pad
    x2 = x1 + tw + (2 * pad)
    y2 = ay - 12
    y1 = y2 - th - baseline - (2 * pad)

    if y1 < 0:
        y1 = ay + 12
        y2 = y1 + th + baseline + (2 * pad)

    # Clamp horizontally
    if x1 < 0:
        shift = -x1
        x1 += shift
        x2 += shift
    if x2 > w:
        shift = x2 - w
        x1 -= shift
        x2 -= shift

    # Clamp vertically
    if y1 < 0:
        shift = -y1
        y1 += shift
        y2 += shift
    if y2 > h:
        shift = y2 - h
        y1 -= shift
        y2 -= shift

    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, -1)
    text_x = int(x1 + pad)
    text_y = int(y2 - baseline - pad)

    # Black outline + white foreground for strong contrast on any background.
    cv2.putText(
        img,
        label,
        (text_x, text_y),
        font,
        font_scale,
        (0, 0, 0),
        outline_thickness,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        label,
        (text_x, text_y),
        font,
        font_scale,
        (255, 255, 255),
        text_thickness,
        cv2.LINE_AA,
    )
    return img


def annotate_frame(
    frame: np.ndarray,
    result,
    class_names: dict[int, str],
    mask_alpha: float = 0.35,
) -> np.ndarray:
    """
    Annotate one inference result frame with masks and class labels.

    Labels are placed near the most concentrated region of each instance mask,
    avoiding bbox-based placement that can fall outside frame bounds.
    """
    annotated = frame.copy()

    boxes = result.boxes
    masks = result.masks
    h, w = annotated.shape[:2]

    if boxes is None:
        return annotated

    for i, box in enumerate(boxes):
        class_id = int(box.cls[0])
        conf = float(box.conf[0])
        name = class_names.get(class_id, str(class_id))
        color = colour_for(class_id)
        label = f"{name} {conf:.2f}"

        anchor = None
        if masks is not None and i < len(masks):
            mask_data = masks[i].data[0].cpu().numpy()
            annotated = overlay_mask(annotated, mask_data, color, alpha=mask_alpha)
            binary = _mask_to_binary(mask_data, h, w)
            anchor = _label_anchor_from_binary(binary)

        # Fallback to box center if mask is unavailable
        if anchor is None:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            anchor = ((x1 + x2) // 2, (y1 + y2) // 2)

        annotated = _draw_label_at(annotated, anchor, label, color)

    return annotated
