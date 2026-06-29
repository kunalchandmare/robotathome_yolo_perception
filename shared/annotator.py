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


def _compute_label_rect(
    anchor: tuple[int, int],
    label: str,
    img_h: int,
    img_w: int,
    font_scale: float | None = None,
) -> tuple[int, int, int, int, float, int, int, int]:
    """
    Compute the label background rectangle for a given anchor and return
    (x1, y1, x2, y2, font_scale, text_thickness, outline_thickness, pad).
    The rect is clamped to image bounds.

    font_scale hint:
        None  → auto (proportional to frame size; ~0.5 for 360p, ~0.9 for 720p,
                       ~1.5 for 1080p)
        0.3–0.5 → small / compact labels
        0.6–0.9 → medium (good for 720p video)
        1.0–1.5 → large (suitable for 1080p+ or presentation output)
        >1.5    → very large; may cover nearby objects
    """
    ax, ay = anchor
    font = cv2.FONT_HERSHEY_SIMPLEX
    if font_scale is None:
        font_scale = max(0.4, min(img_w, img_h) / 700.0)
    text_thickness = max(1, int(font_scale * 1.8))
    outline_thickness = max(2, text_thickness + 2)
    pad = max(4, int(min(img_w, img_h) * 0.008))

    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)

    x1 = ax - (tw // 2) - pad
    x2 = x1 + tw + (2 * pad)
    y2 = ay - 12
    y1 = y2 - th - baseline - (2 * pad)

    if y1 < 0:
        y1 = ay + 12
        y2 = y1 + th + baseline + (2 * pad)

    # Clamp horizontally
    if x1 < 0:
        shift = -x1; x1 += shift; x2 += shift
    if x2 > img_w:
        shift = x2 - img_w; x1 -= shift; x2 -= shift

    # Clamp vertically
    if y1 < 0:
        shift = -y1; y1 += shift; y2 += shift
    if y2 > img_h:
        shift = y2 - img_h; y1 -= shift; y2 -= shift

    return int(x1), int(y1), int(x2), int(y2), font_scale, text_thickness, outline_thickness, pad


def _rects_overlap(a: tuple, b: tuple) -> bool:
    """Return True if two (x1,y1,x2,y2) rects intersect."""
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _resolve_overlaps(
    rects: list,
    img_h: int,
    img_w: int,
    max_iters: int = 10,
) -> list:
    """
    Iteratively nudge label rects apart so they no longer overlap.
    Rects are kept inside image bounds. Each rect is (x1, y1, x2, y2).
    """
    rects = [list(r) for r in rects]  # make mutable

    for _ in range(max_iters):
        moved = False
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                a, b = rects[i], rects[j]
                if not _rects_overlap(a, b):
                    continue

                # Compute overlap on each axis and push apart on the smaller one
                overlap_x = min(a[2], b[2]) - max(a[0], b[0])
                overlap_y = min(a[3], b[3]) - max(a[1], b[1])

                if overlap_y <= overlap_x:
                    # Push vertically: move b down by half overlap, a up by half
                    half = overlap_y // 2 + 1
                    a[1] -= half; a[3] -= half
                    b[1] += half; b[3] += half
                else:
                    # Push horizontally
                    half = overlap_x // 2 + 1
                    a[0] -= half; a[2] -= half
                    b[0] += half; b[2] += half

                # Re-clamp both inside frame
                for r in (a, b):
                    rw = r[2] - r[0]
                    rh = r[3] - r[1]
                    r[0] = max(0, min(r[0], img_w - rw))
                    r[2] = r[0] + rw
                    r[1] = max(0, min(r[1], img_h - rh))
                    r[3] = r[1] + rh

                moved = True

        if not moved:
            break

    return [tuple(r) for r in rects]


def _draw_label_rect(
    img: np.ndarray,
    rect: tuple[int, int, int, int],
    label: str,
    color: tuple,
    font_scale: float,
    text_thickness: int,
    outline_thickness: int,
    pad: int,
) -> np.ndarray:
    """Draw a pre-computed label rect onto img."""
    x1, y1, x2, y2 = rect
    font = cv2.FONT_HERSHEY_SIMPLEX
    (_, th), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)

    cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
    text_x = x1 + pad
    text_y = y2 - baseline - pad

    cv2.putText(img, label, (text_x, text_y), font, font_scale, (0, 0, 0), outline_thickness, cv2.LINE_AA)
    cv2.putText(img, label, (text_x, text_y), font, font_scale, (255, 255, 255), text_thickness, cv2.LINE_AA)
    return img


def _draw_label_at(img: np.ndarray, anchor: tuple[int, int], label: str, color: tuple, font_scale: float | None = None) -> np.ndarray:
    """Draw text label centered near anchor and clamp inside image bounds."""
    h, w = img.shape[:2]
    x1, y1, x2, y2, fs, text_thickness, outline_thickness, pad = _compute_label_rect(anchor, label, h, w, font_scale)
    return _draw_label_rect(img, (x1, y1, x2, y2), label, color, fs, text_thickness, outline_thickness, pad)


def annotate_frame(
    frame: np.ndarray,
    result,
    class_names: dict[int, str],
    mask_alpha: float = 0.35,
    font_scale: float | None = None,
) -> np.ndarray:
    """
    Annotate one inference result frame with masks and class labels.

    Labels are placed near the most concentrated region of each instance mask,
    avoiding bbox-based placement that can fall outside frame bounds.
    Overlapping labels are nudged apart so they remain readable.

    Args:
        frame:       BGR image to annotate.
        result:      Ultralytics inference result for this frame.
        class_names: Mapping of class_id → class name string.
        mask_alpha:  Transparency of the segmentation mask overlay (0=invisible, 1=opaque).
        font_scale:  Label text size.
                     None  → auto-scale proportional to frame resolution.
                     0.3–0.5 → small / compact  (good for thumbnails or small windows)
                     0.6–0.9 → medium            (recommended for 720p video)
                     1.0–1.5 → large             (recommended for 1080p+ or presentations)
                     >1.5   → very large; may obscure nearby detections
    """
    annotated = frame.copy()

    boxes = result.boxes
    masks = result.masks
    h, w = annotated.shape[:2]

    if boxes is None:
        return annotated

    label_info = []  # (rect_params, label, color)

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

        if anchor is None:
            bx1, by1, bx2, by2 = map(int, box.xyxy[0])
            anchor = ((bx1 + bx2) // 2, (by1 + by2) // 2)

        rect_params = _compute_label_rect(anchor, label, h, w, font_scale)
        label_info.append((rect_params, label, color))

    # Resolve overlaps on just the (x1,y1,x2,y2) portion
    raw_rects = [p[0][:4] for p in label_info]
    resolved = _resolve_overlaps(raw_rects, h, w)

    for (rect_params, label, color), rect in zip(label_info, resolved):
        _, _, _, _, font_scale, text_thickness, outline_thickness, pad = rect_params
        _draw_label_rect(annotated, rect, label, color, font_scale, text_thickness, outline_thickness, pad)

    return annotated
