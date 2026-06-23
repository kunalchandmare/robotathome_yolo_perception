"""
Segmentation annotation utilities: overlay masks, draw bboxes, add class labels.
"""
import cv2
import numpy as np


# ── Colour palette (BGR) ────────────────────────────────────────────────────
_PALETTE = [
    (56, 168, 0), (255, 64, 64), (0, 140, 255), (255, 200, 0),
    (180, 0, 180), (0, 220, 220), (255, 128, 0), (100, 100, 255),
    (0, 200, 100), (200, 50, 50), (50, 200, 200), (200, 200, 50),
]


def colour_for(class_id: int) -> tuple[int, int, int]:
    """Return BGR colour for a class ID from palette."""
    return _PALETTE[class_id % len(_PALETTE)]


def overlay_mask(img: np.ndarray, mask: np.ndarray, colour: tuple, alpha: float = 0.35) -> np.ndarray:
    """Overlay a segmentation mask on image with given colour and alpha."""
    overlay = img.copy()
    mask_resized = cv2.resize(
        mask,
        (img.shape[1], img.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    binary = (mask_resized > 0.5).astype(np.uint8)
    overlay[binary == 1] = colour
    
    blended = img.copy()
    cv2.addWeighted(overlay, alpha, blended, 1 - alpha, 0, blended)
    return blended


def draw_bbox_with_label(
    img: np.ndarray,
    bbox: tuple[int, int, int, int],
    label: str,
    colour: tuple,
) -> np.ndarray:
    """Draw bounding box and label on image."""
    x1, y1, x2, y2 = bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
    
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(img, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), colour, -1)
    cv2.putText(
        img, label, (x1 + 2, y1 - baseline - 2),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
    )
    return img


def annotate_frame(
    frame: np.ndarray,
    result,
    class_names: dict[int, str],
    mask_alpha: float = 0.35,
) -> np.ndarray:
    """
    Annotate one inference result frame with masks, bboxes, and class labels.
    
    Args:
        frame:       Original image array.
        result:      Ultralytics prediction result object.
        class_names: {class_id: class_name} mapping.
        mask_alpha:  Transparency for mask overlay [0, 1].
    
    Returns:
        Annotated frame with overlays and labels.
    """
    annotated = frame.copy()
    
    boxes = result.boxes
    masks = result.masks
    
    if boxes is not None:
        for i, box in enumerate(boxes):
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            name = class_names.get(class_id, str(class_id))
            colour = colour_for(class_id)
            
            # Overlay mask if available
            if masks is not None and i < len(masks):
                mask_data = masks[i].data[0].cpu().numpy()
                annotated = overlay_mask(annotated, mask_data, colour, alpha=mask_alpha)
            
            # Draw bbox and label
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = f"{name} {conf:.2f}"
            annotated = draw_bbox_with_label(annotated, (x1, y1, x2, y2), label, colour)
    
    return annotated

