"""
Inference component: run YOLO segmentation on images using best.pt or last.pt,
overlay segmentation masks, and label each instance with its class name.

Supported model_path values:
  - Local path:     C:\\...\\best.pt  or  /data/results/runs/.../best.pt
  - HTTP/HTTPS URL: https://example.com/model/best.pt   (Ultralytics auto-downloads)
  - HuggingFace:    hf://org/repo/best.pt               (Ultralytics auto-downloads)
  - Cloud storage (S3/GCS/Azure): pre-download via 'dvc pull' and pass local path
"""
from pathlib import Path
import sys
import cv2
from ultralytics import YOLO
from tqdm import tqdm

from shared.utils import ensure_dir, load_json
from shared.annotator import annotate_frame


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v", ".webm"}


def load_model(model_path: str) -> YOLO:
    """
    Load YOLO model from a local path, HTTP/HTTPS URL, or HuggingFace URI.

    - Local path: resolved to absolute and validated before loading.
    - HTTP/HTTPS / hf://: passed directly to Ultralytics which handles download.
    """
    p = str(model_path).strip()
    is_remote = p.startswith(("http://", "https://", "hf://"))

    if is_remote:
        return YOLO(p)

    local_path = Path(p).resolve()
    if not local_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {local_path}\n"
            "For cloud storage (S3/GCS/Azure) pre-download with 'dvc pull' "
            "and provide the local path."
        )
    return YOLO(str(local_path))


def load_class_names(mapping_json: str) -> dict[int, str]:
    """Load {class_id: class_name} from JSON."""
    path = Path(mapping_json)
    if path.is_dir():
        raise IsADirectoryError(f"mapping_json is a directory: {path}")
    raw = load_json(path)
    return {int(k): v for k, v in raw.items()}


def _normalize_source(source: str):
    """Normalize source for Ultralytics: webcam index, local path, or stream URL."""
    s = str(source).strip()
    if s.isdigit():
        return int(s)
    p = Path(s)
    if p.exists():
        return str(p.resolve())
    return s


def _estimate_total_frames(source_norm):
    """
    Estimate total frames/items for progress percentage.

    Returns:
      - int total when determinable (local image/video/dir)
      - None for live streams / unknown sources
    """
    # Webcam index or non-local URL-like source -> unknown length
    if isinstance(source_norm, int):
        return None

    s = str(source_norm)
    if s.startswith(("rtsp://", "http://", "https://")):
        return None

    p = Path(s)
    if not p.exists():
        return None

    if p.is_dir():
        return sum(1 for f in p.rglob("*") if f.suffix.lower() in _IMAGE_EXTS)

    suffix = p.suffix.lower()
    if suffix in _IMAGE_EXTS:
        return 1

    if suffix in _VIDEO_EXTS:
        cap = cv2.VideoCapture(str(p))
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            return total if total > 0 else None
        finally:
            cap.release()

    return None


def _local_video_path(source_norm):
    """Return local video Path when source is a local video file, else None."""
    if isinstance(source_norm, int):
        return None
    p = Path(str(source_norm))
    if p.exists() and p.is_file() and p.suffix.lower() in _VIDEO_EXTS:
        return p
    return None


def _build_video_writer(video_path: Path, output_dir: Path, frame_shape):
    """Create a VideoWriter for annotated output using source FPS and extension."""
    h, w = frame_shape[:2]

    cap = cv2.VideoCapture(str(video_path))
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
    finally:
        cap.release()
    if fps <= 0:
        fps = 25.0

    suffix = video_path.suffix.lower()
    out_path = output_dir / f"{video_path.stem}_annotated{suffix}"

    # Basic codec mapping by extension.
    if suffix == ".avi":
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
    else:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    # Fallback to mp4 if writer cannot open requested extension.
    if not writer.isOpened():
        out_path = output_dir / f"{video_path.stem}_annotated.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    if not writer.isOpened():
        raise RuntimeError("Failed to create video writer for annotated output.")

    return writer, out_path


def run_inference(
    source: str,
    output_dir: str,
    model_path: str,
    mapping_json: str,
    conf_threshold: float = 0.25,
    imgsz: int = 640,
    device: int = 0,
    font_scale: float | None = None,
):
    """
    Run segmentation inference on any Ultralytics-compatible source and save
    annotated frames to output_dir.

    Args:
        source:         Local image/video path, directory, webcam index ("0"),
                        RTSP URL, or HTTP stream URL.
        output_dir:     Where annotated frames are written.
        model_path:     Local .pt path, HTTP/HTTPS URL, or hf:// URI.
        mapping_json:   Path to class_id_to_name.json.
        conf_threshold: Minimum confidence to show a detection.
        imgsz:          Inference image size.
        device:         GPU device index.
        font_scale:     Label text size. None = auto (proportional to frame resolution).
                        0.3–0.5 → small/compact, 0.6–0.9 → medium (720p),
                        1.0–1.5 → large (1080p+), >1.5 → very large.
    """
    output_dir = Path(output_dir).resolve()
    ensure_dir(output_dir)

    print(f"Loading model: {model_path}", flush=True)
    model = load_model(model_path)
    class_names = load_class_names(mapping_json)
    source_norm = _normalize_source(source)

    print(f"Loaded {len(class_names)} classes from {mapping_json}", flush=True)
    print(f"Source: {source_norm}", flush=True)

    total_frames = _estimate_total_frames(source_norm)
    if total_frames is not None:
        print(f"Total frames/items: {total_frames}", flush=True)
    else:
        print("Total frames/items: unknown (live stream or undetermined source)", flush=True)

    results = model.predict(
        source=source_norm,
        conf=conf_threshold,
        imgsz=imgsz,
        device=device,
        stream=True,
        verbose=False,
    )

    video_src = _local_video_path(source_norm)
    save_as_video = video_src is not None
    writer = None
    video_out_path = None

    if save_as_video:
        print(f"Video source detected: {video_src.name}", flush=True)
        print("Output mode: annotated video", flush=True)
    else:
        print("Output mode: annotated frames", flush=True)

    saved = 0
    pbar = tqdm(
        results,
        total=total_frames,
        desc="Running inference",
        unit="frame",
        dynamic_ncols=True,
        file=sys.stdout,
    )
    try:
        for idx, result in enumerate(pbar):
            img = result.orig_img.copy()
            annotated = annotate_frame(img, result, class_names, mask_alpha=0.35, font_scale=font_scale)

            if save_as_video:
                if writer is None:
                    writer, video_out_path = _build_video_writer(video_src, output_dir, annotated.shape)
                writer.write(annotated)
            else:
                src_path = getattr(result, "path", None)
                if src_path:
                    out_name = Path(src_path).stem + f"_{idx:06d}.jpg"
                else:
                    out_name = f"frame_{idx:06d}.jpg"
                cv2.imwrite(str(output_dir / out_name), annotated)

            saved += 1
    finally:
        if writer is not None:
            writer.release()

    if save_as_video and video_out_path is not None:
        print(f"Saved {saved} frame(s) into annotated video: {video_out_path}", flush=True)
    else:
        print(f"Saved {saved} annotated frame(s) to {output_dir}", flush=True)


def go(args) -> int:
    run_inference(
        source=args.source,
        output_dir=args.output_dir,
        model_path=args.model_path,
        mapping_json=args.mapping_json,
        conf_threshold=args.conf_threshold,
        imgsz=args.imgsz,
        device=args.device,
        font_scale=args.font_scale,
    )
    return 0


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="YOLO segmentation inference")
    parser.add_argument("--source",         type=str,   required=True,  help="Image/video/dir/webcam index/stream URL")
    parser.add_argument("--output_dir",     type=str,   required=True,  help="Output annotated frames directory")
    parser.add_argument("--model_path",     type=str,   required=True,  help="Local .pt path, HTTP/HTTPS URL, or hf:// URI")
    parser.add_argument("--mapping_json",   type=str,   required=True,  help="Path to class_id_to_name.json")
    parser.add_argument("--conf_threshold", type=float, default=0.25,   help="Detection confidence threshold")
    parser.add_argument("--imgsz",          type=int,   default=640,    help="Inference image size")
    parser.add_argument("--device",         type=int,   default=0,      help="GPU device index")
    parser.add_argument(
        "--font_scale",
        type=float,
        default=None,
        help=(
            "Label font size (default: auto, proportional to frame resolution). "
            "Suggested ranges: 0.3-0.5 small/compact, 0.6-0.9 medium (720p), "
            "1.0-1.5 large (1080p+), >1.5 very large."
        ),
    )

    args = parser.parse_args()
    sys.exit(go(args))
