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
import cv2
from ultralytics import YOLO

from shared.utils import ensure_dir, load_json
from shared.annotator import annotate_frame


def load_model(model_path: str) -> YOLO:
    """
    Load YOLO model from a local path, HTTP/HTTPS URL, or HuggingFace URI.

    - Local path: validated to exist before loading.
    - HTTP/HTTPS / hf://: passed directly to Ultralytics which handles download.
    """
    _p = str(model_path).strip()
    is_remote = _p.startswith(("http://", "https://", "hf://"))
    if not is_remote:
        path = Path(_p)
        if not path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {path}\n"
                "For cloud storage (S3/GCS/Azure) pre-download with 'dvc pull' "
                "and provide the local path."
            )
    return YOLO(_p)


def load_class_names(mapping_json: str) -> dict[int, str]:
    """Load {class_id: class_name} from JSON."""
    path = Path(mapping_json)
    if path.is_dir():
        raise IsADirectoryError(f"mapping_json is a directory: {path}")
    raw = load_json(path)
    return {int(k): v for k, v in raw.items()}


def run_inference(
    source: str,
    output_dir: str,
    model_path: str,
    mapping_json: str,
    conf_threshold: float = 0.25,
    imgsz: int = 640,
    device: int = 0,
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
    """
    output_dir = Path(output_dir).resolve()
    ensure_dir(output_dir)

    print(f"Loading model: {model_path}")
    model = load_model(model_path)
    class_names = load_class_names(mapping_json)

    print(f"Loaded {len(class_names)} classes from {mapping_json}")
    print(f"Source: {source}")

    results = model.predict(
        source=source,
        conf=conf_threshold,
        imgsz=imgsz,
        device=device,
        stream=True,
        verbose=False,
    )

    saved = 0
    for idx, result in enumerate(results):
        img = result.orig_img.copy()
        annotated = annotate_frame(img, result, class_names, mask_alpha=0.35)

        src_path = getattr(result, "path", None)
        if src_path:
            out_name = Path(src_path).stem + f"_{idx:06d}.jpg"
        else:
            out_name = f"frame_{idx:06d}.jpg"

        cv2.imwrite(str(output_dir / out_name), annotated)
        saved += 1

    print(f"Saved {saved} annotated frame(s) to {output_dir}")


def go(args) -> int:
    run_inference(
        source=args.source,
        output_dir=args.output_dir,
        model_path=args.model_path,
        mapping_json=args.mapping_json,
        conf_threshold=args.conf_threshold,
        imgsz=args.imgsz,
        device=args.device,
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

    args = parser.parse_args()
    sys.exit(go(args))
