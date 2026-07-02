#!/usr/bin/env python
"""
Reusable component: inference

Run YOLO segmentation inference on any source (image dir, video file, webcam, RTSP stream) and save annotated output

"""
import argparse
import logging
from pathlib import Path
import subprocess
import sys

import mlflow

def _bootstrap_project_root() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    return project_root

PROJECT_ROOT = _bootstrap_project_root()

import inference as infer

from shared.utils import none_if_null
from shared.mlflow_utils import configure_project_mlflow

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    # No multiplicity handling; all arguments are single set
    configure_project_mlflow(PROJECT_ROOT)

    args.font_scale = none_if_null(args.font_scale)
    if args.font_scale is not None:
        try:
            args.font_scale = float(args.font_scale)
        except (TypeError, ValueError) as exc:
            raise SystemExit(
                f"Invalid --font_scale value {args.font_scale!r}. "
                "Use a number like 0.8 or leave it empty/null for auto."
            ) from exc

    with mlflow.start_run():
        mlflow.set_tag("component_name", "inference")
        try:
            # --- Pull input artifact (DVC) ---
            src_path = Path(args.source)
            mlflow.log_artifact(str(src_path), artifact_path="inference_source")

            model_path = Path(args.model_path)
            logger.info(f"Pulling Inference Model: {model_path}")
            subprocess.run(["dvc", "pull", model_path], check=True)
            mlflow.log_artifact(str(model_path), artifact_path="Models")

            mapping_json_path = Path(args.mapping_json)
            subprocess.run(["dvc", "pull", mapping_json_path], check=True)
            mlflow.set_tag("Class Names to Id", mapping_json_path.stem)

            #infer.go(args)

            # --- Log metrics / params (MLflow tracking) ---
            mlflow.log_param("conf_threshold", args.conf_threshold)
            mlflow.log_param("frame_size", args.imgsz)
            mlflow.log_param("device", args.device)
            # mlflow.log_metric("metric", value)

            # --- Track and push output artifact (DVC) ---
            output_path = Path(args.output_dir)
            logger.info(f"Adding output artifact to DVC: {output_path}")
            subprocess.run(["dvc", "add", str(output_path)], check=True)
            # mlflow.set_tag("output_artifact", output_path)
            logger.info("Pushing output artifact to remote DVC storage")
            subprocess.run(["dvc", "push", "-v", output_path], check=True)

            # --- Optionally also log artifact path to MLflow ---
            mlflow.log_artifact(str(output_path), artifact_path="inference_result")
        except subprocess.CalledProcessError as e:
            logger.error(f"DVC command failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Component: inference")

    
    parser.add_argument(
        "--source", type=str,
        required=True,
        help="Inference source: local image/video path, directory, webcam index (0), or RTSP/HTTP stream URL"
    )
    

    
    parser.add_argument(
        "--output_dir", type=str,
        required=True,
        help="Output directory for annotated frames or images"
    )
    

    
    parser.add_argument(
        "--model_path", type=str,
        required=True,
        help="Path or URL to the model checkpoint (.pt file). See README for supported formats."
    )
    

    
    parser.add_argument(
        "--mapping_json", type=str,
        required=True,
        help="Path to class_id_to_name.json for class name overlay labels"
    )
    

    
    parser.add_argument(
        "--conf_threshold", type=float,
        required=True,
        help="Minimum detection confidence to annotate"
    )
    

    
    parser.add_argument(
        "--imgsz", type=int,
        required=True,
        help="Inference image size"
    )
    

    
    parser.add_argument(
        "--device", type=int,
        required=True,
        help="GPU device index"
    )
    

    
    parser.add_argument(
        "--font_scale", type=str,
        required=False,
        default="",
        help="Label font size (default: auto, proportional to frame resolution). Suggested ranges: 0.3-0.5 small/compact, 0.6-0.9 medium (720p), 1.0-1.5 large (1080p+), >1.5 very large. Empty/null-like values are treated as auto."
    )
    

    args = parser.parse_args()
    sys.exit(go(args))
