#!/usr/bin/env python
"""
Reusable component: training

Train and evaluate YOLO segmentation model

"""
import argparse
import logging
import subprocess
import sys
from pathlib import Path


def _bootstrap_project_root() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    return project_root

PROJECT_ROOT = _bootstrap_project_root()

import mlflow
import training as train

from shared.mlflow_utils import configure_project_mlflow



logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def go(args):
    # No multiplicity handling; all arguments are single set
    configure_project_mlflow(PROJECT_ROOT)

    with mlflow.start_run():
        mlflow.set_tag("component_name", "training")
        try:
            # --- Pull input artifact (DVC) ---
            input_artifact_path = args.dataset_root
            input_dvc_file = "dvc_files/training.dvc"
            logger.info(f"Pulling input artifact: {input_artifact_path}")
            subprocess.run(["dvc", "pull", input_dvc_file], check=True)
            mlflow.set_tag("Training_data", input_artifact_path)

            # --- Log metrics / params (MLflow tracking) ---
            mlflow.log_param("Epochs", args.epochs)
            mlflow.log_param("Image Size", args.imgsz)
            mlflow.log_param("Batch", args.batch)
            mlflow.log_param("Workers", args.workers)
            # mlflow.log_metric("metric", value)

            train.go(args)

            # --- Track and push output artifact (DVC) ---
            output_path = args.output_root  # TODO: set actual output path
            logger.info(f"Adding output results to DVC: {output_path}")
            subprocess.run(["dvc", "add", "--file", "dvc_files/results.dvc", output_path], check=True)
            mlflow.set_tag("Training Results", output_path)
            logger.info("Pushing output artifact to remote DVC storage")
            subprocess.run(["dvc", "push"], check=True)

            # --- Optionally also log artifact path to MLflow ---
            #mlflow.log_artifact(output_path)

        except subprocess.CalledProcessError as e:
            logger.error(f"DVC command failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Component: training")

    
    parser.add_argument(
        "--dataset_root", type=str,
        required=True,
        help="Split dataset root used for training"
    )
    

    
    parser.add_argument(
        "--output_root", type=str,
        required=False,
        default='output',
        help="Directory for run artifacts and metrics"
    )
    

    
    parser.add_argument(
        "--mapping_json", type=str,
        required=True,
        help="class_id_to_name.json used to populate data.yaml (YOLO metadata) names"
    )
    

    
    parser.add_argument(
        "--model_name", type=str,
        required=False,
        default='yolo11s-seg.pt',
        help="Base segmentation model name"
    )
    

    
    parser.add_argument(
        "--run_name", type=str,
        required=False,
        default='robotathome_seg_strat',
        help="Run name under output/runs"
    )
    

    
    parser.add_argument(
        "--epochs", type=int,
        required=False,
        default=50,
        help="Number of training epochs"
    )
    

    
    parser.add_argument(
        "--imgsz", type=int,
        required=False,
        default=640,
        help="Input image size"
    )
    

    
    parser.add_argument(
        "--batch", type=int,
        required=False,
        default=-1,
        help="Batch size (-1 enables AutoBatch)"
    )
    

    
    parser.add_argument(
        "--device", type=int,
        required=False,
        default=0,
        help="GPU device index"
    )
    

    
    parser.add_argument(
        "--workers", type=int,
        required=False,
        default=10,
        help="Data loader workers"
    )
    

    args = parser.parse_args()
    sys.exit(go(args))
