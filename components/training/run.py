#!/usr/bin/env python
"""
Reusable component: training

Train and evaluate YOLO segmentation model

"""
import argparse
import logging
import subprocess
import sys
import training as train
import mlflow

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    # No multiplicity handling; all arguments are single set

    
        with mlflow.start_run():
            mlflow.set_tag("component_name", "training")
            try:
                # --- Pull input artifact (DVC) ---
                # input_artifact_path = "<input_path>"
                # input_dvc_file = "<input_path>.dvc"  # TODO: set actual input DVC file
                # logger.info(f"Pulling input artifact: {input_artifact_path}")
                # subprocess.run(["dvc", "pull", input_dvc_file], check=True)
                # mlflow.set_tag("input_artifact", input_artifact_path)

                train.go(args)

                # --- Log metrics / params (MLflow tracking) ---
                # mlflow.log_param("key", value)
                # mlflow.log_metric("metric", value)

                # --- Track and push output artifact (DVC) ---
                # output_path = "<output_path>"  # TODO: set actual output path
                # logger.info(f"Adding output artifact to DVC: {output_path}")
                # subprocess.run(["dvc", "add", output_path], check=True)
                # mlflow.set_tag("output_artifact", output_path)
                # logger.info("Pushing output artifact to remote DVC storage")
                # subprocess.run(["dvc", "push"], check=True)

                # --- Optionally also log artifact path to MLflow ---
                # mlflow.log_artifact(output_path)

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
        help="Base segmentation model name"
    )
    

    
    parser.add_argument(
        "--run_name", type=str,
        required=False,
        help="Run name under output/runs"
    )
    

    
    parser.add_argument(
        "--epochs", type=int,
        required=False,
        help="Number of training epochs"
    )
    

    
    parser.add_argument(
        "--imgsz", type=int,
        required=False,
        help="Input image size"
    )
    

    
    parser.add_argument(
        "--batch", type=int,
        required=False,
        help="Batch size (-1 enables AutoBatch)"
    )
    

    
    parser.add_argument(
        "--device", type=int,
        required=False,
        help="GPU device index"
    )
    

    
    parser.add_argument(
        "--workers", type=int,
        required=False,
        help="Data loader workers"
    )
    

    args = parser.parse_args()
    sys.exit(go(args))
