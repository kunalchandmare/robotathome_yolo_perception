#!/usr/bin/env python
"""
Pipeline step: training

Train and evaluate YOLO segmentation model

"""
import argparse
import logging

import subprocess
from pathlib import Path

import mlflow

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    # No multiplicity handling; all arguments are single set



    with mlflow.start_run():
        # --- Pull input artifact (DVC) ---
        # subprocess.run(["dvc", "pull", "<file>.dvc"], check=True)

        # TODO: implement step logic here

        # --- Track and push output artifact (DVC) ---
        # subprocess.run(["dvc", "add", "<output_path>"], check=True)
        # subprocess.run(["dvc", "push"], check=True)

        pass




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step: training")

    
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
    go(args)
