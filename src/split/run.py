#!/usr/bin/env python
"""
Pipeline step: split

Split YOLO dataset into train/val/test

"""
import argparse
import logging
import sys

import subprocess
from pathlib import Path

def _bootstrap_project_root() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    return project_root

PROJECT_ROOT = _bootstrap_project_root()

import yolo_data_split as data_split

import mlflow

from shared.mlflow_utils import configure_project_mlflow

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    # No multiplicity handling; all arguments are single set
    configure_project_mlflow(PROJECT_ROOT)

    with mlflow.start_run():
        mlflow.set_tag("component_name", "split")
        # --- Pull input artifact (DVC) ---
        input_files_path = Path(args.source_root)  # get files folder
        subprocess.run(["dvc", "pull", input_files_path], check=True)

        mlflow.set_tag("Yolo annotated Files", input_files_path)
        mlflow.log_param("train_ratio", args.train_ratio)
        mlflow.log_param("val_ratio", args.val_ratio)
        mlflow.log_param("test_ratio", args.test_ratio)
        mlflow.log_param("seed", args.seed)
        mlflow.log_param("stratified", args.use_stratified)
        mlflow.log_param("rarity_threshold", args.rare_threshold)

        #data_split.go(args)

        # --- Track and push output artifact (DVC) ---
        out_files_path = Path(args.output_root)
        mlflow.set_tag("Training Files", out_files_path)
        subprocess.run(["dvc", "add", out_files_path], check=True)
        subprocess.run(["dvc", "push", "-v", out_files_path], check=True)
        mlflow.log_artifact(str(out_files_path.with_suffix(".dvc")),artifact_path="dvc_outputs")
        pass




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step: split")

    
    parser.add_argument(
        "--source_root", type=str,
        required=True,
        help="Source YOLO dataset root containing images/ and labels/"
    )
    

    
    parser.add_argument(
        "--output_root", type=str,
        required=False,
        help="Output split dataset root"
    )
    

    
    parser.add_argument(
        "--train_ratio", type=float,
        required=False,
        help="Training split ratio"
    )
    

    
    parser.add_argument(
        "--val_ratio", type=float,
        required=False,
        help="Validation split ratio"
    )
    

    
    parser.add_argument(
        "--test_ratio", type=float,
        required=False,
        help="Test split ratio"
    )
    

    
    parser.add_argument(
        "--seed", type=int,
        required=False,
        help="Random seed"
    )
    

    
    parser.add_argument(
        "--image_exts", type=str,
        required=False,
        help="Comma-separated image extensions to include"
    )
    

    
    parser.add_argument(
        "--use_stratified", type=lambda x: x.lower() == "true",
        required=False,
        help="Use stratified split with rare-class handling (true/false)"
    )
    

    
    parser.add_argument(
        "--rare_threshold", type=int,
        required=False,
        help="Class frequency threshold treated as rare for stratified split"
    )
    

    args = parser.parse_args()
    go(args)
