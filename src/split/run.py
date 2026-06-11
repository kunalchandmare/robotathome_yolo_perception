#!/usr/bin/env python
"""
Pipeline step: split

Split YOLO dataset into train/val/test

"""
import argparse
import logging

import subprocess
from pathlib import Path

import yolo_data_split as data_split

import mlflow

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    # No multiplicity handling; all arguments are single set



    with mlflow.start_run():
        # --- Pull input artifact (DVC) ---
        # subprocess.run(["dvc", "pull", "<file>.dvc"], check=True)

        data_split.go(args)

        # --- Track and push output artifact (DVC) ---
        # subprocess.run(["dvc", "add", "<output_path>"], check=True)
        # subprocess.run(["dvc", "push"], check=True)

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
