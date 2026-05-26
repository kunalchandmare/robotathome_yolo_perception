#!/usr/bin/env python
"""
Pipeline step: split

Split YOLO dataset into train/val/test

"""
import argparse
import logging

import subprocess
from pathlib import Path

import mlflow

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    with mlflow.start_run():
        # --- Pull input artifact (DVC) ---
        # subprocess.run(["dvc", "pull", "<file>.dvc"], check=True)

        # TODO: implement step logic here

        # --- Track and push output artifact (DVC) ---
        # subprocess.run(["dvc", "add", "<output_path>"], check=True)
        # subprocess.run(["dvc", "push"], check=True)

        pass



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step: split")



    parser.add_argument(
        "--source_root", type=str, required=True,
        help="Source YOLO dataset root containing images/ and labels/"
    )




    parser.add_argument(
        "--output_root", type=str, default="yolo_split_stratified",
        help="Output split dataset root"
    )




    parser.add_argument(
        "--train_ratio", type=float, default=0.8,
        help="Training split ratio"
    )




    parser.add_argument(
        "--val_ratio", type=float, default=0.1,
        help="Validation split ratio"
    )




    parser.add_argument(
        "--test_ratio", type=float, default=0.1,
        help="Test split ratio"
    )




    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )




    parser.add_argument(
        "--image_exts", type=str, default=".jpg,.jpeg,.png",
        help="Comma-separated image extensions to include"
    )




    parser.add_argument(
        "--use_stratified", type=bool, default=True,
        help="Use stratified split with rare-class handling"
    )




    parser.add_argument(
        "--rare_threshold", type=int, default=20,
        help="Class frequency threshold treated as rare for stratified split"
    )



    args = parser.parse_args()
    go(args)
