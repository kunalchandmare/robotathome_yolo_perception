#!/usr/bin/env python
"""
Pipeline step: annotation_convert

Convert Robot@Home labels into YOLO format and remap class ids

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
    parser = argparse.ArgumentParser(description="Step: annotation_convert")



    parser.add_argument(
        "--rh_path", type=str, required=True,
        help="Root Robot@Home path (e.g., data/)"
    )




    parser.add_argument(
        "--rgbd_path", type=str, required=True,
        help="Path to RGBD files directory"
    )




    parser.add_argument(
        "--scene_path", type=str, required=True,
        help="Path to scene files directory"
    )




    parser.add_argument(
        "--output_root", type=str, default="yolo",
        help="Output path of YOLO formatted images and labels"
    )




    parser.add_argument(
        "--rgbd_root", type=str, required=True,
        help="Root path used to preserve RGBD relative folder structure"
    )




    parser.add_argument(
        "--epsilon_ratio", type=float, default=0.002,
        help="Polygon simplification factor for mask-to-YOLO conversion"
    )




    parser.add_argument(
        "--labels_root", type=str, default="yolo/labels",
        help="Label directory to remap instance ids to semantic ids"
    )




    parser.add_argument(
        "--mapping_json", type=str, default="yolo/class_id_to_name.json",
        help="JSON mapping file updated/used during remap"
    )




    parser.add_argument(
        "--name_mode", type=str, default="ot",
        help="Robot@Home name mode passed to id2name during remap"
    )




    parser.add_argument(
        "--backup", type=bool, default=True,
        help="Whether to keep .bak label files before remap overwrite"
    )



    args = parser.parse_args()
    go(args)
