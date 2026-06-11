#!/usr/bin/env python
"""
Pipeline step: annotation_convert

Convert Robot@Home labels into YOLO format and remap class ids

"""
import argparse
import logging
import sys
import subprocess
import annotation_convert as annotation_convert
import mlflow

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    # No multiplicity handling; all arguments are single set



    with mlflow.start_run():
        mlflow.set_tag("step_name", "annotation_convert")
        # --- Pull input artifact (DVC) ---
        # input_artifact_path = "<input_path>"
        # subprocess.run(["dvc", "pull", "<input_path>.dvc"], check=True)
        # mlflow.set_tag("input_artifact", input_artifact_path)

        annotation_convert.go(args)

        # --- Log metrics / params (MLflow tracking) ---
        # mlflow.log_param("key", value)
        # mlflow.log_metric("metric", value)

        # --- Track and push output artifact (DVC) ---
        # output_artifact_path = "<output_path>"
        # subprocess.run(["dvc", "add", output_artifact_path], check=True)
        # subprocess.run(["dvc", "push"], check=True)
        # mlflow.set_tag("output_artifact", output_artifact_path)
        # mlflow.log_artifact(output_artifact_path)

        pass




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step: annotation_convert")

    
    parser.add_argument(
        "--rh_path", type=str,
        required=True,
        help="Root Robot@Home path (e.g., data/)"
    )
    

    
    parser.add_argument(
        "--rgbd_path", type=str,
        required=True,
        help="Path to RGBD files directory for Loading RobotAtHome dataset"
    )
    

    
    parser.add_argument(
        "--scene_path", type=str,
        required=True,
        help="Path to scene files directory"
    )
    

    
    parser.add_argument(
        "--output_root", type=str,
        required=False,
        help="Output path of YOLO formatted images and labels"
    )
    

    
    parser.add_argument(
        "--epsilon_ratio", type=float,
        required=False,
        help="Polygon simplification factor for mask-to-YOLO conversion"
    )
    

    
    parser.add_argument(
        "--labels_root", type=str,
        required=False,
        help="Label directory to remap instance ids to semantic ids"
    )
    

    
    parser.add_argument(
        "--mapping_json", type=str,
        required=False,
        help="JSON mapping file updated/used during remap"
    )
    

    
    parser.add_argument(
        "--name_mode", type=str,
        required=False,
        help="Robot@Home name mode passed to id2name during remap"
    )
    

    
    parser.add_argument(
        "--force_convert", type=lambda x: x.lower() == "true",
        required=False,
        help="Forced conversion and generation of JSON map even if files exists (true/false)"
    )
    

    
    parser.add_argument(
        "--backup", type=lambda x: x.lower() == "true",
        required=False,
        help="Whether to keep .bak label files before remap overwrite (true/false)"
    )
    

    args = parser.parse_args()
    sys.exit(go(args))
