#!/usr/bin/env python
"""
Pipeline step: download

Download and extract Robot@Home archives or one custom source

"""
import argparse
import logging
import subprocess
import sys

from pathlib import Path


import mlflow


def _bootstrap_project_root() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    return project_root

PROJECT_ROOT = _bootstrap_project_root()

from shared.helpers import parse_bool, parse_optional_str
from shared.mlflow_utils import configure_project_mlflow
import download as downloader

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def go(args):
    # No multiplicity handling; all arguments are single set
    configure_project_mlflow(PROJECT_ROOT)

    with mlflow.start_run():
        mlflow.set_tag("component_name", "download")
        # --- Pull input artifact (DVC) ---
        # input_artifact_path = "<input_path>"
        # subprocess.run(["dvc", "pull", "<input_path>.dvc"], check=True)
        # mlflow.set_tag("input_artifact", input_artifact_path)

        logger.info(f"Downloading {args.dataset_filename} from {args.dataset_url} to {args.out_dir}")

        mlflow.set_tag("source_url", args.dataset_url)
        mlflow.set_tag("file_name", args.dataset_filename)
        mlflow.set_tag("file_hash", args.dataset_md5)
        mlflow.set_tag("snapshot_name", f"{Path(args.dataset_filename).stem}_{args.dataset_md5[:8]}")
        # --- Log metrics / params (MLflow tracking) ---
        mlflow.log_param("Forced Download", args.force_download)
        # mlflow.log_metric("metric", value)

        #downloader.go(args)

        # --- Track and push output artifact (DVC) ---
        output_artifact_path = Path(args.extract_root)
        if args.dataset_extract_to=='.':
            data_name = "Raw SQL DB"
            output_artifact_path = output_artifact_path / "rh.db"
            downloaded_artefact = str(output_artifact_path) + ".dvc"
        else:
            data_name = "Raw Files"
            output_artifact_path = output_artifact_path / args.dataset_extract_to
            downloaded_artefact = str(output_artifact_path.with_suffix(".dvc"))

        subprocess.run(["dvc", "add", output_artifact_path], check=True)
        subprocess.run(["dvc", "push", "-v",output_artifact_path], check=True)
        mlflow.set_tag(data_name, output_artifact_path)
        logger.info(f"Pushing {downloaded_artefact} to DVC Remote")
        mlflow.log_artifact(downloaded_artefact, artifact_path="dvc_outputs")

        pass




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step: download")

    
    parser.add_argument(
        "--out_dir", type=str,
        required=True,
        help="Directory where files are downloaded"
    )
    

    
    parser.add_argument(
        "--extract_root", type=str,
        required=False,
        default='.',
        help="Root directory where archives are extracted"
    )
    

    
    parser.add_argument(
        "--force_download", type=parse_bool,
        required=False,
        default=False,
        help="Force re-download even if file already exists (true/false)"
    )
    

    
    parser.add_argument(
        "--dataset_url", type=parse_optional_str,
        required=False,
        default=None,
        help="Optional custom dataset URL (must be paired with dataset_filename and dataset_md5)"
    )
    

    
    parser.add_argument(
        "--dataset_filename", type=parse_optional_str,
        required=False,
        default=None,
        help="Optional custom downloaded filename (must be paired with dataset_url and dataset_md5)"
    )
    

    
    parser.add_argument(
        "--dataset_md5", type=parse_optional_str,
        required=False,
        default=None,
        help="Optional custom MD5 checksum (must be paired with dataset_url and dataset_filename)"
    )
    

    
    parser.add_argument(
        "--dataset_extract_to", type=parse_optional_str,
        required=False,
        default=None,
        help="Optional extraction path relative to extract_root for custom archive"
    )
    

    args = parser.parse_args()
    sys.exit(go(args))
