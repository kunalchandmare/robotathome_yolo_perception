#!/usr/bin/env python
"""
Pipeline step: download

Download and extract Robot@Home archives or one custom source

"""
import argparse
import logging

import subprocess
from pathlib import Path

import mlflow
import download as db_downloader

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    # No multiplicity handling; all arguments are single set

    with mlflow.start_run():
        # --- Pull input artifact (DVC) ---
        # subprocess.run(["dvc", "pull", "<file>.dvc"], check=True)

        # Downloads robot at Home Dataset based on CLI args.
        db_downloader.go(args)

        extract_path = Path(
            args.extract_root) / args.dataset_extract_to if args.dataset_extract_to else args.extract_root
        # --- Track and push output artifact (DVC) ---
        subprocess.run(["dvc", "add", extract_path.resolve()], check=True)
        # subprocess.run(["dvc", "push"], check=True)

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
        help="Root directory where archives are extracted"
    )
    

    
    parser.add_argument(
        "--force_download", type=lambda x: x.lower() == "true",
        required=False,
        help="Force re-download even if file already exists (true/false)"
    )
    

    
    parser.add_argument(
        "--dataset_url", type=str,
        required=False,
        help="Optional custom dataset URL (must be paired with dataset_filename and dataset_md5)"
    )
    

    
    parser.add_argument(
        "--dataset_filename", type=str,
        required=False,
        help="Optional custom downloaded filename (must be paired with dataset_url and dataset_md5)"
    )
    

    
    parser.add_argument(
        "--dataset_md5", type=str,
        required=False,
        help="Optional custom MD5 checksum (must be paired with dataset_url and dataset_filename)"
    )
    

    
    parser.add_argument(
        "--dataset_extract_to", type=str,
        required=False,
        help="Optional extraction path relative to extract_root for custom archive"
    )
    

    args = parser.parse_args()
    go(args)
