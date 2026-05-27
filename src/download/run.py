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

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

import download as rh_db

def go(args):

    with mlflow.start_run():
        # --- Pull input artifact (DVC) ---
        # subprocess.run(["dvc", "pull", "<file>.dvc"], check=True)

        # TODO: implement step logic here

        force_download_flag = None
        if args.force_download:
            force_download_flag = True
        else:
            force_download_flag = False

        custom_fields = [args.dataset_url, args.dataset_filename, args.dataset_md5]
        has_any_custom = any(v is not None for v in custom_fields)
        has_all_required_custom = all(v is not None for v in custom_fields)

        if has_any_custom and not has_all_required_custom:
            parser.error("--dataset-url, --dataset-filename, and --dataset-md5 must be provided together.")

        source_specs = None
        if has_all_required_custom:
            source_specs = [
                {
                    "url": args.dataset_url,
                    "filename": args.dataset_filename,
                    "md5": args.dataset_md5,
                    **({"extract_to": args.dataset_extract_to} if args.dataset_extract_to else {}),
                }
            ]
            print("Using one custom source from CLI arguments.")
        else:
            print("Using built-in Robot@Home default sources.")

        try:
            processed = rh_db.download_rh(
                out_dir=args.out_dir,
                extract_root=args.extract_root,
                force_download=force_download_flag,
                source_specs=source_specs,
            )
            print(f"Successfully processed: {', '.join(processed)}")
        except Exception as e:
            print(f"Download failed: {e}")
            raise SystemExit(1)

        if args.query_sample:
            print("\n" + "=" * 60)
            print("Querying sample annotation...")
            print("=" * 60)
            rh_db.query_sample_annotation()

        extract_path = Path(args.extract_root) / args.dataset_extract_to if args.dataset_extract_to else args.extract_root
        # --- Track and push output artifact (DVC) ---
        subprocess.run(["dvc", "add", extract_path.resolve()], check=True)
        # subprocess.run(["dvc", "push"], check=True)

        pass



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step: download")



    parser.add_argument(
        "--out_dir", type=str, required=True,
        help="Directory where files are downloaded"
    )




    parser.add_argument(
        "--extract_root", type=str, default=".",
        help="Root directory where archives are extracted"
    )




    parser.add_argument(
        "--force_download", type=bool, default=False,
        help="Force re-download even if file already exists"
    )




    parser.add_argument(
        "--dataset_url", type=str, default=None,
        help="Optional custom dataset URL (must be paired with dataset_filename and dataset_md5)"
    )




    parser.add_argument(
        "--dataset_filename", type=str, default=None,
        help="Optional custom downloaded filename (must be paired with dataset_url and dataset_md5)"
    )




    parser.add_argument(
        "--dataset_md5", type=str, default=None,
        help="Optional custom MD5 checksum (must be paired with dataset_url and dataset_filename)"
    )




    parser.add_argument(
        "--dataset_extract_to", type=str, default=None,
        help="Optional extraction path relative to extract_root for custom archive"
    )

    parser.add_argument("--query-sample", action="store_true", help="Query and visualize a sample annotation")



    args = parser.parse_args()
    go(args)
