"""
Pipeline orchestrator for robotathome_yolo_perception.
Driven by Hydra config (params.yaml). Run with:

    mlflow run . -P steps=all
    mlflow run . -P steps=download,training
    mlflow run . -P hydra_options="training.epochs=20"
"""
import os
import sys
import mlflow
import hydra
from omegaconf import DictConfig


_steps = [
    "download",
    "annotation_convert",
    "split",
    "training",
]

@hydra.main(version_base=None, config_name="params", config_path=".")
def go(config: DictConfig):
    main_cfg = config.get("main", {})
    steps_par = str(main_cfg.get("steps", "all"))
    active_steps = [s.strip() for s in steps_par.split(",") if s.strip()] if steps_par != "all" else _steps

    # ── src/ steps ─────────────────────────────────────────────────────────────
    if "download" in active_steps:
        step_cfg_runtime = config.get("download", {})
        for entry in step_cfg_runtime.get("dataset_sources", []):
            mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "download"),
                "main",
                env_manager="local",
                parameters={
                    "dataset_extract_to": str((entry or {}).get("dataset_extract_to", None)),
                    "dataset_filename": str((entry or {}).get("dataset_filename", None)),
                    "dataset_md5": str((entry or {}).get("dataset_md5", None)),
                    "dataset_url": str((entry or {}).get("dataset_url", None)),
                    "extract_root": str((entry or {}).get("extract_root", '.')),
                    "force_download": str((entry or {}).get("force_download", False)),
                    "out_dir": str((entry or {}).get("out_dir", '')),
                },
            )
    if "annotation_convert" in active_steps:
        step_cfg_runtime = config.get("annotation_convert", {})
        mlflow.run(
            os.path.join(hydra.utils.get_original_cwd(), "src", "annotation_convert"),
            "main",
            env_manager="local",
            parameters={
                "rh_path": str(step_cfg_runtime.get("rh_path", '')),
                "rgbd_path": str(step_cfg_runtime.get("rgbd_path", '')),
                "scene_path": str(step_cfg_runtime.get("scene_path", '')),
                "output_root": str(step_cfg_runtime.get("output_root", 'yolo')),
                "rgbd_root": str(step_cfg_runtime.get("rgbd_root", '')),
                "epsilon_ratio": str(step_cfg_runtime.get("epsilon_ratio", 0.002)),
                "labels_root": str(step_cfg_runtime.get("labels_root", 'yolo/labels')),
                "mapping_json": str(step_cfg_runtime.get("mapping_json", 'yolo/class_id_to_name.json')),
                "name_mode": str(step_cfg_runtime.get("name_mode", 'ot')),
                "backup": str(step_cfg_runtime.get("backup", True)),
            },
        )
    if "split" in active_steps:
        step_cfg_runtime = config.get("split", {})
        mlflow.run(
            os.path.join(hydra.utils.get_original_cwd(), "src", "split"),
            "main",
            env_manager="local",
            parameters={
                "source_root": str(step_cfg_runtime.get("source_root", '')),
                "output_root": str(step_cfg_runtime.get("output_root", 'yolo_split_stratified')),
                "train_ratio": str(step_cfg_runtime.get("train_ratio", 0.8)),
                "val_ratio": str(step_cfg_runtime.get("val_ratio", 0.1)),
                "test_ratio": str(step_cfg_runtime.get("test_ratio", 0.1)),
                "seed": str(step_cfg_runtime.get("seed", 42)),
                "image_exts": str(step_cfg_runtime.get("image_exts", '.jpg,.jpeg,.png')),
                "use_stratified": str(step_cfg_runtime.get("use_stratified", True)),
                "rare_threshold": str(step_cfg_runtime.get("rare_threshold", 20)),
            },
        )
    if "training" in active_steps:
        step_cfg_runtime = config.get("training", {})
        mlflow.run(
            os.path.join(hydra.utils.get_original_cwd(), "src", "training"),
            "main",
            env_manager="local",
            parameters={
                "dataset_root": str(step_cfg_runtime.get("dataset_root", '')),
                "output_root": str(step_cfg_runtime.get("output_root", 'output')),
                "mapping_json": str(step_cfg_runtime.get("mapping_json", '')),
                "model_name": str(step_cfg_runtime.get("model_name", 'yolo11s-seg.pt')),
                "run_name": str(step_cfg_runtime.get("run_name", 'robotathome_seg_strat')),
                "epochs": str(step_cfg_runtime.get("epochs", 50)),
                "imgsz": str(step_cfg_runtime.get("imgsz", 640)),
                "batch": str(step_cfg_runtime.get("batch", -1)),
                "device": str(step_cfg_runtime.get("device", 0)),
                "workers": str(step_cfg_runtime.get("workers", 10)),
            },
        )
    # ── components/ ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sys.exit(go())
