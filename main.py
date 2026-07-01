"""
Pipeline orchestrator for robotathome_yolo_perception.
Driven by Hydra config (params.yaml). Run with:

    mlflow run . -P steps=all
    mlflow run . -P steps=download,training
    mlflow run . -P hydra_options="training.epochs=20"
"""
import os
import sys
from pathlib import Path

def _bootstrap_project_root() -> Path:
    # main.py is located at repo root; use its parent directly.
    project_root = Path(__file__).resolve().parent
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    return project_root

PROJECT_ROOT = _bootstrap_project_root()

import mlflow
import hydra
from omegaconf import DictConfig

from shared.mlflow_utils import configure_project_mlflow



def _serialize_param(value):
    return "" if value is None else str(value)


_steps = [
    "download",
    "annotation_convert",
    "split",
    "training",
    "inference",
]

@hydra.main(version_base=None, config_name="params", config_path=".")
def go(config: DictConfig):
    main_cfg = config.get("main", {})
    _, experiment_name = configure_project_mlflow(
        project_root=PROJECT_ROOT,
        experiment_name=main_cfg.get("experiment_name"),
    )
    steps_par = str(main_cfg.get("steps", "all"))
    active_steps = [s.strip() for s in steps_par.split(",") if s.strip()] if steps_par != "all" else _steps

    # ── src/ steps ─────────────────────────────────────────────────────────────
    if "download" in active_steps:
        step_cfg_runtime = config.get("download", {})
        for entry in step_cfg_runtime.get("dataset_sources", []):
            mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "download"),
                "main",
                experiment_name=experiment_name,
                env_manager="local",
                parameters={
                    "dataset_extract_to": _serialize_param((entry or {}).get("dataset_extract_to", None)),
                    "dataset_filename": _serialize_param((entry or {}).get("dataset_filename", None)),
                    "dataset_md5": _serialize_param((entry or {}).get("dataset_md5", None)),
                    "dataset_url": _serialize_param((entry or {}).get("dataset_url", None)),
                    "extract_root": _serialize_param((entry or {}).get("extract_root", '.')),
                    "force_download": _serialize_param((entry or {}).get("force_download", False)),
                    "out_dir": _serialize_param((entry or {}).get("out_dir", '')),
                },
            )
    if "annotation_convert" in active_steps:
        step_cfg_runtime = config.get("annotation_convert", {})
        mlflow.run(
            os.path.join(hydra.utils.get_original_cwd(), "src", "annotation_convert"),
            "main",
            experiment_name=experiment_name,
            env_manager="local",
            parameters={
                "rh_path": _serialize_param(step_cfg_runtime.get("rh_path", '')),
                "rgbd_path": _serialize_param(step_cfg_runtime.get("rgbd_path", '')),
                "scene_path": _serialize_param(step_cfg_runtime.get("scene_path", '')),
                "output_root": _serialize_param(step_cfg_runtime.get("output_root", 'yolo')),
                "epsilon_ratio": _serialize_param(step_cfg_runtime.get("epsilon_ratio", 0.002)),
                "labels_root": _serialize_param(step_cfg_runtime.get("labels_root", 'yolo/labels')),
                "mapping_json": _serialize_param(step_cfg_runtime.get("mapping_json", 'yolo/class_id_to_name.json')),
                "name_mode": _serialize_param(step_cfg_runtime.get("name_mode", 'ot')),
                "force_convert": _serialize_param(step_cfg_runtime.get("force_convert", False)),
                "backup": _serialize_param(step_cfg_runtime.get("backup", True)),
            },
        )
    if "split" in active_steps:
        step_cfg_runtime = config.get("split", {})
        mlflow.run(
            os.path.join(hydra.utils.get_original_cwd(), "src", "split"),
            "main",
            experiment_name=experiment_name,
            env_manager="local",
            parameters={
                "source_root": _serialize_param(step_cfg_runtime.get("source_root", '')),
                "output_root": _serialize_param(step_cfg_runtime.get("output_root", 'yolo_split_stratified')),
                "train_ratio": _serialize_param(step_cfg_runtime.get("train_ratio", 0.8)),
                "val_ratio": _serialize_param(step_cfg_runtime.get("val_ratio", 0.1)),
                "test_ratio": _serialize_param(step_cfg_runtime.get("test_ratio", 0.1)),
                "seed": _serialize_param(step_cfg_runtime.get("seed", 42)),
                "image_exts": _serialize_param(step_cfg_runtime.get("image_exts", '.jpg,.jpeg,.png')),
                "use_stratified": _serialize_param(step_cfg_runtime.get("use_stratified", True)),
                "rare_threshold": _serialize_param(step_cfg_runtime.get("rare_threshold", 20)),
            },
        )
    # ── components/ ────────────────────────────────────────────────────────────
    if "training" in active_steps:
        comp_cfg_runtime = config.get("training", {})
        mlflow.run(
            os.path.join(hydra.utils.get_original_cwd(), "components", "training"),
            "main",
            experiment_name=experiment_name,
            env_manager="local",
            parameters={
                "dataset_root": _serialize_param(comp_cfg_runtime.get("dataset_root", '')),
                "output_root": _serialize_param(comp_cfg_runtime.get("output_root", 'output')),
                "mapping_json": _serialize_param(comp_cfg_runtime.get("mapping_json", '')),
                "model_name": _serialize_param(comp_cfg_runtime.get("model_name", 'yolo11s-seg.pt')),
                "run_name": _serialize_param(comp_cfg_runtime.get("run_name", 'robotathome_seg_strat')),
                "epochs": _serialize_param(comp_cfg_runtime.get("epochs", 50)),
                "imgsz": _serialize_param(comp_cfg_runtime.get("imgsz", 640)),
                "batch": _serialize_param(comp_cfg_runtime.get("batch", -1)),
                "device": _serialize_param(comp_cfg_runtime.get("device", 0)),
                "workers": _serialize_param(comp_cfg_runtime.get("workers", 10)),
            },
        )
    if "inference" in active_steps:
        comp_cfg_runtime = config.get("inference", {})
        mlflow.run(
            os.path.join(hydra.utils.get_original_cwd(), "components", "inference"),
            "main",
            experiment_name=experiment_name,
            env_manager="local",
            parameters={
                "source": _serialize_param(comp_cfg_runtime.get("source", '')),
                "output_dir": _serialize_param(comp_cfg_runtime.get("output_dir", '')),
                "model_path": _serialize_param(comp_cfg_runtime.get("model_path", '')),
                "mapping_json": _serialize_param(comp_cfg_runtime.get("mapping_json", '')),
                "conf_threshold": _serialize_param(comp_cfg_runtime.get("conf_threshold", 0.25)),
                "imgsz": _serialize_param(comp_cfg_runtime.get("imgsz", 640)),
                "device": _serialize_param(comp_cfg_runtime.get("device", 0)),
                "font_scale": _serialize_param(comp_cfg_runtime.get("font_scale", None)),
            },
        )
if __name__ == "__main__":
    sys.exit(go())
