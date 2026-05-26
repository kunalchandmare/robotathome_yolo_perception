"""
Pipeline orchestrator for robotathome_yolo_perception.
Driven by Hydra config (params.yaml). Run with:

    mlflow run . -P steps=all
    mlflow run . -P steps=download,training
    mlflow run . -P hydra_options="training.epochs=20"
"""
import os
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
    steps_par = config["main"]["steps"]
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    # ── src/ steps ─────────────────────────────────────────────────────────────
    if "download" in active_steps:
        for entry in config["download"]["dataset_sources"]:
            mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "download"),
                "main",
                env_manager="local",
                parameters={
                    "dataset_extract_to": str(entry["dataset_extract_to"]),
                    "dataset_filename": str(entry["dataset_filename"]),
                    "dataset_md5": str(entry["dataset_md5"]),
                    "dataset_url": str(entry["dataset_url"]),
                    "extract_root": str(entry["extract_root"]),
                    "force_download": str(entry["force_download"]),
                    "out_dir": str(entry["out_dir"]),
                },
            )
    if "annotation_convert" in active_steps:
        mlflow.run(
            os.path.join(hydra.utils.get_original_cwd(), "src", "annotation_convert"),
            "main",
            env_manager="local",
            parameters={
                "rh_path": str(config["annotation_convert"]["rh_path"]),
                "rgbd_path": str(config["annotation_convert"]["rgbd_path"]),
                "scene_path": str(config["annotation_convert"]["scene_path"]),
                "output_root": str(config["annotation_convert"]["output_root"]),
                "rgbd_root": str(config["annotation_convert"]["rgbd_root"]),
                "epsilon_ratio": str(config["annotation_convert"]["epsilon_ratio"]),
                "labels_root": str(config["annotation_convert"]["labels_root"]),
                "mapping_json": str(config["annotation_convert"]["mapping_json"]),
                "name_mode": str(config["annotation_convert"]["name_mode"]),
                "backup": str(config["annotation_convert"]["backup"]),
            },
        )
    if "split" in active_steps:
        mlflow.run(
            os.path.join(hydra.utils.get_original_cwd(), "src", "split"),
            "main",
            env_manager="local",
            parameters={
                "source_root": str(config["split"]["source_root"]),
                "output_root": str(config["split"]["output_root"]),
                "train_ratio": str(config["split"]["train_ratio"]),
                "val_ratio": str(config["split"]["val_ratio"]),
                "test_ratio": str(config["split"]["test_ratio"]),
                "seed": str(config["split"]["seed"]),
                "image_exts": str(config["split"]["image_exts"]),
                "use_stratified": str(config["split"]["use_stratified"]),
                "rare_threshold": str(config["split"]["rare_threshold"]),
            },
        )
    if "training" in active_steps:
        mlflow.run(
            os.path.join(hydra.utils.get_original_cwd(), "src", "training"),
            "main",
            env_manager="local",
            parameters={
                "dataset_root": str(config["training"]["dataset_root"]),
                "output_root": str(config["training"]["output_root"]),
                "mapping_json": str(config["training"]["mapping_json"]),
                "model_name": str(config["training"]["model_name"]),
                "run_name": str(config["training"]["run_name"]),
                "epochs": str(config["training"]["epochs"]),
                "imgsz": str(config["training"]["imgsz"]),
                "batch": str(config["training"]["batch"]),
                "device": str(config["training"]["device"]),
                "workers": str(config["training"]["workers"]),
            },
        )
    # ── components/ ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    go()
