from pathlib import Path
import argparse
import json
import os
from contextlib import contextmanager
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO, settings

from shared.utils import ensure_dir, print_gpu_info, load_json


def load_class_names(mapping_json):
    """Load semantic id to class-name mapping from JSON and convert keys to int."""
    mapping_path = Path(mapping_json)
    if mapping_path.is_dir():
        raise IsADirectoryError(
            f"mapping_json points to a directory: {mapping_path}. "
            "Provide the JSON file path (for example: .../class_id_to_name.json)."
        )
    mapping = load_json(mapping_path)
    return {int(k): v for k, v in mapping.items()}


def build_data_yaml(dataset_root, names_dict, yaml_path):
    """Create YOLO data.yaml with dataset paths and class-name mapping."""
    dataset_root = Path(dataset_root)
    yaml_path = Path(yaml_path)

    data = {
        "path": str(dataset_root).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": names_dict,
    }

    ensure_dir(yaml_path.parent)
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

    return yaml_path


def load_segmentation_model(model_name="yolo11s-seg.pt"):
    """Load a pretrained YOLO segmentation model checkpoint."""
    return YOLO(model_name)


def resolve_model_source(model_name, weights_root):
    """Resolve bare model names to weights_root so downloaded .pt lands there."""
    model_path = Path(model_name)
    if model_path.is_absolute() or model_path.parent != Path('.'):
        return str(model_path)
    return str(Path(weights_root) / model_path.name)


@contextmanager
def _pushd(path):
    """Temporarily switch CWD so any internal relative downloads land under path."""
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def train_segmentation_model(
    model,
    data_yaml,
    project_dir,
    run_name="seg_train",
    epochs=100,
    imgsz=640,
    batch=8,
    device=0,
    patience=20,
    workers=8,
):
    """Train the segmentation model on the selected NVIDIA GPU."""
    return model.train(
        data=str(data_yaml),
        task="segment",
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        patience=patience,
        pretrained=True,
        amp=True,
        cache=False,
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
        plots=True,
    )


def find_best_weights(project_dir, run_name):
    """Return the path of the best saved checkpoint from training output."""
    best_path = Path(project_dir) / run_name / "weights" / "best.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"best.pt not found: {best_path}")
    return best_path


def validate_on_test(model_path, data_yaml, imgsz=640, batch=8, device=0, workers=8):
    """Evaluate the trained model on the test split and return metrics."""
    model = YOLO(str(model_path))
    return model.val(
        data=str(data_yaml),
        split="test",
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        plots=True,
    )


def collect_metric_summary(metrics):
    """Collect key box and mask metrics with short readable descriptions."""
    return {
        "box_map50_95": {
            "value": float(metrics.box.map),
            "desc": "Overall box mAP across IoU 0.50 to 0.95."
        },
        "box_map50": {
            "value": float(metrics.box.map50),
            "desc": "Loose box accuracy at IoU 0.50."
        },
        "box_map75": {
            "value": float(metrics.box.map75),
            "desc": "Strict box accuracy at IoU 0.75."
        },
        "seg_map50_95": {
            "value": float(metrics.seg.map),
            "desc": "Overall mask mAP across IoU 0.50 to 0.95."
        },
        "seg_map50": {
            "value": float(metrics.seg.map50),
            "desc": "Loose mask accuracy at IoU 0.50."
        },
        "seg_map75": {
            "value": float(metrics.seg.map75),
            "desc": "Strict mask accuracy at IoU 0.75."
        },
    }


def save_metric_summary(summary_dict, output_dir, filename="test_metrics.json"):
    """Save evaluation metrics dictionary to a JSON file."""
    output_dir = Path(output_dir)
    ensure_dir(output_dir)
    out_path = output_dir / filename

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)

    return out_path


def load_training_results_csv(project_dir, run_name):
    """Load YOLO results.csv file produced during training."""
    csv_path = Path(project_dir) / run_name / "results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"results.csv not found: {csv_path}")
    return pd.read_csv(csv_path), csv_path


def plot_training_curves(df, output_dir, filename="training_curves.png"):
    """Plot loss, mAP, precision, and recall curves from training history."""
    output_dir = Path(output_dir)
    ensure_dir(output_dir)
    out_path = output_dir / filename

    if "epoch" not in df.columns:
        df["epoch"] = range(len(df))

    x = df["epoch"]

    train_box_loss_col = next((c for c in df.columns if "train/box_loss" in c), None)
    train_seg_loss_col = next((c for c in df.columns if "train/seg_loss" in c), None)
    val_box_loss_col = next((c for c in df.columns if "val/box_loss" in c), None)
    val_seg_loss_col = next((c for c in df.columns if "val/seg_loss" in c), None)
    seg_map50_col = next((c for c in df.columns if "metrics/seg(mAP50)" in c or "metrics/seg_map50" in c), None)
    seg_map_col = next((c for c in df.columns if "metrics/seg(mAP50-95)" in c or "metrics/seg_map" in c), None)
    box_map50_col = next((c for c in df.columns if "metrics/box(mAP50)" in c or "metrics/box_map50" in c), None)
    precision_col = next((c for c in df.columns if "metrics/precision" in c), None)
    recall_col = next((c for c in df.columns if "metrics/recall" in c), None)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    if train_box_loss_col:
        axes[0, 0].plot(x, df[train_box_loss_col], label="train box loss")
    if val_box_loss_col:
        axes[0, 0].plot(x, df[val_box_loss_col], label="val box loss")
    if train_seg_loss_col:
        axes[0, 0].plot(x, df[train_seg_loss_col], label="train seg loss")
    if val_seg_loss_col:
        axes[0, 0].plot(x, df[val_seg_loss_col], label="val seg loss")
    axes[0, 0].set_title("Loss curves")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    if seg_map50_col:
        axes[0, 1].plot(x, df[seg_map50_col], label="seg mAP50")
    if seg_map_col:
        axes[0, 1].plot(x, df[seg_map_col], label="seg mAP50-95")
    if box_map50_col:
        axes[0, 1].plot(x, df[box_map50_col], label="box mAP50")
    axes[0, 1].set_title("mAP curves")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    if precision_col:
        axes[1, 0].plot(x, df[precision_col], label="precision")
    if recall_col:
        axes[1, 0].plot(x, df[recall_col], label="recall")
    axes[1, 0].set_title("Precision / Recall")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    if seg_map50_col and precision_col and recall_col:
        axes[1, 1].plot(x, df[seg_map50_col], label="seg mAP50")
        axes[1, 1].plot(x, df[precision_col], label="precision")
        axes[1, 1].plot(x, df[recall_col], label="recall")
        axes[1, 1].set_title("Segmentation quality")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return out_path


def run_full_training_pipeline(
    dataset_root,
    output_root,
    mapping_json,
    model_name="yolo11s-seg.pt",
    run_name="robotathome_seg_gpu",
    epochs=100,
    imgsz=640,
    batch=8,
    device=0,
    workers=8,
):
    """Run the full training pipeline using prepared labels and JSON class mapping."""
    dataset_root = Path(dataset_root).resolve()
    output_root = Path(output_root).resolve()
    mapping_json = Path(mapping_json).resolve()
    ensure_dir(output_root)
    run_root = output_root / "runs"
    model_root = dataset_root
    ensure_dir(model_root)

    with _pushd(dataset_root):
        # Keep all Ultralytics artifacts under controlled folders.
        settings.update({
            "runs_dir": str(run_root),
            "weights_dir": str(model_root),
        })

        names_dict = load_class_names(mapping_json)

        yaml_path = build_data_yaml(
            dataset_root=dataset_root,
            names_dict=names_dict,
            yaml_path=output_root / "data.yaml",
        )

        model_source = resolve_model_source(model_name, model_root)
        model = load_segmentation_model(model_name=model_source)

        print_gpu_info(device)

        train_segmentation_model(
            model=model,
            data_yaml=yaml_path,
            project_dir=run_root,
            run_name=run_name,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            workers=workers,
        )

        best_model_path = find_best_weights(run_root, run_name)

        test_metrics = validate_on_test(
            model_path=best_model_path,
            data_yaml=yaml_path,
            imgsz=imgsz,
            batch=32,
            device=device,
            workers=workers,
        )

    summary = collect_metric_summary(test_metrics)
    save_metric_summary(summary, output_root)

    df, _ = load_training_results_csv(run_root, run_name)
    plot_training_curves(df, output_root)


    print("Best model:", best_model_path)
    print("Class mapping:", names_dict)
    print("Test metrics:", summary)

    return {
        "data_yaml": yaml_path,
        "best_model": best_model_path,
        "class_names": names_dict,
        "test_metrics": summary,
        "results_df": df,
    }



def go(args) -> int:

    run_full_training_pipeline(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        mapping_json=args.mapping_json,
        model_name=args.model_name,
        run_name=args.run_name,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate YOLO segmentation model")
    parser.add_argument("--dataset_root", type=str, required=True, help="Split dataset root used for training")
    parser.add_argument("--output_root", type=str, default="output", help="Directory for run artifacts and metrics")
    parser.add_argument("--mapping_json", type=str, required=True,
                        help="class_id_to_name.json used to populate data.yaml names")
    parser.add_argument("--model_name", type=str, default="yolo11s-seg.pt", help="Base segmentation checkpoint")
    parser.add_argument("--run_name", type=str, default="robotathome_seg_strat", help="Run name under output/runs")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--batch", type=int, default=-1, help="Batch size (-1 enables AutoBatch)")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument("--workers", type=int, default=10, help="Data loader workers")

    args = parser.parse_args()

    go(args)