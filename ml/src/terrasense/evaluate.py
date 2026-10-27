"""Evaluate a trained checkpoint on the held-out test split.

Writes reports/<model>/metrics.json and confusion_matrix.png, and logs them to the
same MLflow run the checkpoint came from.

Usage: python -m terrasense.evaluate --model resnet18
"""

import argparse
import json
import time

import matplotlib

matplotlib.use("Agg")  # no display needed, chỉ lưu ra file png
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from terrasense.config import ARTIFACTS_DIR, CLASS_NAMES, PROJECT_ROOT, TrainConfig
from terrasense.data.loaders import build_dataloaders
from terrasense.models import MODEL_NAMES, build_model
from terrasense.tracking import setup_mlflow

REPORTS_DIR = PROJECT_ROOT / "reports"


def load_checkpoint(model_name: str) -> tuple[torch.nn.Module, dict]:
    """Rebuild the model from artifacts/<model>/best.pt (no pretrained download needed)."""
    ckpt = torch.load(ARTIFACTS_DIR / model_name / "best.pt", map_location="cpu")
    model = build_model(ckpt["model_name"], num_classes=len(ckpt["class_names"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def predict_all(model: torch.nn.Module, loader) -> tuple[np.ndarray, np.ndarray]:
    preds, labels = [], []
    for images, targets in loader:
        preds.append(model(images).argmax(dim=1).numpy())
        labels.append(targets.numpy())
    return np.concatenate(preds), np.concatenate(labels)


@torch.no_grad()
def measure_latency(model: torch.nn.Module, image_size: int, runs: int = 200) -> float:
    """Average single-image CPU latency in milliseconds (batch size 1, after warm-up)."""
    x = torch.randn(1, 3, image_size, image_size)
    for _ in range(20):  # warm-up, lần đầu chạy luôn chậm hơn
        model(x)
    start = time.perf_counter()
    for _ in range(runs):
        model(x)
    return (time.perf_counter() - start) / runs * 1000


def plot_confusion(cm: np.ndarray, path) -> None:
    # normalize theo hàng để thấy recall của từng class
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            if cm[i, j]:
                color = "white" if cm_norm[i, j] > 0.5 else "black"
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7, color=color)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate(model_name: str, num_workers: int) -> dict:
    model, ckpt = load_checkpoint(model_name)
    cfg = TrainConfig(model=model_name, num_workers=num_workers)
    data = build_dataloaders(cfg)
    preds, labels = predict_all(model, data.test)

    report = classification_report(
        labels, preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    metrics = {
        "model": model_name,
        "test_accuracy": accuracy_score(labels, preds),
        "test_macro_f1": f1_score(labels, preds, average="macro"),
        "best_val_accuracy": ckpt["val_acc"],
        "best_epoch": ckpt["epoch"],
        "num_params": sum(p.numel() for p in model.parameters()),
        "cpu_latency_ms": measure_latency(model, ckpt["image_size"]),
        "test_images": int(len(labels)),
        "per_class_f1": {name: report[name]["f1-score"] for name in CLASS_NAMES},
    }

    out_dir = REPORTS_DIR / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_confusion(confusion_matrix(labels, preds), out_dir / "confusion_matrix.png")

    # log vào lại đúng run lúc train để xem chung trên MLflow UI
    setup_mlflow()
    with mlflow.start_run(run_id=ckpt["run_id"]):
        mlflow.log_metrics(
            {
                "test_accuracy": metrics["test_accuracy"],
                "test_macro_f1": metrics["test_macro_f1"],
                "cpu_latency_ms": metrics["cpu_latency_ms"],
            }
        )
        mlflow.log_artifacts(str(out_dir), artifact_path="evaluation")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    m = evaluate(args.model, args.num_workers)
    print(
        f"{m['model']}: test acc {m['test_accuracy']:.4f} | macro F1 {m['test_macro_f1']:.4f} "
        f"| {m['num_params']:,} params | {m['cpu_latency_ms']:.2f} ms/image"
    )
    worst = sorted(m["per_class_f1"].items(), key=lambda kv: kv[1])[:3]
    print("lowest F1:", ", ".join(f"{k}={v:.3f}" for k, v in worst))


if __name__ == "__main__":
    main()
