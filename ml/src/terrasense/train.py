"""Train a land-use classifier on EuroSAT and track the run in MLflow.

Usage: python -m terrasense.train --model resnet18 --epochs 8
"""

import argparse
import random
import time
from dataclasses import asdict
from pathlib import Path

import mlflow
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from terrasense.config import CLASS_NAMES, TrainConfig
from terrasense.data.loaders import build_dataloaders
from terrasense.models import MODEL_NAMES, build_model
from terrasense.tracking import setup_mlflow


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    limit_batches: int | None = None,
) -> tuple[float, float]:
    """One pass over `loader`. Trains if an optimizer is given, otherwise just evaluates.

    Returns (mean loss, accuracy).
    """
    training = optimizer is not None
    model.train(training)
    total_loss, correct, seen = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for step, (images, labels) in enumerate(loader):
            if limit_batches is not None and step >= limit_batches:
                break
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            # loss.item() là mean theo batch nên nhân lại với batch size
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            seen += labels.size(0)
    return total_loss / max(seen, 1), correct / max(seen, 1)


def train(cfg: TrainConfig, patience: int, limit_batches: int | None = None) -> Path:
    """Full training loop with early stopping on validation accuracy. Returns the checkpoint path."""
    set_seed(cfg.seed)
    torch.set_num_threads(max(torch.get_num_threads(), 1))
    data = build_dataloaders(cfg)
    model = build_model(cfg.model, num_classes=len(CLASS_NAMES))

    # label smoothing helps a bit since a few classes look alike (Pasture vs HerbaceousVegetation)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    out_dir = cfg.output_dir / cfg.model
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "best.pt"

    setup_mlflow()
    with mlflow.start_run(run_name=cfg.model) as run:
        mlflow.log_params({k: str(v) for k, v in asdict(cfg).items()})
        mlflow.log_param("patience", patience)
        mlflow.log_param("num_params", sum(p.numel() for p in model.parameters()))

        # -1 so the first epoch always saves a checkpoint, even if val acc is 0
        best_acc, bad_epochs = -1.0, 0
        for epoch in range(1, cfg.epochs + 1):
            start = time.perf_counter()
            train_loss, train_acc = run_epoch(model, data.train, criterion, optimizer, limit_batches)
            val_loss, val_acc = run_epoch(model, data.val, criterion, limit_batches=limit_batches)
            lr = scheduler.get_last_lr()[0]
            scheduler.step()
            elapsed = time.perf_counter() - start

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "lr": lr,
                    "epoch_seconds": elapsed,
                },
                step=epoch,
            )
            print(
                f"epoch {epoch:>2}/{cfg.epochs} | train {train_loss:.4f} / {train_acc:.4f} "
                f"| val {val_loss:.4f} / {val_acc:.4f} | {elapsed:.0f}s"
            )

            if val_acc > best_acc:
                best_acc, bad_epochs = val_acc, 0
                torch.save(
                    {
                        "model_name": cfg.model,
                        "state_dict": model.state_dict(),
                        "class_names": CLASS_NAMES,
                        "image_size": cfg.image_size,
                        "mean": data.stats.mean,
                        "std": data.stats.std,
                        "val_acc": val_acc,
                        "epoch": epoch,
                        "run_id": run.info.run_id,
                    },
                    ckpt_path,
                )
            else:
                bad_epochs += 1
                # dừng sớm nếu val không cải thiện sau `patience` epoch
                if bad_epochs >= patience:
                    print(f"early stopping at epoch {epoch}")
                    break

        mlflow.log_metric("best_val_acc", best_acc)
        mlflow.log_artifact(str(ckpt_path), artifact_path="checkpoint")
    print(f"best val acc {best_acc:.4f}, saved to {ckpt_path}")
    return ckpt_path


def main() -> None:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_NAMES, default=defaults.model)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--lr", type=float, default=defaults.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument(
        "--limit-batches", type=int, default=None, help="only for quick smoke runs"
    )
    args = parser.parse_args()

    cfg = TrainConfig(
        model=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
    )
    train(cfg, patience=args.patience, limit_batches=args.limit_batches)


if __name__ == "__main__":
    main()
