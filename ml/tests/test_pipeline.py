"""End-to-end: train on fake tiles -> checkpoint + MLflow run -> ONNX export.

Uses tmp dirs for everything so it never touches real artifacts or mlflow.db.
"""

import json

import mlflow
import pytest
import torch
from torch.utils.data import DataLoader

from terrasense import evaluate, export, tracking
from terrasense import train as train_module
from terrasense.config import TrainConfig
from terrasense.data.loaders import DataBundle, TransformSubset
from terrasense.data.stats import ChannelStats
from terrasense.data.transforms import build_transforms


@pytest.fixture
def isolated(tmp_path, monkeypatch, fake_tiles):
    """Redirect MLflow + artifacts to tmp and feed the trainer fake data."""
    monkeypatch.setattr(tracking, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(evaluate, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(export, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(export, "REPORTS_DIR", tmp_path / "reports")

    stats = ChannelStats(mean=[0.5] * 3, std=[0.25] * 3)

    def fake_loaders(cfg):
        def loader(indices, train):
            ds = TransformSubset(fake_tiles, indices, build_transforms(stats, train))
            return DataLoader(ds, batch_size=32, shuffle=train)

        return DataBundle(
            train=loader(list(range(0, 160)), True),
            val=loader(list(range(160, 180)), False),
            test=loader(list(range(180, 200)), False),
            stats=stats,
        )

    monkeypatch.setattr(train_module, "build_dataloaders", fake_loaders)
    monkeypatch.setattr(evaluate, "build_dataloaders", fake_loaders)
    monkeypatch.setattr(evaluate, "REPORTS_DIR", tmp_path / "reports")
    yield tmp_path
    mlflow.set_tracking_uri(None)


def test_train_writes_checkpoint_and_mlflow_run(isolated):
    cfg = TrainConfig(
        model="simple_cnn", epochs=2, output_dir=isolated / "artifacts", num_workers=0
    )
    ckpt_path = train_module.train(cfg, patience=5)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    assert ckpt["model_name"] == "simple_cnn"
    assert len(ckpt["class_names"]) == 10
    assert ckpt["mean"] == [0.5] * 3

    run = mlflow.get_run(ckpt["run_id"])
    assert run.data.params["model"] == "simple_cnn"
    assert "best_val_acc" in run.data.metrics


def test_evaluate_writes_report(isolated):
    cfg = TrainConfig(
        model="simple_cnn", epochs=1, output_dir=isolated / "artifacts", num_workers=0
    )
    train_module.train(cfg, patience=5)

    metrics = evaluate.evaluate("simple_cnn", num_workers=0)

    out_dir = isolated / "reports" / "simple_cnn"
    assert (out_dir / "confusion_matrix.png").stat().st_size > 0
    saved = json.loads((out_dir / "metrics.json").read_text())
    assert saved["test_images"] == 20
    assert 0.0 <= saved["test_accuracy"] <= 1.0
    assert set(saved["per_class_f1"]) == set(metrics["per_class_f1"])
    assert metrics["cpu_latency_ms"] > 0


def test_export_matches_torch(isolated):
    cfg = TrainConfig(
        model="simple_cnn", epochs=1, output_dir=isolated / "artifacts", num_workers=0
    )
    train_module.train(cfg, patience=5)

    meta = export.export_onnx("simple_cnn")

    assert (isolated / "artifacts" / "model.onnx").exists()
    # export_onnx raises if ORT drifts from torch by > 1e-3, so this is a sanity check on top
    assert meta["max_abs_diff_vs_torch"] < 1e-3
    saved = json.loads((isolated / "artifacts" / "model_meta.json").read_text())
    assert saved["image_size"] == 64
    assert saved["class_names"][0] == "AnnualCrop"
