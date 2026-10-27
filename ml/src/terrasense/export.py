"""Export a trained checkpoint to ONNX for the inference API.

Writes artifacts/model.onnx plus artifacts/model_meta.json (class names, normalization
stats, input size) and checks that ONNX Runtime matches PyTorch.

Usage: python -m terrasense.export --model resnet18
"""

import argparse
import json
import sys
import time

import numpy as np
import onnxruntime as ort
import torch

from terrasense.config import ARTIFACTS_DIR
from terrasense.evaluate import REPORTS_DIR, load_checkpoint
from terrasense.models import MODEL_NAMES


def export_onnx(model_name: str) -> dict:
    model, ckpt = load_checkpoint(model_name)
    size = ckpt["image_size"]
    onnx_path = ARTIFACTS_DIR / "model.onnx"
    dummy = torch.randn(1, 3, size, size)

    torch.onnx.export(
        model,
        (dummy,),
        str(onnx_path),
        input_names=["input"],
        output_names=["logits"],
        # batch size động để API có thể gửi nhiều ảnh 1 lần
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        external_data=False,
    )

    # sanity check: ORT output should match PyTorch up to float error
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    batch = torch.randn(4, 3, size, size)
    with torch.no_grad():
        expected = model(batch).numpy()
    actual = session.run(None, {"input": batch.numpy()})[0]
    max_diff = float(np.abs(expected - actual).max())
    if max_diff > 1e-3:
        raise RuntimeError(f"ONNX output differs from PyTorch by {max_diff}")

    x = np.random.randn(1, 3, size, size).astype(np.float32)
    for _ in range(20):
        session.run(None, {"input": x})
    start = time.perf_counter()
    for _ in range(200):
        session.run(None, {"input": x})
    ort_latency = (time.perf_counter() - start) / 200 * 1000

    meta = {
        "model_name": model_name,
        "class_names": ckpt["class_names"],
        "image_size": size,
        "mean": ckpt["mean"],
        "std": ckpt["std"],
        "val_accuracy": ckpt["val_acc"],
        "onnx_cpu_latency_ms": ort_latency,
        "max_abs_diff_vs_torch": max_diff,
    }
    metrics_path = REPORTS_DIR / model_name / "metrics.json"
    if metrics_path.exists():
        meta["test_accuracy"] = json.loads(metrics_path.read_text())["test_accuracy"]
    (ARTIFACTS_DIR / "model_meta.json").write_text(json.dumps(meta, indent=2))
    meta["onnx_size_mb"] = onnx_path.stat().st_size / 1e6
    return meta


def main() -> None:
    # torch.onnx prints emoji in its progress log; Windows consoles default to cp1252 and crash on it
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    args = parser.parse_args()
    meta = export_onnx(args.model)
    print(
        f"exported {meta['model_name']} -> {ARTIFACTS_DIR / 'model.onnx'} "
        f"({meta['onnx_size_mb']:.1f} MB) | ORT {meta['onnx_cpu_latency_ms']:.2f} ms/image "
        f"| max diff {meta['max_abs_diff_vs_torch']:.2e}"
    )


if __name__ == "__main__":
    main()
