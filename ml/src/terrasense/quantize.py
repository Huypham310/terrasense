"""INT8 quantization of the exported ONNX model, measured against the fp32 original.

Compares three variants on the held-out test split:
  - fp32: artifacts/model.onnx as exported
  - dynamic int8: weights stored as int8, activations quantized on the fly (what
    `quantize_dynamic` does; it mostly targets MatMul/Gemm, so a CNN keeps most of its compute
    in fp32)
  - static int8: QDQ format, per-channel weight scales, activation ranges calibrated on
    validation images, so the convolutions themselves run in int8

Writes artifacts/model_int8.onnx (static) and reports/quantization.json.

Usage: python -m terrasense.quantize --calib-images 512
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

from terrasense.config import ARTIFACTS_DIR, TrainConfig
from terrasense.data.loaders import build_dataloaders
from terrasense.evaluate import REPORTS_DIR


class LoaderReader(CalibrationDataReader):
    """Feeds up to `limit` images from a DataLoader to the calibrator, one batch at a time."""

    def __init__(self, loader, input_name: str, limit: int):
        self.batches = []
        seen = 0
        for images, _ in loader:
            self.batches.append(images.numpy())
            seen += len(images)
            if seen >= limit:
                break
        self.it = iter(self.batches)
        self.input_name = input_name

    def get_next(self):
        batch = next(self.it, None)
        return None if batch is None else {self.input_name: batch}

    def rewind(self) -> None:
        self.it = iter(self.batches)


def session(path: Path) -> ort.InferenceSession:
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def predict(path: Path, loader) -> tuple[np.ndarray, np.ndarray]:
    sess = session(path)
    name = sess.get_inputs()[0].name
    preds, labels = [], []
    for images, targets in loader:
        logits = sess.run(None, {name: images.numpy()})[0]
        preds.append(logits.argmax(axis=1))
        labels.append(targets.numpy())
    return np.concatenate(preds), np.concatenate(labels)


def latency_ms(path: Path, batch: int, image_size: int, runs: int) -> float:
    """Median wall time per call (not per image), after warm-up."""
    sess = session(path)
    name = sess.get_inputs()[0].name
    x = np.random.default_rng(0).standard_normal((batch, 3, image_size, image_size), np.float32)
    for _ in range(10):
        sess.run(None, {name: x})
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        sess.run(None, {name: x})
        times.append(time.perf_counter() - t)
    # median: a laptop's background load makes the mean jumpy
    return float(np.median(times) * 1000)


def build_variants(fp32: Path, calib_loader, calib_images: int) -> dict[str, Path]:
    prepped = ARTIFACTS_DIR / "model_prep.onnx"
    # shape inference + graph cleanup, recommended before quantizing
    quant_pre_process(str(fp32), str(prepped))

    dynamic = ARTIFACTS_DIR / "model_int8_dynamic.onnx"
    quantize_dynamic(str(prepped), str(dynamic), weight_type=QuantType.QInt8)

    static = ARTIFACTS_DIR / "model_int8.onnx"
    input_name = session(prepped).get_inputs()[0].name
    quantize_static(
        str(prepped),
        str(static),
        LoaderReader(calib_loader, input_name, calib_images),
        quant_format=QuantFormat.QDQ,
        per_channel=True,  # một scale cho mỗi output channel, giữ accuracy tốt hơn per-tensor
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
    )
    prepped.unlink()
    return {"fp32": fp32, "int8_dynamic": dynamic, "int8_static": static}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calib-images", type=int, default=512)
    parser.add_argument("--runs", type=int, default=300)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args(argv)

    meta = json.loads((ARTIFACTS_DIR / "model_meta.json").read_text())
    size = meta["image_size"]
    data = build_dataloaders(TrainConfig(model=meta["model_name"], num_workers=args.num_workers))
    # calibrate on validation images: the test split stays untouched for the comparison
    variants = build_variants(ARTIFACTS_DIR / "model.onnx", data.val, args.calib_images)

    base_preds, labels = predict(variants["fp32"], data.test)
    rows = {}
    for name, path in variants.items():
        preds = base_preds if name == "fp32" else predict(path, data.test)[0]
        rows[name] = {
            "size_mb": round(path.stat().st_size / 1e6, 2),
            "test_accuracy": round(float((preds == labels).mean()), 4),
            "agreement_with_fp32": round(float((preds == base_preds).mean()), 4),
            "latency_ms_batch1": round(latency_ms(path, 1, size, args.runs), 3),
            "latency_ms_batch32": round(latency_ms(path, 32, size, max(args.runs // 5, 20)), 2),
        }

    report = {
        "model": meta["model_name"],
        "test_images": int(len(labels)),
        "calibration_images": args.calib_images,
        "onnxruntime": ort.__version__,
        "variants": rows,
    }
    out = REPORTS_DIR / "quantization.json"
    out.write_text(json.dumps(report, indent=2))

    header = ("variant", "size MB", "test acc", "agree", "b1 ms", "b32 ms")
    print("{:14s} {:>8s} {:>9s} {:>7s} {:>7s} {:>7s}".format(*header))
    for name, r in rows.items():
        print(
            f"{name:14s} {r['size_mb']:8.1f} {r['test_accuracy']:9.4f} "
            f"{r['agreement_with_fp32']:7.4f} {r['latency_ms_batch1']:7.2f} "
            f"{r['latency_ms_batch32']:7.1f}"
        )
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
