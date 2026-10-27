import io
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, UnidentifiedImageError


class InvalidImageError(ValueError):
    """Raised when the uploaded bytes can't be decoded as an image."""


@dataclass
class ClassScore:
    label: str
    probability: float


@dataclass
class Classification:
    label: str
    confidence: float
    top_k: list[ClassScore]


class Classifier:
    """Wraps the exported ONNX model + its metadata (classes, normalization, input size).

    Preprocessing has to match training exactly: RGB, resize to the training size,
    scale to [0, 1], then normalize with the train-split mean/std.
    """

    def __init__(self, model_path: Path, meta_path: Path):
        meta = json.loads(meta_path.read_text())
        self.model_name: str = meta["model_name"]
        self.class_names: list[str] = meta["class_names"]
        self.image_size: int = meta["image_size"]
        self.test_accuracy: float | None = meta.get("test_accuracy")
        # reshape sẵn thành (1, 3, 1, 1) để broadcast với batch NCHW
        self.mean = np.array(meta["mean"], dtype=np.float32).reshape(1, 3, 1, 1)
        self.std = np.array(meta["std"], dtype=np.float32).reshape(1, 3, 1, 1)
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def load_image(self, data: bytes) -> np.ndarray:
        """Decode bytes into a (3, H, W) float32 array in [0, 1]."""
        try:
            image = Image.open(io.BytesIO(data))
            image.load()
        except (UnidentifiedImageError, OSError) as exc:
            raise InvalidImageError("File is not a readable image") from exc
        # ảnh png có thể là RGBA hoặc grayscale, ép về RGB cho chắc
        image = image.convert("RGB")
        if image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        return array.transpose(2, 0, 1)  # HWC -> CHW

    def predict(self, images: list[bytes], k: int = 3) -> tuple[list[Classification], float]:
        """Classify a batch of encoded images. Returns results and total inference time in ms."""
        batch = np.stack([self.load_image(data) for data in images])
        batch = (batch - self.mean) / self.std

        start = time.perf_counter()
        logits = self.session.run(None, {self.input_name: batch.astype(np.float32)})[0]
        elapsed_ms = (time.perf_counter() - start) * 1000

        # softmax ổn định số học: trừ max trước khi exp
        shifted = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)

        results = []
        for row in probs:
            order = np.argsort(row)[::-1][:k]
            top = [ClassScore(self.class_names[i], float(row[i])) for i in order]
            results.append(Classification(label=top[0].label, confidence=top[0].probability, top_k=top))
        return results, elapsed_ms
