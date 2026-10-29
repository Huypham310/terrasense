import io
import json

import numpy as np
import onnx
import pytest
from fastapi.testclient import TestClient
from onnx import TensorProto, helper
from PIL import Image

from app.config import Settings
from app.main import create_app

CLASS_NAMES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]


def _tiny_onnx_model() -> onnx.ModelProto:
    """GlobalAveragePool -> Flatten -> Gemm. Mean red/green/blue drive classes 0/1/2.

    Deterministic and ~1 KB, so API tests don't need the real 45 MB ResNet.
    """
    weights = np.zeros((3, len(CLASS_NAMES)), dtype=np.float32)
    weights[0, 0] = weights[1, 1] = weights[2, 2] = 10.0
    nodes = [
        helper.make_node("GlobalAveragePool", ["input"], ["pooled"]),
        helper.make_node("Flatten", ["pooled"], ["flat"]),
        helper.make_node("Gemm", ["flat", "W", "b"], ["logits"]),
    ]
    graph = helper.make_graph(
        nodes,
        "tiny",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 3, 64, 64])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", len(CLASS_NAMES)])],
        initializer=[
            helper.make_tensor("W", TensorProto.FLOAT, weights.shape, weights.flatten()),
            helper.make_tensor(
                "b", TensorProto.FLOAT, [len(CLASS_NAMES)], [0.0] * len(CLASS_NAMES)
            ),
        ],
    )
    # opset 17 is supported by every recent onnxruntime
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


@pytest.fixture(scope="session")
def model_files(tmp_path_factory):
    folder = tmp_path_factory.mktemp("model")
    onnx.save(_tiny_onnx_model(), folder / "model.onnx")
    meta = {
        "model_name": "tiny_test_model",
        "class_names": CLASS_NAMES,
        "image_size": 64,
        # identity normalization keeps the expected class easy to reason about
        "mean": [0.0, 0.0, 0.0],
        "std": [1.0, 1.0, 1.0],
        "test_accuracy": 0.5,
    }
    (folder / "model_meta.json").write_text(json.dumps(meta))
    return folder / "model.onnx", folder / "model_meta.json"


@pytest.fixture
def settings(tmp_path, model_files) -> Settings:
    model_path, meta_path = model_files
    return Settings(
        database_url=f"sqlite:///{(tmp_path / 'test.db').as_posix()}",
        model_path=model_path,
        meta_path=meta_path,
        max_upload_mb=1,
        max_batch_size=4,
    )


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as c:
        yield c


@pytest.fixture
def no_model_client(tmp_path):
    settings = Settings(
        database_url=f"sqlite:///{(tmp_path / 'empty.db').as_posix()}",
        model_path=tmp_path / "missing.onnx",
        meta_path=tmp_path / "missing.json",
    )
    with TestClient(create_app(settings)) as c:
        yield c


def make_image(color=(255, 0, 0), size=64, mode="RGB", fmt="PNG") -> bytes:
    """Encode a solid-color image in memory."""
    image = Image.new("RGB", (size, size), color).convert(mode)
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return buf.getvalue()


def upload(client, data: bytes, name="tile.png", content_type="image/png"):
    return client.post("/predict", files={"file": (name, data, content_type)})
