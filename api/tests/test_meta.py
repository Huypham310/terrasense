import numpy as np
import pytest

from app.inference import Classifier, InvalidImageError
from tests.conftest import CLASS_NAMES, make_image, upload


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True, "database": True}


def test_health_degraded_without_model(no_model_client):
    body = no_model_client.get("/health").json()
    assert body["status"] == "degraded"
    assert body["model_loaded"] is False


def test_model_info(client):
    info = client.get("/model").json()
    assert info["model_name"] == "tiny_test_model"
    assert info["class_names"] == CLASS_NAMES
    assert info["image_size"] == 64


def test_stats_empty(client):
    assert client.get("/stats").json() == {
        "total_predictions": 0,
        "by_label": [],
        "avg_latency_ms": None,
    }


def test_stats_counts(client):
    for color in [(255, 0, 0), (255, 0, 0), (0, 255, 0)]:
        upload(client, make_image(color))
    stats = client.get("/stats").json()
    assert stats["total_predictions"] == 3
    # sorted by count, most frequent first
    assert [(r["label"], r["count"]) for r in stats["by_label"]] == [
        ("AnnualCrop", 2),
        ("Forest", 1),
    ]
    assert stats["avg_latency_ms"] >= 0


def test_classifier_probabilities_sum_to_one(model_files):
    classifier = Classifier(*model_files)
    results, latency = classifier.predict([make_image((10, 20, 30))], k=len(CLASS_NAMES))
    total = sum(s.probability for s in results[0].top_k)
    assert total == pytest.approx(1.0, abs=1e-5)
    assert latency >= 0


def test_classifier_load_image_shape_and_range(model_files):
    classifier = Classifier(*model_files)
    array = classifier.load_image(make_image((255, 255, 255), size=100))
    assert array.shape == (3, 64, 64)
    assert array.dtype == np.float32
    assert array.max() == pytest.approx(1.0)


def test_classifier_rejects_garbage(model_files):
    with pytest.raises(InvalidImageError):
        Classifier(*model_files).load_image(b"\x89PNG broken")
