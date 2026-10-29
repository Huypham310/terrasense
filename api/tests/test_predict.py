import pytest

from tests.conftest import make_image, upload


def test_predict_red_tile(client):
    res = upload(client, make_image((255, 0, 0)))
    assert res.status_code == 201
    body = res.json()
    # tiny model maps mean red -> class 0
    assert body["label"] == "AnnualCrop"
    assert body["model_name"] == "tiny_test_model"
    assert len(body["top_k"]) == 3
    assert body["top_k"][0]["label"] == body["label"]


def test_top_k_sorted_and_valid_probabilities(client):
    body = upload(client, make_image((40, 200, 90))).json()
    probs = [s["probability"] for s in body["top_k"]]
    assert probs == sorted(probs, reverse=True)
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert body["confidence"] == pytest.approx(probs[0])
    assert body["label"] == "Forest"  # green dominates


@pytest.mark.parametrize(
    ("mode", "fmt", "content_type"),
    [("RGBA", "PNG", "image/png"), ("L", "PNG", "image/png"), ("RGB", "JPEG", "image/jpeg")],
)
def test_accepts_other_color_modes_and_formats(client, mode, fmt, content_type):
    res = upload(client, make_image((0, 0, 255), mode=mode, fmt=fmt), content_type=content_type)
    assert res.status_code == 201


def test_resizes_larger_images(client):
    # 256x256 blue tile still classifies (resized down to 64x64 first)
    res = upload(client, make_image((0, 0, 255), size=256))
    assert res.status_code == 201
    assert res.json()["label"] == "HerbaceousVegetation"


def test_unsupported_content_type(client):
    res = upload(client, b"hello", name="notes.txt", content_type="text/plain")
    assert res.status_code == 415


def test_file_too_large(client):
    # settings fixture caps uploads at 1 MB
    res = upload(client, b"\x00" * (1024 * 1024 + 10))
    assert res.status_code == 413


def test_corrupt_image(client):
    res = upload(client, b"definitely not a png")
    assert res.status_code == 422
    assert "readable image" in res.json()["detail"]


def test_batch_predict_keeps_order(client):
    files = [
        ("files", ("r.png", make_image((255, 0, 0)), "image/png")),
        ("files", ("g.png", make_image((0, 255, 0)), "image/png")),
        ("files", ("b.png", make_image((0, 0, 255)), "image/png")),
    ]
    res = client.post("/predict/batch", files=files)
    assert res.status_code == 201
    labels = [p["label"] for p in res.json()]
    assert labels == ["AnnualCrop", "Forest", "HerbaceousVegetation"]
    assert [p["filename"] for p in res.json()] == ["r.png", "g.png", "b.png"]


def test_batch_over_limit(client):
    files = [("files", (f"{i}.png", make_image(), "image/png")) for i in range(5)]
    assert client.post("/predict/batch", files=files).status_code == 422


def test_predict_without_model_returns_503(no_model_client):
    assert upload(no_model_client, make_image()).status_code == 503
    assert no_model_client.get("/model").status_code == 503
