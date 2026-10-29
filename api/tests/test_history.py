from datetime import datetime

from tests.conftest import make_image, upload


def _seed(client):
    """3 red (AnnualCrop) + 2 green (Forest) predictions."""
    for color in [(255, 0, 0)] * 3 + [(0, 255, 0)] * 2:
        upload(client, make_image(color))


def test_list_newest_first_with_paging(client):
    _seed(client)
    page = client.get("/predictions", params={"limit": 2, "offset": 0}).json()
    assert page["total"] == 5
    assert len(page["items"]) == 2
    ids = [p["id"] for p in page["items"]]
    assert ids == sorted(ids, reverse=True)

    last = client.get("/predictions", params={"limit": 2, "offset": 4}).json()
    assert len(last["items"]) == 1


def test_filter_by_label(client):
    _seed(client)
    page = client.get("/predictions", params={"label": "Forest"}).json()
    assert page["total"] == 2
    assert {p["label"] for p in page["items"]} == {"Forest"}


def test_limit_validation(client):
    assert client.get("/predictions", params={"limit": 0}).status_code == 422
    assert client.get("/predictions", params={"limit": 101}).status_code == 422


def test_get_and_delete(client):
    created = upload(client, make_image()).json()
    assert client.get(f"/predictions/{created['id']}").json()["label"] == "AnnualCrop"

    assert client.delete(f"/predictions/{created['id']}").status_code == 204
    assert client.get(f"/predictions/{created['id']}").status_code == 404
    assert client.delete(f"/predictions/{created['id']}").status_code == 404


def test_created_at_is_timezone_aware(client):
    created = upload(client, make_image()).json()
    # regression: sqlite dropped tzinfo and the UI showed UTC as local time
    fetched = client.get(f"/predictions/{created['id']}").json()
    assert datetime.fromisoformat(fetched["created_at"]).tzinfo is not None
