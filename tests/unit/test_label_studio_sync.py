from unittest.mock import patch

from vit.labeling.label_studio_sync import (
    bbox_to_percent,
    build_auth_header,
    build_prediction_payload,
    is_jwt,
    match_task_by_filename,
)

FAKE_JWT = "eyJhbGciOiJIUzI1NiJ9.eyJ0b2tlbl90eXBlIjoicmVmcmVzaCJ9.fake-signature"


def test_is_jwt_detects_three_segment_token():
    assert is_jwt(FAKE_JWT) is True
    assert is_jwt("plain-legacy-token-123") is False


@patch("vit.labeling.label_studio_sync.requests.post")
def test_build_auth_header_exchanges_jwt_refresh_token(mock_post):
    mock_post.return_value.json.return_value = {"access": "short-lived-access-token"}
    mock_post.return_value.raise_for_status.return_value = None

    header = build_auth_header("http://localhost:8080", FAKE_JWT)

    mock_post.assert_called_once_with(
        "http://localhost:8080/api/token/refresh/", json={"refresh": FAKE_JWT}
    )
    assert header == {"Authorization": "Bearer short-lived-access-token"}


def test_build_auth_header_uses_legacy_token_directly():
    header = build_auth_header("http://localhost:8080", "plain-legacy-token-123")

    assert header == {"Authorization": "Token plain-legacy-token-123"}


def test_bbox_to_percent_converts_pixel_bbox():
    percent = bbox_to_percent(
        (100.0, 50.0, 200.0, 100.0), img_width=1000, img_height=500
    )

    assert percent == {"x": 10.0, "y": 10.0, "width": 20.0, "height": 20.0}


def test_match_task_by_filename_finds_task_with_url_encoded_path():
    tasks = [
        {"id": 1, "data": {"image": "/data/local-files/?d=data%2Fraw%2Fother.jpg"}},
        {
            "id": 2,
            "data": {"image": "/data/local-files/?d=data%2Fraw%2Fbillboard-sample.png"},
        },
    ]

    task = match_task_by_filename(tasks, "billboard-sample.png")

    assert task is not None
    assert task["id"] == 2


def test_match_task_by_filename_returns_none_when_not_found():
    tasks = [
        {"id": 1, "data": {"image": "/data/local-files/?d=data%2Fraw%2Fother.jpg"}}
    ]

    assert match_task_by_filename(tasks, "missing.jpg") is None


def test_build_prediction_payload_maps_category_and_score():
    annotations = [
        {"bbox": [10, 20, 100, 50], "category_id": 1, "score": 0.87},
    ]

    payload = build_prediction_payload(
        task_id=5,
        annotations=annotations,
        category_names_by_id={1: "billboard"},
        img_width=1000,
        img_height=500,
        model_version="grounding-dino-auto-label",
    )

    assert payload["task"] == 5
    assert payload["model_version"] == "grounding-dino-auto-label"
    result = payload["result"][0]
    assert result["type"] == "rectanglelabels"
    assert result["value"]["rectanglelabels"] == ["billboard"]
    assert result["score"] == 0.87
