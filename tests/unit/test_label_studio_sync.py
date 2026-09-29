from vit.labeling.label_studio_sync import (
    bbox_to_percent,
    build_prediction_payload,
    match_task_by_filename,
)


def test_bbox_to_percent_converts_pixel_bbox():
    percent = bbox_to_percent((100.0, 50.0, 200.0, 100.0), img_width=1000, img_height=500)

    assert percent == {"x": 10.0, "y": 10.0, "width": 20.0, "height": 20.0}


def test_match_task_by_filename_finds_task_with_url_encoded_path():
    tasks = [
        {"id": 1, "data": {"image": "/data/local-files/?d=data%2Fraw%2Fother.jpg"}},
        {"id": 2, "data": {"image": "/data/local-files/?d=data%2Fraw%2Fbillboard-sample.png"}},
    ]

    task = match_task_by_filename(tasks, "billboard-sample.png")

    assert task is not None
    assert task["id"] == 2


def test_match_task_by_filename_returns_none_when_not_found():
    tasks = [{"id": 1, "data": {"image": "/data/local-files/?d=data%2Fraw%2Fother.jpg"}}]

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
