"""Push COCO-format auto-labels as pre-annotations onto existing Label Studio tasks.

Label Studio's built-in COCO import creates a *new* set of tasks from the
JSON file instead of attaching annotations to the tasks already created by
a Local Storage sync, which results in duplicate/unlinked images. This
module instead matches auto-labels to already-synced tasks by file name and
pushes the boxes as "predictions" via the API, so a human only has to
review/correct them in the labeling UI.
"""
from typing import Any
from urllib.parse import unquote

import requests


def bbox_to_percent(
    bbox_xywh: tuple[float, float, float, float], img_width: int, img_height: int
) -> dict[str, float]:
    """Convert a pixel-space COCO bbox (x, y, w, h) to Label Studio's percent format."""
    x, y, w, h = bbox_xywh
    return {
        "x": x / img_width * 100,
        "y": y / img_height * 100,
        "width": w / img_width * 100,
        "height": h / img_height * 100,
    }


def match_task_by_filename(
    tasks: list[dict[str, Any]], file_name: str, image_field: str = "image"
) -> dict[str, Any] | None:
    """Find the Label Studio task whose image URL references this file name."""
    for task in tasks:
        image_url = unquote(task.get("data", {}).get(image_field, ""))
        if file_name in image_url:
            return task
    return None


def build_prediction_payload(
    task_id: int,
    annotations: list[dict[str, Any]],
    category_names_by_id: dict[int, str],
    img_width: int,
    img_height: int,
    model_version: str,
) -> dict[str, Any]:
    """Build the /api/predictions/ request body for one image's annotations."""
    results = []
    for ann in annotations:
        percent_box = bbox_to_percent(tuple(ann["bbox"]), img_width, img_height)
        results.append(
            {
                "from_name": "label",
                "to_name": "image",
                "type": "rectanglelabels",
                "value": {
                    **percent_box,
                    "rotation": 0,
                    "rectanglelabels": [category_names_by_id[ann["category_id"]]],
                },
                "score": ann.get("score"),
            }
        )
    return {"task": task_id, "result": results, "model_version": model_version}


def push_predictions(
    label_studio_url: str,
    api_key: str,
    project_id: int,
    coco: dict[str, Any],
    model_version: str = "grounding-dino-auto-label",
) -> dict[str, int]:
    """Push every image's annotations in `coco` as predictions on matching LS tasks.

    Returns a summary dict: {"pushed": N, "unmatched": N, "total_images": N}.
    """
    session = requests.Session()
    session.headers.update({"Authorization": f"Token {api_key}"})

    tasks_resp = session.get(
        f"{label_studio_url}/api/tasks",
        params={"project": project_id, "page_size": 10000},
    )
    tasks_resp.raise_for_status()
    tasks = tasks_resp.json()["tasks"]

    images_by_id = {img["id"]: img for img in coco["images"]}
    category_names_by_id = {c["id"]: c["name"] for c in coco["categories"]}
    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    for ann in coco["annotations"]:
        annotations_by_image.setdefault(ann["image_id"], []).append(ann)

    pushed, unmatched = 0, 0
    for image_id, image_meta in images_by_id.items():
        annotations = annotations_by_image.get(image_id, [])
        if not annotations:
            continue

        task = match_task_by_filename(tasks, image_meta["file_name"])
        if task is None:
            unmatched += 1
            continue

        payload = build_prediction_payload(
            task_id=task["id"],
            annotations=annotations,
            category_names_by_id=category_names_by_id,
            img_width=image_meta["width"],
            img_height=image_meta["height"],
            model_version=model_version,
        )
        response = session.post(f"{label_studio_url}/api/predictions/", json=payload)
        response.raise_for_status()
        pushed += 1

    return {"pushed": pushed, "unmatched": unmatched, "total_images": len(images_by_id)}
