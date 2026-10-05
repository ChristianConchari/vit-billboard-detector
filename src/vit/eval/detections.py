"""Shared helpers to turn a Detector's proposals into COCO results and evaluate them."""

import contextlib
import io
from pathlib import Path
from typing import Any

from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from vit.inference.detection import Detector


def collect_detections(
    detector: Detector, ground_truth: dict[str, Any], image_dir: str | Path
) -> list[dict[str, Any]]:
    """Predict on every ground-truth image and return COCO-format results.

    Class-agnostic: the dataset must have a single category, and every
    proposal counts for it regardless of the detector's own label text
    (Grounding DINO labels are prompt phrases, not dataset categories).
    """
    categories = ground_truth["categories"]
    if len(categories) != 1:
        raise ValueError(
            f"Expected a single-category dataset, got {len(categories)} categories"
        )
    category_id = categories[0]["id"]

    detections = []
    for image_info in ground_truth["images"]:
        image = Image.open(Path(image_dir) / image_info["file_name"]).convert("RGB")
        for proposal in detector.predict(image):
            x_min, y_min, x_max, y_max = proposal.box_xyxy
            detections.append(
                {
                    "image_id": image_info["id"],
                    "category_id": category_id,
                    "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                    "score": proposal.score,
                }
            )
    return detections


def run_cocoeval(
    ground_truth: dict[str, Any], detections: list[dict[str, Any]]
) -> COCOeval:
    """Evaluate and accumulate COCO bbox metrics in memory, silencing pycocotools."""
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = ground_truth
        coco_gt.createIndex()
        coco_eval = COCOeval(coco_gt, coco_gt.loadRes(detections), iouType="bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
    return coco_eval
