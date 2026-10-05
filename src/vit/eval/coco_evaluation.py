"""COCO-style evaluation (mAP, AP50, AP75, AP by object size) for any Detector."""

import contextlib
import io
from pathlib import Path
from typing import Any

from vit.eval.detections import collect_detections, run_cocoeval
from vit.inference.detection import Detector

METRIC_NAMES = ("mAP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large")


def coco_metrics(
    ground_truth: dict[str, Any], detections: list[dict[str, Any]]
) -> dict[str, float]:
    """COCO summary metrics for in-memory ground truth and detections.

    Follows the COCO convention of reporting -1 for a size range with no
    ground-truth objects.
    """
    if not detections:
        return dict.fromkeys(METRIC_NAMES, 0.0)

    coco_eval = run_cocoeval(ground_truth, detections)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_eval.summarize()
    summary_stats = coco_eval.stats[: len(METRIC_NAMES)]
    return {name: float(value) for name, value in zip(METRIC_NAMES, summary_stats, strict=True)}


def evaluate_detector(
    detector: Detector, ground_truth: dict[str, Any], image_dir: str | Path
) -> dict[str, float]:
    """Predict on every ground-truth image and score the proposals (class-agnostic)."""
    return coco_metrics(ground_truth, collect_detections(detector, ground_truth, image_dir))
