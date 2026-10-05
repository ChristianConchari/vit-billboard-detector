"""Pick the score threshold that maximizes F1 on a labeled split."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision.ops import box_iou

from vit.inference.detection import BoxProposal, Detector

CALIBRATION_FILE = "calibration.json"

BoxXYXY = tuple[float, float, float, float]


@dataclass
class ThresholdCalibration:
    score_threshold: float
    precision: float
    recall: float
    f1: float
    iou_threshold: float

    def save(self, checkpoint_dir: Path) -> Path:
        path = checkpoint_dir / CALIBRATION_FILE
        path.write_text(json.dumps(asdict(self), indent=2))
        return path


def load_calibrated_threshold(checkpoint_dir: Path) -> float | None:
    path = checkpoint_dir / CALIBRATION_FILE
    if not path.exists():
        return None
    return json.loads(path.read_text())["score_threshold"]


def match_proposals(
    proposals: list[BoxProposal], ground_truth_xyxy: list[BoxXYXY], iou_threshold: float
) -> list[tuple[float, bool]]:
    """Greedily match proposals (highest score first) to unmatched ground truth.

    Returns (score, is_true_positive) for every proposal.
    """
    ranked = sorted(proposals, key=lambda p: p.score, reverse=True)
    if not ranked or not ground_truth_xyxy:
        return [(p.score, False) for p in ranked]

    ious = box_iou(
        torch.tensor([p.box_xyxy for p in ranked], dtype=torch.float32),
        torch.tensor(ground_truth_xyxy, dtype=torch.float32),
    )
    matched_ground_truth: set[int] = set()
    results = []
    for row, proposal in enumerate(ranked):
        candidates = [
            (iou, column)
            for column, iou in enumerate(ious[row].tolist())
            if iou >= iou_threshold and column not in matched_ground_truth
        ]
        if candidates:
            matched_ground_truth.add(max(candidates)[1])
        results.append((proposal.score, bool(candidates)))
    return results


def best_f1_threshold(
    scored_matches: list[tuple[float, bool]],
    total_ground_truth: int,
    iou_threshold: float,
) -> ThresholdCalibration:
    if total_ground_truth == 0:
        raise ValueError("Cannot calibrate a threshold without ground-truth boxes")

    best = ThresholdCalibration(1.0, 0.0, 0.0, 0.0, iou_threshold)
    true_positives = 0
    for rank, (score, is_true_positive) in enumerate(
        sorted(scored_matches, reverse=True), start=1
    ):
        true_positives += is_true_positive
        precision = true_positives / rank
        recall = true_positives / total_ground_truth
        f1 = 2 * precision * recall / (precision + recall) if true_positives else 0.0
        if f1 > best.f1:
            best = ThresholdCalibration(score, precision, recall, f1, iou_threshold)
    return best


def calibrate_threshold(
    detector: Detector,
    ground_truth: dict[str, Any],
    image_dir: str | Path,
    iou_threshold: float = 0.5,
) -> ThresholdCalibration:
    """Run a permissive `detector` on every image and pick the best-F1 score cutoff."""
    boxes_by_image: dict[int, list[BoxXYXY]] = {
        img["id"]: [] for img in ground_truth["images"]
    }
    for annotation in ground_truth["annotations"]:
        x, y, w, h = annotation["bbox"]
        boxes_by_image[annotation["image_id"]].append((x, y, x + w, y + h))

    scored_matches = []
    for image_info in ground_truth["images"]:
        image = Image.open(Path(image_dir) / image_info["file_name"]).convert("RGB")
        scored_matches += match_proposals(
            detector.predict(image), boxes_by_image[image_info["id"]], iou_threshold
        )

    total_ground_truth = sum(len(boxes) for boxes in boxes_by_image.values())
    return best_f1_threshold(scored_matches, total_ground_truth, iou_threshold)
