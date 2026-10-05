"""Localization diagnostics: where AP drops across IoU thresholds, and systematic box bias.

Used to show that the high-IoU ceiling comes from a labeling convention: models
trained on corrected Grounding DINO pre-labels draw boxes in that model's
(tighter) style, while the test split was drawn by hand.
"""

import statistics
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torchvision.ops import box_iou

from vit.eval.detections import run_cocoeval

IOU_THRESHOLDS = tuple(round(float(t), 2) for t in np.linspace(0.5, 0.95, 10))

BoxXYXY = list[float]
MatchedPair = tuple[BoxXYXY, BoxXYXY, float]


@dataclass
class EdgeBias:
    """Mean signed edge offsets normalized by the ground-truth box size.

    Positive means the predicted edge lies outside the ground-truth box (box too
    large), negative means inside (box too small).
    """

    left: float
    top: float
    right: float
    bottom: float
    width_ratio: float
    height_ratio: float
    median_iou: float
    matches: int


def ap_per_iou_threshold(
    ground_truth: dict[str, Any], detections: list[dict[str, Any]]
) -> dict[float, float]:
    """AP (all areas, up to 100 detections) at each COCO IoU threshold from 0.50 to 0.95."""
    if not detections:
        return dict.fromkeys(IOU_THRESHOLDS, 0.0)
    precision = run_cocoeval(ground_truth, detections).eval["precision"][:, :, 0, 0, -1]
    return {
        threshold: float(np.mean(values[values > -1])) if (values > -1).any() else 0.0
        for threshold, values in zip(IOU_THRESHOLDS, precision, strict=True)
    }


def boxes_by_image(
    detections: list[dict[str, Any]], min_score: float = 0.0
) -> dict[int, list[BoxXYXY]]:
    """Group COCO-format results (or annotations) as xyxy boxes per image id."""
    grouped: dict[int, list[BoxXYXY]] = {}
    for item in detections:
        if item.get("score", 1.0) < min_score:
            continue
        x, y, w, h = item["bbox"]
        grouped.setdefault(item["image_id"], []).append([x, y, x + w, y + h])
    return grouped


def match_to_ground_truth(
    ground_truth_boxes: dict[int, list[BoxXYXY]],
    candidate_boxes: dict[int, list[BoxXYXY]],
    min_iou: float = 0.5,
) -> list[MatchedPair]:
    """Pair each ground-truth box with its highest-IoU candidate, if that IoU reaches `min_iou`."""
    pairs = []
    for image_id, gt_boxes in ground_truth_boxes.items():
        candidates = candidate_boxes.get(image_id)
        if not candidates:
            continue
        ious = box_iou(torch.tensor(gt_boxes), torch.tensor(candidates))
        for gt_box, row in zip(gt_boxes, ious, strict=True):
            iou, index = row.max(dim=0)
            if iou >= min_iou:
                pairs.append((candidates[int(index)], gt_box, float(iou)))
    return pairs


def edge_bias(pairs: list[MatchedPair]) -> EdgeBias:
    if not pairs:
        raise ValueError("No matched boxes to measure")
    left, top, right, bottom, widths, heights = [], [], [], [], [], []
    for pred, gt, _ in pairs:
        gt_width, gt_height = gt[2] - gt[0], gt[3] - gt[1]
        left.append((gt[0] - pred[0]) / gt_width)
        right.append((pred[2] - gt[2]) / gt_width)
        top.append((gt[1] - pred[1]) / gt_height)
        bottom.append((pred[3] - gt[3]) / gt_height)
        widths.append((pred[2] - pred[0]) / gt_width)
        heights.append((pred[3] - pred[1]) / gt_height)
    return EdgeBias(
        left=statistics.fmean(left),
        top=statistics.fmean(top),
        right=statistics.fmean(right),
        bottom=statistics.fmean(bottom),
        width_ratio=statistics.fmean(widths),
        height_ratio=statistics.fmean(heights),
        median_iou=statistics.median(iou for _, _, iou in pairs),
        matches=len(pairs),
    )
