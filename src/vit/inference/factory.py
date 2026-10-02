"""Builders for the project's detectors from their configs."""

from pathlib import Path
from typing import Any

import torch

from vit.inference.rtdetr_detector import RtDetrDetector
from vit.labeling.grounding_dino_labeler import GroundingDinoAutoLabeler
from vit.models.rtdetr import load_rtdetr


def build_rtdetr_detector(
    checkpoint: str | Path, label_names: list[str], score_threshold: float
) -> RtDetrDetector:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, image_processor = load_rtdetr(checkpoint, label_names)
    return RtDetrDetector(model.to(device), image_processor, device, score_threshold)


def build_grounding_dino_detector(
    config: dict[str, Any], box_threshold: float
) -> GroundingDinoAutoLabeler:
    return GroundingDinoAutoLabeler(
        checkpoint=config["model"]["checkpoint"],
        prompts=config["prompts"],
        box_threshold=box_threshold,
        nms_iou_threshold=config["thresholds"]["nms_iou_threshold"],
    )
