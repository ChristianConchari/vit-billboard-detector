"""Render qualitative figures over a labeled split: comparisons and attention maps.

The bottom `overlay_fraction` of every frame is cropped because the source
camera burns a timestamp/GPS overlay into it.
"""

from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from vit.eval.attention_maps import explain_detections
from vit.eval.visualization import (
    crop_bottom,
    draw_detections,
    draw_encoder_attention,
    draw_sampling_points,
    side_by_side,
)
from vit.inference.detection import BoxProposal, Detector
from vit.utils.device import select_device


def render_detector_comparison(
    detectors: dict[str, Detector],
    ground_truth: dict[str, Any],
    image_dir: Path,
    output_dir: Path,
    overlay_fraction: float,
) -> int:
    """Save a side-by-side figure per image (one panel per detector); return count."""
    boxes_by_image = defaultdict(list)
    for annotation in ground_truth["annotations"]:
        boxes_by_image[annotation["image_id"]].append(annotation["bbox"])

    output_dir.mkdir(parents=True, exist_ok=True)
    for image_info in ground_truth["images"]:
        image = Image.open(image_dir / image_info["file_name"]).convert("RGB")
        overlay_height = round(image.height * overlay_fraction)
        panels = [
            crop_bottom(
                draw_detections(
                    image,
                    boxes_by_image[image_info["id"]],
                    detector.predict(image),
                    title,
                ),
                overlay_height,
            )
            for title, detector in detectors.items()
        ]
        side_by_side(panels).save(output_dir / image_info["file_name"], quality=90)
    return len(ground_truth["images"])


def render_attention_maps(
    checkpoint: Path,
    ground_truth: dict[str, Any],
    image_dir: Path,
    output_dir: Path,
    score_threshold: float,
    max_detections: int,
    overlay_fraction: float,
) -> int:
    """Save detection, encoder attention and decoder sampling panels per detection.

    Returns the number of figures written.
    """
    device = select_device()
    model = AutoModelForObjectDetection.from_pretrained(
        checkpoint, attn_implementation="eager"
    )
    model.to(device)
    image_processor = AutoImageProcessor.from_pretrained(checkpoint)

    output_dir.mkdir(parents=True, exist_ok=True)
    rendered = 0
    for image_info in ground_truth["images"]:
        image = Image.open(image_dir / image_info["file_name"]).convert("RGB")
        overlay_height = round(image.height * overlay_fraction)
        detections = explain_detections(
            model, image_processor, image, device, score_threshold, max_detections
        )
        for rank, detection in enumerate(detections, start=1):
            proposal = BoxProposal("billboard", detection.score, detection.box_xyxy)
            panels = [
                draw_detections(image, [], [proposal], "Detection"),
                draw_encoder_attention(
                    image, detection, "Encoder self-attention (box center token)"
                ),
                draw_sampling_points(
                    image, detection, "Decoder deformable sampling points"
                ),
            ]
            stem = Path(image_info["file_name"]).stem
            side_by_side(
                [crop_bottom(p, overlay_height) for p in panels], max_panel_width=800
            ).save(output_dir / f"{stem}_det{rank}.jpg", quality=90)
            rendered += 1
    return rendered
