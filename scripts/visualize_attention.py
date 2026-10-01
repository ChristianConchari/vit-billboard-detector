"""CLI: render RT-DETR attention maps for the detections on a reviewed split.

For each detection (up to --max-detections per image) writes one image with
three panels: the detection, the encoder self-attention from the box center,
and the decoder's deformable sampling points. Output goes to
reports/figures/attention_<split>/, with the timestamp/GPS overlay cropped.

Usage:
    python scripts/visualize_attention.py --checkpoint checkpoints/rtdetr/<run>/best
"""
import argparse
import json
from pathlib import Path

import torch
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
from vit.inference.detection import BoxProposal
from vit.utils.config import load_config
from vit.utils.logging import get_logger

logger = get_logger(__name__, log_file="evaluation.log")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize RT-DETR attention maps")
    parser.add_argument("--checkpoint", required=True, help="Fine-tuned RT-DETR checkpoint dir")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--score-threshold", type=float, default=0.15)
    parser.add_argument("--max-detections", type=int, default=3)
    parser.add_argument("--overlay-fraction", type=float, default=0.08)
    parser.add_argument("--rtdetr-config", default="configs/model/rtdetr.yaml")
    parser.add_argument("--output-dir", default="reports/figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_config = load_config(args.rtdetr_config)["data"]
    ground_truth = json.loads(Path(data_config[f"{args.split}_annotations"]).read_text())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForObjectDetection.from_pretrained(
        args.checkpoint, attn_implementation="eager"
    ).to(device)
    image_processor = AutoImageProcessor.from_pretrained(args.checkpoint)

    output_dir = Path(args.output_dir) / f"attention_{args.split}"
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered = 0
    for image_info in ground_truth["images"]:
        image = Image.open(Path(data_config["image_dir"]) / image_info["file_name"]).convert("RGB")
        overlay_height = round(image.height * args.overlay_fraction)
        detections = explain_detections(
            model, image_processor, image, device, args.score_threshold, args.max_detections
        )

        for rank, detection in enumerate(detections, start=1):
            proposal = BoxProposal("billboard", detection.score, detection.box_xyxy)
            panels = [
                draw_detections(image, [], [proposal], "Detection"),
                draw_encoder_attention(image, detection, "Encoder self-attention (box center token)"),
                draw_sampling_points(image, detection, "Decoder deformable sampling points"),
            ]
            panels = [crop_bottom(panel, overlay_height) for panel in panels]
            stem = Path(image_info["file_name"]).stem
            side_by_side(panels, max_panel_width=800).save(output_dir / f"{stem}_det{rank}.jpg", quality=90)
            rendered += 1

    logger.info("Saved %d attention figure(s) to %s", rendered, output_dir)


if __name__ == "__main__":
    main()
