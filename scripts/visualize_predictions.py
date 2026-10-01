"""CLI: render RT-DETR vs. Grounding DINO predictions next to the ground truth.

Writes one side-by-side image per split image to reports/figures/predictions_<split>/.
Green boxes are ground truth, red boxes are predictions with their score. The
bottom band of each frame is cropped because it carries the camera's
timestamp/GPS overlay.

Usage:
    python scripts/visualize_predictions.py --checkpoint checkpoints/rtdetr/<run>/best
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image

from vit.eval.visualization import crop_bottom, draw_detections, side_by_side
from vit.inference.factory import build_grounding_dino_detector, build_rtdetr_detector
from vit.utils.config import load_config
from vit.utils.logging import get_logger

logger = get_logger(__name__, log_file="evaluation.log")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize detector predictions")
    parser.add_argument("--checkpoint", required=True, help="Fine-tuned RT-DETR checkpoint dir")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--rtdetr-threshold", type=float, default=0.5)
    parser.add_argument("--grounding-dino-threshold", type=float, default=0.35)
    parser.add_argument("--rtdetr-config", default="configs/model/rtdetr.yaml")
    parser.add_argument("--grounding-dino-config", default="configs/model/grounding_dino.yaml")
    parser.add_argument(
        "--overlay-fraction",
        type=float,
        default=0.08,
        help="Bottom fraction of each frame to crop (timestamp/GPS overlay); 0 keeps it",
    )
    parser.add_argument("--output-dir", default="reports/figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rtdetr_config = load_config(args.rtdetr_config)
    data_config = rtdetr_config["data"]
    ground_truth = json.loads(Path(data_config[f"{args.split}_annotations"]).read_text())

    detectors = {
        f"RT-DETR fine-tuned (score >= {args.rtdetr_threshold})": build_rtdetr_detector(
            args.checkpoint, rtdetr_config["model"]["label_names"], args.rtdetr_threshold
        ),
        f"Grounding DINO zero-shot (score >= {args.grounding_dino_threshold})": (
            build_grounding_dino_detector(
                load_config(args.grounding_dino_config), args.grounding_dino_threshold
            )
        ),
    }

    boxes_by_image = defaultdict(list)
    for annotation in ground_truth["annotations"]:
        boxes_by_image[annotation["image_id"]].append(annotation["bbox"])

    output_dir = Path(args.output_dir) / f"predictions_{args.split}"
    output_dir.mkdir(parents=True, exist_ok=True)
    for image_info in ground_truth["images"]:
        image = Image.open(Path(data_config["image_dir"]) / image_info["file_name"]).convert("RGB")
        overlay_height = round(image.height * args.overlay_fraction)
        panels = [
            crop_bottom(
                draw_detections(
                    image, boxes_by_image[image_info["id"]], detector.predict(image), title
                ),
                overlay_height,
            )
            for title, detector in detectors.items()
        ]
        side_by_side(panels).save(output_dir / image_info["file_name"], quality=90)

    logger.info("Saved %d comparison image(s) to %s", len(ground_truth["images"]), output_dir)


if __name__ == "__main__":
    main()
