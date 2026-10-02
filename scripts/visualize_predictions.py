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
from pathlib import Path

from vit.eval.figures import render_detector_comparison
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

    output_dir = Path(args.output_dir) / f"predictions_{args.split}"
    rendered = render_detector_comparison(
        detectors, ground_truth, Path(data_config["image_dir"]), output_dir, args.overlay_fraction
    )
    logger.info("Saved %d comparison image(s) to %s", rendered, output_dir)


if __name__ == "__main__":
    main()
