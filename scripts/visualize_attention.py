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

from vit.data.dataset_split import subset_by_file_names
from vit.eval.figures import render_attention_maps
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
    parser.add_argument(
        "--images",
        nargs="+",
        metavar="FILE_NAME",
        help="Render only these images of the split (e.g. the hand-picked README examples)",
    )
    parser.add_argument("--output-dir", default="reports/figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_config = load_config(args.rtdetr_config)["data"]
    ground_truth = json.loads(Path(data_config[f"{args.split}_annotations"]).read_text())
    if args.images:
        ground_truth = subset_by_file_names(ground_truth, args.images)

    output_dir = Path(args.output_dir) / f"attention_{args.split}"
    rendered = render_attention_maps(
        Path(args.checkpoint),
        ground_truth,
        Path(data_config["image_dir"]),
        output_dir,
        args.score_threshold,
        args.max_detections,
        args.overlay_fraction,
    )
    logger.info("Saved %d attention figure(s) to %s", rendered, output_dir)


if __name__ == "__main__":
    main()
