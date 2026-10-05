"""Detect billboards in an image, an image folder or a video with a fine-tuned RT-DETR.

Writes annotated copies plus detections.json (boxes in pixels, xyxy) to --output-dir.

Usage:
    billboard-detect path/to/video.mp4 --checkpoint checkpoints/rtdetr/<run>/best
    billboard-detect path/to/images/ --checkpoint checkpoints/rtdetr/<run>/best
"""

import argparse
import json
import time
from pathlib import Path

from vit.eval.threshold import load_calibrated_threshold
from vit.inference.factory import build_rtdetr_detector
from vit.inference.media import (
    UnreadableVideoError,
    detect_in_images,
    detect_in_video,
    is_video,
    list_images,
)
from vit.utils.config import load_config
from vit.utils.logging import get_logger

logger = get_logger(__name__, log_file="inference.log")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect billboards in images or videos"
    )
    parser.add_argument(
        "source", type=Path, help="Image file, folder of images, or video file"
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Fine-tuned RT-DETR checkpoint dir"
    )
    parser.add_argument("--config", default="configs/model/rtdetr.yaml")
    parser.add_argument(
        "--score-threshold",
        type=float,
        help="Defaults to the checkpoint's calibrated threshold, else "
        "inference.score_threshold",
    )
    parser.add_argument(
        "--overlay-fraction",
        type=float,
        default=0.08,
        help="Bottom fraction of each frame to crop (camera timestamp/GPS "
        "overlay); 0 keeps it",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/detections"))
    return parser.parse_args()


def resolve_score_threshold(
    requested: float | None, checkpoint_dir: Path, config: dict
) -> float:
    if requested is not None:
        return requested
    calibrated = load_calibrated_threshold(checkpoint_dir)
    if calibrated is not None:
        return calibrated
    return config["inference"]["score_threshold"]


def main() -> None:
    args = parse_args()
    if not args.source.exists():
        raise SystemExit(f"Source not found: {args.source}")

    config = load_config(args.config)
    score_threshold = resolve_score_threshold(
        args.score_threshold, Path(args.checkpoint), config
    )
    logger.info("Score threshold: %.3f", score_threshold)
    detector = build_rtdetr_detector(
        args.checkpoint, config["model"]["label_names"], score_threshold
    )

    start = time.perf_counter()
    if is_video(args.source):
        try:
            records = detect_in_video(
                detector, args.source, args.output_dir, args.overlay_fraction
            )
        except UnreadableVideoError as error:
            raise SystemExit(str(error)) from None
    else:
        images = list_images(args.source)
        if not images:
            raise SystemExit(f"No images found in {args.source}")
        records = detect_in_images(
            detector, images, args.output_dir, args.overlay_fraction
        )
    elapsed = time.perf_counter() - start

    (args.output_dir / "detections.json").write_text(json.dumps(records, indent=2))
    logger.info(
        "%d frame(s)/image(s), %d detection(s) in %.1f s (%.1f per second) -> %s",
        len(records),
        sum(len(r["detections"]) for r in records),
        elapsed,
        len(records) / elapsed,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
