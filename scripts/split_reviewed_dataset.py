"""CLI: split data/annotations/reviewed/reviewed_master.json into train/val/test.

Whole videos are assigned to splits (see docs/decisions/0002-split-by-video.md).
The assignment is computed once from the full image pool (--pool-dir) and saved,
so re-running after reviewing more images keeps every video in the same split.

Usage:
    python scripts/split_reviewed_dataset.py
    python scripts/split_reviewed_dataset.py --rebuild-assignment  # after adding new videos
"""

import argparse
import json
from pathlib import Path

from vit.data.dataset_split import (
    SPLIT_NAMES,
    load_or_build_assignment,
    split_coco_by_video,
    write_splits,
)
from vit.utils.logging import get_logger

logger = get_logger(__name__, log_file="data.log")

MIN_RECOMMENDED_IMAGES = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a reviewed COCO dataset into train/val/test"
    )
    parser.add_argument("--input", default="data/annotations/reviewed/reviewed_master.json")
    parser.add_argument("--output-dir", default="data/annotations/reviewed")
    parser.add_argument("--assignment", default="data/annotations/reviewed/split_assignment.json")
    parser.add_argument(
        "--pool-dir", default="data/raw", help="Full image pool used to build the assignment"
    )
    parser.add_argument("--rebuild-assignment", action="store_true")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    coco = json.loads(Path(args.input).read_text())

    n_images = len(coco["images"])
    if n_images < MIN_RECOMMENDED_IMAGES:
        logger.warning(
            "Only %d reviewed image(s): a fixed train/val/test split is not statistically "
            "meaningful yet. Proceeding anyway so the pipeline can be exercised end-to-end.",
            n_images,
        )

    assignment = load_or_build_assignment(
        Path(args.assignment),
        Path(args.pool_dir),
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed,
        rebuild=args.rebuild_assignment,
    )
    splits = split_coco_by_video(coco, assignment)
    paths = {name: Path(args.output_dir) / f"{name}.json" for name in SPLIT_NAMES}
    write_splits(splits, paths)

    for name, subset in splits.items():
        if not subset["images"]:
            logger.warning("%s split is empty: no reviewed images from its videos yet", name)
        logger.info(
            "%s: %d image(s), %d annotation(s) -> %s",
            name,
            len(subset["images"]),
            len(subset["annotations"]),
            paths[name],
        )


if __name__ == "__main__":
    main()
