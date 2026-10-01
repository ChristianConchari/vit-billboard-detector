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

from vit.data.dataset_split import assign_videos_to_splits, split_coco_by_video
from vit.utils.logging import get_logger

logger = get_logger(__name__, log_file="data.log")

MIN_RECOMMENDED_IMAGES = 20
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split a reviewed COCO dataset into train/val/test")
    parser.add_argument("--input", default="data/annotations/reviewed/reviewed_master.json")
    parser.add_argument("--output-dir", default="data/annotations/reviewed")
    parser.add_argument("--assignment", default="data/annotations/reviewed/split_assignment.json")
    parser.add_argument("--pool-dir", default="data/raw", help="Full image pool used to build the assignment")
    parser.add_argument("--rebuild-assignment", action="store_true")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_or_build_assignment(args: argparse.Namespace) -> dict[str, str]:
    assignment_path = Path(args.assignment)
    if assignment_path.exists() and not args.rebuild_assignment:
        logger.info("Using existing split assignment %s", assignment_path)
        return json.loads(assignment_path.read_text())

    pool = [p.name for p in Path(args.pool_dir).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
    if not pool:
        raise SystemExit(f"No images in {args.pool_dir} to build the split assignment from")

    assignment = assign_videos_to_splits(
        pool,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    assignment_path.parent.mkdir(parents=True, exist_ok=True)
    assignment_path.write_text(json.dumps(assignment, indent=2, sort_keys=True))
    logger.info(
        "Built split assignment for %d video(s) from %d pool image(s) -> %s",
        len(assignment),
        len(pool),
        assignment_path,
    )
    return assignment


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

    splits = split_coco_by_video(coco, load_or_build_assignment(args))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, subset in splits.items():
        if not subset["images"]:
            logger.warning("%s split is empty: no reviewed images from its videos yet", name)
        path = output_dir / f"{name}.json"
        path.write_text(json.dumps(subset, indent=2))
        logger.info(
            "%s: %d image(s), %d annotation(s) -> %s",
            name,
            len(subset["images"]),
            len(subset["annotations"]),
            path,
        )


if __name__ == "__main__":
    main()
