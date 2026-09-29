"""CLI: split data/annotations/reviewed/reviewed_master.json into train/val/test.

Usage:
    python scripts/split_reviewed_dataset.py
"""
import argparse
import json
from pathlib import Path

from vit.data.dataset_split import split_coco_dataset
from vit.utils.logging import get_logger

logger = get_logger(__name__, log_file="data.log")

MIN_RECOMMENDED_IMAGES = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split a reviewed COCO dataset into train/val/test")
    parser.add_argument("--input", default="data/annotations/reviewed/reviewed_master.json")
    parser.add_argument("--output-dir", default="data/annotations/reviewed")
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
            "meaningful yet (k-fold cross-validation is recommended once the dataset stays "
            "this small -- see project notes). Proceeding anyway so the pipeline can be "
            "exercised end-to-end.",
            n_images,
        )

    splits = split_coco_dataset(
        coco,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, subset in splits.items():
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
