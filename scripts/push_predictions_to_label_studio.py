"""CLI: push data/annotations/auto/*.json as predictions onto existing Label
Studio tasks (matched by image file name), so review only requires
correcting boxes instead of drawing them from scratch.

Usage:
    export LABEL_STUDIO_API_KEY=xxxxxxxx
    python scripts/push_predictions_to_label_studio.py --project-id 1 --exclude-split test

Test videos are labeled from scratch (docs/decisions/0003-label-test-from-scratch.md),
so they must not receive pre-labels.
"""
import argparse
import json
import os
from pathlib import Path

from vit.data.dataset_split import exclude_split
from vit.labeling.label_studio_sync import push_predictions
from vit.utils.logging import get_logger

logger = get_logger(__name__, log_file="labeling.log")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Push COCO auto-labels as Label Studio predictions")
    parser.add_argument("--label-studio-url", default="http://localhost:8080")
    parser.add_argument("--api-key", default=os.environ.get("LABEL_STUDIO_API_KEY"))
    parser.add_argument("--project-id", type=int, required=True)
    parser.add_argument("--coco-file", default="data/annotations/auto/auto_labels.json")
    parser.add_argument("--model-version", default="grounding-dino-auto-label")
    parser.add_argument(
        "--exclude-split",
        choices=["train", "val", "test"],
        help="Don't pre-label images whose video belongs to this split",
    )
    parser.add_argument("--split-assignment", default="data/annotations/reviewed/split_assignment.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise SystemExit(
            "Missing API key: pass --api-key or set LABEL_STUDIO_API_KEY "
            "(Account & Settings -> Access Token, in the Label Studio UI)"
        )

    coco = json.loads(Path(args.coco_file).read_text())
    if args.exclude_split:
        assignment = json.loads(Path(args.split_assignment).read_text())
        coco = exclude_split(coco, assignment, args.exclude_split)
        logger.info("Excluding %s videos: %d image(s) left to pre-label", args.exclude_split, len(coco["images"]))

    summary = push_predictions(
        label_studio_url=args.label_studio_url,
        api_key=args.api_key,
        project_id=args.project_id,
        coco=coco,
        model_version=args.model_version,
    )

    logger.info(
        "Pushed predictions for %d/%d image(s), %d unmatched (sync Local Storage first if >0)",
        summary["pushed"],
        summary["total_images"],
        summary["unmatched"],
    )


if __name__ == "__main__":
    main()
