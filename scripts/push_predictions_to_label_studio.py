"""CLI: push data/annotations/auto/*.json as predictions onto existing Label
Studio tasks (matched by image file name), so review only requires
correcting boxes instead of drawing them from scratch.

Usage:
    export LABEL_STUDIO_API_KEY=xxxxxxxx
    python scripts/push_predictions_to_label_studio.py --project-id 1
"""
import argparse
import json
import os
from pathlib import Path

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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise SystemExit(
            "Missing API key: pass --api-key or set LABEL_STUDIO_API_KEY "
            "(Account & Settings -> Access Token, in the Label Studio UI)"
        )

    coco = json.loads(Path(args.coco_file).read_text())

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
