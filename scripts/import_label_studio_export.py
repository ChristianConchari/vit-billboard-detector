"""CLI: import a Label Studio COCO export into data/annotations/reviewed.

Usage:
    python scripts/import_label_studio_export.py \
        --export-path ~/Downloads/project-1-at-....zip
"""

import argparse
from pathlib import Path

from vit.data.label_studio_export import import_label_studio_export
from vit.utils.logging import get_logger

logger = get_logger(__name__, log_file="data.log")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import a Label Studio COCO export")
    parser.add_argument(
        "--export-path", required=True, help="Path to the exported .zip or .json"
    )
    parser.add_argument(
        "--output", default="data/annotations/reviewed/reviewed_master.json"
    )
    parser.add_argument("--images-out-dir", default="data/processed")
    parser.add_argument("--extract-dir", default="data/interim/label_studio_export")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    coco, missing = import_label_studio_export(
        Path(args.export_path).expanduser(),
        output_path=Path(args.output),
        images_out_dir=Path(args.images_out_dir),
        extract_dir=Path(args.extract_dir),
        fallback_image_dirs=[Path("data/raw")],
    )
    if missing:
        logger.warning(
            "Could not find image file(s), copy manually into %s: %s",
            args.images_out_dir,
            missing,
        )
    logger.info(
        "Wrote %d image(s) / %d annotation(s) to %s",
        len(coco["images"]),
        len(coco["annotations"]),
        args.output,
    )


if __name__ == "__main__":
    main()
