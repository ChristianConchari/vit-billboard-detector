"""CLI: import a Label Studio COCO export into data/annotations/reviewed.

Usage:
    python scripts/import_label_studio_export.py --export-path ~/Downloads/project-1-at-....zip
"""
import argparse
import json
from pathlib import Path

from vit.data.label_studio_export import (
    copy_referenced_images,
    extract_if_zip,
    find_coco_json,
    normalize_file_names,
)
from vit.utils.logging import get_logger

logger = get_logger(__name__, log_file="data.log")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import a Label Studio COCO export")
    parser.add_argument("--export-path", required=True, help="Path to the exported .zip or .json")
    parser.add_argument("--output", default="data/annotations/reviewed/reviewed_master.json")
    parser.add_argument("--images-out-dir", default="data/processed")
    parser.add_argument("--extract-dir", default="data/interim/label_studio_export")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_path = Path(args.export_path).expanduser()

    if export_path.suffix.lower() == ".zip":
        search_root = extract_if_zip(export_path, Path(args.extract_dir))
        coco_json_path = find_coco_json(search_root)
    else:
        coco_json_path = export_path
        search_root = export_path.parent

    coco = json.loads(coco_json_path.read_text())
    coco = normalize_file_names(coco)

    # Fall back to data/raw in case the export didn't bundle image bytes
    # (e.g. Local Storage-backed tasks exported as annotations-only).
    missing = copy_referenced_images(
        coco, search_dirs=[search_root, Path("data/raw")], output_dir=Path(args.images_out_dir)
    )
    if missing:
        logger.warning(
            "Could not find image file(s), copy manually into %s: %s",
            args.images_out_dir,
            missing,
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(coco, indent=2))

    logger.info(
        "Wrote %d image(s) / %d annotation(s) to %s",
        len(coco["images"]),
        len(coco["annotations"]),
        output_path,
    )


if __name__ == "__main__":
    main()
