"""CLI entry point: run Grounding DINO auto-labeling over data/raw.

Usage:
    python scripts/run_auto_labeling.py \
        --config configs/model/grounding_dino.yaml \
        --images-dir data/raw \
        --output data/annotations/auto/auto_labels.json
"""

import argparse
from pathlib import Path

from PIL import Image

from vit.labeling.coco_writer import CocoDatasetBuilder
from vit.labeling.grounding_dino_labeler import GroundingDinoAutoLabeler
from vit.utils.config import load_config
from vit.utils.logging import get_logger

logger = get_logger(__name__, log_file="labeling.log")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
CATEGORY_NAME = "billboard"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grounding DINO auto-labeling pipeline"
    )
    parser.add_argument("--config", default="configs/model/grounding_dino.yaml")
    parser.add_argument("--images-dir", default="data/raw")
    parser.add_argument(
        "--output",
        default=None,
        help="Overrides output.auto_labels_dir/auto_labels.json from config",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    images_dir = Path(args.images_dir)
    image_paths = sorted(
        p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        logger.warning("No images found in %s", images_dir)
        return

    output_path = (
        Path(args.output)
        if args.output
        else Path(config["output"]["auto_labels_dir"]) / "auto_labels.json"
    )

    labeler = GroundingDinoAutoLabeler(
        checkpoint=config["model"]["checkpoint"],
        prompts=config["prompts"],
        box_threshold=config["thresholds"]["box_threshold"],
        nms_iou_threshold=config["thresholds"]["nms_iou_threshold"],
    )
    builder = CocoDatasetBuilder(category_names=[CATEGORY_NAME])

    for image_path in image_paths:
        logger.info("Processing %s", image_path.name)
        image = Image.open(image_path).convert("RGB")
        image_id = builder.add_image(
            file_name=image_path.name, width=image.width, height=image.height
        )

        proposals = labeler.predict(image)
        logger.info("  -> %d proposal(s)", len(proposals))

        for proposal in proposals:
            x_min, y_min, x_max, y_max = proposal.box_xyxy
            builder.add_annotation(
                image_id=image_id,
                category_name=CATEGORY_NAME,
                bbox_xywh=(x_min, y_min, x_max - x_min, y_max - y_min),
                score=proposal.score,
            )

    builder.save(output_path)
    logger.info(
        "Saved %d image(s) / %d annotation(s) to %s",
        len(builder.images),
        len(builder.annotations),
        output_path,
    )
    logger.info(
        "Review and correct these boxes before moving them to data/annotations/reviewed"
    )


if __name__ == "__main__":
    main()
