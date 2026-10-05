"""CLI: download the trained RT-DETR checkpoint published as a GitHub release.

The URL, checksum and destination come from the `release` section of
configs/model/rtdetr.yaml. After downloading, run the demo with:

    billboard-detect path/to/image.jpg --checkpoint checkpoints/rtdetr-billboard

Usage:
    python scripts/download_model.py
"""

import argparse
from pathlib import Path

from vit.inference.weights import download_checkpoint
from vit.utils.config import load_config
from vit.utils.logging import get_logger

logger = get_logger(__name__, log_file="inference.log")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the trained RT-DETR checkpoint"
    )
    parser.add_argument("--config", default="configs/model/rtdetr.yaml")
    return parser.parse_args()


def main() -> None:
    release = load_config(parse_args().config)["release"]
    checkpoint_dir = Path(release["checkpoint_dir"])
    logger.info("Downloading %s", release["weights_url"])
    download_checkpoint(
        release["weights_url"], release["weights_sha256"], checkpoint_dir
    )
    logger.info("Checkpoint ready (checksum verified) -> %s", checkpoint_dir)


if __name__ == "__main__":
    main()
