"""CLI: fine-tune RT-DETR on the reviewed billboard dataset.

Usage:
    python scripts/train_rtdetr.py --config configs/model/rtdetr.yaml
"""
import argparse

from vit.train.trainer import train_rtdetr
from vit.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune RT-DETR")
    parser.add_argument("--config", default="configs/model/rtdetr.yaml")
    parser.add_argument("--epochs", type=int, help="Overrides training.epochs from the config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    train_rtdetr(config)


if __name__ == "__main__":
    main()
