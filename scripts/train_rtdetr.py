"""CLI: fine-tune RT-DETR on the reviewed billboard dataset.

Usage:
    python scripts/train_rtdetr.py --config configs/model/rtdetr.yaml
"""

import argparse

import mlflow

from vit.train.trainer import new_run_name, train_rtdetr
from vit.utils.config import load_config
from vit.utils.tracking import flatten, git_tags, start_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune RT-DETR")
    parser.add_argument("--config", default="configs/model/rtdetr.yaml")
    parser.add_argument("--epochs", type=int, help="Overrides training.epochs from the config")
    parser.add_argument("--note", help="MLflow tag to group related runs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    run_name = new_run_name()
    with start_run(config["mlflow"], run_name):
        mlflow.set_tags({**git_tags(), "stage": "train"})
        if args.note:
            mlflow.set_tags({"note": args.note, "mlflow.note.content": args.note})
        mlflow.log_params(flatten(config, prefix="rtdetr."))
        mlflow.log_dict(config, "configs/rtdetr.yaml")
        train_rtdetr(config, run_name=run_name)


if __name__ == "__main__":
    main()
