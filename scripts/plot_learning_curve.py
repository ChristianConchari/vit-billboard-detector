"""CLI: plot RT-DETR test mAP vs. training images from MLflow runs.

Writes the chart (PNG) and the same points as CSV. Runs must share one test split.

Usage:
    python scripts/plot_learning_curve.py --note learning-curve
"""

import argparse
from pathlib import Path

from vit.eval.learning_curve import (
    build_curve,
    fetch_run_results,
    plot_learning_curve,
    write_curve_table,
)
from vit.utils.config import load_config
from vit.utils.logging import get_logger

logger = get_logger(__name__, log_file="evaluation.log")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the learning curve from MLflow runs"
    )
    parser.add_argument("--note", help="Only use runs tagged with this note")
    parser.add_argument("--rtdetr-config", default="configs/model/rtdetr.yaml")
    parser.add_argument(
        "--output", type=Path, default=Path("reports/figures/learning_curve.png")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = fetch_run_results(load_config(args.rtdetr_config)["mlflow"], args.note)
    points, zero_shot_map = build_curve(results)

    plot_learning_curve(points, zero_shot_map, args.output)
    write_curve_table(points, zero_shot_map, args.output.with_suffix(".csv"))
    logger.info(
        "%d run(s), %d training-set size(s) -> %s",
        len(results),
        len(points),
        args.output,
    )


if __name__ == "__main__":
    main()
