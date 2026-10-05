"""CLI: export the MLflow experiment record to reports/experiments/ for versioning.

Writes runs.csv (one row per run), epochs.csv (loss and val metrics per epoch),
summary.md (learning curve and backbone ablation
with mean and range over seeds) and training_curves.png. Only runs evaluated on the
current test split are included (matched by its content fingerprint).

Usage:
    python scripts/export_experiments.py
"""

import argparse
import json
from pathlib import Path

from vit.data.dataset_split import split_fingerprint
from vit.eval.experiments import (
    fetch_run_records,
    plot_training_curves,
    summary_markdown,
    write_epochs_csv,
    write_runs_csv,
)
from vit.utils.config import load_config
from vit.utils.logging import get_logger

logger = get_logger(__name__, log_file="evaluation.log")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export MLflow runs to versionable reports"
    )
    parser.add_argument("--rtdetr-config", default="configs/model/rtdetr.yaml")
    parser.add_argument(
        "--test-fingerprint",
        help="Defaults to the fingerprint of the current test split",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports/experiments"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.rtdetr_config)
    fingerprint = args.test_fingerprint or split_fingerprint(
        json.loads(Path(config["data"]["test_annotations"]).read_text())
    )
    records = fetch_run_records(config["mlflow"], fingerprint)
    if not records:
        raise SystemExit(f"No tracked runs evaluated on the test split {fingerprint}")

    write_runs_csv(records, args.output_dir / "runs.csv")
    write_epochs_csv(records, args.output_dir / "epochs.csv")
    (args.output_dir / "summary.md").write_text(summary_markdown(records))
    plot_training_curves(records, args.output_dir / "training_curves.png")
    logger.info(
        "%d run(s) on test split %s -> %s", len(records), fingerprint, args.output_dir
    )


if __name__ == "__main__":
    main()
