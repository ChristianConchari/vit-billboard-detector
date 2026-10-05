"""CLI: headline charts for the README and the report from a pipeline run.

Reads reports/runs/<run>/summary.json (and, optionally, the localization JSON) and
writes model_comparison.png, latency.png and ap_per_iou.png to reports/figures/.

Usage:
    python scripts/plot_results.py --run-dir reports/runs/<run> \
        --localization reports/metrics/localization_test.json
"""

import argparse
import json
from pathlib import Path

from vit.eval.result_charts import plot_ap_per_iou, plot_latency, plot_model_comparison
from vit.utils.logging import get_logger

logger = get_logger(__name__, log_file="evaluation.log")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot headline result charts")
    parser.add_argument(
        "--run-dir", type=Path, required=True, help="Directory with summary.json"
    )
    parser.add_argument(
        "--localization",
        type=Path,
        help="JSON written by scripts/analyze_localization.py",
    )
    parser.add_argument(
        "--checkpoint-run",
        help="Run name whose AP curves to plot (defaults to the run of --run-dir)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports/figures"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = json.loads((args.run_dir / "summary.json").read_text())
    plot_model_comparison(
        summary["test_metrics"], args.output_dir / "model_comparison.png"
    )
    plot_latency(summary["latency"], args.output_dir / "latency.png")

    if args.localization:
        run_name = args.checkpoint_run or summary["run"]
        curves = [
            curve
            for curve in json.loads(args.localization.read_text())["ap_per_iou"]
            if curve["model"] in (run_name, "Grounding DINO zero-shot")
        ]
        if not any(curve["model"] == run_name for curve in curves):
            raise SystemExit(f"No AP curves for run {run_name} in {args.localization}")
        plot_ap_per_iou(curves, args.output_dir / "ap_per_iou.png")

    logger.info("Result charts -> %s", args.output_dir)


if __name__ == "__main__":
    main()
