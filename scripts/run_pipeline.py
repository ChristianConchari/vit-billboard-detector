"""CLI: run the whole pipeline from a Label Studio export to reports/runs/<run>/.

Usage:
    python scripts/run_pipeline.py --export-path ~/Downloads/project-1-at-....zip
    python scripts/run_pipeline.py  # reuse the imported dataset
    python scripts/run_pipeline.py --checkpoint checkpoints/rtdetr/<run>/best
    python scripts/run_pipeline.py --export-path <export>.zip --note learning-curve
    python scripts/run_pipeline.py --set training.freeze_backbone=false --note ablation
"""

import argparse
from pathlib import Path

from vit.pipeline import PipelineConfigs, run_pipeline
from vit.utils.config import apply_overrides, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the end-to-end billboard detection pipeline"
    )
    parser.add_argument(
        "--export-path", type=Path, help="Label Studio COCO export (.zip or .json)"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Evaluate this RT-DETR checkpoint instead of training",
    )
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--pipeline-config", default="configs/pipeline.yaml")
    parser.add_argument("--rtdetr-config", default="configs/model/rtdetr.yaml")
    parser.add_argument(
        "--grounding-dino-config", default="configs/model/grounding_dino.yaml"
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override an RT-DETR config value, e.g. "
        "training.freeze_backbone=false (repeatable)",
    )
    parser.add_argument(
        "--note", help='MLflow tag to group related runs, e.g. "learning-curve"'
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configs = PipelineConfigs(
        pipeline=load_config(args.pipeline_config),
        rtdetr=load_config(args.rtdetr_config),
        grounding_dino=load_config(args.grounding_dino_config),
    )
    apply_overrides(configs.rtdetr, args.overrides)

    run_pipeline(
        configs,
        export_path=args.export_path.expanduser() if args.export_path else None,
        checkpoint=args.checkpoint,
        render_figures=not args.skip_figures,
        note=args.note,
    )


if __name__ == "__main__":
    main()
