"""CLI: evaluate a detector on a reviewed split with COCO metrics.

Compares the fine-tuned RT-DETR against the Grounding DINO zero-shot baseline
on the same ground truth. Metrics are written to reports/metrics/ and MLflow.

Usage:
    python scripts/evaluate_detector.py --model rtdetr --checkpoint checkpoints/rtdetr/<run>/best
    python scripts/evaluate_detector.py --model grounding-dino
"""
import argparse
import json
from pathlib import Path

import mlflow

from vit.eval.coco_evaluation import evaluate_detector
from vit.inference.detection import Detector
from vit.inference.factory import build_grounding_dino_detector, build_rtdetr_detector
from vit.utils.config import load_config
from vit.utils.logging import get_logger

logger = get_logger(__name__, log_file="evaluation.log")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a detector with COCO metrics")
    parser.add_argument("--model", choices=["rtdetr", "grounding-dino"], required=True)
    parser.add_argument("--checkpoint", help="Fine-tuned RT-DETR checkpoint dir (required for rtdetr)")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--rtdetr-config", default="configs/model/rtdetr.yaml")
    parser.add_argument("--grounding-dino-config", default="configs/model/grounding_dino.yaml")
    parser.add_argument("--output-dir", default="reports/metrics")
    return parser.parse_args()


def build_detector(args: argparse.Namespace, rtdetr_config: dict) -> Detector:
    if args.model == "rtdetr":
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required for --model rtdetr")
        return build_rtdetr_detector(
            args.checkpoint,
            rtdetr_config["model"]["label_names"],
            score_threshold=rtdetr_config["evaluation"]["score_threshold"],
        )

    gdino_config = load_config(args.grounding_dino_config)
    return build_grounding_dino_detector(
        gdino_config, box_threshold=gdino_config["evaluation"]["box_threshold"]
    )


def main() -> None:
    args = parse_args()
    rtdetr_config = load_config(args.rtdetr_config)
    data_config = rtdetr_config["data"]
    ground_truth = json.loads(Path(data_config[f"{args.split}_annotations"]).read_text())

    metrics = evaluate_detector(build_detector(args, rtdetr_config), ground_truth, data_config["image_dir"])

    output_path = Path(args.output_dir) / f"{args.model}_{args.split}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2))

    mlflow.set_tracking_uri(rtdetr_config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(rtdetr_config["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name=f"eval-{args.model}-{args.split}"):
        mlflow.log_params(
            {"model": args.model, "split": args.split, "checkpoint": args.checkpoint or "zero-shot"}
        )
        mlflow.log_metrics({f"{args.split}_{k}": v for k, v in metrics.items()})

    logger.info(
        "%s on %s (%d images): %s -> %s",
        args.model,
        args.split,
        len(ground_truth["images"]),
        " | ".join(f"{k} {v:.4f}" for k, v in metrics.items()),
        output_path,
    )


if __name__ == "__main__":
    main()
