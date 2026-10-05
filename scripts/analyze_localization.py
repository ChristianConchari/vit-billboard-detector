"""CLI: localization diagnostics for one or more RT-DETR checkpoints.

Writes reports/metrics/localization_<split>.md (two tables) and the same data as JSON:
- AP at every IoU threshold (0.50-0.95): each checkpoint on the evaluation and a
  reference split, and Grounding DINO zero-shot on the evaluation split;
- systematic edge bias of each model's boxes against the ground truth, next to the
  bias of the Grounding DINO pre-labels themselves.

Usage:
    python scripts/analyze_localization.py \
        --checkpoint checkpoints/rtdetr/<run-a>/best \
        --checkpoint checkpoints/rtdetr/<run-b>/best
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from vit.eval.detections import collect_detections
from vit.eval.localization import (
    IOU_THRESHOLDS,
    EdgeBias,
    ap_per_iou_threshold,
    boxes_by_image,
    edge_bias,
    match_to_ground_truth,
)
from vit.eval.threshold import load_calibrated_threshold
from vit.inference.factory import build_grounding_dino_detector, build_rtdetr_detector
from vit.utils.config import load_config
from vit.utils.logging import get_logger

logger = get_logger(__name__, log_file="evaluation.log")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Localization diagnostics for RT-DETR")
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument(
        "--reference-split", choices=["train", "val", "test"], default="val"
    )
    parser.add_argument(
        "--auto-labels",
        type=Path,
        default=Path("data/annotations/auto/auto_labels.json"),
    )
    parser.add_argument("--rtdetr-config", default="configs/model/rtdetr.yaml")
    parser.add_argument(
        "--grounding-dino-config", default="configs/model/grounding_dino.yaml"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports/metrics"))
    return parser.parse_args()


def pre_labels_for(split: dict, auto_labels: dict) -> dict[int, list[list[float]]]:
    """Grounding DINO pre-label boxes keyed by the split's image ids (via file name)."""
    auto_image_ids = {
        image["file_name"]: image["id"] for image in auto_labels["images"]
    }
    auto_boxes = boxes_by_image(auto_labels["annotations"])
    return {
        image["id"]: auto_boxes.get(auto_image_ids[image["file_name"]], [])
        for image in split["images"]
    }


def bias_row(name: str, bias: EdgeBias) -> str:
    return (
        f"| {name} | {bias.left:+.3f} | {bias.top:+.3f} "
        f"| {bias.right:+.3f} | {bias.bottom:+.3f} "
        f"| {bias.width_ratio:.3f} | {bias.height_ratio:.3f} | {bias.median_iou:.3f} "
        f"| {bias.matches} |"
    )


def report_markdown(
    split: str, ap_curves: list[dict], biases: dict[str, EdgeBias]
) -> str:
    return "\n".join(
        [
            f"# Localization diagnostics ({split})",
            "",
            "## AP per IoU threshold (all areas, up to 100 detections)",
            "",
            "| Model | Split | "
            + " | ".join(f"{t:.2f}" for t in IOU_THRESHOLDS)
            + " |",
            "|---|---|" + "--:|" * len(IOU_THRESHOLDS),
            *(
                f"| {c['model']} | {c['split']} | "
                + " | ".join(f"{v:.2f}" for v in c["ap"].values())
                + " |"
                for c in ap_curves
            ),
            "",
            f"## Edge bias against the {split} ground truth",
            "",
            "Signed edge offsets normalized by the ground-truth box size: positive "
            "means the predicted edge lies outside the ground-truth box (box too "
            "large), negative inside (box too small). Models use their calibrated "
            "score threshold.",
            "",
            "| Boxes | Left | Top | Right | Bottom | Width ratio | Height ratio "
            "| Median IoU "
            "| Matches |",
            "|---|--:|--:|--:|--:|--:|--:|--:|--:|",
            *(bias_row(name, bias) for name, bias in biases.items()),
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.rtdetr_config)
    data_cfg = config["data"]
    splits = {
        name: json.loads(Path(data_cfg[f"{name}_annotations"]).read_text())
        for name in (args.split, args.reference_split)
    }
    ground_truth_boxes = boxes_by_image(splits[args.split]["annotations"])

    ap_curves: list[dict] = []
    biases: dict[str, EdgeBias] = {}
    for checkpoint in args.checkpoint:
        run_name = checkpoint.parent.name
        detector = build_rtdetr_detector(
            checkpoint,
            config["model"]["label_names"],
            config["evaluation"]["score_threshold"],
        )
        for split_name, split in splits.items():
            detections = collect_detections(detector, split, data_cfg["image_dir"])
            ap_curves.append(
                {
                    "model": run_name,
                    "split": split_name,
                    "ap": ap_per_iou_threshold(split, detections),
                }
            )
            if split_name == args.split:
                threshold = (
                    load_calibrated_threshold(checkpoint)
                    or config["inference"]["score_threshold"]
                )
                predicted = boxes_by_image(detections, min_score=threshold)
                biases[run_name] = edge_bias(
                    match_to_ground_truth(ground_truth_boxes, predicted)
                )

    gdino_config = load_config(args.grounding_dino_config)
    zero_shot = build_grounding_dino_detector(
        gdino_config, gdino_config["evaluation"]["box_threshold"]
    )
    zero_shot_detections = collect_detections(
        zero_shot, splits[args.split], data_cfg["image_dir"]
    )
    ap_curves.append(
        {
            "model": "Grounding DINO zero-shot",
            "split": args.split,
            "ap": ap_per_iou_threshold(splits[args.split], zero_shot_detections),
        }
    )

    pre_labels = pre_labels_for(
        splits[args.split], json.loads(args.auto_labels.read_text())
    )
    biases["Grounding DINO pre-labels"] = edge_bias(
        match_to_ground_truth(ground_truth_boxes, pre_labels)
    )

    output_stem = args.output_dir / f"localization_{args.split}"
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    output_stem.with_suffix(".md").write_text(
        report_markdown(args.split, ap_curves, biases)
    )
    output_stem.with_suffix(".json").write_text(
        json.dumps(
            {
                "ap_per_iou": ap_curves,
                "edge_bias": {name: asdict(bias) for name, bias in biases.items()},
            },
            indent=2,
        )
    )
    logger.info("Localization diagnostics -> %s.{md,json}", output_stem)


if __name__ == "__main__":
    main()
