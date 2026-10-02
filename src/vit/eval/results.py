"""Render a pipeline run summary as Markdown tables for the README and the report."""

from typing import Any

from vit.eval.coco_evaluation import METRIC_NAMES

MODEL_TITLES = {"rtdetr": "RT-DETR fine-tuned", "grounding-dino": "Grounding DINO zero-shot"}


def results_markdown(summary: dict[str, Any]) -> str:
    calibration = summary["calibration"]
    latency = summary["latency"]
    lines = [
        f"# Results: {summary['run']}",
        "",
        f"Checkpoint: `{summary['checkpoint']}`",
        "",
        "## Dataset",
        "",
        "| Split | Images | Boxes |",
        "|---|--:|--:|",
        *(
            f"| {split} | {counts['images']} | {counts['boxes']} |"
            for split, counts in summary["dataset"].items()
        ),
        "",
        "## Test metrics (COCO)",
        "",
        f"| Model | {' | '.join(METRIC_NAMES)} |",
        f"|---|{'--:|' * len(METRIC_NAMES)}",
        *(
            f"| {MODEL_TITLES[model]} | {' | '.join(_metric(metrics[m]) for m in METRIC_NAMES)} |"
            for model, metrics in summary["test_metrics"].items()
        ),
        "",
        "`-` means there are no ground-truth objects in that size range.",
        "",
        "## RT-DETR score threshold (calibrated on val for best F1)",
        "",
        "| Threshold | Precision | Recall | F1 | IoU |",
        "|--:|--:|--:|--:|--:|",
        f"| {calibration['score_threshold']:.3f} | {calibration['precision']:.3f} "
        f"| {calibration['recall']:.3f} | {calibration['f1']:.3f} | {calibration['iou_threshold']} |",
        "",
        f"## Inference latency ({latency['device']}, batch size 1, end to end)",
        "",
        "| Model | Mean (ms) | Median (ms) | p95 (ms) | FPS |",
        "|---|--:|--:|--:|--:|",
        *(
            f"| {MODEL_TITLES[model]} | {stats['mean_ms']:.1f} | {stats['median_ms']:.1f} "
            f"| {stats['p95_ms']:.1f} | {stats['fps']:.1f} |"
            for model, stats in latency["models"].items()
        ),
        "",
    ]
    return "\n".join(lines)


def _metric(value: float) -> str:
    return "-" if value < 0 else f"{value:.3f}"
