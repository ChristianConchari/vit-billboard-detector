"""Headline charts for a pipeline run: model comparison, latency and AP per IoU."""

from pathlib import Path
from typing import Any

import numpy as np

from vit.eval.chart_style import (
    RTDETR_COLOR,
    SURFACE_COLOR,
    TEXT_PRIMARY,
    ZERO_SHOT_COLOR,
    add_legend,
    new_figure,
    save,
    style_axes,
)

MODELS = (
    ("rtdetr", "RT-DETR fine-tuned", RTDETR_COLOR),
    ("grounding-dino", "Grounding DINO zero-shot", ZERO_SHOT_COLOR),
)
COMPARED_METRICS = ("mAP", "AP50", "AP75")


def plot_model_comparison(
    test_metrics: dict[str, dict[str, float]], output_path: Path
) -> None:
    """Grouped bars of test mAP, AP50 and AP75 for both models, each bar labeled."""
    figure, axes = new_figure()
    positions = np.arange(len(COMPARED_METRICS))
    width = 0.36
    for offset, (key, label, color) in zip(
        (-width / 2, width / 2), MODELS, strict=True
    ):
        values = [test_metrics[key][metric] for metric in COMPARED_METRICS]
        bars = axes.bar(
            positions + offset,
            values,
            width - 0.02,
            color=color,
            edgecolor=SURFACE_COLOR,
            linewidth=2,
            label=label,
        )
        axes.bar_label(bars, fmt="%.2f", padding=3, color=TEXT_PRIMARY, fontsize=9)
    axes.set_xticks(positions, ["mAP@[.5:.95]", "AP50", "AP75"])
    axes.set_ylim(0, 1.05)
    style_axes(
        axes, "Test accuracy: fine-tuned vs. zero-shot", "", "Average precision (COCO)"
    )
    add_legend(axes, "upper left")
    save(figure, output_path)


def plot_latency(latency: dict[str, Any], output_path: Path) -> None:
    """Horizontal bars of end-to-end milliseconds per image, labeled with ms and FPS."""
    figure, axes = new_figure(height=3)
    labels, values, colors = [], [], []
    for key, label, color in reversed(MODELS):
        labels.append(label)
        values.append(latency["models"][key]["mean_ms"])
        colors.append(color)
    bars = axes.barh(
        labels, values, height=0.5, color=colors, edgecolor=SURFACE_COLOR, linewidth=2
    )
    axes.bar_label(
        bars,
        labels=[f"{ms:.0f} ms · {1000 / ms:.0f} FPS" for ms in values],
        padding=4,
        color=TEXT_PRIMARY,
        fontsize=9,
    )
    axes.set_xlim(0, max(values) * 1.3)
    style_axes(
        axes,
        "Inference latency per image (batch size 1)",
        f"Milliseconds, end to end, on {latency['device']}",
        "",
        grid_axis="x",
    )
    save(figure, output_path)


def plot_ap_per_iou(curves: list[dict[str, Any]], output_path: Path) -> None:
    """AP at each IoU threshold: solid lines with markers for test, dashed for val."""
    figure, axes = new_figure()
    for curve in curves:
        color = (
            ZERO_SHOT_COLOR
            if curve["model"].startswith("Grounding DINO")
            else RTDETR_COLOR
        )
        name = (
            "Grounding DINO zero-shot"
            if color == ZERO_SHOT_COLOR
            else "RT-DETR fine-tuned"
        )
        thresholds = [float(t) for t in curve["ap"]]
        is_test = curve["split"] == "test"
        axes.plot(
            thresholds,
            list(curve["ap"].values()),
            color=color,
            linewidth=2,
            linestyle="-" if is_test else (0, (4, 3)),
            marker="o" if is_test else None,
            markersize=6,
            markeredgecolor=SURFACE_COLOR,
            markeredgewidth=1.5,
            label=f"{name} · {curve['split']}",
        )
    axes.set_ylim(0, 1.02)
    style_axes(
        axes,
        "Where mAP is lost: AP per IoU threshold",
        "IoU threshold",
        "Average precision",
    )
    add_legend(axes, "lower left")
    save(figure, output_path)
