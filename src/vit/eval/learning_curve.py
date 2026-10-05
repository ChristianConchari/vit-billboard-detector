"""Learning curve from MLflow runs: RT-DETR test mAP vs. training images.

Only runs that trained a model and share the exact same test split are
comparable; mixing test sets would make the curve meaningless, so it's refused.
"""

import csv
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from mlflow.tracking import MlflowClient

from vit.eval.chart_style import (
    RTDETR_COLOR,
    SURFACE_COLOR,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    ZERO_SHOT_COLOR,
    add_legend,
    new_figure,
    save,
    style_axes,
)


@dataclass
class RunResult:
    run_name: str
    train_images: int
    test_split_fingerprint: str
    rtdetr_map: float
    zero_shot_map: float


@dataclass
class CurvePoint:
    train_images: int
    mean_map: float
    min_map: float
    max_map: float
    runs: int


def fetch_run_results(
    mlflow_config: dict[str, Any], note: str | None = None
) -> list[RunResult]:
    client = MlflowClient(tracking_uri=mlflow_config["tracking_uri"])
    experiment = client.get_experiment_by_name(mlflow_config["experiment_name"])
    if experiment is None:
        return []

    query = "tags.stage = 'train+evaluate'"
    if note:
        query += f" and tags.note = '{note}'"
    runs = client.search_runs(
        [experiment.experiment_id], filter_string=query, max_results=1000
    )
    return [
        RunResult(
            run_name=run.info.run_name,
            train_images=int(run.data.params["data.train.images"]),
            test_split_fingerprint=run.data.tags["data.test_fingerprint"],
            rtdetr_map=run.data.metrics["test/rtdetr/mAP"],
            zero_shot_map=run.data.metrics["test/grounding-dino/mAP"],
        )
        for run in runs
        if "data.test_fingerprint" in run.data.tags
        and "test/rtdetr/mAP" in run.data.metrics
    ]


def build_curve(results: list[RunResult]) -> tuple[list[CurvePoint], float]:
    """Aggregate runs per training-set size; return the points and zero-shot mAP."""
    if not results:
        raise ValueError("No finished train+evaluate runs to plot")
    test_splits = {r.test_split_fingerprint for r in results}
    if len(test_splits) > 1:
        raise ValueError(
            f"Runs were evaluated on {len(test_splits)} different test splits; "
            "filter them with --note so the curve compares like with like"
        )

    maps_by_size: dict[int, list[float]] = defaultdict(list)
    for result in results:
        maps_by_size[result.train_images].append(result.rtdetr_map)
    points = [
        CurvePoint(size, statistics.fmean(maps), min(maps), max(maps), len(maps))
        for size, maps in sorted(maps_by_size.items())
    ]
    return points, statistics.fmean(r.zero_shot_map for r in results)


def plot_learning_curve(
    points: list[CurvePoint], zero_shot_map: float, output_path: Path
) -> None:
    sizes = [p.train_images for p in points]
    means = [p.mean_map for p in points]

    figure, axes = new_figure()
    axes.axhline(
        zero_shot_map, color=ZERO_SHOT_COLOR, linewidth=2, solid_capstyle="round"
    )
    axes.vlines(
        sizes,
        [p.min_map for p in points],
        [p.max_map for p in points],
        color=RTDETR_COLOR,
        linewidth=2,
        alpha=0.35,
    )
    axes.plot(
        sizes,
        means,
        color=RTDETR_COLOR,
        linewidth=2,
        marker="o",
        markersize=8,
        markeredgecolor=SURFACE_COLOR,
        markeredgewidth=2,
        solid_joinstyle="round",
        solid_capstyle="round",
        label="RT-DETR fine-tuned",
    )
    axes.plot(
        [], [], color=ZERO_SHOT_COLOR, linewidth=2, label="Grounding DINO zero-shot"
    )

    last = points[-1]
    axes.annotate(
        f"RT-DETR {last.mean_map:.2f}",
        (last.train_images, last.mean_map),
        xytext=(8, -14),
        textcoords="offset points",
        color=TEXT_PRIMARY,
        fontsize=9,
    )
    axes.annotate(
        f"Grounding DINO zero-shot {zero_shot_map:.2f}",
        (sizes[-1], zero_shot_map),
        xytext=(0, 6 if last.mean_map < zero_shot_map else -14),
        textcoords="offset points",
        ha="right",
        color=TEXT_PRIMARY,
        fontsize=9,
    )

    style_axes(
        axes,
        "Test mAP vs. reviewed training images",
        "Training images",
        "Test mAP (COCO 0.50:0.95)",
    )
    if any(p.runs > 1 for p in points):
        axes.text(
            0,
            1.01,
            "Points: mean over repeated runs · vertical bars: min–max",
            transform=axes.transAxes,
            color=TEXT_SECONDARY,
            fontsize=8,
        )
    axes.set_ylim(0, 1)
    add_legend(axes)
    save(figure, output_path)


def write_curve_table(
    points: list[CurvePoint], zero_shot_map: float, output_path: Path
) -> None:
    """Same data as the plot, as a table for the report and screen-reader users."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file, [f.name for f in fields(CurvePoint)] + ["zero_shot_map"]
        )
        writer.writeheader()
        for point in points:
            writer.writerow({**asdict(point), "zero_shot_map": zero_shot_map})
