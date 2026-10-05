"""Export the MLflow experiment record so it can be versioned and read without MLflow.

Only finished runs evaluated on one frozen test split (matched by content
fingerprint) are included, so every number in the export is comparable.
"""

import csv
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from mlflow.tracking import MlflowClient

from vit.eval.chart_style import (
    RTDETR_COLOR,
    UNFROZEN_COLOR,
    add_legend,
    new_figure,
    save,
    style_axes,
)

TEST_METRICS = ("mAP", "AP50", "AP75")
EPOCH_METRICS = ("train/loss", "val/mAP", "val/AP50", "val/AP75")


@dataclass
class RunRecord:
    run_name: str
    note: str
    freeze_backbone: bool
    seed: int
    train_images: int
    test_mAP: float
    test_AP50: float
    test_AP75: float
    zero_shot_mAP: float
    zero_shot_AP50: float
    val_best_mAP: float
    best_epoch: int
    score_threshold: float
    latency_ms: float
    git_commit: str
    git_dirty: str
    train_fingerprint: str
    test_fingerprint: str
    epoch_history: dict[str, list[float]] = field(default_factory=dict, repr=False)

    @property
    def val_map_by_epoch(self) -> list[float]:
        return self.epoch_history.get("val/mAP", [])


@dataclass
class ConfigSummary:
    freeze_backbone: bool
    train_images: int
    runs: int
    stats: dict[str, tuple[float, float, float]]  # metric -> (mean, min, max)


def fetch_run_records(
    mlflow_config: dict[str, Any], test_fingerprint: str
) -> list[RunRecord]:
    client = MlflowClient(tracking_uri=mlflow_config["tracking_uri"])
    experiment = client.get_experiment_by_name(mlflow_config["experiment_name"])
    if experiment is None:
        return []
    conditions = [
        "attributes.status = 'FINISHED'",
        "tags.stage = 'train+evaluate'",
        f"tags.`data.test_fingerprint` = '{test_fingerprint}'",
    ]
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=" and ".join(conditions),
        order_by=["attributes.start_time ASC"],
        max_results=1000,
    )
    return [_record(client, run) for run in runs]


def _record(client: MlflowClient, run) -> RunRecord:
    tags, params, metrics = run.data.tags, run.data.params, run.data.metrics
    epoch_history = {
        key: [
            m.value
            for m in sorted(
                client.get_metric_history(run.info.run_id, key), key=lambda m: m.step
            )
        ]
        for key in EPOCH_METRICS
    }
    return RunRecord(
        run_name=run.info.run_name,
        note=tags.get("note", ""),
        freeze_backbone=params["rtdetr.training.freeze_backbone"] == "True",
        seed=int(params["rtdetr.training.seed"]),
        train_images=int(params["data.train.images"]),
        test_mAP=metrics["test/rtdetr/mAP"],
        test_AP50=metrics["test/rtdetr/AP50"],
        test_AP75=metrics["test/rtdetr/AP75"],
        zero_shot_mAP=metrics["test/grounding-dino/mAP"],
        zero_shot_AP50=metrics["test/grounding-dino/AP50"],
        val_best_mAP=metrics["val/best_mAP"],
        best_epoch=int(metrics["val/best_epoch"]),
        score_threshold=metrics["calibration/score_threshold"],
        latency_ms=metrics["latency/rtdetr/mean_ms"],
        git_commit=tags["git.commit"][:7],
        git_dirty=tags["git.dirty"],
        train_fingerprint=tags.get("data.train_fingerprint", ""),
        test_fingerprint=tags["data.test_fingerprint"],
        epoch_history=epoch_history,
    )


def write_runs_csv(records: list[RunRecord], output_path: Path) -> None:
    columns = [f.name for f in fields(RunRecord) if f.name != "epoch_history"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, columns, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    k: round(v, 4) if isinstance(v, float) else v
                    for k, v in asdict(record).items()
                }
            )


def write_epochs_csv(records: list[RunRecord], output_path: Path) -> None:
    """One row per run and epoch, so training curves can be redrawn without MLflow."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["run_name", "epoch", *EPOCH_METRICS])
        for record in records:
            epochs = max(map(len, record.epoch_history.values()), default=0)
            for epoch in range(epochs):
                values = [
                    round(history[epoch], 4) if epoch < len(history) else ""
                    for history in map(record.epoch_history.get, EPOCH_METRICS)
                ]
                writer.writerow([record.run_name, epoch + 1, *values])


def summarize_by_config(records: list[RunRecord]) -> list[ConfigSummary]:
    """Group runs by (backbone frozen, training-set size); seeds share one row."""
    groups: dict[tuple[bool, int], list[RunRecord]] = defaultdict(list)
    for record in records:
        groups[(record.freeze_backbone, record.train_images)].append(record)
    return [
        ConfigSummary(
            freeze_backbone=frozen,
            train_images=train_images,
            runs=len(group),
            stats={
                metric: _mean_and_range([getattr(r, f"test_{metric}") for r in group])
                for metric in TEST_METRICS
            },
        )
        for (frozen, train_images), group in sorted(
            groups.items(), key=lambda item: (item[0][1], not item[0][0])
        )
    ]


def _mean_and_range(values: list[float]) -> tuple[float, float, float]:
    return statistics.fmean(values), min(values), max(values)


def summary_markdown(records: list[RunRecord]) -> str:
    curve = sorted(
        (r for r in records if r.note == "learning-curve"), key=lambda r: r.train_images
    )
    final_size = max(r.train_images for r in records)
    ablation = [s for s in summarize_by_config(records) if s.train_images == final_size]
    lines = [
        "# Experiment record",
        "",
        f"{len(records)} tracked runs evaluated on the frozen test split "
        f"(fingerprint `{records[0].test_fingerprint}`). "
        "Full per-run data: `runs.csv`.",
        "",
        "## Learning curve (backbone frozen, seed 42)",
        "",
        "| Train images | RT-DETR mAP | AP50 | AP75 | Grounding DINO mAP | Run |",
        "|--:|--:|--:|--:|--:|---|",
        *(
            f"| {r.train_images} | {r.test_mAP:.3f} | {r.test_AP50:.3f} "
            f"| {r.test_AP75:.3f} "
            f"| {r.zero_shot_mAP:.3f} | `{r.run_name}` |"
            for r in curve
        ),
        "",
        f"## Backbone ablation ({final_size} training images, "
        "mean and range over seeds)",
        "",
        "| Backbone | Runs | mAP | AP50 | AP75 |",
        "|---|--:|--:|--:|--:|",
        *(
            f"| {'frozen' if s.freeze_backbone else 'unfrozen'} | {s.runs} | "
            + " | ".join(
                f"{mean:.3f} ({low:.3f}–{high:.3f})"
                for mean, low, high in s.stats.values()
            )
            + " |"
            for s in ablation
        ),
        "",
    ]
    return "\n".join(lines)


def plot_training_curves(records: list[RunRecord], output_path: Path) -> None:
    """Validation mAP per epoch for every run at the largest training-set size."""
    final_size = max(r.train_images for r in records)
    figure, axes = new_figure()

    plotted = [r for r in records if r.train_images == final_size]
    for frozen, color, label in (
        (True, RTDETR_COLOR, "frozen"),
        (False, UNFROZEN_COLOR, "unfrozen"),
    ):
        runs = [r for r in plotted if r.freeze_backbone == frozen]
        if not runs:
            continue
        best = max(max(r.val_map_by_epoch) for r in runs)
        seeds = f"{len(runs)} {'seed' if len(runs) == 1 else 'seeds'}"
        legend_label = f"Backbone {label} · {seeds} · best {best:.2f}"
        for index, run in enumerate(runs):
            axes.plot(
                range(1, len(run.val_map_by_epoch) + 1),
                run.val_map_by_epoch,
                color=color,
                linewidth=2,
                alpha=0.8,
                solid_capstyle="round",
                label=legend_label if index == 0 else None,
            )

    values = [v for r in plotted for v in r.val_map_by_epoch]
    axes.set_ylim(max(0.0, min(values) - 0.05), min(1.0, max(values) + 0.05))
    style_axes(
        axes,
        f"Validation mAP per epoch ({final_size} training images)",
        "Epoch",
        "Val mAP (COCO 0.50:0.95)",
    )
    add_legend(axes)
    save(figure, output_path)
