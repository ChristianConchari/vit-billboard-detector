"""End-to-end run, from a Label Studio export to a results report.

Steps: import export -> split by video -> fine-tune RT-DETR -> calibrate its
score threshold on val -> evaluate RT-DETR and Grounding DINO on test ->
benchmark latency -> render figures -> write reports/runs/<run>/.

Each run is a single MLflow run holding the configs, the code commit, dataset
fingerprints, per-epoch training metrics, test metrics, latency and reports.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mlflow
import torch
from PIL import Image

from vit.data.dataset_split import (
    SPLIT_NAMES,
    load_or_build_assignment,
    split_coco_by_video,
    split_fingerprint,
    write_splits,
)
from vit.data.label_studio_export import import_label_studio_export
from vit.eval.coco_evaluation import evaluate_detector
from vit.eval.figures import render_attention_maps, render_detector_comparison
from vit.eval.latency import measure_latency
from vit.eval.results import results_markdown
from vit.eval.threshold import CALIBRATION_FILE, calibrate_threshold
from vit.inference.factory import build_grounding_dino_detector, build_rtdetr_detector
from vit.train.trainer import new_run_name, train_rtdetr
from vit.utils.logging import get_logger
from vit.utils.tracking import file_digest, flatten, git_tags, prefixed, start_run

logger = get_logger(__name__, log_file="pipeline.log")


@dataclass
class PipelineConfigs:
    pipeline: dict[str, Any]
    rtdetr: dict[str, Any]
    grounding_dino: dict[str, Any]


def run_pipeline(
    configs: PipelineConfigs,
    export_path: Path | None = None,
    checkpoint: Path | None = None,
    render_figures: bool = True,
    note: str | None = None,
) -> Path:
    """Run every step and return the run directory holding summary.json and results.md.

    Without `export_path` the existing reviewed dataset is split again; with
    `checkpoint` training is skipped and that model is evaluated instead. `note`
    tags the MLflow run (e.g. "learning-curve") so related runs can be filtered.
    """
    image_dir = Path(configs.rtdetr["data"]["image_dir"])
    train = checkpoint is None
    run_name = new_run_name() if train else checkpoint.parent.name
    run_dir = Path(configs.pipeline["output_dir"]) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if export_path is not None:
        _step("Import Label Studio export")
        _import_export(configs, export_path, image_dir)

    _step("Split reviewed dataset by video")
    splits = _split(configs)

    mlflow_run_name = run_name if train else f"{run_name}-evaluation"
    with start_run(configs.rtdetr["mlflow"], mlflow_run_name):
        _log_run_context(
            configs,
            splits,
            export_path,
            stage="train+evaluate" if train else "evaluate",
            note=note,
        )

        if train:
            _step("Fine-tune RT-DETR")
            checkpoint = train_rtdetr(configs.rtdetr, run_name=run_name)
        mlflow.log_param("checkpoint", str(checkpoint))

        label_names = configs.rtdetr["model"]["label_names"]
        permissive_rtdetr = build_rtdetr_detector(
            checkpoint, label_names, configs.rtdetr["evaluation"]["score_threshold"]
        )

        _step("Calibrate RT-DETR score threshold on val")
        iou_threshold = configs.pipeline["calibration"]["iou_threshold"]
        calibration = calibrate_threshold(
            permissive_rtdetr, splits["val"], image_dir, iou_threshold
        )
        calibration.save(checkpoint)
        mlflow.log_metrics(
            prefixed(
                {k: v for k, v in asdict(calibration).items() if k != "iou_threshold"},
                "calibration",
            )
        )
        logger.info(
            "Threshold %.3f (F1 %.3f on val)",
            calibration.score_threshold,
            calibration.f1,
        )

        _step("Evaluate on test")
        grounding_dino = build_grounding_dino_detector(
            configs.grounding_dino,
            configs.grounding_dino["evaluation"]["box_threshold"],
        )
        detectors = {"rtdetr": permissive_rtdetr, "grounding-dino": grounding_dino}
        test_metrics = {
            name: evaluate_detector(detector, splits["test"], image_dir)
            for name, detector in detectors.items()
        }
        for name, metrics in test_metrics.items():
            mlflow.log_metrics(prefixed(metrics, f"test/{name}"))

        _step("Benchmark inference latency")
        latency = _benchmark(configs, detectors, splits["test"], image_dir)
        for name, stats in latency["models"].items():
            mlflow.log_metrics(prefixed(stats, f"latency/{name}"))

        if render_figures:
            _step("Render figures")
            _render_figures(
                configs,
                checkpoint,
                splits["test"],
                run_dir,
                calibration.score_threshold,
            )

        summary = {
            "run": run_name,
            "checkpoint": str(checkpoint),
            "dataset": _dataset_counts(splits),
            "test_metrics": test_metrics,
            "calibration": asdict(calibration),
            "latency": latency,
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        (run_dir / "results.md").write_text(results_markdown(summary))
        mlflow.log_artifact(str(run_dir / "summary.json"), artifact_path="reports")
        mlflow.log_artifact(str(run_dir / "results.md"), artifact_path="reports")
        mlflow.log_artifact(str(checkpoint / CALIBRATION_FILE), artifact_path="reports")
        if configs.pipeline["tracking"]["log_checkpoint"]:
            mlflow.log_artifacts(str(checkpoint), artifact_path="checkpoint")

    logger.info("Run complete -> %s", run_dir)
    return run_dir


def _import_export(
    configs: PipelineConfigs, export_path: Path, image_dir: Path
) -> None:
    data_cfg = configs.pipeline["data"]
    coco, missing = import_label_studio_export(
        export_path,
        output_path=Path(data_cfg["reviewed_annotations"]),
        images_out_dir=image_dir,
        extract_dir=Path(data_cfg["label_studio_extract_dir"]),
        fallback_image_dirs=[Path(data_cfg["raw_image_dir"])],
    )
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} exported image(s) not found, e.g. {missing[:3]}"
        )
    logger.info(
        "%d reviewed image(s), %d box(es)",
        len(coco["images"]),
        len(coco["annotations"]),
    )


def _split(configs: PipelineConfigs) -> dict[str, dict[str, Any]]:
    data_cfg = configs.pipeline["data"]
    ratios = data_cfg["split_ratios"]
    assignment = load_or_build_assignment(
        Path(data_cfg["split_assignment"]),
        Path(data_cfg["raw_image_dir"]),
        ratios["train"],
        ratios["val"],
        ratios["test"],
        data_cfg["split_seed"],
    )
    reviewed = json.loads(Path(data_cfg["reviewed_annotations"]).read_text())
    splits = split_coco_by_video(reviewed, assignment)

    empty = [name for name, split in splits.items() if not split["images"]]
    if empty:
        raise ValueError(
            f"Splits without reviewed images: {empty}. Review images from their videos."
        )

    write_splits(
        splits,
        {
            name: Path(configs.rtdetr["data"][f"{name}_annotations"])
            for name in SPLIT_NAMES
        },
    )
    for name, counts in _dataset_counts(splits).items():
        logger.info(
            "%s: %d image(s), %d box(es)", name, counts["images"], counts["boxes"]
        )
    return splits


def _log_run_context(
    configs: PipelineConfigs,
    splits: dict[str, dict[str, Any]],
    export_path: Path | None,
    stage: str,
    note: str | None,
) -> None:
    """Record what is needed to reproduce the run: code, configs and dataset version."""
    data_cfg = configs.pipeline["data"]
    tags = {
        **git_tags(),
        "stage": stage,
        "data.reviewed_annotations_sha256": file_digest(
            Path(data_cfg["reviewed_annotations"])
        ),
        "data.split_assignment_sha256": file_digest(Path(data_cfg["split_assignment"])),
        **{
            f"data.{name}_fingerprint": split_fingerprint(split)
            for name, split in splits.items()
        },
    }
    if export_path is not None:
        tags["data.export_sha256"] = file_digest(export_path)
    if note:
        tags.update({"note": note, "mlflow.note.content": note})
    mlflow.set_tags(tags)

    for name, config in asdict(configs).items():
        mlflow.log_params(flatten(config, prefix=f"{name}."))
        mlflow.log_dict(config, f"configs/{name}.yaml")
    mlflow.log_params(
        {
            f"data.{split}.{key}": value
            for split, counts in _dataset_counts(splits).items()
            for key, value in counts.items()
        }
    )


def _benchmark(
    configs: PipelineConfigs,
    detectors: dict[str, Any],
    test_split: dict[str, Any],
    image_dir: Path,
) -> dict[str, Any]:
    latency_cfg = configs.pipeline["latency"]
    images = [
        Image.open(image_dir / info["file_name"]).convert("RGB")
        for info in test_split["images"]
    ]
    return {
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "models": {
            name: measure_latency(
                detector, images, latency_cfg["warmup"], latency_cfg["repeats"]
            ).to_dict()
            for name, detector in detectors.items()
        },
    }


def _render_figures(
    configs: PipelineConfigs,
    checkpoint: Path,
    test_split: dict[str, Any],
    run_dir: Path,
    rtdetr_threshold: float,
) -> None:
    figures_cfg = configs.pipeline["figures"]
    image_dir = Path(configs.rtdetr["data"]["image_dir"])
    gdino_threshold = configs.grounding_dino["thresholds"]["box_threshold"]
    rtdetr = build_rtdetr_detector(
        checkpoint, configs.rtdetr["model"]["label_names"], rtdetr_threshold
    )
    grounding_dino = build_grounding_dino_detector(
        configs.grounding_dino, gdino_threshold
    )
    detectors = {
        f"RT-DETR fine-tuned (score >= {rtdetr_threshold:.2f})": rtdetr,
        f"Grounding DINO zero-shot (score >= {gdino_threshold:.2f})": grounding_dino,
    }
    render_detector_comparison(
        detectors,
        test_split,
        image_dir,
        run_dir / "figures" / "predictions",
        figures_cfg["overlay_fraction"],
    )
    render_attention_maps(
        checkpoint,
        test_split,
        image_dir,
        run_dir / "figures" / "attention",
        rtdetr_threshold,
        figures_cfg["max_attention_detections"],
        figures_cfg["overlay_fraction"],
    )


def _dataset_counts(splits: dict[str, dict[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        name: {"images": len(split["images"]), "boxes": len(split["annotations"])}
        for name, split in splits.items()
    }


def _step(title: str) -> None:
    logger.info("==== %s ====", title)
