"""End-to-end pipeline run: export, split, train, calibrate, evaluate, report."""

import json

import pytest
from mlflow.tracking import MlflowClient

from vit import pipeline as pipeline_module
from vit.inference.detection import BoxProposal
from vit.pipeline import PipelineConfigs, run_pipeline


class FixedZeroShotDetector:
    """Stands in for Grounding DINO, whose weights are too large to download in CI."""

    def predict(self, image):
        return [BoxProposal("billboard", 0.8, (60.0, 40.0, 180.0, 120.0))]


@pytest.fixture
def configs(workspace, monkeypatch) -> PipelineConfigs:
    monkeypatch.setattr(
        pipeline_module,
        "build_grounding_dino_detector",
        lambda *_: FixedZeroShotDetector(),
    )
    return PipelineConfigs(
        workspace["pipeline"], workspace["rtdetr"], workspace["grounding_dino"]
    )


def _runs(configs: PipelineConfigs):
    client = MlflowClient(tracking_uri=configs.rtdetr["mlflow"]["tracking_uri"])
    experiment = client.get_experiment_by_name(
        configs.rtdetr["mlflow"]["experiment_name"]
    )
    return client, client.search_runs([experiment.experiment_id])


def test_pipeline_trains_evaluates_and_reports(configs, workspace):
    run_dir = run_pipeline(
        configs, export_path=workspace["export_path"], note="integration"
    )

    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["dataset"] == {
        split: {"images": 3, "boxes": 3} for split in ("train", "val", "test")
    }
    assert set(summary["test_metrics"]) == {"rtdetr", "grounding-dino"}
    assert summary["test_metrics"]["grounding-dino"]["mAP"] == pytest.approx(1.0)
    assert set(summary["latency"]["models"]) == {"rtdetr", "grounding-dino"}
    assert "## Test metrics (COCO)" in (run_dir / "results.md").read_text()
    assert any((run_dir / "figures" / "predictions").iterdir())

    checkpoint = workspace["tmp_path"] / "checkpoints" / run_dir.name / "best"
    assert (checkpoint / "calibration.json").is_file()

    client, runs = _runs(configs)
    assert len(runs) == 1
    run = runs[0]
    assert run.data.tags["stage"] == "train+evaluate"
    assert run.data.tags["note"] == "integration"
    assert "data.test_fingerprint" in run.data.tags
    assert {
        "train/loss",
        "val/mAP",
        "test/rtdetr/mAP",
        "latency/rtdetr/mean_ms",
    } <= set(run.data.metrics)
    artifacts = {a.path for a in client.list_artifacts(run.info.run_id, "reports")}
    assert artifacts == {
        "reports/results.md",
        "reports/summary.json",
        "reports/calibration.json",
    }


def test_pipeline_evaluates_an_existing_checkpoint_without_training(configs, workspace):
    first_run_dir = run_pipeline(
        configs, export_path=workspace["export_path"], render_figures=False
    )
    checkpoint = workspace["tmp_path"] / "checkpoints" / first_run_dir.name / "best"

    run_pipeline(configs, checkpoint=checkpoint, render_figures=False)

    _, runs = _runs(configs)
    names_by_stage = {run.data.tags["stage"]: run.info.run_name for run in runs}
    assert names_by_stage == {
        "train+evaluate": first_run_dir.name,
        "evaluate": f"{first_run_dir.name}-evaluation",
    }


def test_pipeline_refuses_splits_without_reviewed_images(configs, workspace):
    export = json.loads(workspace["export_path"].read_text())
    test_ids = {
        img["id"] for img in export["images"] if "testvideo" in img["file_name"]
    }
    export["images"] = [img for img in export["images"] if img["id"] not in test_ids]
    export["annotations"] = [
        a for a in export["annotations"] if a["image_id"] not in test_ids
    ]
    workspace["export_path"].write_text(json.dumps(export))

    with pytest.raises(ValueError, match="test"):
        run_pipeline(configs, export_path=workspace["export_path"])
