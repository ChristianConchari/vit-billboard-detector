import csv

import mlflow
import pytest

from vit.eval.experiments import (
    RunRecord,
    fetch_run_records,
    plot_training_curves,
    summarize_by_config,
    summary_markdown,
    write_runs_csv,
)


def _record(name, frozen, seed, train_images, test_map, note="ablation"):
    return RunRecord(
        run_name=name,
        note=note,
        freeze_backbone=frozen,
        seed=seed,
        train_images=train_images,
        test_mAP=test_map,
        test_AP50=test_map + 0.3,
        test_AP75=test_map + 0.1,
        zero_shot_mAP=0.48,
        zero_shot_AP50=0.72,
        val_best_mAP=0.76,
        best_epoch=7,
        score_threshold=0.13,
        latency_ms=19.3,
        git_commit="abc1234",
        git_dirty="false",
        train_fingerprint="train-fp",
        test_fingerprint="test-fp",
        val_map_by_epoch=[0.5, 0.7, 0.76],
    )


RECORDS = [
    _record("lc-66", True, 42, 66, 0.546, note="learning-curve"),
    _record("lc-441", True, 42, 441, 0.637, note="learning-curve"),
    _record("frozen-1", True, 1, 441, 0.640),
    _record("unfrozen-42", False, 42, 441, 0.645),
    _record("unfrozen-1", False, 1, 441, 0.655),
]


def test_seeds_of_one_config_aggregate_into_mean_and_range():
    summaries = {(s.freeze_backbone, s.train_images): s for s in summarize_by_config(RECORDS)}

    frozen = summaries[(True, 441)]
    assert frozen.runs == 2
    assert frozen.stats["mAP"] == pytest.approx((0.6385, 0.637, 0.640))
    assert summaries[(False, 441)].stats["mAP"] == pytest.approx((0.650, 0.645, 0.655))


def test_summary_has_learning_curve_and_ablation_tables():
    markdown = summary_markdown(RECORDS)

    assert "| 66 | 0.546 |" in markdown and "| 441 | 0.637 |" in markdown
    assert "| frozen | 2 | 0.639 (0.637–0.640) |" in markdown
    assert "| unfrozen | 2 | 0.650 (0.645–0.655) |" in markdown


def test_csv_and_training_curves_are_written(tmp_path):
    write_runs_csv(RECORDS, tmp_path / "runs.csv")
    plot_training_curves(RECORDS, tmp_path / "curves.png")

    rows = list(csv.DictReader((tmp_path / "runs.csv").open()))
    assert [row["run_name"] for row in rows] == [r.run_name for r in RECORDS]
    assert "val_map_by_epoch" not in rows[0]
    assert (tmp_path / "curves.png").stat().st_size > 0


def test_fetch_keeps_only_finished_runs_on_the_given_test_split(tmp_path):
    config = {"tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}", "experiment_name": "exp"}
    mlflow.set_tracking_uri(config["tracking_uri"])
    mlflow.set_experiment(config["experiment_name"])

    def log_run(name, test_fingerprint, finished=True):
        mlflow.start_run(run_name=name)
        mlflow.set_tags(
            {
                "stage": "train+evaluate",
                "data.test_fingerprint": test_fingerprint,
                "git.commit": "abc1234def",
                "git.dirty": "false",
            }
        )
        mlflow.log_params(
            {
                "rtdetr.training.freeze_backbone": True,
                "rtdetr.training.seed": 42,
                "data.train.images": 66,
            }
        )
        for step, value in enumerate([0.4, 0.6], start=1):
            mlflow.log_metric("val/mAP", value, step=step)
        metric_names = (
            "test/rtdetr/mAP test/rtdetr/AP50 test/rtdetr/AP75 test/grounding-dino/mAP "
            "test/grounding-dino/AP50 val/best_mAP val/best_epoch calibration/score_threshold "
            "latency/rtdetr/mean_ms"
        ).split()
        mlflow.log_metrics(dict.fromkeys(metric_names, 0.5))
        mlflow.end_run(status="FINISHED" if finished else "KILLED")

    log_run("kept", "test-fp")
    log_run("other-test-split", "old-fp")
    log_run("aborted", "test-fp", finished=False)

    records = fetch_run_records(config, "test-fp")

    assert [r.run_name for r in records] == ["kept"]
    assert records[0].val_map_by_epoch == [0.4, 0.6]
    assert records[0].freeze_backbone is True
