import csv

import pytest

from vit.eval.learning_curve import (
    RunResult,
    build_curve,
    plot_learning_curve,
    write_curve_table,
)


def _run(train_images: int, rtdetr_map: float, test_sha: str = "abc") -> RunResult:
    return RunResult(
        f"run-{train_images}", train_images, test_sha, rtdetr_map, zero_shot_map=0.68
    )


def test_runs_with_the_same_training_size_are_aggregated():
    points, zero_shot = build_curve([_run(100, 0.4), _run(34, 0.2), _run(100, 0.6)])

    assert [p.train_images for p in points] == [34, 100]
    assert points[1].mean_map == pytest.approx(0.5)
    assert (points[1].min_map, points[1].max_map, points[1].runs) == (0.4, 0.6, 2)
    assert zero_shot == pytest.approx(0.68)


def test_runs_on_different_test_splits_are_refused():
    with pytest.raises(ValueError, match="different test splits"):
        build_curve([_run(34, 0.2, "abc"), _run(100, 0.5, "xyz")])


def test_no_runs_is_an_error():
    with pytest.raises(ValueError):
        build_curve([])


def test_plot_and_table_are_written(tmp_path):
    points, zero_shot = build_curve([_run(34, 0.2), _run(100, 0.5)])

    plot_learning_curve(points, zero_shot, tmp_path / "curve.png")
    write_curve_table(points, zero_shot, tmp_path / "curve.csv")

    assert (tmp_path / "curve.png").stat().st_size > 0
    rows = list(csv.DictReader((tmp_path / "curve.csv").open()))
    assert [row["train_images"] for row in rows] == ["34", "100"]
    assert rows[0]["zero_shot_map"] == "0.68"
