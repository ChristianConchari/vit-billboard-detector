import pytest

from vit.eval.threshold import (
    ThresholdCalibration,
    best_f1_threshold,
    load_calibrated_threshold,
    match_proposals,
)
from vit.inference.detection import BoxProposal


def test_each_ground_truth_box_matches_at_most_one_proposal():
    proposals = [
        BoxProposal("billboard", 0.6, (0, 0, 10, 10)),
        BoxProposal("billboard", 0.9, (1, 1, 10, 10)),
        BoxProposal("billboard", 0.3, (50, 50, 60, 60)),
    ]

    matches = match_proposals(proposals, [(0, 0, 10, 10)], iou_threshold=0.5)

    assert matches == [(0.9, True), (0.6, False), (0.3, False)]


def test_proposals_without_ground_truth_are_false_positives():
    assert match_proposals([BoxProposal("billboard", 0.5, (0, 0, 1, 1))], [], 0.5) == [(0.5, False)]


def test_best_f1_threshold_cuts_below_the_last_useful_true_positive():
    scored = [(0.9, True), (0.8, True), (0.4, False), (0.3, False), (0.2, True)]

    calibration = best_f1_threshold(scored, total_ground_truth=3, iou_threshold=0.5)

    assert calibration.score_threshold == 0.8
    assert calibration.precision == 1.0
    assert calibration.recall == pytest.approx(2 / 3)
    assert calibration.f1 == pytest.approx(0.8)


def test_calibration_requires_ground_truth():
    with pytest.raises(ValueError):
        best_f1_threshold([(0.9, False)], total_ground_truth=0, iou_threshold=0.5)


def test_calibration_round_trips_through_the_checkpoint_dir(tmp_path):
    assert load_calibrated_threshold(tmp_path) is None

    ThresholdCalibration(0.27, 0.8, 0.6, 0.69, 0.5).save(tmp_path)

    assert load_calibrated_threshold(tmp_path) == 0.27
