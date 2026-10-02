from interfaces.cli.detect import resolve_score_threshold
from vit.eval.threshold import ThresholdCalibration

CONFIG = {"inference": {"score_threshold": 0.15}}


def test_explicit_threshold_wins(tmp_path):
    ThresholdCalibration(0.3, 1, 1, 1, 0.5).save(tmp_path)

    assert resolve_score_threshold(0.0, tmp_path, CONFIG) == 0.0


def test_calibrated_threshold_beats_config_default(tmp_path):
    ThresholdCalibration(0.3, 1, 1, 1, 0.5).save(tmp_path)

    assert resolve_score_threshold(None, tmp_path, CONFIG) == 0.3


def test_config_default_without_calibration(tmp_path):
    assert resolve_score_threshold(None, tmp_path, CONFIG) == 0.15
