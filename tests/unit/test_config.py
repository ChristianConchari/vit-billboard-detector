import pytest

from vit.utils.config import apply_overrides, load_config


def test_load_config_reads_yaml(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model:\n  name: test-model\n")

    config = load_config(config_path)

    assert config == {"model": {"name": "test-model"}}


def test_load_config_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config("does/not/exist.yaml")


def test_overrides_are_parsed_as_yaml_values():
    config = {"training": {"freeze_backbone": True, "learning_rate": 1e-4, "seed": 42}}

    apply_overrides(
        config, ["training.freeze_backbone=false", "training.learning_rate=5e-5", "training.seed=1"]
    )

    assert config == {"training": {"freeze_backbone": False, "learning_rate": 5e-5, "seed": 1}}


def test_unknown_keys_are_rejected():
    with pytest.raises(KeyError):
        apply_overrides({"training": {"epochs": 50}}, ["training.epoch=3"])


def test_overrides_need_an_equals_sign():
    with pytest.raises(ValueError):
        apply_overrides({"training": {"epochs": 50}}, ["training.epochs"])
