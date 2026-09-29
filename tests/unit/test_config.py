import pytest

from vit.utils.config import load_config


def test_load_config_reads_yaml(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model:\n  name: test-model\n")

    config = load_config(config_path)

    assert config == {"model": {"name": "test-model"}}


def test_load_config_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config("does/not/exist.yaml")
