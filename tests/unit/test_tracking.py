import mlflow
import pytest

from vit.utils.tracking import file_digest, flatten, git_tags, prefixed, require_active_run


def test_flatten_uses_dotted_keys_and_prefix():
    config = {"training": {"epochs": 50, "lr": 1e-4}, "labels": ["billboard"]}

    assert flatten(config, prefix="rtdetr.") == {
        "rtdetr.training.epochs": 50,
        "rtdetr.training.lr": 1e-4,
        "rtdetr.labels": ["billboard"],
    }


def test_prefixed_namespaces_metric_names():
    assert prefixed({"mAP": 0.5}, "test/rtdetr") == {"test/rtdetr/mAP": 0.5}


def test_file_digest_changes_with_content(tmp_path):
    path = tmp_path / "a.json"
    path.write_text("{}")
    first = file_digest(path)
    path.write_text('{"x": 1}')

    assert len(first) == 16 and file_digest(path) != first


def test_git_tags_report_the_current_commit():
    tags = git_tags()

    assert set(tags) == {"git.commit", "git.dirty"}
    assert tags["git.dirty"] in {"true", "false", "unknown"}


def test_logging_outside_a_run_is_rejected():
    assert mlflow.active_run() is None
    with pytest.raises(RuntimeError):
        require_active_run()
