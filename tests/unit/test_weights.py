import zipfile

import pytest

from vit.inference.weights import download_checkpoint, sha256_of


@pytest.fixture
def archive(tmp_path):
    path = tmp_path / "release" / "model.zip"
    path.parent.mkdir()
    with zipfile.ZipFile(path, "w") as zip_file:
        zip_file.writestr("rtdetr-billboard/config.json", "{}")
        zip_file.writestr("rtdetr-billboard/model.safetensors", b"weights")
    return path


def test_download_unpacks_the_checkpoint_folder(tmp_path, archive):
    checkpoint_dir = tmp_path / "checkpoints" / "rtdetr-billboard"

    download_checkpoint(archive.as_uri(), sha256_of(archive), checkpoint_dir)

    assert sorted(p.name for p in checkpoint_dir.iterdir()) == [
        "config.json",
        "model.safetensors",
    ]
    assert sorted(p.name for p in checkpoint_dir.parent.iterdir()) == [
        "rtdetr-billboard"
    ]


def test_checksum_mismatch_aborts_without_unpacking(tmp_path, archive):
    checkpoint_dir = tmp_path / "checkpoints" / "rtdetr-billboard"

    with pytest.raises(ValueError, match="Checksum mismatch"):
        download_checkpoint(archive.as_uri(), "0" * 64, checkpoint_dir)

    assert not checkpoint_dir.exists()
    assert list(checkpoint_dir.parent.iterdir()) == []
