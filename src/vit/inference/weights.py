"""Download the released RT-DETR checkpoint and verify it before use."""

import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_checkpoint(url: str, expected_sha256: str, checkpoint_dir: Path) -> Path:
    """Fetch the zipped checkpoint, verify its SHA-256 and unpack it.

    The archive holds one top-level folder with the model files; its contents end
    up directly in checkpoint_dir. A checksum mismatch aborts before unpacking.
    """
    checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
    archive = checkpoint_dir.parent / f"{checkpoint_dir.name}.zip"
    urllib.request.urlretrieve(url, archive)

    actual_sha256 = sha256_of(archive)
    if actual_sha256 != expected_sha256:
        archive.unlink()
        raise ValueError(
            f"Checksum mismatch for {url}: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )

    extract_dir = checkpoint_dir.parent / f".{checkpoint_dir.name}-extract"
    shutil.rmtree(extract_dir, ignore_errors=True)
    with zipfile.ZipFile(archive) as zip_file:
        zip_file.extractall(extract_dir)
    (top_level,) = extract_dir.iterdir()
    shutil.rmtree(checkpoint_dir, ignore_errors=True)
    top_level.rename(checkpoint_dir)
    extract_dir.rmdir()
    archive.unlink()
    return checkpoint_dir
