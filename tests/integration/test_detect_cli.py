"""End-to-end run of the billboard-detect CLI on a video and on an image folder."""

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]


def _detect(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "interfaces.cli.detect", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def test_detect_annotates_a_video_and_writes_one_record_per_frame(
    tmp_path, tiny_checkpoint
):
    video = tmp_path / "drive.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 5, (320, 240))
    for _ in range(4):
        writer.write(np.full((240, 320, 3), 128, dtype=np.uint8))
    writer.release()

    result = _detect(
        str(video),
        "--checkpoint",
        str(tiny_checkpoint),
        "--output-dir",
        str(tmp_path / "out"),
    )

    assert result.returncode == 0, result.stderr
    records = json.loads((tmp_path / "out" / "detections.json").read_text())
    assert [r["frame"] for r in records] == [0, 1, 2, 3]
    assert (tmp_path / "out" / "drive_detections.mp4").is_file()


def test_detect_annotates_every_image_in_a_folder(tmp_path, tiny_checkpoint):
    images = tmp_path / "images"
    images.mkdir()
    for name in ("a.jpg", "b.png"):
        Image.new("RGB", (320, 240)).save(images / name)

    result = _detect(
        str(images),
        "--checkpoint",
        str(tiny_checkpoint),
        "--output-dir",
        str(tmp_path / "out"),
        "--score-threshold",
        "0.0",
    )

    assert result.returncode == 0, result.stderr
    records = json.loads((tmp_path / "out" / "detections.json").read_text())
    assert sorted(r["source"] for r in records) == ["a.jpg", "b.png"]
    assert all(r["frame"] is None for r in records)


def test_detect_fails_cleanly_on_a_missing_source(tmp_path, tiny_checkpoint):
    result = _detect(
        str(tmp_path / "missing.jpg"), "--checkpoint", str(tiny_checkpoint)
    )

    assert result.returncode != 0
    assert "Source not found" in result.stderr
    assert "Traceback" not in result.stderr
