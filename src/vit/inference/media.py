"""Run a Detector over image files, image folders and videos, saving annotated copies."""
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from vit.eval.visualization import crop_bottom, draw_predictions
from vit.inference.detection import BoxProposal, Detector

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

DetectionRecord = dict[str, Any]


def is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS


def list_images(path: Path) -> list[Path]:
    candidates = [path] if path.is_file() else sorted(path.iterdir())
    return [p for p in candidates if p.suffix.lower() in IMAGE_EXTENSIONS]


def detect_in_images(
    detector: Detector, image_paths: list[Path], output_dir: Path, overlay_fraction: float = 0.0
) -> list[DetectionRecord]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for path in image_paths:
        image = _without_overlay(Image.open(path).convert("RGB"), overlay_fraction)
        proposals = detector.predict(image)
        draw_predictions(image, proposals).save(output_dir / path.name)
        records.append(_record(path.name, None, proposals))
    return records


def detect_in_video(
    detector: Detector, video_path: Path, output_dir: Path, overlay_fraction: float = 0.0
) -> list[DetectionRecord]:
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0

    writer = None
    records = []
    try:
        for frame_index, frame in enumerate(_read_frames(capture)):
            image = _without_overlay(frame, overlay_fraction)
            proposals = detector.predict(image)
            if writer is None:
                writer = cv2.VideoWriter(
                    str(output_dir / f"{video_path.stem}_detections.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    image.size,
                )
            writer.write(cv2.cvtColor(np.asarray(draw_predictions(image, proposals)), cv2.COLOR_RGB2BGR))
            records.append(_record(video_path.name, frame_index, proposals))
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    return records


def _read_frames(capture: cv2.VideoCapture) -> Iterator[Image.Image]:
    while True:
        ok, frame = capture.read()
        if not ok:
            return
        yield Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def _without_overlay(image: Image.Image, overlay_fraction: float) -> Image.Image:
    return crop_bottom(image, round(image.height * overlay_fraction))


def _record(source: str, frame: int | None, proposals: list[BoxProposal]) -> DetectionRecord:
    return {
        "source": source,
        "frame": frame,
        "detections": [
            {"box_xyxy": [round(c, 1) for c in p.box_xyxy], "score": round(p.score, 4)}
            for p in proposals
        ],
    }
