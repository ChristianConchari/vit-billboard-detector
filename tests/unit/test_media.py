import cv2
import numpy as np
import pytest
from PIL import Image

from vit.inference.detection import BoxProposal
from vit.inference.media import (
    UnreadableVideoError,
    detect_in_images,
    detect_in_video,
    is_video,
    list_images,
)


class FixedDetector:
    def __init__(self):
        self.seen_sizes = []

    def predict(self, image):
        self.seen_sizes.append(image.size)
        return [BoxProposal("billboard", 0.87654, (1.23, 2.0, 30.0, 20.0))]


@pytest.fixture
def video_path(tmp_path):
    path = tmp_path / "drive.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for _ in range(5):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()
    return path


def test_media_type_detection_and_image_listing(tmp_path):
    for name in ("b.jpg", "a.png", "notes.txt"):
        (tmp_path / name).touch()

    assert is_video(tmp_path / "x.MP4") and not is_video(tmp_path / "x.jpg")
    assert [p.name for p in list_images(tmp_path)] == ["a.png", "b.jpg"]
    assert list_images(tmp_path / "b.jpg") == [tmp_path / "b.jpg"]
    assert list_images(tmp_path / "notes.txt") == []


def test_detect_in_images_saves_annotated_copies_and_records(tmp_path):
    Image.new("RGB", (64, 50)).save(tmp_path / "a_1.jpg")
    detector = FixedDetector()

    records = detect_in_images(
        detector, [tmp_path / "a_1.jpg"], tmp_path / "out", overlay_fraction=0.1
    )

    assert detector.seen_sizes == [(64, 45)]
    assert Image.open(tmp_path / "out" / "a_1.jpg").size == (64, 45)
    assert records == [
        {
            "source": "a_1.jpg",
            "frame": None,
            "detections": [{"box_xyxy": [1.2, 2.0, 30.0, 20.0], "score": 0.8765}],
        }
    ]


def test_detect_in_video_writes_annotated_video_and_one_record_per_frame(
    tmp_path, video_path
):
    records = detect_in_video(FixedDetector(), video_path, tmp_path / "out")

    assert [r["frame"] for r in records] == [0, 1, 2, 3, 4]
    output = cv2.VideoCapture(str(tmp_path / "out" / "drive_detections.mp4"))
    assert int(output.get(cv2.CAP_PROP_FRAME_COUNT)) == 5
    output.release()


def test_detect_in_video_rejects_unreadable_files(tmp_path):
    broken = tmp_path / "broken.mp4"
    broken.write_text("not a video")

    with pytest.raises(UnreadableVideoError):
        detect_in_video(FixedDetector(), broken, tmp_path / "out")
