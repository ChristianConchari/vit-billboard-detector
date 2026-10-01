import time

import pytest
from PIL import Image

from vit.eval.latency import measure_latency


class SleepingDetector:
    def __init__(self, seconds: float):
        self.seconds = seconds
        self.calls = 0

    def predict(self, image):
        self.calls += 1
        time.sleep(self.seconds)
        return []


def test_measure_latency_times_every_image_after_warmup():
    detector = SleepingDetector(0.01)
    images = [Image.new("RGB", (8, 8))] * 4

    stats = measure_latency(detector, images, warmup=2, repeats=2)

    assert detector.calls == 2 + 4 * 2
    assert stats.images == 8
    assert stats.mean_ms == pytest.approx(10, rel=0.5)
    assert stats.fps == pytest.approx(1000 / stats.mean_ms)
    assert stats.median_ms <= stats.p95_ms
