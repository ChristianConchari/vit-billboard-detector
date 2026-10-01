"""End-to-end inference latency (preprocessing + forward + postprocessing) of a Detector."""
import statistics
import time
from dataclasses import asdict, dataclass

from PIL import Image

from vit.inference.detection import Detector


@dataclass
class LatencyStats:
    images: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    fps: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def measure_latency(
    detector: Detector, images: list[Image.Image], warmup: int = 5, repeats: int = 1
) -> LatencyStats:
    """Time `detector.predict` per image, one image at a time (batch size 1).

    Warmup runs are discarded so CUDA initialization and kernel autotuning
    don't inflate the numbers.
    """
    for image in images[:warmup]:
        detector.predict(image)

    timings_ms = []
    for _ in range(repeats):
        for image in images:
            start = time.perf_counter()
            detector.predict(image)
            timings_ms.append((time.perf_counter() - start) * 1000)

    mean_ms = statistics.fmean(timings_ms)
    return LatencyStats(
        images=len(timings_ms),
        mean_ms=mean_ms,
        median_ms=statistics.median(timings_ms),
        p95_ms=statistics.quantiles(timings_ms, n=20)[-1] if len(timings_ms) > 1 else mean_ms,
        fps=1000 / mean_ms,
    )
