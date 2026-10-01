"""Model-agnostic detection types shared by every detector in the project."""
from dataclasses import dataclass
from typing import Protocol

from PIL import Image


@dataclass
class BoxProposal:
    label: str
    score: float
    box_xyxy: tuple[float, float, float, float]  # (x_min, y_min, x_max, y_max), pixels


class Detector(Protocol):
    def predict(self, image: Image.Image) -> list[BoxProposal]: ...
