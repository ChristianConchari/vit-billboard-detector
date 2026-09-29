"""Zero-shot bounding box proposals for billboard images using Grounding DINO.

Used as the first stage of the labeling pipeline: generates candidate boxes
via text prompts so a human only has to review/correct them instead of
annotating from scratch.
"""
from dataclasses import dataclass

from PIL import Image
from transformers import pipeline

from vit.utils.logging import get_logger

logger = get_logger(__name__, log_file="labeling.log")


@dataclass
class BoxProposal:
    label: str
    score: float
    box_xyxy: tuple[float, float, float, float]  # (x_min, y_min, x_max, y_max)


class GroundingDinoAutoLabeler:
    """Wraps a Hugging Face zero-shot-object-detection pipeline for Grounding DINO."""

    def __init__(
        self,
        checkpoint: str = "IDEA-Research/grounding-dino-tiny",
        prompts: list[str] | None = None,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        device: str | int | None = None,
    ):
        self.prompts = prompts or ["billboard"]
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        logger.info("Loading Grounding DINO checkpoint: %s", checkpoint)
        self._pipe = pipeline(
            task="zero-shot-object-detection",
            model=checkpoint,
            device=device,
        )

    def predict(self, image: Image.Image) -> list[BoxProposal]:
        """Run zero-shot detection on a single image and return box proposals."""
        candidate_labels = [f"{p}." for p in self.prompts]
        results = self._pipe(
            image,
            candidate_labels=candidate_labels,
            threshold=self.box_threshold,
        )

        proposals = []
        for result in results:
            box = result["box"]
            proposals.append(
                BoxProposal(
                    label=result["label"].rstrip("."),
                    score=result["score"],
                    box_xyxy=(box["xmin"], box["ymin"], box["xmax"], box["ymax"]),
                )
            )
        return proposals
