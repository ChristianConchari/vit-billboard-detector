"""Zero-shot bounding box proposals for billboard images using Grounding DINO.

Used as the first stage of the labeling pipeline: generates candidate boxes
via text prompts so a human only has to review/correct them instead of
annotating from scratch.
"""

import torch
from PIL import Image
from torchvision.ops import nms
from transformers import pipeline

from vit.inference.detection import BoxProposal
from vit.utils.logging import get_logger

logger = get_logger(__name__, log_file="labeling.log")


def deduplicate_proposals(
    proposals: list[BoxProposal], iou_threshold: float = 0.5
) -> list[BoxProposal]:
    """Class-agnostic NMS over proposals.

    Querying multiple near-synonym prompts (e.g. "billboard", "advertising
    sign") makes Grounding DINO return several near-identical boxes for the
    same physical object. This collapses them to the highest-scoring box per
    overlapping cluster so manual review doesn't have to deal with
    duplicates.
    """
    if not proposals:
        return proposals

    boxes = torch.tensor([p.box_xyxy for p in proposals], dtype=torch.float32)
    scores = torch.tensor([p.score for p in proposals], dtype=torch.float32)
    keep_indices = nms(boxes, scores, iou_threshold).tolist()
    return [proposals[i] for i in keep_indices]


class GroundingDinoAutoLabeler:
    """Wraps a Hugging Face zero-shot-object-detection pipeline for Grounding DINO."""

    def __init__(
        self,
        checkpoint: str = "IDEA-Research/grounding-dino-tiny",
        prompts: list[str] | None = None,
        box_threshold: float = 0.35,
        nms_iou_threshold: float = 0.5,
        device: str | int | None = None,
    ):
        self.prompts = prompts or ["billboard"]
        self.box_threshold = box_threshold
        self.nms_iou_threshold = nms_iou_threshold

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
        return deduplicate_proposals(proposals, self.nms_iou_threshold)
