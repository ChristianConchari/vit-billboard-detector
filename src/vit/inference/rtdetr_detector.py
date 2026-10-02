"""Detector adapter around a (fine-tuned) RT-DETR model."""

import torch
from PIL import Image
from transformers import BaseImageProcessor, PreTrainedModel

from vit.inference.detection import BoxProposal


class RtDetrDetector:
    def __init__(
        self,
        model: PreTrainedModel,
        image_processor: BaseImageProcessor,
        device: torch.device,
        score_threshold: float = 0.5,
    ):
        self.model = model
        self.image_processor = image_processor
        self.device = device
        self.score_threshold = score_threshold

    @torch.no_grad()
    def predict(self, image: Image.Image) -> list[BoxProposal]:
        self.model.eval()
        inputs = self.image_processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        result = self.image_processor.post_process_object_detection(
            outputs, threshold=self.score_threshold, target_sizes=[(image.height, image.width)]
        )[0]

        boxes = result["boxes"].clamp(min=0)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(max=image.width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(max=image.height)
        return [
            BoxProposal(
                label=self.model.config.id2label[label],
                score=score,
                box_xyxy=tuple(box),
            )
            for box, score, label in zip(
                boxes.tolist(), result["scores"].tolist(), result["labels"].tolist(), strict=True
            )
        ]
