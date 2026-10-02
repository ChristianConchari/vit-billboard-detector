"""RT-DETR loading helpers for fine-tuning on the billboard classes."""

import math
from pathlib import Path

import torch
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    BaseImageProcessor,
    PreTrainedModel,
)


def load_rtdetr(
    checkpoint: str | Path, label_names: list[str]
) -> tuple[PreTrainedModel, BaseImageProcessor]:
    """Load RT-DETR with a classification head sized for `label_names`.

    Works for both the COCO-pretrained hub checkpoint (the 80-class head is
    re-initialized) and a fine-tuned local checkpoint.
    """
    model = AutoModelForObjectDetection.from_pretrained(
        checkpoint,
        id2label=dict(enumerate(label_names)),
        label2id={name: idx for idx, name in enumerate(label_names)},
        ignore_mismatched_sizes=True,
    )
    image_processor = AutoImageProcessor.from_pretrained(checkpoint)
    return model, image_processor


def freeze_backbone(model: PreTrainedModel) -> None:
    for parameter in model.model.backbone.parameters():
        parameter.requires_grad = False


def init_classification_bias(model: PreTrainedModel, prior_probability: float = 0.01) -> None:
    """Reset the freshly initialized class heads to RT-DETR's focal-loss prior.

    With a zero bias all 300 queries start at probability 0.5, which makes the
    varifocal loss explode during the first fine-tuning steps.
    """
    bias = -math.log((1 - prior_probability) / prior_probability)
    for head in [*model.model.decoder.class_embed, model.model.enc_score_head]:
        torch.nn.init.constant_(head.bias, bias)
