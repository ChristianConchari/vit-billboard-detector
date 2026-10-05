"""Box-aware training augmentations built on torchvision.transforms.v2."""

from typing import Any

import torch
from PIL import Image
from torchvision import tv_tensors
from torchvision.transforms import v2

BoxesXYWH = list[list[float]]


def build_train_augmentations(config: dict[str, Any]) -> v2.Compose:
    color = config["color_jitter"]
    return v2.Compose(
        [
            v2.RandomHorizontalFlip(p=config["horizontal_flip"]),
            v2.RandomAffine(
                degrees=0,
                translate=(config["translate"], config["translate"]),
                scale=tuple(config["scale_range"]),
            ),
            v2.ColorJitter(brightness=color, contrast=color, saturation=color),
            v2.ClampBoundingBoxes(),
            v2.SanitizeBoundingBoxes(min_size=2),
        ]
    )


def apply_augmentations(
    augmentations: v2.Compose,
    image: Image.Image,
    boxes_xywh: BoxesXYWH,
    labels: list[int],
) -> tuple[Image.Image, BoxesXYWH, list[int]]:
    boxes = tv_tensors.BoundingBoxes(
        torch.tensor(boxes_xywh, dtype=torch.float32).reshape(-1, 4),
        format="XYWH",
        canvas_size=(image.height, image.width),
    )
    target = {"boxes": boxes, "labels": torch.tensor(labels, dtype=torch.int64)}
    image, target = augmentations(image, target)
    return image, target["boxes"].tolist(), target["labels"].tolist()
