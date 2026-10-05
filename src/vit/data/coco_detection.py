"""PyTorch dataset over a COCO detection file, yielding image processor inputs."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2
from transformers import BaseImageProcessor, BatchFeature

from vit.transforms.augmentations import apply_augmentations


def map_categories_to_labels(
    categories: list[dict[str, Any]], label_names: list[str]
) -> dict[int, int]:
    """Map COCO category ids to model label indices by category name.

    Label Studio exports 0-based category ids while other tools use 1-based
    ones, so ids are never trusted as label indices.
    """
    unknown = [c["name"] for c in categories if c["name"] not in label_names]
    if unknown:
        raise ValueError(
            f"Categories {unknown} are not in the model labels {label_names}"
        )
    return {c["id"]: label_names.index(c["name"]) for c in categories}


class CocoDetectionDataset(Dataset):
    """Yields (image, target) pairs in the processor's COCO detection format."""

    def __init__(
        self,
        annotations_path: str | Path,
        image_dir: str | Path,
        label_names: list[str],
        augmentations: v2.Compose | None = None,
    ):
        coco = json.loads(Path(annotations_path).read_text())
        self.image_dir = Path(image_dir)
        self.augmentations = augmentations
        self.images = coco["images"]
        self.label_by_category_id = map_categories_to_labels(
            coco["categories"], label_names
        )
        self.annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for annotation in coco["annotations"]:
            self.annotations_by_image[annotation["image_id"]].append(annotation)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[Image.Image, dict[str, Any]]:
        image_info = self.images[index]
        image = Image.open(self.image_dir / image_info["file_name"]).convert("RGB")
        annotations = self.annotations_by_image[image_info["id"]]
        boxes = [a["bbox"] for a in annotations]
        labels = [self.label_by_category_id[a["category_id"]] for a in annotations]

        if self.augmentations is not None:
            image, boxes, labels = apply_augmentations(
                self.augmentations, image, boxes, labels
            )

        target = {
            "image_id": image_info["id"],
            "annotations": [
                {
                    "bbox": box,
                    "category_id": label,
                    "area": box[2] * box[3],
                    "iscrowd": 0,
                }
                for box, label in zip(boxes, labels, strict=True)
            ],
        }
        return image, target


class DetectionCollator:
    """Batches (image, target) pairs through the image processor (resize, labels)."""

    def __init__(self, image_processor: BaseImageProcessor):
        self.image_processor = image_processor

    def __call__(self, batch: list[tuple[Image.Image, dict[str, Any]]]) -> BatchFeature:
        images, targets = zip(*batch, strict=True)
        return self.image_processor(
            images=list(images), annotations=list(targets), return_tensors="pt"
        )
