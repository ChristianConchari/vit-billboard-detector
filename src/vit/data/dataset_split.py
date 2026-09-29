"""Split a COCO dataset into train/val/test subsets."""
import random
from typing import Any


def split_coco_dataset(
    coco: dict[str, Any],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, dict[str, Any]]:
    """Shuffle images (seeded) and partition images+annotations into train/val/test."""
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    image_ids = [img["id"] for img in coco["images"]]
    random.Random(seed).shuffle(image_ids)

    n = len(image_ids)
    n_train = round(n * train_ratio)
    n_val = round(n * val_ratio)

    split_ids = {
        "train": set(image_ids[:n_train]),
        "val": set(image_ids[n_train : n_train + n_val]),
        "test": set(image_ids[n_train + n_val :]),
    }

    return {name: _subset(coco, ids) for name, ids in split_ids.items()}


def _subset(coco: dict[str, Any], image_ids: set[int]) -> dict[str, Any]:
    return {
        "images": [img for img in coco["images"] if img["id"] in image_ids],
        "annotations": [a for a in coco["annotations"] if a["image_id"] in image_ids],
        "categories": coco["categories"],
    }
