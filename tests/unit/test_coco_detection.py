import json

import pytest
from PIL import Image

from vit.data.coco_detection import CocoDetectionDataset, map_categories_to_labels
from vit.transforms.augmentations import apply_augmentations, build_train_augmentations

AUGMENTATION_CONFIG = {
    "horizontal_flip": 0.5,
    "scale_range": [0.75, 1.25],
    "translate": 0.1,
    "color_jitter": 0.2,
}


@pytest.fixture
def coco_dir(tmp_path):
    for name in ("a_1.jpg", "a_2.jpg"):
        Image.new("RGB", (200, 100)).save(tmp_path / name)
    coco = {
        "images": [
            {"id": 0, "file_name": "a_1.jpg", "width": 200, "height": 100},
            {"id": 1, "file_name": "a_2.jpg", "width": 200, "height": 100},
        ],
        "annotations": [
            {
                "id": 0,
                "image_id": 0,
                "category_id": 0,
                "bbox": [10, 20, 50, 40],
                "area": 2000,
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 0, "name": "billboard"}],
    }
    (tmp_path / "annotations.json").write_text(json.dumps(coco))
    return tmp_path


def test_categories_map_to_labels_by_name_not_by_id():
    categories = [{"id": 1, "name": "billboard"}]

    assert map_categories_to_labels(categories, ["billboard"]) == {1: 0}


def test_unknown_category_name_is_rejected():
    with pytest.raises(ValueError):
        map_categories_to_labels([{"id": 0, "name": "car"}], ["billboard"])


def test_dataset_yields_processor_ready_targets(coco_dir):
    dataset = CocoDetectionDataset(
        coco_dir / "annotations.json", coco_dir, ["billboard"]
    )

    image, target = dataset[0]

    assert image.size == (200, 100)
    assert target == {
        "image_id": 0,
        "annotations": [
            {"bbox": [10, 20, 50, 40], "category_id": 0, "area": 2000, "iscrowd": 0}
        ],
    }


def test_dataset_keeps_images_without_boxes_as_negatives(coco_dir):
    dataset = CocoDetectionDataset(
        coco_dir / "annotations.json", coco_dir, ["billboard"]
    )

    _, target = dataset[1]

    assert len(dataset) == 2
    assert target["annotations"] == []


def test_augmented_boxes_stay_inside_the_image():
    augmentations = build_train_augmentations(AUGMENTATION_CONFIG)
    image = Image.new("RGB", (200, 100))

    for _ in range(20):
        augmented, boxes, labels = apply_augmentations(
            augmentations, image, [[150, 60, 50, 40]], [0]
        )
        assert len(boxes) == len(labels)
        for x, y, w, h in boxes:
            assert x >= 0 and y >= 0 and w > 0 and h > 0
            assert x + w <= augmented.width and y + h <= augmented.height


def test_augmentations_accept_images_without_boxes():
    augmentations = build_train_augmentations(AUGMENTATION_CONFIG)

    _, boxes, labels = apply_augmentations(
        augmentations, Image.new("RGB", (200, 100)), [], []
    )

    assert boxes == [] and labels == []
