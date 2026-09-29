import pytest

from vit.data.dataset_split import split_coco_dataset


def _make_coco(n_images: int) -> dict:
    return {
        "images": [
            {"id": i, "file_name": f"img_{i}.jpg", "width": 100, "height": 100}
            for i in range(1, n_images + 1)
        ],
        "annotations": [
            {"id": i, "image_id": i, "category_id": 1, "bbox": [0, 0, 10, 10], "area": 100, "iscrowd": 0}
            for i in range(1, n_images + 1)
        ],
        "categories": [{"id": 1, "name": "billboard"}],
    }


def test_split_ratios_must_sum_to_one():
    with pytest.raises(ValueError):
        split_coco_dataset(_make_coco(10), train_ratio=0.5, val_ratio=0.5, test_ratio=0.5)


def test_split_partitions_all_images_without_overlap():
    coco = _make_coco(20)

    splits = split_coco_dataset(coco, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)

    train_ids = {img["id"] for img in splits["train"]["images"]}
    val_ids = {img["id"] for img in splits["val"]["images"]}
    test_ids = {img["id"] for img in splits["test"]["images"]}

    assert train_ids | val_ids | test_ids == set(range(1, 21))
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    assert len(train_ids) == 14
    assert len(val_ids) == 3
    assert len(test_ids) == 3


def test_split_is_deterministic_given_same_seed():
    coco = _make_coco(20)

    first = split_coco_dataset(coco, seed=7)
    second = split_coco_dataset(coco, seed=7)

    assert [img["id"] for img in first["train"]["images"]] == [
        img["id"] for img in second["train"]["images"]
    ]


def test_split_keeps_only_matching_annotations_per_subset():
    coco = _make_coco(4)

    splits = split_coco_dataset(coco, train_ratio=0.5, val_ratio=0.25, test_ratio=0.25, seed=1)

    for name, subset in splits.items():
        image_ids = {img["id"] for img in subset["images"]}
        annotation_image_ids = {a["image_id"] for a in subset["annotations"]}
        assert annotation_image_ids <= image_ids


def test_split_preserves_categories_in_every_subset():
    coco = _make_coco(4)

    splits = split_coco_dataset(coco, train_ratio=0.5, val_ratio=0.25, test_ratio=0.25, seed=1)

    for subset in splits.values():
        assert subset["categories"] == [{"id": 1, "name": "billboard"}]
