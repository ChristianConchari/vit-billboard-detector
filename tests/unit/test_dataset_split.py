import pytest

from vit.data.dataset_split import (
    assign_videos_to_splits,
    split_coco_by_video,
    video_id_from_file_name,
)


def _file_names(frames_per_video: dict[str, int]) -> list[str]:
    return [f"{video}_{frame}.jpg" for video, n in frames_per_video.items() for frame in range(n)]


def _make_coco(file_names: list[str]) -> dict:
    return {
        "images": [
            {"id": i, "file_name": name, "width": 100, "height": 100}
            for i, name in enumerate(file_names, start=1)
        ],
        "annotations": [
            {"id": i, "image_id": i, "category_id": 1, "bbox": [0, 0, 10, 10], "area": 100, "iscrowd": 0}
            for i in range(1, len(file_names) + 1)
        ],
        "categories": [{"id": 1, "name": "billboard"}],
    }


def test_video_id_is_everything_before_the_frame_number():
    assert video_id_from_file_name("0e99-c98_ab_2096.jpg") == "0e99-c98_ab"


def test_video_id_rejects_names_without_frame_number():
    with pytest.raises(ValueError):
        video_id_from_file_name("billboard-sample.png")


def test_assignment_ratios_must_sum_to_one():
    with pytest.raises(ValueError):
        assign_videos_to_splits(["a_1.jpg"], train_ratio=0.5, val_ratio=0.5, test_ratio=0.5)


def test_assignment_balances_image_counts_across_splits():
    pool = _file_names({f"v{i}": 10 for i in range(20)})

    assignment = assign_videos_to_splits(pool, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

    images_per_split = {s: 10 * list(assignment.values()).count(s) for s in ("train", "val", "test")}
    assert images_per_split == {"train": 140, "val": 30, "test": 30}


def test_assignment_is_deterministic_given_same_seed():
    pool = _file_names({f"v{i}": i + 1 for i in range(10)})

    assert assign_videos_to_splits(pool, seed=7) == assign_videos_to_splits(list(reversed(pool)), seed=7)


def test_split_never_puts_one_video_in_two_splits():
    pool = _file_names({"a": 5, "b": 5, "c": 5, "d": 5})
    assignment = {"a": "train", "b": "train", "c": "val", "d": "test"}

    splits = split_coco_by_video(_make_coco(pool), assignment)

    for name, subset in splits.items():
        videos = {video_id_from_file_name(img["file_name"]) for img in subset["images"]}
        assert videos == {v for v, s in assignment.items() if s == name}


def test_split_of_partial_export_keeps_videos_in_their_assigned_split():
    assignment = {"a": "train", "b": "val", "c": "test"}
    partial = _make_coco(["a_1.jpg", "c_7.jpg"])

    splits = split_coco_by_video(partial, assignment)

    assert [img["file_name"] for img in splits["train"]["images"]] == ["a_1.jpg"]
    assert splits["val"]["images"] == []
    assert [img["file_name"] for img in splits["test"]["images"]] == ["c_7.jpg"]


def test_split_rejects_images_from_unassigned_videos():
    with pytest.raises(ValueError):
        split_coco_by_video(_make_coco(["new_1.jpg"]), {"a": "train"})


def test_split_keeps_only_matching_annotations_and_categories_per_subset():
    pool = _file_names({"a": 2, "b": 2, "c": 2})

    splits = split_coco_by_video(_make_coco(pool), {"a": "train", "b": "val", "c": "test"})

    for subset in splits.values():
        image_ids = {img["id"] for img in subset["images"]}
        assert {a["image_id"] for a in subset["annotations"]} == image_ids
        assert subset["categories"] == [{"id": 1, "name": "billboard"}]
