import pytest

from vit.data.dataset_split import (
    assign_videos_to_splits,
    exclude_split,
    load_or_build_assignment,
    split_coco_by_video,
    split_fingerprint,
    subset_by_file_names,
    video_id_from_file_name,
    write_splits,
)


def _file_names(frames_per_video: dict[str, int]) -> list[str]:
    return [
        f"{video}_{frame}.jpg"
        for video, n in frames_per_video.items()
        for frame in range(n)
    ]


def _make_coco(file_names: list[str]) -> dict:
    return {
        "images": [
            {"id": i, "file_name": name, "width": 100, "height": 100}
            for i, name in enumerate(file_names, start=1)
        ],
        "annotations": [
            {
                "id": i,
                "image_id": i,
                "category_id": 1,
                "bbox": [0, 0, 10, 10],
                "area": 100,
                "iscrowd": 0,
            }
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
        assign_videos_to_splits(
            ["a_1.jpg"], train_ratio=0.5, val_ratio=0.5, test_ratio=0.5
        )


def test_assignment_balances_image_counts_across_splits():
    pool = _file_names({f"v{i}": 10 for i in range(20)})

    assignment = assign_videos_to_splits(
        pool, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
    )

    images_per_split = {
        s: 10 * list(assignment.values()).count(s) for s in ("train", "val", "test")
    }
    assert images_per_split == {"train": 140, "val": 30, "test": 30}


def test_assignment_is_deterministic_given_same_seed():
    pool = _file_names({f"v{i}": i + 1 for i in range(10)})

    assert assign_videos_to_splits(pool, seed=7) == assign_videos_to_splits(
        list(reversed(pool)), seed=7
    )


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

    splits = split_coco_by_video(
        _make_coco(pool), {"a": "train", "b": "val", "c": "test"}
    )

    for subset in splits.values():
        image_ids = {img["id"] for img in subset["images"]}
        assert {a["image_id"] for a in subset["annotations"]} == image_ids
        assert subset["categories"] == [{"id": 1, "name": "billboard"}]


def test_assignment_is_built_once_and_then_reused(tmp_path):
    pool_dir = tmp_path / "raw"
    pool_dir.mkdir()
    for name in _file_names({"a": 3, "b": 1}):
        (pool_dir / name).touch()
    assignment_path = tmp_path / "assignment.json"

    built = load_or_build_assignment(assignment_path, pool_dir)
    (pool_dir / "c_0.jpg").touch()
    reused = load_or_build_assignment(assignment_path, pool_dir)
    rebuilt = load_or_build_assignment(assignment_path, pool_dir, rebuild=True)

    assert set(built) == set(reused) == {"a", "b"}
    assert set(rebuilt) == {"a", "b", "c"}


def test_assignment_needs_pool_images(tmp_path):
    with pytest.raises(ValueError):
        load_or_build_assignment(tmp_path / "assignment.json", tmp_path)


def test_write_splits_writes_each_split_to_its_path(tmp_path):
    splits = split_coco_by_video(_make_coco(["a_1.jpg"]), {"a": "train"})
    paths = {name: tmp_path / "out" / f"{name}.json" for name in splits}

    write_splits(splits, paths)

    assert all(path.is_file() for path in paths.values())


def test_exclude_split_drops_images_and_boxes_of_that_split():
    coco = _make_coco(["a_1.jpg", "b_1.jpg", "c_1.jpg"])

    kept = exclude_split(coco, {"a": "train", "b": "test", "c": "val"}, "test")

    assert [img["file_name"] for img in kept["images"]] == ["a_1.jpg", "c_1.jpg"]
    assert {a["image_id"] for a in kept["annotations"]} == {1, 3}


def test_split_fingerprint_ignores_ids_and_order_but_not_boxes():
    coco = _make_coco(["a_1.jpg", "b_1.jpg"])
    renumbered = {
        "images": [{**img, "id": img["id"] + 100} for img in reversed(coco["images"])],
        "annotations": [
            {**a, "id": a["id"] + 50, "image_id": a["image_id"] + 100}
            for a in coco["annotations"]
        ],
        "categories": coco["categories"],
    }
    moved_box = {
        **coco,
        "annotations": [{**coco["annotations"][0], "bbox": [1, 0, 10, 10]}],
    }

    assert split_fingerprint(renumbered) == split_fingerprint(coco)
    assert split_fingerprint(moved_box) != split_fingerprint(coco)


def test_subset_by_file_names_keeps_only_the_named_images():
    coco = _make_coco(["a_1.jpg", "b_1.jpg", "c_1.jpg"])

    subset = subset_by_file_names(coco, ["c_1.jpg", "a_1.jpg"])

    assert sorted(img["file_name"] for img in subset["images"]) == [
        "a_1.jpg",
        "c_1.jpg",
    ]
    assert {a["image_id"] for a in subset["annotations"]} == {1, 3}


def test_subset_by_file_names_rejects_unknown_images():
    with pytest.raises(ValueError, match="z_9.jpg"):
        subset_by_file_names(_make_coco(["a_1.jpg"]), ["z_9.jpg"])
