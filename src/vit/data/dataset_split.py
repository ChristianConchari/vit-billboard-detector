"""Split a COCO dataset into train/val/test subsets, grouped by source video.

Images are consecutive video frames, so near-duplicates must never cross
splits (see docs/decisions/0002-split-by-video.md). Whole videos are assigned
to a split once, over the full image pool, and that assignment is reused for
every (partial) reviewed export so splits stay stable while labeling grows.
"""

import json
import random
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

SPLIT_NAMES = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def video_id_from_file_name(file_name: str) -> str:
    """Extract the video id from a `<video_id>_<frame_number>.<ext>` file name."""
    stem = Path(file_name).stem
    video_id, sep, frame = stem.rpartition("_")
    if not sep or not video_id or not frame.isdigit():
        raise ValueError(f"File name does not match '<video_id>_<frame>': {file_name}")
    return video_id


def assign_videos_to_splits(
    file_names: Iterable[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, str]:
    """Assign each video to a split so image counts approximate the given ratios.

    Greedy: videos are visited largest first (seeded shuffle breaks ties) and
    each goes to the split furthest below its target image count. Returns
    {video_id: split_name}.
    """
    ratios: dict[str, float] = dict(
        zip(SPLIT_NAMES, (train_ratio, val_ratio, test_ratio), strict=True)
    )
    if abs(sum(ratios.values()) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    images_per_video = Counter(video_id_from_file_name(name) for name in file_names)
    total = sum(images_per_video.values())

    videos = sorted(images_per_video)
    random.Random(seed).shuffle(videos)
    videos.sort(key=lambda v: images_per_video[v], reverse=True)

    filled: dict[str, int] = dict.fromkeys(SPLIT_NAMES, 0)
    assignment = {}
    for video in videos:
        split = max(
            (s for s in SPLIT_NAMES if ratios[s] > 0),
            key=lambda s: ratios[s] * total - filled[s],
        )
        assignment[video] = split
        filled[split] += images_per_video[video]
    return assignment


def load_or_build_assignment(
    assignment_path: Path,
    pool_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    rebuild: bool = False,
) -> dict[str, str]:
    """Reuse the saved video assignment, or build it from the full image pool and save it."""
    if assignment_path.exists() and not rebuild:
        return json.loads(assignment_path.read_text())

    pool = [p.name for p in pool_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
    if not pool:
        raise ValueError(f"No images in {pool_dir} to build the split assignment from")

    assignment = assign_videos_to_splits(pool, train_ratio, val_ratio, test_ratio, seed)
    assignment_path.parent.mkdir(parents=True, exist_ok=True)
    assignment_path.write_text(json.dumps(assignment, indent=2, sort_keys=True))
    return assignment


def split_coco_by_video(
    coco: dict[str, Any], assignment: dict[str, str]
) -> dict[str, dict[str, Any]]:
    """Partition images+annotations into train/val/test using a video assignment."""
    split_ids: dict[str, set[int]] = {name: set() for name in SPLIT_NAMES}
    unassigned = []
    for image in coco["images"]:
        split = assignment.get(video_id_from_file_name(image["file_name"]))
        if split is None:
            unassigned.append(image["file_name"])
        else:
            split_ids[split].add(image["id"])

    if unassigned:
        raise ValueError(
            f"{len(unassigned)} image(s) belong to videos missing from the split assignment "
            f"(rebuild it from the full image pool), e.g. {unassigned[:3]}"
        )

    return {name: _subset(coco, ids) for name, ids in split_ids.items()}


def _subset(coco: dict[str, Any], image_ids: set[int]) -> dict[str, Any]:
    return {
        "images": [img for img in coco["images"] if img["id"] in image_ids],
        "annotations": [a for a in coco["annotations"] if a["image_id"] in image_ids],
        "categories": coco["categories"],
    }


def write_splits(splits: dict[str, dict[str, Any]], paths: dict[str, Path]) -> None:
    for name, subset in splits.items():
        paths[name].parent.mkdir(parents=True, exist_ok=True)
        paths[name].write_text(json.dumps(subset, indent=2))


def exclude_split(coco: dict[str, Any], assignment: dict[str, str], split: str) -> dict[str, Any]:
    """Drop the images (and their annotations) whose video is assigned to `split`."""
    kept_ids = {
        image["id"]
        for image in coco["images"]
        if assignment.get(video_id_from_file_name(image["file_name"])) != split
    }
    return _subset(coco, kept_ids)
