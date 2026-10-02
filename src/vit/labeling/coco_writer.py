"""Incremental builder for COCO-format detection datasets."""

import json
from pathlib import Path
from typing import Any


class CocoDatasetBuilder:
    """Accumulates images/annotations and writes a COCO-format JSON file.

    Annotation scores are kept (non-standard COCO field) so auto-generated
    labels can be filtered/reviewed by confidence before promoting them
    from data/annotations/auto to data/annotations/reviewed.
    """

    def __init__(self, category_names: list[str]):
        self.categories = [{"id": idx + 1, "name": name} for idx, name in enumerate(category_names)]
        self._category_id_by_name = {c["name"]: c["id"] for c in self.categories}
        self.images: list[dict[str, Any]] = []
        self.annotations: list[dict[str, Any]] = []
        self._next_image_id = 1
        self._next_annotation_id = 1

    def add_image(self, file_name: str, width: int, height: int) -> int:
        image_id = self._next_image_id
        self._next_image_id += 1
        self.images.append(
            {"id": image_id, "file_name": file_name, "width": width, "height": height}
        )
        return image_id

    def add_annotation(
        self,
        image_id: int,
        category_name: str,
        bbox_xywh: tuple[float, float, float, float],
        score: float | None = None,
    ) -> int:
        if category_name not in self._category_id_by_name:
            raise ValueError(f"Unknown category: {category_name!r}")

        x, y, w, h = bbox_xywh
        annotation_id = self._next_annotation_id
        self._next_annotation_id += 1
        annotation = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": self._category_id_by_name[category_name],
            "bbox": [x, y, w, h],
            "area": w * h,
            "iscrowd": 0,
        }
        if score is not None:
            annotation["score"] = score
        self.annotations.append(annotation)
        return annotation_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "images": self.images,
            "annotations": self.annotations,
            "categories": self.categories,
        }

    def save(self, output_path: str | Path) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
