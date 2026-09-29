from vit.labeling.coco_writer import CocoDatasetBuilder


def test_add_image_assigns_incrementing_ids():
    builder = CocoDatasetBuilder(category_names=["billboard"])

    first_id = builder.add_image("a.jpg", width=800, height=600)
    second_id = builder.add_image("b.jpg", width=1024, height=768)

    assert first_id == 1
    assert second_id == 2
    assert [img["file_name"] for img in builder.images] == ["a.jpg", "b.jpg"]


def test_add_annotation_converts_bbox_and_area():
    builder = CocoDatasetBuilder(category_names=["billboard"])
    image_id = builder.add_image("a.jpg", width=800, height=600)

    builder.add_annotation(
        image_id=image_id,
        category_name="billboard",
        bbox_xywh=(10.0, 20.0, 100.0, 50.0),
        score=0.87,
    )

    annotation = builder.annotations[0]
    assert annotation["bbox"] == [10.0, 20.0, 100.0, 50.0]
    assert annotation["area"] == 5000.0
    assert annotation["score"] == 0.87
    assert annotation["category_id"] == 1


def test_add_annotation_rejects_unknown_category():
    builder = CocoDatasetBuilder(category_names=["billboard"])
    image_id = builder.add_image("a.jpg", width=800, height=600)

    try:
        builder.add_annotation(
            image_id=image_id,
            category_name="not-a-category",
            bbox_xywh=(0, 0, 1, 1),
        )
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_save_writes_valid_coco_json(tmp_path):
    builder = CocoDatasetBuilder(category_names=["billboard"])
    image_id = builder.add_image("a.jpg", width=800, height=600)
    builder.add_annotation(
        image_id=image_id, category_name="billboard", bbox_xywh=(0, 0, 10, 10)
    )

    output_path = tmp_path / "nested" / "auto_labels.json"
    builder.save(output_path)

    assert output_path.is_file()

    import json

    data = json.loads(output_path.read_text())
    assert len(data["images"]) == 1
    assert len(data["annotations"]) == 1
    assert data["categories"] == [{"id": 1, "name": "billboard"}]
