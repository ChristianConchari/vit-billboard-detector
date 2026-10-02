import json
import zipfile

from vit.data.label_studio_export import (
    copy_referenced_images,
    extract_if_zip,
    find_coco_json,
    import_label_studio_export,
    normalize_file_names,
)

VALID_COCO = {
    "images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}],
    "annotations": [],
    "categories": [{"id": 1, "name": "billboard"}],
}


def test_extract_if_zip_extracts_and_returns_dir(tmp_path):
    zip_path = tmp_path / "export.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("result.json", json.dumps(VALID_COCO))
        zf.writestr("images/a.jpg", b"fake-bytes")

    extract_dir = tmp_path / "extracted"
    result_dir = extract_if_zip(zip_path, extract_dir)

    assert result_dir == extract_dir
    assert (extract_dir / "result.json").is_file()
    assert (extract_dir / "images" / "a.jpg").is_file()


def test_extract_if_zip_passthrough_for_non_zip(tmp_path):
    json_path = tmp_path / "export.json"
    json_path.write_text("{}")

    result_dir = extract_if_zip(json_path, tmp_path / "unused")

    assert result_dir == tmp_path


def test_find_coco_json_picks_the_coco_formatted_file(tmp_path):
    (tmp_path / "not_coco.json").write_text(json.dumps({"foo": "bar"}))
    coco_path = tmp_path / "result.json"
    coco_path.write_text(json.dumps(VALID_COCO))

    found = find_coco_json(tmp_path)

    assert found == coco_path


def test_find_coco_json_raises_when_none_found(tmp_path):
    (tmp_path / "not_coco.json").write_text(json.dumps({"foo": "bar"}))

    try:
        find_coco_json(tmp_path)
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_normalize_file_names_strips_directory_prefixes():
    coco = {
        "images": [{"id": 1, "file_name": "task-42/uploads/hash-a.jpg"}],
    }

    normalized = normalize_file_names(coco)

    assert normalized["images"][0]["file_name"] == "hash-a.jpg"


def test_copy_referenced_images_copies_matches_and_reports_missing(tmp_path):
    search_dir = tmp_path / "search"
    (search_dir / "nested").mkdir(parents=True)
    (search_dir / "nested" / "found.jpg").write_bytes(b"data")

    output_dir = tmp_path / "out"
    coco = {"images": [{"file_name": "found.jpg"}, {"file_name": "missing.jpg"}]}

    missing = copy_referenced_images(coco, search_dirs=[search_dir], output_dir=output_dir)

    assert missing == ["missing.jpg"]
    assert (output_dir / "found.jpg").is_file()


def test_import_label_studio_export_falls_back_to_raw_images(tmp_path):
    zip_path = tmp_path / "export.zip"
    exported = {
        **VALID_COCO,
        "images": [{**VALID_COCO["images"][0], "file_name": "../../raw/a.jpg"}],
    }
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("result.json", json.dumps(exported))
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "a.jpg").write_bytes(b"fake-bytes")

    coco, missing = import_label_studio_export(
        zip_path,
        output_path=tmp_path / "reviewed" / "master.json",
        images_out_dir=tmp_path / "processed",
        extract_dir=tmp_path / "extracted",
        fallback_image_dirs=[raw_dir],
    )

    assert missing == []
    assert coco["images"][0]["file_name"] == "a.jpg"
    assert (tmp_path / "processed" / "a.jpg").read_bytes() == b"fake-bytes"
    assert json.loads((tmp_path / "reviewed" / "master.json").read_text()) == coco
