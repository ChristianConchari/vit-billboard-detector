"""Import a Label Studio COCO export into the project's reviewed dataset."""

import json
import shutil
import zipfile
from pathlib import Path
from typing import Any


def extract_if_zip(export_path: Path, extract_dir: Path) -> Path:
    """If export_path is a zip, extract it into extract_dir and return extract_dir.

    Otherwise return export_path's parent directory unchanged.
    """
    if export_path.suffix.lower() == ".zip":
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(export_path) as zf:
            zf.extractall(extract_dir)
        return extract_dir
    return export_path.parent


def find_coco_json(directory: Path) -> Path:
    """Find the COCO-format annotation file inside an extracted export dir."""
    candidates = sorted(directory.rglob("*.json"))
    if not candidates:
        raise FileNotFoundError(f"No .json file found under {directory}")

    for path in candidates:
        data = json.loads(path.read_text())
        if {"images", "annotations", "categories"} <= data.keys():
            return path

    raise FileNotFoundError(
        "No COCO-format json (with images/annotations/categories) found under "
        f"{directory}"
    )


def normalize_file_names(coco: dict[str, Any]) -> dict[str, Any]:
    """Rewrite each image's file_name to its basename.

    Label Studio exports often prefix file_name with task id / hash
    directories; downstream code expects a flat data/processed/ layout.
    """
    for image in coco["images"]:
        image["file_name"] = Path(image["file_name"]).name
    return coco


def copy_referenced_images(
    coco: dict[str, Any], search_dirs: list[Path], output_dir: Path
) -> list[str]:
    """Copy every image referenced in `coco` into output_dir (flat), by basename.

    Searches search_dirs in order (first match wins). Returns the list of
    file names that could not be found in any search dir.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    index: dict[str, Path] = {}
    for directory in search_dirs:
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if path.is_file():
                index.setdefault(path.name, path)

    missing = []
    for image in coco["images"]:
        file_name = image["file_name"]
        source = index.get(file_name)
        if source is None:
            missing.append(file_name)
            continue
        shutil.copy2(source, output_dir / file_name)

    return missing


def import_label_studio_export(
    export_path: Path,
    output_path: Path,
    images_out_dir: Path,
    extract_dir: Path,
    fallback_image_dirs: list[Path],
) -> tuple[dict[str, Any], list[str]]:
    """Import a Label Studio COCO export (.zip or .json) into the reviewed dataset.

    Images are looked up in the export first, then in `fallback_image_dirs`:
    Local Storage-backed projects export annotations without image bytes.
    Returns the normalized COCO dict and the file names that weren't found.
    """
    if export_path.suffix.lower() == ".zip":
        search_root = extract_if_zip(export_path, extract_dir)
        coco_json_path = find_coco_json(search_root)
    else:
        coco_json_path, search_root = export_path, export_path.parent

    coco = normalize_file_names(json.loads(coco_json_path.read_text()))
    missing = copy_referenced_images(
        coco, [search_root, *fallback_image_dirs], images_out_dir
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(coco, indent=2))
    return coco, missing
