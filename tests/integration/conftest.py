"""Fixtures for end-to-end tests: a tiny randomly initialized RT-DETR and a synthetic dataset.

Nothing is downloaded, so the tests run on CPU in CI in a few seconds.
"""

import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw
from transformers import (
    RTDetrConfig,
    RTDetrForObjectDetection,
    RTDetrImageProcessor,
    RTDetrResNetConfig,
)

from vit.utils.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[2]
VIDEOS = {"trainvideo": "train", "valvideo": "val", "testvideo": "test"}
FRAMES_PER_VIDEO = 3
BILLBOARD_XYWH = [60, 40, 120, 80]


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory) -> Path:
    backbone = RTDetrResNetConfig(
        embedding_size=16,
        hidden_sizes=[16, 32, 64, 128],
        depths=[1, 1, 1, 1],
        layer_type="basic",
        out_features=["stage2", "stage3", "stage4"],
    )
    config = RTDetrConfig(
        backbone_config=backbone,
        encoder_in_channels=[32, 64, 128],
        decoder_in_channels=[64, 64, 64],
        d_model=64,
        encoder_hidden_dim=64,
        encoder_ffn_dim=128,
        decoder_ffn_dim=128,
        encoder_layers=1,
        decoder_layers=2,
        num_queries=20,
        num_denoising=10,
        num_labels=1,
        id2label={0: "billboard"},
        label2id={"billboard": 0},
    )
    checkpoint = tmp_path_factory.mktemp("tiny-rtdetr")
    RTDetrForObjectDetection(config).save_pretrained(checkpoint)
    RTDetrImageProcessor(size={"height": 128, "width": 128}).save_pretrained(checkpoint)
    return checkpoint


def _frame(file_path: Path) -> None:
    image = Image.new("RGB", (320, 240), (90, 130, 180))
    x, y, w, h = BILLBOARD_XYWH
    ImageDraw.Draw(image).rectangle([x, y, x + w, y + h], fill=(240, 220, 40))
    image.save(file_path)


@pytest.fixture
def workspace(tmp_path, tiny_checkpoint) -> dict:
    """Synthetic raw pool, Label Studio export and configs pointing at tmp_path."""
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    names = [f"{video}_{frame}.jpg" for video in VIDEOS for frame in range(FRAMES_PER_VIDEO)]
    for name in names:
        _frame(raw_dir / name)

    export = {
        "images": [
            {
                "id": i,
                "file_name": f"../../label-studio/files/data/raw/{name}",
                "width": 320,
                "height": 240,
            }
            for i, name in enumerate(names)
        ],
        "annotations": [
            {
                "id": i,
                "image_id": i,
                "category_id": 0,
                "bbox": BILLBOARD_XYWH,
                "area": 9600,
                "iscrowd": 0,
            }
            for i in range(len(names))
        ],
        "categories": [{"id": 0, "name": "billboard"}],
    }
    export_path = tmp_path / "export.json"
    export_path.write_text(json.dumps(export))

    reviewed_dir = tmp_path / "data" / "annotations" / "reviewed"
    reviewed_dir.mkdir(parents=True)
    (reviewed_dir / "split_assignment.json").write_text(json.dumps(VIDEOS))

    pipeline = load_config(REPO_ROOT / "configs" / "pipeline.yaml")
    pipeline["data"].update(
        label_studio_extract_dir=str(tmp_path / "data" / "interim"),
        reviewed_annotations=str(reviewed_dir / "reviewed_master.json"),
        raw_image_dir=str(raw_dir),
        split_assignment=str(reviewed_dir / "split_assignment.json"),
    )
    pipeline["latency"].update(warmup=0, repeats=1)
    pipeline["output_dir"] = str(tmp_path / "reports" / "runs")

    rtdetr = load_config(REPO_ROOT / "configs" / "model" / "rtdetr.yaml")
    rtdetr["model"]["pretrained_checkpoint"] = str(tiny_checkpoint)
    rtdetr["data"].update(
        {f"{split}_annotations": str(reviewed_dir / f"{split}.json") for split in VIDEOS.values()},
        image_dir=str(tmp_path / "data" / "processed"),
    )
    rtdetr["training"].update(
        epochs=1, batch_size=2, num_workers=0, checkpoint_dir=str(tmp_path / "checkpoints")
    )
    rtdetr["mlflow"]["tracking_uri"] = f"sqlite:///{tmp_path / 'mlflow.db'}"

    return {
        "export_path": export_path,
        "pipeline": pipeline,
        "rtdetr": rtdetr,
        "grounding_dino": load_config(REPO_ROOT / "configs" / "model" / "grounding_dino.yaml"),
        "tmp_path": tmp_path,
    }
