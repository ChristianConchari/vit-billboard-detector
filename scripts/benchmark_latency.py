"""CLI: compare end-to-end inference latency of RT-DETR and Grounding DINO.

Batch size 1 on the images of a reviewed split. Results are written to
reports/metrics/latency_<split>.json (quick checks; tracked results come from
scripts/run_pipeline.py).

Usage:
    python scripts/benchmark_latency.py --checkpoint checkpoints/rtdetr/<run>/best
"""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from vit.eval.latency import measure_latency
from vit.inference.factory import build_grounding_dino_detector, build_rtdetr_detector
from vit.utils.config import load_config
from vit.utils.logging import get_logger

logger = get_logger(__name__, log_file="evaluation.log")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark detector inference latency")
    parser.add_argument("--checkpoint", required=True, help="Fine-tuned RT-DETR checkpoint dir")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--rtdetr-config", default="configs/model/rtdetr.yaml")
    parser.add_argument("--grounding-dino-config", default="configs/model/grounding_dino.yaml")
    parser.add_argument("--output-dir", default="reports/metrics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rtdetr_config = load_config(args.rtdetr_config)
    gdino_config = load_config(args.grounding_dino_config)
    data_config = rtdetr_config["data"]
    ground_truth = json.loads(Path(data_config[f"{args.split}_annotations"]).read_text())
    images = [
        Image.open(Path(data_config["image_dir"]) / info["file_name"]).convert("RGB")
        for info in ground_truth["images"]
    ]

    detectors = {
        "rtdetr": build_rtdetr_detector(
            args.checkpoint,
            rtdetr_config["model"]["label_names"],
            rtdetr_config["evaluation"]["score_threshold"],
        ),
        "grounding-dino": build_grounding_dino_detector(
            gdino_config, gdino_config["evaluation"]["box_threshold"]
        ),
    }
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

    results = {"device": device, "image_size": [images[0].width, images[0].height]}
    for name, detector in detectors.items():
        stats = measure_latency(detector, images, warmup=args.warmup, repeats=args.repeats)
        results[name] = stats.to_dict()
        logger.info(
            "%s on %s: mean %.1f ms | median %.1f ms | p95 %.1f ms | %.1f FPS (%d runs)",
            name,
            device,
            stats.mean_ms,
            stats.median_ms,
            stats.p95_ms,
            stats.fps,
            stats.images,
        )

    output_path = Path(args.output_dir) / f"latency_{args.split}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))

    logger.info("Latency results -> %s", output_path)


if __name__ == "__main__":
    main()
