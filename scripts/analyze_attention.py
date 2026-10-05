"""CLI: does RT-DETR's encoder attend to other billboards in the same frame?

For every detection matched to a ground-truth billboard (IoU >= 0.5) in images
with at least two billboards, measures the attention lift of the token under the
detection's center: attention share inside the *other* billboards divided by the
area share they cover (1.0 = chance), and the same for its own billboard.
Writes reports/metrics/attention_<split>.json.

Usage:
    python scripts/analyze_attention.py --checkpoint checkpoints/rtdetr/<run>/best
"""

import argparse
import json
import statistics
from pathlib import Path

import torch
from PIL import Image
from torchvision.ops import box_iou
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from vit.eval.attention_maps import attention_lift, explain_detections
from vit.eval.threshold import load_calibrated_threshold
from vit.utils.config import load_config
from vit.utils.logging import get_logger

logger = get_logger(__name__, log_file="evaluation.log")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure encoder attention lift")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument(
        "--score-threshold",
        type=float,
        help="Defaults to the checkpoint's calibrated threshold",
    )
    parser.add_argument("--max-detections", type=int, default=10)
    parser.add_argument("--rtdetr-config", default="configs/model/rtdetr.yaml")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/metrics"))
    return parser.parse_args()


def summarize(lifts: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(lifts),
        "mean": statistics.fmean(lifts),
        "above_chance": sum(lift > 1 for lift in lifts) / len(lifts),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.rtdetr_config)
    data_cfg = config["data"]
    split = json.loads(Path(data_cfg[f"{args.split}_annotations"]).read_text())
    threshold = args.score_threshold
    if threshold is None:
        threshold = load_calibrated_threshold(args.checkpoint)
    if threshold is None:
        threshold = config["inference"]["score_threshold"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForObjectDetection.from_pretrained(
        args.checkpoint, attn_implementation="eager"
    ).to(device)
    image_processor = AutoImageProcessor.from_pretrained(args.checkpoint)

    boxes_by_image: dict[int, list[tuple[float, float, float, float]]] = {}
    for annotation in split["annotations"]:
        x, y, w, h = annotation["bbox"]
        boxes_by_image.setdefault(annotation["image_id"], []).append(
            (x, y, x + w, y + h)
        )

    own_lifts, other_lifts = [], []
    for image_info in split["images"]:
        boxes = boxes_by_image.get(image_info["id"], [])
        if len(boxes) < 2:
            continue
        image = Image.open(Path(data_cfg["image_dir"]) / image_info["file_name"])
        image = image.convert("RGB")
        detections = explain_detections(
            model, image_processor, image, device, threshold, args.max_detections
        )
        for detection in detections:
            ious = box_iou(torch.tensor([detection.box_xyxy]), torch.tensor(boxes))[0]
            if ious.max() < 0.5:
                continue
            matched = int(ious.argmax())
            others = [box for index, box in enumerate(boxes) if index != matched]
            own = attention_lift(
                detection.encoder_attention, [boxes[matched]], image.size
            )
            other = attention_lift(detection.encoder_attention, others, image.size)
            if own is not None and other is not None:
                own_lifts.append(own)
                other_lifts.append(other)

    if not other_lifts:
        raise SystemExit("No matched detections in images with two or more billboards")
    report = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "score_threshold": threshold,
        "detections": len(other_lifts),
        "other_billboards": summarize(other_lifts),
        "own_billboard": summarize(own_lifts),
    }
    output_path = args.output_dir / f"attention_{args.split}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    logger.info(
        "%d detections | lift on other billboards: median %.2f, above chance %.0f%% "
        "| own billboard: median %.2f -> %s",
        report["detections"],
        report["other_billboards"]["median"],
        100 * report["other_billboards"]["above_chance"],
        report["own_billboard"]["median"],
        output_path,
    )


if __name__ == "__main__":
    main()
