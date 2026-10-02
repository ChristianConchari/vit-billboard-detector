from vit.eval.results import results_markdown

SUMMARY = {
    "run": "rtdetr-test",
    "checkpoint": "checkpoints/rtdetr/rtdetr-test/best",
    "dataset": {"train": {"images": 10, "boxes": 12}, "val": {"images": 3, "boxes": 4}},
    "test_metrics": {
        "rtdetr": {
            "mAP": 0.5,
            "AP50": 0.6,
            "AP75": 0.4,
            "AP_small": -1.0,
            "AP_medium": 0.2,
            "AP_large": 0.55,
        },
    },
    "calibration": {
        "score_threshold": 0.168,
        "precision": 0.9,
        "recall": 0.7,
        "f1": 0.8,
        "iou_threshold": 0.5,
    },
    "latency": {
        "device": "GPU",
        "models": {"rtdetr": {"mean_ms": 19.8, "median_ms": 19.6, "p95_ms": 20.5, "fps": 50.5}},
    },
}


def test_results_markdown_renders_every_section():
    markdown = results_markdown(SUMMARY)

    assert "| train | 10 | 12 |" in markdown
    assert "| RT-DETR fine-tuned | 0.500 | 0.600 | 0.400 | - | 0.200 | 0.550 |" in markdown
    assert "| 0.168 | 0.900 | 0.700 | 0.800 | 0.5 |" in markdown
    assert "| RT-DETR fine-tuned | 19.8 | 19.6 | 20.5 | 50.5 |" in markdown
