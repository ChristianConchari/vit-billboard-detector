from vit.eval.result_charts import plot_ap_per_iou, plot_latency, plot_model_comparison

TEST_METRICS = {
    "rtdetr": {"mAP": 0.64, "AP50": 0.93, "AP75": 0.78},
    "grounding-dino": {"mAP": 0.48, "AP50": 0.72, "AP75": 0.59},
}
LATENCY = {
    "device": "GPU",
    "models": {"rtdetr": {"mean_ms": 19.3}, "grounding-dino": {"mean_ms": 472.0}},
}
CURVES = [
    {"model": "rtdetr-run", "split": "test", "ap": {"0.5": 0.93, "0.95": 0.01}},
    {"model": "rtdetr-run", "split": "val", "ap": {"0.5": 0.89, "0.95": 0.48}},
    {
        "model": "Grounding DINO zero-shot",
        "split": "test",
        "ap": {"0.5": 0.72, "0.95": 0.01},
    },
]


def test_result_charts_are_written(tmp_path):
    plot_model_comparison(TEST_METRICS, tmp_path / "comparison.png")
    plot_latency(LATENCY, tmp_path / "latency.png")
    plot_ap_per_iou(CURVES, tmp_path / "ap_per_iou.png")

    for name in ("comparison.png", "latency.png", "ap_per_iou.png"):
        assert (tmp_path / name).stat().st_size > 0
