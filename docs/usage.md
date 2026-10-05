# Usage: pipeline, scripts and demo

Run every command from the repository root, with the virtual environment active.

## Training and evaluation

### One-command run

Once the review in Label Studio is exported, one command rebuilds every result:

```bash
python scripts/run_pipeline.py --export-path ~/Downloads/<export>.zip
```

It imports the export, splits it by video, fine-tunes RT-DETR, calibrates its score threshold on val (best F1 at IoU 0.5, saved as `calibration.json` next to the checkpoint), evaluates RT-DETR and the Grounding DINO baseline on test, benchmarks latency, and renders figures. Everything lands in `reports/runs/<run>/`:

- `results.md`: dataset, test metrics, calibrated threshold and latency tables, ready for the README and the report;
- `summary.json`: the same numbers, machine-readable;
- `figures/`: prediction comparisons and attention maps (not tracked by git; publish only a few hand-picked frames).

Without `--export-path` it reuses the dataset already imported. `--checkpoint checkpoints/rtdetr/<run>/best` skips training and evaluates that model. `--skip-figures` saves a few minutes. `--set KEY=VALUE` (repeatable) overrides any RT-DETR config value for that run, e.g. `--set training.freeze_backbone=false --set training.seed=1`; the effective config is what gets logged to MLflow. Settings live in `configs/pipeline.yaml` and `configs/model/*.yaml`. The run stops early if a split has no reviewed images.

### Step by step

The same steps can be run one at a time:

1. Fine-tune RT-DETR on `data/annotations/reviewed/train.json` (settings in `configs/model/rtdetr.yaml`):
   ```bash
   python scripts/train_rtdetr.py
   ```
   Each epoch is validated with COCO mAP on `val.json`. The best epoch is saved to `checkpoints/rtdetr/<run>/best`, and the run is tracked in MLflow (see [Experiment tracking](#experiment-tracking)).
2. Evaluate the fine-tuned model and the zero-shot baseline on the same split:
   ```bash
   python scripts/evaluate_detector.py --model rtdetr --checkpoint checkpoints/rtdetr/<run>/best --split test
   python scripts/evaluate_detector.py --model grounding-dino --split test
   ```
   Metrics (mAP, AP50, AP75, AP by object size; `-1` means there are no ground-truth objects in that size range) are written to `reports/metrics/<model>_<split>.json`. These standalone evaluation, latency and figure scripts are for quick checks and don't create MLflow runs; tracked results come from the one-command run.

   **Caveat:** the reviewed ground truth starts from Grounding DINO proposals, and boxes accepted unchanged match them exactly. This biases the comparison in favor of the zero-shot baseline, especially AP75.
3. Inspect predictions qualitatively (green = ground truth, red = prediction):
   ```bash
   python scripts/visualize_predictions.py --checkpoint checkpoints/rtdetr/<run>/best --rtdetr-threshold 0.15
   python scripts/visualize_attention.py --checkpoint checkpoints/rtdetr/<run>/best
   ```
   The attention figures show, for each RT-DETR detection, the encoder self-attention from the token under the box center and the decoder's deformable sampling points, sized by weight. Frames are cropped at the bottom to remove the camera's timestamp/GPS overlay (`--overlay-fraction`). Figures are written to `reports/figures/` and are not tracked by git.
4. Benchmark end-to-end inference latency (batch size 1, preprocessing included):
   ```bash
   python scripts/benchmark_latency.py --checkpoint checkpoints/rtdetr/<run>/best
   ```
   Results go to `reports/metrics/latency_<split>.json`. The Grounding DINO pipeline runs one forward pass per text prompt, so its latency grows with the number of prompts.

### Localization diagnostics

```bash
python scripts/analyze_localization.py \
    --checkpoint checkpoints/rtdetr/<run-a>/best --checkpoint checkpoints/rtdetr/<run-b>/best
```

Writes `reports/metrics/localization_test.md` with AP at every IoU threshold (0.50–0.95) on test and val, and the systematic edge bias of each model's boxes against the test ground truth, next to the bias of the Grounding DINO pre-labels. It shows where mAP is lost: detection at IoU 0.5 is near its ceiling, while boxes trained on corrected pre-labels inherit Grounding DINO's tighter style and diverge from the hand-drawn test boxes at high IoU.

### Attention analysis

```bash
python scripts/analyze_attention.py --checkpoint checkpoints/rtdetr/<run>/best
```

Writes `reports/metrics/attention_test.json`. For each detection matched to a billboard in frames with two or more billboards, it compares the encoder attention of the token under the box center that falls on the *other* billboards with the share of the image they cover (1.0 = chance), and the same for its own billboard.

### Experiment tracking

MLflow uses a local SQLite backend (`mlflow.db`, with artifacts under `mlruns/`; neither is tracked by git). Browse runs with:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Each `run_pipeline.py` execution is **one MLflow run** with:
- **tags** for reproducibility: `git.commit`, `git.dirty` (uncommitted changes when it ran), `stage`, SHA-256 hashes of the reviewed annotations, the split assignment and the Label Studio export, and a content fingerprint per split (`data.<split>_fingerprint`);
- **params**: every setting of the three configs (`pipeline.*`, `rtdetr.*`, `grounding_dino.*`) and images/boxes per split (`data.<split>.*`);
- **metrics**: `train/loss` and `val/*` per epoch, `val/best_mAP`, `calibration/*`, `test/<model>/*` and `latency/<model>/*`;
- **artifacts**: the effective configs (`configs/`), `results.md`, `summary.json` and `calibration.json` (`reports/`). The checkpoint itself is logged only if `tracking.log_checkpoint` is enabled in `configs/pipeline.yaml`, because it weighs ~170 MB.

In the MLflow UI, the runs table, the per-epoch charts and the run page with its reproducibility tags look like this:

![Runs table](../reports/figures/mlflow/runs.png)
![Validation mAP per epoch, ablation runs](../reports/figures/mlflow/val_map_ablation.png)
![Final model's run](../reports/figures/mlflow/final_run.png)

`scripts/train_rtdetr.py` creates a training-only run with the same naming. A pipeline run with `--checkpoint` is named `<run>-evaluation`.

### Exporting the experiment record

`mlflow.db` stays local, so the record is exported to versionable files:

```bash
python scripts/export_experiments.py
```

This writes `reports/experiments/` with:
- `runs.csv`: one row per finished run, with config, test metrics, calibrated threshold, latency, `git.commit` and data fingerprints;
- `summary.md`: the learning curve and the backbone ablation, with mean and range over seeds;
- `training_curves.png`: validation mAP per epoch for every run at the largest training-set size.

Only finished runs evaluated on the current test split (matched by content fingerprint) are included, so runs scored on an earlier, biased test split never mix in.

### Learning curve

To show how much labeling RT-DETR needs to match the zero-shot baseline, run the pipeline after each labeling batch with the same note, then plot:

```bash
python scripts/run_pipeline.py --export-path ~/Downloads/<export>.zip --note learning-curve
python scripts/plot_learning_curve.py --note learning-curve
```

This writes `reports/figures/learning_curve.png` and the same points as `learning_curve.csv`: test mAP against the number of training images, with the Grounding DINO zero-shot mAP as a reference line. Runs repeated at one size (e.g. different seeds) are averaged, and their min–max range is drawn. The script refuses runs evaluated on different test splits. Commit before each run, so every point is tied to a clean `git.commit`.

## Demo: detect billboards in images or video

`billboard-detect` (installed by `pip install -e .`) runs a fine-tuned RT-DETR on an image, a folder of images or a video. It needs a checkpoint from [Training and evaluation](#training-and-evaluation); checkpoints are not tracked by git.

```bash
billboard-detect path/to/drive.mp4 --checkpoint checkpoints/rtdetr/<run>/best
billboard-detect path/to/images/ --checkpoint checkpoints/rtdetr/<run>/best --score-threshold 0.3
billboard-detect --help
```

| Option | Default | Meaning |
|---|---|---|
| `--score-threshold` | the checkpoint's `calibration.json`, else `inference.score_threshold` in `configs/model/rtdetr.yaml` | Minimum score to keep a box |
| `--overlay-fraction` | `0.08` | Bottom fraction of each frame cropped *before* detection, to drop the camera's timestamp/GPS overlay; use `0` for footage without it |
| `--output-dir` | `outputs/detections` | Where results are written (not tracked by git) |

Output:
- annotated copies of the images, or `<name>_detections.mp4` for a video;
- `detections.json`, with one record per image or frame:
  ```json
  {"source": "drive.mp4", "frame": 12, "detections": [{"box_xyxy": [609.7, 367.7, 918.6, 735.3], "score": 0.27}]}
  ```
  `frame` is `null` for images. Boxes are in pixels of the cropped frame.

### Trying it out

1. Unit tests for the demo (image folders, a synthetic video, an unreadable file):
   ```bash
   pytest -q tests/unit/test_media.py
   ```
2. A single image, then open the annotated copy:
   ```bash
   billboard-detect data/processed/<image>.jpg --checkpoint checkpoints/rtdetr/<run>/best
   xdg-open outputs/detections/<image>.jpg
   ```
3. A video. Any dashcam or phone clip recorded from a car works; add `--overlay-fraction 0` if it has no timestamp/GPS band:
   ```bash
   billboard-detect path/to/drive.mp4 --checkpoint checkpoints/rtdetr/<run>/best --output-dir outputs/demo_video
   xdg-open outputs/demo_video/drive_detections.mp4
   ```
4. Threshold trade-off: a higher threshold gives fewer false positives but misses more billboards:
   ```bash
   billboard-detect data/processed --checkpoint checkpoints/rtdetr/<run>/best --score-threshold 0.25 --output-dir outputs/th025
   ```
5. Error handling: these should exit with a clear message, without a traceback:
   ```bash
   billboard-detect missing.jpg --checkpoint checkpoints/rtdetr/<run>/best
   billboard-detect docs/ --checkpoint checkpoints/rtdetr/<run>/best
   billboard-detect README.md --checkpoint checkpoints/rtdetr/<run>/best
   ```

What to check: boxes sit on billboards, no timestamp/GPS overlay is left in the outputs, and `detections.json` has one record per image or frame.

To judge real quality, use images from the test split (`data/annotations/reviewed/test.json`) or new footage. `data/processed/` also contains training images, where the model looks better than it is.
