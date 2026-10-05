# vit-billboard-detector

[![CI](https://github.com/ChristianConchari/vit-billboard-detector/actions/workflows/ci.yml/badge.svg)](https://github.com/ChristianConchari/vit-billboard-detector/actions/workflows/ci.yml)

Vision Transformer-based object detection system for identifying **out-of-home (OOH) advertising billboards** in images. Final project for the **Computer Vision II** course at MIA-CEIA (FIUBA).

## Problem

Detect billboards / OOH advertising signage in real-world images, under a tight constraint: a small pool of unlabeled images and limited time to annotate them.

## Strategy

Two-stage pipeline designed around data scarcity:

1. **Auto-labeling** — [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) (open-set, text-prompted detection) generates candidate bounding boxes on unlabeled images ("billboard", "advertising sign", "OOH ad"). Boxes are manually reviewed/corrected instead of labeled from scratch.
2. **Fine-tuning** — [RT-DETR](https://github.com/lyuwenyu/RT-DETR) (COCO-pretrained) is fine-tuned on the resulting dataset. Chosen over vanilla DETR for faster convergence with limited data, given the lack of inductive bias in pure Transformer detectors.

Evaluation compares **Grounding DINO zero-shot** vs. **RT-DETR fine-tuned** on the same held-out ground truth (mAP, AP50, AP75, AP by object size).

## Project structure

```
.
├── configs/
│   ├── pipeline.yaml          # end-to-end run: data paths, split, calibration, latency, figures
│   └── model/                 # rtdetr.yaml (training/eval), grounding_dino.yaml (auto-labeling)
├── data/                      # not tracked by git
│   ├── raw/                   # full image pool (input to auto-labeling and the split)
│   ├── interim/               # extracted Label Studio exports
│   ├── processed/             # reviewed images used for training/evaluation
│   └── annotations/
│       ├── auto/              # Grounding DINO pre-labels (COCO)
│       └── reviewed/          # reviewed COCO labels, train/val/test splits, split_assignment.json
├── src/
│   ├── interfaces/cli/        # billboard-detect: end-user CLI for images and videos
│   └── vit/
│       ├── data/              # Label Studio import, split by video, PyTorch dataset
│       ├── labeling/          # Grounding DINO auto-labeling, Label Studio sync
│       ├── models/            # RT-DETR loading, backbone freezing, head init
│       ├── train/             # fine-tuning loop
│       ├── eval/              # COCO metrics, calibration, latency, localization, figures, learning curve
│       ├── inference/         # Detector protocol, RT-DETR adapter, image/video runner
│       ├── transforms/        # box-aware augmentations
│       ├── utils/             # config, logging, MLflow tracking
│       └── pipeline.py        # one-command run, from export to report
├── scripts/                   # thin CLIs over src/vit (pipeline, training, evaluation, analysis)
├── tests/
│   ├── unit/                  # fast tests per module
│   └── integration/           # end-to-end runs with a tiny RT-DETR, no downloads
├── docs/decisions/            # architecture decision records (ADRs)
├── reports/                   # metrics, run reports and figures (generated)
├── checkpoints/               # fine-tuned weights (not tracked by git)
└── logs/                      # run logs (not tracked by git)
```

## Tooling

- PyTorch / torchvision (training loop, box-aware augmentations)
- Hugging Face `transformers` (RT-DETR, Grounding DINO)
- pycocotools (COCO mAP), OpenCV (video I/O), matplotlib (attention colormaps)
- MLflow for experiment tracking (SQLite backend, `mlflow.db`)
- Label Studio for box review
- pytest for testing

Dependencies are pinned in `requirements.txt` to the versions the pipeline was validated with (Python 3.12).

## Deliverables (course requirements)

- Functional, modular codebase (pre-production level)
- Technical report (PDF): objective, architecture, implementation, evaluation, results, conclusions, team planning
- This README
- 15-minute final presentation (class 8)

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e . -r requirements.txt
pytest -q   # unit + integration tests, no model downloads or GPU needed (~25 s)
```

Run every command from the repository root: config and data paths are relative to it.

### Code quality and CI

[Ruff](https://docs.astral.sh/ruff/) formats and lints the code (settings in `pyproject.toml`: line length 100, pyflakes, pycodestyle, bugbear, import sorting, pyupgrade):

```bash
ruff format src scripts tests
ruff check src scripts tests
```

GitHub Actions (`.github/workflows/ci.yml`) runs on every push to `main` and on pull requests, with two jobs:
- **lint:** `ruff check` and `ruff format --check`;
- **test:** installs the pinned dependencies (CPU-only PyTorch, no GPU needed) and runs `pytest`: unit tests per module, plus integration tests that run the whole pipeline and the `billboard-detect` CLI end to end with a tiny randomly initialized RT-DETR.

## Design decisions

Architecture decision records live in [`docs/decisions/`](docs/decisions/).

## Labeling workflow

1. Drop unlabeled images into `data/raw/`.
2. Generate candidate boxes:
   ```bash
   python scripts/run_auto_labeling.py
   ```
   Output: `data/annotations/auto/auto_labels.json` (COCO format, NMS-deduplicated, includes a confidence `score` per box). These are proposals, not ground truth.
3. Review/correct the boxes in **Label Studio** (see below), then export the corrected COCO file(s) into `data/annotations/reviewed/`.

### Running Label Studio (local, via Docker)

```bash
docker run -d \
  --name vit-billboard-label-studio \
  -p 8080:8080 \
  -v "$(pwd)/.label-studio/data:/label-studio/data" \
  -v "$(pwd)/data:/label-studio/files/data:ro" \
  -e LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true \
  -e LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/label-studio/files \
  heartexlabs/label-studio:latest
```

If the container already exists (e.g. after a reboot), restart it with `docker start vit-billboard-label-studio` instead.

Open **http://localhost:8080**, create a local account (first run only), then:

1. **Create project** → name it (e.g. `billboard-detection`).
2. **Settings → Labeling Interface** → start from the *Object Detection with Bounding Boxes* template. Set the single label to `billboard`. Keep the template's tag names (`<Image name="image">`, `<RectangleLabels name="label">`): the push script targets them.
3. **Settings → Cloud Storage → Add Source Storage** → type *Local files*, path `/label-studio/files/data/raw`. In Import Settings, set **Import Method** to treat every file as a source file (not "Tasks/JSON") and filter to image extensions, e.g. `.*\.(jpg|jpeg|png)$`, and leave **Recursive scan** off so only top-level files in `data/raw/` are imported → **Sync Storage**. This pulls in the raw images without copying them. Syncing only adds tasks: if files were moved or deleted from `data/raw/`, delete their stale tasks in the Data Manager.
4. Push the auto-labels as **predictions** onto the synced tasks (don't use Label Studio's built-in COCO import — it creates duplicate tasks instead of attaching boxes to the ones Local Storage already created):
   ```bash
   export LABEL_STUDIO_API_KEY=<your access/refresh token, from Account & Settings>
   python scripts/push_predictions_to_label_studio.py --project-id <id> --exclude-split test
   ```
   Opening a task now shows the Grounding DINO boxes pre-drawn. The script isn't idempotent: running it again adds a second set of predictions, so delete the existing ones first (Data Manager → select tasks → *Delete Predictions*).
5. Review each task: correct/add/remove boxes, submit. Test videos are the exception: label them from scratch (see [Labeling the test split](#labeling-the-test-split)).
6. **Data Manager → Export** → format *COCO* → download the zip.
7. Import the export into the project (normalizes file names, copies images into `data/processed/`):
   ```bash
   python scripts/import_label_studio_export.py --export-path ~/Downloads/<export>.zip
   ```
8. Split into train/val/test by video (writes `data/annotations/reviewed/{train,val,test}.json`, matching `configs/model/rtdetr.yaml`):
   ```bash
   python scripts/split_reviewed_dataset.py
   ```
   Whole videos are assigned to one split ([ADR 0002](docs/decisions/0002-split-by-video.md)). The first run builds the assignment from the full image pool in `data/raw/` (≈70/15/15 by image count) and saves it to `data/annotations/reviewed/split_assignment.json`. Later runs reuse it, so you can review incrementally: export again, re-run steps 7–8, and every video stays in its split. Pass `--rebuild-assignment` only when new videos are added to the pool. A split with no reviewed images yet is written empty and logged as a warning.

Stop the container with `docker stop vit-billboard-label-studio` (state persists in `.label-studio/data`, ignored by git); remove it with `docker rm vit-billboard-label-studio`.

### Labeling the test split

Test images are annotated by hand, without Grounding DINO pre-labels, so the zero-shot baseline isn't scored against its own boxes ([ADR 0003](docs/decisions/0003-label-test-from-scratch.md)). The test videos are the keys with value `"test"` in `data/annotations/reviewed/split_assignment.json`.

1. Push pre-labels with `--exclude-split test`, as in step 4, so test tasks don't get them.
2. If test tasks already have predictions or annotations from an earlier push, filter them in the Data Manager by file name (each test video id), select them, and run **Actions → Delete predictions** and **Actions → Delete annotations**.
3. Label every image of every test video, including images without billboards (submit them empty).
4. From then on, don't edit the test videos: every pipeline run must see the same test split, and its content fingerprint (file names and boxes, independent of the ids Label Studio renumbers on every export) is tracked in MLflow as `data.test_fingerprint`.

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
