# vit-billboard-detector

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
├── configs/                # env + model configuration (yaml)
├── data/
│   ├── raw/                 # original, unlabeled images
│   ├── interim/              # intermediate/preprocessed data
│   ├── processed/            # final datasets ready for training
│   └── annotations/
│       ├── auto/              # Grounding DINO auto-generated labels
│       └── reviewed/          # manually corrected, final COCO-format labels
├── src/
│   ├── interfaces/cli/       # CLI entry points
│   └── vit/
│       ├── data/              # dataset loaders
│       ├── labeling/          # auto-labeling pipeline (Grounding DINO)
│       ├── models/            # model definitions / wrappers
│       ├── train/             # training loop
│       ├── eval/               # metrics (mAP, AP50, AP75...)
│       ├── inference/          # inference scripts
│       ├── transforms/         # augmentations, preprocessing
│       └── utils/              # logging, config, shared helpers
├── notebooks/               # exploration notebooks
├── scripts/                 # standalone automation scripts
├── experiments/             # experiment configs/outputs
├── reports/figures/         # generated plots/visualizations
├── checkpoints/             # trained model weights
├── logs/                    # local run logs
├── docs/                    # technical documentation
└── tests/                   # unit / integration / e2e tests
```

## Tooling

- PyTorch / torchvision / timm
- Hugging Face `transformers` (RT-DETR, Grounding DINO)
- MLflow for experiment tracking
- pytest for testing

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
```

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
   python scripts/push_predictions_to_label_studio.py --project-id <id>
   ```
   Opening a task now shows the Grounding DINO boxes pre-drawn. The script isn't idempotent: running it again adds a second set of predictions, so delete the existing ones first (Data Manager → select tasks → *Delete Predictions*).
5. Review each task: correct/add/remove boxes, submit.
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

## Training and evaluation

1. Fine-tune RT-DETR on `data/annotations/reviewed/train.json` (settings in `configs/model/rtdetr.yaml`):
   ```bash
   python scripts/train_rtdetr.py
   ```
   Each epoch is validated with COCO mAP on `val.json`. The best epoch is saved to `checkpoints/rtdetr/<run>/best`, and params and metrics go to MLflow (`mlflow ui --backend-store-uri sqlite:///mlflow.db`).
2. Evaluate the fine-tuned model and the zero-shot baseline on the same split:
   ```bash
   python scripts/evaluate_detector.py --model rtdetr --checkpoint checkpoints/rtdetr/<run>/best --split test
   python scripts/evaluate_detector.py --model grounding-dino --split test
   ```
   Metrics (mAP, AP50, AP75, AP by object size; `-1` means there are no ground-truth objects in that size range) are written to `reports/metrics/<model>_<split>.json` and logged to MLflow.

   **Caveat:** the reviewed ground truth starts from Grounding DINO proposals, and boxes accepted unchanged match them exactly. This biases the comparison in favor of the zero-shot baseline, especially AP75.
