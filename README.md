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

Open **http://localhost:8080**, create a local account (first run only), then:

1. **Create project** → name it (e.g. `billboard-detection`).
2. **Settings → Labeling Interface** → start from the *Object Detection with Bounding Boxes* template. Set the single label to `billboard`.
3. **Settings → Cloud Storage → Add Source Storage** → type *Local files*, path `/label-studio/files/data/raw`. In Import Settings, set **Import Method** to treat every file as a source file (not "Tasks/JSON") and filter to image extensions, e.g. `.*\.(jpg|jpeg|png)$` → **Sync Storage**. This pulls in the raw images without copying them.
4. Push the auto-labels as **predictions** onto the synced tasks (don't use Label Studio's built-in COCO import — it creates duplicate tasks instead of attaching boxes to the ones Local Storage already created):
   ```bash
   export LABEL_STUDIO_API_KEY=<your access/refresh token, from Account & Settings>
   python scripts/push_predictions_to_label_studio.py --project-id <id>
   ```
   Opening a task now shows the Grounding DINO boxes pre-drawn.
5. Review each task: correct/add/remove boxes, submit.
6. **Data Manager → Export** → format *COCO* → download the zip.
7. Import the export into the project (normalizes file names, copies images into `data/processed/`):
   ```bash
   python scripts/import_label_studio_export.py --export-path ~/Downloads/<export>.zip
   ```
8. Split into train/val/test (writes `data/annotations/reviewed/{train,val,test}.json`, matching `configs/model/rtdetr.yaml`):
   ```bash
   python scripts/split_reviewed_dataset.py
   ```
   Note: with a very small reviewed set, a fixed split isn't statistically meaningful — the script warns and still runs so the pipeline can be exercised end-to-end; re-run once you have more reviewed images, or switch to k-fold cross-validation.

Stop the container with `docker stop vit-billboard-label-studio` (state persists in `.label-studio/data`, ignored by git); remove it with `docker rm vit-billboard-label-studio`.
