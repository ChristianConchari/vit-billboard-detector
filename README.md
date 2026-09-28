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
pip install -r requirements.txt
```
