# vit-billboard-detector

[![CI](https://github.com/ChristianConchari/vit-billboard-detector/actions/workflows/ci.yml/badge.svg)](https://github.com/ChristianConchari/vit-billboard-detector/actions/workflows/ci.yml)

Detection of **out-of-home (OOH) advertising billboards** in street-level video frames, with Transformer-based detectors. Final project for **Computer Vision III** (MIA-CEIA, FIUBA).

The challenge is the data: 632 unlabeled frames and little time to annotate them. The project therefore works in two stages:

1. **Auto-labeling.** [Grounding DINO](https://arxiv.org/abs/2303.05499), an open-vocabulary detector, proposes boxes from text prompts ("billboard", "advertising sign", "ooh ad"). A human only reviews and corrects them.
2. **Fine-tuning.** [RT-DETR](https://arxiv.org/abs/2304.08069), a real-time DETR pretrained on COCO, is fine-tuned on the reviewed boxes and compared with Grounding DINO used zero-shot.

![Demo: RT-DETR detections on consecutive frames](reports/figures/examples/demo.gif)

## Results

Evaluated on **95 frames from 7 videos never seen in training**, labeled by hand from scratch so that neither model influenced the ground truth.

| Model | mAP@[.5:.95] | AP50 | AP75 | Speed (RTX 4070 SUPER) |
|---|--:|--:|--:|--:|
| **RT-DETR fine-tuned** | **0.649** | **0.930** | **0.794** | **51 FPS** (19.5 ms) |
| Grounding DINO zero-shot | 0.481 | 0.723 | 0.588 | 2 FPS (471 ms) |

- **Fine-tuning wins on both accuracy and speed.** RT-DETR finds 93% of the billboards (AP50) and runs 24× faster, fast enough for real-time video. Grounding DINO is useful only offline, to pre-label data. Over 3 training seeds, RT-DETR scores 0.648 mAP (range 0.645–0.649).

  ![Test accuracy: fine-tuned vs. zero-shot](reports/figures/model_comparison.png)

- **About 200 labeled images are enough.** RT-DETR already beats the zero-shot model with 66 training images and stops improving after ~200.

  ![Learning curve](reports/figures/learning_curve.png)

- **The remaining error is box tightness, not detection.** Both models detect almost every billboard, but their boxes are 5–9% smaller than the hand-drawn ones. RT-DETR learned this style from the Grounding DINO pre-labels it was trained on. This lowers mAP, which demands very tight boxes, while AP50 barely changes.

  ![AP per IoU threshold](reports/figures/ap_per_iou.png)

- **Unfreezing the ResNet-50 backbone does not help.** It changes mAP by less than the variation between seeds ([training curves](reports/experiments/training_curves.png)).

### Examples

Green: ground truth. Red: predictions with their confidence. Left: RT-DETR; right: Grounding DINO.

| | |
|---|---|
| Several billboards: RT-DETR finds all three; Grounding DINO also marks a shop sign. | ![Several billboards](reports/figures/examples/multiple_billboards.jpg) |
| A distant billboard, the typical case. | ![Distant billboard](reports/figures/examples/distant_billboard.jpg) |
| A shared mistake: a glass facade taken for a billboard. | ![False positive on a facade](reports/figures/examples/false_positive_facade.jpg) |

### What the Transformer looks at

![Attention maps](reports/figures/examples/attention_maps.jpg)

Left to right: a detection, the **encoder self-attention** from the token at the center of the billboard, and the **decoder's deformable attention**, which reads only a few points per object (dot size = weight). The decoder focuses on the billboard and its edges. Measured over the test set, the encoder pays 1.6× more attention to the *other* billboards in the frame than chance would give: self-attention links similar regions across the whole image.

Full numbers: [`reports/experiments/summary.md`](reports/experiments/summary.md) (all tracked runs) and [`reports/metrics/`](reports/metrics/) (localization and attention analysis).

## How it works

```mermaid
flowchart LR
    raw["Raw video frames<br/>(632 images, 26 videos)"]

    subgraph labeling["1 · Labeling"]
        gdino["Grounding DINO<br/>zero-shot proposals"]
        review["Label Studio<br/>human review"]
        scratch["Test videos<br/>drawn from scratch"]
    end

    subgraph data["2 · Dataset"]
        import["Import COCO export"]
        split["Split by video<br/>(frozen assignment)"]
    end

    subgraph training["3 · Training"]
        rtdetr["RT-DETR fine-tuning<br/>(COCO-pretrained)"]
        calib["Best epoch + score<br/>threshold on val"]
    end

    subgraph evaluation["4 · Evaluation"]
        eval["COCO metrics on test<br/>RT-DETR vs. Grounding DINO"]
        diag["Latency, localization,<br/>attention maps"]
    end

    tracking[("MLflow<br/>runs, tags, artifacts")]
    report["reports/<br/>results, figures"]
    cli["billboard-detect<br/>images and video"]

    raw --> gdino --> review
    raw --> scratch
    review --> import
    scratch --> import
    import --> split --> rtdetr --> calib --> eval --> diag --> report
    calib --> cli
    rtdetr -.-> tracking
    eval -.-> tracking
    diag -.-> tracking
```

1. **Labeling.** Grounding DINO proposes boxes; they are corrected in Label Studio. The test videos are drawn from scratch instead.
2. **Dataset.** Frames are split into train/val/test **by video**, so near-identical consecutive frames never leak from training into evaluation.
3. **Training.** RT-DETR is fine-tuned for one class; the best epoch and the score threshold are chosen on validation.
4. **Evaluation.** Both models are scored with COCO metrics on the same test split; latency, localization errors and attention are analyzed.

One command runs steps 2 to 4 and records every run in MLflow, with the code version, the data version and all settings. Details of each component, including RT-DETR's internals, are in [`docs/architecture/`](docs/architecture/README.md).

### Experiment tracking

Every run of the pipeline is an MLflow run. The table below lists the learning-curve and ablation runs with their settings and scores; each run also stores the git commit and a fingerprint of every data split, so any number can be traced back to the exact code and data that produced it ([final model's run](reports/figures/mlflow/final_run.png)).

![MLflow runs](reports/figures/mlflow/runs.png)

The [validation curves](reports/experiments/training_curves.png) show that four of the six runs with 441 training images lose validation mAP after epoch ~10; keeping the best validation epoch protects the final checkpoints from that overfitting. The [MLflow view of the ablation](reports/figures/mlflow/val_map_ablation.png) shows five of them: the sixth, frozen with seed 42, is tracked as the last point of the learning curve.

## Getting started

Tested with Python 3.12. The tests run on CPU and download no model:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e . -r requirements.txt
pytest -q
```

### Try the trained model

The final checkpoint (164 MB) is published in the
[`model-v1` release](https://github.com/ChristianConchari/vit-billboard-detector/releases/tag/model-v1).
It runs on GPU or CPU.

1. Install the project as shown above.
2. Download the weights. The script verifies the SHA-256 and unpacks them into
   `checkpoints/rtdetr-billboard/`:

   ```bash
   python scripts/download_model.py
   ```

   To do it by hand instead, download `rtdetr-billboard.zip` from the release
   and unzip it into `checkpoints/`.
3. Run the detector on an image, a folder of images or a video:

   ```bash
   billboard-detect path/to/street.jpg --checkpoint checkpoints/rtdetr-billboard --overlay-fraction 0
   billboard-detect path/to/frames/    --checkpoint checkpoints/rtdetr-billboard --overlay-fraction 0
   billboard-detect path/to/drive.mp4  --checkpoint checkpoints/rtdetr-billboard --overlay-fraction 0
   ```

Annotated copies and a `detections.json` file (boxes in pixels, `xyxy`, with
scores) are written to `outputs/detections/`. Useful options:

| Option | Default | Meaning |
|--|--|--|
| `--score-threshold` | 0.15, calibrated on validation and stored with the checkpoint | Minimum score to keep a box |
| `--overlay-fraction` | 0.08 | Bottom part of each frame to crop. The dataset cameras print a timestamp there; use `0` for other images |
| `--output-dir` | `outputs/detections` | Where to write the results |

### Reproduce the results

The dataset is private and not included. With a Label Studio export in place:

```bash
python scripts/run_pipeline.py --export-path <export>.zip
```

## Repository structure

```
configs/        model and pipeline settings (YAML)
src/vit/        library: data, models, training, evaluation, inference, pipeline
src/interfaces/ billboard-detect command-line tool
scripts/        command-line entry points (pipeline, training, evaluation, figures)
tests/          unit and end-to-end tests (run in CI)
docs/           architecture, design decisions, guides
reports/        results, figures and the exported experiment record
```

## Documentation

- [Architecture](docs/architecture/README.md): system flow, RT-DETR internals, components.
- [Design decisions](docs/decisions/): why the provided labels were discarded, why the split is by video, why the test set was labeled from scratch.
- [Usage](docs/usage.md): the pipeline, analysis scripts, MLflow tracking, the demo.
- [Labeling workflow](docs/labeling.md): pre-labeling and review in Label Studio.
- [Development](docs/development.md): setup, code style, tests and CI.
