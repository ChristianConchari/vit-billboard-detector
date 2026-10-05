# Experiment record

9 tracked runs evaluated on the frozen test split (fingerprint `736070363e3635fb`). Full per-run data: `runs.csv`.

## Learning curve (backbone frozen, seed 42)

| Train images | RT-DETR mAP | AP50 | AP75 | Grounding DINO mAP | Run |
|--:|--:|--:|--:|--:|---|
| 66 | 0.546 | 0.784 | 0.657 | 0.481 | `rtdetr-20261001-232751` |
| 205 | 0.637 | 0.922 | 0.779 | 0.481 | `rtdetr-20261004-200415` |
| 346 | 0.642 | 0.927 | 0.777 | 0.481 | `rtdetr-20261004-204025` |
| 441 | 0.637 | 0.928 | 0.778 | 0.481 | `rtdetr-20261004-221908` |

## Backbone ablation (441 training images, mean and range over seeds)

| Backbone | Runs | mAP | AP50 | AP75 |
|---|--:|--:|--:|--:|
| frozen | 3 | 0.643 (0.637–0.653) | 0.929 (0.925–0.933) | 0.784 (0.778–0.794) |
| unfrozen | 3 | 0.648 (0.645–0.649) | 0.934 (0.930–0.938) | 0.783 (0.777–0.794) |
