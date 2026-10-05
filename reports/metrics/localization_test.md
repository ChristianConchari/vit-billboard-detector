# Localization diagnostics (test)

## AP per IoU threshold (all areas, up to 100 detections)

| Model | Split | 0.50 | 0.55 | 0.60 | 0.65 | 0.70 | 0.75 | 0.80 | 0.85 | 0.90 | 0.95 |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| rtdetr-20261005-004038 | test | 0.93 | 0.93 | 0.93 | 0.89 | 0.87 | 0.79 | 0.66 | 0.35 | 0.13 | 0.00 |
| rtdetr-20261005-004038 | val | 0.93 | 0.93 | 0.93 | 0.92 | 0.90 | 0.84 | 0.79 | 0.75 | 0.66 | 0.51 |
| Grounding DINO zero-shot | test | 0.72 | 0.72 | 0.72 | 0.69 | 0.65 | 0.59 | 0.43 | 0.21 | 0.06 | 0.01 |

## Edge bias against the test ground truth

Signed edge offsets normalized by the ground-truth box size: positive means the predicted edge lies outside the ground-truth box (box too large), negative inside (box too small). Models use their calibrated score threshold.

| Boxes | Left | Top | Right | Bottom | Width ratio | Height ratio | Median IoU | Matches |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| rtdetr-20261005-004038 | -0.022 | -0.030 | -0.026 | -0.039 | 0.952 | 0.931 | 0.859 | 122 |
| Grounding DINO pre-labels | -0.026 | -0.041 | -0.035 | -0.049 | 0.938 | 0.910 | 0.859 | 90 |
