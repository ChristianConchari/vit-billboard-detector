# 0002. Split train/val/test by video, not by image

- **Status:** Accepted
- **Date:** 2026-10-01

## Context

The images are consecutive frames from 26 videos, so frames from the same video are nearly identical. A random split by image puts near-duplicates in both train and test. This leaks training data into evaluation and inflates mAP.

## Decision

Group images by video (the `<video_id>` prefix in each file name) and assign whole videos to a single split. The assignment is computed once over the full image pool, balancing image counts, and reused for every partial export.

## Consequences

- Test metrics measure generalization to unseen scenes.
- With only 26 videos, splits are coarse and image counts per split can't be exact. If the variance turns out too high, consider grouped k-fold.
- Splits stay stable while labeling progresses: reviewing more images never moves a video between splits.
