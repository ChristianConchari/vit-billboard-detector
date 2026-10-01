# 0001. Discard the labels shipped with the source dataset

- **Status:** Accepted
- **Date:** 2026-10-01

## Context

The source dataset (632 video frames, 26 videos) ships with COCO boxes. Each box has a review status:

- 72% of boxes are still unreviewed.
- 3 boxes marked as rejected are included as positives.
- 30 of the 57 test images contain unreviewed boxes.

Ground truth this noisy would make the evaluation metrics unreliable.

## Decision

Ignore the provided labels. Treat the images as unlabeled and run them through the project's own pipeline: Grounding DINO auto-labeling, then manual review in Label Studio.

## Consequences

- All ground truth goes through a single, documented review process.
- Manual review effort grows: every image needs a review pass.
