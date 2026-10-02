# 0003. Label the test split from scratch, without pre-labels

- **Status:** Accepted
- **Date:** 2026-10-01

## Context

Reviewed boxes start from Grounding DINO proposals. Boxes accepted unchanged match those proposals exactly: in the first test images, 67% of ground-truth boxes had IoU > 0.98 with a proposal. This biases the comparison in favor of the zero-shot baseline, especially AP75.

The test split also kept growing with every partial export, so runs weren't comparable over time.

## Decision

- Annotate every image of the test videos by hand, with Grounding DINO pre-labels hidden or deleted.
- Freeze the test split once it is complete. Later labeling only adds train and val images.

## Consequences

- RT-DETR and Grounding DINO are measured against ground truth that neither model influenced.
- Runs share one test split (tracked by its SHA-256), so a learning curve over training-set size is valid.
- Labeling the test videos takes longer than correcting proposals.
