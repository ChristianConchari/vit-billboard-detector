# Labeling workflow


1. Drop unlabeled images into `data/raw/`.
2. Generate candidate boxes:
   ```bash
   python scripts/run_auto_labeling.py
   ```
   Output: `data/annotations/auto/auto_labels.json` (COCO format, NMS-deduplicated, includes a confidence `score` per box). These are proposals, not ground truth.
3. Review/correct the boxes in **Label Studio** (see below), then export the corrected COCO file(s) into `data/annotations/reviewed/`.

## Running Label Studio (local, via Docker)

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
   Whole videos are assigned to one split ([ADR 0002](decisions/0002-split-by-video.md)). The first run builds the assignment from the full image pool in `data/raw/` (≈70/15/15 by image count) and saves it to `data/annotations/reviewed/split_assignment.json`. Later runs reuse it, so you can review incrementally: export again, re-run steps 7–8, and every video stays in its split. Pass `--rebuild-assignment` only when new videos are added to the pool. A split with no reviewed images yet is written empty and logged as a warning.

Stop the container with `docker stop vit-billboard-label-studio` (state persists in `.label-studio/data`, ignored by git); remove it with `docker rm vit-billboard-label-studio`.

## Labeling the test split

Test images are annotated by hand, without Grounding DINO pre-labels, so the zero-shot baseline isn't scored against its own boxes ([ADR 0003](decisions/0003-label-test-from-scratch.md)). The test videos are the keys with value `"test"` in `data/annotations/reviewed/split_assignment.json`.

1. Push pre-labels with `--exclude-split test`, as in step 4, so test tasks don't get them.
2. If test tasks already have predictions or annotations from an earlier push, filter them in the Data Manager by file name (each test video id), select them, and run **Actions → Delete predictions** and **Actions → Delete annotations**.
3. Label every image of every test video, including images without billboards (submit them empty).
4. From then on, don't edit the test videos: every pipeline run must see the same test split, and its content fingerprint (file names and boxes, independent of the ids Label Studio renumbers on every export) is tracked in MLflow as `data.test_fingerprint`.
