# Labeling

Grounding DINO proposes boxes and a human corrects them in Label Studio. Test videos are labeled from scratch ([ADR 0003](decisions/0003-label-test-from-scratch.md)).

1. Put the frames in `data/raw/` and generate proposals:
   ```bash
   python scripts/run_auto_labeling.py
   ```
2. Start Label Studio:
   ```bash
   docker run -d --name vit-billboard-label-studio -p 8080:8080 \
     -v "$(pwd)/.label-studio/data:/label-studio/data" \
     -v "$(pwd)/data:/label-studio/files/data:ro" \
     -e LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true \
     -e LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/label-studio/files \
     heartexlabs/label-studio:latest
   ```
   At http://localhost:8080, create a project from the *Object Detection with Bounding Boxes* template with one label, `billboard`, and add *Local files* storage at `/label-studio/files/data/raw`.
3. Push the proposals as pre-annotations, skipping the test videos:
   ```bash
   LABEL_STUDIO_API_KEY=<token> python scripts/push_predictions_to_label_studio.py \
     --project-id <id> --exclude-split test
   ```
4. Correct every task; draw test tasks from scratch. Export as COCO.
5. Import the export and split by video ([ADR 0002](decisions/0002-split-by-video.md)):
   ```bash
   python scripts/import_label_studio_export.py --export-path <export>.zip
   python scripts/split_reviewed_dataset.py
   ```

The video-to-split assignment is saved in `data/annotations/reviewed/split_assignment.json` and reused on every export, so labeling can continue in batches. Once labeled, the test videos are not edited again; their fingerprint is tracked in MLflow as `data.test_fingerprint`.
