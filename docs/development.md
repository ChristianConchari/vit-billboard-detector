# Development

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e . -r requirements.txt
pytest -q   # unit + integration tests, no model downloads or GPU needed (~25 s)
```

Run every command from the repository root: config and data paths are relative to it.

## Code quality and CI

[Ruff](https://docs.astral.sh/ruff/) formats and lints the code (settings in `pyproject.toml`: line length 88, pyflakes, pycodestyle, bugbear, import sorting, pyupgrade):

```bash
ruff format src scripts tests
ruff check src scripts tests
```

GitHub Actions (`.github/workflows/ci.yml`) runs on every push to `main` and on pull requests, with two jobs:
- **lint:** `ruff check` and `ruff format --check`;
- **test:** installs the pinned dependencies (CPU-only PyTorch, no GPU needed) and runs `pytest`: unit tests per module, plus integration tests that run the whole pipeline and the `billboard-detect` CLI end to end with a tiny randomly initialized RT-DETR.

## Tooling

- PyTorch / torchvision (training loop, box-aware augmentations)
- Hugging Face `transformers` (RT-DETR, Grounding DINO)
- pycocotools (COCO mAP), OpenCV (video I/O), matplotlib (attention colormaps)
- MLflow for experiment tracking (SQLite backend, `mlflow.db`)
- Label Studio for box review
- pytest for testing

Dependencies are pinned in `requirements.txt` to the versions the pipeline was validated with (Python 3.12).
