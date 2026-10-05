# Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e . -r requirements.txt
pytest -q                      # unit + integration, CPU only, no downloads
ruff format src scripts tests
ruff check src scripts tests
```

CI (`.github/workflows/ci.yml`) runs ruff and pytest on every push and pull request. The integration tests run the full pipeline and the `billboard-detect` CLI with a tiny random RT-DETR.

Dependencies are pinned in `requirements.txt` (Python 3.12).
