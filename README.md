# traceable-demo (DVC + MLflow + FastAPI)

Minimal, production-style demo showing **traceable predictions** from a **retrainable** ML model:

- **Model**: intentionally trivial, always predicts **y = 1.5**
- **Data versioning**: DVC tracks `data/family_heights.csv`
- **Experiment tracking**: MLflow logs params/metrics and the model artifact
- **Serving**: FastAPI with two endpoints: `/predict` and `/retrain`
- **Single source of truth**: `metadata/production.json` (used by inference)

Everything runs locally (no cloud services, no Docker).

## Folder layout

```
traceable-demo/
  app/
    main.py
  src/
    model.py
    train.py
    registry.py
  data/
    family_heights.csv
  metadata/
    production.json
    retrain_log.jsonl
  models/
    production/
      model.pkl
  requirements.txt
  README.md
```

## What gets recorded (traceability)

Each retrain writes/records:

- **trainer** (who retrained)
- **data_dvc_md5** (which exact DVC-tracked dataset version)
- **git_commit** (the code version, if git is available)
- **run_id** (the MLflow run that produced the model artifact)

These are stored in:

- `metadata/production.json` (single source of truth for inference)
- `metadata/retrain_log.jsonl` (append-only history)
- MLflow file store under `./mlruns/`

## Setup (exact commands)

Run these from the `traceable-demo/` directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
git init
dvc init
dvc add data/family_heights.csv
git add .
git commit -m "init"
uvicorn app.main:app --reload --port 8000
```

### Retrain + predict (curl)

```bash
curl -X POST http://127.0.0.1:8000/retrain -H "Content-Type: application/json" -d "{\"trainer\":\"alice\"}"
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"height_cm\":172}"
curl http://127.0.0.1:8000/health
```

### MLflow UI

```bash
mlflow ui --backend-store-uri ./mlruns --port 5000
```

Open the UI at `http://127.0.0.1:5000` and inspect the latest run under the
experiment **family-heights-traceable-demo**.

## Dataset update workflow (DVC-required)

Every dataset change must be tracked by DVC:

1. Edit `data/family_heights.csv`
2. Recompute the DVC hash:

```bash
dvc add data/family_heights.csv
```

3. Commit the updated `.dvc` file (and `dvc.lock` if created/updated):

```bash
git add data/family_heights.csv.dvc dvc.lock
git commit -m "update dataset"
```

Then call `POST /retrain` again to promote a new production model and metadata.

## Traceability checklist

Where to find each traceability field:

- **MLflow run_id**
  - `metadata/production.json` → `run_id`
  - MLflow UI → run details
- **DVC md5**
  - `metadata/production.json` → `data_dvc_md5`
  - `data/family_heights.csv.dvc` → `outs[0].md5` (parsed via `yaml.safe_load`)
- **Git commit**
  - `metadata/production.json` → `git_commit`
  - `git log -1` (if git is available)
- **Trainer**
  - `metadata/production.json` → `trainer`
  - `metadata/retrain_log.jsonl` → each line is a retrain event

## Common issues

### Missing git

If `git` is not installed or the repo has no commits yet, training records:

- `git_commit: ""`

This is intentional; the demo fails gracefully.

### Missing DVC `.dvc` file / md5

If you haven't run:

```bash
dvc add data/family_heights.csv
```

then `data/family_heights.csv.dvc` won’t exist, and training records:

- `data_dvc_md5: ""`

Run `dvc add` before retraining to get a real DVC md5 in MLflow + metadata.

### First run: no production model yet

`/predict` returns **HTTP 503** until you create a production model by calling:

- `POST /retrain`

The single source of truth is `metadata/production.json`.

### Windows shell differences

- **Activate venv (PowerShell)**:

```powershell
.\.venv\Scripts\Activate.ps1
```

- **Activate venv (cmd.exe)**:

```bat
.\.venv\Scripts\activate
```

- If `curl` behaves differently in PowerShell, use:
  - `Invoke-RestMethod` / `irm` as an alternative, but the README keeps `curl` for portability.

