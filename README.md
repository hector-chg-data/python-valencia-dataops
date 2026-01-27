# Python Valencia
## Demo dataops (DVC + MLflow + FastAPI + Streamlit)

Minimal, production-style demo showing **traceable predictions** from a **retrainable** ML model:

- **Model**: intentionally trivial  
  - `constant` model always predicts **y = 1.5** (configurable)  
  - `mean` model predicts the mean height of the current dataset (in meters)
- **Data versioning**: DVC tracks `data/family_heights.csv`
- **Experiment tracking**: MLflow logs params/metrics and the model artifact
- **Serving**: FastAPI with three endpoints: `GET /health`, `POST /predict`, `POST /retrain`
- **Single source of truth**: `metadata/production.json` (used by inference)

Everything runs locally or via Docker Compose.

## Folder layout

```
python-valencia-dataops/
  app/
    main.py
  data/
    family_heights.csv
  src/
    model.py
    train.py
    registry.py
  metadata/
    production.json
    retrain_log.jsonl
  models/
    production/
      model.pkl
  ui/
    streamlit_app.py
  requirements.txt
  README.md
```

## What gets recorded (traceability)

Each retrain writes or records:

- **trainer** (who retrained)
- **data_dvc_md5** (which exact DVC-tracked dataset version)
- **git_commit** (the code version, if git is available)
- **run_id** (the MLflow run that produced the model artifact)

These are stored in:

- `metadata/production.json` (single source of truth for inference)
- `metadata/retrain_log.jsonl` (append-only history)
- MLflow file store under `./mlruns/`

## Local setup (venv)

[!NOTE]  
Presumes you have cloned the repo first.

Run these from the `python-valencia-dataops/` directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
streamlit run ui/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

### Retrain and predict (curl)

```bash
curl -X POST http://127.0.0.1:8000/retrain \
  -H "Content-Type: application/json" \
  -d '{"trainer":"alice","model_type":"constant","y_value":1.5}'

curl -X POST http://127.0.0.1:8000/retrain \
  -H "Content-Type: application/json" \
  -d '{"trainer":"alice","model_type":"mean"}'

curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"height_cm":1.72}'

curl http://127.0.0.1:8000/health
```

### MLflow UI (local file store)

```bash
mlflow ui --backend-store-uri ./mlruns --port 5000
```

Open the UI at `http://127.0.0.1:5000` and inspect the latest run under the
experiment **family-heights-traceable-demo**.

## Dataset update workflow (DVC)

Every dataset change should be tracked by DVC:

1. Edit `data/family_heights.csv`
2. Recompute the DVC hash:

   ```bash
   dvc add data/family_heights.csv
   ```

3. Commit the updated `.dvc` file (and `dvc.lock` if created or updated):

   ```bash
   git add data/family_heights.csv.dvc dvc.lock
   git commit -m "update dataset"
   ```

Then call `POST /retrain` again (from the UI or curl) to promote a new production model and metadata.

### Why `dvc pull` can fail on fresh clones

If you clone the repo on a new machine and run:

```bash
dvc pull data/family_heights.csv
```

you can see errors like:

- `No remote provided and no default remote set`
- `missing-files checkout failed`

This happens when:

- the repo only has `.dvc` pointer files
- the actual data was never pushed to any DVC remote

Fix:

1. Configure a DVC remote (see the DVC remote section below).
2. On the machine that already has the data:

   ```bash
   dvc push
   ```

3. On any new machine:

   ```bash
   dvc pull
   ```

If the original data is lost entirely, you can recreate `data/family_heights.csv`, run `dvc add data/family_heights.csv` and `dvc push` again to repopulate the remote.

## DVC remotes (MinIO)

This repo is configured to use MinIO as an S3 compatible DVC remote.

The DVC config in `.dvc/config` defines:

- **Default remote**: `minio-docker` pointing to `s3://dvc` with endpoint `http://minio:9000`
- **Host remote**: `minio-host` pointing to `s3://dvc` with endpoint `http://localhost:9000`

Usage:

- Inside Docker containers, just run:

  ```bash
  dvc push
  dvc pull
  ```

  which use the default `minio-docker` remote.

- On the host machine, if you have MinIO running locally on port 9000, you can run:

  ```bash
  dvc push -r minio-host
  dvc pull -r minio-host
  ```

  which uses the `minio-host` remote.

If no remote is set and you see `No remote provided and no default remote set`, make sure `.dvc/config` is committed and present, or add the remote again with:

```bash
dvc remote add -d minio-docker s3://dvc
dvc remote modify minio-docker endpointurl http://minio:9000
```

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

### Missing DVC `.dvc` file or md5

If you have not run:

```bash
dvc add data/family_heights.csv
```

then `data/family_heights.csv.dvc` will not exist, and training records:

- `data_dvc_md5: ""`

Run `dvc add` before retraining to get a real DVC md5 in MLflow and metadata.

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
  - `Invoke-RestMethod` or `irm` as an alternative, but the README keeps `curl` for portability.

## Docker Compose workflow

You can run the full stack (Postgres, MinIO, MLflow, API, and Streamlit UI) with Docker Compose.

From the repo root:

```bash
docker compose up --build
```

Services:

- **postgres**: PostgreSQL 18, internal port 5432, exposed on host as 5436
- **minio**: S3 compatible object store, console at `http://localhost:9001`
- **mlflow**: MLflow server at `http://localhost:5000`
- **api**: FastAPI app at `http://localhost:8000`
- **ui**: Streamlit UI at `http://localhost:8501`

Notes:

- Postgres uses `PGDATA=/var/lib/postgresql/18/docker` with a volume mounted at `/var/lib/postgresql`.
- The `minio_init` service creates the buckets `mlflow` and `dvc` automatically (no error if they already exist).
- The API and UI containers mount the repo into `/workspace` so `data/` and `.dvc/` are shared.

To stop everything:

```bash
docker compose down
```

# Python Valencia
## Demo dataops (DVC + MLflow + FastAPI)

Minimal, production-style demo showing **traceable predictions** from a **retrainable** ML model:

- **Model**: intentionally trivial, always predicts **y = 1.5**
- **Data versioning**: DVC tracks `data/family_heights.csv`
- **Experiment tracking**: MLflow logs params/metrics and the model artifact
- **Serving**: FastAPI with two endpoints: `/predict` and `/retrain`
- **Single source of truth**: `metadata/production.json` (used by inference)

Everything runs locally (no cloud services, no Docker).

## Folder layout

```
python-valencia-dataops/
  app/
    main.py
  data/
    family_heights.csv
  src/
    model.py
    train.py
    registry.py
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
[!NOTE]  
Presumes you have cloned the repo first.

Run these from the `python-valencia-dataops/` directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
dvc init
dvc add data/family_heights.csv
uvicorn app.main:app --reload --port 8000
```

### Retrain + predict (curl)

```bash
curl -X POST http://127.0.0.1:8000/retrain -H "Content-Type: application/json" -d "{"trainer":"alice"}"
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{"height_cm":1.72}"
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

