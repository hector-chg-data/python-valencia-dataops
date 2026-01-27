from __future__ import annotations

import csv
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import mlflow
import yaml

from src.model import ConstantModel, MeanModel
from src.registry import append_retrain_log, ensure_dirs, write_production


def _project_root() -> Path:
    # .../traceable-demo/src/train.py -> .../traceable-demo
    return Path(__file__).resolve().parents[1]


def _utc_iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _get_git_commit_sha() -> str:
    """Return `git rev-parse HEAD`, or empty string if unavailable."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return ""


def _get_data_dvc_md5(root_dir: Path) -> str:
    """
    Parse `data/family_heights.csv.dvc` and return `outs[0].md5` if present.
    Returns empty string if the `.dvc` file does not exist or is invalid.
    """
    dvc_path = root_dir / "data" / "family_heights.csv.dvc"
    if not dvc_path.exists():
        return ""

    try:
        doc = yaml.safe_load(dvc_path.read_text(encoding="utf-8"))
        if not isinstance(doc, dict):
            return ""

        outs = doc.get("outs")
        if not isinstance(outs, list) or not outs:
            return ""

        first = outs[0]
        if not isinstance(first, dict):
            return ""

        md5 = first.get("md5")
        return str(md5) if md5 else ""
    except Exception:
        return ""


def _ensure_dataset_materialized(root_dir: Path) -> Path:
    """
    Ensure `data/family_heights.csv` exists.

    If the CSV is missing but the `.dvc` pointer exists, run:
      dvc pull data/family_heights.csv

    If that fails, raise a clear error including stdout/stderr.
    """
    csv_path = root_dir / "data" / "family_heights.csv"
    dvc_path = root_dir / "data" / "family_heights.csv.dvc"

    if csv_path.exists():
        return csv_path

    if dvc_path.exists():
        proc = subprocess.run(
            ["dvc", "pull", "data/family_heights.csv"],
            cwd=str(root_dir),
            text=True,
            capture_output=True,
        )
        if proc.returncode != 0:
            out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
            raise RuntimeError(
                "Failed to materialize dataset with `dvc pull data/family_heights.csv`.\n"
                f"Exit code: {proc.returncode}\n"
                f"Output:\n{out.strip()}"
            )

        if csv_path.exists():
            return csv_path

        raise RuntimeError(
            "dvc pull reported success but `data/family_heights.csv` is still missing."
        )

    # No CSV and no .dvc pointer: let the caller see a simple error.
    raise FileNotFoundError(
        f"Dataset not found and no DVC pointer present: {csv_path} (and {dvc_path})"
    )


def _read_heights_csv_normalized(csv_path: Path) -> list[float]:
    """
    Read heights from CSV and normalize to meters in a canonical `height_m` column.

    Accepts one of: `height_cm`, `height_m`, or `height`.
    If values look like centimeters (> 10), convert to meters by dividing by 100.
    If values look like meters (<= 10), keep as is.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    heights_raw: list[float] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row")

        columns = {name.strip(): name for name in reader.fieldnames}
        candidate_names = ["height_cm", "height_m", "height"]
        height_col_key = None
        for cand in candidate_names:
            if cand in columns:
                height_col_key = columns[cand]
                break

        if height_col_key is None:
            raise ValueError(
                "CSV is missing a height column. Expected one of: "
                "'height_cm', 'height_m', or 'height'."
            )

        for i, row in enumerate(reader, start=1):
            raw = (row.get(height_col_key) or "").strip()
            if not raw:
                continue
            try:
                heights_raw.append(float(raw))
            except Exception as e:
                raise ValueError(
                    f"Invalid height value on data row {i}: {raw!r}"
                ) from e

    if not heights_raw:
        raise ValueError("CSV contained 0 valid height values")

    max_val = max(heights_raw)
    if max_val > 10:
        # Treat as centimeters -> convert to meters.
        heights_m = [v / 100.0 for v in heights_raw]
    else:
        # Already in meters.
        heights_m = list(heights_raw)

    return heights_m


def train_and_promote(
    trainer: str,
    model_type: str = "constant",
    y_value: float = 1.5,
) -> dict[str, Any]:
    """
    Retrain model and promote it to production.

    Supported model types:
    - "constant": always predicts `y_value`
    - "mean": predicts the mean height (in meters) from the current dataset

    Records traceability:
    - trainer
    - data DVC md5 (from `.dvc` file)
    - git commit SHA (if git is available)
    - mlflow run_id (artifact source of truth for the model)
    """
    trainer_clean = (trainer or "").strip()
    if not trainer_clean:
        raise ValueError("trainer must be a non-empty string")

    model_type_clean = (model_type or "constant").strip().lower()
    if model_type_clean not in {"constant", "mean"}:
        raise ValueError("model_type must be either 'constant' or 'mean'")

    root_dir = _project_root()
    ensure_dirs(root_dir)

    data_path = _ensure_dataset_materialized(root_dir)
    heights_m = _read_heights_csv_normalized(data_path)

    n_rows = int(len(heights_m))
    mean_height_m = float(sum(heights_m) / n_rows)

    data_dvc_md5 = _get_data_dvc_md5(root_dir)
    git_commit = _get_git_commit_sha()

    # Respect external MLflow tracking URI when provided, otherwise use local file store.
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        mlflow.set_tracking_uri(f"file:{(root_dir / 'mlruns').as_posix()}")

    mlflow.set_experiment("family-heights-traceable-demo")

    model_path = root_dir / "models" / "production" / "model.pkl"
    model_rel_path = model_path.relative_to(root_dir).as_posix()

    with mlflow.start_run() as run:
        run_id = run.info.run_id

        mlflow.log_param("model_type", model_type_clean)
        mlflow.log_param("trainer", trainer_clean)
        mlflow.log_param("data_dvc_md5", data_dvc_md5)
        mlflow.log_param("git_commit", git_commit)
        if model_type_clean == "constant":
            mlflow.log_param("y_value", float(y_value))

        mlflow.log_metric("n_rows", n_rows)
        mlflow.log_metric("mean_height_m", mean_height_m)

        if model_type_clean == "constant":
            model = ConstantModel(y_value=float(y_value))
            effective_y_value = float(y_value)
        else:
            model = MeanModel(mean_value=mean_height_m)
            # The model predicts the dataset mean; expose it as y_value for traceability.
            effective_y_value = mean_height_m

        joblib.dump(model, model_path)

        # Artifact path "model" -> .../artifacts/model/model.pkl
        mlflow.log_artifact(str(model_path), artifact_path="model")

    production_meta: dict[str, Any] = {
        "run_id": run_id,
        "trainer": trainer_clean,
        "model_type": model_type_clean,
        "y_value": float(effective_y_value),
        "mean_height_m": mean_height_m,
        "n_rows": n_rows,
        "data_dvc_md5": data_dvc_md5,
        "git_commit": git_commit,
        "model_path": model_rel_path,
        "promoted_at_utc": _utc_iso_now(),
    }

    write_production(production_meta, root_dir)

    retrain_entry = dict(production_meta)
    retrain_entry["logged_at_utc"] = _utc_iso_now()
    append_retrain_log(retrain_entry, root_dir)

    return production_meta

