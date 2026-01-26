from __future__ import annotations

import csv
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import mlflow
import yaml

from src.model import ConstantModel
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


def _read_heights_csv(csv_path: Path) -> list[float]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    heights: list[float] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row")

        required = {"member", "height_cm"}
        missing = required.difference(set(reader.fieldnames))
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

        for i, row in enumerate(reader, start=1):
            raw = (row.get("height_cm") or "").strip()
            try:
                heights.append(float(raw))
            except Exception as e:
                raise ValueError(f"Invalid height_cm on data row {i}: {raw!r}") from e

    if not heights:
        raise ValueError("CSV contained 0 data rows")

    return heights


def train_and_promote(trainer: str, y_value: float = 1.5) -> dict[str, Any]:
    """
    Retrain (trivial) model and promote it to production.

    Records traceability:
    - trainer
    - data DVC md5 (from `.dvc` file)
    - git commit SHA (if git is available)
    - mlflow run_id (artifact source of truth for the model)
    """
    trainer_clean = (trainer or "").strip()
    if not trainer_clean:
        raise ValueError("trainer must be a non-empty string")

    root_dir = _project_root()
    ensure_dirs(root_dir)

    data_path = root_dir / "data" / "family_heights.csv"
    heights = _read_heights_csv(data_path)

    n_rows = int(len(heights))
    mean_height_cm = float(sum(heights) / n_rows)

    data_dvc_md5 = _get_data_dvc_md5(root_dir)
    git_commit = _get_git_commit_sha()

    # Local file store (no cloud services).
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("family-heights-traceable-demo")

    model_path = root_dir / "models" / "production" / "model.pkl"
    model_rel_path = model_path.relative_to(root_dir).as_posix()

    with mlflow.start_run() as run:
        run_id = run.info.run_id

        mlflow.log_param("model_type", "constant")
        mlflow.log_param("y_value", float(y_value))
        mlflow.log_param("trainer", trainer_clean)
        mlflow.log_param("data_dvc_md5", data_dvc_md5)
        mlflow.log_param("git_commit", git_commit)

        mlflow.log_metric("n_rows", n_rows)
        mlflow.log_metric("mean_height_cm", mean_height_cm)

        model = ConstantModel(y_value=float(y_value))
        joblib.dump(model, model_path)

        # Artifact path "model" -> .../artifacts/model/model.pkl
        mlflow.log_artifact(str(model_path), artifact_path="model")

    production_meta: dict[str, Any] = {
        "run_id": run_id,
        "trainer": trainer_clean,
        "y_value": float(y_value),
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

