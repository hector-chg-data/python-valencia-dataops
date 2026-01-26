from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.registry import read_production
from src.train import train_and_promote


def _project_root() -> Path:
    # .../traceable-demo/app/main.py -> .../traceable-demo
    return Path(__file__).resolve().parents[1]


_state_lock = threading.Lock()
_model: Any | None = None
_production_meta: dict[str, Any] = {}
_model_load_error: str = ""


def _load_model() -> None:
    """
    Load production metadata and model into module-global state.
    Never raises; records errors into `_model_load_error`.
    """
    global _model, _production_meta, _model_load_error

    root_dir = _project_root()
    meta = read_production(root_dir)

    # Default state when no model is available.
    model_obj: Any | None = None
    err = ""

    try:
        model_path_raw = (meta.get("model_path") or "").strip()
        if not model_path_raw:
            err = "No production model available yet. Call POST /retrain first to create and promote a model."
        else:
            model_path = Path(model_path_raw)
            if not model_path.is_absolute():
                model_path = root_dir / model_path

            model_obj = joblib.load(model_path)
            if not hasattr(model_obj, "predict_one"):
                raise TypeError("Loaded object does not implement predict_one(height_cm).")
    except Exception as e:
        model_obj = None
        err = f"Failed to load production model: {e}"

    with _state_lock:
        _production_meta = meta if isinstance(meta, dict) else {}
        _model = model_obj
        _model_load_error = err


app = FastAPI(title="Traceable Demo (DVC + MLflow + FastAPI)")


@app.on_event("startup")
def _startup() -> None:
    _load_model()


class RetrainRequest(BaseModel):
    trainer: str = Field(..., min_length=1, max_length=200)


class PredictRequest(BaseModel):
    height_cm: float = Field(..., ge=30, le=300)


@app.get("/health")
def health() -> dict[str, Any]:
    with _state_lock:
        meta = dict(_production_meta) if _production_meta else {}

    return {"ok": True, "production": meta}


@app.post("/retrain")
def retrain(req: RetrainRequest) -> dict[str, Any]:
    try:
        production = train_and_promote(trainer=req.trainer, y_value=1.5)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrain failed: {e}") from e

    _load_model()

    with _state_lock:
        if _model is None:
            raise HTTPException(status_code=503, detail=_model_load_error or "Model not available after retrain.")
        meta = dict(_production_meta) if _production_meta else dict(production)

    return {"ok": True, "production": meta}


@app.post("/predict")
def predict(req: PredictRequest) -> dict[str, Any]:
    with _state_lock:
        model_obj = _model
        meta = dict(_production_meta) if _production_meta else {}
        err = _model_load_error

    if model_obj is None:
        raise HTTPException(
            status_code=503,
            detail=err
            or "No production model available yet. Call POST /retrain first to create and promote a model.",
        )

    y = float(model_obj.predict_one(req.height_cm))
    return {"y": y, "model": meta}

