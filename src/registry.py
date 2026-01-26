from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def _project_root(root: Path | None = None) -> Path:
    if root is not None:
        return Path(root)
    # .../traceable-demo/src/registry.py -> .../traceable-demo
    return Path(__file__).resolve().parents[1]


def ensure_dirs(root: Path | None = None) -> dict[str, Path]:
    """Create required directories if missing."""
    root_dir = _project_root(root)
    metadata_dir = root_dir / "metadata"
    production_model_dir = root_dir / "models" / "production"

    metadata_dir.mkdir(parents=True, exist_ok=True)
    production_model_dir.mkdir(parents=True, exist_ok=True)

    return {
        "root": root_dir,
        "metadata": metadata_dir,
        "production_model_dir": production_model_dir,
    }


def read_production(root: Path | None = None) -> dict[str, Any]:
    """Read `metadata/production.json`. Returns {} if missing or invalid."""
    root_dir = _project_root(root)
    path = root_dir / "metadata" / "production.json"
    if not path.exists():
        return {}

    try:
        raw = path.read_text(encoding="utf-8").strip()
        if not raw:
            return {}
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def write_production(meta: dict[str, Any], root: Path | None = None) -> Path:
    """
    Atomically write `metadata/production.json` as the single source of truth.

    Implementation: write to temp file next to target, then `os.replace`.
    """
    dirs = ensure_dirs(root)
    target = dirs["metadata"] / "production.json"

    payload = json.dumps(meta, sort_keys=True, indent=2, ensure_ascii=False) + "\n"

    fd, tmp_path = tempfile.mkstemp(
        prefix="production.",
        suffix=".json.tmp",
        dir=str(dirs["metadata"]),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_path, target)
    finally:
        # If anything failed before replace, try to remove the temp file.
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

    return target


def append_retrain_log(entry: dict[str, Any], root: Path | None = None) -> Path:
    """Append a JSONL line to `metadata/retrain_log.jsonl`."""
    dirs = ensure_dirs(root)
    path = dirs["metadata"] / "retrain_log.jsonl"

    line = json.dumps(entry, sort_keys=True, ensure_ascii=False)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())

    return path

