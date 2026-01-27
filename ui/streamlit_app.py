import json
import os
import subprocess
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = REPO_ROOT / "data" / "family_heights.csv"
DATA_DVC = REPO_ROOT / "data" / "family_heights.csv.dvc"


def _default_api_base() -> str:
    env_base = os.getenv("API_BASE")
    if env_base:
        return env_base.strip()
    return "http://127.0.0.1:8000"


st.set_page_config(page_title="Traceable Demo UI", layout="wide")
st.title("Traceable model demo")


def run_cmd(cmd: list[str]) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
        out = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
        return p.returncode, out.strip()
    except Exception as e:
        return 1, str(e)


def read_dvc_md5() -> str:
    if not DATA_DVC.exists():
        return ""
    try:
        d = yaml.safe_load(DATA_DVC.read_text(encoding="utf-8"))
    except Exception:
        return ""
    if not isinstance(d, dict):
        return ""
    outs = d.get("outs", [])
    if not outs:
        return ""
    first = outs[0] or {}
    return first.get("md5", "") or ""


def api_get(base_url: str, path: str):
    url = f"{base_url.rstrip('/')}{path}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.HTTPError as e:
        text = r.text if "r" in locals() else ""
        raise RuntimeError(f"GET {url} failed: {e}\n{text}") from e
    except Exception as e:
        raise RuntimeError(f"GET {url} failed: {e}") from e


def api_post(base_url: str, path: str, payload: dict):
    url = f"{base_url.rstrip('/')}{path}"
    try:
        r = requests.post(url, json=payload, timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.HTTPError as e:
        text = r.text if "r" in locals() else ""
        raise RuntimeError(f"POST {url} failed: {e}\n{text}") from e
    except Exception as e:
        raise RuntimeError(f"POST {url} failed: {e}") from e


col_data, col_model = st.columns(2)

with col_data:
    st.subheader("Dataset")

    if DATA_CSV.exists():
        df = pd.read_csv(DATA_CSV)
    else:
        st.warning(
            "Dataset file not found on disk. If you use DVC, run: dvc pull data/family_heights.csv"
        )
        df = pd.DataFrame({"member": [], "height_cm": []})

    edited = st.data_editor(df, num_rows="dynamic", use_container_width=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button("Save CSV", use_container_width=True):
            edited.to_csv(DATA_CSV, index=False)
            st.success("Saved data/family_heights.csv")

    with c2:
        if st.button("Commit dataset (DVC + git)", use_container_width=True):
            edited.to_csv(DATA_CSV, index=False)
            code_add, out_add = run_cmd(["dvc", "add", "data/family_heights.csv"])

            code_git_add, out_git_add = run_cmd(
                ["git", "add", "data/family_heights.csv", "data/family_heights.csv.dvc"]
            )

            commit_with_git = st.checkbox(
                "Create git commit for dataset pointer", value=True, key="commit_with_git"
            )
            push_with_dvc = st.checkbox(
                "Run dvc push to configured remote", value=False, key="push_with_dvc"
            )

            combined_output = []
            combined_output.append("=== dvc add ===")
            combined_output.append(out_add or "(no output)")
            combined_output.append("\n=== git add ===")
            combined_output.append(out_git_add or "(no output)")

            if commit_with_git:
                name = st.text_input("git user.name", "demo-user", key="git_name")
                email = st.text_input(
                    "git user.email", "demo-user@example.com", key="git_email"
                )
                msg = st.text_input(
                    "commit message", "data: update family heights", key="git_msg"
                )
                code_git_commit, out_git_commit = run_cmd(
                    [
                        "git",
                        "-c",
                        f"user.name={name}",
                        "-c",
                        f"user.email={email}",
                        "commit",
                        "-m",
                        msg,
                    ]
                )
                combined_output.append("\n=== git commit ===")
                combined_output.append(out_git_commit or "(no output)")
                if code_git_commit == 0:
                    st.success("Committed dataset change")
                else:
                    st.warning("Git commit step failed; see output below.")

            if push_with_dvc:
                code_push, out_push = run_cmd(["dvc", "push"])
                combined_output.append("\n=== dvc push ===")
                combined_output.append(out_push or "(no output)")
                if code_push == 0:
                    st.success("Pushed dataset to DVC remote")
                else:
                    st.warning("dvc push failed; see output below.")

            if code_add == 0:
                st.success("DVC pointer updated")
            else:
                st.error("DVC add failed")

            st.code("\n".join(combined_output).strip() or "(no output)")

    with c3:
        if st.button("DVC pull dataset", use_container_width=True):
            code_pull, out_pull = run_cmd(["dvc", "pull", "data/family_heights.csv"])
            if code_pull == 0:
                st.success("dvc pull successful")
            else:
                st.error("dvc pull failed")
            st.code(out_pull or "(no output)")

    st.markdown("### DVC info")
    st.write(f"Current DVC md5: `{read_dvc_md5()}`")

    code_status, out_status = run_cmd(["dvc", "status"])
    st.markdown("`dvc status`:")
    st.code(out_status or "(no output)")

    st.markdown("### Dataset history (git log of .dvc pointer)")
    code_log, out_log = run_cmd(
        ["git", "log", "-n", "10", "--oneline", "--", "data/family_heights.csv.dvc"]
    )
    st.code(out_log or "(no output)")

with col_model:
    st.subheader("Model and API")

    api_base = st.text_input("API base URL", _default_api_base()).strip()

    c1, _ = st.columns(2)
    with c1:
        if st.button("Refresh /health"):
            try:
                health = api_get(api_base, "/health")
                st.session_state["health"] = health
            except Exception as e:
                st.error(str(e))

    health = st.session_state.get("health")
    if health:
        st.markdown("#### /health response")
        st.json(health)

        production = health.get("production") or {}
        if production:
            st.markdown("#### Production model metadata")
            summary_fields = {
                "run_id": production.get("run_id"),
                "trainer": production.get("trainer"),
                "model_type": production.get("model_type"),
                "n_rows": production.get("n_rows"),
                "mean_height_m": production.get("mean_height_m"),
                "data_dvc_md5": production.get("data_dvc_md5"),
                "git_commit": production.get("git_commit"),
            }
            st.json(summary_fields)

    st.markdown("### Retrain")
    trainer = st.text_input("trainer", "alice")
    model_type = st.selectbox("model_type", ["constant", "mean"], index=0)
    y_value = st.number_input("y_value (constant only)", value=1.5, step=0.1)

    if st.button("Call /retrain"):
        try:
            payload = {
                "trainer": trainer,
                "model_type": model_type,
                "y_value": float(y_value),
            }
            res = api_post(api_base, "/retrain", payload)
            st.success("Retrained")
            st.json(res)
            try:
                st.session_state["health"] = api_get(api_base, "/health")
            except Exception:
                pass
        except Exception as e:
            st.error(str(e))

    st.markdown("### Predict")
    height = st.number_input("height (meters)", value=1.72, step=0.01)
    if st.button("Call /predict"):
        try:
            res = api_post(api_base, "/predict", {"height_cm": float(height)})
            st.json(res)
        except Exception as e:
            st.error(str(e))
