from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, Generator
import uuid
import json
import logging
import re
import threading
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

import httpx

from state import MLState
from graph import step_graph
from agents import MLAgent
from models_config import get_available_models, resolve_model, get_api_key

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Lab Agent", version="4.0.0")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lab-agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# ── In-memory stores ──────────────────────────────────────────────────────────
agent_states: dict[str, MLState] = {}
v2_sessions:  dict[str, dict]    = {}

LAB_BACKEND     = "http://localhost:8010"
MAX_FIX_ATTEMPTS = 7   # guard retry budget per failing cell
MAX_AGENT_CALL_RETRIES = 3


# ── Shared helpers ────────────────────────────────────────────────────────────

def _exec(session_id: str, code: str) -> tuple[str, str | None, list[str]]:
    """Execute code in the lab-backend. Returns (stdout, error, charts)."""
    try:
        with httpx.Client(timeout=90.0) as c:
            resp = c.post(
                f"{LAB_BACKEND}/execute",
                json={"session_id": session_id, "cell_id": str(uuid.uuid4()), "code": code},
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("stdout", ""), data.get("error") or None, data.get("charts", [])
    except Exception as exc:
        return "", str(exc), []


# ── Persistent fix memory (lightweight "reinforcement") ───────────────────────
# When the guard successfully fixes an error, the (fingerprint → fix) pair is
# saved to fix_memory.json.  On the next run, matching errors skip straight to
# the proven fix instead of starting from scratch with the LLM.

_FIX_MEMORY_PATH = Path(__file__).parent / "fix_memory.json"
_fix_memory_lock = threading.Lock()


def _error_fingerprint(error: str, stage: str) -> str:
    """Produce a stable, short key for an error type + stage combination."""
    # Extract the exception class name (ValueError, NameError, …)
    m = re.search(r'\b([A-Z][a-zA-Z]+Error)\b', error)
    exc_type = m.group(1) if m else "Error"
    # Extract the first quoted token (e.g. the bad column value or module name)
    q = re.search(r"['\"]([^'\"]{1,60})['\"]", error)
    token = q.group(1) if q else ""
    # Normalise numbers/IDs so 'AG-2011-2040' and 'US-2013-100328' hash the same
    token_norm = re.sub(r'\d+', 'N', token)
    return f"{stage}|{exc_type}|{token_norm}"


def _load_fix_memory() -> list[dict]:
    if not _FIX_MEMORY_PATH.exists():
        return []
    try:
        with _FIX_MEMORY_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save_fix_memory(entries: list[dict]) -> None:
    with _fix_memory_lock:
        try:
            with _FIX_MEMORY_PATH.open("w", encoding="utf-8") as f:
                json.dump(entries, f, indent=2, ensure_ascii=False)
        except Exception:
            pass


def _recall_fix(error: str, stage: str) -> dict | None:
    """Return the best memorised fix for this error/stage, or None."""
    fp = _error_fingerprint(error, stage)
    entries = _load_fix_memory()
    # Exact fingerprint match — return the one with most successes
    candidates = [e for e in entries if e.get("fingerprint") == fp]
    if candidates:
        return max(candidates, key=lambda e: e.get("success_count", 0))
    return None


def _record_fix(error: str, stage: str, fixed_code: str, explanation: str) -> None:
    """Persist a successful fix so the guard can reuse it later."""
    fp = _error_fingerprint(error, stage)
    entries = _load_fix_memory()
    # Update existing entry if fingerprint already known
    for e in entries:
        if e.get("fingerprint") == fp:
            e["success_count"] = e.get("success_count", 0) + 1
            e["fixed_code"]   = fixed_code    # keep the most recent working version
            e["explanation"]  = explanation
            _save_fix_memory(entries)
            return
    # New pattern — append
    entries.append({
        "fingerprint":   fp,
        "error_sample":  error[:300],
        "stage":         stage,
        "fixed_code":    fixed_code,
        "explanation":   explanation,
        "success_count": 1,
    })
    _save_fix_memory(entries)


def _get_agent(model_id: Optional[str] = None) -> MLAgent:
    try:
        cfg = resolve_model(model_id)
        key = get_api_key(model_id)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return MLAgent(api_key=key, model=cfg["model_name"], provider=cfg["provider"])


def _profile_code(file_path: str, filename: str) -> str:
    ext = Path(filename).suffix.lower()
    read_call = (
        f'pd.read_excel("{file_path}")'
        if ext in (".xlsx", ".xls")
        else f'pd.read_csv("{file_path}")'
    )
    return f"""\
import pandas as pd
import numpy as np

df = {read_call}

print("=== SHAPE ===")
print(f"Rows: {{df.shape[0]}}, Columns: {{df.shape[1]}}")

print("\\n=== COLUMN INFO ===")
for col in df.columns:
    print(f"  {{col}} [{{df[col].dtype}}]  unique={{df[col].nunique()}}  null={{df[col].isnull().sum()}}")

print("\\n=== FIRST 4 ROWS ===")
print(df.head(4).to_string())

print("\\n=== NUMERIC STATISTICS ===")
num = df.select_dtypes(include='number')
if not num.empty:
    print(num.describe().to_string())

print("\\n=== CATEGORICAL DISTRIBUTIONS ===")
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
for col in cat_cols[:6]:
    print(f"  {{col}}: {{dict(df[col].value_counts().head(5))}}")
"""


def _sse(event: dict) -> str:
    """Format a dict as a single SSE data line."""
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def _fallback_models(problem_type: str) -> list[dict]:
    if (problem_type or "").lower() == "regression":
        return [{
            "rank": 1,
            "name": "RandomForestRegressor",
            "display_name": "Random Forest Regressor",
            "reasoning": "Reliable baseline when detailed model analysis fails.",
            "pros": ["Handles non-linearity", "Works with mixed feature scales"],
            "cons": ["May need tuning for best performance"],
            "expected_performance": "Baseline quality regression output",
        }]
    return [{
        "rank": 1,
        "name": "RandomForestClassifier",
        "display_name": "Random Forest Classifier",
        "reasoning": "Reliable baseline when detailed model analysis fails.",
        "pros": ["Robust to noisy features", "Strong baseline accuracy"],
        "cons": ["Can be less interpretable"],
        "expected_performance": "Baseline quality classification output",
    }]


def _fallback_analysis_result(target_column: str | None, problem_type: str = "classification") -> dict:
    target = (target_column or "target").strip() or "target"
    ptype = (problem_type or "classification").strip() or "classification"
    return {
        "problem_type": ptype,
        "target_column": target,
        "dataset_summary": "Fallback summary: dataset loaded, but AI analysis failed after retries.",
        "feature_analysis": "Fallback feature analysis enabled to continue pipeline execution.",
        "missing_values_note": "Use imputation and safe defaults during pipeline build.",
        "feature_engineering_reasoning": "Proceeding with conservative preprocessing fallback.",
        "feature_engineering_code": "print('Fallback: skipped advanced feature engineering due to upstream analysis issue.')",
        "models": _fallback_models(ptype),
    }


def _fallback_pipeline_result(file_path: str, target_column: str, problem_type: str) -> dict:
    target = (target_column or "target").replace("'", "")
    is_reg = (problem_type or "").lower() == "regression"
    model_cls = "RandomForestRegressor" if is_reg else "RandomForestClassifier"
    metric_line = "print('R2:', r2_score(y_test, preds))" if is_reg else "print('Accuracy:', accuracy_score(y_test, preds))"

    code = f"""\
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import {model_cls}
from sklearn.metrics import accuracy_score, r2_score

if 'df_processed' in globals():
    _work_df = df_processed.copy()
elif 'df' in globals():
    _work_df = df.copy()
else:
    _work_df = pd.read_csv(r'{file_path}')

if '{target}' not in _work_df.columns:
    raise ValueError("Target column '{target}' not found in dataset")

y = _work_df['{target}']
X = _work_df.drop(columns=['{target}'])
X = pd.get_dummies(X, drop_first=False)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
trained_columns = X_train.columns.tolist()
feature_names = X.columns.tolist()

model = {model_cls}(random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)
{metric_line}
print('Fallback pipeline executed. Features:', len(trained_columns))
"""

    return {
        "reasoning": "Fallback pipeline generated after AI pipeline generation failed.",
        "feature_columns": [],
        "cells": [{
            "title": "Fallback Baseline Pipeline",
            "description": "Loads data, trains a baseline model, and evaluates results.",
            "code": code,
            "has_chart": False,
        }],
    }


def _guard_fix(
    agent: MLAgent,
    session_id: str,
    code: str,
    error: str,
    context: dict,
    title: str,
) -> Generator[str, None, tuple[str, str, str | None, list[str]]]:
    """
    Guard agent retry loop.  Yields SSE events and returns the final
    (code, stdout, error, charts) tuple via StopIteration value.

    Usage inside a generator:
        result = yield from _guard_fix(agent, sid, code, error, state, title)
        code, stdout, error, charts = result
    """
    _RESET_MARKERS = (
        "kernel was reset",
        "execution timed out",
        "timed out",
        "name 'df' is not defined",       # df lost after kernel reset
        "name 'df_processed' is not defined",
    )

    def _is_kernel_reset(err: str) -> bool:
        lo = err.lower()
        return any(m in lo for m in _RESET_MARKERS)

    def _df_reload_code(fp: str) -> str:
        ext = fp.rsplit(".", 1)[-1].lower() if "." in fp else ""
        fn = "pd.read_excel" if ext in ("xlsx", "xls") else "pd.read_csv"
        return f"import pandas as pd\ndf = {fn}(r'{fp}')\nprint('Dataset restored:', df.shape)"

    stage = context.get("stage", title)

    for attempt in range(MAX_FIX_ATTEMPTS):
        # Notify frontend that guard is analysing
        yield _sse({
            "type": "guard_analyzing",
            "title": title,
            "error_preview": error[:400],
            "attempt": attempt + 1,
        })

        # ── Fix memory: try a proven fix before calling the LLM ──────────────
        memory_hit = _recall_fix(error, stage) if attempt == 0 else None
        if memory_hit:
            fixed       = memory_hit["fixed_code"]
            explanation = f"[Memory] {memory_hit['explanation']}"
            yield _sse({"type": "fix_attempt", "code": fixed,
                        "explanation": explanation, "attempt": attempt + 1})
        else:
            # ── LLM fix ──────────────────────────────────────────────────────
            fix = agent.fix_error(code, error, context)

            # Optionally escalate to Tavily web search
            if fix.get("needs_search") and fix.get("search_query"):
                query = fix["search_query"]
                yield _sse({"type": "web_searching", "query": query})
                sr = agent.web_search(query)
                if sr:
                    fix = agent.fix_error(code, error, context, sr)

            fixed       = (fix.get("fixed_code") or "").strip()
            explanation = fix.get("explanation", "")

            # Guard couldn't produce a different fix — bail out
            if not fixed or fixed == code:
                break

            yield _sse({"type": "fix_attempt", "code": fixed,
                        "explanation": explanation, "attempt": attempt + 1})

        # ── Kernel-reset recovery: restore df before running any retry ────────
        if _is_kernel_reset(error):
            file_path = context.get("file_path", "")
            if file_path:
                yield _sse({"type": "status", "text": "Restoring dataset after kernel reset…"})
                _exec(session_id, _df_reload_code(file_path))
        # ── End kernel-reset recovery ─────────────────────────────────────────

        new_stdout, new_err, new_charts = _exec(session_id, fixed)
        if not new_err:
            yield _sse({"type": "fix_success", "title": title, "explanation": explanation})
            # Persist this successful fix so future runs skip the LLM
            _record_fix(error, stage, fixed, explanation)
            return fixed, new_stdout, None, new_charts   # type: ignore[return-value]

        # Still failing — update code/error for next attempt
        code, error = fixed, new_err

    # All retries exhausted
    yield _sse({"type": "guard_give_up", "title": title, "error": error[:400]})
    return code, "", error, []   # type: ignore[return-value]


# ── Legacy v1 endpoints ───────────────────────────────────────────────────────

class RunStepRequest(BaseModel):
    session_id: str
    target_column: Optional[str] = None
    uploaded_file_path: Optional[str] = None
    uploaded_filename: Optional[str] = None

class RunStepResponse(BaseModel):
    stage: str; code: str; output: str; finished: bool


@app.post("/run-step", response_model=RunStepResponse)
def run_step(req: RunStepRequest):
    state = agent_states.get(req.session_id)
    if state is None:
        if not req.uploaded_file_path:
            raise HTTPException(status_code=400, detail="Upload a dataset first.")
        state = MLState(
            session_id=req.session_id, stage="", dataset_loaded=False,
            target_column=req.target_column, problem_type=None,
            last_output="", last_code="", finished=False,
            uploaded_file_path=req.uploaded_file_path,
            uploaded_filename=req.uploaded_filename,
        )
    overrides: dict = {}
    if req.target_column: overrides["target_column"] = req.target_column
    if req.uploaded_file_path and not state.get("uploaded_file_path"):
        overrides["uploaded_file_path"] = req.uploaded_file_path
        overrides["uploaded_filename"]  = req.uploaded_filename
    if overrides: state = {**state, **overrides}  # type: ignore

    if state.get("finished"):
        return RunStepResponse(stage="done", code="", output="Pipeline complete.", finished=True)

    result: MLState = step_graph.invoke(state)  # type: ignore
    agent_states[req.session_id] = result
    return RunStepResponse(stage=result.get("stage",""), code=result.get("last_code",""),
                           output=result.get("last_output",""), finished=result.get("finished",False))


@app.delete("/agent-session/{session_id}")
def delete_agent_session(session_id: str):
    agent_states.pop(session_id, None)
    v2_sessions.pop(session_id, None)
    return {"ok": True}

@app.get("/agent-state/{session_id}")
def get_agent_state(session_id: str):
    s = agent_states.get(session_id)
    if not s: raise HTTPException(404, "No legacy agent state")
    return s

@app.get("/health")
def health():
    return {"status": "ok", "v1_sessions": len(agent_states), "v2_sessions": len(v2_sessions)}


# ── V2 request models ─────────────────────────────────────────────────────────

class V2AnalyzeRequest(BaseModel):
    session_id: str
    uploaded_file_path: str
    uploaded_filename: str
    target_column: Optional[str] = None
    model_id: Optional[str] = None

class V2BuildPipelineRequest(BaseModel):
    session_id: str
    selected_model: str
    target_column: Optional[str] = None
    model_id: Optional[str] = None

class V2ChatRequest(BaseModel):
    session_id: str
    message: str
    model_id: Optional[str] = None

class V2PredictRequest(BaseModel):
    session_id: str
    input_values: dict   # {column_name: value}
    model_id: Optional[str] = None


# ── V2: Analyse (SSE streaming) ───────────────────────────────────────────────

@app.post("/v2/analyze-stream")
def v2_analyze_stream(req: V2AnalyzeRequest):
    """
    SSE endpoint that streams the analysis step-by-step:
      status → code_cell (profile) → status → analysis → fe_cell [guard] → models → done
    """
    def generate():
        try:
            agent = _get_agent(req.model_id)
        except HTTPException as exc:
            yield _sse({"type": "error", "text": exc.detail}); yield _sse({"type": "done"}); return

        # Step 1 – profile dataset
        yield _sse({"type": "status", "text": "📊 Loading and profiling your dataset…"})
        profile_code = _profile_code(req.uploaded_file_path, req.uploaded_filename)
        p_out, p_err, _ = _exec(req.session_id, profile_code)
        profile_output = p_out if not p_err else f"{p_out}\n[ERROR] {p_err}"
        yield _sse({"type": "code_cell", "code": profile_code, "output": p_out,
                    "error": p_err, "title": "Dataset Profile", "charts": []})

        if not p_out and p_err:
            yield _sse({"type": "error", "text": f"Failed to load dataset: {p_err}"})
            yield _sse({"type": "done"}); return

        # Step 2 – AI analysis
        yield _sse({"type": "status", "text": "🤔 AI is analysing your data…"})
        result = None
        last_analysis_error = None
        for _ in range(MAX_AGENT_CALL_RETRIES):
            try:
                result = agent.analyze(req.uploaded_filename, profile_output, req.target_column)
                break
            except Exception as exc:
                last_analysis_error = exc

        if result is None:
            yield _sse({"type": "status", "text": f"⚠️ AI analysis failed after retries. Switching to fallback analysis: {last_analysis_error}"})
            result = _fallback_analysis_result(req.target_column)

        yield _sse({"type": "analysis", "data": {
            "problem_type":                  result.get("problem_type"),
            "target_column":                 result.get("target_column") or req.target_column,
            "dataset_summary":               result.get("dataset_summary", ""),
            "feature_analysis":              result.get("feature_analysis", ""),
            "missing_values_note":           result.get("missing_values_note", ""),
            "feature_engineering_reasoning": result.get("feature_engineering_reasoning", ""),
        }})

        # Step 3 – feature engineering
        fe_code = result.get("feature_engineering_code", "").strip()
        fe_out, fe_err, fe_charts = "", None, []
        if fe_code and fe_code not in ("pass", "# (no feature engineering needed)"):
            yield _sse({"type": "status", "text": "⚙️ Applying feature engineering…"})
            fe_out, fe_err, fe_charts = _exec(req.session_id, fe_code)

            # ── Guard: auto-fix feature-engineering errors ────────────────────
            if fe_err:
                ctx = {
                    "filename": req.uploaded_filename,
                    "file_path": req.uploaded_file_path,
                    "problem_type": result.get("problem_type", "unknown"),
                    "target_column": result.get("target_column") or req.target_column or "",
                    "stage": "feature_engineering",
                    "dataset_summary": result.get("dataset_summary", ""),
                }
                fix_result = yield from _guard_fix(
                    agent, req.session_id, fe_code, fe_err, ctx, "Feature Engineering"
                )
                fe_code, fe_out, fe_err, fe_charts = fix_result
            # ── End guard ─────────────────────────────────────────────────────

            yield _sse({"type": "fe_cell", "code": fe_code, "output": fe_out,
                        "error": fe_err, "charts": fe_charts})

        # Step 3b – Exploratory Data Analysis (EDA)
        yield _sse({"type": "status", "text": "🔍 Running Exploratory Data Analysis…"})
        eda_summary_data = {}
        eda_chart_insights: list[dict] = []
        try:
            eda_result = agent.run_eda(
                filename=req.uploaded_filename,
                problem_type=result.get("problem_type", "classification"),
                target_column=result.get("target_column") or req.target_column or "",
                profile_output=profile_output,
                feature_analysis=result.get("feature_analysis", ""),
            )
            for eda_cell in eda_result.get("cells", []):
                eda_code = (eda_cell.get("code") or "").strip()
                eda_title = eda_cell.get("title", "EDA")
                if not eda_code:
                    continue
                eda_out, eda_err, eda_charts = _exec(req.session_id, eda_code)

                # Guard: auto-fix EDA cell errors
                if eda_err:
                    ctx = {
                        "filename": req.uploaded_filename,
                        "file_path": req.uploaded_file_path,
                        "problem_type": result.get("problem_type", "unknown"),
                        "target_column": result.get("target_column") or req.target_column or "",
                        "stage": "eda",
                        "dataset_summary": result.get("dataset_summary", ""),
                    }
                    fix_result = yield from _guard_fix(
                        agent, req.session_id, eda_code, eda_err, ctx, f"EDA: {eda_title}"
                    )
                    eda_code, eda_out, eda_err, eda_charts = fix_result

                yield _sse({
                    "type": "eda_cell",
                    "code": eda_code, "output": eda_out,
                    "error": eda_err, "title": eda_title,
                    "charts": eda_charts,
                })

                # Add chart-level narrative so reports are evidence-driven.
                if eda_charts:
                    try:
                        insight = agent.explain_chart(eda_code, {
                            "filename": req.uploaded_filename,
                            "problem_type": result.get("problem_type", "unknown"),
                            "target_column": result.get("target_column") or req.target_column or "",
                            "stage": "eda",
                            "dataset_summary": result.get("dataset_summary", ""),
                        })
                        insight_obj = {
                            "title": eda_title,
                            "charts_count": len(eda_charts),
                            "insight": insight,
                        }
                        eda_chart_insights.append(insight_obj)
                        yield _sse({"type": "eda_insight", "data": insight_obj})
                    except Exception:
                        pass

            eda_summary_data = {
                "summary": eda_result.get("summary", ""),
                "feature_importance": eda_result.get("feature_importance_notes", ""),
                "preprocessing": eda_result.get("preprocessing_recommendations", ""),
                "executive_summary": eda_result.get("executive_summary", ""),
                "data_quality_findings": eda_result.get("data_quality_findings", []),
                "key_patterns": eda_result.get("key_patterns", []),
                "risk_flags": eda_result.get("risk_flags", []),
                "recommendations": eda_result.get("recommendations", []),
                "chart_narrative": eda_result.get("chart_narrative", ""),
                "chart_insights": eda_chart_insights,
            }
            yield _sse({"type": "eda_summary", "data": eda_summary_data})
            yield _sse({"type": "report_summary", "data": eda_summary_data})
        except Exception as exc:
            logger.warning("EDA step failed (non-fatal): %s", exc)
            yield _sse({"type": "status", "text": "⚠️ EDA step skipped due to an error."})

        # Step 4 – model suggestions
        yield _sse({"type": "models", "models": result.get("models", [])})

        # Persist session
        v2_sessions[req.session_id] = {
            "filename":                      req.uploaded_filename,
            "file_path":                     req.uploaded_file_path,
            "target_column":                 result.get("target_column") or req.target_column or "",
            "problem_type":                  result.get("problem_type", "classification"),
            "profile_output":                profile_output,
            "dataset_summary":               result.get("dataset_summary", ""),
            "feature_analysis":              result.get("feature_analysis", ""),
            "feature_engineering_reasoning": result.get("feature_engineering_reasoning", ""),
            "feature_engineering_code":      fe_code,
            "missing_values_note":           result.get("missing_values_note", ""),
            "models":                        result.get("models", []),
            "eda_summary":                   eda_summary_data.get("summary", ""),
            "feature_importance_notes":      eda_summary_data.get("feature_importance", ""),
            "preprocessing_recommendations": eda_summary_data.get("preprocessing", ""),
            "report_executive_summary":      eda_summary_data.get("executive_summary", ""),
            "report_data_quality_findings":  eda_summary_data.get("data_quality_findings", []),
            "report_key_patterns":           eda_summary_data.get("key_patterns", []),
            "report_risk_flags":             eda_summary_data.get("risk_flags", []),
            "report_recommendations":        eda_summary_data.get("recommendations", []),
            "report_chart_narrative":        eda_summary_data.get("chart_narrative", ""),
            "report_chart_insights":         eda_summary_data.get("chart_insights", []),
            "selected_model":                None,
            "feature_columns":               [],
            "stage":                         "analyzed",
            "chat_history":                  [],
        }
        yield _sse({"type": "done"})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── V2: Build pipeline (SSE streaming) ────────────────────────────────────────

@app.post("/v2/build-pipeline-stream")
def v2_build_pipeline_stream(req: V2BuildPipelineRequest):
    """
    SSE endpoint that streams the pipeline cell-by-cell:
      status → reasoning → (status → [guard] → code_cell [→ insight]) × N → done
    """
    def generate():
        state = v2_sessions.get(req.session_id)
        if not state:
            yield _sse({"type": "error", "text": "No analysis session found. Run Analyse first."})
            yield _sse({"type": "done"}); return

        try:
            agent = _get_agent(req.model_id)
        except HTTPException as exc:
            yield _sse({"type": "error", "text": exc.detail}); yield _sse({"type": "done"}); return

        target = req.target_column or state.get("target_column") or "label"
        yield _sse({"type": "status", "text": f"🧠 Planning pipeline with {req.selected_model}…"})

        result = None
        last_pipeline_error = None
        for _ in range(MAX_AGENT_CALL_RETRIES):
            try:
                result = agent.build_pipeline(
                    filename=state["filename"],
                    problem_type=state.get("problem_type") or "classification",
                    target_column=target,
                    model_name=req.selected_model,
                    profile_output=state.get("profile_output", ""),
                    fe_code=state.get("feature_engineering_code", ""),
                )
                break
            except Exception as exc:
                last_pipeline_error = exc

        if result is None:
            yield _sse({"type": "status", "text": f"⚠️ Pipeline generation failed after retries. Running fallback pipeline: {last_pipeline_error}"})
            result = _fallback_pipeline_result(
                file_path=state.get("file_path", ""),
                target_column=target,
                problem_type=state.get("problem_type") or "classification",
            )

        feature_columns = result.get("feature_columns", [])

        # Stream reasoning
        if result.get("reasoning"):
            yield _sse({"type": "reasoning", "text": result["reasoning"]})

        # Execute each cell one at a time, with guard protection
        for cell in result.get("cells", []):
            title = cell.get("title", "Step")
            yield _sse({"type": "status", "text": f"📊 Executing: {title}…"})
            code = cell.get("code", "")
            stdout, error, charts = _exec(req.session_id, code)

            # ── Guard: auto-fix loop ──────────────────────────────────────────
            if error:
                fix_result = yield from _guard_fix(
                    agent, req.session_id, code, error, state, title
                )
                code, stdout, error, charts = fix_result
            # ── End guard ────────────────────────────────────────────────────

            yield _sse({
                "type": "code_cell",
                "code": code, "output": stdout, "error": error,
                "title": title, "description": cell.get("description", ""),
                "charts": charts,
            })
            # If the cell produced a chart, request an AI interpretation
            if charts and cell.get("has_chart"):
                try:
                    insight = agent.explain_chart(code, state)
                    yield _sse({"type": "insight", "text": insight})
                except Exception:
                    pass   # insight is optional

        # Persist updated state
        state["selected_model"]  = req.selected_model
        state["feature_columns"] = feature_columns
        state["stage"]           = "pipeline_built"

        yield _sse({
            "type": "done",
            "feature_columns": feature_columns,
            "target_column":   target,
        })

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── V2: Chat (with code-execution capability) ─────────────────────────────────

@app.post("/v2/chat")
def v2_chat(req: V2ChatRequest):
    """
    Decide whether the question needs an explanation or code execution.
    If code is needed, execute it (with guard auto-fix on failure) and return
    the code + output alongside the reply.
    """
    state   = v2_sessions.get(req.session_id, {})
    history = state.get("chat_history", [])

    try:
        agent  = _get_agent(req.model_id)
        result = agent.chat_with_code(req.message, state, history)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chat error: {exc}")

    code    = result.get("code", "")
    stdout, error, charts = "", None, []
    guard_events: list[dict] = []

    if result.get("action") == "execute" and code:
        stdout, error, charts = _exec(req.session_id, code)

        # ── Guard: auto-fix chat-generated code errors ────────────────────────
        if error:
            # Collect guard SSE events in a list (non-streaming endpoint)
            fixed_code, fixed_stdout, fixed_error, fixed_charts = code, stdout, error, charts
            ctx = state if state else {}
            for attempt in range(MAX_FIX_ATTEMPTS):
                guard_events.append({
                    "type": "guard_analyzing",
                    "title": result.get("title", "Chat Code"),
                    "error_preview": fixed_error[:400] if fixed_error else "",
                    "attempt": attempt + 1,
                })
                fix = agent.fix_error(fixed_code, fixed_error or "", ctx)
                if fix.get("needs_search") and fix.get("search_query"):
                    query = fix["search_query"]
                    guard_events.append({"type": "web_searching", "query": query})
                    sr = agent.web_search(query)
                    if sr:
                        fix = agent.fix_error(fixed_code, fixed_error or "", ctx, sr)

                new_fixed = (fix.get("fixed_code") or "").strip()
                explanation = fix.get("explanation", "")
                if not new_fixed or new_fixed == fixed_code:
                    break
                guard_events.append({"type": "fix_attempt", "code": new_fixed, "explanation": explanation, "attempt": attempt + 1})
                new_stdout, new_err, new_charts = _exec(req.session_id, new_fixed)
                if not new_err:
                    guard_events.append({"type": "fix_success", "title": result.get("title", "Chat Code"), "explanation": explanation})
                    fixed_code, fixed_stdout, fixed_error, fixed_charts = new_fixed, new_stdout, None, new_charts
                    break
                fixed_code, fixed_error = new_fixed, new_err
            else:
                guard_events.append({"type": "guard_give_up", "title": result.get("title", "Chat Code"), "error": (fixed_error or "")[:400]})

            code, stdout, error, charts = fixed_code, fixed_stdout, fixed_error, fixed_charts
        # ── End guard ─────────────────────────────────────────────────────────

        # Empty output fallback: if code ran successfully but produced nothing
        if not stdout and not error and not charts:
            reply = result.get("reply", "")
            result["reply"] = (reply + "\n\n⚠️ The code ran successfully but produced no visible output. "
                               "Try adding `print()` statements to see results.").strip()

        # If chart produced, append a graph interpretation to the reply
        if charts:
            try:
                insight = agent.explain_chart(code, state)
                result["chart_insight"] = insight
            except Exception:
                pass

    # Update history
    history = history + [
        {"role": "user",      "content": req.message},
        {"role": "assistant", "content": result.get("reply", "")},
    ]
    if req.session_id in v2_sessions:
        v2_sessions[req.session_id]["chat_history"] = history[-20:]

    return {
        "action":        result.get("action", "explain"),
        "reply":         result.get("reply", ""),
        "code":          code,
        "title":         result.get("title", ""),
        "is_chart":      result.get("is_chart", False),
        "output":        stdout,
        "error":         error,
        "charts":        charts,
        "chart_insight": result.get("chart_insight", ""),
        "guard_events":  guard_events,   # frontend can render these as guard bubbles
    }


# ── V2: Predict ───────────────────────────────────────────────────────────────

@app.post("/v2/predict")
def v2_predict(req: V2PredictRequest):
    """
    Generate and execute prediction code using the trained model.
    """
    state = v2_sessions.get(req.session_id)
    if not state:
        raise HTTPException(404, "No pipeline session found.")
    if state.get("stage") != "pipeline_built":
        raise HTTPException(400, "Train a model first by running the full pipeline.")

    try:
        agent = _get_agent(req.model_id)
        code  = agent.generate_predict(
            feature_columns=state.get("feature_columns", []),
            input_values=req.input_values,
            problem_type=state.get("problem_type", "classification"),
            target_column=state.get("target_column", "label"),
        )
        # Strip stray markdown fences the model might include
        code = code.strip().lstrip("```python").lstrip("```").rstrip("```").strip()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Could not generate prediction code: {exc}")

    stdout, error, _ = _exec(req.session_id, code)
    return {"code": code, "output": stdout, "error": error}


# ── V2: Available models ──────────────────────────────────────────────────────

@app.get("/v2/models")
def v2_models():
    """Return the list of AI models whose API keys are configured in .env."""
    return {"models": get_available_models()}


# ── V2: Context ───────────────────────────────────────────────────────────────

@app.get("/v2/context/{session_id}")
def v2_context(session_id: str):
    state = v2_sessions.get(session_id)
    if not state:
        raise HTTPException(404, "No v2 session found")
    return {k: v for k, v in state.items() if k != "chat_history"}


# ── V2: Python Script — generate ──────────────────────────────────────────────

class V2GenerateScriptRequest(BaseModel):
    session_id: str
    cells: list[str]   # raw code strings from each notebook cell

@app.post("/v2/generate-script")
def v2_generate_script(req: V2GenerateScriptRequest):
    """Combine notebook cells into a single standalone .py script."""
    state    = v2_sessions.get(req.session_id, {})
    filename = state.get("filename", "dataset.csv")

    header = (
        "# ═══════════════════════════════════════════════════════════════════\n"
        "# Auto-generated by Ownquesta\n"
        f"# Dataset : {filename}\n"
        "# Run     : python script.py\n"
        "# ═══════════════════════════════════════════════════════════════════\n\n"
    )

    body_parts: list[str] = []
    for i, code in enumerate(req.cells, 1):
        code = code.strip()
        if not code:
            continue
        body_parts.append(
            f"# ── Cell {i} {'─' * max(0, 60 - len(str(i)))}\n"
            f"{code}\n"
        )

    return {"script": header + "\n".join(body_parts)}


# ── V2: Python Script — execute (SSE streaming) ───────────────────────────────

class V2ExecuteScriptRequest(BaseModel):
    session_id: str
    script: str

@app.post("/v2/execute-script-stream")
def v2_execute_script_stream(req: V2ExecuteScriptRequest):
    """Execute the script in the existing kernel and stream output line-by-line."""
    def generate():
        yield _sse({"type": "status", "text": "▶ Running script…"})
        stdout, error, charts = _exec(req.session_id, req.script)
        for line in (stdout or "").splitlines():
            yield _sse({"type": "output", "line": line})
        if error:
            for line in error.splitlines():
                yield _sse({"type": "error_line", "line": line})
        yield _sse({"type": "done", "charts": charts})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── V2: Python Script — download as notebook (.ipynb) ─────────────────────────

class V2GenerateNotebookRequest(BaseModel):
    session_id: str
    cells: list[str]

@app.post("/v2/generate-notebook")
def v2_generate_notebook(req: V2GenerateNotebookRequest):
    """Return a minimal .ipynb JSON for the given cells."""
    from fastapi.responses import Response as FastAPIResponse

    state    = v2_sessions.get(req.session_id, {})
    filename = state.get("filename", "pipeline")
    stem     = Path(filename).stem

    nb_cells = []
    for code in req.cells:
        code = code.strip()
        if not code:
            continue
        nb_cells.append({
            "cell_type":       "code",
            "source":          code,
            "metadata":        {},
            "outputs":         [],
            "execution_count": None,
        })

    notebook = {
        "nbformat":       4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "cells": nb_cells,
    }
    nb_json = json.dumps(notebook, indent=2)
    return FastAPIResponse(
        content=nb_json,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{stem}_pipeline.ipynb"'},
    )


# ── V2: Python Script — AI chat for script modification ───────────────────────

class V2ScriptChatRequest(BaseModel):
    session_id: str
    message: str
    script: str
    model_id: Optional[str] = None

@app.post("/v2/script-chat")
def v2_script_chat(req: V2ScriptChatRequest):
    """AI assistant that can explain or rewrite the user's Python script."""
    try:
        agent = _get_agent(req.model_id)
    except HTTPException:
        raise

    state = v2_sessions.get(req.session_id, {})
    try:
        result = agent.modify_script(req.message, req.script, state)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Script chat error: {exc}")

    return {
        "reply":          result.get("reply", ""),
        "updated_script": result.get("updated_script") or req.script,
        "changed":        bool(result.get("updated_script")),
    }
