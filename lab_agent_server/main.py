from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, Generator
import os
import uuid
import json
import logging
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

import httpx

from state import MLState
from graph import step_graph
from agents import MLAgent

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
MAX_FIX_ATTEMPTS = 3   # guard retry budget per failing cell


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


def _get_agent() -> MLAgent:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY not set. Create lab-agent/.env with OPENAI_API_KEY=sk-...",
        )
    return MLAgent(api_key=key)


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
    for attempt in range(MAX_FIX_ATTEMPTS):
        # Notify frontend that guard is analysing
        yield _sse({
            "type": "guard_analyzing",
            "title": title,
            "error_preview": error[:400],
            "attempt": attempt + 1,
        })

        fix = agent.fix_error(code, error, context)

        # Optionally escalate to Tavily web search
        if fix.get("needs_search") and fix.get("search_query"):
            query = fix["search_query"]
            yield _sse({"type": "web_searching", "query": query})
            sr = agent.web_search(query)
            if sr:
                fix = agent.fix_error(code, error, context, sr)

        fixed = (fix.get("fixed_code") or "").strip()
        explanation = fix.get("explanation", "")

        # Guard couldn't produce a different fix — bail out
        if not fixed or fixed == code:
            break

        yield _sse({"type": "fix_attempt", "code": fixed, "explanation": explanation, "attempt": attempt + 1})

        new_stdout, new_err, new_charts = _exec(session_id, fixed)
        if not new_err:
            yield _sse({"type": "fix_success", "title": title, "explanation": explanation})
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

class V2BuildPipelineRequest(BaseModel):
    session_id: str
    selected_model: str
    target_column: Optional[str] = None

class V2ChatRequest(BaseModel):
    session_id: str
    message: str

class V2PredictRequest(BaseModel):
    session_id: str
    input_values: dict   # {column_name: value}


# ── V2: Analyse (SSE streaming) ───────────────────────────────────────────────

@app.post("/v2/analyze-stream")
def v2_analyze_stream(req: V2AnalyzeRequest):
    """
    SSE endpoint that streams the analysis step-by-step:
      status → code_cell (profile) → status → analysis → fe_cell [guard] → models → done
    """
    def generate():
        try:
            agent = _get_agent()
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
        try:
            result = agent.analyze(req.uploaded_filename, profile_output, req.target_column)
        except Exception as exc:
            yield _sse({"type": "error", "text": f"AI analysis failed: {exc}"})
            yield _sse({"type": "done"}); return

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

            eda_summary_data = {
                "summary": eda_result.get("summary", ""),
                "feature_importance": eda_result.get("feature_importance_notes", ""),
                "preprocessing": eda_result.get("preprocessing_recommendations", ""),
            }
            yield _sse({"type": "eda_summary", "data": eda_summary_data})
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
            agent = _get_agent()
        except HTTPException as exc:
            yield _sse({"type": "error", "text": exc.detail}); yield _sse({"type": "done"}); return

        target = req.target_column or state.get("target_column") or "label"
        yield _sse({"type": "status", "text": f"🧠 Planning pipeline with {req.selected_model}…"})

        try:
            result = agent.build_pipeline(
                filename=state["filename"],
                problem_type=state.get("problem_type") or "classification",
                target_column=target,
                model_name=req.selected_model,
                profile_output=state.get("profile_output", ""),
                fe_code=state.get("feature_engineering_code", ""),
            )
        except Exception as exc:
            yield _sse({"type": "error", "text": f"Pipeline generation failed: {exc}"})
            yield _sse({"type": "done"}); return

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
        agent  = _get_agent()
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
        agent = _get_agent()
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


# ── V2: Context ───────────────────────────────────────────────────────────────

@app.get("/v2/context/{session_id}")
def v2_context(session_id: str):
    state = v2_sessions.get(session_id)
    if not state:
        raise HTTPException(404, "No v2 session found")
    return {k: v for k, v in state.items() if k != "chat_history"}
