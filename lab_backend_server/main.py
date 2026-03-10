import uuid
import json
import re
import logging
import subprocess
import sys
import threading
import tempfile
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Lab Backend")

# configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lab-backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions: dict[str, subprocess.Popen] = {}

EXEC_TIMEOUT = 120
EXECUTOR_PATH = Path(__file__).parent / "worker" / "executor.py"

# ── Server-side code restrictions ─────────────────────────────────
BLOCKED: list[tuple[re.Pattern, str]] = [
    (re.compile(r'(!|%)?\bpip3?\s+(install|download)', re.I),
     "Package installation is not allowed. Use built-in Python libraries."),
    (re.compile(r'\b(conda|apt-get?|brew|npm|yarn)\s+install', re.I),
     "System package installation is not allowed."),
    (re.compile(r'\b(import|from)\s+(tensorflow|tf|torch|keras|jax|mxnet|paddle|caffe|theano|cntk|transformers|diffusers|ultralytics)\b', re.I),
     "Heavy deep-learning libraries are not available in this playground."),
    (re.compile(r'\b__import__\s*\(', re.I),
     "Dynamic imports via __import__ are not allowed."),
    (re.compile(r'\bos\s*\.\s*system\s*\(', re.I),
     "os.system() calls are not allowed."),
    (re.compile(r'\bsubprocess\b', re.I),
     "subprocess module is not allowed."),
]


def validate_code(code: str) -> str | None:
    for pattern, message in BLOCKED:
        if pattern.search(code):
            return message
    return None


def _infer_schema(obj):
    """Infer a simple JSON-compatible schema from a Python object.
    Produces a mapping of field -> {type, example} for dicts and recurses.
    """
    if obj is None:
        return {"type": "null", "example": None}
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out[k] = _infer_schema(v)
        return {"type": "object", "properties": out}
    if isinstance(obj, list):
        items = [_infer_schema(obj[0])] if obj else [{"type": "null"}]
        return {"type": "array", "items": items}
    t = type(obj)
    if t is bool:
        return {"type": "boolean", "example": obj}
    if t in (int, float):
        return {"type": "number", "example": obj}
    if t is str:
        return {"type": "string", "example": obj}
    return {"type": "string", "example": str(obj)}


class ExecuteRequest(BaseModel):
    session_id: str
    cell_id: str
    code: str


def _read_line_timeout(proc: subprocess.Popen, timeout: float) -> bytes | None:
    result: list[bytes | None] = [None]

    def _reader():
        result[0] = proc.stdout.readline()

    t = threading.Thread(target=_reader, daemon=True)
    t.start()
    t.join(timeout)
    return result[0] if not t.is_alive() else None


@app.post("/session")
def create_session():
    session_id = str(uuid.uuid4())
    proc = subprocess.Popen(
        [sys.executable, "-u", str(EXECUTOR_PATH)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    sessions[session_id] = proc
    return {"session_id": session_id}


@app.get("/health")
def health():
    return {"status": "ok", "sessions": len(sessions)}


@app.post("/execute")
def execute_cell(req: ExecuteRequest):
    # Log the declared Pydantic model schema and an instance-derived schema
    try:
        logger.info("ExecuteRequest model schema: %s", json.dumps(ExecuteRequest.schema(), indent=2))
    except Exception:
        logger.exception("Failed to serialize ExecuteRequest.model schema")
    try:
        logger.info("ExecuteRequest instance schema: %s", json.dumps(_infer_schema(req.dict()), indent=2))
    except Exception:
        logger.exception("Failed to infer ExecuteRequest instance schema")

    violation = validate_code(req.code)
    if violation:
        return {"stdout": "", "error": f"Restricted: {violation}"}

    proc = sessions.get(req.session_id)
    if proc is None or proc.poll() is not None:
        # ── Auto-recover: re-create the Python worker instead of returning 404 ──
        # This happens when the server restarts and loses in-memory session state.
        proc = subprocess.Popen(
            [sys.executable, "-u", str(EXECUTOR_PATH)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        sessions[req.session_id] = proc

    payload = json.dumps({"code": req.code}) + "\n"
    try:
        proc.stdin.write(payload.encode())
        proc.stdin.flush()
    except (BrokenPipeError, OSError):
        raise HTTPException(status_code=500, detail="Worker process died")

    raw = _read_line_timeout(proc, EXEC_TIMEOUT)
    if raw is None:
        # The worker is stuck — kill it and restart a fresh one so the session
        # stays usable. Without this, the next execution reads the stale result
        # from this timed-out run and the session falls permanently out of sync.
        try:
            proc.kill()
            proc.wait(timeout=3)
        except Exception:
            pass
        new_proc = subprocess.Popen(
            [sys.executable, "-u", str(EXECUTOR_PATH)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        sessions[req.session_id] = new_proc
        return {
            "stdout": "",
            "error": (
                f"Execution timed out ({EXEC_TIMEOUT}s limit). "
                "Session kernel was reset — all variables cleared. "
                "The guard agent will retry with more efficient code."
            ),
        }
    if not raw:
        return {"stdout": "", "error": "Worker returned empty response"}

    try:
        result = json.loads(raw.decode())
    except json.JSONDecodeError:
        return {"stdout": "", "error": f"Malformed response: {raw.decode()[:200]}"}

    return {
        "stdout": result.get("stdout", ""),
        "error":  result.get("error"),
        "charts": result.get("charts", []),   # list of base64-encoded PNG strings
    }


@app.post("/upload")
async def upload_dataset(
    session_id: str = Form(...),
    file: UploadFile = File(...),
):
    """
    Accept a CSV or Excel file upload, save it to a per-session temp folder,
    and return the absolute file path so the agent can pass it to codegen.
    """
    # Log what the caller sent (form fields + file metadata) as an inferred schema
    try:
        logger.info("Upload request model/schema: %s", json.dumps(_infer_schema({
            "session_id": session_id,
            "filename": file.filename,
            "content_type": getattr(file, "content_type", None),
        }), indent=2))
    except Exception:
        logger.exception("Failed to log upload request schema")

    allowed_extensions = {".csv", ".xlsx", ".xls"}
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Upload a .csv, .xlsx, or .xls file.",
        )

    upload_dir = Path(tempfile.gettempdir()) / "lab_uploads" / session_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    save_path = upload_dir / (file.filename or f"dataset{suffix}")
    content = await file.read()
    save_path.write_bytes(content)

    return {
        "filename": file.filename,
        "file_path": save_path.as_posix(),   # forward-slash path, works on all OS
        "size_kb": round(len(content) / 1024, 1),
    }


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    proc = sessions.pop(session_id, None)
    if proc:
        proc.terminate()
    return {"ok": True}
