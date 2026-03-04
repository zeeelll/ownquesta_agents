import uuid
import httpx
from state import MLState

LAB_BACKEND_URL = "http://localhost:8010"


def _do_execute(client: httpx.Client, session_id: str, code: str) -> dict:
    resp = client.post(
        f"{LAB_BACKEND_URL}/execute",
        json={"session_id": session_id, "cell_id": str(uuid.uuid4()), "code": code},
    )
    resp.raise_for_status()
    return resp.json()


def executor_node(state: MLState) -> dict:
    """
    Sends last_code to lab-backend /execute and stores stdout in last_output.
    Auto-recovers if the backend session was lost (e.g. after a server restart):
    the backend now auto-creates a fresh Python worker on any missing session,
    so a simple retry is enough.
    """
    code  = state.get("last_code", "")
    stage = state.get("stage", "")

    output = ""
    try:
        with httpx.Client(timeout=60.0) as client:
            try:
                data = _do_execute(client, state["session_id"], code)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code in (404, 500):
                    # Backend lost the session — retry once (backend auto-recreates it)
                    data = _do_execute(client, state["session_id"], code)
                else:
                    raise

        stdout = data.get("stdout") or ""
        error  = data.get("error")
        output = f"ERROR:\n{error}" if error else stdout

    except Exception as exc:
        output = f"ERROR: could not reach lab-backend — {exc}"

    updates: dict = {"last_output": output}

    # Advance persistent state flags after successful execution
    if stage == "load_dataset":
        updates["dataset_loaded"] = True
    elif stage == "inspect_dataset":
        # Default to classification; frontend/user can override via target_column
        updates["problem_type"] = "classification"

    return updates
