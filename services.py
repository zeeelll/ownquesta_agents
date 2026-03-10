"""services.py — Lifecycle manager for lab-backend and lab-agent sub-servers.

Automatically started by ownquesta_agents on startup and stopped on shutdown.
Both servers run as child processes of the main ownquesta_agents process.
"""

import sys
import time
import logging
import subprocess
from pathlib import Path

import httpx

logger = logging.getLogger("ownquesta_agents.services")

_BASE = Path(__file__).parent

LAB_BACKEND_DIR  = _BASE / "lab_backend_server"
LAB_AGENT_DIR    = _BASE / "lab_agent_server"

LAB_BACKEND_PORT = 8010
LAB_AGENT_PORT   = 8020

# Tracks running child processes so they can be stopped on shutdown.
_procs: list[subprocess.Popen] = []


def _wait_ready(url: str, retries: int = 20, delay: float = 0.5) -> bool:
    """Poll a /health endpoint until it responds, timing out after retries × delay seconds."""
    for _ in range(retries):
        try:
            r = httpx.get(url, timeout=2.0)
            if r.status_code < 500:
                return True
        except Exception:
            pass
        time.sleep(delay)
    return False


def start_services() -> None:
    """Start lab-backend then lab-agent as child processes.

    Called from the ownquesta_agents startup event (run in a thread so it
    does not block the asyncio event loop).
    """
    global _procs

    # 1. Start lab-backend (code-execution kernel, no external deps)
    backend_proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "main:app",
            "--host", "127.0.0.1",
            "--port", str(LAB_BACKEND_PORT),
        ],
        cwd=str(LAB_BACKEND_DIR),
    )
    _procs.append(backend_proc)
    logger.info(
        "Started lab-backend (pid=%d) on port %d",
        backend_proc.pid,
        LAB_BACKEND_PORT,
    )

    if _wait_ready(f"http://127.0.0.1:{LAB_BACKEND_PORT}/health"):
        logger.info("lab-backend is ready.")
    else:
        logger.warning(
            "lab-backend did not respond in time — "
            "lab-agent may fail to execute code cells."
        )

    # 2. Start lab-agent (AI orchestration, depends on lab-backend)
    agent_proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "main:app",
            "--host", "127.0.0.1",
            "--port", str(LAB_AGENT_PORT),
        ],
        cwd=str(LAB_AGENT_DIR),
    )
    _procs.append(agent_proc)
    logger.info(
        "Started lab-agent (pid=%d) on port %d",
        agent_proc.pid,
        LAB_AGENT_PORT,
    )

    if _wait_ready(f"http://127.0.0.1:{LAB_AGENT_PORT}/health"):
        logger.info("lab-agent is ready.")
    else:
        logger.warning("lab-agent did not respond in time.")


def stop_services() -> None:
    """Terminate all managed child processes (called on shutdown)."""
    for proc in _procs:
        try:
            proc.terminate()
            proc.wait(timeout=5)
            logger.info("Stopped process pid=%d", proc.pid)
        except Exception as exc:
            logger.warning("Error stopping pid=%d: %s", proc.pid, exc)
    _procs.clear()
