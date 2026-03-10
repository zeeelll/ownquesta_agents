from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import importlib
import importlib.util
import logging
import os
import asyncio
from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

from services import start_services, stop_services

# Load environment variables from .env file
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_MAX_TOKENS = os.getenv('OPENAI_MAX_TOKENS') or os.getenv('MAX_TOKENS')
OPENAI_TEMPERATURE = os.getenv('OPENAI_TEMPERATURE') or os.getenv('TEMPERATURE')

_default_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000"
]
frontend_origins_env = os.getenv('FRONTEND_ORIGINS')
if frontend_origins_env:
    try:
        origins = [o.strip() for o in frontend_origins_env.split(',') if o.strip()]
    except Exception:
        origins = _default_origins
else:
    origins = _default_origins

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ownquesta_agents")


def try_import_router(module_path: str, attr: str = "router") -> Optional[object]:
    try:
        mod = importlib.import_module(module_path)
    except ImportError as e:
        logger.warning(f"Router module import failed: {module_path} - {e}")
        try:
            module_file = Path(__file__).parent / (module_path.replace(".", "/") + ".py")
            if module_file.exists():
                spec = importlib.util.spec_from_file_location(module_path, str(module_file))
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    logger.info(f"Loaded router module from file: {module_file}")
                else:
                    logger.warning(f"Could not create spec for module file: {module_file}")
                    return None
            else:
                logger.warning(f"Router module not available: {module_path} - {e}")
                return None
        except Exception as ex:
            logger.error(f"Fallback loader failed for {module_path}: {ex}")
            return None
    except Exception as e:
        logger.error(f"Unexpected error importing {module_path}: {e}")
        return None

    router = getattr(mod, attr, None)
    if router is None:
        logger.warning(f"Module '{module_path}' does not expose attribute '{attr}'")
        return None

    if not hasattr(router, "routes"):
        logger.error(f"Imported attribute '{attr}' from '{module_path}' is not a valid router")
        return None

    logger.info(f"Successfully loaded router: {module_path}.{attr}")
    return router


# ─────────────────────────────────────────────
# QUESTA AGENT — Loaded separately
# The router already has prefix="/questa" in endpoint.py
# So we include it WITHOUT adding a prefix here
# ─────────────────────────────────────────────
questa_router = try_import_router("questa_agent.endpoint", "router")
if questa_router:
    logger.info("✅  Questa AI Agent loaded successfully")
else:
    logger.warning("⚠️  Questa AI Agent not available — ensure questa_agent/ folder exists with endpoint.py and config.py")


# ─────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────
app = FastAPI(
    title="OwnQuesta Agent API",
    description="AI-powered machine learning platform APIs.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_available_agents() -> List[Dict[str, Any]]:
    agents: List[Dict[str, Any]] = []
    if questa_router:
        agents.append({"name": "Questa AI Assistant", "prefix": "/questa"})
    return agents


@app.get("/")
def root():
    return {
        "message": "Welcome to OwnQuesta Agent API",
        "version": "1.0.0",
        "documentation": "/docs",
        "available_agents": get_available_agents(),
        "health_check": "/health"
    }


@app.get("/health")
def health():
    overall = "ok" if questa_router else "degraded"

    return {
        "status": overall,
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "agents": {
            "questa_agent": "available" if questa_router else "unavailable"
        },
        "version": "1.0.0",
        "openai_configured": bool(OPENAI_API_KEY),
        "questa_agent_available": bool(questa_router),
        "questa_endpoints": {
            "chat":   "POST /questa/chat",
            "info":   "GET  /questa/info",
            "health": "GET  /questa/health",
        } if questa_router else None,
    }


@app.get("/meta.json")
def meta():
    endpoints = ["/", "/health", "/meta.json"]
    if questa_router:
        endpoints.extend(["/questa/chat", "/questa/info", "/questa/health"])
    return {
        "name": "ownquesta-agent-api",
        "version": "1.0.0",
        "status": "ok",
        "endpoints": endpoints,
        "agents": get_available_agents()
    }


# ─────────────────────────────────────────────
# REGISTER QUESTA ROUTER
# NOTE: No prefix here — endpoint.py already declares prefix="/questa"
# ─────────────────────────────────────────────
if questa_router is not None:
    try:
        app.include_router(questa_router)  # ← No prefix! Already set in endpoint.py
        logger.info("✅  Questa AI Assistant registered at /questa")
    except Exception as e:
        logger.exception(f"Failed to register Questa router: {e}")
else:
    logger.warning("⚠️  Questa AI Agent router not registered — agent unavailable")


@app.on_event("startup")
async def startup_event():
    logger.info("=" * 55)
    logger.info("  OwnQuesta Agent API Starting Up")
    logger.info("=" * 55)
    logger.info(f"Available agents: {[a['name'] for a in get_available_agents()]}")
    logger.info(f"OpenAI configured: {bool(OPENAI_API_KEY)}")
    logger.info(f"Questa AI Agent:   {'✅ Active' if questa_router else '❌ Unavailable'}")
    logger.info("=" * 55)

    # Start lab-backend and lab-agent services
    await asyncio.to_thread(start_services)


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("OwnQuesta Agent API is shutting down...")
    stop_services()