from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import importlib
import logging
import os
from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ownquesta_agents")

def try_import_router(module_path: str, attr: str = "router") -> Optional[object]:
    """
    Safely import router modules with proper error handling
    
    Args:
        module_path: Python module path to import
        attr: Attribute name to get from module (default: "router")
        
    Returns:
        Router object if successful, None otherwise
    """
    try:
        mod = importlib.import_module(module_path)
    except ImportError as e:
        logger.warning(f"Router module import failed: {module_path} - {e}")
        # Fallback: attempt to load module by file path (useful when running from different CWDs)
        try:
            module_file = Path(__file__).parent / (module_path + ".py")
            if module_file.exists():
                spec = importlib.util.spec_from_file_location(module_path, str(module_file))
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)  # type: ignore
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

    # Basic validation: ensure the object exposes 'routes' (APIRouter-compatible)
    try:
        if not hasattr(router, "routes"):
            logger.error(f"Imported attribute '{attr}' from {module_path}' is not a valid router (missing 'routes')")
            return None
    except Exception as e:
        logger.error(f"Error validating router from {module_path}: %s", e)
        return None

    logger.info(f"Successfully loaded router: {module_path}.{attr}")
    return router

# Agent registry and controlled loading - dynamically discovered
from pathlib import Path

def discover_agents():
    """Dynamically discover available agents in the agents directory."""
    agents_dir = Path(__file__).parent
    agent_registry = {}
    
    for item in agents_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.') and item.name not in ['__pycache__', 'data', 'scripts']:
            endpoint_file = item / 'endpoint.py'
            if endpoint_file.exists():
                key = item.name
                # Use the part before '_agent' if present, else the name
                prefix_name = item.name.replace('_agent', '') if '_agent' in item.name else item.name
                agent_registry[key] = {
                    "module": f"{item.name}.endpoint",
                    "attr": "router",
                    "prefix": f"/{prefix_name}",
                    "name": f"{prefix_name.replace('_', ' ').title()} Agent"
                }
                logger.info(f"Discovered agent: {key} at {item.name}")
    
    return agent_registry

AGENT_REGISTRY = discover_agents()

# Read ENABLED_AGENTS from environment (comma-separated keys from AGENT_REGISTRY)
enabled_env = os.getenv("ENABLED_AGENTS")
if enabled_env:
    enabled_keys = {k.strip() for k in enabled_env.split(",") if k.strip()}
    logger.info("ENABLED_AGENTS set; attempting to load: %s", enabled_keys)
else:
    enabled_keys = set(AGENT_REGISTRY.keys())
    logger.info("Loading optional router modules... (all enabled)")

# Attempt to import configured agents
agent_routers: Dict[str, Optional[object]] = {}
for key, meta in AGENT_REGISTRY.items():
    if key not in enabled_keys:
        agent_routers[key] = None
        logger.info("Skipping agent '%s' (disabled by ENABLED_AGENTS)", key)
        continue
    agent_routers[key] = try_import_router(meta["module"], meta.get("attr", "router"))

# Expose variables used elsewhere for backwards compatibility
# Map commonly-used agent keys to discovered routers. discovery uses folder names
# like 'validation_agent' or 'conversation_agent', so match by substring.
conversation_router = next((r for k, r in agent_routers.items() if r is not None and 'conversation' in k), None)
validation_router = next((r for k, r in agent_routers.items() if r is not None and 'validation' in k), None)

# Create main FastAPI application
app = FastAPI(
    title="OwnQuesta Agent API",
    description="AI-powered machine learning platform APIs including advanced ML validation with comprehensive EDA (shape, size, statistical summaries, data distribution, correlations, insights), conversation agents, and comprehensive ML assistance",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to OwnQuesta Agent API",
        "version": "1.0.0",
        "documentation": "/docs",
        "available_agents": get_available_agents(),
        "health_check": "/health"
    }


@app.get("/health")
def health():
    """Comprehensive health check for all agents"""
    agent_status = {}
    
    # Check all discovered agents
    for key, router in agent_routers.items():
        agent_status[key] = "available" if router else "unavailable"
    
    overall_status = "ok" if any(status == "available" for status in agent_status.values()) else "degraded"

    return {
        "status": overall_status,
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "agents": agent_status,
        "version": "1.0.0"
    }


def get_available_agents() -> List[Dict[str, Any]]:
    """Get list of available agent endpoints"""
    agents: List[Dict[str, Any]] = []
    
    for key, meta in AGENT_REGISTRY.items():
        if key in enabled_keys and agent_routers.get(key):
            agents.append({
                "name": meta["name"],
                "prefix": meta["prefix"],
                "description": f"AI-powered {meta['name'].lower()} for advanced ML workflows"
            })
    
    return agents


@app.get("/meta.json")
def meta():
    """Legacy metadata endpoint for frontend compatibility"""
    endpoints = ["/", "/health", "/meta.json"]
    
    # Add available agent endpoints
    for key, router in agent_routers.items():
        if router:
            endpoints.append(AGENT_REGISTRY[key]["prefix"])
    
    return {
        "name": "ownquesta-agent-api",
        "version": "1.0.0",
        "status": "ok",
        "endpoints": endpoints,
        "agents": get_available_agents()
    }


# Include routers for available agents (safe registration)
logger.info("Registering available agent routers...")
for key, router in agent_routers.items():
    if router is not None:
        try:
            meta = AGENT_REGISTRY[key]
            app.include_router(
                router,
                prefix=meta["prefix"],
                tags=[meta["name"]],
            )
            logger.info(f"{meta['name']} registered at {meta['prefix']}")
        except Exception as e:
            logger.exception(f"Failed to register {key} router: %s", e)

# Log startup summary
total_agents = sum(router is not None for router in agent_routers.values())

# Add startup event
@app.on_event("startup")
async def startup_event():
    logger.info("OwnQuesta Agent API is starting up...")
    logger.info(f"Available agents: {[agent['name'] for agent in get_available_agents()]}")

# Add shutdown event  
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("OwnQuesta Agent API is shutting down...")