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
        logger.warning(f"Router module not available: {module_path} - {e}")
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

# Agent registry and controlled loading
AGENT_REGISTRY = {
    
    "conversation": {"module": "conversation_agent.endpoint", "attr": "router", "prefix": "/conversation", "name": "Conversation Agent"},
}

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
conversation_router = agent_routers.get("conversation")

# Create main FastAPI application
app = FastAPI(
    title="OwnQuesta Agent API",
    description="AI-powered machine learning platform APIs including ML validation, conversation agents, and comprehensive ML assistance",
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
    
    # Check other agents
    agent_status["conversation"] = "available" if conversation_router else "unavailable"
    
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
    
    
    
    if conversation_router:
        agents.append({
            "name": "Conversation Agent",
            "prefix": "/conversation",
            "description": "AI-powered conversational assistant with RAG capabilities"
        })
    
    
    return agents


@app.get("/meta.json")
def meta():
    """Legacy metadata endpoint for frontend compatibility"""
    endpoints = ["/", "/health", "/meta.json"]
    
    # Add available agent endpoints
    if conversation_router:
        endpoints.append("/conversation")
    
    return {
        "name": "ownquesta-agent-api",
        "version": "1.0.0",
        "status": "ok",
        "endpoints": endpoints,
        "agents": get_available_agents()
    }

# Log startup summary
total_agents = sum([ 
    conversation_router is not None,
])

# Add startup event
@app.on_event("startup")
async def startup_event():
    logger.info("OwnQuesta Agent API is starting up...")
    logger.info(f"Available agents: {[agent['name'] for agent in get_available_agents()]}")

# Add shutdown event  
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("OwnQuesta Agent API is shutting down...")