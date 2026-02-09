from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import importlib
import logging
from typing import Optional
from pathlib import Path
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
        router = getattr(mod, attr)
        logger.info(f"Successfully loaded router: {module_path}")
        return router
    except ImportError as e:
        logger.warning(f"Router module not available: {module_path} - {e}")
        return None
    except AttributeError as e:
        logger.error(f"Router attribute '{attr}' not found in {module_path} - {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading {module_path}: {e}")
        return None

# Load optional routers with better logging
logger.info("Loading optional router modules...")
ml_validation_router = try_import_router("ml_validation_agent.endpoint")
conversation_router = try_import_router("conversation_agent.endpoint")
eda_router = try_import_router("eda_agent.endpoint")

# Load ML Assistant Agent
logger.info("Loading ML Assistant Agent...")
ml_assistant_router = None

# Load ML Assistant router
try:
    from ml_assistant_agent.router import router as ml_assistant_router
    logger.info("ML Assistant router loaded successfully")
except ImportError as e:
    logger.error(f"Failed to import ML Assistant router: {e}")
    ml_assistant_router = None
except Exception as e:
    logger.error(f"Unexpected error loading ML Assistant router: {e}")
    ml_assistant_router = None

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
    
    # Check ML Assistant Agent
    if ml_assistant_router:
        agent_status["ml_assistant"] = "available"
    else:
        agent_status["ml_assistant"] = "unavailable"
    
    # Check other agents
    agent_status["ml_validation"] = "available" if ml_validation_router else "unavailable"
    agent_status["conversation"] = "available" if conversation_router else "unavailable"
    agent_status["eda"] = "available" if eda_router else "unavailable"
    
    overall_status = "ok" if any(status == "available" for status in agent_status.values()) else "degraded"
    
    return {
        "status": overall_status,
        "timestamp": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
        "agents": agent_status,
        "version": "1.0.0"
    }


def get_available_agents():
    """Get list of available agent endpoints"""
    agents = []
    
    if ml_assistant_router:
        agents.append({
            "name": "ML Assistant", 
            "prefix": "/ml-assistant",
            "description": "Advanced ML workflow assistance with goal-based task detection, EDA, validation, and AI chat",
            "features": [
                "Automatic ML task detection (classification, regression, clustering, anomaly detection)",
                "Goal-based model recommendations", 
                "Advanced dataset validation",
                "Comprehensive EDA with statistical analysis",
                "AI-powered chat assistance",
                "Code documentation generation"
            ],
            "endpoints": ["/upload", "/eda", "/validate", "/advanced-validate", "/goal-analyze", "/chat", "/docs", "/health", "/files"]
        })
    
    if ml_validation_router:
        agents.append({
            "name": "ML Validation",
            "prefix": "/ml-validation",
            "description": "Machine learning model validation services"
        })
    
    if conversation_router:
        agents.append({
            "name": "Conversation Agent",
            "prefix": "/conversation",
            "description": "AI-powered conversational assistant with RAG capabilities"
        })
    
    if eda_router:
        agents.append({
            "name": "EDA Agent", 
            "prefix": "/eda",
            "description": "Exploratory data analysis services"
        })
    
    return agents


@app.get("/meta.json")
def meta():
    """Legacy metadata endpoint for frontend compatibility"""
    endpoints = ["/", "/health", "/meta.json"]
    
    # Add available agent endpoints
    if ml_assistant_router:
        endpoints.append("/ml-assistant")
    if ml_validation_router:
        endpoints.append("/ml-validation")
    if conversation_router:
        endpoints.append("/conversation")
    if eda_router:
        endpoints.append("/eda")
    
    return {
        "name": "ownquesta-agent-api",
        "version": "1.0.0",
        "status": "ok",
        "endpoints": endpoints,
        "agents": get_available_agents()
    }


# Include routers for available agents
logger.info("Registering available agent routers...")

if ml_assistant_router is not None:
    app.include_router(
        ml_assistant_router, 
        prefix="/ml-assistant", 
        tags=["ML Assistant Agent"]
    )
    logger.info("ML Assistant Agent registered at /ml-assistant")

if ml_validation_router is not None:
    app.include_router(
        ml_validation_router, 
        prefix="/ml-validation", 
        tags=["ML Validation Agent"]
    )
    logger.info("ML Validation Agent registered at /ml-validation")

if conversation_router is not None:
    app.include_router(
        conversation_router, 
        prefix="/conversation", 
        tags=["Conversation Agent"]
    )
    logger.info("Conversation Agent registered at /conversation")

if eda_router is not None:
    app.include_router(
        eda_router, 
        prefix="/eda", 
        tags=["EDA Agent"]
    )
    logger.info("EDA Agent registered at /eda")

# Log startup summary
total_agents = sum([
    ml_assistant_router is not None,
    ml_validation_router is not None, 
    conversation_router is not None,
    eda_router is not None
])

logger.info(f"OwnQuesta Agent API started with {total_agents} active agents")
if total_agents == 0:
    logger.warning("No agents are available! Check your dependencies and configuration.")

# Add startup event
@app.on_event("startup")
async def startup_event():
    logger.info("OwnQuesta Agent API is starting up...")
    logger.info(f"Available agents: {[agent['name'] for agent in get_available_agents()]}")

# Add shutdown event  
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("OwnQuesta Agent API is shutting down...")