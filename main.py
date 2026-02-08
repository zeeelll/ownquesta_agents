from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ml_validation_agent.endpoint import router as ml_validation_router
from conversation_agent.endpoint import router as conversation_router
from eda_agent.endpoint import router as eda_router

app = FastAPI(
    title="OwnQuesta Agent API",
    description="AI-powered machine learning platform APIs including ML validation and conversation agents",
    version="0.0.1"
)

# Allow your Next.js frontend to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/hello")
def hello():
    return {"message": "Hello!"}


@app.get("/meta.json")
def meta():
    # Lightweight service metadata used by the frontend for a quick health probe.
    return {
        "name": "ownquesta-agent-api",
        "version": "0.0.1",
        "status": "ok",
        "endpoints": ["/hello", "/meta.json", "/ml-validation", "/conversation"],
    }

# Include routers
app.include_router(ml_validation_router, prefix="/ml-validation", tags=["ML Validation Agent"])
app.include_router(conversation_router, prefix="/conversation", tags=["OwnQuesta Conversation Agent"])
app.include_router(eda_router, prefix="/eda", tags=["EDA Agent"])