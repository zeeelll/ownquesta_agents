from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from .eda import summarize_df
from .validate import simple_validate
from .docs_gen import generate_docs
from . import config
from openai import OpenAI
from pydantic import BaseModel, Field
import time
import hashlib

# Setup logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)

# Initialize OpenAI client
openai_client = None
if config.OPENAI_API_KEY:
    try:
        openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
        logger.info("OpenAI client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
else:
    logger.warning("OpenAI client not initialized - API key not found")

app = FastAPI(
    title="ML Assistant Agent",
    description="Advanced ML Assistant for data analysis, validation, and documentation",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging middleware
@app.middleware("http")
async def log_requests(request, call_next):
    if config.ENABLE_REQUEST_LOGGING:
        start_time = time.time()
        logger.info(f"Request: {request.method} {request.url}")
        
    response = await call_next(request)
    
    if config.ENABLE_REQUEST_LOGGING:
        process_time = time.time() - start_time
        logger.info(f"Response: {response.status_code} in {process_time:.2f}s")
        
    return response

def validate_file_size(file: UploadFile) -> None:
    """Validate uploaded file size"""
    if hasattr(file, 'size') and file.size:
        if file.size > config.MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size is {config.MAX_FILE_SIZE_MB}MB"
            )

def validate_file_extension(filename: str) -> None:
    """Validate file extension"""
    if not filename:
        raise HTTPException(status_code=400, detail="No filename provided")
        
    extension = Path(filename).suffix.lower().lstrip('.')
    if extension not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(config.ALLOWED_EXTENSIONS)}"
        )

def get_file_path(filename: str) -> Path:
    """Get and validate file path"""
    path = config.UPLOAD_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return path


@app.get("/health")
def health():
    """Health check endpoint with system status"""
    openai_status = "available" if openai_client else "unavailable"
    
    return {
        "status": "ok",
        "agent": "ml_assistant",
        "version": "1.0.0",
        "openai_status": openai_status,
        "upload_dir": str(config.UPLOAD_DIR),
        "max_file_size_mb": config.MAX_FILE_SIZE_MB,
        "allowed_extensions": config.ALLOWED_EXTENSIONS
    }


@app.post("/start")
def start_process(confirm: bool = Form(False)):
    """Start the ML assistant process"""
    if not confirm:
        return {
            "message": "Please confirm to start the ML assistant process (send confirm=true).",
            "capabilities": [
                "Dataset upload and analysis",
                "Exploratory Data Analysis (EDA)",
                "Model validation",
                "Documentation generation",
                "AI-powered chat assistance"
            ]
        }
    
    logger.info("ML assistant process started")
    return {
        "message": "ML assistant started successfully!",
        "next_steps": [
            "Upload a dataset using /upload",
            "Perform EDA using /eda",
            "Validate models using /validate",
            "Get help using /docs or /chat"
        ]
    }


class UploadResponse(BaseModel):
    filename: str
    size_mb: float
    summary: Dict[str, Any]
    upload_time: str


@app.post("/upload", response_model=UploadResponse)
async def upload_dataset(file: UploadFile = File(...)):
    """Upload and analyze a dataset"""
    try:
        # Validate file
        validate_file_extension(file.filename)
        validate_file_size(file)
        
        # Save file
        dest = config.UPLOAD_DIR / file.filename
        
        # Read file content
        content = await file.read()
        size_mb = len(content) / (1024 * 1024)
        
        with dest.open("wb") as f:
            f.write(content)
        
        logger.info(f"File uploaded: {file.filename} ({size_mb:.2f}MB)")
        
        # Generate summary
        try:
            summary = summarize_df(str(dest))
            logger.info(f"Dataset summary generated for {file.filename}")
        except Exception as e:
            logger.error(f"Failed to generate summary for {file.filename}: {e}")
            raise HTTPException(status_code=422, detail=f"Error analyzing dataset: {str(e)}")
        
        return UploadResponse(
            filename=file.filename,
            size_mb=round(size_mb, 2),
            summary=summary,
            upload_time=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


class EDAResponse(BaseModel):
    filename: str
    eda: Dict[str, Any]
    processing_time: float


@app.post("/eda", response_model=EDAResponse)
def do_eda(filename: str = Form(...)):
    """Perform Exploratory Data Analysis on uploaded dataset"""
    try:
        start_time = time.time()
        path = get_file_path(filename)
        
        logger.info(f"Starting EDA for {filename}")
        summary = summarize_df(str(path))
        
        processing_time = time.time() - start_time
        logger.info(f"EDA completed for {filename} in {processing_time:.2f}s")
        
        return EDAResponse(
            filename=filename,
            eda=summary,
            processing_time=round(processing_time, 2)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"EDA failed for {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"EDA failed: {str(e)}")


class ValidationResponse(BaseModel):
    filename: str
    target_column: str
    result: Dict[str, Any]
    processing_time: float


@app.post("/validate", response_model=ValidationResponse)
def do_validate(filename: str = Form(...), target: str = Form(...)):
    """Validate ML model performance on dataset"""
    try:
        start_time = time.time()
        path = get_file_path(filename)
        
        if not target or not target.strip():
            raise HTTPException(status_code=400, detail="Target column name is required")
        
        logger.info(f"Starting validation for {filename} with target: {target}")
        result = simple_validate(str(path), target.strip())
        
        # Check if validation returned an error
        if isinstance(result, dict) and "error" in result:
            raise HTTPException(status_code=422, detail=result["error"])
        
        processing_time = time.time() - start_time
        logger.info(f"Validation completed for {filename} in {processing_time:.2f}s")
        
        return ValidationResponse(
            filename=filename,
            target_column=target.strip(),
            result=result,
            processing_time=round(processing_time, 2)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Validation failed for {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")


class DocsResponse(BaseModel):
    topic: str
    explanation: str
    code: str
    timestamp: str


@app.post("/docs", response_model=DocsResponse)
def docs(topic: str = Form(None)):
    """Generate documentation and code examples"""
    try:
        result = generate_docs(topic)
        
        return DocsResponse(
            topic=result["topic"],
            explanation=result["explanation"],
            code=result["code"],
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
    except Exception as e:
        logger.error(f"Documentation generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Documentation generation failed: {str(e)}")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    context: Optional[str] = Field(None, max_length=2000)


class ChatResponse(BaseModel):
    reply: str
    model: str
    timestamp: str
    tokens_used: Optional[int] = None


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """AI-powered chat assistance for ML and data science questions"""
    
    if not openai_client:
        raise HTTPException(
            status_code=503,
            detail="OpenAI service unavailable. Please check API key configuration."
        )
    
    try:
        # Prepare the messages
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert ML and data science assistant. "
                    "Provide helpful, accurate answers about machine learning, "
                    "data analysis, Python programming, and related topics. "
                    "Keep responses concise but informative."
                )
            }
        ]
        
        # Add context if provided
        if req.context:
            messages.append({
                "role": "user", 
                "content": f"Context: {req.context}"
            })
        
        # Add user message
        messages.append({
            "role": "user",
            "content": req.message
        })
        
        logger.info(f"Processing chat request: {req.message[:50]}...")
        
        # Call OpenAI API with new client
        response = openai_client.chat.completions.create(
            model=config.OPENAI_CHAT_MODEL,
            messages=messages,
            max_tokens=config.OPENAI_MAX_TOKENS,
            temperature=config.OPENAI_TEMPERATURE,
            n=1,
        )
        
        reply = response.choices[0].message.content
        tokens_used = response.usage.total_tokens if response.usage else None
        
        logger.info(f"Chat response generated (tokens: {tokens_used})")
        
        return ChatResponse(
            reply=reply,
            model=config.OPENAI_CHAT_MODEL,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            tokens_used=tokens_used
        )
        
    except Exception as e:
        logger.error(f"Chat request failed: {e}")
        raise HTTPException(status_code=500, detail=f"Chat request failed: {str(e)}")


@app.get("/files")
def list_files():
    """List uploaded files"""
    try:
        files = []
        for file_path in config.UPLOAD_DIR.glob("*"):
            if file_path.is_file():
                stat = file_path.stat()
                files.append({
                    "filename": file_path.name,
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))
                })
        
        return {"files": files, "upload_dir": str(config.UPLOAD_DIR)}
        
    except Exception as e:
        logger.error(f"Failed to list files: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list files: {str(e)}")


@app.delete("/files/{filename}")
def delete_file(filename: str):
    """Delete an uploaded file"""
    try:
        path = config.UPLOAD_DIR / filename
        if not path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        path.unlink()
        logger.info(f"File deleted: {filename}")
        
        return {"message": f"File {filename} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete file {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")


@app.get("/config")
def get_config():
    """Get current configuration (excluding sensitive data)"""
    return {
        "max_file_size_mb": config.MAX_FILE_SIZE_MB,
        "allowed_extensions": config.ALLOWED_EXTENSIONS,
        "default_test_size": config.DEFAULT_TEST_SIZE,
        "openai_model": config.OPENAI_CHAT_MODEL,
        "max_tokens": config.OPENAI_MAX_TOKENS,
        "temperature": config.OPENAI_TEMPERATURE,
        "min_samples": config.MIN_SAMPLES_FOR_VALIDATION,
        "min_features": config.MIN_FEATURES_FOR_VALIDATION,
        "version": "1.0.0"
    }
