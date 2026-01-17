from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pathlib import Path
import tempfile
import os
from .models import ValidationResponse, DatasetSummary, GoalUnderstanding
from .graph import validation_graph
from .config import settings


router = APIRouter()

ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}

@router.post("/validate", response_model=ValidationResponse)
async def validate_dataset(
    goal: str = Form(..., description="User's ML goal statement"),
    file: UploadFile = File(..., description="CSV or XLSX dataset file")
):
    """
    Validate whether ML prediction is possible with the given dataset and goal.
    
    Returns:
        ValidationResponse with status (PROCEED/PAUSE/REJECT), scores, and recommendations
    """
    
    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Create temporary file
    temp_file = None
    try:
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            content = await file.read()
            
            # Check file size
            file_size_mb = len(content) / (1024 * 1024)
            if file_size_mb > settings.MAX_FILE_SIZE_MB:
                raise HTTPException(
                    status_code=400,
                    detail=f"File too large ({file_size_mb:.1f}MB). Max: {settings.MAX_FILE_SIZE_MB}MB"
                )
            
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Initialize agent state
        initial_state = {
            "goal": goal,
            "file_path": temp_file_path,
            "file_extension": file_ext,
            "df": None,
            "rows": 0,
            "columns": 0,
            "file_size_mb": 0.0,
            "column_types": {},
            "missing_percent": {},
            "interpreted_task": "unknown",
            "target_column_guess": None,
            "goal_confidence": 0.0,
            "validation_issues": [],
            "satisfaction_score": 0,
            "status": "REJECT",
            "clarification_questions": [],
            "optional_questions": [],
            "agent_answer": "",
            "user_view_report": "",
            "error": None
        }
        
        # Run LangGraph workflow
        result = validation_graph.invoke(initial_state)
        
        # Handle errors
        if result.get("error"):
            return ValidationResponse(
                status="REJECT",
                satisfaction_score=0,
                dataset_summary=DatasetSummary(
                    rows=0,
                    columns=0,
                    file_size_mb=0.0,
                    column_types={},
                    missing_percent={}
                ),
                goal_understanding=GoalUnderstanding(
                    interpreted_task="unknown",
                    target_column_guess=None,
                    confidence=0.0
                ),
                clarification_questions=[],
                optional_questions=[],
                agent_answer=f"**Error:** {result['error']}\n\nPlease check your file and try again.",
                user_view_report=f"# Validation Failed\n\nError: {result['error']}\n\nPlease ensure your file is a valid CSV or XLSX file with data."
            )
        
        # Build response
        response = ValidationResponse(
            status=result["status"],
            satisfaction_score=result["satisfaction_score"],
            dataset_summary=DatasetSummary(
                rows=result["rows"],
                columns=result["columns"],
                file_size_mb=result["file_size_mb"],
                column_types=result["column_types"],
                missing_percent=result["missing_percent"]
            ),
            goal_understanding=GoalUnderstanding(
                interpreted_task=result["interpreted_task"],
                target_column_guess=result["target_column_guess"],
                confidence=result["goal_confidence"]
            ),
            clarification_questions=result["clarification_questions"],
            optional_questions=result["optional_questions"],
            agent_answer=result["agent_answer"],
            user_view_report=result["user_view_report"]
        )
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
    
    finally:
        # Cleanup temporary file
        if temp_file and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except:
                pass