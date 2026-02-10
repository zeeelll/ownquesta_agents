from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional, Dict, Any
from validation_agent import perform_advanced_eda_from_csv_text
import json

router = APIRouter()


class ValidationRequest(BaseModel):
    csv_text: str
    goal: Optional[Dict[str, Any]] = None


@router.post("/analyze")
async def analyze_csv(req: ValidationRequest):
    """Analyze CSV text and return an EDA summary produced by the validation agent."""
    try:
        result = perform_advanced_eda_from_csv_text(req.csv_text, goal=req.goal or {})
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error performing validation: {str(e)}"
        )


@router.post("/validate")
async def validate_upload(file: UploadFile = File(...), goal: Optional[str] = Form(None)):
    """Accept a multipart file upload (CSV) and optional goal form field, run EDA and return result."""
    try:
        raw = await file.read()
        text = raw.decode('utf-8', errors='ignore')

        parsed_goal = {}
        if goal:
            try:
                parsed_goal = json.loads(goal)
            except Exception:
                parsed_goal = {"description": goal}

        result = perform_advanced_eda_from_csv_text(text, goal=parsed_goal)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error performing validation upload: {str(e)}"
        )
