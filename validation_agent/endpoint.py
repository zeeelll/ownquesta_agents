from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional, Dict, Any
from .config import perform_advanced_eda_from_csv_text, analyze_user_question, perform_ml_validation_from_eda, generate_ai_insights, generate_ml_ai_insights
import json
from pathlib import Path
from fastapi.responses import PlainTextResponse

router = APIRouter()


class ValidationRequest(BaseModel):
    csv_text: str
    goal: Optional[Dict[str, Any]] = None


class QuestionRequest(BaseModel):
    question: str
    eda_results: Dict[str, Any]


class AskRequest(BaseModel):
    csv_text: str
    question: str
    goal: Optional[Dict[str, Any]] = None


@router.post("/analyze")
async def analyze_csv(req: ValidationRequest):
    """Analyze CSV text and return an enhanced EDA summary produced by the validation agent."""
    try:
        print(f"DEBUG: Received request with csv_text length: {len(req.csv_text)}")
        result = perform_advanced_eda_from_csv_text(req.csv_text, goal=req.goal or {})
        print(f"DEBUG: EDA completed, result keys: {list(result.keys())}")

        # Add AI-powered EDA insights when available (graceful fallback if OpenAI not configured)
        try:
            ai_insights = generate_ai_insights(result)
            result['aiInsights'] = ai_insights
        except Exception as e:
            print(f"DEBUG: generate_ai_insights failed: {e}")

        # Also perform ML validation based on EDA so callers get both EDA + ML validation report
        try:
            ml_result = perform_ml_validation_from_eda(result)
            # Add ML-specific AI insights as well
            try:
                ml_ai = generate_ml_ai_insights(result, ml_result)
                ml_result['aiInsights'] = ml_ai
            except Exception as e:
                print(f"DEBUG: generate_ml_ai_insights failed: {e}")
        except Exception as e:
            print(f"DEBUG: ML validation failed during analyze: {e}")
            ml_result = {"error": str(e)}

        response_data = {"status": "success", "result": result, "ml_result": ml_result}
        print(f"DEBUG: Response data created, size: {len(str(response_data))}")
        return response_data
    except Exception as e:
        print(f"DEBUG: Error in analyze_csv: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error performing validation: {str(e)}"
        )


@router.post("/ml_validate")
async def ml_validate(req: ValidationRequest):
    """Perform ML validation based on EDA results."""
    try:
        # First perform EDA
        eda_result = perform_advanced_eda_from_csv_text(req.csv_text, goal=req.goal or {})

        # Add AI-powered EDA insights when available
        try:
            eda_result['aiInsights'] = generate_ai_insights(eda_result)
        except Exception as e:
            print(f"DEBUG: generate_ai_insights failed in ml_validate: {e}")

        # Then perform ML validation
        ml_result = perform_ml_validation_from_eda(eda_result)

        # Add ML AI insights (strategy/recommendations)
        try:
            ml_result['aiInsights'] = generate_ml_ai_insights(eda_result, ml_result)
        except Exception as e:
            print(f"DEBUG: generate_ml_ai_insights failed in ml_validate: {e}")

        return {
            "status": "success",
            "eda_result": eda_result,
            "ml_result": ml_result
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error performing ML validation: {str(e)}"
        )


@router.post("/validate")
async def validate_upload(file: UploadFile = File(...), goal: Optional[str] = Form(None)):
    """Accept a multipart file upload (CSV) and optional goal form field, run enhanced EDA and return result."""
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

        # Add AI-powered EDA insights when available
        try:
            result['aiInsights'] = generate_ai_insights(result)
        except Exception as e:
            print(f"DEBUG: generate_ai_insights failed in validate_upload: {e}")

        # Also run ML validation so the caller receives EDA + ML report in one response
        try:
            ml_result = perform_ml_validation_from_eda(result)
            try:
                ml_result['aiInsights'] = generate_ml_ai_insights(result, ml_result)
            except Exception as e:
                print(f"DEBUG: generate_ml_ai_insights failed in validate_upload: {e}")
        except Exception as e:
            ml_result = {"error": str(e)}

        return {"status": "success", "result": result, "ml_result": ml_result}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error performing validation upload: {str(e)}"
        )


@router.get("/ui")
async def get_ui_component():
    """Return the raw React component source for the Validation Agent UI."""
    try:
        p = Path(__file__).parent / 'ui_component.jsx'
        if not p.exists():
            raise HTTPException(status_code=404, detail='UI component not found')
        text = p.read_text(encoding='utf-8')
        return PlainTextResponse(text, media_type='text/javascript')
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/ask")
async def ask_question(req: AskRequest):
    """Answer a user question given CSV text by running EDA then an analysis function."""
    try:
        # Run EDA to build context
        eda_result = perform_advanced_eda_from_csv_text(req.csv_text, goal=req.goal or {})
        # Use rule-based/question analyzer (or AI if configured)
        answer = analyze_user_question(req.question, eda_result)
        return {"status": "success", "answer": answer, "eda": eda_result}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/question")
async def question_endpoint(req: QuestionRequest):
    """Accept a question plus existing EDA results and return an answer."""
    try:
        answer = analyze_user_question(req.question, req.eda_results)
        return {"status": "success", "answer": answer}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
