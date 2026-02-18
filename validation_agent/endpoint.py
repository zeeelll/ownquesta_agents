from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional, Dict, Any, Union
from .config import perform_advanced_eda_from_csv_text, analyze_user_question, perform_ml_validation_from_eda, generate_ai_insights, generate_ml_ai_insights
import json
import base64
from pathlib import Path
from fastapi.responses import PlainTextResponse

router = APIRouter()


class ValidationRequest(BaseModel):
    csv_text: Optional[str] = None
    file_data: Optional[str] = None  # Base64 encoded file data for Excel
    file_type: Optional[str] = 'csv'  # 'csv' or 'excel'
    filename: Optional[str] = None
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
    """Analyze CSV or Excel file and return an enhanced EDA summary produced by the validation agent."""
    try:
        # Handle different input formats
        data_input = None
        
        if req.file_type == 'excel' and req.file_data:
            # Decode base64 Excel data
            print(f"DEBUG: Received Excel file (base64 length: {len(req.file_data)})")
            file_bytes = base64.b64decode(req.file_data)
            data_input = file_bytes
            print(f"DEBUG: Decoded to {len(file_bytes)} bytes")
        elif req.csv_text:
            # CSV text
            print(f"DEBUG: Received CSV text (length: {len(req.csv_text)})")
            data_input = req.csv_text
        else:
            raise HTTPException(status_code=400, detail="Either csv_text or file_data must be provided")
        
        result = perform_advanced_eda_from_csv_text(data_input, goal=req.goal or {})
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

        # Provide consistent top-level keys for callers: `eda_result` and `ml_result`.
        # Also include a human-readable agent_answer and a user_view_report markdown for display.
        agent_answer = None
        if isinstance(ml_result, dict) and ml_result.get('agent_answer'):
            agent_answer = ml_result.get('agent_answer')
        elif isinstance(result, dict) and result.get('aiInsights'):
            agent_answer = result.get('aiInsights')
        else:
            # Fallback short summary
            agent_answer = f"EDA completed: {result.get('shape', {}).get('rows', 'N/A')} rows × {result.get('shape', {}).get('columns', 'N/A')} columns."

        # Build a simple user-facing markdown report combining EDA insights and ML summary
        try:
            report_lines = ["# Validation Report", ""]
            report_lines.append(f"**Dataset:** {result.get('shape', {}).get('rows', 'N/A')} rows × {result.get('shape', {}).get('columns', 'N/A')} columns")
            if result.get('insights'):
                report_lines.append('\n## Key EDA Insights')
                for k, v in result.get('insights', {}).items():
                    if isinstance(v, list) and v:
                        report_lines.append(f"- **{k}**: {('; '.join(v))}")
            if isinstance(ml_result, dict):
                if ml_result.get('modelRecommendations'):
                    report_lines.append('\n## Model Recommendations')
                    for m in ml_result.get('modelRecommendations', [])[:5]:
                        report_lines.append(f"- {m.get('algorithm', m.get('type', 'Model'))}: {m.get('use_case') or m.get('description') or ''}")
                if ml_result.get('performanceEstimates'):
                    pe = ml_result.get('performanceEstimates')
                    report_lines.append('\n## Performance Estimates')
                    report_lines.append(f"- Confidence: {pe.get('confidence')}")
                    report_lines.append(f"- Expected Accuracy: {pe.get('expected_accuracy')}")

            user_view_report = '\n'.join(report_lines)
        except Exception as e:
            user_view_report = None

        response_data = {"status": "success", "result": result, "eda_result": result, "ml_result": ml_result, "agent_answer": agent_answer, "user_view_report": user_view_report}
        print(f"DEBUG: Response data created, size: {len(str(response_data))}")
        # Expose convenient top-level aliases for UI clients
        response_data.update({
            'summary': result.get('summary'),
            'info': result.get('info'),
            'correlationPairs': result.get('correlationPairs'),
            'numericalSummary': result.get('numericalSummary')
        })
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

        # Also include a short agent_answer and user_view_report for display
        agent_answer = ml_result.get('agent_answer') if isinstance(ml_result, dict) else None
        if not agent_answer and isinstance(eda_result, dict):
            agent_answer = eda_result.get('aiInsights')

        # Build lightweight report
        report = f"Dataset: {eda_result.get('shape', {}).get('rows', 'N/A')} rows × {eda_result.get('shape', {}).get('columns', 'N/A')} columns."

        return {
            "status": "success",
            "eda_result": eda_result,
            "ml_result": ml_result,
            "agent_answer": agent_answer,
            "user_view_report": report,
            "summary": eda_result.get('summary'),
            "info": eda_result.get('info'),
            "correlationPairs": eda_result.get('correlationPairs'),
            "numericalSummary": eda_result.get('numericalSummary')
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

        # Determine file type: if Excel (xls/xlsx) then pass raw bytes to EDA parser
        filename = (file.filename or '').lower()
        content_type = (file.content_type or '').lower()
        is_excel = False
        if filename.endswith(('.xls', '.xlsx')):
            is_excel = True
        elif 'spreadsheet' in content_type or 'excel' in content_type:
            is_excel = True

        if is_excel:
            parsed_input = raw  # bytes -> perform_advanced_eda_from_csv_text will handle Excel bytes
        else:
            parsed_input = raw.decode('utf-8', errors='ignore')

        parsed_goal = {}
        if goal:
            try:
                parsed_goal = json.loads(goal)
            except Exception:
                parsed_goal = {"description": goal}

        result = perform_advanced_eda_from_csv_text(parsed_input, goal=parsed_goal)

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

        # Provide combined response with aliases and human-readable report
        agent_answer = ml_result.get('agent_answer') if isinstance(ml_result, dict) else None
        if not agent_answer and isinstance(result, dict):
            agent_answer = result.get('aiInsights')

        user_view_report = None
        try:
            report_lines = ["# Validation Report", ""]
            report_lines.append(f"**Dataset:** {result.get('shape', {}).get('rows', 'N/A')} rows × {result.get('shape', {}).get('columns', 'N/A')} columns")
            if result.get('insights'):
                report_lines.append('\n## Key EDA Insights')
                for k, v in result.get('insights', {}).items():
                    if isinstance(v, list) and v:
                        report_lines.append(f"- **{k}**: {('; '.join(v))}")
            user_view_report = '\n'.join(report_lines)
        except Exception:
            user_view_report = None

        # Add convenient top-level aliases for UI
        response = {"status": "success", "result": result, "eda_result": result, "ml_result": ml_result, "agent_answer": agent_answer, "user_view_report": user_view_report}
        response.update({
            'summary': result.get('summary'),
            'info': result.get('info'),
            'correlationPairs': result.get('correlationPairs'),
            'numericalSummary': result.get('numericalSummary')
        })
        return response
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
