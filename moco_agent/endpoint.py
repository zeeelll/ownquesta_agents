from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging

router = APIRouter()
logger = logging.getLogger("moco_agent")


class MocoRequest(BaseModel):
    model_summaries: List[Dict[str, Any]]
    eda_result: Optional[Dict[str, Any]] = None
    goal: Optional[Dict[str, Any]] = None


@router.post("/compare")
async def compare_models(req: MocoRequest):
    """Compare models using provided summaries and choose the best one.

    Expects a list of model summaries with numeric metrics (e.g., accuracy).
    Returns the best model info and a short explanation of choice.
    """
    try:
        if not req.model_summaries:
            raise HTTPException(status_code=400, detail="Provide `model_summaries` with metrics")

        # Choose best by accuracy if available, else fallback to first
        best = None
        best_score = -1.0
        for m in req.model_summaries:
            score = None
            for k in ['accuracy', 'f1', 'roc_auc', 'score']:
                if k in m and isinstance(m[k], (int, float)):
                    score = float(m[k])
                    break
            if score is None:
                score = -1.0
            if score > best_score:
                best_score = score
                best = m

        explanation = f"Selected model '{best.get('name')}' with score {best_score:.4f}. Chosen by highest validation metric among provided models."

        return {'status': 'success', 'best_model': best, 'best_score': best_score, 'explanation': explanation}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Model comparison failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
