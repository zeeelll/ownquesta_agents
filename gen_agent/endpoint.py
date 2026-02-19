from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import logging
import os
from pathlib import Path

router = APIRouter()
logger = logging.getLogger("gen_agent")


class GenRequest(BaseModel):
    best_model: Dict[str, Any]
    eda_result: Optional[Dict[str, Any]] = None
    goal: Optional[Dict[str, Any]] = None


@router.post("/explain")
async def explain_model(req: GenRequest):
    """Explain why the provided model is best in human-friendly language and provide download/deploy options.
    This is a lightweight generator; in a full system it would call an LLM to produce polished prose.
    """
    try:
        model = req.best_model
        if not model:
            raise HTTPException(status_code=400, detail="Provide `best_model`")

        name = model.get('name', 'model')
        score = model.get('accuracy') or model.get('score') or model.get('f1') or 'N/A'

        explanation = (
            f"We selected '{name}' as the best model because it achieved the highest validation metric "
            f"({score}). It balances predictive performance and robustness for the provided dataset and goal."
        )

        # Provide simple download link placeholder (in real system return a stored path or presigned URL)
        model_path = model.get('artifact_path') or f"/models/{name}.pkl"

        return {
            'status': 'success',
            'explanation': explanation,
            'download_path': model_path,
            'deploy_options': ['docker_container', 'aws_sagemaker', 'download_and_run']
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Gen explain failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
