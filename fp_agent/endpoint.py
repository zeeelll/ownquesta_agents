from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import logging

router = APIRouter()
logger = logging.getLogger("fp_agent")


class FPRequest(BaseModel):
    eda_result: Optional[Dict[str, Any]] = None
    csv_text: Optional[str] = None
    goal: Optional[Dict[str, Any]] = None


@router.post("/process")
async def process_features(req: FPRequest):
    """Feature processing and preprocessing agent.

    Accepts either an `eda_result` produced by the validation agent or raw CSV text.
    Returns a lightweight description of feature transforms and a small sample of processed data.
    """
    try:
        if not req.eda_result and not req.csv_text:
            raise HTTPException(status_code=400, detail="Provide `eda_result` or `csv_text`")

        # Minimal placeholder preprocessing: collect feature types and suggested transforms
        eda = req.eda_result or {}
        numeric = eda.get('numericColumns', []) or eda.get('numericalColumns', []) or []
        categorical = eda.get('objectColumns', []) or eda.get('categoricalColumns', []) or []

        suggested = {"scaling": [], "encoding": [], "imputation": []}
        if numeric:
            suggested['scaling'].append('StandardScaler')
            suggested['imputation'].append('median')
        if categorical:
            suggested['encoding'].append('one-hot')
            suggested['imputation'].append('most_frequent')

        result = {
            'status': 'success',
            'suggested_transforms': suggested,
            'feature_summary': {
                'numeric': numeric,
                'categorical': categorical,
                'count': len(numeric) + len(categorical)
            },
            'processed_sample': []
        }

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("FP processing failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
