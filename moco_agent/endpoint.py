from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging

router = APIRouter()
logger = logging.getLogger("moco_agent")


class MocoRequest(BaseModel):
    # Either pass `model_summaries` as returned by model_agent or provide
    # `processed_sample` (list of dict rows) and model paths to re-evaluate.
    model_summaries: Optional[List[Dict[str, Any]]] = None
    processed_sample: Optional[List[Dict[str, Any]]] = None
    eda_result: Optional[Dict[str, Any]] = None
    goal: Optional[Dict[str, Any]] = None


@router.post("/compare")
async def compare_models(req: MocoRequest):
    """Compare models using provided summaries and choose the best one.

    Expects a list of model summaries with numeric metrics (e.g., accuracy).
    Returns the best model info and a short explanation of choice.
    """
    try:
        summaries = req.model_summaries or []

        # If processed_sample and model paths exist, try to re-evaluate pickled models
        eval_results = []
        if req.processed_sample and summaries:
            # Prepare X,y from processed_sample
            rows = req.processed_sample
            if len(rows) < 3:
                raise HTTPException(status_code=400, detail='processed_sample needs at least a few rows')

            # determine target column
            sample_keys = list(rows[0].keys())
            target_col = None
            if '_target' in rows[0]:
                target_col = '_target'
            else:
                # try to infer from summaries.goal or last column
                target_col = rows[0].get('_target') and '_target' or sample_keys[-1]

            cols = [c for c in sample_keys if c != target_col]

            X = [[r.get(c) for c in cols] for r in rows]
            y = [r.get(target_col) for r in rows]

            # attempt to load pickled models from summaries paths
            for m in summaries:
                path = m.get('path') or m.get('model_path')
                name = m.get('name') or 'unknown'
                if not path:
                    eval_results.append({**m, 're_eval': 'no_path'})
                    continue
                try:
                    import pickle
                    with open(path, 'rb') as fh:
                        mdl = pickle.load(fh)
                    # lazy sklearn metrics
                    try:
                        from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
                        preds = mdl.predict(X)
                        # decide metric type by y
                        if all(isinstance(v, (int, float)) for v in y):
                            # regression
                            rmse = float(mean_squared_error(y, preds, squared=False))
                            r2 = float(r2_score(y, preds))
                            eval_results.append({'name': name, 'path': path, 'rmse': rmse, 'r2': r2})
                        else:
                            acc = float(accuracy_score(y, preds))
                            eval_results.append({'name': name, 'path': path, 'accuracy': acc})
                    except Exception as me:
                        eval_results.append({'name': name, 'path': path, 'error': str(me)})
                except Exception as e:
                    eval_results.append({'name': name, 'path': path, 'error': 'load_failed: ' + str(e)})

            # Choose best from eval_results
            best = None
            best_score = None
            for r in eval_results:
                if 'accuracy' in r:
                    score = r['accuracy']
                elif 'rmse' in r:
                    score = -r['rmse']
                else:
                    continue
                if best is None or score > best_score:
                    best = r
                    best_score = score

            return {'status': 'success', 'evaluations': eval_results, 'best_model': best, 'best_score': best_score}

        # Fallback: choose best from provided summaries by common metrics
        if summaries:
            best = None
            best_score = None
            for m in summaries:
                score = None
                for k in ['accuracy', 'score', 'f1', 'roc_auc']:
                    if k in m and isinstance(m[k], (int, float)):
                        score = float(m[k])
                        break
                if score is None:
                    continue
                if best is None or score > best_score:
                    best = m
                    best_score = score
            explanation = f"Selected model '{best.get('name')}' with score {best_score:.4f}. Chosen by highest validation metric among provided models."
            return {'status': 'success', 'best_model': best, 'best_score': best_score, 'explanation': explanation}

        raise HTTPException(status_code=400, detail='No models provided to compare')

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Model comparison failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
