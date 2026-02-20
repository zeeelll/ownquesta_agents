from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import logging
import os
from pathlib import Path

router = APIRouter()
logger = logging.getLogger("gen_agent")


class GenRequest(BaseModel):
    best_model: Dict[str, Any]
    eda_result: Optional[Dict[str, Any]] = None
    goal: Optional[Dict[str, Any]] = None
    processed_sample: Optional[List[Dict[str, Any]]] = None
    model_summaries: Optional[List[Dict[str, Any]]] = None


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
        # collect basic metrics
        metrics = {k: model.get(k) for k in ['accuracy', 'score', 'f1', 'roc_auc', 'rmse', 'r2'] if k in model}

        details = []
        # If a model artifact path is provided, try to inspect it
        model_path = model.get('path') or model.get('artifact_path') or model.get('model_path') or None
        artifact_info = None
        feature_info = None
        model_type = None
        try:
            if model_path and Path(model_path).exists():
                import pickle
                with open(model_path, 'rb') as fh:
                    mdl = pickle.load(fh)
                model_type = type(mdl).__name__
                artifact_info = {'type': model_type}
                # try to extract common attributes
                if hasattr(mdl, 'feature_importances_'):
                    fi = list(getattr(mdl, 'feature_importances_'))
                    feature_info = {'importances': fi}
                elif hasattr(mdl, 'coef_'):
                    coef = getattr(mdl, 'coef_')
                    # flatten
                    if hasattr(coef, 'tolist'):
                        coef = coef.tolist()
                    feature_info = {'coef': coef}
                # include number of trees if ensemble
                if hasattr(mdl, 'n_estimators'):
                    artifact_info['n_estimators'] = getattr(mdl, 'n_estimators')
        except Exception as e:
            logger.info('Could not load or inspect model artifact %s: %s', model_path, e)

        # Build a human-friendly explanation
        explanation_lines = []
        # Opening: Why selected
        if metrics:
            metric_k, metric_v = next(iter(metrics.items()))
            explanation_lines.append(f"The model we selected is '{name}'. It achieved a {metric_k} of {metric_v} on the validation set, which was the strongest performance among candidates.")
        else:
            explanation_lines.append(f"The model we selected is '{name}'. It was chosen as the best candidate based on comparative evaluation of the available models.")

        # Add model type insights if available
        if model_type:
            explanation_lines.append(f"This model is a {model_type}. {('Tree-based models like RandomForest are strong for tabular data and capture non-linear interactions.' if 'Forest' in model_type or 'Tree' in model_type else '')}")

        # Feature importance summary
        top_features_text = ''
        if feature_info:
            if 'importances' in feature_info and req.processed_sample:
                import numpy as np
                imps = np.array(feature_info['importances'])
                # try to get feature names from processed_sample keys
                sample_keys = list(req.processed_sample[0].keys())
                # remove target if present
                sample_keys = [k for k in sample_keys if k != '_target']
                if len(imps) == len(sample_keys):
                    idxs = list(reversed(imps.argsort()))
                    top = [(sample_keys[i], float(imps[i])) for i in idxs[:5]]
                    top_features_text = ', '.join([f"{t[0]} ({t[1]:.2f})" for t in top])
                    explanation_lines.append(f"Feature importance highlights: {top_features_text}. These features contributed most to the model's predictions.")
            elif 'coef' in feature_info and req.processed_sample:
                coeffs = feature_info['coef']
                # flatten
                try:
                    coeffs_arr = coeffs[0] if isinstance(coeffs[0], list) else coeffs
                except Exception:
                    coeffs_arr = coeffs
                sample_keys = list(req.processed_sample[0].keys())
                sample_keys = [k for k in sample_keys if k != '_target']
                if len(coeffs_arr) == len(sample_keys):
                    # top positive contributors
                    pairs = list(zip(sample_keys, coeffs_arr))
                    pairs_sorted = sorted(pairs, key=lambda x: -abs(float(x[1])))[:5]
                    pf = ', '.join([f"{p[0]} ({float(p[1]):.2f})" for p in pairs_sorted])
                    explanation_lines.append(f"Important coefficients: {pf}. These features have the largest linear effect on predictions.")

        # Compare against alternatives if model_summaries provided
        if req.model_summaries:
            try:
                # find second best
                others = [m for m in req.model_summaries if m.get('name') != name]
                if others:
                    # compare best metric
                    def score_of(m):
                        for k in ['accuracy', 'score', 'f1', 'roc_auc']:
                            if k in m and isinstance(m[k], (int, float)):
                                return float(m[k])
                        return None
                    best_score = score_of(model) or 0
                    other_scores = [(m.get('name'), score_of(m) or 0) for m in others]
                    if other_scores:
                        other_scores_sorted = sorted(other_scores, key=lambda x: -x[1])
                        runner_up = other_scores_sorted[0]
                        explanation_lines.append(f"Compared to the next-best model ('{runner_up[0]}', score={runner_up[1]}), '{name}' offers better validation performance ({best_score} vs {runner_up[1]}).")
            except Exception as e:
                logger.info('Failed to compare to other summaries: %s', e)

        # Practical notes and deployment suggestions
        explanation_lines.append("Practical notes: consider testing the model on unseen production-like data, calibrating probabilities if needed, and monitoring for data drift.")
        # Deployment options heuristic
        deploy_opts = ['docker_container', 'aws_sagemaker', 'download_and_run']
        if artifact_info and artifact_info.get('n_estimators'):
            deploy_opts = ['aws_sagemaker', 'docker_container', 'download_and_run']

        explanation = '\n\n'.join(explanation_lines)

        return {
            'status': 'success',
            'explanation': explanation,
            'download_path': model_path or f"/models/{name}.pkl",
            'deploy_options': deploy_opts,
            'artifact_info': artifact_info,
            'feature_summary': top_features_text
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Gen explain failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
