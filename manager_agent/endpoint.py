from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging
import json

router = APIRouter()
logger = logging.getLogger("manager_agent")


class PipelineRequest(BaseModel):
    csv_text: Optional[str] = None
    goal: Optional[Dict[str, Any]] = None


@router.get("/ml_config")
async def ml_config():
    """Return ML configuration page layout describing preprocessing, modeling, and comparison steps.

    This supplies a JSON structure the frontend can render as a three-step ML configuration UI:
      1) Preprocessing & Feature Engineering
      2) Model Creation, Training & Evaluation
      3) Model Comparison & Selection
    """
    try:
        layout = {
            "status": "success",
            "page": "ml_config",
            "title": "Model Configuration",
            "steps": [
                {
                    "step": 1,
                    "id": "preprocessing",
                    "title": "Preprocessing & Feature Engineering",
                    "description": "Select target, choose imputers, scaling, encoding and feature selection strategies.",
                    "fields": [
                        {"name": "target_column", "type": "string", "label": "Target column"},
                        {"name": "imputation_numeric", "type": "select", "label": "Numeric imputation", "options": ["median", "mean", "knn"]},
                        {"name": "imputation_categorical", "type": "select", "label": "Categorical imputation", "options": ["most_frequent", "constant"]},
                        {"name": "scaling", "type": "select", "label": "Scaling", "options": ["none", "standard", "minmax"]},
                        {"name": "encoding", "type": "select", "label": "Categorical encoding", "options": ["one-hot", "ordinal", "target"]},
                        {"name": "feature_selection", "type": "select", "label": "Feature selection", "options": ["none", "variance_threshold", "select_k_best", "pca"]}
                    ]
                },
                {
                    "step": 2,
                    "id": "modeling",
                    "title": "Model Creation, Training & Evaluation",
                    "description": "Choose candidate algorithms, tuning strategy, cross-validation and evaluation metrics.",
                    "fields": [
                        {"name": "candidate_models", "type": "multiselect", "label": "Candidate models", "options": ["logistic_regression", "random_forest", "xgboost", "svm"]},
                        {"name": "hyperparameter_search", "type": "select", "label": "Hyperparameter search", "options": ["none", "grid", "random", "bayes"]},
                        {"name": "cv_folds", "type": "number", "label": "CV folds", "default": 5},
                        {"name": "metrics", "type": "multiselect", "label": "Evaluation metrics", "options": ["accuracy", "f1", "roc_auc", "precision", "recall"]}
                    ]
                },
                {
                    "step": 3,
                    "id": "comparison",
                    "title": "Model Comparison & Selection",
                    "description": "Compare trained models by selected metrics, inspect feature importance and choose the best model to deploy.",
                    "fields": [
                        {"name": "compare_by", "type": "select", "label": "Compare by", "options": ["accuracy", "f1", "roc_auc"]},
                        {"name": "explainability", "type": "select", "label": "Explainability", "options": ["shap", "lime", "none"]},
                        {"name": "deploy_options", "type": "multiselect", "label": "Deploy options", "options": ["docker", "sagemaker", "download"]}
                    ]
                }
            ]
        }

        return layout
    except Exception as e:
        logger.exception("Failed to build ml_config layout: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/run_pipeline")
async def run_pipeline(req: PipelineRequest):
    """Run the full pipeline: validation -> fp -> model training -> compare -> explain.

    This endpoint orchestrates the other agents created in this package and returns
    a combined result. It's defensive and will return partial outputs if any step fails.
    """
    try:
        if not req.csv_text:
            raise HTTPException(status_code=400, detail="Provide `csv_text` in request body")

        # Step 1: Validation (reuse validation_agent.config utilities if available)
        try:
            from validation_agent.config import perform_advanced_eda_from_csv_text
            eda = perform_advanced_eda_from_csv_text(req.csv_text, goal=req.goal or {})
        except Exception as e:
            logger.warning("Validation step failed or not available: %s", e)
            eda = {'error': str(e)}

        # Step 2: Feature processing
        try:
            from fp_agent.endpoint import process_features, FPRequest as _FPReq
            fp_req = _FPReq(eda_result=eda, csv_text=req.csv_text, goal=req.goal or {})
            fp_res = await process_features(fp_req)
        except Exception as e:
            logger.warning("FP step failed: %s", e)
            fp_res = {'error': str(e)}

        # Step 3: Model create & train (placeholder: expects features + labels)
        try:
            from model_agent.endpoint import create_and_train, ModelTrainRequest as _ModelReq

            # In a simple demo, try to extract a target column from eda (best-effort)
            # Real system: frontend supplies features/labels after FP step
            features = []
            labels = []
            model_req = _ModelReq(features=features, labels=labels, eda_result=eda, goal=req.goal or {})
            model_res = await create_and_train(model_req)
        except Exception as e:
            logger.warning("Model step failed: %s", e)
            model_res = {'error': str(e)}

        # Step 4: Model comparison
        try:
            from moco_agent.endpoint import compare_models, MocoRequest as _MocoReq
            # Build model summaries from model_res if available
            summaries = []
            if isinstance(model_res, dict) and model_res.get('summary'):
                summaries = model_res.get('summary')
            moco_req = _MocoReq(model_summaries=summaries, eda_result=eda, goal=req.goal or {})
            moco_res = await compare_models(moco_req)
        except Exception as e:
            logger.warning("Moco step failed: %s", e)
            moco_res = {'error': str(e)}

        # Step 5: Explanation / generation
        try:
            from gen_agent.endpoint import explain_model, GenRequest as _GenReq
            best = moco_res.get('best_model') if isinstance(moco_res, dict) else None
            if not best and isinstance(model_res, dict) and model_res.get('summary'):
                best = model_res.get('summary')[0]
            gen_req = _GenReq(best_model=best or {}, eda_result=eda, goal=req.goal or {})
            gen_res = await explain_model(gen_req)
        except Exception as e:
            logger.warning("Gen step failed: %s", e)
            gen_res = {'error': str(e)}

        return {
            'status': 'completed',
            'validation': eda,
            'feature_processing': fp_res,
            'modeling': model_res,
            'comparison': moco_res,
            'explanation': gen_res
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Pipeline failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
