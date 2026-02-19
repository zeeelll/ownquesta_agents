from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import logging

router = APIRouter()
logger = logging.getLogger("model_agent")


class ModelTrainRequest(BaseModel):
    features: Optional[List[Dict[str, Any]]] = None
    labels: Optional[List[Any]] = None
    eda_result: Optional[Dict[str, Any]] = None
    goal: Optional[Dict[str, Any]] = None


@router.post("/create_and_train")
async def create_and_train(req: ModelTrainRequest):
    """Create multiple simple models and evaluate them.

    This endpoint trains a few sklearn models (if available) on provided data.
    The implementation is defensive: if numeric training arrays aren't provided,
    it returns a placeholder indicating next steps.
    """
    try:
        # If no features/labels provided, return scaffolding info
        if not req.features or not req.labels:
            return {
                'status': 'waiting_for_data',
                'message': 'Provide `features` (list of dict rows) and `labels` (list) to train models',
                'available_models': ['logistic_regression', 'random_forest']
            }

        # Convert incoming list-of-dict into X, y (very small helper)
        X = []
        for row in req.features:
            # Ensure deterministic ordering
            keys = sorted(row.keys())
            X.append([row[k] for k in keys])
        y = req.labels

        # Lazy import sklearn
        results = []
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.linear_model import LogisticRegression
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            models = {
                'logistic_regression': LogisticRegression(max_iter=200),
                'random_forest': RandomForestClassifier(n_estimators=50)
            }

            for name, m in models.items():
                try:
                    m.fit(X_train, y_train)
                    preds = m.predict(X_test)
                    acc = float(accuracy_score(y_test, preds))
                    results.append({'name': name, 'accuracy': acc, 'model_obj': m})
                except Exception as me:
                    logger.warning("Model %s failed: %s", name, me)
                    results.append({'name': name, 'error': str(me)})

            # Do not attempt to JSON-serialize model objects in API response; provide pickled path or id in real system
            summary = [{'name': r.get('name'), 'accuracy': r.get('accuracy')} for r in results]
            return {'status': 'success', 'summary': summary, 'raw': results}

        except ImportError:
            return {'status': 'sklearn_missing', 'message': 'scikit-learn not installed in environment'}

    except Exception as e:
        logger.exception("Model training failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
