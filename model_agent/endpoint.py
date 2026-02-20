from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import logging

router = APIRouter()
logger = logging.getLogger("model_agent")


class ModelTrainRequest(BaseModel):
    # Accept processed sample (list of dict rows) or raw features/labels
    processed_sample: Optional[List[Dict[str, Any]]] = None
    csv_text: Optional[str] = None
    features: Optional[List[Dict[str, Any]]] = None
    labels: Optional[List[Any]] = None
    eda_result: Optional[Dict[str, Any]] = None
    goal: Optional[Dict[str, Any]] = None


@router.post("/create_and_train")
async def create_and_train(req: ModelTrainRequest):
    """Create multiple models, train and evaluate them based on provided preprocessing.

    Accepts either a `processed_sample` (list of dict rows) or `csv_text`, or
    `features` + `labels`. Returns evaluation metrics and a path to the best model
    pickled under the local `models/` directory.
    """
    try:
        # Prepare data from whichever source is provided
        rows = None
        if req.processed_sample:
            rows = req.processed_sample
        elif req.features and req.labels:
            # Merge features and labels into rows for convenience
            rows = []
            for f, l in zip(req.features, req.labels):
                r = dict(f)
                r['_target'] = l
                rows.append(r)
        elif req.csv_text:
            # Parse small CSV text
            try:
                import csv
                from io import StringIO
                reader = csv.DictReader(StringIO(req.csv_text))
                rows = [r for r in reader]
            except Exception:
                rows = None

        if not rows or len(rows) < 3:
            return {
                'status': 'waiting_for_data',
                'message': 'Provide a processed dataset (processed_sample or csv_text) with at least a few rows to train models',
                'received_rows': len(rows) if rows else 0
            }

        # Infer task from goal or data (classification by default)
        task = 'classification'
        metric = 'accuracy'
        if req.goal and isinstance(req.goal, dict):
            task = req.goal.get('task', task)
            metric = req.goal.get('metric', metric)

        # Convert list-of-dict rows into X, y with deterministic column ordering
        cols = sorted([c for c in rows[0].keys() if c != '_target'])
        # If target column is named differently, try to detect
        if '_target' not in rows[0]:
            # pick a likely target from goal or eda
            target_name = None
            if req.goal and isinstance(req.goal, dict):
                target_name = req.goal.get('target')
            if not target_name:
                # fallback: choose last column
                target_name = [c for c in rows[0].keys()][-1]
            if target_name in cols:
                cols = [c for c in cols if c != target_name]
                target_col = target_name
            else:
                target_col = '_target'
        else:
            target_col = '_target'

        X = []
        y = []
        for r in rows:
            if target_col not in r and '_target' not in r:
                continue
            y.append(r.get(target_col) or r.get('_target'))
            X.append([r.get(c) for c in cols])

        # Lazy sklearn imports and model selection
        try:
            import os
            import pickle
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score, r2_score, mean_squared_error

            # Choose model candidates based on task
            models = {}
            if task == 'classification':
                from sklearn.linear_model import LogisticRegression
                from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
                models = {
                    'logistic_regression': LogisticRegression(max_iter=400),
                    'random_forest': RandomForestClassifier(n_estimators=100),
                    'gradient_boosting': GradientBoostingClassifier()
                }
            else:
                # regression
                from sklearn.linear_model import LinearRegression
                from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
                models = {
                    'linear_regression': LinearRegression(),
                    'random_forest_reg': RandomForestRegressor(n_estimators=100),
                    'gradient_boosting_reg': GradientBoostingRegressor()
                }

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            results = []
            best_score = None
            best_model_name = None
            best_model_path = None

            os.makedirs('models', exist_ok=True)

            for name, m in models.items():
                try:
                    m.fit(X_train, y_train)
                    preds = m.predict(X_test)
                    if task == 'classification':
                        score = float(accuracy_score(y_test, preds))
                        results.append({'name': name, 'score': score})
                    else:
                        rmse = float(mean_squared_error(y_test, preds, squared=False))
                        r2 = float(r2_score(y_test, preds))
                        results.append({'name': name, 'rmse': rmse, 'r2': r2})

                    # Track best according to metric
                    compare = score if task == 'classification' else -rmse
                    if best_score is None or compare > best_score:
                        best_score = compare
                        best_model_name = name
                        path = os.path.join('models', f"best_{name}.pkl")
                        with open(path, 'wb') as fh:
                            pickle.dump(m, fh)
                        best_model_path = path
                except Exception as me:
                    logger.warning("Model %s failed: %s", name, me)
                    results.append({'name': name, 'error': str(me)})

            summary = results
            return {
                'status': 'success',
                'task': task,
                'metric': metric,
                'results': summary,
                'best_model': {'name': best_model_name, 'path': best_model_path}
            }
        except ImportError:
            return {'status': 'sklearn_missing', 'message': 'scikit-learn not installed in environment'}

    except Exception as e:
        logger.exception("Model training failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
