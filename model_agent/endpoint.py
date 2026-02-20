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
        import os
        import pickle
        import math

        # Prefer pandas for easier handling
        try:
            import pandas as pd
        except Exception:
            pd = None

        logger.info(f"Model.create_and_train called: processed_sample_present={bool(req.processed_sample)}, csv_text_present={bool(req.csv_text)}, eda_result_present={bool(req.eda_result)}")

        # If no eda_result was provided, try to call the validation agent to analyze the csv_text
        validation_info = None
        if not req.eda_result and req.csv_text:
            try:
                import httpx
                base = os.environ.get('ML_VALIDATION_URL', 'http://127.0.0.1:8000')
                url = f"{base.rstrip('/')}/validation/validate"
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.post(url, json={
                        'csv_text': req.csv_text,
                        'goal': req.goal or {}
                    })
                    if resp.status_code == 200:
                        validation_info = resp.json()
            except Exception:
                validation_info = None

        # Merge eda_result from request or validation_info
        eda = req.eda_result or (validation_info.get('eda_result') if isinstance(validation_info, dict) else None) or validation_info

        # Determine task: prefer explicit goal.task, then validation result, default to classification
        task = 'classification'
        metric = req.goal.get('metric') if req.goal and isinstance(req.goal, dict) else None
        if req.goal and isinstance(req.goal, dict) and req.goal.get('task'):
            task = req.goal.get('task')
        else:
            try:
                if isinstance(validation_info, dict):
                    ml = validation_info.get('ml_result') or validation_info.get('ml') or validation_info
                    if ml and isinstance(ml, dict) and ml.get('goal_understanding'):
                        task = ml['goal_understanding'].get('interpreted_task', task)
            except Exception:
                pass

        task = (task or 'classification').lower()

        # Build DataFrame
        df = None
        if req.processed_sample:
            if pd:
                df = pd.DataFrame(req.processed_sample)
            else:
                # fallback to manual conversion
                cols = sorted(req.processed_sample[0].keys())
                df = None
        # fallback: eda_result may contain a sample or processed_sample list
        if df is None and req.eda_result and isinstance(req.eda_result, dict):
            if req.eda_result.get('processed_sample'):
                try:
                    if pd:
                        df = pd.DataFrame(req.eda_result.get('processed_sample'))
                except Exception:
                    df = None
            elif req.eda_result.get('sample'):
                try:
                    if pd:
                        df = pd.DataFrame(req.eda_result.get('sample'))
                except Exception:
                    df = None
        elif req.features and req.labels:
            if pd:
                df = pd.DataFrame(req.features)
                df['_target'] = req.labels
        elif req.csv_text and pd:
            from io import StringIO
            try:
                df = pd.read_csv(StringIO(req.csv_text))
            except Exception:
                df = None

        if df is None:
            logger.info("No dataframe could be constructed for training; returning no_data")
            return {'status': 'no_data', 'message': 'No usable dataframe available for training. Provide processed_sample or csv_text.'}

        # Determine target column
        target_col = None
        if req.goal and isinstance(req.goal, dict) and req.goal.get('target'):
            target_col = req.goal.get('target')
        # try common target guesses
        if not target_col:
            guesses = ['target', 'label', 'y', 'outcome']
            for g in guesses:
                if g in df.columns:
                    target_col = g
                    break
        # fallback: if df has a column named like last column and task is supervised
        if not target_col and task in ('classification', 'regression', 'supervised'):
            target_col = df.columns[-1]

        X = df.drop(columns=[target_col]) if target_col in df.columns else df.copy()
        y = df[target_col] if target_col in df.columns else None

        # Convert X to numeric where possible
        try:
            X_num = X.select_dtypes(include=['number'])
            if X_num.shape[1] == 0:
                # try coercing
                X = X.apply(pd.to_numeric, errors='coerce')
        except Exception:
            pass

        # Lazy import sklearn pieces
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, r2_score, silhouette_score
        except Exception:
            return {'status': 'sklearn_missing', 'message': 'scikit-learn not installed'}

        # Candidate models per task (aim for 5 models)
        models: Dict[str, Any] = {}
        # Supervised: classification
        if task == 'classification' or task == 'supervised':
            from sklearn.linear_model import LogisticRegression
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
            from sklearn.svm import SVC
            from sklearn.neighbors import KNeighborsClassifier

            models = {
                'logistic_regression': LogisticRegression(max_iter=400),
                'random_forest': RandomForestClassifier(n_estimators=100),
                'gradient_boosting': GradientBoostingClassifier(),
                'svc': SVC(probability=True),
                'knn': KNeighborsClassifier(n_neighbors=5)
            }
        elif task == 'regression':
            from sklearn.linear_model import LinearRegression
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
            from sklearn.svm import SVR
            from sklearn.neighbors import KNeighborsRegressor

            models = {
                'linear_regression': LinearRegression(),
                'random_forest_reg': RandomForestRegressor(n_estimators=100),
                'gradient_boosting_reg': GradientBoostingRegressor(),
                'svr': SVR(),
                'knn_reg': KNeighborsRegressor(n_neighbors=5)
            }
        elif task in ('clustering', 'unsupervised'):
            from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
            try:
                from sklearn.mixture import GaussianMixture
            except Exception:
                GaussianMixture = None

            models = {
                'kmeans': KMeans(n_clusters=3),
                'dbscan': DBSCAN(),
                'agglomerative': AgglomerativeClustering(n_clusters=3),
                'spectral': SpectralClustering(n_clusters=3, assign_labels='kmeans')
            }
            if GaussianMixture is not None:
                models['gmm'] = GaussianMixture(n_components=3)
        elif task in ('association',):
            # Association rules are not standard sklearn models; attempt with mlxtend if available
            try:
                from mlxtend.frequent_patterns import apriori, association_rules
                models = {'apriori': 'apriori'}
            except Exception:
                models = {}

        # If user requested a high-level ensemble goal, add ensemble models
        if req.goal and isinstance(req.goal, dict) and req.goal.get('high_level_goal') in ('ensemble', 'boost'):
            try:
                if task in ('classification', 'supervised'):
                    from sklearn.ensemble import VotingClassifier
                    # use a simple voting ensemble composed of earlier candidates
                    estimators = []
                    for i, (n, m) in enumerate(list(models.items())[:3]):
                        estimators.append((f"m{i}", m))
                    models['voting_classifier'] = VotingClassifier(estimators=estimators, voting='soft')
                if task == 'regression':
                    from sklearn.ensemble import VotingRegressor
                    estimators = []
                    for i, (n, m) in enumerate(list(models.items())[:3]):
                        estimators.append((f"m{i}", m))
                    models['voting_regressor'] = VotingRegressor(estimators=estimators)
            except Exception:
                pass

        # Training and evaluation
        results = []
        best_score = None
        best_model_name = None
        best_model_path = None
        from pathlib import Path
        root = Path(__file__).parent.parent
        models_dir = root / 'models'
        models_dir.mkdir(parents=True, exist_ok=True)

        # Supervised flow
        if task in ('classification', 'regression', 'supervised'):
            test_size = 0.2
            if req.goal and isinstance(req.goal, dict) and req.goal.get('split'):
                test_size = float(req.goal.get('split'))

            X_vals = X.values if hasattr(X, 'values') else X
            y_vals = y.values if hasattr(y, 'values') else y

            X_train, X_test, y_train, y_test = train_test_split(X_vals, y_vals, test_size=test_size, random_state=42)

            for name, m in models.items():
                try:
                    # Skip non-fit models (e.g., placeholders)
                    if isinstance(m, str):
                        results.append({'name': name, 'note': 'placeholder/not trained'})
                        continue
                    m.fit(X_train, y_train)
                    preds = m.predict(X_test)

                    if task == 'classification':
                        acc = float(accuracy_score(y_test, preds))
                        try:
                            f1 = float(f1_score(y_test, preds, average='weighted'))
                        except Exception:
                            f1 = None
                        try:
                            prob = m.predict_proba(X_test)
                            auc = float(roc_auc_score(y_test, prob[:, 1])) if prob is not None and prob.shape[1] > 1 else None
                        except Exception:
                            auc = None
                        score_item = {'name': name, 'accuracy': acc, 'f1': f1, 'auc': auc}
                        results.append(score_item)
                        compare = acc
                    else:
                        rmse = float(mean_squared_error(y_test, preds, squared=False))
                        r2 = float(r2_score(y_test, preds))
                        results.append({'name': name, 'rmse': rmse, 'r2': r2})
                        compare = -rmse

                    # persist each trained model so other agents can reload and compare
                    try:
                        model_path = str((models_dir / f"{name}.pkl").resolve())
                        with open(model_path, 'wb') as fh:
                            pickle.dump(m, fh)
                        logger.info(f"Saved model artifact: {name} -> {model_path}")
                    except Exception:
                        model_path = None

                    # attach path to the reported result
                    if isinstance(score_item, dict):
                        score_item['path'] = model_path
                    else:
                        # ensure dict fallback
                        results.append({'name': name, 'path': model_path})

                    # save model if best
                    if best_score is None or compare > best_score:
                        best_score = compare
                        best_model_name = name
                        best_model_path = model_path
                except Exception as me:
                    logger.warning("Model %s training failed: %s", name, me)
                    results.append({'name': name, 'error': str(me)})

        # Clustering flow
        elif task in ('clustering', 'unsupervised'):
            from sklearn.preprocessing import StandardScaler
            try:
                X_scaled = StandardScaler().fit_transform(X.values if hasattr(X, 'values') else X)
            except Exception:
                X_scaled = X

            for name, m in models.items():
                try:
                    if name == 'gmm' and m is None:
                        results.append({'name': name, 'note': 'GaussianMixture unavailable'})
                        continue
                    if hasattr(m, 'fit_predict'):
                        labels = m.fit_predict(X_scaled)
                    else:
                        # some models (GaussianMixture) use fit then predict
                        m.fit(X_scaled)
                        labels = m.predict(X_scaled)

                    # Evaluate clustering with silhouette if possible
                    try:
                        sil = float(silhouette_score(X_scaled, labels)) if len(set(labels)) > 1 and len(labels) > 10 else None
                    except Exception:
                        sil = None
                        try:
                            model_path_c = str((models_dir / f"{name}.pkl").resolve())
                            with open(model_path_c, 'wb') as fh:
                                pickle.dump(m, fh)
                            logger.info(f"Saved clustering artifact: {name} -> {model_path_c}")
                        except Exception:
                            model_path_c = None
                        results.append({'name': name, 'silhouette': sil, 'path': model_path_c})
                    if sil is not None and (best_score is None or sil > best_score):
                        best_score = sil
                        best_model_name = name
                        best_model_path = str((models_dir / f"{name}.pkl").resolve())
                except Exception as me:
                    logger.warning("Clustering model %s failed: %s", name, me)
                    results.append({'name': name, 'error': str(me)})

        # Association flow (basic frequent itemset extraction)
        elif task == 'association':
            try:
                from mlxtend.frequent_patterns import apriori, association_rules
                # Expect one-hot encoded transactional dataframe; if not, try to transform
                trans = X.copy()
                if trans.select_dtypes(include=['number']).shape[1] == 0:
                    # try dummy encoding
                    trans = pd.get_dummies(X)
                freq = apriori(trans, min_support=0.1, use_colnames=True)
                rules = association_rules(freq, metric='lift', min_threshold=1.0)
                results.append({'name': 'apriori', 'frequent_itemsets': freq.head(10).to_dict(orient='records'), 'rules_count': len(rules)})
                best_model_name = 'apriori'
                best_model_path = None
            except Exception as e:
                results.append({'name': 'association', 'error': str(e)})

        summary = results

        # Write a manifest to models/ for other agents to discover metadata
        try:
            import json
            from datetime import datetime
            manifest = []
            for r in results:
                entry = {
                    'name': r.get('name'),
                    'path': r.get('path'),
                    'metrics': {k: v for k, v in r.items() if k not in ('name', 'path', 'error', 'note')},
                    'note': r.get('note') or r.get('error') or None,
                    'created_at': datetime.utcnow().isoformat() + 'Z'
                }
                if entry['name']:
                    manifest.append(entry)
            manifest_path = models_dir / 'manifest.json'
            with open(manifest_path, 'w', encoding='utf-8') as mh:
                json.dump(manifest, mh, indent=2)
            logger.info(f"Wrote models manifest: {manifest_path}")
        except Exception:
            pass

        return {
            'status': 'success',
            'task': task,
            'metric': metric,
            'trained_models': [r.get('name') for r in results if r.get('name')],
            'results': summary,
            'best_model': {'name': best_model_name, 'path': best_model_path}
        }
    except Exception as e:
        logger.exception("Model training failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
