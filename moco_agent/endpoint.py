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

        import os
        from pathlib import Path
        # Debug: log incoming payload summary
        try:
            logger.info(f"Moco.compare called: summaries={len(summaries) if summaries else 0}, processed_sample_present={bool(req.processed_sample)}")
        except Exception:
            pass

        # If caller didn't provide summaries, try to discover models on disk regardless of processed_sample
        if not summaries or len(summaries) == 0:
            models_dir = Path(__file__).parent.parent / 'models'
            eval_summaries = []
            # Try to read manifest.json first for richer metadata
            try:
                manifest_file = models_dir / 'manifest.json'
                if manifest_file.exists():
                    import json
                    with open(manifest_file, 'r', encoding='utf-8') as mf:
                        manifest = json.load(mf)
                        for e in manifest:
                            if e.get('path'):
                                eval_summaries.append({'name': e.get('name'), 'path': e.get('path'), 'metrics': e.get('metrics')})
            except Exception:
                pass

            if not eval_summaries and models_dir.exists() and models_dir.is_dir():
                for f in models_dir.iterdir():
                    if f.suffix.lower() in ('.pkl', '.joblib'):
                        eval_summaries.append({'name': f.stem, 'path': str(f.resolve())})
            summaries = eval_summaries

        # If summaries provided but missing paths, attempt to resolve by name in models/ folder
        if summaries and any(not s.get('path') for s in summaries):
            models_dir = Path(__file__).parent.parent / 'models'
            # try manifest for mapping names to paths
            manifest_map = {}
            try:
                manifest_file = models_dir / 'manifest.json'
                if manifest_file.exists():
                    import json
                    with open(manifest_file, 'r', encoding='utf-8') as mf:
                        for e in json.load(mf):
                            if e.get('name') and e.get('path'):
                                manifest_map[e['name']] = e['path']
            except Exception:
                manifest_map = {}

            for s in summaries:
                if not s.get('path') and s.get('name'):
                    if s.get('name') in manifest_map:
                        s['path'] = manifest_map[s['name']]
                        continue
                    candidate = models_dir / f"{s['name']}.pkl"
                    if candidate.exists():
                        s['path'] = str(candidate.resolve())

        # If still no summaries and no model files discoverable, return a friendly response
        if not summaries or len(summaries) == 0:
            logger.info("Moco.compare: no model artifacts discovered in models/ directory")
            return {
                'status': 'no_models',
                'evaluations': [],
                'best_model': None,
                'message': 'No model artifacts found under ownquesta_agents/models/ to compare'
            }

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
                    # try to infer path by searching common patterns in models/ directory
                    models_dir = Path(__file__).parent.parent / 'models'
                    candidate = None
                    if models_dir.exists():
                        # try exact name.pkl, best_name.pkl, or any file containing the model stem
                        patterns = [f"{name}.pkl", f"best_{name}.pkl"]
                        for p in patterns:
                            c = models_dir / p
                            if c.exists():
                                candidate = str(c.resolve())
                                break
                        if not candidate:
                            # fallback: find first file containing the name
                            for f in models_dir.iterdir():
                                if name.lower() in f.stem.lower() and f.suffix.lower() in ('.pkl', '.joblib'):
                                    candidate = str(f.resolve())
                                    break
                    if candidate:
                        path = candidate
                    else:
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


@router.get("/list_artifacts")
def list_artifacts():
    """Return a quick inventory of model artifacts and manifest for debugging."""
    try:
        from pathlib import Path
        import json
        models_dir = Path(__file__).parent.parent / 'models'
        manifest = None
        manifest_path = models_dir / 'manifest.json'
        if manifest_path.exists():
            try:
                with open(manifest_path, 'r', encoding='utf-8') as mf:
                    manifest = json.load(mf)
            except Exception:
                manifest = None

        files = []
        if models_dir.exists() and models_dir.is_dir():
            for f in models_dir.iterdir():
                if f.is_file():
                    files.append({'name': f.name, 'path': str(f.resolve()), 'size': f.stat().st_size})

        return {'status': 'ok', 'models_dir': str(models_dir.resolve()), 'files': files, 'manifest': manifest}
    except Exception as e:
        logger.exception("list_artifacts failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
