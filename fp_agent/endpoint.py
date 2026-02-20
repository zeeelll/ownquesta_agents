from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import logging
import csv
import io
from statistics import mean, pstdev

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

        eda = req.eda_result or {}

        # Helper lookups for common eda keys
        def _get_list(keys):
            for k in keys:
                v = eda.get(k)
                if v:
                    return v
            return []

        numeric = _get_list(['numericColumns', 'numericalColumns', 'numeric_columns', 'numerical_columns'])
        categorical = _get_list(['objectColumns', 'categoricalColumns', 'object_columns', 'categorical_columns'])

        # Try to obtain per-column stats if available
        numerical_summary = eda.get('numericalSummary', {}) or eda.get('numerical_summary', {})
        categorical_summary = eda.get('categoricalSummary', {}) or eda.get('categorical_summary', {})

        suggested = {'scaling': {}, 'encoding': {}, 'imputation': {}}

        # Feature selection heuristics
        selected_features = []
        # Exclude constant or id-like features
        def is_id_like(col_stats):
            try:
                unique = col_stats.get('unique') if isinstance(col_stats, dict) else None
                rows = eda.get('shape', {}).get('rows') or eda.get('row_count') or None
                if unique and rows and rows > 0 and unique / rows > 0.9:
                    return True
            except Exception:
                pass
            return False

        # Analyze numeric columns
        for col in numeric:
            stats = numerical_summary.get(col, {}) if isinstance(numerical_summary, dict) else {}
            skew = stats.get('skew', 0)
            missing_pct = stats.get('missing_pct', stats.get('missing_percent', 0)) or 0

            # Choose scaler based on skew
            scaler = 'StandardScaler'
            try:
                if abs(float(skew)) > 1.0:
                    scaler = 'RobustScaler'
            except Exception:
                scaler = 'StandardScaler'

            suggested['scaling'][col] = scaler
            suggested['imputation'][col] = 'median' if missing_pct >= 5 else 'mean'

            # Select feature unless id-like or constant
            col_stats = stats
            if not is_id_like(col_stats) and col_stats.get('unique', col_stats.get('nunique', 1)) != 1:
                selected_features.append(col)

        # Analyze categorical columns
        for col in categorical:
            stats = categorical_summary.get(col, {}) if isinstance(categorical_summary, dict) else {}
            missing_pct = stats.get('missing_pct', stats.get('missing_percent', 0)) or 0
            unique = stats.get('unique') or stats.get('nunique') or stats.get('unique_count')

            # Determine encoding strategy
            encoding = 'one-hot'
            try:
                if unique and int(unique) > 30:
                    encoding = 'target'  # high-cardinality
                elif unique and int(unique) > 10:
                    encoding = 'frequency'
            except Exception:
                encoding = 'one-hot'

            suggested['encoding'][col] = encoding
            suggested['imputation'][col] = 'most_frequent' if missing_pct < 50 else 'constant:unknown'

            if not is_id_like(stats):
                selected_features.append(col)

        # Deduplicate selected features preserving order
        seen = set()
        selected_features = [x for x in selected_features if not (x in seen or seen.add(x))]

        # Build processed_sample from csv_text or from small eda sample if present
        processed_sample = []
        sample_rows = []
        sample_header = []
        if req.csv_text:
            try:
                f = io.StringIO(req.csv_text)
                reader = csv.reader(f)
                rows = list(reader)
                if rows:
                    sample_header = [c.strip().strip('"\'') for c in rows[0]]
                    for r in rows[1:6]:
                        # map header -> value
                        row = {sample_header[i]: (r[i] if i < len(r) else '') for i in range(len(sample_header))}
                        sample_rows.append(row)
            except Exception:
                sample_rows = []

        # If no csv_text but eda contains small sample
        if not sample_rows and isinstance(eda.get('sample'), list) and eda.get('sample'):
            sample_rows = eda.get('sample')[:5]

        # Apply lightweight transforms to sample rows to illustrate preprocessing
        for row in sample_rows:
            new_row = {}
            for k, v in row.items():
                if k in suggested['scaling']:
                    # attempt numeric conversion and scale using sample mean/std
                    try:
                        vals = [float(r.get(k)) for r in sample_rows if r.get(k) is not None and r.get(k) != '']
                        m = mean(vals) if vals else 0
                        sd = pstdev(vals) if len(vals) > 1 else 1
                        new_row[k] = (float(v) - m) / sd if v != '' else None
                    except Exception:
                        new_row[k] = v
                elif k in suggested['encoding'] and suggested['encoding'][k] == 'one-hot':
                    # For one-hot, keep original value and also create a single one-hot placeholder for first value
                    new_row[k] = v
                    if v is not None:
                        key = f"{k}__{v}"
                        new_row[key] = 1
                else:
                    new_row[k] = v
            processed_sample.append(new_row)

        result = {
            'status': 'success',
            'suggested_transforms': suggested,
            'feature_summary': {
                'numeric': numeric,
                'categorical': categorical,
                'selected_features': selected_features,
                'count': len(numeric) + len(categorical)
            },
            'processed_sample': processed_sample
        }

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("FP processing failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
