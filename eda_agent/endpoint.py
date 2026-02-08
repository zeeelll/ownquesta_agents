from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import json
import shutil
import os

from .config import (
    dataset_overview,
    dataset_shape,
    column_analysis,
    column_names,
    dataset_info,
    comprehensive_statistics,
    summary_statistics,
    statistical_measures,
    advanced_distribution_analysis,
    distribution_analysis,
    advanced_correlation_analysis,
    correlation_matrix,
    data_quality_analysis,
    outlier_detection,
    unique_values,
    handle_missing_values,
    drop_duplicates,
    data_distribution,
)

router = APIRouter()

UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


# NOTE: `/run` endpoint removed to keep API surface minimal. Use `/upload_and_run` only.


@router.post("/upload_and_run")
async def upload_and_run(file: UploadFile = File(...)):
    """Upload a CSV and run EDA locally. This endpoint runs the EDA tools directly
    (no LLM). It returns a clear, structured JSON response with a short summary
    and full tool outputs."""
    if not file.filename.lower().endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV uploads are supported")

    dest = UPLOAD_DIR / file.filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    response = {
        "filename": file.filename,
        "path": str(dest),
        "status": "running",
        "results": None,
    }

    # Run local tools and collect JSON-friendly outputs
    tools = [
        dataset_overview,
        dataset_shape,
        column_analysis,
        column_names,
        dataset_info,
        comprehensive_statistics,
        summary_statistics,
        statistical_measures,
        advanced_distribution_analysis,
        distribution_analysis,
        advanced_correlation_analysis,
        correlation_matrix,
        data_quality_analysis,
        outlier_detection,
        unique_values,
        handle_missing_values,
        drop_duplicates,
        data_distribution,
    ]
    results = {}
    for fn in tools:
        name = getattr(fn, "name", None) or getattr(fn, "__name__", None) or repr(fn)
        try:
            result = None
            # Try calling common call patterns to support StructuredTool and plain functions
            if callable(fn):
                try:
                    result = fn(str(dest))
                except TypeError:
                    try:
                        result = fn(file_path=str(dest))
                    except Exception:
                        try:
                            result = fn({"file_path": str(dest)})
                        except Exception:
                            # Last resort: attempt to call with no args
                            result = fn()
            elif hasattr(fn, "func") and callable(fn.func):
                try:
                    result = fn.func(str(dest))
                except TypeError:
                    result = fn.func(file_path=str(dest))
            elif hasattr(fn, "run") and callable(fn.run):
                result = fn.run(str(dest))
            else:
                # Not callable; store its representation
                result = repr(fn)

            # Attempt to parse JSON strings into native structures
            try:
                if isinstance(result, str):
                    parsed = json.loads(result)
                    results[name] = parsed
                else:
                    results[name] = result
            except Exception:
                results[name] = result
        except Exception as e:
            results[name] = {"error": str(e)}

    response["status"] = "done"
    response["results"] = results
    # Friendly summary
    dataset_shape_res = results.get("dataset_shape")
    try:
        numeric_cols = list(results.get("summary_statistics", {}).get("numeric", {}).keys())
    except Exception:
        numeric_cols = None

    response["summary"] = {
        "rows": dataset_shape_res.get("rows") if isinstance(dataset_shape_res, dict) else None,
        "columns": dataset_shape_res.get("columns") if isinstance(dataset_shape_res, dict) else None,
        "numeric_columns": numeric_cols,
    }

    return JSONResponse(response)
