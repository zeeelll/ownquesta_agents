import sys
import json
import io
import ast
import traceback
import base64
import os
import re

_globals = {}


def _safe_load_feature_names() -> list[str]:
    """Load feature names from globals or feature_names.pkl when available."""
    names = _globals.get("feature_names")
    if isinstance(names, list) and all(isinstance(n, str) for n in names):
        return names

    path = _globals.get("feature_names_path") or "feature_names.pkl"
    if not isinstance(path, str) or not os.path.exists(path):
        return []

    try:
        import joblib

        loaded = joblib.load(path)
        if isinstance(loaded, list) and all(isinstance(n, str) for n in loaded):
            _globals["feature_names"] = loaded
            return loaded
    except Exception:
        pass

    try:
        import pickle

        with open(path, "rb") as f:
            loaded = pickle.load(f)
        if isinstance(loaded, list) and all(isinstance(n, str) for n in loaded):
            _globals["feature_names"] = loaded
            return loaded
    except Exception:
        pass

    return []


def _pick_model_obj():
    for key in ("model", "pipeline", "clf", "reg", "best_model", "estimator"):
        obj = _globals.get(key)
        if obj is not None and hasattr(obj, "predict"):
            return obj
    return None


def _pick_input_obj():
    for key in ("input_data", "predict_input", "payload", "request_data", "features", "X_input", "X_test"):
        if key in _globals:
            return _globals.get(key)
    return None


def _to_df(input_obj):
    import pandas as pd

    if isinstance(input_obj, pd.DataFrame):
        return input_obj.copy()
    if isinstance(input_obj, pd.Series):
        return pd.DataFrame([input_obj.to_dict()])
    if isinstance(input_obj, dict):
        return pd.DataFrame([input_obj])
    if isinstance(input_obj, (list, tuple)):
        if input_obj and isinstance(input_obj[0], dict):
            return pd.DataFrame(list(input_obj))
        return pd.DataFrame([list(input_obj)])
    return pd.DataFrame([input_obj])


def _align_to_features(df, feature_names: list[str]):
    # Add any missing training-time features with default 0 values.
    for feature in feature_names:
        if feature not in df.columns:
            df[feature] = 0
    return df[feature_names]


def auto_fix_features(model, input_data):
    expected_features = getattr(model, "n_features_in_", None)

    if hasattr(model, "feature_names_in_"):
        feature_names = [str(name) for name in list(model.feature_names_in_)]
    else:
        feature_names = _safe_load_feature_names()

    df = _to_df(input_data)
    input_feature_count = len(df.columns)

    if expected_features is not None and input_feature_count == int(expected_features) and not feature_names:
        return df

    if not feature_names:
        # Fall back to count-only fix when feature names are unavailable.
        if expected_features is not None and input_feature_count < int(expected_features):
            missing = int(expected_features) - input_feature_count
            for i in range(missing):
                df[f"_missing_feature_{i}"] = 0
        if expected_features is not None and input_feature_count > int(expected_features):
            return df.iloc[:, : int(expected_features)]
        return df

    return _align_to_features(df, feature_names)


def _extract_feature_mismatch(error_text: str):
    text = str(error_text or "")
    pattern = r"X has\s+(\d+)\s+features?,\s+but\s+.*?expecting\s+(\d+)\s+features?"
    m = re.search(pattern, text, flags=re.IGNORECASE)
    if m:
        try:
            return int(m.group(1)), int(m.group(2))
        except Exception:
            return None
    return None


def _attempt_feature_alignment_recovery(error_text: str):
    mismatch = _extract_feature_mismatch(error_text)
    if not mismatch:
        return None

    model = _pick_model_obj()
    input_obj = _pick_input_obj()
    if model is None or input_obj is None:
        return None

    feature_names = _safe_load_feature_names()
    if not feature_names and not hasattr(model, "feature_names_in_"):
        return None

    try:
        original_df = _to_df(input_obj)
        aligned = auto_fix_features(model, input_obj)
        pred = model.predict(aligned)
        _globals["_last_aligned_input"] = aligned
        _globals["_last_prediction"] = pred
        return (
            "Auto-aligned prediction input using training feature names.\n"
            f"Expected features: {getattr(model, 'n_features_in_', len(aligned.columns))} | Received: {len(original_df.columns)}\n"
            f"Prediction: {pred}\n"
        )
    except Exception:
        return None


def _auto_print(code: str, buf: io.StringIO) -> None:
    """Jupyter-like behaviour: if the last statement is an expression and
    nothing was printed, evaluate it and write its repr to *buf*.
    Handles pandas DataFrames / Series with pretty-print.
    """
    try:
        tree = ast.parse(code)
        if not tree.body:
            return
        last = tree.body[-1]
        if not isinstance(last, ast.Expr):
            return
        # Compile the last expression in 'eval' mode
        expr_code = compile(ast.Expression(body=last.value), "<cell>", "eval")
        result = eval(expr_code, _globals)
        if result is None:
            return
        # Pretty-print pandas objects
        try:
            import pandas as _pd
            if isinstance(result, (_pd.DataFrame, _pd.Series)):
                buf.write(result.to_string() + "\n")
                return
        except ImportError:
            pass
        # Pretty-print numpy arrays
        try:
            import numpy as _np
            if isinstance(result, _np.ndarray):
                buf.write(_np.array2string(result, threshold=200) + "\n")
                return
        except ImportError:
            pass
        buf.write(repr(result) + "\n")
    except Exception:
        pass  # auto-print is best-effort; never crash the cell


def _capture_charts() -> list[str]:
    """Capture all open matplotlib/seaborn figures as base64-encoded PNG strings.
    Returns an empty list if matplotlib is not imported or no figures are open.
    """
    charts: list[str] = []
    try:
        import matplotlib
        matplotlib.use("Agg")          # non-interactive, file-only backend
        import matplotlib.pyplot as plt
        fig_nums = plt.get_fignums()
        if not fig_nums:
            return charts
        for num in fig_nums:
            fig = plt.figure(num)
            buf = io.BytesIO()
            fig.savefig(
                buf, format="png", bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none", dpi=100,
            )
            buf.seek(0)
            charts.append(base64.b64encode(buf.read()).decode("utf-8"))
        plt.close("all")
    except Exception:
        pass   # matplotlib may not be installed or no figures exist
    return charts


def run():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            code = payload.get("code", "")
        except Exception:
            sys.stdout.write(json.dumps({"stdout": "", "error": "invalid json", "charts": []}) + "\n")
            sys.stdout.flush()
            continue

        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        error = None
        try:
            exec(compile(code, "<cell>", "exec"), _globals)
        except Exception:
            error = traceback.format_exc()
            recovered = _attempt_feature_alignment_recovery(error)
            if recovered:
                error = None
                buf.write(recovered)
        finally:
            sys.stdout = old_stdout

        # Jupyter-like: auto-print last expression if nothing was printed
        if not buf.getvalue() and not error:
            _auto_print(code, buf)

        # Capture any matplotlib figures produced by the code
        charts = _capture_charts()

        sys.stdout.write(
            json.dumps({"stdout": buf.getvalue(), "error": error, "charts": charts}) + "\n"
        )
        sys.stdout.flush()


if __name__ == "__main__":
    run()
