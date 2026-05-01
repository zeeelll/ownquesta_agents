"""
agents.py - OpenAI-powered AutoML agents for OwnQuesta

Roles:
  1. analyze()          – profile data, feature engineering, top 3 model suggestions
  2. build_pipeline()   – generate full sklearn pipeline cell-by-cell
  3. chat_with_code()   – decide: explain in text OR generate+execute code in notebook
  4. explain_chart()    – interpret a seaborn/matplotlib chart from its code + context
  5. generate_predict() – generate prediction code for the test UI
  6. fix_error()        – guard agent: analyze cell error and return corrected code
  7. web_search()       – Tavily web search (optional; graceful no-op if key not set)
"""
from __future__ import annotations

import json
import os
import re
import hashlib
import time
from typing import Any, Optional


# ── JSON extraction ───────────────────────────────────────────────────────────

def _extract_json(text: str) -> dict:
    """Pull the first complete JSON object out of a GPT reply."""
    # 1. strip markdown fences
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    candidate = fenced.group(1) if fenced else text

    # 2. find outermost { ... }
    start = candidate.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in response:\n{text[:400]}")

    depth = 0
    for i, ch in enumerate(candidate[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(candidate[start : i + 1])
                except json.JSONDecodeError:
                    pass  # keep scanning

    # 3. last resort – try the whole text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        raise ValueError(f"Could not parse JSON from response:\n{text[:400]}")


# ── Prompt templates ──────────────────────────────────────────────────────────

_ANALYSIS_SYS = """\
You are a senior ML engineer and data scientist.
Respond ONLY with valid JSON — no prose before or after the JSON block.
When analysing a dataset you always:
  • Identify the ML problem type (classification or regression)
  • Spot quality issues (nulls, skew, cardinality)
  • Propose feature-engineering steps and write clean Python code for them
  • Recommend the top 3 scikit-learn models with clear reasoning
  • Detect leakage risks (target leakage, post-outcome columns, IDs)
  • Check class imbalance and mention balancing strategy when needed
  • Explain trade-offs between bias/variance and interpretability

CRITICAL — always use these modern API patterns in feature_engineering_code:
  • OneHotEncoder: use sparse_output=False  (NOT sparse=False — removed in sklearn 1.2)
  • DataFrame merge/combine: use pd.concat([...])  (NOT DataFrame.append() — removed in pandas 2.0)
  • fillna with numeric value on object columns: cast first, e.g. df[col].astype(float).fillna(0)
  • LabelEncoder: fit on df[col].astype(str) to avoid mixed-type issues
  • The feature_engineering_code must be a complete, runnable Python script from the very first line.
    Define ALL variables at the top — never reference a variable before it is defined.
    End with a print() confirming the result shape, e.g. print('FE done. Shape:', df_processed.shape)

MANDATORY DATA CLEANING — always include this block at the TOP of feature_engineering_code
BEFORE any other transformation, to catch common real-world data quality issues.
IMPORTANT: use the sample-first pattern to avoid slow processing on large datasets.

    import numpy as np
    df_processed = df.copy()

    # 1. Strip comma/currency/percent from numeric-looking string columns
    #    Sample 20 rows first — only scan full column if sample confirms numeric.
    for _col in df_processed.select_dtypes(include='object').columns:
        _s = df_processed[_col].dropna().head(20).astype(str).str.strip()
        _s = _s.str.replace(r'[\\$\\£\\€,]', '', regex=True).str.replace('%', '', regex=False)
        if pd.to_numeric(_s, errors='coerce').notna().sum() / max(len(_s), 1) >= 0.7:
            _full = (df_processed[_col].astype(str).str.strip()
                     .str.replace(r'[\\$\\£\\€,]', '', regex=True)
                     .str.replace('%', '', regex=False))
            df_processed[_col] = pd.to_numeric(_full, errors='coerce')

    # 2. Convert datetime-looking string columns to Unix timestamp
    #    Sample 5 rows first to avoid slow pd.to_datetime on non-date columns.
    for _col in df_processed.select_dtypes(include='object').columns:
        try:
            _s5 = df_processed[_col].dropna().head(5)
            if pd.to_datetime(_s5, errors='coerce').notna().all():
                df_processed[_col] = pd.to_datetime(
                    df_processed[_col], errors='coerce'
                ).astype('int64') // 10**9
        except Exception:
            pass

    # 3. Drop ID-like / high-cardinality string columns BEFORE any encoding
    #    Columns with > 50 unique values or unique ratio > 0.5 are IDs / free-text — drop them.
    for _col in df_processed.select_dtypes(include='object').columns.tolist():
        if df_processed[_col].nunique() > 50 or df_processed[_col].nunique() / max(len(df_processed), 1) > 0.5:
            df_processed = df_processed.drop(columns=[_col])

    # 4. Replace inf values
    df_processed = df_processed.replace([np.inf, -np.inf], np.nan)
"""

_ANALYSIS_USER = """\
Dataset file: {filename}
Target column hint: {target}

Python execution output (shape / dtypes / head / describe / distributions):
---
{profile}
---

Return EXACTLY the following JSON (no extra keys, no markdown outside the block):
{{
  "problem_type": "classification",
  "target_column": "<confirmed or best-guess column name>",
  "dataset_summary": "<2-3 sentence plain-English overview>",
  "feature_analysis": "<what columns exist, their types, any quality issues>",
  "missing_values_note": "<how to handle NaN — drop / impute / flag>",
  "feature_engineering_reasoning": "<what transformations are needed and why>",
  "feature_engineering_code": "<complete Python — 'df' is already in scope; use pandas/sklearn; end with print('Feature engineering done. Shape:', df_processed.shape) or similar. IMPORTANT: when using OneHotEncoder always fit on a DataFrame slice (not .values) so feature_names_in_ is set, and call get_feature_names_out() with NO arguments — never pass column names explicitly to get_feature_names_out().>",
  "models": [
    {{
      "rank": 1,
      "name": "<sklearn class e.g. RandomForestClassifier>",
      "display_name": "<human-friendly label>",
      "reasoning": "<why this model suits this specific dataset>",
      "pros": ["<pro1>", "<pro2>"],
      "cons": ["<con1>"],
      "expected_performance": "<brief expectation>"
    }},
    {{ "rank": 2, "name": "...", "display_name": "...", "reasoning": "...", "pros": [], "cons": [], "expected_performance": "..." }},
    {{ "rank": 3, "name": "...", "display_name": "...", "reasoning": "...", "pros": [], "cons": [], "expected_performance": "..." }}
  ]
}}
"""

_PIPELINE_SYS = """\
You are a senior ML engineer. Build complete, runnable scikit-learn ML pipelines.
Variables already in scope: 'df' (original load) and possibly 'df_processed' (after feature engineering).
Use whichever is appropriate. Respond ONLY with valid JSON — no prose outside the block.

SPEED CONTRACTS — every model cell must finish in < 90 seconds:
  • RandomForest / ExtraTrees   → n_estimators=100, max_depth=15, n_jobs=-1
  • GradientBoosting            → n_estimators=100, max_depth=6
  • HistGradientBoosting        → max_iter=100, max_depth=6
  • LogisticRegression          → solver='lbfgs', max_iter=500
  • SVC / SVR on > 2000 rows    → switch to LinearSVC(max_iter=2000) / LinearSVR(max_iter=2000)
  • KNeighbors                  → n_neighbors=5, algorithm='ball_tree', n_jobs=-1
  • DecisionTree                → max_depth=15
  • NEVER use GridSearchCV, RandomizedSearchCV — multiply training time unacceptably
  • Cross-validation: ONLY for datasets < 3000 rows, max cv=3 folds; otherwise skip it
  • Hyperparameter tuning: use a tiny manual candidate set (max 3 configs) and choose best validation metric
  • For datasets >= 5000 rows: skip tuning loop and use one fast default config

Always use seaborn with dark theme for any visualisation:
  import seaborn as sns, matplotlib.pyplot as plt
  sns.set_theme(style='darkgrid', palette='muted')
  plt.style.use('dark_background')
  plt.rcParams.update({{'figure.facecolor':'#0d1117','axes.facecolor':'#161b22','text.color':'#e6edf3','axes.labelcolor':'#8b949e','xtick.color':'#8b949e','ytick.color':'#8b949e'}})
Never call plt.show() — the backend captures figures automatically.

CRITICAL chart quality rules for pipeline visualisations:
  • Confusion matrix: ALWAYS use cmap='Blues', annot=True, fmt='d'.
    Add plt.xlabel('Predicted Label'), plt.ylabel('True Label'), plt.title('Confusion Matrix'), plt.tight_layout().
  • Residual plot: use alpha=0.5, add a horizontal line at y=0 with plt.axhline(0, color='red', linestyle='--').
    Label axes (Predicted vs Residuals) and add a title.
  • Every chart must have a descriptive plt.title() and axis labels. Call plt.tight_layout() before the end of each cell.
  • After each visualisation, print() a 1–2 sentence insight about what the chart reveals.
"""

_PIPELINE_USER = """\
Dataset: {filename}
Problem type: {problem_type}
Target column: {target}
Selected model: {model_name}

Dataset profile (first lines of execution output):
---
{profile}
---

Feature-engineering code that already ran (df_processed may exist):
```python
{fe_code}
```

Generate a complete pipeline split into logical cells.
For classification: include accuracy_score, classification_report, and a seaborn confusion matrix heatmap.
For regression: include RMSE, MAE, R² score and a seaborn residual plot.
Include StratifiedKFold cross-validation for classification (KFold for regression) ONLY if dataset has < 3000 rows.
For datasets >= 3000 rows: SKIP cross-validation entirely — a single train/test split is sufficient and much faster.
Save the trained model to a variable called `model`. List the exact feature column names used for X.
Add print statements so each cell has visible output. Do NOT call plt.show().

PERFORMANCE RULES — training must complete within 90 seconds on any dataset size:
  • RandomForest / ExtraTrees: ALWAYS use n_estimators=100, max_depth=15, n_jobs=-1
  • GradientBoosting / HistGradientBoosting: max_iter=100, max_depth=6
  • LogisticRegression: solver='lbfgs', max_iter=500, C=1.0
  • SVC / SVR: NEVER use kernel='rbf' on datasets > 2000 rows — use LinearSVC/LinearSVR instead
  • KNeighbors: n_neighbors=5, algorithm='ball_tree', n_jobs=-1
  • Any other model: add the fastest equivalent hyperparameters to prevent runaway training
  • For datasets > 5000 rows: add n_jobs=-1 wherever supported
  • NEVER use GridSearchCV or RandomizedSearchCV — they multiply training time by n_iter × n_folds

ACCURACY RULES — improve predictions without violating the 90-second budget:
  • For classification: if class imbalance ratio > 1.8, use class_weight='balanced' where supported
  • Add a lightweight validation split from training data (or cv<=3 on small data) to compare up to 3 parameter candidates
  • Select the final model config using the relevant metric: F1/ROC-AUC for classification, RMSE/MAE for regression
  • Print why the chosen config won and the top 2 alternatives

MANDATORY — every pipeline cell that builds X MUST apply ALL steps below in order:

  STEP A — Numeric string cleaning (sample-first for performance on large datasets):
    for _col in df_processed.select_dtypes(include='object').columns:
        _s = df_processed[_col].dropna().head(20).astype(str).str.strip()
        _s = _s.str.replace(r'[\\$\\£\\€,]', '', regex=True).str.replace('%', '', regex=False)
        if pd.to_numeric(_s, errors='coerce').notna().sum() / max(len(_s), 1) >= 0.7:
            _full = (df_processed[_col].astype(str).str.strip()
                     .str.replace(r'[\\$\\£\\€,]', '', regex=True)
                     .str.replace('%', '', regex=False))
            df_processed[_col] = pd.to_numeric(_full, errors='coerce')

  STEP B — Datetime column encoding (sample 5 rows to avoid slow parsing):
    for _col in df_processed.select_dtypes(include=['datetime64','object']).columns:
        try:
            _s5 = df_processed[_col].dropna().head(5)
            if pd.to_datetime(_s5, errors='coerce').notna().all():
                df_processed[_col] = pd.to_datetime(
                    df_processed[_col], errors='coerce'
                ).astype('int64') // 10**9
        except Exception: pass

  STEP C — Drop ID-like / high-cardinality string columns, then one-hot encode, then cast:
    X = df_processed.drop(columns=[target_column])
    _obj = X.select_dtypes(include='object').columns.tolist()
    # Drop ID-like columns: unique ratio > 0.5 OR nunique > 50 (prevents ArrayMemoryError)
    _ids = [c for c in _obj if X[c].nunique() / max(len(X), 1) > 0.5 or X[c].nunique() > 50]
    X = X.drop(columns=_ids, errors='ignore')
    _cats = X.select_dtypes(include='object').columns.tolist()
    if _cats:
        X = pd.get_dummies(X, columns=_cats, drop_first=True)
    X = X.astype(float)

  STEP D — Remove inf/NaN:
    import numpy as np
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

  STEP E — After train/test split, align columns (prevents shape mismatch after get_dummies):
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

  Never pass a DataFrame with string/object dtype to sklearn — it will always raise ValueError.

Return EXACTLY:
{{
  "reasoning": "<step-by-step explanation of the approach and every key decision>",
  "feature_columns": ["<col1>", "<col2>"],
  "cells": [
    {{
      "title": "<short title>",
      "description": "<what this cell does and why>",
      "code": "<complete, runnable Python>",
      "has_chart": true
    }}
  ]
}}
"""

_CHAT_DECISION_SYS = """\
You are an expert ML assistant embedded in a Jupyter notebook environment.
When the user asks a question, decide whether to:
  - "explain" : give a concise but useful text answer (typically 120-260 words, no code block needed)
  - "execute"  : write Python code that should run in the notebook (plots, computations, new analyses)

Rules:
• Questions about WHY / WHAT / HOW something works → "explain"
• Requests to plot, compute, explore, or show new results → "execute"
• Always use seaborn dark-theme for plots; never call plt.show()
• The executed code must be self-contained (use variables already in scope: df, df_processed, model, X_test, y_test, etc.)
• ALWAYS use print() to display results — never rely on bare variable names as the last line
• Wrap numeric results, dataframes, and computation outputs in print()
• If no useful printable output can be produced, use action="explain" instead

MODEL-SELECTION REASONING RULES (critical):
• If the user asks "why this model", "why these top 3", "which is best for my dataset", or any model-comparison question,
  ALWAYS choose action="explain" and provide dataset-grounded reasoning.
• Use current context fields (problem_type, feature_analysis, missing_values_note, models, eda summary/risk flags) to justify decisions.
• Structure the explanation in this order:
  1) Dataset signals that influenced choice (size, feature types, class balance, noise, missingness, cardinality)
  2) Explainability trade-off (interpretability vs complexity for this dataset)
  3) Performance expectation (why chosen model should generalize better than alternatives)
  4) Risks and fallback strategy (when to prefer model #2 or #3)
• Do not give generic textbook reasons; tie each point to known dataset context.
• Mention at least one concrete reason for each of the top-3 models when asked about ranking.

Respond ONLY with valid JSON:
{{
  "action": "explain" | "execute",
  "reply": "<concise explanation shown in chat — no code blocks here>",
  "code": "<complete Python if action=execute, else omit>",
  "title": "<short cell title if action=execute, else omit>",
  "is_chart": true | false
}}
"""

_CHART_EXPLAIN_SYS = """\
You are a data scientist explaining a chart to a non-expert.
Given the Python code that generated a seaborn/matplotlib chart and the ML context,
explain in 3–5 sentences what the chart reveals about the data or model performance.
Focus on data insights, not just "this is a heatmap". Be specific and actionable.
Respond with plain text only — no markdown headers, no bullet points.
"""

_PREDICT_CODE_SYS = """\
You are an ML engineer. Given a trained sklearn model (variable 'model') in scope,
feature column names, and user-provided input values, generate Python code
to make a prediction and print the result clearly.
Handle type conversions (int/float for numeric, str for categorical).
Also print a confidence / probability breakdown if the model supports predict_proba.
Respond ONLY with the Python code — no markdown fences, no explanation.
"""

_FIX_SYS = """\
You are an expert Python/ML debugging guard agent.
Given failing code and its error traceback, identify the EXACT root cause using the patterns
below and return fully corrected code. Match the error message literally before applying a fix.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PATTERN 1 — COMMA / CURRENCY / PERCENT FORMATTED NUMERIC STRINGS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Trigger: "could not convert string to float: '1,648'" OR the bad value contains
         commas ('1,234'), currency symbols ('$45'), percent signs ('12%'),
         spaces (' 99 '), or other non-numeric characters that look numeric.
Root cause: A column that should be numeric was read as object/string because of
            formatting. pandas astype(float) or sklearn cannot handle these.
THE FIX — apply AT THE TOP of the cell, right after imports, before ANY X/y split:

    import re as _re
    def _clean_numeric_col(series):
        \"\"\"Strip currency symbols, commas, percent signs; return float Series.\"\"\"
        cleaned = series.astype(str).str.strip()
        cleaned = cleaned.str.replace(r'[\\$\\£\\€\\,]', '', regex=True)
        cleaned = cleaned.str.replace('%', '', regex=False)
        cleaned = cleaned.str.replace(r'\\s+', '', regex=True)
        return pd.to_numeric(cleaned, errors='coerce')

    # Auto-detect and fix all numeric-looking string columns in df/df_processed
    _src = df_processed if 'df_processed' in dir() and df_processed is not None else df
    for _col in _src.select_dtypes(include='object').columns:
        _cleaned = _clean_numeric_col(_src[_col])
        if _cleaned.notna().sum() / max(len(_cleaned), 1) >= 0.7:
            _src[_col] = _cleaned
    df_processed = _src

ALSO: if the error names a specific column (e.g. "could not convert string to float: '1,648'"),
apply _clean_numeric_col() to that column directly as well.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PATTERN 2 — CATEGORICAL / STRING COLUMNS NOT ENCODED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Trigger: "could not convert string to float: 'some_text'" where the bad value is
         clearly a category label or ID (not a formatted number).
Root cause: object/string columns were not one-hot encoded before sklearn fitting,
            OR an ID-like column slipped through because its cardinality ratio was < 0.9.
THE FIX — FIRST, identify the column containing the bad value from the error message,
then apply the block below right before building X (after PATTERN 1 cleaning):

    # Step 1: Parse the bad column name from the error if possible
    # e.g. "could not convert string to float: 'AG-2011-2040'" — scan all object columns
    # for that exact value and drop any column that contains it.
    import re as _re
    _bad_val_match = _re.search(r"could not convert string to float: '([^']+)'", str(error_msg))
    if _bad_val_match:
        _bv = _bad_val_match.group(1)
        _bad_cols = [c for c in df_processed.select_dtypes(include='object').columns
                     if df_processed[c].astype(str).str.contains(_re.escape(_bv), na=False).any()]
        df_processed = df_processed.drop(columns=_bad_cols, errors='ignore')

    # Step 2: Drop ID-like / high-cardinality string columns
    # Threshold: unique ratio > 0.5 OR more than 50 unique values (prevents ArrayMemoryError)
    _obj_cols = df_processed.select_dtypes(include='object').columns.tolist()
    _id_cols  = [c for c in _obj_cols
                 if df_processed[c].nunique() / max(len(df_processed), 1) > 0.5
                 or df_processed[c].nunique() > 50]
    df_processed = df_processed.drop(columns=_id_cols, errors='ignore')
    X = df_processed.drop(columns=[target_column], errors='ignore')

    # Step 3: One-hot encode remaining low-cardinality object columns
    _cat_cols = X.select_dtypes(include='object').columns.tolist()
    if _cat_cols:
        X = pd.get_dummies(X, columns=_cat_cols, drop_first=True)
    X = X.astype(float)

  • Apply the SAME transformations to X_train and X_test.
  • If X is already split, rebuild from df_processed then re-split.
  • Common ID column names to always drop: any col matching r'^[A-Z]+-\\d+(-\\d+)?$', or
    named like 'order_id', 'customer_id', 'row_id', 'id', 'order id', 'customer id'.
  • NEVER call pd.get_dummies on y.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PATTERN 3 — TRAIN/TEST COLUMN MISMATCH AFTER GET_DUMMIES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Trigger: "ValueError: X has N features but model is expecting M features" OR
         KeyError / shape mismatch after get_dummies on separate train/test splits.
Root cause: get_dummies on X_train and X_test independently can produce different
            column sets if some categories only appear in one split.
THE FIX:
    X_train = pd.get_dummies(X_train, drop_first=True)
    X_test  = pd.get_dummies(X_test,  drop_first=True)
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PATTERN 4 — NaN / INFINITY IN FEATURES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Trigger: "Input X contains NaN" OR "Input X contains infinity or a value too large"
Root cause: Missing values or inf remain in feature matrix after encoding.
THE FIX — after building X (or X_train/X_test):
    import numpy as np
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    # or for train/test:
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test  = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PATTERN 5 — DATETIME COLUMNS PASSED TO SKLEARN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Trigger: "could not convert string to float" where the value looks like a date
         ('2023-01-15', '01/15/2023') OR dtype is datetime64.
Root cause: Datetime columns must be converted to numeric before sklearn.
THE FIX:
    _dt_cols = X.select_dtypes(include=['datetime64', 'datetime']).columns.tolist()
    for _dc in _dt_cols:
        X[_dc] = pd.to_datetime(X[_dc], errors='coerce').astype('int64') // 10**9
    # Also detect string columns that look like dates:
    for _sc in X.select_dtypes(include='object').columns:
        try:
            _parsed = pd.to_datetime(X[_sc], errors='coerce')
            if _parsed.notna().sum() / max(len(X), 1) > 0.7:
                X[_sc] = _parsed.astype('int64') // 10**9
        except Exception:
            pass

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PATTERN 6 — SKLEARN / PANDAS API CHANGES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  - sklearn >= 1.2: OneHotEncoder(sparse=False) → OneHotEncoder(sparse_output=False)
  - sklearn >= 1.2: many transformers renamed 'sparse' kwarg → 'sparse_output'
  - scipy/sklearn: sparse matrices need .toarray() when dense array expected
  - pandas >= 2.0: DataFrame.append() removed → use pd.concat()
  - pandas >= 2.0: fillna() with numeric on object column → cast first
  - matplotlib: NEVER call plt.show() — backend captures automatically

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PATTERN 7 — NAMEERROR / MISSING VARIABLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  - NameError for df_processed → add: df_processed = df.copy()
  - NameError for X_train/X_test/y_train/y_test → add df_processed = df.copy()
    before the train_test_split line.
  - NameError on any other variable → check if it was defined in a previous cell
    and recreate it inline at the top of fixed_code.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PATTERN 8 — TARGET COLUMN LEAKING INTO FEATURES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Trigger: Model accuracy is exactly 1.0, or shape/index errors on y.
Root cause: Target column was included in X.
THE FIX: X = df_processed.drop(columns=[target_column])

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PATTERN 9 — EXECUTION TIMEOUT (code too slow)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Trigger: "Execution timed out" in the error message.
Root cause: The code is correct but too slow — likely caused by one of:
  a) pd.to_datetime() called on every object column of a large DataFrame
  b) A nested loop iterating over rows (never use row-level loops with pandas)
  c) Overly complex feature engineering on a large dataset
THE FIX — replace slow patterns with fast vectorised equivalents:

  a) Slow datetime detection → use sample-first approach:
     REPLACE:
       _parsed = pd.to_datetime(df_processed[_col], errors='coerce')
       if _parsed.notna().sum() / max(len(df_processed), 1) > 0.7: ...
     WITH:
       _s5 = df_processed[_col].dropna().head(5)
       if pd.to_datetime(_s5, errors='coerce').notna().all():
           df_processed[_col] = pd.to_datetime(df_processed[_col], errors='coerce').astype('int64') // 10**9

  b) Slow numeric cleaning → use sample-first approach:
     REPLACE: iterate and apply to_numeric on full columns
     WITH: check a 20-row sample first; only convert full column if sample confirms numeric
       _s = df_processed[_col].dropna().head(20).astype(str).str.strip()
       _s = _s.str.replace(r'[$£€,]', '', regex=True).str.replace('%', '', regex=False)
       if pd.to_numeric(_s, errors='coerce').notna().sum() / max(len(_s), 1) >= 0.7:
           # apply full column conversion

  c) Row loops → replace with vectorised pandas operations (apply, map, str methods)

  NOTE: The session kernel was reset by the timeout. The fixed code must be
  fully self-contained — all variables (df_processed, X, y, etc.) must be
  recreated from scratch starting from `df` (the globally loaded DataFrame).
  Begin fixed_code with: df_processed = df.copy()

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PATTERN 10 — MODEL TRAINING TIMEOUT (model too slow to fit)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Trigger: "timed out" error AND the failing cell contains model.fit() or cross_val_score().
Root cause: Model hyperparameters are too expensive for the dataset size, or
            cross-validation multiplies training time beyond the timeout budget.
THE FIX — apply ALL of these in the fixed_code:

  1. Add fast hyperparameters to the model:
       RandomForestClassifier/Regressor  → n_estimators=100, max_depth=15, n_jobs=-1
       GradientBoostingClassifier        → n_estimators=100, max_depth=6, learning_rate=0.1
       HistGradientBoostingClassifier    → max_iter=100, max_depth=6
       LogisticRegression                → solver='lbfgs', max_iter=500, C=1.0
       SVC / SVR                         → replace with LinearSVC(max_iter=2000) / LinearSVR(max_iter=2000)
       KNeighborsClassifier/Regressor    → n_neighbors=5, algorithm='ball_tree', n_jobs=-1
       DecisionTreeClassifier/Regressor  → max_depth=15
       ExtraTreesClassifier/Regressor    → n_estimators=100, max_depth=15, n_jobs=-1

  2. Remove cross-validation entirely — replace cross_val_score / StratifiedKFold loop
     with a single train/test split:
       REPLACE:  scores = cross_val_score(model, X, y, cv=5)
       WITH:     model.fit(X_train, y_train); scores = [model.score(X_test, y_test)]

  3. Rebuild all variables from df (kernel was reset):
       df_processed = df.copy()
       # ... apply STEP A-E cleaning ...
       X = df_processed.drop(columns=[target_column])
       # ... encode, clean inf/NaN ...
       X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PATTERN 11 — ARRAYMEMORYERROR / MEMORYERROR FROM GET_DUMMIES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Trigger: "ArrayMemoryError" OR "MemoryError" OR "Unable to allocate" in error message,
         especially when `pd.get_dummies` or `astype(float)` is in the failing code.
Root cause: A high-cardinality string column (e.g. Order ID with 10 000+ unique values)
            was passed to pd.get_dummies, producing tens of thousands of new columns
            that exhaust available memory.
THE FIX — before ANY call to pd.get_dummies, unconditionally drop all object columns
           with more than 50 unique values:

    _src = df_processed if 'df_processed' in dir() else df.copy()
    # Nuclear drop: any object column with > 50 unique values is an ID / free-text → drop it
    for _c in _src.select_dtypes(include='object').columns.tolist():
        if _src[_c].nunique() > 50:
            _src = _src.drop(columns=[_c])
    df_processed = _src
    X = df_processed.drop(columns=[target_column], errors='ignore')
    _remaining_cats = X.select_dtypes(include='object').columns.tolist()
    if _remaining_cats:
        X = pd.get_dummies(X, columns=_remaining_cats, drop_first=True)
    X = X.astype(float)
    import numpy as np
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

  After this, continue with the normal train/test split and model fitting.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PATTERN 12 — get_feature_names_out() INPUT MISMATCH
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Trigger: "ValueError: input_features is not equal to feature_names_in_"
Root cause: get_feature_names_out(input_features) was called with column names that
            do not exactly match what the encoder was fitted on (its feature_names_in_
            attribute). This mismatch also occurs when the encoder was fitted on a
            numpy array (so feature_names_in_ is not set) but explicit names are passed.
THE FIX — two sub-cases:

  Sub-case A (most common): encoder fitted on a DataFrame, names passed to get_feature_names_out don't match.
    REPLACE: encoder.get_feature_names_out(['col1', 'col2', ...])
    WITH:    encoder.get_feature_names_out()   # uses feature_names_in_ automatically

  Sub-case B: encoder fitted on a numpy array (.values), so feature_names_in_ is unset.
    REPLACE: ohe.fit_transform(df[cols].values)
             ohe.get_feature_names_out(cols)
    WITH:    ohe.fit_transform(df[cols])       # DataFrame input sets feature_names_in_
             ohe.get_feature_names_out()       # now works without args

  Universal safe pattern for OneHotEncoder feature engineering:
    _cat_cols = ['Contract_Type', 'Payment_Method']  # or whatever categorical cols
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded = ohe.fit_transform(df_processed[_cat_cols])   # fit on DataFrame
    encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(), index=df_processed.index)
    df_processed = pd.concat([df_processed.drop(columns=_cat_cols), encoded_df], axis=1)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL RULES FOR fixed_code
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • fixed_code MUST be the ENTIRE original cell from the very first line to the last.
    Do NOT truncate, shorten, or start from the middle. Copy everything, then apply
    the minimal targeted fix for the specific error pattern matched above.
  • All imports at the top — never omit them.
  • Every variable used must be defined earlier in the same block.
  • NEVER call pd.read_csv() / pd.read_excel() with a bare filename. The dataset is
    already loaded as the global `df`. Use the full file_path from context if reload needed.
  • NEVER call pd.get_dummies on y (target) — only on X feature columns.
  • After fixing, add X = X.replace([np.inf, -np.inf], np.nan).fillna(0) as a safety net.
  • If the error contains "timed out" or "kernel was reset": the session kernel was cleared.
    fixed_code MUST start with df_processed = df.copy() and rebuild all variables from df.
    Do NOT reference df_processed, X, y, model, X_train, X_test without recreating them first.
  • If the error is truly obscure and needs current documentation, set needs_search=true.

Respond ONLY with valid JSON — no prose before or after:
{{
  "fixed_code": "<COMPLETE corrected Python — full cell, not a partial snippet>",
  "explanation": "<root cause identified + exact pattern applied + what changed>",
  "needs_search": false,
  "search_query": "<web search query — only populate if needs_search is true>"
}}
"""

_FIX_USER = """\
Failed code:
```python
{code}
```

Error traceback:
```
{error}
```

ML pipeline context: {context}

Web search results (empty = not searched yet):
---
{search_results}
---

Generate the corrected code now.
"""


# ── EDA prompt templates ──────────────────────────────────────────────────────

_EDA_SYS = """\
You are a senior data scientist performing Exploratory Data Analysis.
Given a dataset profile, generate 2-4 focused, runnable Python code cells that explore
the data visually and statistically BEFORE any modelling.

Each cell MUST:
  • Use pandas / seaborn / matplotlib ONLY
  • Use variables already in scope: 'df' (raw) or 'df_processed' (after FE)
  • Use seaborn dark-theme:
      import seaborn as sns, matplotlib.pyplot as plt
      sns.set_theme(style='darkgrid', palette='muted')
      plt.style.use('dark_background')
      plt.rcParams.update({{'figure.facecolor':'#0d1117','axes.facecolor':'#161b22','text.color':'#e6edf3','axes.labelcolor':'#8b949e','xtick.color':'#8b949e','ytick.color':'#8b949e'}})
  • Never call plt.show()
  • Always print() a 1-2 sentence insight about what the chart or analysis reveals
  • Be self-contained (all imports at the top of each cell)
  • Every chart MUST have plt.title(), appropriate axis labels, and plt.tight_layout()

CRITICAL chart quality rules — ALWAYS apply these before generating any chart:
  1. Missing-value heatmap: ONLY generate if df.isnull().sum().sum() > 0.
     If the dataset has NO missing values, skip the plot entirely and instead print:
     "✅ No missing values found in this dataset — no imputation required."
  2. Correlation heatmap: ALWAYS use numeric columns only (df.select_dtypes(include='number')).
     ALWAYS specify cmap='coolwarm', vmin=-1, vmax=1, center=0, annot=True, fmt='.2f', linewidths=0.5.
     This ensures the colour scale always spans the full -1 to +1 range — never a single flat colour.
  3. Before generating ANY chart, verify the plotted data has meaningful variance.
     If all values are identical (e.g. std == 0), skip the plot and print a descriptive message instead.
  4. Target distribution: for classification use countplot; for regression use histplot with kde=True.
     Always label axes and add a title.

Focus on:
  1. Distribution of the target variable
  2. Correlation heatmap for numeric columns (with the quality rules above)
  3. Top feature distributions or box-plots by target
  4. Missing-value summary (heatmap only if nulls exist, otherwise print the ✅ message)

REPORTING RULES:
  • Build a detailed report grounded in observed chart evidence.
  • For every major finding, reference the chart type/source (target distribution, correlation heatmap, box-plot, etc.).
  • Separate facts from recommendations.
  • Include both ML impact (model quality/generalization) and business impact.
  • Keep conclusions actionable and prioritized.

Respond ONLY with valid JSON — no prose before or after.
"""

_EDA_USER = """\
Dataset: {filename}
Problem type: {problem_type}
Target column: {target}

Dataset profile:
---
{profile}
---

Feature analysis notes: {feature_analysis}

Generate EDA cells and a plain-English summary.
Return EXACTLY:
{{
  "cells": [
    {{
      "title": "<short title>",
      "code": "<complete runnable Python>"
    }}
  ],
  "summary": "<2-4 sentence overview of what EDA should reveal>",
  "feature_importance_notes": "<which features seem most predictive and why>",
  "preprocessing_recommendations": "<data cleaning / transformation advice based on EDA>",
  "executive_summary": "<concise, decision-ready summary with model and business impact>",
  "data_quality_findings": ["<finding 1>", "<finding 2>"],
  "key_patterns": ["<pattern backed by chart evidence>", "<pattern backed by chart evidence>"],
  "risk_flags": ["<risk 1>", "<risk 2>"],
  "recommendations": ["<prioritized action 1>", "<prioritized action 2>"],
  "chart_narrative": "<how the charts together support the final conclusion>"
}}
"""


# ── Agent class ───────────────────────────────────────────────────────────────

class MLAgent:
    """OpenAI / Anthropic chat completions wrapper for every AutoML agent role."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini", provider: str = "openai"):
        self.model    = model
        self.provider = provider
        self.cache_enabled = os.getenv("AUTOML_AGENT_CACHE", "1").strip() != "0"
        self._cache_ttl_seconds = int(os.getenv("AUTOML_AGENT_CACHE_TTL", "900"))
        self._cache: dict[str, tuple[float, dict]] = {}
        self.client: Any
        if provider == "anthropic":
            from anthropic import Anthropic
            self.client = Anthropic(api_key=api_key)
        else:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)

    # ── private helpers ───────────────────────────────────────────────────────

    def _cache_key(self, scope: str, payload: str) -> str:
        digest = hashlib.sha256(payload.encode("utf-8", errors="ignore")).hexdigest()
        return f"{scope}:{self.provider}:{self.model}:{digest}"

    def _cache_get(self, key: str) -> Optional[dict]:
        if not self.cache_enabled:
            return None
        item = self._cache.get(key)
        if not item:
            return None
        ts, data = item
        if (ts + self._cache_ttl_seconds) < time.time():
            self._cache.pop(key, None)
            return None
        return data

    def _cache_set(self, key: str, value: dict) -> None:
        if not self.cache_enabled:
            return
        self._cache[key] = (time.time(), value)

    def _complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 3000,
        json_mode: bool = False,
    ) -> str:
        if self.provider == "anthropic":
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return resp.content[0].text if resp.content else ""
        else:
            params: dict[str, Any] = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                "max_tokens": max_tokens,
                "temperature": 0.15,
            }
            if json_mode:
                params["response_format"] = {"type": "json_object"}
            try:
                resp = self.client.chat.completions.create(**params)
            except Exception as exc:
                # Some models/providers reject response_format=json_object.
                if json_mode and "response_format" in str(exc).lower():
                    params.pop("response_format", None)
                    resp = self.client.chat.completions.create(**params)
                else:
                    raise
            return resp.choices[0].message.content or ""

    def _ctx_str(self, context: dict) -> str:
        models = context.get("models") or []
        model_brief: list[dict[str, Any]] = []
        if isinstance(models, list):
            for m in models[:3]:
                if isinstance(m, dict):
                    model_brief.append({
                        "rank": m.get("rank"),
                        "name": m.get("name") or m.get("display_name"),
                        "reasoning": (m.get("reasoning") or "")[:220],
                        "expected_performance": (m.get("expected_performance") or "")[:120],
                    })

        compact = {
            "filename": context.get("filename"),
            "file_path": context.get("file_path"),
            "problem_type": context.get("problem_type"),
            "target_column": context.get("target_column"),
            "stage": context.get("stage"),
            "dataset_summary": context.get("dataset_summary"),
            "feature_analysis": (context.get("feature_analysis") or "")[:1200],
            "missing_values_note": (context.get("missing_values_note") or "")[:500],
            "feature_engineering_reasoning": (context.get("feature_engineering_reasoning") or "")[:700],
            "eda_summary": (context.get("eda_summary") or "")[:700],
            "risk_flags": context.get("report_risk_flags") or context.get("risk_flags") or [],
            "top_models": model_brief,
            "selected_model": context.get("selected_model"),
        }
        return json.dumps(
            compact,
            ensure_ascii=False,
        )

    # ── 1. Dataset analysis ───────────────────────────────────────────────────

    def analyze(
        self,
        filename: str,
        profile_output: str,
        target_column: Optional[str] = None,
    ) -> dict:
        """Analyse the dataset and suggest top 3 models."""
        payload = json.dumps(
            {
                "filename": filename,
                "target_column": target_column,
                "profile": profile_output[:7000],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        ck = self._cache_key("analyze", payload)
        cached = self._cache_get(ck)
        if cached is not None:
            return cached

        user = _ANALYSIS_USER.format(
            filename=filename,
            target=target_column or "not specified – please infer from the data",
            profile=profile_output[:7000],
        )
        raw = self._complete(_ANALYSIS_SYS, user, max_tokens=3000, json_mode=True)
        result = _extract_json(raw)
        self._cache_set(ck, result)
        return result

    # ── 2. Pipeline generation ────────────────────────────────────────────────

    def build_pipeline(
        self,
        filename: str,
        problem_type: str,
        target_column: str,
        model_name: str,
        profile_output: str,
        fe_code: str,
    ) -> dict:
        """Generate a complete ML pipeline (cell-by-cell) for the chosen model."""
        payload = json.dumps(
            {
                "filename": filename,
                "problem_type": problem_type,
                "target_column": target_column,
                "model_name": model_name,
                "profile": profile_output[:3500],
                "fe_code": (fe_code or "")[:3500],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        ck = self._cache_key("build_pipeline", payload)
        cached = self._cache_get(ck)
        if cached is not None:
            return cached

        user = _PIPELINE_USER.format(
            filename=filename,
            problem_type=problem_type,
            target=target_column,
            model_name=model_name,
            profile=profile_output[:3500],
            fe_code=fe_code or "# (no feature engineering applied)",
        )
        raw = self._complete(_PIPELINE_SYS, user, max_tokens=4096, json_mode=True)
        result = _extract_json(raw)
        self._cache_set(ck, result)
        return result

    # ── 3. Exploratory Data Analysis ─────────────────────────────────────────

    def run_eda(
        self,
        filename: str,
        problem_type: str,
        target_column: str,
        profile_output: str,
        feature_analysis: str = "",
    ) -> dict:
        """Generate EDA code cells and a summary.
      Returns detailed report data including:
      {cells, summary, feature_importance_notes, preprocessing_recommendations,
       executive_summary, data_quality_findings, key_patterns, risk_flags,
       recommendations, chart_narrative}
        """
        payload = json.dumps(
            {
                "filename": filename,
                "problem_type": problem_type,
                "target_column": target_column,
                "profile": profile_output[:5000],
                "feature_analysis": feature_analysis[:2000],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        ck = self._cache_key("run_eda", payload)
        cached = self._cache_get(ck)
        if cached is not None:
            return cached

        user = _EDA_USER.format(
            filename=filename,
            problem_type=problem_type,
            target=target_column,
            profile=profile_output[:5000],
            feature_analysis=feature_analysis[:2000],
        )
        raw = self._complete(_EDA_SYS, user, max_tokens=2200, json_mode=True)
        result = _extract_json(raw)
        self._cache_set(ck, result)
        return result

    # ── 4. Chat with code-detection ───────────────────────────────────────────

    def chat_with_code(
        self,
        message: str,
        context: dict,
        history: list[dict],
    ) -> dict:
        """
        Decide whether the user question needs an explanation or code execution.
        Returns: {"action": "explain"|"execute", "reply": str,
                  "code"?: str, "title"?: str, "is_chart"?: bool}
        """
        ctx = self._ctx_str(context)
        system = (
            _CHAT_DECISION_SYS
            + f"\n\nCurrent workflow context:\n{ctx}"
        )
        messages: list[dict] = [{"role": "system", "content": system}]
        for m in history[-10:]:
            messages.append({"role": m["role"], "content": m["content"]})
        messages.append({"role": "user", "content": message})

        if self.provider == "anthropic":
            # Extract system message, keep only user/assistant turns
            sys_content = next(
                (m["content"] for m in messages if m["role"] == "system"), system
            )
            user_messages = [m for m in messages if m["role"] != "system"]
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=700,
                system=sys_content,
                messages=user_messages,
            )
            raw = resp.content[0].text if resp.content else ""
        else:
          params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 700,
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
          }
          try:
            resp = self.client.chat.completions.create(**params)
          except Exception as exc:
            if "response_format" in str(exc).lower():
              params.pop("response_format", None)
              resp = self.client.chat.completions.create(**params)
            else:
              raise
          raw = resp.choices[0].message.content or ""
        try:
            return _extract_json(raw)
        except ValueError:
            # Fallback: treat as plain explanation
            return {"action": "explain", "reply": raw, "is_chart": False}

    # ── 5. Chart interpretation ───────────────────────────────────────────────

    def explain_chart(
        self,
        chart_code: str,
        context: dict,
    ) -> str:
        """
        Given the Python code that produced a seaborn/matplotlib chart,
        return a plain-English insight about what the chart reveals.
        """
        ctx = self._ctx_str(context)
        user = (
            f"ML context: {ctx}\n\n"
            f"Python code that generated the chart:\n```python\n{chart_code[:2000]}\n```\n\n"
            "Explain what this chart reveals about the data or model performance."
        )
        return self._complete(_CHART_EXPLAIN_SYS, user, max_tokens=300)

    # ── 6. Prediction code generation ─────────────────────────────────────────

    def generate_predict(
        self,
        feature_columns: list[str],
        input_values: dict,
        problem_type: str,
        target_column: str,
    ) -> str:
        """Generate Python code to make a prediction using the trained model."""
        user = (
            f"Feature columns: {feature_columns}\n"
            f"User input values: {json.dumps(input_values)}\n"
            f"Problem type: {problem_type}\n"
            f"Target column: {target_column}\n\n"
            "Write Python code that creates a prediction using the in-scope variable 'model'. "
            "Print the result clearly with labels."
        )
        return self._complete(_PREDICT_CODE_SYS, user, max_tokens=450)

    # ── 7. Guard: error analysis & fix ────────────────────────────────────────

    def fix_error(
        self,
        failed_code: str,
        error: str,
        context: dict,
        search_results: str = "",
    ) -> dict:
        """
        Guard agent: analyse a cell error and return corrected code.
        Returns: {fixed_code, explanation, needs_search, search_query}
        """
        ctx = self._ctx_str(context)
        user = _FIX_USER.format(
            code=failed_code[:4000],
            error=error[:2000],
            context=ctx,
            search_results=search_results[:2000] if search_results else "(none — LLM knowledge only)",
        )
        raw = self._complete(_FIX_SYS, user, max_tokens=2500, json_mode=True)
        try:
            return _extract_json(raw)
        except ValueError:
            return {
                "fixed_code": "",
                "explanation": raw[:300],
                "needs_search": False,
                "search_query": "",
            }

    # ── 8. Web search via Tavily (optional) ───────────────────────────────────

    def modify_script(self, message: str, script: str, state: dict) -> dict:
        """
        AI chat for the script editor.
        Returns {"reply": str, "updated_script": str | None}.
        If the user asks for a code change, updated_script contains the full
        rewritten script; otherwise it is None (explanation-only response).
        """
        sys_prompt = (
            "You are a senior Python/ML engineer helping the user improve a machine-learning script.\n"
            "When the user asks you to modify, fix, add, or rewrite code, return the COMPLETE updated script.\n"
            "When the user asks a question, explain clearly without rewriting the script unless necessary.\n"
            "Respond ONLY with valid JSON — no prose outside the JSON block.\n"
            '{\n'
            '  "reply": "<plain-English explanation or summary of changes>",\n'
            '  "updated_script": "<full updated Python script, or null if no code changes>"\n'
            '}'
        )
        context_snippet = (
            f"Dataset: {state.get('filename', 'unknown')}\n"
            f"Problem type: {state.get('problem_type', 'unknown')}\n"
            f"Target column: {state.get('target_column', 'unknown')}\n"
        )
        user_prompt = (
            f"Context:\n{context_snippet}\n"
            f"Current script:\n```python\n{script[:6000]}\n```\n\n"
            f"User request: {message}"
        )
        raw = self._complete(sys_prompt, user_prompt, json_mode=True)
        try:
            result = _extract_json(raw)
        except Exception:
            result = {"reply": raw, "updated_script": None}
        return result

    def web_search(self, query: str) -> str:
        """
        Search the web using Tavily.
        Returns an empty string if TAVILY_API_KEY is not set or tavily-python is
        not installed — the guard degrades gracefully to LLM-only fixes.
        """
        try:
            from tavily import TavilyClient  # type: ignore[import]
            key = os.getenv("TAVILY_API_KEY", "").strip()
            if not key:
                return ""
            client = TavilyClient(api_key=key)
            results = client.search(query, max_results=3, search_depth="basic")
            snippets: list[str] = []
            for r in results.get("results", []):
                title   = r.get("title", "")
                content = r.get("content", "")[:600]
                snippets.append(f"[{title}]\n{content}")
            return "\n\n---\n\n".join(snippets)
        except Exception:
            return ""
