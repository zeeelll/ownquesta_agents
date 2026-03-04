"""
agents.py – OpenAI-powered ML agents for Lab Playground

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
from typing import Optional


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
  "feature_engineering_code": "<complete Python — 'df' is already in scope; use pandas/sklearn; end with print('Feature engineering done. Shape:', df_processed.shape) or similar>",
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
Always use seaborn with dark theme for any visualisation:
  import seaborn as sns, matplotlib.pyplot as plt
  sns.set_theme(style='darkgrid', palette='muted')
  plt.style.use('dark_background')
  plt.rcParams.update({{'figure.facecolor':'#0d1117','axes.facecolor':'#161b22','text.color':'#e6edf3','axes.labelcolor':'#8b949e','xtick.color':'#8b949e','ytick.color':'#8b949e'}})
Never call plt.show() — the backend captures figures automatically.
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
Include StratifiedKFold cross-validation for classification (KFold for regression) if dataset has < 5000 rows.
Save the trained model to a variable called `model`. List the exact feature column names used for X.
Add print statements so each cell has visible output. Do NOT call plt.show().

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
  - "explain" : give a concise text answer (< 150 words, no code block needed)
  - "execute"  : write Python code that should run in the notebook (plots, computations, new analyses)

Rules:
• Questions about WHY / WHAT / HOW something works → "explain"
• Requests to plot, compute, explore, or show new results → "execute"
• Always use seaborn dark-theme for plots; never call plt.show()
• The executed code must be self-contained (use variables already in scope: df, df_processed, model, X_test, y_test, etc.)
• ALWAYS use print() to display results — never rely on bare variable names as the last line
• Wrap numeric results, dataframes, and computation outputs in print()
• If no useful printable output can be produced, use action="explain" instead

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
Given failing code and its error traceback, identify the root cause and return corrected code.

Known API changes to fix automatically:
  - sklearn >= 1.2: OneHotEncoder(sparse=False) → OneHotEncoder(sparse_output=False)
  - sklearn >= 1.2: many transformers renamed 'sparse' kwarg → 'sparse_output'
  - scipy/sklearn: sparse matrices may need .toarray() when dense array expected
  - pandas >= 2.0: DataFrame.append() removed → use pd.concat()
  - pandas >= 2.0: fillna() with numeric value on object column needs explicit cast
  - matplotlib: plt.show() must never be called (backend captures automatically)

Rules:
  • Fix ONLY the broken part — keep everything else identical
  • The fixed code must be complete and self-contained (all imports included)
  • Notebook globals already in scope: df, df_processed, model, X_train, X_test, y_train, y_test, etc.
  • If the error is truly obscure and you need up-to-date web information, set needs_search=true

Respond ONLY with valid JSON — no prose before or after:
{{
  "fixed_code": "<complete corrected Python>",
  "explanation": "<1–2 sentences: root cause and what was changed>",
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
  • Always print() a summary of what the chart/analysis reveals
  • Be self-contained (all imports at the top of each cell)

Focus on:
  1. Distribution of the target variable
  2. Correlation heatmap for numeric columns
  3. Top feature distributions or box-plots by target
  4. Missing-value heatmap (if relevant)

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
  "preprocessing_recommendations": "<data cleaning / transformation advice based on EDA>"
}}
"""


# ── Agent class ───────────────────────────────────────────────────────────────

class MLAgent:
    """OpenAI chat completions wrapper for every lab-agent role."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model

    # ── private helpers ───────────────────────────────────────────────────────

    def _complete(self, system: str, user: str, max_tokens: int = 3000) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            max_tokens=max_tokens,
            temperature=0.2,
        )
        return resp.choices[0].message.content or ""

    def _ctx_str(self, context: dict) -> str:
        return json.dumps(
            {k: context.get(k) for k in
             ["filename", "problem_type", "target_column", "stage", "dataset_summary"]},
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
        user = _ANALYSIS_USER.format(
            filename=filename,
            target=target_column or "not specified – please infer from the data",
            profile=profile_output[:7000],
        )
        raw = self._complete(_ANALYSIS_SYS, user, max_tokens=2800)
        return _extract_json(raw)

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
        user = _PIPELINE_USER.format(
            filename=filename,
            problem_type=problem_type,
            target=target_column,
            model_name=model_name,
            profile=profile_output[:3500],
            fe_code=fe_code or "# (no feature engineering applied)",
        )
        raw = self._complete(_PIPELINE_SYS, user, max_tokens=3500)
        return _extract_json(raw)

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
        Returns: {cells: [{title, code}], summary, feature_importance_notes, preprocessing_recommendations}
        """
        user = _EDA_USER.format(
            filename=filename,
            problem_type=problem_type,
            target=target_column,
            profile=profile_output[:5000],
            feature_analysis=feature_analysis[:2000],
        )
        raw = self._complete(_EDA_SYS, user, max_tokens=3000)
        return _extract_json(raw)

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

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=900,
            temperature=0.3,
        )
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
        return self._complete(_PREDICT_CODE_SYS, user, max_tokens=500)

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
        raw = self._complete(_FIX_SYS, user, max_tokens=2500)
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
