import json
import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import numpy as np
try:
    from scipy import stats
    _SCIPY_AVAILABLE = True
except Exception:
    stats = None
    _SCIPY_AVAILABLE = False
from langchain.tools import tool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _read_csv_safe(file_path: str) -> pd.DataFrame:
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(p)


def _to_json(o: Any) -> str:
    return json.dumps(o, default=str)


@tool
def dataset_shape(file_path: str) -> str:
    """Return dataset shape (rows, columns) as JSON."""
    df = _read_csv_safe(file_path)
    return _to_json({"rows": int(df.shape[0]), "columns": int(df.shape[1])})


@tool
def column_names(file_path: str) -> str:
    """Return column names as JSON list."""
    df = _read_csv_safe(file_path)
    return _to_json(list(df.columns))


@tool
def dataset_info(file_path: str) -> str:
    """Return data types and non-null counts as JSON."""
    df = _read_csv_safe(file_path)
    info = {"dtypes": df.dtypes.astype(str).to_dict(), "non_null": df.count().to_dict()}
    return _to_json(info)


@tool
def summary_statistics(file_path: str) -> str:
    """Return describe() for numeric and categorical columns as JSON."""
    df = _read_csv_safe(file_path)
    numeric = df.describe().to_dict()
    categorical = df.describe(include=[object]).to_dict()
    return _to_json({"numeric": numeric, "categorical": categorical})


@tool
def statistical_measures(file_path: str) -> str:
    """Return mean, median, mode, std, variance, range, IQR for numeric cols as JSON."""
    df = _read_csv_safe(file_path)
    numeric = df.select_dtypes(include=np.number)

    measures: Dict[str, Dict[str, Any]] = {}
    for col in numeric.columns:
        col_series = numeric[col].dropna()
        if col_series.empty:
            measures[col] = {}
            continue
        mode_val = None
        m = col_series.mode()
        if not m.empty:
            mode_val = m.iloc[0]

        measures[col] = {
            "mean": float(col_series.mean()),
            "median": float(col_series.median()),
            "mode": mode_val if mode_val is None else (mode_val.item() if hasattr(mode_val, "item") else mode_val),
            "std": float(col_series.std()),
            "variance": float(col_series.var()),
            "range": float(col_series.max() - col_series.min()),
            "iqr": float(col_series.quantile(0.75) - col_series.quantile(0.25)),
            "min": float(col_series.min()),
            "max": float(col_series.max()),
        }
    return _to_json(measures)


@tool
def distribution_analysis(file_path: str) -> str:
    """Check skewness and (approx) normality using Shapiro test; downsample large series for Shapiro."""
    df = _read_csv_safe(file_path)
    numeric = df.select_dtypes(include=np.number)

    dist: Dict[str, Dict[str, Any]] = {}
    for col in numeric.columns:
        col_series = numeric[col].dropna()
        if col_series.empty:
            dist[col] = {"skewness": None, "shapiro_p": None, "normal_distribution": None}
            continue
        skewness = None
        stat = None
        p_value = None
        normal = None

        try:
            skewness = float(stats.skew(col_series)) if _SCIPY_AVAILABLE else None
        except Exception:
            skewness = None

        # Shapiro can fail for very large samples; downsample deterministically if needed
        if _SCIPY_AVAILABLE:
            try:
                sample = col_series
                if sample.shape[0] > 5000:
                    sample = sample.sample(5000, random_state=0)
                stat, p_value = stats.shapiro(sample)
                normal = bool(p_value > 0.05)
            except Exception as e:
                logger.warning("Shapiro test failed for %s: %s", col, e)
                stat, p_value, normal = None, None, None
        else:
            logger.info("scipy is not available; skipping Shapiro test for %s", col)

        dist[col] = {"skewness": skewness, "shapiro_stat": stat, "shapiro_p": p_value, "normal_distribution": normal}
    return _to_json(dist)


@tool
def unique_values(file_path: str) -> str:
    """Return number of unique values per column as JSON."""
    df = _read_csv_safe(file_path)
    return _to_json(df.nunique().to_dict())


@tool
def handle_missing_values(file_path: str) -> str:
    """Report missing values and (optionally) fill them with 0. This tool reports actions only."""
    df = _read_csv_safe(file_path)
    missing_before = df.isnull().sum().to_dict()
    # We don't overwrite the file here; agent may instruct how to handle missing values.
    return _to_json({"missing_before": missing_before, "suggested_action": "fillna(0)"})


@tool
def drop_duplicates(file_path: str) -> str:
    """Report number of duplicate rows; does not mutate file."""
    df = _read_csv_safe(file_path)
    before = int(df.shape[0])
    after = int(df.drop_duplicates().shape[0])
    return _to_json({"duplicates_removed": before - after, "rows_before": before, "rows_after": after})


@tool
def data_distribution(file_path: str) -> str:
    """Return distribution description for numeric columns as JSON."""
    df = _read_csv_safe(file_path)
    numeric = df.select_dtypes(include=np.number)
    return _to_json(numeric.describe(percentiles=[0.25, 0.5, 0.75]).to_dict())


@tool
def correlation_matrix(file_path: str) -> str:
    """Return correlation matrix as JSON."""
    df = _read_csv_safe(file_path)
    corr = df.select_dtypes(include=np.number).corr()
    return _to_json(corr.to_dict())


import argparse
import os
import sys


def create_agent(temperature: float = 0.0, verbose: bool = True):
    """Create and return a configured EDA agent (tools are the functions above).

    Imports from `langchain` are done lazily so the module can be imported without
    a hard dependency during static checks. If the required langchain symbols are
    missing, a helpful ImportError is raised at runtime when attempting to create
    the agent.
    """
    # Prefer the newer `create_agent` API when available (langchain>=1.2+)
    try:
        from langchain.agents import create_agent as lc_create_agent  # type: ignore
        # Use a model spec string so we don't depend on langchain.llms import path
        model_spec = "openai:gpt-3.5-turbo"

        tools = [
            dataset_shape,
            column_names,
            dataset_info,
            summary_statistics,
            statistical_measures,
            distribution_analysis,
            unique_values,
            handle_missing_values,
            drop_duplicates,
            data_distribution,
            correlation_matrix,
        ]

        compiled = lc_create_agent(model=model_spec, tools=tools, debug=verbose)

        class _AgentWrapper:
            def __init__(self, graph):
                self._graph = graph

            def run(self, prompt: str):
                # Many compiled agent graphs expect a `messages` list shaped like chat messages.
                try:
                    out = self._graph.invoke({"messages": [{"type": "human", "content": prompt}]})
                except Exception:
                    out = self._graph.invoke({"input": prompt})
                if isinstance(out, dict):
                    for key in ("output", "result", "response", "text"):
                        if key in out and isinstance(out[key], str):
                            return out[key]
                    for v in out.values():
                        if isinstance(v, str):
                            return v
                    return json.dumps(out, default=str)
                return str(out)

        return _AgentWrapper(compiled)
    except Exception:
        # Fallback to the older initialize_agent API
        try:
            from langchain.agents import initialize_agent  # type: ignore
            from langchain.llms import OpenAI  # type: ignore
        except Exception as e:
            raise ImportError(
                "Required langchain symbols not available: ensure `langchain` and the "
                "OpenAI integration are installed and up-to-date."
            ) from e

        llm = OpenAI(temperature=temperature)
        tools = [
            dataset_shape,
            column_names,
            dataset_info,
            summary_statistics,
            statistical_measures,
            distribution_analysis,
            unique_values,
            handle_missing_values,
            drop_duplicates,
            data_distribution,
            correlation_matrix,
        ]
        agent = initialize_agent(tools=tools, llm=llm, agent="zero-shot-react-description", verbose=verbose)
        return agent


def main():
    parser = argparse.ArgumentParser(description="Run EDA agent on a CSV file.")
    parser.add_argument("file", help="Path to CSV file to analyze")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true", help="Run tools locally without contacting the LLM/provider")
    args = parser.parse_args()

    # Basic environment check
    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY not set; the agent will not be able to call OpenAI.")

    file_path = args.file

    if args.dry_run:
        # Execute tools locally and print their JSON outputs. Useful for testing without LLM.
        local_tools = [
            dataset_shape,
            column_names,
            dataset_info,
            summary_statistics,
            statistical_measures,
            distribution_analysis,
            unique_values,
            handle_missing_values,
            drop_duplicates,
            data_distribution,
            correlation_matrix,
        ]
        for fn in local_tools:
            # Friendly name for the tool (StructuredTool may not expose __name__)
            name = getattr(fn, "name", None) or getattr(fn, "__name__", None) or repr(fn)
            print(f"--- {name} ---")
            try:
                # Try calling the tool in several ways to support plain functions and StructuredTool wrappers
                result = None
                if callable(fn):
                    try:
                        result = fn(file_path)
                    except TypeError:
                        try:
                            result = fn(file_path=file_path)
                        except Exception:
                            result = fn({"file_path": file_path})
                elif hasattr(fn, "func"):
                    try:
                        result = fn.func(file_path)
                    except TypeError:
                        result = fn.func(file_path=file_path)
                elif hasattr(fn, "run"):
                    result = fn.run(file_path)
                else:
                    result = fn

                print(result)
            except Exception as e:
                logger.exception("Tool %s failed", name)
                print(name, "error:", e)
        return

    try:
        agent = create_agent(temperature=args.temperature, verbose=True)
    except ImportError as e:
        logger.error("Failed to create agent: %s", e)
        print("Failed to create agent:", e)
        sys.exit(1)

    prompt = f"Perform complete exploratory data analysis on file {file_path}"

    logger.info("Running EDA agent on %s", file_path)
    try:
        result = agent.run(prompt)
        print(result)
    except Exception as e:
        logger.exception("Agent run failed")
        print("Agent run failed:", e)
        print("Check your OPENAI_API_KEY, model access, and that the CSV path is valid.")
        sys.exit(1)


if __name__ == "__main__":
    main()
