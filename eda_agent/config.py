import os
import logging
import json
import warnings
from dotenv import load_dotenv
from langchain.tools import tool

# Setup logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

load_dotenv()

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import numpy as np
from collections import Counter
import math

# Try to import scipy
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
    logger.info("scipy is available")
except ImportError:
    logger.warning("scipy is not available. Installing scipy...")
    import subprocess
    import sys
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy"])
        from scipy import stats
        SCIPY_AVAILABLE = True
        logger.info("scipy installed successfully")
    except Exception as e:
        logger.warning(f"Could not install scipy: {e}. Some statistical tests will be skipped.")
        SCIPY_AVAILABLE = False

_SCIPY_AVAILABLE = SCIPY_AVAILABLE  # Make it accessible to functions

def _read_csv_safe(file_path: str) -> pd.DataFrame:
    """Safely read CSV with proper encoding handling."""
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Try different encodings
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    for encoding in encodings:
        try:
            return pd.read_csv(p, encoding=encoding)
        except UnicodeDecodeError:
            continue
    
    # If all encodings fail, use default with error handling
    return pd.read_csv(p, encoding='utf-8', errors='replace')


def _to_json(o: Any) -> str:
    """Convert object to JSON with proper handling of special values."""
    def serialize_special(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        elif obj == np.inf:
            return "infinity"
        elif obj == -np.inf:
            return "-infinity"
        return str(obj)
    
    return json.dumps(o, default=serialize_special, ensure_ascii=False, indent=2)


def _calculate_mode(series: pd.Series) -> Any:
    """Calculate mode with proper handling of edge cases."""
    try:
        if series.empty:
            return None
        mode_values = series.mode()
        if mode_values.empty:
            return None
        return mode_values.iloc[0] if hasattr(mode_values.iloc[0], 'item') else mode_values.iloc[0]
    except Exception:
        return None


def _detect_outliers(series: pd.Series, method: str = 'iqr') -> Dict[str, Any]:
    """Detect outliers using IQR or Z-score methods."""
    if series.empty or not np.issubdtype(series.dtype, np.number):
        return {"outliers": [], "outlier_count": 0, "method": method}
    
    clean_series = series.dropna()
    if len(clean_series) == 0:
        return {"outliers": [], "outlier_count": 0, "method": method}
    
    if method == 'iqr':
        Q1 = clean_series.quantile(0.25)
        Q3 = clean_series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = clean_series[(clean_series < lower_bound) | (clean_series > upper_bound)]
    else:  # z-score method
        z_scores = np.abs((clean_series - clean_series.mean()) / clean_series.std())
        outliers = clean_series[z_scores > 3]
    
    return {
        "outliers": outliers.tolist()[:20],  # Limit to first 20 outliers
        "outlier_count": len(outliers),
        "percentage": round((len(outliers) / len(clean_series)) * 100, 2),
        "method": method
    }


@tool
def dataset_overview(file_path: str) -> str:
    """Return comprehensive dataset overview including size, shape, memory usage, and basic information."""
    df = _read_csv_safe(file_path)
    
    # Calculate memory usage
    memory_usage = df.memory_usage(deep=True).sum()
    memory_mb = memory_usage / (1024 * 1024)
    
    # Get file size
    file_size = Path(file_path).stat().st_size / (1024 * 1024)  # MB
    
    overview = {
        "shape": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1])
        },
        "size": {
            "memory_usage_mb": round(memory_mb, 2),
            "file_size_mb": round(file_size, 2),
            "total_cells": int(df.shape[0] * df.shape[1])
        },
        "data_quality": {
            "missing_values_count": int(df.isnull().sum().sum()),
            "missing_percentage": round((df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100, 2),
            "duplicate_rows": int(df.duplicated().sum()),
            "duplicate_percentage": round((df.duplicated().sum() / df.shape[0]) * 100, 2)
        },
        "column_types": {
            "numeric": len(df.select_dtypes(include=[np.number]).columns),
            "categorical": len(df.select_dtypes(include=['object', 'category']).columns),
            "datetime": len(df.select_dtypes(include=['datetime']).columns),
            "boolean": len(df.select_dtypes(include=['bool']).columns)
        }
    }
    
    return _to_json(overview)


@tool
def dataset_shape(file_path: str) -> str:
    """Return dataset shape (rows, columns) as JSON."""
    df = _read_csv_safe(file_path)
    return _to_json({"rows": int(df.shape[0]), "columns": int(df.shape[1])})


@tool
def column_analysis(file_path: str) -> str:
    """Return detailed column analysis including names, types, and basic statistics."""
    df = _read_csv_safe(file_path)
    
    column_info = {}
    for col in df.columns:
        series = df[col]
        col_info = {
            "name": col,
            "dtype": str(series.dtype),
            "non_null_count": int(series.count()),
            "null_count": int(series.isnull().sum()),
            "null_percentage": round((series.isnull().sum() / len(series)) * 100, 2),
            "unique_count": int(series.nunique()),
            "unique_percentage": round((series.nunique() / len(series)) * 100, 2)
        }
        
        # Add type-specific information
        if np.issubdtype(series.dtype, np.number):
            clean_series = series.dropna()
            if len(clean_series) > 0:
                col_info.update({
                    "min": float(clean_series.min()),
                    "max": float(clean_series.max()),
                    "mean": float(clean_series.mean()),
                    "median": float(clean_series.median()),
                    "std": float(clean_series.std()),
                    "zeros_count": int((clean_series == 0).sum()),
                    "negative_count": int((clean_series < 0).sum()),
                    "positive_count": int((clean_series > 0).sum())
                })
        elif series.dtype == 'object':
            col_info.update({
                "max_length": int(series.astype(str).str.len().max()) if not series.empty else 0,
                "min_length": int(series.astype(str).str.len().min()) if not series.empty else 0,
                "avg_length": round(series.astype(str).str.len().mean(), 2) if not series.empty else 0,
                "most_common": series.value_counts().head(3).to_dict() if not series.empty else {}
            })
        
        column_info[col] = col_info
    
    return _to_json(column_info)


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
def comprehensive_statistics(file_path: str) -> str:
    """Return comprehensive statistical measures for numeric and categorical columns."""
    df = _read_csv_safe(file_path)
    
    result = {
        "numeric_statistics": {},
        "categorical_statistics": {},
        "summary": {}
    }
    
    # Numeric statistics
    numeric_cols = df.select_dtypes(include=np.number)
    for col in numeric_cols.columns:
        col_series = numeric_cols[col].dropna()
        if col_series.empty:
            continue
            
        # Basic statistics
        stats_dict = {
            "count": int(len(col_series)),
            "mean": float(col_series.mean()),
            "median": float(col_series.median()),
            "mode": _calculate_mode(col_series),
            "std": float(col_series.std()),
            "variance": float(col_series.var()),
            "min": float(col_series.min()),
            "max": float(col_series.max()),
            "range": float(col_series.max() - col_series.min()),
            "q1": float(col_series.quantile(0.25)),
            "q3": float(col_series.quantile(0.75)),
            "iqr": float(col_series.quantile(0.75) - col_series.quantile(0.25)),
        }
        
        # Additional measures
        try:
            stats_dict.update({
                "skewness": float(stats.skew(col_series)) if _SCIPY_AVAILABLE else None,
                "kurtosis": float(stats.kurtosis(col_series)) if _SCIPY_AVAILABLE else None,
                "coefficient_of_variation": float(col_series.std() / col_series.mean()) if col_series.mean() != 0 else None,
            })
        except Exception as e:
            logger.warning(f"Error calculating advanced statistics for {col}: {e}")
        
        # Percentiles
        percentiles = [5, 10, 25, 50, 75, 90, 95, 99]
        stats_dict["percentiles"] = {f"p{p}": float(col_series.quantile(p/100)) for p in percentiles}
        
        # Outlier detection
        stats_dict["outliers"] = _detect_outliers(col_series)
        
        result["numeric_statistics"][col] = stats_dict
    
    # Categorical statistics
    categorical_cols = df.select_dtypes(include=['object', 'category'])
    for col in categorical_cols.columns:
        col_series = categorical_cols[col].dropna()
        if col_series.empty:
            continue
            
        value_counts = col_series.value_counts()
        stats_dict = {
            "count": int(len(col_series)),
            "unique_count": int(col_series.nunique()),
            "top_value": str(value_counts.index[0]) if len(value_counts) > 0 else None,
            "top_frequency": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
            "mode": _calculate_mode(col_series),
            "entropy": float(-np.sum((value_counts / len(col_series)) * np.log2(value_counts / len(col_series) + 1e-10))),
            "top_10_values": dict(value_counts.head(10)),
            "value_distribution": {
                "rare_values": int(sum(value_counts == 1)),  # Values appearing only once
                "common_values": int(sum(value_counts >= len(col_series) * 0.05))  # Values appearing in >5% of records
            }
        }
        
        result["categorical_statistics"][col] = stats_dict
    
    # Summary statistics
    result["summary"] = {
        "total_numeric_columns": len(result["numeric_statistics"]),
        "total_categorical_columns": len(result["categorical_statistics"]),
        "most_variable_numeric": max(result["numeric_statistics"].keys(), 
                                     key=lambda x: result["numeric_statistics"][x].get("coefficient_of_variation", 0)) 
                                if result["numeric_statistics"] else None,
        "most_diverse_categorical": max(result["categorical_statistics"].keys(),
                                       key=lambda x: result["categorical_statistics"][x].get("unique_count", 0)) 
                                  if result["categorical_statistics"] else None
    }
    
    return _to_json(result)


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
        mode_val = _calculate_mode(col_series)

        measures[col] = {
            "mean": float(col_series.mean()),
            "median": float(col_series.median()),
            "mode": mode_val,
            "std": float(col_series.std()),
            "variance": float(col_series.var()),
            "range": float(col_series.max() - col_series.min()),
            "iqr": float(col_series.quantile(0.75) - col_series.quantile(0.25)),
            "min": float(col_series.min()),
            "max": float(col_series.max()),
        }
    return _to_json(measures)


@tool
def advanced_distribution_analysis(file_path: str) -> str:
    """Comprehensive distribution analysis including normality tests, skewness, kurtosis, and distribution identification."""
    df = _read_csv_safe(file_path)
    numeric = df.select_dtypes(include=np.number)

    dist: Dict[str, Dict[str, Any]] = {}
    
    for col in numeric.columns:
        col_series = numeric[col].dropna()
        if col_series.empty:
            dist[col] = {"error": "No valid data"}
            continue
            
        analysis = {
            "sample_size": int(len(col_series)),
            "distribution_shape": {},
            "normality_tests": {},
            "distribution_properties": {},
            "chart_data": {}
        }
        
        # Basic distribution shape measures
        try:
            if _SCIPY_AVAILABLE:
                analysis["distribution_shape"] = {
                    "skewness": float(stats.skew(col_series)),
                    "kurtosis": float(stats.kurtosis(col_series)),
                    "skewness_interpretation": _interpret_skewness(stats.skew(col_series)),
                    "kurtosis_interpretation": _interpret_kurtosis(stats.kurtosis(col_series))
                }
            else:
                # Manual calculation fallback
                mean = col_series.mean()
                std = col_series.std()
                if std > 0:
                    skew = np.mean(((col_series - mean) / std) ** 3)
                    kurt = np.mean(((col_series - mean) / std) ** 4) - 3
                    analysis["distribution_shape"] = {
                        "skewness": float(skew),
                        "kurtosis": float(kurt),
                        "skewness_interpretation": _interpret_skewness(skew),
                        "kurtosis_interpretation": _interpret_kurtosis(kurt)
                    }
        except Exception as e:
            logger.warning(f"Error calculating distribution shape for {col}: {e}")
        
        # Normality tests
        if _SCIPY_AVAILABLE and len(col_series) >= 3:
            try:
                # Sample for large datasets
                sample = col_series.sample(min(5000, len(col_series)), random_state=42)
                
                # Shapiro-Wilk test
                if len(sample) <= 5000:
                    shapiro_stat, shapiro_p = stats.shapiro(sample)
                    analysis["normality_tests"]["shapiro_wilk"] = {
                        "statistic": float(shapiro_stat),
                        "p_value": float(shapiro_p),
                        "is_normal": bool(shapiro_p > 0.05),
                        "sample_size": len(sample)
                    }
                
                # Kolmogorov-Smirnov test against normal distribution
                normalized = (sample - sample.mean()) / sample.std()
                ks_stat, ks_p = stats.kstest(normalized, 'norm')
                analysis["normality_tests"]["kolmogorov_smirnov"] = {
                    "statistic": float(ks_stat),
                    "p_value": float(ks_p),
                    "is_normal": bool(ks_p > 0.05)
                }
                
                # D'Agostino's normality test
                if len(sample) >= 8:
                    dagostino_stat, dagostino_p = stats.normaltest(sample)
                    analysis["normality_tests"]["dagostino"] = {
                        "statistic": float(dagostino_stat),
                        "p_value": float(dagostino_p),
                        "is_normal": bool(dagostino_p > 0.05)
                    }
                
            except Exception as e:
                logger.warning(f"Error in normality tests for {col}: {e}")
        
        # Distribution properties
        analysis["distribution_properties"] = {
            "mean": float(col_series.mean()),
            "median": float(col_series.median()),
            "mode": _calculate_mode(col_series),
            "range": float(col_series.max() - col_series.min()),
            "iqr": float(col_series.quantile(0.75) - col_series.quantile(0.25)),
            "coefficient_of_variation": float(col_series.std() / col_series.mean()) if col_series.mean() != 0 else None,
            "distribution_type": _identify_distribution_type(col_series)
        }
        
        # Prepare chart data for histograms and box plots
        analysis["chart_data"] = _prepare_chart_data(col_series)
        
        # Overall normality conclusion
        normality_tests = analysis.get("normality_tests", {})
        normal_count = sum(1 for test in normality_tests.values() if test.get("is_normal", False))
        total_tests = len(normality_tests)
        
        analysis["conclusion"] = {
            "likely_normal": bool(normal_count > total_tests / 2) if total_tests > 0 else None,
            "confidence": "high" if total_tests >= 2 else "low",
            "recommendation": _recommend_transformation(analysis)
        }
        
        dist[col] = analysis
    
    return _to_json(dist)


def _interpret_skewness(skewness: float) -> str:
    """Interpret skewness value."""
    if abs(skewness) < 0.5:
        return "approximately symmetric"
    elif skewness > 0.5:
        return "moderately right skewed" if skewness < 1 else "highly right skewed"
    else:
        return "moderately left skewed" if skewness > -1 else "highly left skewed"


def _interpret_kurtosis(kurtosis: float) -> str:
    """Interpret kurtosis value."""
    if abs(kurtosis) < 0.5:
        return "normal (mesokurtic)"
    elif kurtosis > 0.5:
        return "heavy-tailed (leptokurtic)"
    else:
        return "light-tailed (platykurtic)"


def _identify_distribution_type(series: pd.Series) -> str:
    """Attempt to identify the type of distribution."""
    if len(series) < 10:
        return "insufficient data"
    
    mean = series.mean()
    median = series.median()
    std = series.std()
    
    # Check for discrete vs continuous
    unique_ratio = series.nunique() / len(series)
    
    if unique_ratio < 0.05:
        return "discrete/categorical-like"
    
    # Check for uniformity
    if _SCIPY_AVAILABLE:
        try:
            # Test uniformity with Kolmogorov-Smirnov
            normalized = (series - series.min()) / (series.max() - series.min())
            uniform_stat, uniform_p = stats.kstest(normalized, 'uniform')
            if uniform_p > 0.05:
                return "approximately uniform"
        except Exception:
            pass
    
    # Simple heuristics
    if abs(mean - median) < 0.1 * std:
        return "approximately normal"
    elif mean > median:
        return "right-skewed"
    else:
        return "left-skewed"


def _prepare_chart_data(series: pd.Series, bins: int = 30) -> Dict[str, Any]:
    """Prepare data for histogram and box plot visualization."""
    try:
        # Histogram data
        hist_counts, bin_edges = np.histogram(series, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Box plot data
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_whisker = max(series.min(), q1 - 1.5 * iqr)
        upper_whisker = min(series.max(), q3 + 1.5 * iqr)
        
        outliers = series[(series < lower_whisker) | (series > upper_whisker)]
        
        return {
            "histogram": {
                "bin_centers": bin_centers.tolist(),
                "counts": hist_counts.tolist(),
                "bin_edges": bin_edges.tolist()
            },
            "box_plot": {
                "min": float(series.min()),
                "q1": float(q1),
                "median": float(series.median()),
                "q3": float(q3),
                "max": float(series.max()),
                "lower_whisker": float(lower_whisker),
                "upper_whisker": float(upper_whisker),
                "outliers": outliers.tolist()[:50]  # Limit outliers for performance
            },
            "density_plot": {
                "x_values": np.linspace(series.min(), series.max(), 100).tolist(),
                "description": "Use for kernel density estimation"
            }
        }
    except Exception as e:
        logger.warning(f"Error preparing chart data: {e}")
        return {"error": str(e)}


def _recommend_transformation(analysis: Dict[str, Any]) -> str:
    """Recommend data transformation based on distribution analysis."""
    try:
        shape = analysis.get("distribution_shape", {})
        skewness = shape.get("skewness", 0)
        
        if abs(skewness) < 0.5:
            return "No transformation needed - data is approximately symmetric"
        elif skewness > 1:
            return "Consider log transformation or square-root transformation to reduce right skew"
        elif skewness < -1:
            return "Consider square transformation or exponential transformation to reduce left skew"
        elif skewness > 0.5:
            return "Consider square-root transformation to reduce moderate right skew"
        else:
            return "Consider reciprocal transformation to reduce moderate left skew"
    except Exception:
        return "Unable to determine transformation recommendation"


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
def advanced_correlation_analysis(file_path: str) -> str:
    """Comprehensive correlation analysis including different correlation methods and insights."""
    df = _read_csv_safe(file_path)
    numeric_df = df.select_dtypes(include=np.number)
    
    if numeric_df.empty:
        return _to_json({"error": "No numeric columns found for correlation analysis"})
    
    result = {
        "correlation_matrices": {},
        "correlation_insights": {},
        "feature_relationships": {},
        "multicollinearity_analysis": {}
    }
    
    # Calculate different types of correlations
    correlation_methods = ['pearson', 'spearman']
    if _SCIPY_AVAILABLE:
        correlation_methods.append('kendall')
    
    for method in correlation_methods:
        try:
            if method == 'kendall' and len(numeric_df) > 1000:
                # Kendall is computationally expensive for large datasets
                sample_df = numeric_df.sample(1000, random_state=42)
                corr_matrix = sample_df.corr(method=method)
            else:
                corr_matrix = numeric_df.corr(method=method)
            
            result["correlation_matrices"][method] = corr_matrix.to_dict()
        except Exception as e:
            logger.warning(f"Error calculating {method} correlation: {e}")
            result["correlation_matrices"][method] = {"error": str(e)}
    
    # Analyze correlations (using Pearson as primary)
    try:
        pearson_corr = numeric_df.corr(method='pearson')
        
        # Find strong correlations
        strong_correlations = []
        high_correlations = []
        
        for i in range(len(pearson_corr.columns)):
            for j in range(i+1, len(pearson_corr.columns)):
                col1, col2 = pearson_corr.columns[i], pearson_corr.columns[j]
                corr_val = pearson_corr.iloc[i, j]
                
                if abs(corr_val) > 0.7:
                    strong_correlations.append({
                        "variable_1": col1,
                        "variable_2": col2,
                        "correlation": round(float(corr_val), 4),
                        "strength": "strong",
                        "direction": "positive" if corr_val > 0 else "negative"
                    })
                elif abs(corr_val) > 0.5:
                    high_correlations.append({
                        "variable_1": col1,
                        "variable_2": col2,
                        "correlation": round(float(corr_val), 4),
                        "strength": "moderate",
                        "direction": "positive" if corr_val > 0 else "negative"
                    })
        
        result["correlation_insights"] = {
            "strong_correlations": strong_correlations[:10],  # Limit to top 10
            "moderate_correlations": high_correlations[:15],  # Limit to top 15
            "correlation_summary": {
                "max_positive_correlation": float(pearson_corr.where(pearson_corr != 1.0).max().max()),
                "max_negative_correlation": float(pearson_corr.where(pearson_corr != 1.0).min().min()),
                "average_correlation": float(pearson_corr.where(np.triu(np.ones(pearson_corr.shape), k=1).astype(bool)).stack().mean()),
                "number_of_strong_correlations": len(strong_correlations),
                "number_of_moderate_correlations": len(high_correlations)
            }
        }
        
        # Feature relationship analysis
        feature_analysis = {}
        for col in pearson_corr.columns:
            col_correlations = pearson_corr[col].drop(col)  # Remove self-correlation
            feature_analysis[col] = {
                "most_correlated_with": {
                    "variable": str(col_correlations.abs().idxmax()),
                    "correlation": float(col_correlations.abs().max())
                },
                "average_correlation": float(col_correlations.abs().mean()),
                "max_positive_correlation": float(col_correlations.max()),
                "max_negative_correlation": float(col_correlations.min()),
                "correlation_count_strong": int(sum(col_correlations.abs() > 0.7)),
                "correlation_count_moderate": int(sum(col_correlations.abs() > 0.5))
            }
        
        result["feature_relationships"] = feature_analysis
        
        # Multicollinearity analysis
        try:
            # Calculate VIF (Variance Inflation Factor) approximation
            vif_data = []
            for i, col in enumerate(numeric_df.columns):
                if numeric_df[col].var() > 0:  # Avoid constant columns
                    other_cols = [c for c in numeric_df.columns if c != col]
                    if len(other_cols) > 0:
                        corr_with_others = pearson_corr.loc[col, other_cols].abs().mean()
                        # Simplified VIF approximation
                        vif_approx = 1 / (1 - corr_with_others ** 2) if corr_with_others ** 2 < 0.99 else float('inf')
                        vif_data.append({
                            "variable": col,
                            "vif_approximation": float(vif_approx),
                            "multicollinearity_risk": "high" if vif_approx > 10 else "moderate" if vif_approx > 5 else "low"
                        })
            
            result["multicollinearity_analysis"] = {
                "vif_analysis": sorted(vif_data, key=lambda x: x["vif_approximation"], reverse=True)[:10],
                "recommendation": "Consider removing variables with VIF > 10 to reduce multicollinearity"
            }
        except Exception as e:
            logger.warning(f"Error in multicollinearity analysis: {e}")
            result["multicollinearity_analysis"] = {"error": str(e)}
    
    except Exception as e:
        logger.warning(f"Error in correlation insights: {e}")
        result["correlation_insights"] = {"error": str(e)}
    
    # Prepare chart data for correlation heatmap
    try:
        if 'pearson' in result["correlation_matrices"]:
            corr_matrix = pearson_corr
            result["chart_data"] = {
                "heatmap": {
                    "columns": list(corr_matrix.columns),
                    "data": [[float(corr_matrix.iloc[i, j]) for j in range(len(corr_matrix.columns))] 
                             for i in range(len(corr_matrix.index))],
                    "min_correlation": float(corr_matrix.min().min()),
                    "max_correlation": float(corr_matrix.max().max())
                }
            }
    except Exception as e:
        logger.warning(f"Error preparing correlation chart data: {e}")
    
    return _to_json(result)


@tool
def correlation_matrix(file_path: str) -> str:
    """Return correlation matrix as JSON."""
    df = _read_csv_safe(file_path)
    corr = df.select_dtypes(include=np.number).corr()
    return _to_json(corr.to_dict())


@tool
def data_quality_analysis(file_path: str) -> str:
    """Comprehensive data quality analysis including missing values, duplicates, and data consistency."""
    df = _read_csv_safe(file_path)
    
    result = {
        "missing_values": {},
        "duplicate_analysis": {},
        "data_consistency": {},
        "data_types": {},
        "recommendations": []
    }
    
    # Missing values analysis
    missing_counts = df.isnull().sum()
    missing_percentages = (missing_counts / len(df)) * 100
    
    result["missing_values"] = {
        "total_missing": int(missing_counts.sum()),
        "percentage_missing": round(float(missing_percentages.sum() / len(df.columns)), 2),
        "by_column": {
            col: {
                "count": int(missing_counts[col]),
                "percentage": round(float(missing_percentages[col]), 2)
            }
            for col in df.columns if missing_counts[col] > 0
        },
        "columns_with_missing": list(missing_counts[missing_counts > 0].index),
        "complete_columns": list(missing_counts[missing_counts == 0].index)
    }
    
    # Duplicate analysis
    total_duplicates = df.duplicated().sum()
    duplicate_subset_analysis = {}
    
    # Check duplicates in key combinations
    for col_combo_size in [1, 2, 3]:
        if col_combo_size <= len(df.columns):
            for cols in pd.DataFrame(df.columns).sample(min(5, len(df.columns))).values:
                if len(cols) >= col_combo_size:
                    subset_cols = cols[:col_combo_size].tolist()
                    subset_dups = df.duplicated(subset=subset_cols).sum()
                    if subset_dups > 0:
                        key = "_".join(subset_cols)
                        duplicate_subset_analysis[key] = {
                            "columns": subset_cols,
                            "duplicate_count": int(subset_dups),
                            "percentage": round((subset_dups / len(df)) * 100, 2)
                        }
    
    result["duplicate_analysis"] = {
        "total_duplicates": int(total_duplicates),
        "percentage_duplicates": round((total_duplicates / len(df)) * 100, 2),
        "unique_rows": int(len(df) - total_duplicates),
        "subset_duplicates": duplicate_subset_analysis
    }
    
    # Data consistency checks
    consistency_issues = []
    
    # Check for mixed data types in object columns
    for col in df.select_dtypes(include=['object']).columns:
        sample_values = df[col].dropna().head(100)
        if len(sample_values) > 0:
            # Check for mixed numeric/text
            numeric_count = sum(1 for val in sample_values if str(val).replace('.', '').replace('-', '').isdigit())
            if 0 < numeric_count < len(sample_values):
                consistency_issues.append({
                    "column": col,
                    "issue": "mixed_numeric_text",
                    "description": f"Column contains both numeric and text values ({numeric_count}/{len(sample_values)} numeric)"
                })
            
            # Check for inconsistent casing
            text_values = [str(val) for val in sample_values if not str(val).replace('.', '').replace('-', '').isdigit()]
            if len(text_values) > 1:
                lower_count = sum(1 for val in text_values if val.islower())
                upper_count = sum(1 for val in text_values if val.isupper())
                mixed_case = len(text_values) - lower_count - upper_count
                if mixed_case > 0 and (lower_count > 0 or upper_count > 0):
                    consistency_issues.append({
                        "column": col,
                        "issue": "inconsistent_casing",
                        "description": f"Column has inconsistent casing: {lower_count} lowercase, {upper_count} uppercase, {mixed_case} mixed"
                    })
    
    # Check for potential date columns that aren't parsed as dates
    import re
    for col in df.select_dtypes(include=['object']).columns:
        sample_values = df[col].dropna().head(50).astype(str)
        date_like_count = sum(1 for val in sample_values 
                             if bool(re.search(r'\d{1,4}[-/]\d{1,2}[-/]\d{1,4}|\d{1,2}[-/]\d{1,2}[-/]\d{4}', str(val))))
        if date_like_count > len(sample_values) * 0.8:  # 80% look like dates
            consistency_issues.append({
                "column": col,
                "issue": "potential_date_column",
                "description": f"Column appears to contain dates ({date_like_count}/{len(sample_values)} date-like values) but is stored as text"
            })
    
    result["data_consistency"] = {
        "issues_found": len(consistency_issues),
        "issues": consistency_issues
    }
    
    # Data types summary
    dtype_counts = df.dtypes.value_counts()
    result["data_types"] = {
        "summary": {str(dtype): int(count) for dtype, count in dtype_counts.items()},
        "details": {col: str(df[col].dtype) for col in df.columns},
        "memory_usage": {
            "total_mb": round(df.memory_usage(deep=True).sum() / (1024*1024), 2),
            "by_column_mb": {col: round(df[col].memory_usage(deep=True) / (1024*1024), 4) 
                           for col in df.columns}
        }
    }
    
    # Generate recommendations
    recommendations = []
    
    if result["missing_values"]["total_missing"] > 0:
        high_missing_cols = [col for col, info in result["missing_values"]["by_column"].items() 
                           if info["percentage"] > 50]
        if high_missing_cols:
            recommendations.append(f"Consider dropping columns with >50% missing values: {', '.join(high_missing_cols)}")
        
        medium_missing_cols = [col for col, info in result["missing_values"]["by_column"].items() 
                             if 10 < info["percentage"] <= 50]
        if medium_missing_cols:
            recommendations.append(f"Consider imputation strategies for columns with 10-50% missing: {', '.join(medium_missing_cols)}")
    
    if result["duplicate_analysis"]["total_duplicates"] > 0:
        recommendations.append(f"Remove {result['duplicate_analysis']['total_duplicates']} duplicate rows to improve data quality")
    
    for issue in consistency_issues:
        if issue["issue"] == "potential_date_column":
            recommendations.append(f"Parse column '{issue['column']}' as datetime for better analysis")
        elif issue["issue"] == "mixed_numeric_text":
            recommendations.append(f"Clean column '{issue['column']}' to separate numeric and text data")
    
    result["recommendations"] = recommendations[:10]  # Limit recommendations
    
    return _to_json(result)


@tool
def outlier_detection(file_path: str) -> str:
    """Advanced outlier detection using multiple methods for numeric columns."""
    df = _read_csv_safe(file_path)
    numeric_df = df.select_dtypes(include=np.number)
    
    if numeric_df.empty:
        return _to_json({"error": "No numeric columns found for outlier analysis"})
    
    result = {
        "outlier_analysis": {},
        "summary": {},
        "visualization_data": {}
    }
    
    for col in numeric_df.columns:
        col_series = numeric_df[col].dropna()
        if len(col_series) < 4:  # Need at least 4 points for meaningful analysis
            continue
        
        col_analysis = {
            "column": col,
            "data_points": len(col_series),
            "methods": {}
        }
        
        # Method 1: IQR Method
        iqr_outliers = _detect_outliers(col_series, method='iqr')
        col_analysis["methods"]["iqr"] = iqr_outliers
        
        # Method 2: Z-Score Method
        zscore_outliers = _detect_outliers(col_series, method='zscore')
        col_analysis["methods"]["zscore"] = zscore_outliers
        
        # Method 3: Modified Z-Score (using median)
        try:
            median = col_series.median()
            mad = np.median(np.abs(col_series - median))  # Median Absolute Deviation
            if mad > 0:
                modified_z_scores = 0.6745 * (col_series - median) / mad
                modified_outliers = col_series[np.abs(modified_z_scores) > 3.5]
                col_analysis["methods"]["modified_zscore"] = {
                    "outliers": modified_outliers.tolist()[:20],
                    "outlier_count": len(modified_outliers),
                    "percentage": round((len(modified_outliers) / len(col_series)) * 100, 2),
                    "method": "modified_zscore"
                }
        except Exception as e:
            logger.warning(f"Modified Z-score failed for {col}: {e}")
        
        # Method 4: Isolation Forest (if scipy available and enough data)
        if _SCIPY_AVAILABLE and len(col_series) >= 10:
            try:
                # Simple isolation forest approximation
                # Use interquartile range expansion as a proxy
                Q1, Q3 = col_series.quantile([0.25, 0.75])
                IQR = Q3 - Q1
                extended_lower = Q1 - 2.5 * IQR
                extended_upper = Q3 + 2.5 * IQR
                isolation_outliers = col_series[(col_series < extended_lower) | (col_series > extended_upper)]
                
                col_analysis["methods"]["isolation_approximation"] = {
                    "outliers": isolation_outliers.tolist()[:20],
                    "outlier_count": len(isolation_outliers),
                    "percentage": round((len(isolation_outliers) / len(col_series)) * 100, 2),
                    "method": "isolation_approximation"
                }
            except Exception as e:
                logger.warning(f"Isolation forest approximation failed for {col}: {e}")
        
        # Consensus outliers (outliers detected by multiple methods)
        all_outliers = set()
        method_outliers = {}
        
        for method, method_data in col_analysis["methods"].items():
            if "outliers" in method_data:
                method_outliers[method] = set(method_data["outliers"])
                all_outliers.update(method_data["outliers"])
        
        # Find consensus outliers (detected by at least 2 methods)
        consensus_outliers = []
        for outlier in all_outliers:
            detection_count = sum(1 for outlier_set in method_outliers.values() if outlier in outlier_set)
            if detection_count >= 2:
                consensus_outliers.append(outlier)
        
        col_analysis["consensus"] = {
            "outliers": consensus_outliers[:20],
            "outlier_count": len(consensus_outliers),
            "percentage": round((len(consensus_outliers) / len(col_series)) * 100, 2),
            "confidence": "high" if len(consensus_outliers) > 0 else "low"
        }
        
        # Statistical impact of outliers
        without_outliers = col_series[~col_series.isin(consensus_outliers)]
        if len(without_outliers) > 0 and len(consensus_outliers) > 0:
            col_analysis["impact"] = {
                "mean_change": float(col_series.mean() - without_outliers.mean()),
                "std_change": float(col_series.std() - without_outliers.std()),
                "median_change": float(col_series.median() - without_outliers.median()),
                "range_reduction": float((col_series.max() - col_series.min()) - (without_outliers.max() - without_outliers.min()))
            }
        
        result["outlier_analysis"][col] = col_analysis
    
    # Summary across all columns
    total_outliers = sum(data.get("consensus", {}).get("outlier_count", 0) 
                        for data in result["outlier_analysis"].values())
    total_data_points = sum(data.get("data_points", 0) 
                           for data in result["outlier_analysis"].values())
    
    result["summary"] = {
        "total_outliers": total_outliers,
        "total_data_points": total_data_points,
        "overall_outlier_percentage": round((total_outliers / total_data_points) * 100, 2) if total_data_points > 0 else 0,
        "columns_with_outliers": [col for col, data in result["outlier_analysis"].items() 
                                 if data.get("consensus", {}).get("outlier_count", 0) > 0],
        "most_outlier_prone": max(result["outlier_analysis"].keys(), 
                                 key=lambda x: result["outlier_analysis"][x].get("consensus", {}).get("percentage", 0)) 
                             if result["outlier_analysis"] else None
    }
    
    # Prepare visualization data
    try:
        viz_data = {}
        for col, analysis in result["outlier_analysis"].items():
            if analysis.get("consensus", {}).get("outlier_count", 0) > 0:
                col_series = numeric_df[col].dropna()
                outliers = analysis["consensus"]["outliers"]
                
                viz_data[col] = {
                    "scatter_plot": {
                        "x_values": list(range(len(col_series))),
                        "y_values": col_series.tolist(),
                        "outlier_indices": [i for i, val in enumerate(col_series) if val in outliers],
                        "outlier_values": outliers
                    },
                    "box_plot_data": _prepare_chart_data(col_series)["box_plot"]
                }
        
        result["visualization_data"] = viz_data
    except Exception as e:
        logger.warning(f"Error preparing outlier visualization data: {e}")
    
    return _to_json(result)


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
