"""
Lightweight Validation agent module for OwnQuesta agents.
Provides functions to parse CSV text and compute basic EDA summaries.
This file is intended as a starting point — you can import and call
`perform_advanced_eda_from_csv_text(csv_text, goal)` from `main.py` or other services.

Requires: pandas, numpy
"""

from typing import Dict, Any
import io
import pandas as pd
import numpy as np


def perform_advanced_eda_from_csv_text(csv_text: str, goal: Dict[str, Any] = None) -> Dict[str, Any]:
    """Parse CSV text and return EDA summary compatible with the frontend structure.

    Args:
        csv_text: Raw CSV contents as string.
        goal: Optional dict describing the goal (type/description).

    Returns:
        A dict containing shape, numerical and categorical summaries, missing values,
        correlation matrix (as nested dict), validationChecks, recommendations and insights.
    """
    if goal is None:
        goal = {"type": "eda", "description": "Exploratory Data Analysis"}

    df = pd.read_csv(io.StringIO(csv_text))

    shape = {"rows": int(df.shape[0]), "columns": int(df.shape[1])}

    # Column types
    column_types = df.dtypes.apply(lambda x: 'numerical' if np.issubdtype(x, np.number) else 'categorical').to_dict()
    numeric_cols = [c for c, t in column_types.items() if t == 'numerical']
    object_cols = [c for c, t in column_types.items() if t == 'categorical']

    # Missing values
    missing_values = {}
    for col in df.columns:
        missing = int(df[col].isna().sum())
        pct = round((missing / max(1, len(df))) * 100, 2)
        severity = 'High' if pct > 50 else 'Medium' if pct > 20 else 'Low'
        missing_values[col] = {"count": missing, "percentage": pct, "severity": severity}

    # Numerical summary
    numerical_summary = {}
    for col in numeric_cols:
        series = df[col].dropna().astype(float)
        if len(series) == 0:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = int(((series < lower) | (series > upper)).sum())
        variance = float(series.var())
        std = float(series.std())
        mean = float(series.mean())
        numerical_summary[col] = {
            "count": int(series.count()),
            "unique": int(series.nunique()),
            "mean": round(mean, 2),
            "median": round(float(series.median()), 2),
            "std": round(std, 2),
            "variance": round(variance, 2),
            "min": round(float(series.min()), 2),
            "max": round(float(series.max()), 2),
            "range": round(float(series.max() - series.min()), 2),
            "q1": round(float(q1), 2),
            "q3": round(float(q3), 2),
            "iqr": round(float(iqr), 2),
            "lowerBound": round(float(lower), 2),
            "upperBound": round(float(upper), 2),
            "skewness": round(float(series.skew()), 2),
            "outliers": outliers,
            "outliersPercentage": round((outliers / max(1, len(series))) * 100, 2),
            "coefficientOfVariation": round((std / mean) * 100 if mean != 0 else 0, 2),
            "isNormalDist": 'Approximately Normal' if abs(series.skew()) < 0.5 else 'Moderately Skewed' if abs(series.skew()) < 1 else 'Highly Skewed'
        }

    # Categorical/object summary
    object_summary = {}
    for col in object_cols:
        vals = df[col].dropna().astype(str)
        freq = vals.value_counts().to_dict()
        unique = int(vals.nunique())
        entropy = 0.0
        if len(vals) > 0:
            ps = np.array(list(freq.values())) / len(vals)
            entropy = -float(np.sum(ps * np.log2(ps)))
        top_values = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:5]
        object_summary[col] = {
            "count": int(len(vals)),
            "unique": unique,
            "uniquePercentage": round((unique / max(1, len(vals))) * 100, 2),
            "entropy": round(entropy, 2),
            "topValues": [{"value": v, "count": c, "percentage": round((c / max(1, len(vals))) * 100, 2)} for v, c in top_values]
        }

    # Correlation
    correlation = {}
    if len(numeric_cols) > 1:
        corr_df = df[numeric_cols].corr().fillna(0)
        for c1 in corr_df.columns:
            correlation[c1] = {c2: round(float(corr_df.loc[c1, c2]), 2) for c2 in corr_df.columns}

    # Validation checks
    validation_checks = {
        'hasData': len(df) > 0,
        'hasColumns': df.shape[1] > 0,
        'noEmptyColumns': all(df[c].dropna().shape[0] > 0 for c in df.columns),
        'missingDataLevel': 'Acceptable' if all(v['percentage'] < 50 for v in missing_values.values()) else 'High',
        'dataQuality': 'Good' if len(df) > 30 else 'Fair' if len(df) > 10 else 'Limited',
        'sufficientSamples': 'Excellent' if len(df) >= 100 else 'Good' if len(df) >= 30 else 'Limited'
    }

    # Basic recommendations (short)
    recommendations = []
    if goal.get('type') == 'supervised':
        recommendations.append('Consider target column selection and handle class imbalance if present')
    elif goal.get('type') == 'unsupervised':
        recommendations.append('Scale numerical features before clustering and consider PCA when many features')
    else:
        recommendations.append('Run feature-level checks and handle missing data')

    # Simple lightweight "insights" stub
    insights = {
        'dataQuality': [],
        'featureInsights': [],
        'correlationInsights': [],
        'distributionInsights': [],
        'actionableRecommendations': recommendations
    }

    result = {
        'shape': shape,
        'columns': list(df.columns),
        'columnTypes': column_types,
        'numericColumns': numeric_cols,
        'objectColumns': object_cols,
        'missingValues': missing_values,
        'uniqueValues': {},
        'numericalSummary': numerical_summary,
        'objectSummary': object_summary,
        'correlation': correlation,
        'validationChecks': validation_checks,
        'isValid': validation_checks['hasData'] and validation_checks['hasColumns'],
        'goal': goal,
        'recommendations': recommendations,
        'insights': insights
    }

    return result
