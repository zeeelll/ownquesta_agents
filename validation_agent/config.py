"""
Enhanced Validation agent module for OwnQuesta agents.
Provides intelligent functions to parse CSV text, compute advanced EDA summaries,
and answer user questions about datasets.

Requires: pandas, numpy, scipy
"""

from typing import Dict, Any, List
import io
import pandas as pd
import numpy as np
from scipy import stats
import re
import warnings
warnings.filterwarnings('ignore')


def analyze_user_question(question: str, eda_results: Dict[str, Any]) -> str:
    """Analyze user questions about the dataset and provide intelligent responses."""
    q_lower = question.lower().strip()
    
    # Dataset overview questions
    if any(keyword in q_lower for keyword in ['overview', 'summary', 'describe', 'tell me about']):
        shape = eda_results.get('shape', {})
        rows = shape.get('rows', 0)
        cols = shape.get('columns', 0)
        missing_pct = sum(v['percentage'] for v in eda_results.get('missingValues', {}).values()) / max(1, cols)
        
        return f"""📊 **Dataset Overview**:
• **Size**: {rows:,} rows × {cols} columns 
• **Data Quality**: {eda_results.get('validationChecks', {}).get('dataQuality', 'Unknown')}
• **Missing Data**: {missing_pct:.1f}% average across columns
• **Numeric Columns**: {len(eda_results.get('numericColumns', []))}
• **Categorical Columns**: {len(eda_results.get('objectColumns', []))}

This dataset appears suitable for {eda_results.get('goal', {}).get('description', 'analysis')}."""
    
    # Missing data questions
    if any(keyword in q_lower for keyword in ['missing', 'null', 'empty', 'incomplete']):
        missing_vals = eda_results.get('missingValues', {})
        high_missing = {col: info for col, info in missing_vals.items() if info['percentage'] > 20}
        
        if not high_missing:
            return "✅ **Good news!** Your dataset has minimal missing values (all columns <20% missing). This is excellent for most ML tasks."
        else:
            problems = [f"• **{col}**: {info['percentage']}% missing ({info['count']:,} values)" 
                       for col, info in high_missing.items()]
            return f"⚠️ **Missing Data Issues Found**:\n" + "\n".join(problems) + "\n\n**Recommendations**: Consider imputation, dropping columns >50% missing, or using algorithms that handle missing values."
    
    # Correlation questions
    if any(keyword in q_lower for keyword in ['correlation', 'correlate', 'relationship', 'related']):
        corr = eda_results.get('correlation', {})
        if not corr:
            return "📈 **Correlation Analysis**: Not available - need at least 2 numeric columns for correlation analysis."
        
        # Find strong correlations
        strong_corrs = []
        for col1, row in corr.items():
            for col2, val in row.items():
                if col1 != col2 and abs(val) > 0.7:
                    strong_corrs.append((col1, col2, val))
        
        if strong_corrs:
            corr_text = "\n".join([f"• **{col1}** ↔ **{col2}**: {val:.2f}" for col1, col2, val in strong_corrs[:5]])
            return f"🔗 **Strong Correlations Found**:\n{corr_text}\n\n**Note**: Consider feature selection or PCA for highly correlated features."
        else:
            return "📊 **Correlation Analysis**: No strong correlations found (|r| > 0.7). Features appear relatively independent."
    
    # Outliers questions
    if any(keyword in q_lower for keyword in ['outlier', 'outliers', 'extreme', 'anomaly']):
        numeric_summary = eda_results.get('numericalSummary', {})
        outlier_cols = {col: info for col, info in numeric_summary.items() if info.get('outliers', 0) > 0}
        
        if not outlier_cols:
            return "✅ **Outlier Analysis**: No significant outliers detected using IQR method. Your data looks clean!"
        
        outlier_text = "\n".join([f"• **{col}**: {info['outliers']} outliers ({info['outliersPercentage']}%)" 
                                 for col, info in list(outlier_cols.items())[:5]])
        return f"⚠️ **Outliers Detected**:\n{outlier_text}\n\n**Suggestions**: Investigate outliers - they might be data entry errors or important edge cases."
    
    # Data quality questions  
    if any(keyword in q_lower for keyword in ['quality', 'clean', 'good', 'problems', 'issues']):
        checks = eda_results.get('validationChecks', {})
        quality = checks.get('dataQuality', 'Unknown')
        missing_level = checks.get('missingDataLevel', 'Unknown')
        samples = checks.get('sufficientSamples', 'Unknown')
        
        return f"""🔍 **Data Quality Assessment**:
• **Overall Quality**: {quality}
• **Missing Data Level**: {missing_level}
• **Sample Size**: {samples}
• **Data Integrity**: {'✅ Passed' if checks.get('hasData') and checks.get('hasColumns') else '❌ Issues found'}

{'🎉 Your dataset looks ready for machine learning!' if quality == 'Good' and missing_level == 'Acceptable' else '⚠️ Consider data cleaning before modeling.'}"""
    
    # Distribution questions
    if any(keyword in q_lower for keyword in ['distribution', 'skew', 'normal', 'bell curve']):
        numeric_summary = eda_results.get('numericalSummary', {})
        skewed_cols = []
        normal_cols = []
        
        for col, info in numeric_summary.items():
            skew_val = info.get('skewness', 0)
            if abs(skew_val) < 0.5:
                normal_cols.append(col)
            elif abs(skew_val) > 1:
                skewed_cols.append((col, skew_val))
        
        response = "📊 **Distribution Analysis**:\n"
        if normal_cols:
            response += f"✅ **Normal-ish**: {', '.join(normal_cols[:3])}\n"
        if skewed_cols:
            skew_text = ", ".join([f"{col} ({skew:.1f})" for col, skew in skewed_cols[:3]])
            response += f"⚠️ **Highly Skewed**: {skew_text}\n"
        
        return response + "\n**Tip**: Consider log transformation for highly skewed features before modeling."
    
    # Feature engineering questions
    if any(keyword in q_lower for keyword in ['feature', 'engineer', 'transform', 'preprocess']):
        recommendations = []
        
        # Check for categorical features with high cardinality
        obj_summary = eda_results.get('objectSummary', {})
        high_card = [col for col, info in obj_summary.items() if info.get('unique', 0) > 50]
        if high_card:
            recommendations.append(f"• **High cardinality categorical features** ({', '.join(high_card[:2])}): Consider target encoding or feature hashing")
        
        # Check for skewed numeric features
        numeric_summary = eda_results.get('numericalSummary', {})
        skewed = [col for col, info in numeric_summary.items() if abs(info.get('skewness', 0)) > 1]
        if skewed:
            recommendations.append(f"• **Skewed numeric features** ({', '.join(skewed[:2])}): Consider log/sqrt transformation")
        
        # Check for features with outliers
        outlier_features = [col for col, info in numeric_summary.items() if info.get('outliersPercentage', 0) > 10]
        if outlier_features:
            recommendations.append(f"• **Features with outliers** ({', '.join(outlier_features[:2])}): Consider robust scaling")
        
        if not recommendations:
            return "✨ **Feature Engineering**: Your features look well-behaved! Standard scaling should be sufficient for most algorithms."
        
        return "🔧 **Feature Engineering Recommendations**:\n" + "\n".join(recommendations[:5])
    
    # Model recommendations
    if any(keyword in q_lower for keyword in ['model', 'algorithm', 'recommend', 'best', 'choose']):
        goal_type = eda_results.get('goal', {}).get('type', 'eda')
        rows = eda_results.get('shape', {}).get('rows', 0)
        cols = eda_results.get('shape', {}).get('columns', 0)
        
        if goal_type == 'supervised':
            if rows < 1000:
                return "🤖 **Model Recommendations** (Small dataset):\n• **Random Forest**: Great for small datasets\n• **SVM**: Good for high-dimensional data\n• **Gradient Boosting**: Often performs well\n\n**Tip**: Use cross-validation due to limited data."
            elif cols > rows:
                return "🤖 **Model Recommendations** (High-dimensional):\n• **Regularized Linear Models**: L1/L2 regularization\n• **SVM**: Handles high dimensions well\n• **PCA + Models**: Reduce dimensionality first"
            else:
                return "🤖 **Model Recommendations** (Standard dataset):\n• **Random Forest**: Robust baseline\n• **XGBoost/LightGBM**: Often best performance\n• **Neural Networks**: If you have enough data\n\n**Tip**: Try ensemble methods for best results."
        
        elif goal_type == 'unsupervised':
            return "🔍 **Unsupervised Learning Recommendations**:\n• **K-Means**: For spherical clusters\n• **DBSCAN**: For arbitrary-shaped clusters\n• **PCA**: For dimensionality reduction\n\n**Tip**: Scale your features and determine optimal number of clusters."
        
        return "📊 **Analysis Recommendations**:\n• Start with descriptive statistics\n• Visualize distributions and relationships\n• Check for data quality issues\n• Define your target variable clearly"
    
    # Default response for unrecognized questions
    return f"""🤔 **I'd love to help!** I can answer questions about:
• Dataset overview & quality
• Missing data & outliers  
• Correlations & relationships
• Feature distributions
• Model recommendations
• Feature engineering tips

Try asking something like: *"What's the data quality?"* or *"Are there any outliers?"*"""


def perform_advanced_eda_from_csv_text(csv_text: str, goal: Dict[str, Any] = None) -> Dict[str, Any]:
    """Enhanced CSV analysis with intelligent insights and recommendations."""
    """Enhanced CSV analysis with intelligent insights and recommendations."""
    if goal is None:
        goal = {"type": "eda", "description": "Exploratory Data Analysis"}

    try:
        df = pd.read_csv(io.StringIO(csv_text))
    except Exception as e:
        return {"error": f"Failed to parse CSV: {str(e)}", "isValid": False}

    if df.empty:
        return {"error": "Dataset is empty", "isValid": False}

    shape = {"rows": int(df.shape[0]), "columns": int(df.shape[1])}

    # Enhanced column type detection
    column_types = {}
    numeric_cols = []
    object_cols = []
    datetime_cols = []
    
    for col in df.columns:
        # Check for datetime
        if df[col].dtype == 'object':
            sample_vals = df[col].dropna().head(5)
            if all(pd.to_datetime(val, errors='coerce') is not pd.NaT for val in sample_vals):
                column_types[col] = 'datetime'
                datetime_cols.append(col)
                continue
        
        # Check for numeric
        if np.issubdtype(df[col].dtype, np.number):
            column_types[col] = 'numerical'
            numeric_cols.append(col)
        else:
            column_types[col] = 'categorical'
            object_cols.append(col)

    # Enhanced missing values analysis
    missing_values = {}
    for col in df.columns:
        missing = int(df[col].isna().sum())
        pct = round((missing / max(1, len(df))) * 100, 2)
        severity = 'High' if pct > 50 else 'Medium' if pct > 20 else 'Low'
        
        # Analyze missing patterns
        if missing > 0:
            # Check if missing values are clustered
            is_clustered = df[col].isna().astype(int).diff().abs().sum() < missing * 0.5
            pattern = 'Clustered' if is_clustered else 'Random'
        else:
            pattern = 'None'
            
        missing_values[col] = {
            "count": missing, 
            "percentage": pct, 
            "severity": severity,
            "pattern": pattern
        }

    # Enhanced numerical summary with statistical tests
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
        
        # Statistical tests
        normality_p = stats.normaltest(series)[1] if len(series) > 8 else 1.0
        is_normal = normality_p > 0.05
        
        # Calculate percentiles for better insights
        percentiles = {
            'p5': float(series.quantile(0.05)),
            'p25': float(q1),
            'p50': float(series.median()),
            'p75': float(q3),
            'p95': float(series.quantile(0.95))
        }
        
        # Data quality score
        quality_score = 100
        if series.nunique() == 1:
            quality_score -= 50  # Constant column
        if outliers / len(series) > 0.1:
            quality_score -= 20  # Many outliers
        if abs(series.skew()) > 2:
            quality_score -= 15  # Highly skewed
            
        numerical_summary[col] = {
            "count": int(series.count()),
            "unique": int(series.nunique()),
            "mean": round(mean, 4),
            "median": round(float(series.median()), 4),
            "std": round(std, 4),
            "variance": round(variance, 4),
            "min": round(float(series.min()), 4),
            "max": round(float(series.max()), 4),
            "range": round(float(series.max() - series.min()), 4),
            "q1": round(float(q1), 4),
            "q3": round(float(q3), 4),
            "iqr": round(float(iqr), 4),
            "lowerBound": round(float(lower), 4),
            "upperBound": round(float(upper), 4),
            "skewness": round(float(series.skew()), 4),
            "kurtosis": round(float(series.kurtosis()), 4),
            "outliers": outliers,
            "outliersPercentage": round((outliers / max(1, len(series))) * 100, 2),
            "coefficientOfVariation": round((std / mean) * 100 if mean != 0 else 0, 2),
            "isNormalDist": 'Normal' if is_normal else 'Skewed',
            "normalityTest": round(normality_p, 4),
            "percentiles": percentiles,
            "qualityScore": max(0, quality_score),
            "isConstant": series.nunique() == 1,
            "hasZeros": (series == 0).any(),
            "zerosCount": int((series == 0).sum())
        }

    # Enhanced categorical summary
    object_summary = {}
    for col in object_cols:
        vals = df[col].dropna().astype(str)
        if len(vals) == 0:
            continue
            
        freq = vals.value_counts()
        unique = int(vals.nunique())
        
        # Calculate entropy
        if len(vals) > 0:
            ps = freq.values / len(vals)
            entropy = -float(np.sum(ps * np.log2(ps + 1e-10)))
        else:
            entropy = 0.0
        
        # Analyze distribution balance
        most_frequent_pct = (freq.iloc[0] / len(vals)) * 100 if len(freq) > 0 else 0
        is_balanced = most_frequent_pct < 50
        
        # Check for potential identifiers
        is_identifier = unique == len(vals) and unique > 10
        
        top_values = [
            {
                "value": str(val), 
                "count": int(count), 
                "percentage": round((count / len(vals)) * 100, 2)
            } 
            for val, count in freq.head(5).items()
        ]
        
        # Data quality for categorical
        quality_score = 100
        if is_identifier:
            quality_score -= 40  # Likely an ID column
        if most_frequent_pct > 90:
            quality_score -= 30  # Very imbalanced
        if unique == 1:
            quality_score -= 50  # Constant
            
        object_summary[col] = {
            "count": int(len(vals)),
            "unique": unique,
            "uniquePercentage": round((unique / max(1, len(vals))) * 100, 2),
            "entropy": round(entropy, 4),
            "topValues": top_values,
            "isBalanced": is_balanced,
            "dominantValuePct": round(most_frequent_pct, 2),
            "isIdentifier": is_identifier,
            "qualityScore": max(0, quality_score),
            "cardinality": "High" if unique > 50 else "Medium" if unique > 10 else "Low"
        }
    # Enhanced correlation analysis
    correlation = {}
    if len(numeric_cols) > 1:
        corr_df = df[numeric_cols].corr().fillna(0)
        for c1 in corr_df.columns:
            correlation[c1] = {c2: round(float(corr_df.loc[c1, c2]), 4) for c2 in corr_df.columns}

    # Comprehensive validation checks
    total_missing_pct = sum(v['percentage'] for v in missing_values.values()) / max(1, len(missing_values))
    constant_cols = [col for col in df.columns if df[col].nunique() == 1]
    duplicate_rows = df.duplicated().sum()
    
    validation_checks = {
        'hasData': len(df) > 0,
        'hasColumns': df.shape[1] > 0,
        'noEmptyColumns': all(df[c].dropna().shape[0] > 0 for c in df.columns),
        'missingDataLevel': 'Low' if total_missing_pct < 5 else 'Medium' if total_missing_pct < 20 else 'High',
        'dataQuality': 'Excellent' if len(df) > 1000 and total_missing_pct < 5 else 'Good' if len(df) > 100 and total_missing_pct < 20 else 'Fair' if len(df) > 30 else 'Limited',
        'sufficientSamples': 'Excellent' if len(df) >= 1000 else 'Good' if len(df) >= 100 else 'Fair' if len(df) >= 30 else 'Limited',
        'hasConstantColumns': len(constant_cols) > 0,
        'constantColumns': constant_cols,
        'duplicateRows': int(duplicate_rows),
        'duplicatePercentage': round((duplicate_rows / max(1, len(df))) * 100, 2),
        'balanceScore': 100 - min(100, total_missing_pct + len(constant_cols) * 10 + (duplicate_rows / len(df)) * 50),
        'readinessScore': min(100, max(0, 100 - total_missing_pct - len(constant_cols) * 15))
    }

    # Intelligent recommendations based on analysis
    recommendations = []
    
    # Data quality recommendations
    if validation_checks['duplicatePercentage'] > 5:
        recommendations.append("Remove duplicate rows to improve data quality")
    
    if validation_checks['hasConstantColumns']:
        recommendations.append(f"Consider dropping constant columns: {', '.join(constant_cols[:3])}")
    
    # Missing data recommendations
    high_missing_cols = [col for col, info in missing_values.items() if info['percentage'] > 30]
    if high_missing_cols:
        recommendations.append(f"Address high missing data in: {', '.join(high_missing_cols[:3])}")
    
    # Feature engineering recommendations
    if goal.get('type') == 'supervised':
        if len(numeric_cols) > 0:
            recommendations.append("Scale numeric features before training")
        
        high_card_cats = [col for col, info in object_summary.items() if info.get('unique', 0) > 50]
        if high_card_cats:
            recommendations.append(f"Consider encoding high-cardinality categorical features: {', '.join(high_card_cats[:2])}")
        
        # Check for potential target leakage
        identifier_cols = [col for col, info in object_summary.items() if info.get('isIdentifier', False)]
        if identifier_cols:
            recommendations.append(f"Remove potential ID columns to prevent data leakage: {', '.join(identifier_cols[:3])}")
    
    elif goal.get('type') == 'unsupervised':
        recommendations.append("Standardize all numeric features for clustering")
        recommendations.append("Consider dimensionality reduction (PCA) for high-dimensional data")
    
    # Model-specific recommendations
    if validation_checks['sufficientSamples'] in ['Limited', 'Fair']:
        recommendations.append("Consider cross-validation due to limited sample size")
    
    if len(numeric_cols) > len(df):
        recommendations.append("High-dimensional data detected - consider feature selection")

    # Generate intelligent insights
    insights = {
        'dataQuality': [
            f"Dataset contains {shape['rows']:,} samples with {shape['columns']} features",
            f"Missing data level: {validation_checks['missingDataLevel']} ({total_missing_pct:.1f}% average)",
            f"Data quality score: {validation_checks['dataQuality']}"
        ],
        'featureInsights': [],
        'correlationInsights': [],
        'distributionInsights': [],
        'actionableRecommendations': recommendations[:5]  # Limit to top 5
    }
    
    # Feature-specific insights
    if numeric_cols:
        outlier_features = [col for col, info in numerical_summary.items() if info.get('outliersPercentage', 0) > 10]
        if outlier_features:
            insights['featureInsights'].append(f"Features with significant outliers: {', '.join(outlier_features[:3])}")
        
        skewed_features = [col for col, info in numerical_summary.items() if abs(info.get('skewness', 0)) > 1]
        if skewed_features:
            insights['featureInsights'].append(f"Highly skewed features: {', '.join(skewed_features[:3])}")
    
    # Correlation insights
    if correlation:
        strong_corrs = []
        for col1, row in correlation.items():
            for col2, val in row.items():
                if col1 < col2 and abs(val) > 0.7:  # Avoid duplicates
                    strong_corrs.append((col1, col2, val))
        
        if strong_corrs:
            insights['correlationInsights'].append(f"Strong correlations found between {len(strong_corrs)} feature pairs")
        else:
            insights['correlationInsights'].append("No strong feature correlations detected")

    # Distribution insights
    if object_cols:
        imbalanced_cats = [col for col, info in object_summary.items() if info.get('dominantValuePct', 0) > 90]
        if imbalanced_cats:
            insights['distributionInsights'].append(f"Highly imbalanced categorical features: {', '.join(imbalanced_cats[:3])}")

    result = {
        'shape': shape,
        'columns': list(df.columns),
        'columnTypes': column_types,
        'numericColumns': numeric_cols,
        'objectColumns': object_cols,
        'datetimeColumns': datetime_cols,
        'missingValues': missing_values,
        'uniqueValues': {},
        'numericalSummary': numerical_summary,
        'objectSummary': object_summary,
        'correlation': correlation,
        'validationChecks': validation_checks,
        'isValid': validation_checks['hasData'] and validation_checks['hasColumns'] and validation_checks['readinessScore'] > 30,
        'goal': goal,
        'recommendations': recommendations,
        'insights': insights
    }

    return result
