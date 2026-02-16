"""
Enhanced Validation agent module for OwnQuesta agents.
Provides intelligent functions to parse CSV text, compute advanced EDA summaries,
and answer user questions about datasets.

Requires: pandas, numpy, scipy, openai
"""

from typing import Dict, Any, List
import io
import pandas as pd
import numpy as np
from scipy import stats
import re
import warnings
import os
from pathlib import Path
from dotenv import load_dotenv
import logging


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively convert numpy/pandas types to native Python types for JSON serialization."""
    # numpy scalar -> python native
    try:
        if isinstance(obj, (np.generic,)):
            return obj.item()
    except Exception:
        pass

    # numpy arrays -> lists
    try:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass

    # pandas Series/DataFrame -> lists/dicts
    try:
        if isinstance(obj, pd.Series):
            return _sanitize_for_json(obj.tolist())
        if isinstance(obj, pd.DataFrame):
            return _sanitize_for_json(obj.to_dict(orient='records'))
    except Exception:
        pass

    # dict -> sanitize values
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}

    # list/tuple -> sanitize elements
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]

    return obj

# Load environment variables from project root
project_root = Path(__file__).parent.parent
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHAT_MODEL = "gpt-5-mini"  # Using GPT-5-mini for better performance
TEMPERATURE = 0.3
MAX_TOKENS = 1000

# Initialize OpenAI client
openai_client = None
if OPENAI_API_KEY:
    try:
        from openai import OpenAI
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except ImportError:
        logging.warning("OpenAI package not installed. AI-enhanced features will be limited.")
    except Exception as e:
        logging.warning(f"Failed to initialize OpenAI client: {e}")
else:
    logging.warning("OPENAI_API_KEY not set. AI-enhanced features will be limited.")

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


def generate_ai_insights(eda_results: Dict[str, Any]) -> str:
    """Generate AI-powered insights and recommendations using OpenAI."""
    if not openai_client:
        return "🤖 **AI Insights**: OpenAI API not configured. Using rule-based analysis only."

    try:
        # Prepare context for AI analysis
        context = f"""
Dataset Overview:
- Shape: {eda_results.get('shape', {}).get('rows', 0)} rows × {eda_results.get('shape', {}).get('columns', 0)} columns
- Goal: {eda_results.get('goal', {}).get('description', 'General analysis')}
- Data Quality: {eda_results.get('validationChecks', {}).get('dataQuality', 'Unknown')}

Key Statistics:
- Missing Data: {eda_results.get('validationChecks', {}).get('missingDataLevel', 'Unknown')}
- Sample Size: {eda_results.get('validationChecks', {}).get('sufficientSamples', 'Unknown')}
- Numeric Columns: {len(eda_results.get('numericColumns', []))}
- Categorical Columns: {len(eda_results.get('objectColumns', []))}

Top Insights:
"""

        # Add key insights from EDA
        insights = []
        if eda_results.get('validationChecks', {}).get('duplicatePercentage', 0) > 5:
            insights.append(f"- High duplicate data: {eda_results['validationChecks']['duplicatePercentage']}%")
        if eda_results.get('validationChecks', {}).get('missingDataLevel') == 'High':
            insights.append("- Significant missing data requiring attention")
        if len(eda_results.get('correlation', {})) > 0:
            insights.append("- Correlation analysis available for numeric features")

        context += "\n".join(insights) if insights else "- Standard dataset characteristics"

        prompt = f"""As an expert data scientist, analyze this dataset summary and provide 3-5 key insights and actionable recommendations:

{context}

Focus on:
1. Data quality assessment
2. Feature engineering opportunities
3. Modeling considerations
4. Potential challenges
5. Best practices for this dataset

Keep your response concise but insightful, using bullet points."""

        response = openai_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert data scientist providing actionable insights about datasets. Be concise, practical, and focus on what matters most for successful machine learning."},
                {"role": "user", "content": prompt}
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            timeout=15  # Add 15 second timeout
        )

        ai_insights = response.choices[0].message.content.strip()
        return f"🤖 **AI-Powered Insights**:\n{ai_insights}"

    except Exception as e:
        return f"🤖 **AI Insights**: Error generating insights - {str(e)}. Using rule-based analysis."


def generate_ml_ai_insights(eda_results: Dict[str, Any], ml_validation: Dict[str, Any]) -> str:
    """Generate AI-powered ML-specific insights and recommendations."""
    if not openai_client:
        return "🤖 **ML AI Insights**: OpenAI API not configured. Using rule-based recommendations."

    try:
        goal_type = ml_validation.get('goalType', 'eda')
        shape = eda_results.get('shape', {})
        data_quality = eda_results.get('validationChecks', {}).get('dataQuality', 'Unknown')

        context = f"""
ML Task: {goal_type}
Dataset: {shape.get('rows', 0)} rows × {shape.get('columns', 0)} columns
Data Quality: {data_quality}
Preprocessing Steps: {len(ml_validation.get('preprocessingSteps', []))}
Feature Engineering: {len(ml_validation.get('featureEngineering', []))}
Recommended Models: {len(ml_validation.get('modelRecommendations', []))}
Risks Identified: {len(ml_validation.get('risksAndWarnings', []))}
"""

        prompt = f"""As an expert machine learning engineer, provide strategic insights for this {goal_type} project:

{context}

Provide 4-6 key strategic recommendations covering:
1. Model selection rationale
2. Feature engineering priorities
3. Potential pitfalls to avoid
4. Performance optimization strategies
5. Deployment considerations
6. Monitoring and maintenance needs

Be specific to the {goal_type} task and dataset characteristics. Keep recommendations actionable and practical."""

        response = openai_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "You are a senior ML engineer providing strategic guidance for machine learning projects. Focus on practical, implementable recommendations that drive success."},
                {"role": "user", "content": prompt}
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            timeout=15  # Add 15 second timeout
        )

        ai_insights = response.choices[0].message.content.strip()
        return f"🎯 **AI-Powered ML Strategy**:\n{ai_insights}"

    except Exception as e:
        return f"🤖 **ML AI Insights**: Error generating insights - {str(e)}. Using rule-based recommendations."


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
        return _sanitize_for_json({"error": "Dataset is empty", "isValid": False})

    shape = {"rows": int(df.shape[0]), "columns": int(df.shape[1])}
    size = shape['rows'] * shape['columns']

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
        
        # Calculate mode (handling multiple modes)
        mode_series = series.mode()
        mode = float(mode_series.iloc[0]) if len(mode_series) > 0 else None
        
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
            "mode": round(mode, 4) if mode is not None else None,
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
            "isConstant": bool(series.nunique() == 1),
            "hasZeros": bool((series == 0).any()),
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
        'size': size,
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

    # Derive a user-friendly satisfaction score and simple goal-understanding hints
    try:
        satisfaction = int(validation_checks.get('readinessScore', 0))
    except Exception:
        satisfaction = 0

    # Simple target/goal detection heuristics
    target_guess = None
    lower_cols = [c.lower() for c in df.columns]
    for keyword in ('target', 'label', 'y', 'outcome', 'class'):
        if keyword in lower_cols:
            idx = lower_cols.index(keyword)
            target_guess = df.columns[idx]
            break

    interpreted_task = 'Auto-detected'
    confidence = 0.0
    if goal and goal.get('type'):
        if goal.get('type') == 'supervised':
            interpreted_task = 'Predictive (supervised)'
        elif goal.get('type') == 'unsupervised':
            interpreted_task = 'Unsupervised (clustering)'
        confidence = 0.9
    else:
        # Heuristic: if a target-like column exists -> supervised
        if target_guess:
            interpreted_task = 'Predictive (supervised)'
            confidence = 0.9
        elif len(object_cols) > 0 and len(numeric_cols) > 0:
            interpreted_task = 'Predictive (supervised)'
            confidence = 0.6
        elif len(numeric_cols) > 1:
            interpreted_task = 'Predictive (supervised)'
            confidence = 0.5
        else:
            interpreted_task = 'Exploratory / Unsupervised'
            confidence = 0.4

    result['satisfaction_score'] = satisfaction
    result['goal_understanding'] = {
        'interpreted_task': interpreted_task,
        'target_column_guess': target_guess or 'To be determined',
        'confidence': float(confidence)
    }

    # Add AI-powered insights after result is created
    # result['aiInsights'] = generate_ai_insights(result)  # Disabled for testing

    # Sanitize result to ensure JSON serializable types
    return _sanitize_for_json(result)


def perform_ml_validation_from_eda(eda_results: Dict[str, Any]) -> Dict[str, Any]:
    """Perform comprehensive ML validation based on EDA results."""
    try:
        # Reconstruct dataframe from EDA results (simplified approach)
        # In a real implementation, you'd store the original dataframe
        # For now, we'll work with the statistical summaries

        goal = eda_results.get('goal', {})
        goal_type = goal.get('type', 'eda')

        ml_validation = {
            'goalType': goal_type,
            'preprocessingSteps': [],
            'featureEngineering': [],
            'modelRecommendations': [],
            'validationMetrics': [],
            'risksAndWarnings': [],
            'implementationCode': {},
            'performanceEstimates': {}
        }

        # Preprocessing recommendations
        missing_data = eda_results.get('missingValues', {})
        high_missing_cols = [col for col, info in missing_data.items() if info.get('percentage', 0) > 20]

        if high_missing_cols:
            ml_validation['preprocessingSteps'].append({
                'step': 'Handle Missing Data',
                'description': f'Columns with high missing data: {", ".join(high_missing_cols)}',
                'methods': ['Drop columns (>50% missing)', 'Imputation (mean/median/mode)', 'Advanced imputation (KNN, MICE)'],
                'priority': 'High'
            })

        # Feature engineering based on numerical summaries
        numerical_summary = eda_results.get('numericalSummary', {})
        skewed_features = []
        outlier_features = []

        for col, stats in numerical_summary.items():
            if abs(stats.get('skewness', 0)) > 1:
                skewed_features.append(col)
            if stats.get('outliersPercentage', 0) > 10:
                outlier_features.append(col)

        if skewed_features:
            ml_validation['featureEngineering'].append({
                'type': 'Transformation',
                'description': f'Apply transformations to skewed features: {", ".join(skewed_features)}',
                'methods': ['Log transformation', 'Square root transformation', 'Box-Cox transformation'],
                'reason': 'Reduce skewness for better model performance'
            })

        if outlier_features:
            ml_validation['featureEngineering'].append({
                'type': 'Outlier Handling',
                'description': f'Handle outliers in: {", ".join(outlier_features)}',
                'methods': ['IQR method', 'Z-score method', 'Robust scaling', 'Winsorization'],
                'reason': 'Prevent outlier influence on model training'
            })

        # Encoding recommendations for categorical features
        object_summary = eda_results.get('objectSummary', {})
        high_cardinality = [col for col, info in object_summary.items() if info.get('unique', 0) > 20]

        if high_cardinality:
            ml_validation['featureEngineering'].append({
                'type': 'Categorical Encoding',
                'description': f'High cardinality categorical features: {", ".join(high_cardinality)}',
                'methods': ['Target encoding', 'Frequency encoding', 'Feature hashing', 'Binary encoding'],
                'reason': 'Convert categorical to numerical features'
            })

        # Model recommendations based on goal
        if goal_type == 'supervised':
            ml_validation['modelRecommendations'] = [
                {
                    'algorithm': 'Random Forest',
                    'type': 'Ensemble',
                    'pros': ['Handles mixed data types', 'Feature importance', 'Robust to outliers'],
                    'cons': ['Can overfit', 'Less interpretable'],
                    'use_case': 'General purpose, good baseline'
                },
                {
                    'algorithm': 'XGBoost',
                    'type': 'Ensemble',
                    'pros': ['High performance', 'Handles missing values', 'Feature selection'],
                    'cons': ['Complex hyperparameters', 'Computationally intensive'],
                    'use_case': 'High-performance modeling'
                },
                {
                    'algorithm': 'Logistic Regression',
                    'type': 'Linear',
                    'pros': ['Interpretable', 'Fast training', 'Probabilistic outputs'],
                    'cons': ['Assumes linear relationships', 'Sensitive to outliers'],
                    'use_case': 'Interpretable models, probability estimates'
                }
            ]

            # Validation metrics for supervised learning
            ml_validation['validationMetrics'] = [
                'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC',
                'Confusion Matrix', 'Classification Report', 'Cross-validation scores'
            ]

        elif goal_type == 'unsupervised':
            ml_validation['modelRecommendations'] = [
                {
                    'algorithm': 'K-Means Clustering',
                    'type': 'Centroid-based',
                    'pros': ['Simple and fast', 'Scalable', 'Easy to interpret'],
                    'cons': ['Assumes spherical clusters', 'Sensitive to initialization'],
                    'use_case': 'Spherical cluster detection'
                },
                {
                    'algorithm': 'DBSCAN',
                    'type': 'Density-based',
                    'pros': ['Handles arbitrary shapes', 'No need to specify k', 'Handles noise'],
                    'cons': ['Sensitive to parameters', 'Struggles with varying densities'],
                    'use_case': 'Arbitrary shaped clusters, noise detection'
                },
                {
                    'algorithm': 'PCA',
                    'type': 'Dimensionality Reduction',
                    'pros': ['Linear transformation', 'Preserves variance', 'Fast computation'],
                    'cons': ['Linear method only', 'May not capture non-linear relationships'],
                    'use_case': 'Feature reduction, visualization'
                }
            ]

            # Validation metrics for unsupervised learning
            ml_validation['validationMetrics'] = [
                'Silhouette Score', 'Calinski-Harabasz Index', 'Davies-Bouldin Index',
                'Elbow Method', 'Gap Statistics', 'Explained Variance Ratio'
            ]

        # Risks and warnings
        shape = eda_results.get('shape', {})
        if shape.get('rows', 0) < 100:
            ml_validation['risksAndWarnings'].append({
                'level': 'High',
                'issue': 'Limited Sample Size',
                'description': f'Only {shape.get("rows", 0)} samples may lead to overfitting',
                'mitigation': 'Use cross-validation, consider data augmentation, or collect more data'
            })

        if len(eda_results.get('numericColumns', [])) > shape.get('rows', 0):
            ml_validation['risksAndWarnings'].append({
                'level': 'High',
                'issue': 'High Dimensionality',
                'description': f'{len(eda_results.get("numericColumns", []))} features vs {shape.get("rows", 0)} samples',
                'mitigation': 'Use feature selection, dimensionality reduction (PCA), or regularization'
            })

        # Performance estimates
        sample_size = shape.get('rows', 0)
        if sample_size >= 1000:
            ml_validation['performanceEstimates'] = {
                'confidence': 'High',
                'expected_accuracy': 'Good model performance possible with proper preprocessing',
                'data_sufficiency': 'Sufficient data for robust model training and validation'
            }
        elif sample_size >= 100:
            ml_validation['performanceEstimates'] = {
                'confidence': 'Medium',
                'expected_accuracy': 'Moderate performance expected, use cross-validation',
                'data_sufficiency': 'Adequate for basic modeling, consider more data for better results'
            }
        else:
            ml_validation['performanceEstimates'] = {
                'confidence': 'Low',
                'expected_accuracy': 'Limited performance expected due to small sample size',
                'data_sufficiency': 'Insufficient data, results may not generalize well'
            }

        # Generate implementation code
        ml_validation['implementationCode'] = generate_ml_implementation_code(eda_results)

        # Add AI-powered insights for ML validation
        ml_validation['aiInsights'] = generate_ml_ai_insights(eda_results, ml_validation)

        # Ensure returned structure is JSON-serializable
        return _sanitize_for_json(ml_validation)

    except Exception as e:
        return _sanitize_for_json({
            'error': f'ML validation failed: {str(e)}',
            'isValid': False
        })


def generate_ml_implementation_code(eda_results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate Python code for ML implementation based on EDA results."""
    goal = eda_results.get('goal', {})
    goal_type = goal.get('type', 'eda')

    code = {
        'eda_code': generate_eda_code(eda_results),
        'preprocessing_code': '',
        'model_code': '',
        'validation_code': '',
        'full_pipeline': ''
    }

    # Preprocessing code
    preprocessing_steps = []

    # Handle missing values
    missing_data = eda_results.get('missingValues', {})
    high_missing_cols = [col for col, info in missing_data.items() if info.get('percentage', 0) > 20]

    if high_missing_cols:
        preprocessing_steps.append(f"""
# Handle missing values for high-missing columns
high_missing_cols = {high_missing_cols}
df = df.drop(columns=high_missing_cols)  # Drop columns with >20% missing data
""")

    # Handle remaining missing values
    preprocessing_steps.append("""
# Handle remaining missing values
from sklearn.impute import SimpleImputer

# For numerical columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
if len(num_cols) > 0:
    num_imputer = SimpleImputer(strategy='median')
    df[num_cols] = num_imputer.fit_transform(df[num_cols])

# For categorical columns
cat_cols = df.select_dtypes(include=['object']).columns
if len(cat_cols) > 0:
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
""")

    # Encoding categorical variables
    object_summary = eda_results.get('objectSummary', {})
    cat_cols = list(object_summary.keys())

    if cat_cols:
        preprocessing_steps.append(f"""
# Encode categorical variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

categorical_cols = {cat_cols}

# Label encoding for ordinal/binary categories
label_encoder = LabelEncoder()
for col in categorical_cols:
    if df[col].nunique() <= 2:  # Binary categories
        df[col] = label_encoder.fit_transform(df[col])
    else:
        # One-hot encoding for multi-class categories
        df = pd.get_dummies(df, columns=[col], prefix=col, drop_first=True)
""")

    # Feature scaling
    numerical_cols = eda_results.get('numericColumns', [])
    if numerical_cols:
        preprocessing_steps.append("""
# Feature scaling
from sklearn.preprocessing import StandardScaler, RobustScaler

# Use RobustScaler if there are outliers
scaler = StandardScaler()  # or RobustScaler() for outlier robustness
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
""")

    code['preprocessing_code'] = '\n'.join(preprocessing_steps)

    # Model code based on goal type
    if goal_type == 'supervised':
        code['model_code'] = """
# Supervised Learning Models
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import classification_report, mean_squared_error, r2_score

# Assume 'target' is your target column - replace with actual column name
X = df.drop('target', axis=1)
y = df['target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Try different models
models = {
    'Random Forest': RandomForestClassifier(random_state=42) if y.nunique() <= 10 else RandomForestRegressor(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42) if y.nunique() <= 10 else LinearRegression(),
    'XGBoost': XGBClassifier(random_state=42) if y.nunique() <= 10 else XGBRegressor(random_state=42)
}

# Train and evaluate models
for name, model in models.items():
    print(f"\\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if y.nunique() <= 10:  # Classification
        print(f"{name} Classification Report:")
        print(classification_report(y_test, y_pred))
    else:  # Regression
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"{name} - MSE: {mse:.4f}, R²: {r2:.4f}")
"""
    elif goal_type == 'unsupervised':
        code['model_code'] = """
# Unsupervised Learning Models
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Prepare data
X = df.select_dtypes(include=['int64', 'float64'])

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means Clustering
print("Performing K-Means Clustering...")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

# Evaluate clustering
silhouette = silhouette_score(X_scaled, kmeans_labels)
ch_score = calinski_harabasz_score(X_scaled, kmeans_labels)
db_score = davies_bouldin_score(X_scaled, kmeans_labels)

print(f"K-Means Evaluation:")
print(f"Silhouette Score: {silhouette:.4f}")
print(f"Calinski-Harabasz Score: {ch_score:.4f}")
print(f"Davies-Bouldin Score: {db_score:.4f}")

# DBSCAN Clustering
print("\\nPerforming DBSCAN Clustering...")
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

# Count clusters (excluding noise labeled as -1)
n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)

print(f"DBSCAN Results:")
print(f"Number of clusters: {n_clusters}")
print(f"Number of noise points: {n_noise}")

# PCA for dimensionality reduction
print("\\nPerforming PCA...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.4f}")
"""

    # Validation code
    if goal_type == 'supervised':
        code['validation_code'] = """
# Model Validation for Supervised Learning
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

# Cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy' if y.nunique() <= 10 else 'r2')
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Hyperparameter tuning example
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\\nTop 10 Feature Importances:")
print(feature_importance.head(10))
"""
    elif goal_type == 'unsupervised':
        code['validation_code'] = """
# Model Validation for Unsupervised Learning
from sklearn.metrics import silhouette_samples
import matplotlib.pyplot as plt
import seaborn as sns

# Elbow method for optimal k in K-Means
inertias = []
silhouette_scores = []
K = range(2, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    if k > 1:
        silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Plot elbow curve
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(K, inertias, 'bx-')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')

plt.subplot(1, 2, 2)
plt.plot(K[1:], silhouette_scores, 'rx-')
plt.xlabel('k')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis')

plt.tight_layout()
plt.show()

# Detailed silhouette analysis for chosen k
chosen_k = 3  # Adjust based on elbow method
kmeans_final = KMeans(n_clusters=chosen_k, random_state=42, n_init=10)
final_labels = kmeans_final.fit_predict(X_scaled)

# Silhouette analysis
silhouette_vals = silhouette_samples(X_scaled, final_labels)

plt.figure(figsize=(8, 6))
y_lower = 10
for i in range(chosen_k):
    cluster_silhouette_vals = silhouette_vals[final_labels == i]
    cluster_silhouette_vals.sort()
    y_upper = y_lower + len(cluster_silhouette_vals)
    plt.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_vals)
    plt.text(-0.05, y_lower + 0.5 * len(cluster_silhouette_vals), str(i))
    y_lower = y_upper + 10

plt.axvline(x=silhouette_score(X_scaled, final_labels), color="red", linestyle="--")
plt.xlabel("Silhouette coefficient values")
plt.ylabel("Cluster label")
plt.title("Silhouette plot for clustering")
plt.show()
"""

    # Full pipeline code
    code['full_pipeline'] = f"""
# Complete ML Pipeline
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, silhouette_score
import matplotlib.pyplot as plt

# Load your data
df = pd.read_csv('your_dataset.csv')
print("Dataset loaded successfully!")
print(f"Shape: {{df.shape}}")

# Step 1: Preprocessing
print("\\n=== PREPROCESSING ===")

# Handle missing values
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = df.select_dtypes(include=['object']).columns

if len(num_cols) > 0:
    num_imputer = SimpleImputer(strategy='median')
    df[num_cols] = num_imputer.fit_transform(df[num_cols])

if len(cat_cols) > 0:
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

# Encode categorical variables
for col in cat_cols:
    if df[col].nunique() <= 2:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    else:
        df = pd.get_dummies(df, columns=[col], prefix=col, drop_first=True)

# Scale numerical features
scaler = StandardScaler()
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
if len(num_cols) > 0:
    df[num_cols] = scaler.fit_transform(df[num_cols])

print("Preprocessing completed!")

# Step 2: Model Training
print("\\n=== MODEL TRAINING ===")

{code['model_code']}

# Step 3: Validation
print("\\n=== MODEL VALIDATION ===")

{code['validation_code']}

print("\\n=== PIPELINE COMPLETED ===")
print("Your ML model is ready for predictions!")
"""

    return code


def generate_eda_code(eda_results: Dict[str, Any]) -> str:
    """Generate comprehensive EDA Python code."""
    return """
# Complete Exploratory Data Analysis (EDA) Code
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Load your dataset
df = pd.read_csv('your_dataset.csv')

print("=== DATASET OVERVIEW ===")
print(f"Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\\nData Types:\\n{df.dtypes}")
print(f"\\nFirst 5 rows:\\n{df.head()}")

# Basic info
print(f"\\n=== BASIC INFORMATION ===")
print(df.info())

# Missing values analysis
print(f"\\n=== MISSING VALUES ANALYSIS ===")
missing_data = df.isnull().sum()
missing_percentage = (missing_data / len(df)) * 100
missing_summary = pd.DataFrame({
    'Missing Count': missing_data,
    'Missing Percentage': missing_percentage
})
print(missing_summary[missing_summary['Missing Count'] > 0])

# Numerical columns analysis
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
if len(num_cols) > 0:
    print(f"\\n=== NUMERICAL COLUMNS ANALYSIS ===")
    print(f"\\nDescriptive Statistics:\\n{df[num_cols].describe()}")

    # Distribution plots
    fig, axes = plt.subplots(len(num_cols), 2, figsize=(15, 5*len(num_cols)))
    if len(num_cols) == 1:
        axes = [axes]

    for i, col in enumerate(num_cols):
        # Histogram
        sns.histplot(df[col], kde=True, ax=axes[i][0])
        axes[i][0].set_title(f'Distribution of {col}')
        axes[i][0].axvline(df[col].mean(), color='red', linestyle='--', label=f'Mean: {df[col].mean():.2f}')
        axes[i][0].axvline(df[col].median(), color='green', linestyle='--', label=f'Median: {df[col].median():.2f}')
        axes[i][0].legend()

        # Box plot
        sns.boxplot(x=df[col], ax=axes[i][1])
        axes[i][1].set_title(f'Box Plot of {col}')

        # Calculate outliers using IQR
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        print(f"\\n{col} - Outliers: {len(outliers)} ({(len(outliers)/len(df)*100):.2f}%)")

    plt.tight_layout()
    plt.show()

# Categorical columns analysis
cat_cols = df.select_dtypes(include=['object']).columns
if len(cat_cols) > 0:
    print(f"\\n=== CATEGORICAL COLUMNS ANALYSIS ===")

    for col in cat_cols:
        print(f"\\n{col}:")
        print(f"Unique values: {df[col].nunique()}")
        print(f"Most common: {df[col].mode()[0]}")
        print(f"Value counts:\\n{df[col].value_counts().head()}")

        # Bar plot
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, y=col, order=df[col].value_counts().index)
        plt.title(f'Distribution of {col}')
        plt.show()

# Correlation analysis
if len(num_cols) > 1:
    print(f"\\n=== CORRELATION ANALYSIS ===")

    # Correlation matrix
    corr_matrix = df[num_cols].corr()
    print("Correlation Matrix:")
    print(corr_matrix)

    # Heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.show()

    # Find strong correlations
    strong_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:
                strong_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))

    if strong_corr:
        print("\\nStrong correlations (|r| > 0.7):")
        for col1, col2, corr in strong_corr:
            print(f"{col1} ↔ {col2}: {corr:.3f}")
    else:
        print("\\nNo strong correlations found.")

print("\\n=== EDA COMPLETED ===")
print("Your exploratory data analysis is complete!")
print("Key insights have been generated and visualizations created.")
"""
