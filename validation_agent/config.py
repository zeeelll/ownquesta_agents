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
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")  # Use gpt-4o-mini for better compatibility
# Allow external configuration for temperature and max tokens
# Default temperature to 0.7 for faster, more deterministic responses
TEMPERATURE = float(os.getenv('OPENAI_TEMPERATURE', os.getenv('TEMPERATURE', '0.7')))
try:
    MAX_TOKENS = int(os.getenv('OPENAI_MAX_TOKENS') or os.getenv('MAX_TOKENS') or '4096')
except Exception:
    MAX_TOKENS = 4096

# Timeout configuration (set high to avoid interruptions)
API_TIMEOUT = int(os.getenv('OPENAI_TIMEOUT', '120'))  # 2 minutes default

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


def _safe_chat_completion_call(messages, model, **kwargs):
    """Call the OpenAI chat completion API with defensive retries for unsupported params.

    If the model rejects parameters like `temperature` or `max_completion_tokens`,
    this helper will retry the call with those parameters removed.
    """
    if not openai_client:
        raise RuntimeError("OpenAI client not configured")

    try:
        # Use max_tokens instead of max_completion_tokens for better compatibility
        if 'max_completion_tokens' in kwargs:
            kwargs['max_tokens'] = kwargs.pop('max_completion_tokens')
        
        # Remove temperature if it's the default (1.0) to avoid issues
        if kwargs.get('temperature') == 1.0:
            kwargs.pop('temperature', None)
        
        return openai_client.chat.completions.create(model=model, messages=messages, **kwargs)
    except Exception as e:
        err = str(e)
        logging.warning("OpenAI chat completion failed on first attempt: %s", err)

        # If the error mentions unsupported parameter(s), attempt retries without them
        retry_kwargs = dict(kwargs)

        # Common problematic params: temperature, max_completion_tokens, max_tokens
        removed = []
        for param in ('temperature', 'max_completion_tokens', 'max_tokens', 'timeout'):
            if param in retry_kwargs:
                retry_kwargs.pop(param)
                removed.append(param)

        if removed:
            try:
                logging.info("Retrying OpenAI chat completion without params: %s", removed)
                return openai_client.chat.completions.create(model=model, messages=messages, **retry_kwargs)
            except Exception as e2:
                logging.error("OpenAI retry without params %s failed: %s", removed, e2)
                raise

        # No known params to remove or retry failed - re-raise
        raise


def analyze_user_question(question: str, eda_results: Dict[str, Any]) -> Dict[str, str]:
    """Comprehensive validation agent - answers ALL validation-related questions intelligently.
    
    Returns:
        Dict with 'concise' (2-line chatbot answer), 'detailed' (full main page answer), 
        and optional 'code_section' to trigger code display
    """
    q_lower = question.lower().strip()
    shape = eda_results.get('shape', {})
    rows = shape.get('rows', 0)
    cols = shape.get('columns', 0)
    numeric_cols = eda_results.get('numericColumns', [])
    categorical_cols = eda_results.get('objectColumns', [])
    missing_vals = eda_results.get('missingValues', {})
    numeric_summary = eda_results.get('numericalSummary', {})
    
    # ============================================
    # DATASET OVERVIEW & STRUCTURE
    # ============================================
    if any(kw in q_lower for kw in ['overview', 'summary', 'describe', 'tell me about', 'about', 'dataset', 'structure', 'shape', 'size']):
        missing_pct = sum(v.get('percentage', 0) for v in missing_vals.values()) / max(1, cols)
        quality_score = "Excellent" if missing_pct < 5 else "Good" if missing_pct < 10 else "Fair" if missing_pct < 30 else "Needs attention"
        
        concise = f"📊 Dataset: {rows:,} rows × {cols} columns. Quality: {quality_score}. {len(numeric_cols)} numeric, {len(categorical_cols)} categorical features."
        
        detailed = f"""📊 **Complete Dataset Validation Report**:

🔢 **Dataset Structure**:
• **Total Records**: {rows:,} rows
• **Total Features**: {cols} columns
• **Numeric Features**: {len(numeric_cols)} (for quantitative analysis)
• **Categorical Features**: {len(categorical_cols)} (for grouping/classification)
• **Memory Size**: ~{(rows * cols * 8 / 1024 / 1024):.1f} MB

📈 **Data Quality Score: {quality_score}**:
• **Completeness**: {100 - missing_pct:.1f}% (Missing: {missing_pct:.1f}%)
• **Data Integrity**: {eda_results.get('validationChecks', {}).get('dataQuality', 'Validated')}
• **Duplicates**: {eda_results.get('duplicates', 0)} rows

🎯 **ML Readiness Assessment**:
• Sample Size: {'✅ Excellent' if rows > 10000 else '✅ Good' if rows > 1000 else '⚠️ Limited' if rows > 100 else '❌ Too small'}
• Feature Count: {'✅ Optimal' if 5 <= cols <= 100 else '⚠️ Review' if cols > 100 else '⚠️ Few features'}
• Balance: {'✅ Well-balanced' if 0.3 <= len(numeric_cols)/max(1,cols) <= 0.7 else '⚠️ Imbalanced'}

💡 **Validation Summary**: Dataset is {'ready for machine learning' if quality_score in ['Excellent', 'Good'] and rows > 100 else 'needs preprocessing before ML'}."""
        
        return {"concise": concise, "detailed": detailed}
    
    # ============================================
    # DATA TYPES & COLUMNS
    # ============================================
    if any(kw in q_lower for kw in ['data type', 'datatype', 'dtype', 'column type', 'feature type', 'type']):
        concise = f"📋 {len(numeric_cols)} numeric columns, {len(categorical_cols)} categorical columns, {len(eda_results.get('datetimeColumns', []))} datetime."
        
        detailed = f"""📋 **Column Data Types Analysis**:

🔢 **Numerical Columns** ({len(numeric_cols)}):
{chr(10).join([f'• {col} - Continuous numeric data' for col in numeric_cols[:10]])}
{'...' if len(numeric_cols) > 10 else ''}

🏷️ **Categorical Columns** ({len(categorical_cols)}):
{chr(10).join([f'• {col} - Discrete categories' for col in categorical_cols[:10]])}
{'...' if len(categorical_cols) > 10 else ''}

📅 **DateTime Columns**: {len(eda_results.get('datetimeColumns', []))}

✅ **Validation Check**: All columns properly typed for ML processing."""
        
        return {"concise": concise, "detailed": detailed}
    
    # ============================================
    # MISSING VALUES - COMPREHENSIVE ANALYSIS
    # ============================================
    if any(kw in q_lower for kw in ['missing', 'null', 'empty', 'incomplete', 'nan', 'na value', 'blank']):
        if not missing_vals or all(v.get('count', 0) == 0 for v in missing_vals.values()):
            return {
                "concise": "✅ Perfect! No missing values detected. Dataset is 100% complete and ready for ML.",
                "detailed": "✅ **Perfect Data Completeness!**\n\nNo missing values detected in any column. Your dataset is 100% complete and ready for machine learning without imputation.",
                "code_section": "missing_values"  # Show missing values code anyway for reference
            }
            
        total_missing_cols = sum(1 for v in missing_vals.values() if v.get('count', 0) > 0)
        critical = {col: info for col, info in missing_vals.items() if info.get('percentage', 0) > 50}
        high = {col: info for col, info in missing_vals.items() if 20 < info.get('percentage', 0) <= 50}
        moderate = {col: info for col, info in missing_vals.items() if 5 < info.get('percentage', 0) <= 20}
        low = {col: info for col, info in missing_vals.items() if 0 < info.get('percentage', 0) <= 5}
        
        concise = f"⚠️ {total_missing_cols}/{len(missing_vals)} columns have missing data. Critical: {len(critical)}, High: {len(high)}, Moderate: {len(moderate)}, Low: {len(low)}."
        
        detailed = f"""⚠️ **Comprehensive Missing Data Validation**:

📊 **Summary**: {total_missing_cols} out of {len(missing_vals)} columns affected

"""
        if critical:
            detailed += f"""🔴 **CRITICAL - Drop Consider** (>50% missing):
{chr(10).join([f'• **{col}**: {info.get("percentage", 0):.1f}% missing ({info.get("count", 0):,}/{rows} values)' for col, info in list(critical.items())[:5]])}
**Recommendation**: Consider dropping these columns - too much missing data to impute reliably.

"""
        if high:
            detailed += f"""🟠 **HIGH PRIORITY** (20-50% missing):
{chr(10).join([f'• **{col}**: {info.get("percentage", 0):.1f}% missing ({info.get("count", 0):,} values)' for col, info in list(high.items())[:5]])}
**Recommendation**: Advanced imputation (KNN, iterative) or consider dropping.

"""
        if moderate:
            detailed += f"""🟡 **MODERATE** (5-20% missing):
{chr(10).join([f'• **{col}**: {info.get("percentage", 0):.1f}% missing ({info.get("count", 0):,} values)' for col, info in list(moderate.items())[:5]])}
**Recommendation**: Mean/median imputation for numeric, mode for categorical.

"""
        if low:
            detailed += f"""🟢 **LOW IMPACT** (<5% missing):
{chr(10).join([f'• **{col}**: {info.get("percentage", 0):.1f}% missing' for col, info in list(low.items())[:5]])}
**Recommendation**: Simple imputation will work perfectly.

"""
        
        detailed += """💡 **Imputation Strategies**:
• **Numeric**: Mean (normal dist), Median (skewed), KNN (correlated features)
• **Categorical**: Mode (most common), Forward/backward fill (time series)
• **Advanced**: Iterative imputation, predictive modeling

🔧 **Python**: Use `df.fillna()`, `SimpleImputer`, or `KNNImputer` from sklearn."""
        
        return {"concise": concise, "detailed": detailed, "code_section": "missing_values"}
    
    # ============================================
    # CORRELATION & RELATIONSHIPS
    # ============================================
    if any(kw in q_lower for kw in ['correlation', 'correlate', 'relationship', 'related', 'feature', 'multicollinearity', 'heatmap']):
        corr = eda_results.get('correlation', {})
        if not corr:
            return {
                "concise": "⚠️ Correlation not available - need at least 2 numeric columns.",
                "detailed": "⚠️ **Correlation Analysis Unavailable**: Requires at least 2 numeric features. Your dataset may be categorical-heavy."
            }
        
        very_strong = []
        strong = []
        moderate = []
        
        for col1, row in corr.items():
            for col2, val in row.items():
                if col1 < col2:  # Avoid duplicates
                    if abs(val) > 0.9:
                        very_strong.append((col1, col2, val))
                    elif abs(val) > 0.7:
                        strong.append((col1, col2, val))
                    elif abs(val) > 0.5:
                        moderate.append((col1, col2, val))
        
        concise = f"🔗 Correlations: {len(very_strong)} very strong (>0.9), {len(strong)} strong (0.7-0.9), {len(moderate)} moderate (0.5-0.7)."
        
        detailed = f"""🔗 **Feature Correlation Validation**:

📊 **Correlation Strength Distribution**:
• Very Strong (|r| > 0.9): {len(very_strong)} pairs
• Strong (0.7 < |r| ≤ 0.9): {len(strong)} pairs  
• Moderate (0.5 < |r| ≤ 0.7): {len(moderate)} pairs

"""
        if very_strong:
            detailed += f"""🔴 **MULTICOLLINEARITY ALERT** (|r| > 0.9):
{chr(10).join([f'• **{c1}** ↔ **{c2}**: {v:.3f}' for c1, c2, v in very_strong[:5]])}
⚠️ **ML Impact**: High multicollinearity! Consider:
   • Feature selection (keep one, drop redundant)
   • Principal Component Analysis (PCA) 
   • Regularization (Ridge/Lasso regression)

"""
        if strong:
            detailed += f"""🟡 **Strong Correlations** (0.7-0.9):
{chr(10).join([f'• **{c1}** ↔ **{c2}**: {v:.3f}' for c1, c2, v in strong[:5]])}
📊 **Note**: Features carry similar information. Monitor for redundancy.

"""
        if moderate:
            detailed += f"""🟢 **Moderate Correlations** (0.5-0.7):
{chr(10).join([f'• **{c1}** ↔ **{c2}**: {v:.3f}' for c1, c2, v in moderate[:5]])}
✅ **Good**: Useful relationships without excessive redundancy.

"""
        if not (very_strong or strong or moderate):
            detailed += """✅ **Independent Features**: No strong correlations (all |r| < 0.5)
🎯 **Excellent for ML**: Each feature provides unique information!

"""
        
        detailed += "💡 **Recommendation**: Review correlation heatmap on main page for visual analysis."
        
        return {"concise": concise, "detailed": detailed, "code_section": "correlation"}
    
    # ============================================
    # OUTLIERS & ANOMALIES
    # ============================================
    if any(kw in q_lower for kw in ['outlier', 'outliers', 'extreme', 'anomaly', 'anomalies', 'abnormal', 'unusual']):
        outlier_analysis = []
        
        for col, info in numeric_summary.items():
            outlier_count = info.get('outliers', 0)
            outlier_pct = info.get('outliersPercentage', 0)
            if outlier_count > 0:
                outlier_analysis.append((col, outlier_count, outlier_pct, info))
        
        if not outlier_analysis:
            return {
                "concise": "✅ No significant outliers detected using IQR method. Data distribution is healthy!",
                "detailed": "✅ **Clean Data Distribution**:\n\nNo significant outliers detected using IQR method (Q1-1.5×IQR, Q3+1.5×IQR). Your data looks healthy and normally distributed!"
            }
        
        severe = [(c, cnt, pct, i) for c, cnt, pct, i in outlier_analysis if pct > 10]
        moderate = [(c, cnt, pct, i) for c, cnt, pct, i in outlier_analysis if 3 < pct <= 10]
        minor = [(c, cnt, pct, i) for c, cnt, pct, i in outlier_analysis if pct <= 3]
        
        total_outliers = sum(cnt for _, cnt, _, _ in outlier_analysis)
        concise = f"🎯 {total_outliers:,} outliers in {len(outlier_analysis)} columns. Severe: {len(severe)}, Moderate: {len(moderate)}, Minor: {len(minor)}."
        
        detailed = f"""🎯 **Outlier Detection & Validation** (IQR Method):

📊 **Overall**: {total_outliers:,} outliers across {len(outlier_analysis)}/{len(numeric_cols)} numeric columns

"""
        if severe:
            detailed += f"""🔴 **SEVERE** (>10% outliers):
{chr(10).join([f'• **{col}**: {cnt:,} outliers ({pct:.1f}%) - Range: [{info.get("min"):.2f}, {info.get("max"):.2f}]' for col, cnt, pct, info in severe])}
**Action**: Investigate data collection issues. Consider robust algorithms or capping.

"""
        if moderate:
            detailed += f"""🟡 **MODERATE** (3-10% outliers):
{chr(10).join([f'• **{col}**: {cnt:,} outliers ({pct:.1f}%)' for col, cnt, pct, _ in moderate])}
**Options**: Winsorization, log transformation, or outlier-robust models.

"""
        if minor:
            detailed += f"""🟢 **MINOR** (≤3% outliers):
{chr(10).join([f'• **{col}**: {cnt:,} outliers ({pct:.1f}%)' for col, cnt, pct, _ in minor])}
**Status**: Normal for real-world data. Usually keep these values.

"""
        
        detailed += """💡 **ML Strategy**:
• **Keep**: Tree-based models (Random Forest, XGBoost) handle outliers well
• **Transform**: Log/sqrt transformation for positively skewed data
• **Cap**: Winsorization at 1st/99th percentile
• **Remove**: Only if proven data errors"""
        
        return {"concise": concise, "detailed": detailed}
    
    # ============================================
    # DATA QUALITY & VALIDATION
    # ============================================
    if any(kw in q_lower for kw in ['quality', 'clean', 'good', 'problems', 'issues', 'health', 'valid', 'check']):
        checks = eda_results.get('validationChecks', {})
        
        # Calculate comprehensive scores
        missing_pct = sum(v.get('percentage', 0) for v in missing_vals.values()) / max(1, cols)
        completeness_score = 100 - missing_pct
        size_score = min(100, (rows / 10000) * 100) if rows > 0 else 0
        balance_score = 80 if 0.2 <= len(numeric_cols)/max(1,cols) <= 0.8 else 50
        duplicate_pct = (eda_results.get('duplicates', 0) / max(1, rows)) * 100
        duplicate_score = 100 if duplicate_pct == 0 else max(0, 100 - duplicate_pct * 10)
        
        overall_score = (completeness_score * 0.35 + size_score * 0.25 + balance_score * 0.2 + duplicate_score * 0.2)
        
        grade = "A+ (Excellent)" if overall_score >= 90 else "A (Very Good)" if overall_score >= 80 else "B (Good)" if overall_score >= 70 else "C (Fair)" if overall_score >= 60 else "D (Needs Work)"
        
        concise = f"🔍 Data Quality: {overall_score:.0f}/100 ({grade}). Completeness: {completeness_score:.0f}%, Size: {size_score:.0f}%, Balance: {balance_score:.0f}%."
        
        detailed = f"""🔍 **Comprehensive Data Validation Report**:

📊 **Overall Quality Score: {overall_score:.1f}/100** - **{grade}**

**Detailed Breakdown**:

✅ **Data Completeness**: {completeness_score:.1f}/100
   • Missing Data: {missing_pct:.1f}% average
   • Status: {('Excellent' if completeness_score > 95 else 'Good' if completeness_score > 85 else 'Fair' if completeness_score > 70 else 'Poor')}

✅ **Sample Size**: {size_score:.1f}/100
   • Total Records: {rows:,}
   • Status: {('Excellent' if rows > 10000 else 'Good' if rows > 1000 else 'Adequate' if rows > 100 else 'Limited')}

✅ **Feature Balance**: {balance_score:.1f}/100
   • Numeric: {len(numeric_cols)} ({len(numeric_cols)/max(1,cols)*100:.0f}%)
   • Categorical: {len(categorical_cols)} ({len(categorical_cols)/max(1,cols)*100:.0f}%)
   • Status: {('Balanced' if 0.3 <= len(numeric_cols)/max(1,cols) <= 0.7 else 'Imbalanced')}

✅ **Data Integrity**: {duplicate_score:.1f}/100
   • Duplicates: {eda_results.get('duplicates', 0)} rows ({duplicate_pct:.2f}%)
   • Status: {('Clean' if duplicate_pct < 1 else 'Moderate' if duplicate_pct < 5 else 'High')}

🎯 **ML Readiness**: {('Ready for production' if overall_score >= 80 else 'Ready with preprocessing' if overall_score >= 60 else 'Significant work needed')}

💡 **Recommendations**:"""
        
        if overall_score >= 80:
            detailed += "\n• ✅ Excellent quality! Proceed with model training"
        else:
            if completeness_score < 85:
                detailed += "\n• ⚠️ Address missing values before modeling"
            if size_score < 50:
                detailed += "\n• ⚠️ Consider collecting more data"
            if balance_score < 60:
                detailed += "\n• ⚠️ Review feature engineering needs"
            if duplicate_score < 90:
                detailed += "\n• ⚠️ Remove duplicate records"
        
        return {"concise": concise, "detailed": detailed}
    
    # ============================================
    # STATISTICAL ANALYSIS & DISTRIBUTIONS
    # ============================================
    if any(kw in q_lower for kw in ['distribution', 'skew', 'skewness', 'kurtosis', 'normal', 'bell curve', 'gaussian', 'statistics', 'stats']):
        if not numeric_summary:
            return {
                "concise": "⚠️ No numeric columns for statistical analysis.",
                "detailed": "⚠️ **Statistical Analysis Unavailable**: Dataset has no numeric columns."
            }
        
        normal_dist = []
        left_skewed = []
        right_skewed = []
        heavy_tailed = []
        
        for col, info in numeric_summary.items():
            skew = info.get('skewness', 0)
            kurt = info.get('kurtosis', 0)
            is_normal = info.get('isNormal', False)
            
            if is_normal or abs(skew) < 0.5:
                normal_dist.append(col)
            elif skew < -0.5:
                left_skewed.append((col, skew))
            elif skew > 0.5:
                right_skewed.append((col, skew))
            
            if abs(kurt) > 3:
                heavy_tailed.append((col, kurt))
        
        concise = f"📈 Distributions: {len(normal_dist)} normal, {len(right_skewed)} right-skewed, {len(left_skewed)} left-skewed, {len(heavy_tailed)} heavy-tailed."
        
        detailed = f"""📈 **Statistical Distribution Analysis**:

📊 **Distribution Summary** ({len(numeric_summary)} numeric features):

"""
        if normal_dist:
            detailed += f"""✅ **Normal/Symmetric** ({len(normal_dist)}):
{chr(10).join([f'• {col}' for col in normal_dist[:8]])}
**ML Note**: Ready for linear models without transformation.

"""
        if right_skewed:
            detailed += f"""📊 **Right-Skewed** ({len(right_skewed)}):
{chr(10).join([f'• **{col}**: skew = {skew:.2f}' for col, skew in right_skewed[:5]])}
**Transform**: Log, square root, or Box-Cox transformation recommended.

"""
        if left_skewed:
            detailed += f"""📊 **Left-Skewed** ({len(left_skewed)}):
{chr(10).join([f'• **{col}**: skew = {skew:.2f}' for col, skew in left_skewed[:5]])}
**Transform**: Square or exponential transformation may help.

"""
        if heavy_tailed:
            detailed += f"""⚠️ **Heavy-Tailed** ({len(heavy_tailed)}):
{chr(10).join([f'• **{col}**: kurtosis = {kurt:.2f}' for col, kurt in heavy_tailed[:5]])}
**Note**: Outliers present. Consider robust methods.

"""
        
        detailed += """💡 **Recommendations**:
• Normal distributions work well with all models
• Skewed features benefit from transformation
• Tree-based models handle non-normal distributions naturally"""
        
        return {"concise": concise, "detailed": detailed, "code_section": "eda"}
    
    # ============================================
    # DUPLICATES VALIDATION
    # ============================================
    if any(kw in q_lower for kw in ['duplicate', 'duplicates', 'repeated', 'same', 'identical']):
        dup_count = eda_results.get('duplicates', 0)
        dup_pct = (dup_count / max(1, rows)) * 100
        
        if dup_count == 0:
            return {
                "concise": "✅ No duplicate rows found. Dataset has unique records only.",
                "detailed": "✅ **Perfect Data Integrity**: No duplicate rows detected. All records are unique!"
            }
        
        concise = f"⚠️ {dup_count:,} duplicate rows found ({dup_pct:.2f}% of dataset). Recommend removal before ML."
        
        detailed = f"""⚠️ **Duplicate Records Detected**:

📊 **Summary**:
• **Duplicate Rows**: {dup_count:,}
• **Percentage**: {dup_pct:.2f}%
• **Unique Rows**: {rows - dup_count:,}

{'🔴 **HIGH DUPLICATION** - Significant data quality issue!' if dup_pct > 5 else '🟡 **Moderate duplication** - Should be addressed' if dup_pct > 1 else '🟢 **Minor duplication** - Low impact but clean is better'}

💡 **Impact on ML**:
• Biases training data toward duplicated patterns
• Inflates dataset size artificially
• May cause overfitting
• Affects validation accuracy

🔧 **Removal Strategy**:
```python
# Remove duplicates
df_clean = df.drop_duplicates()

# Keep first occurrence
df_clean = df.drop_duplicates(keep='first')

# Remove based on specific columns
df_clean = df.drop_duplicates(subset=['col1', 'col2'])
```

✅ **Recommendation**: Remove duplicates before training models."""
        
        return {"concise": concise, "detailed": detailed, "code_section": "preprocessing"}
    
    # ============================================
    # ML MODEL RECOMMENDATIONS
    # ============================================
    if any(kw in q_lower for kw in ['model', 'algorithm', 'ml', 'machine learning', 'recommend', 'best', 'which', 'suitable']):
        task_type = eda_results.get('goal', {}).get('interpreted_task', 'Unknown')
        
        concise = f"🤖 For {rows:,} samples: "
        detailed = f"""🤖 **ML Model Recommendations**:

📊 **Dataset Profile**:
• Samples: {rows:,}
• Features: {len(numeric_cols)} numeric, {len(categorical_cols)} categorical
• Task: {task_type}

🎯 **Recommended Models**:

"""
        
        # Size-based recommendations
        if rows < 100:
            concise += "Simple models (Logistic Reg, Decision Tree). Dataset too small for complex models."
            detailed += """⚠️ **Small Dataset (<100 samples)**:
✅ **Use**: Logistic Regression, KNN, Simple Decision Trees
❌ **Avoid**: Deep Learning, Complex ensembles (overfitting risk)
💡 **Tip**: Consider data augmentation or collecting more samples"""
        elif rows < 1000:
            concise += "Random Forest, SVM, Gradient Boosting recommended."
            detailed += """📊 **Medium Dataset (100-1K samples)**:
✅ **Best**: Random Forest, Support Vector Machines, Gradient Boosting
✅ **Good**: Logistic/Linear Regression (with regularization)
❌ **Avoid**: Deep Neural Networks (insufficient data)
💡 **Tip**: Use cross-validation for reliable performance"""
        elif rows < 10000:
            concise += "XGBoost, LightGBM, Random Forest excellent choices."
            detailed += """🚀 **Large Dataset (1K-10K samples)**:
✅ **Excellent**: XGBoost, LightGBM, CatBoost
✅ **Good**: Random Forest, SVM, Neural Networks
✅ **Fast**: LightGBM (faster than XGBoost)
💡 **Tip**: Hyperparameter tuning will significantly improve performance"""
        else:
            concise += "XGBoost, Neural Networks, LightGBM - all advanced models suitable."
            detailed += """🔥 **Very Large Dataset (>10K samples)**:
✅ **Top Choice**: XGBoost, LightGBM (scalable & accurate)
✅ **Deep Learning**: Neural Networks, if features >50
✅ **Ensemble**: Stacking multiple models
💡 **Tip**: GPU acceleration recommended for neural networks"""
        
        # Feature-based recommendations
        detailed += "\n\n"
        if len(categorical_cols) > len(numeric_cols) * 2:
            detailed += """🏷️ **Categorical-Heavy Data**:
• **Perfect**: CatBoost, Tree-based models
• **Good**: Random Forest, XGBoost (with encoding)
• **Preprocessing**: OneHot encoding for linear models"""
        elif len(numeric_cols) > len(categorical_cols) * 2:
            detailed += """🔢 **Numeric-Heavy Data**:
• **Perfect**: Linear/Polynomial Regression, SVM
• **Good**: Any model works well
• **Preprocessing**: Feature scaling recommended"""
        else:
            detailed += """⚖️ **Balanced Features**:
• **Universal**: Random Forest, XGBoost
• **Flexible**: Works with most algorithms
• **Preprocessing**: Encode categorical, scale numeric"""
        
        detailed += "\n\n💡 **Pro Strategy**: Start with Random Forest baseline, then optimize with XGBoost!"
        
        return {"concise": concise, "detailed": detailed}
    
    # ============================================
    # PYTHON CODE REQUESTS - COMPREHENSIVE
    # ============================================
    if any(kw in q_lower for kw in ['python', 'code', 'implementation', 'script', 'pandas', 'show', 'how to', 'example']):
        # Detect specific section
        section_detected = None
        section_name = "Complete Python Code"
        
        # Comprehensive section detection
        if any(kw in q_lower for kw in ['missing', 'null', 'nan', 'imputation', 'fillna', 'impute']):
            section_detected = 'missing_values'
            section_name = "Missing Values Handling"
        elif any(kw in q_lower for kw in ['correlation', 'correlate', 'heatmap', 'corr', 'feature relationship']):
            section_detected = 'correlation'
            section_name = "Correlation Analysis"
        elif any(kw in q_lower for kw in ['visualiz', 'plot', 'chart', 'graph', 'histogram', 'boxplot', 'scatter']):
            section_detected = 'visualization'
            section_name = "Data Visualization"
        elif any(kw in q_lower for kw in ['model', 'train', 'fit', 'machine learning', 'ml', 'algorithm', 'predict']):
            section_detected = 'model_training'
            section_name = "Model Training"
        elif any(kw in q_lower for kw in ['preprocess', 'clean', 'encode', 'scale', 'transform', 'normalize']):
            section_detected = 'preprocessing'
            section_name = "Data Preprocessing"
        elif any(kw in q_lower for kw in ['eda', 'explore', 'exploration', 'exploratory', 'statistic']):
            section_detected = 'eda'
            section_name = "Exploratory Data Analysis"
        
        if section_detected:
            concise = f"💻 Displaying **{section_name}** code on main page. Full implementation ready!"
            detailed = f"💻 **{section_name} - Python Code**\n\nI'm showing the {section_name.lower()} implementation on the main page. Scroll up to see complete, production-ready code with comments and best practices!"
            
            return {"concise": concise, "detailed": detailed, "code_section": section_detected}
        else:
            concise = "💻 Displaying complete Python implementation. All sections included!"
            detailed = "💻 **Complete Python Implementation**\n\nShowing full end-to-end code on main page:\n• Data loading\n• EDA & statistics\n• Missing value handling\n• Visualization\n• Preprocessing\n• Model training\n• Evaluation\n\nScroll up to see everything!"
            
            return {"concise": concise, "detailed": detailed, "code_section": "full"}
    
    # ============================================
    # CARDINALITY & UNIQUENESS
    # ============================================
    if any(kw in q_lower for kw in ['cardinality', 'unique', 'distinct', 'values', 'categorical']):
        if not categorical_cols:
            return {
                "concise": "No categorical columns for cardinality analysis.",
                "detailed": "Dataset has no categorical columns. All features are numeric."
            }
            
        high_card = []
        low_card = []
        
        for col in categorical_cols:
            if col in eda_results.get('categoricalSummary', {}):
                info = eda_results['categoricalSummary'][col]
                unique_count = info.get('uniqueValues', 0)
                card_pct = (unique_count / rows) * 100
                
                if card_pct > 50:
                    high_card.append((col, unique_count, card_pct))
                else:
                    low_card.append((col, unique_count, card_pct))
        
        concise = f"🏷️ {len(high_card)} high-cardinality columns (may be IDs), {len(low_card)} suitable for encoding."
        
        detailed = f"""🏷️ **Cardinality Analysis**:

"""
        if high_card:
            detailed += f"""⚠️ **High Cardinality** (>50% unique):
{chr(10).join([f'• **{col}**: {cnt:,} unique values ({pct:.1f}%)' for col, cnt, pct in high_card])}
**Note**: Likely identifiers or text fields. Consider dropping or feature hashing.

"""
        if low_card:
            detailed += f"""✅ **Good Cardinality** (≤50% unique):
{chr(10).join([f'• **{col}**: {cnt} unique values ({pct:.1f}%)' for col, cnt, pct in low_card[:10]])}
**Status**: Perfect for one-hot or label encoding.

"""
        return {"concise": concise, "detailed": detailed}
    
    # ============================================
    # FEATURE IMPORTANCE & ENGINEERING
    # ============================================
    if any(kw in q_lower for kw in ['feature importance', 'important feature', 'feature engineering', 'feature selection', 'which feature']):
        concise = "🎯 Feature analysis available. Check correlation matrix and statistical summaries for importance indicators."
        
        detailed = """🎯 **Feature Importance & Engineering**:

📊 **Identifying Important Features**:
• **Correlation**: High correlation with target = important
• **Variance**: Low variance = less useful
• **Missing Data**: High missing % = consider dropping

💡 **Feature Engineering Ideas**:
• **Interaction**: Multiply related features
• **Polynomial**: Square/cube of numeric features
• **Binning**: Convert continuous to categorical
• **DateTime**: Extract day, month, year, day_of_week
• **Text**: Extract length, word count, special chars

🔧 **Feature Selection Methods**:
• SelectKBest for top K features
• Recursive Feature Elimination (RFE)
• L1 Regularization (Lasso) for automatic selection
• Tree-based feature importance (Random Forest)

📈 **Check main page** for correlation heatmap showing feature relationships!"""
        
        return {"concise": concise, "detailed": detailed, "code_section": "preprocessing"}
    
    # ============================================
    # DEFAULT INTELLIGENT RESPONSE
    # ============================================
    # Handle any other validation-related question intelligently
    concise = f"🤔 Based on your dataset ({rows:,} rows, {cols} cols): I can provide detailed analysis. Try asking about specific validation topics!"
    
    detailed = f"""🤔 **I'm Your Validation Agent - Ask Me Anything!**

📊 **Your Dataset**: {rows:,} rows × {cols} columns

💡 **What I Can Validate**:

**Data Quality:**
• "What's the data quality?" - Overall health score
• "Any missing values?" - Completeness analysis
• "Check for duplicates" - Integrity validation

**Statistical Analysis:**
• "Show correlations" - Feature relationships
• "Any outliers?" - Anomaly detection  
• "Check distributions" - Statistical properties
• "Data types?" - Column type validation

**ML Readiness:**
• "Best models?" - Algorithm recommendations
• "Feature importance?" - Key feature identification
• "Is it ML-ready?" - Comprehensive assessment

**Code & Implementation:**
• "Show Python code" - Full implementation
• "How to handle missing values?" - Specific code
• "Correlation code" - Visualization examples

🚀 **Try asking**: "What's the quality score?", "Show correlations", "Best ML model?", "Python code for preprocessing?"

📈 **Main page shows**: Complete validation results, statistics, and visualizations!"""
    
    return {"concise": concise, "detailed": detailed}
    
    # Dataset overview questions
    if any(keyword in q_lower for keyword in ['overview', 'summary', 'describe', 'tell me about', 'about']):
        shape = eda_results.get('shape', {})
        rows = shape.get('rows', 0)
        cols = shape.get('columns', 0)
        missing_pct = sum(v.get('percentage', 0) for v in eda_results.get('missingValues', {}).values()) / max(1, cols)
        
        quality_score = "Good" if missing_pct < 10 else "Fair" if missing_pct < 30 else "Needs attention"
        
        concise = f"📊 Your dataset has {rows:,} rows × {cols} columns. Data quality: {quality_score}. Check the main page for detailed analysis."
        
        detailed = f"""📊 **Comprehensive Dataset Overview**:

🔢 **Size & Structure**:
• **{rows:,} rows** × **{cols} columns**
• **{len(eda_results.get('numericColumns', []))} numeric** features for quantitative analysis
• **{len(eda_results.get('objectColumns', []))} categorical** features for classification

📈 **Data Quality Assessment**:
• **Missing Data**: {missing_pct:.1f}% average (Quality: {quality_score})
• **Data Integrity**: {eda_results.get('validationChecks', {}).get('dataQuality', 'Analyzing...')}
• **Memory Footprint**: ~{(rows * cols * 8 / 1024 / 1024):.1f} MB estimated

🎯 **ML Readiness**:
• Dataset size: {'✅ Sufficient' if rows > 1000 else '⚠️ Small sample'} for machine learning
• Feature variety: {'✅ Balanced' if 2 <= len(eda_results.get('numericColumns', [])) <= 50 else '⚠️ Review features'}
• **Recommendation**: {eda_results.get('goal', {}).get('description', 'Ready for analysis')}

💡 **Next Steps**: This dataset appears suitable for {'supervised learning' if len(eda_results.get('numericColumns', [])) > 0 else 'exploratory analysis'}."""
        
        return {"concise": concise, "detailed": detailed}
    
    # Missing data questions - enhanced analysis
    if any(keyword in q_lower for keyword in ['missing', 'null', 'empty', 'incomplete', 'nan']):
        missing_vals = eda_results.get('missingValues', {})
        if not missing_vals:
            return {
                "concise": "✅ No missing values detected. Your dataset is clean and ready for ML modeling.",
                "detailed": "✅ **Excellent Data Quality!** No missing values detected. Your dataset is clean and ready for ML modeling."
            }
            
        total_cols = len(missing_vals)
        high_missing = {col: info for col, info in missing_vals.items() if info.get('percentage', 0) > 20}
        moderate_missing = {col: info for col, info in missing_vals.items() if 5 < info.get('percentage', 0) <= 20}
        low_missing = {col: info for col, info in missing_vals.items() if 0 < info.get('percentage', 0) <= 5}
        
        concise = f"⚠️ Found missing values in {len(high_missing) + len(moderate_missing) + len(low_missing)}/{total_cols} columns. See main page for detailed analysis and recommendations."
        
        detailed = "⚠️ **Missing Data Analysis**:\n\n"
        
        if high_missing:
            detailed += "🔴 **Critical Issues** (>20% missing):\n"
            for col, info in list(high_missing.items())[:3]:
                detailed += f"• **{col}**: {info.get('percentage', 0):.1f}% missing ({info.get('count', 0):,} values)\n"
            detailed += "\n📋 **Action Required**: Consider dropping these columns or advanced imputation.\n\n"
        
        if moderate_missing:
            detailed += "🟡 **Moderate Issues** (5-20% missing):\n"
            for col, info in list(moderate_missing.items())[:3]:
                detailed += f"• **{col}**: {info.get('percentage', 0):.1f}% missing ({info.get('count', 0):,} values)\n"
            detailed += "\n📋 **Recommended**: Use mean/median imputation or predictive modeling.\n\n"
            
        if low_missing:
            detailed += "🟢 **Minor Issues** (<5% missing):\n"
            for col, info in list(low_missing.items())[:3]:
                detailed += f"• **{col}**: {info.get('percentage', 0):.1f}% missing ({info.get('count', 0):,} values)\n"
            detailed += "\n📋 **Solution**: Simple mean/mode imputation will work well.\n\n"
            
        detailed += f"**Overall**: {len(high_missing) + len(moderate_missing) + len(low_missing)}/{total_cols} columns affected. "
        detailed += "Use pandas `fillna()`, sklearn `SimpleImputer`, or advanced techniques like KNN imputation."
        
        return {"concise": concise, "detailed": detailed}
    
    # Correlation questions - enhanced with ML insights
    if any(keyword in q_lower for keyword in ['correlation', 'correlate', 'relationship', 'related', 'feature']):
        corr = eda_results.get('correlation', {})
        if not corr:
            return "📈 **Correlation Analysis**: Not available - requires at least 2 numeric columns. Upload dataset with numeric features for correlation insights."
        
        # Find various correlation strengths
        very_strong = []  # >0.9
        strong_corrs = []  # 0.7-0.9
        moderate_corrs = []  # 0.5-0.7
        
        for col1, row in corr.items():
            for col2, val in row.items():
                if col1 != col2 and abs(val) > 0.5:
                    if abs(val) > 0.9:
                        very_strong.append((col1, col2, val))
                    elif abs(val) > 0.7:
                        strong_corrs.append((col1, col2, val))
                    else:
                        moderate_corrs.append((col1, col2, val))
        
        result = "🔗 **Feature Correlation Analysis**:\n\n"
        
        if very_strong:
            result += "🔴 **Very Strong Correlations** (|r| > 0.9) - **Multicollinearity Risk**:\n"
            for col1, col2, val in very_strong[:3]:
                result += f"• **{col1}** ↔ **{col2}**: {val:.3f}\n"
            result += "⚠️ **ML Impact**: May cause overfitting. Consider feature selection/PCA.\n\n"
        
        if strong_corrs:
            result += "🟡 **Strong Correlations** (0.7 < |r| ≤ 0.9):\n"
            for col1, col2, val in strong_corrs[:3]:
                result += f"• **{col1}** ↔ **{col2}**: {val:.3f}\n"
            result += "📊 **ML Insight**: These features carry similar information. Monitor for redundancy.\n\n"
            
        if moderate_corrs:
            result += "🟢 **Moderate Correlations** (0.5 < |r| ≤ 0.7):\n"
            for col1, col2, val in moderate_corrs[:3]:
                result += f"• **{col1}** ↔ **{col2}**: {val:.3f}\n"
            result += "✅ **Good Balance**: Useful relationships without excessive redundancy.\n\n"
            
        if not (very_strong or strong_corrs or moderate_corrs):
            result += "✅ **Independent Features**: No strong correlations found (|r| > 0.5).\n"
            result += "🎯 **ML Advantage**: Features provide unique information - excellent for modeling!\n\n"
            
        detailed += "💡 **Recommendation**: Use correlation matrix for feature engineering and selection."
        
        concise = f"🔗 Found {len(very_strong)} very strong, {len(strong_corrs)} strong correlations. Check main page for detailed correlation matrix."
        return {"concise": concise, "detailed": detailed}
    
    # Outliers questions - comprehensive analysis
    if any(keyword in q_lower for keyword in ['outlier', 'outliers', 'extreme', 'anomaly', 'anomalies']):
        numeric_summary = eda_results.get('numericalSummary', {})
        outlier_analysis = []
        
        for col, info in numeric_summary.items():
            outlier_count = info.get('outliers', 0)
            outlier_pct = info.get('outliersPercentage', 0)
            if outlier_count > 0:
                outlier_analysis.append((col, outlier_count, outlier_pct, info))
        
        if not outlier_analysis:
            msg = "✅ No significant outliers detected. Your data distribution looks healthy!"
            return {"concise": msg, "detailed": "✅ **Clean Dataset**: No significant outliers detected using IQR method (Q1-1.5×IQR, Q3+1.5×IQR). Your data distribution looks healthy!"}
        
        result = "🎯 **Outlier Analysis Report**:\n\n"
        
        # Categorize by severity
        severe = [(col, count, pct, info) for col, count, pct, info in outlier_analysis if pct > 10]
        moderate = [(col, count, pct, info) for col, count, pct, info in outlier_analysis if 3 < pct <= 10]
        minor = [(col, count, pct, info) for col, count, pct, info in outlier_analysis if pct <= 3]
        
        if severe:
            result += "🔴 **High Outlier Concentration** (>10%):\n"
            for col, count, pct, info in severe:
                result += f"• **{col}**: {count:,} outliers ({pct:.1f}%) - Range: {info.get('min', 'N/A')} to {info.get('max', 'N/A')}\n"
            result += "⚠️ **Action**: Investigate data collection issues or consider robust algorithms.\n\n"
            
        if moderate:
            result += "🟡 **Moderate Outliers** (3-10%):\n"
            for col, count, pct, info in moderate:
                result += f"• **{col}**: {count:,} outliers ({pct:.1f}%)\n"
            result += "📊 **Options**: Cap values, transform data, or use outlier-robust models.\n\n"
            
        if minor:
            result += "🟢 **Minor Outliers** (≤3%):\n"
            for col, count, pct, info in minor:
                result += f"• **{col}**: {count:,} outliers ({pct:.1f}%)\n"
            result += "✅ **Normal**: Expected in real-world data. Monitor but likely keep.\n\n"
            
        detailed += "**ML Strategy**: Tree-based models handle outliers well. For linear models, consider preprocessing."
        
        total_outliers = sum(count for _, count, _, _ in outlier_analysis)
        concise = f"🎯 Detected {total_outliers:,} outliers across {len(outlier_analysis)} columns. See main page for detailed analysis."
        return {"concise": concise, "detailed": detailed}
    
    # Data quality questions - comprehensive assessment
    if any(keyword in q_lower for keyword in ['quality', 'clean', 'good', 'problems', 'issues', 'health']):
        checks = eda_results.get('validationChecks', {})
        shape = eda_results.get('shape', {})
        rows, cols = shape.get('rows', 0), shape.get('columns', 0)
        
        # Calculate comprehensive quality score
        missing_score = 100 - (sum(v.get('percentage', 0) for v in eda_results.get('missingValues', {}).values()) / max(1, cols))
        size_score = min(100, (rows / 1000) * 50) if rows > 0 else 0
        balance_score = 80 if 2 <= len(eda_results.get('numericColumns', [])) <= 50 else 40
        
        overall_score = (missing_score * 0.4 + size_score * 0.3 + balance_score * 0.3)
        
        result = f"🔍 **Comprehensive Data Quality Report**:\n\n"
        result += f"📊 **Overall Quality Score: {overall_score:.1f}/100**\n\n"
        
        result += "**Component Analysis**:\n"
        result += f"• **Data Completeness**: {missing_score:.1f}/100 {'✅' if missing_score > 80 else '⚠️' if missing_score > 60 else '❌'}\n"
        result += f"• **Sample Size**: {size_score:.1f}/100 {'✅' if size_score > 40 else '⚠️' if size_score > 20 else '❌'}\n"
        result += f"• **Feature Balance**: {balance_score:.1f}/100 {'✅' if balance_score > 60 else '⚠️'}\n\n"
        
        result += "**Detailed Assessment**:\n"
        result += f"• **Rows**: {rows:,} {'(Good size)' if rows > 1000 else '(Consider more data)' if rows > 100 else '(Small sample)'}\n"
        result += f"• **Features**: {cols} {'(Well-balanced)' if 5 <= cols <= 50 else '(Review feature count)'}\n"
        result += f"• **Data Types**: {len(eda_results.get('numericColumns', []))} numeric, {len(eda_results.get('objectColumns', []))} categorical\n"
        result += f"• **Missing Data**: {checks.get('missingDataLevel', 'Calculating...')}\n\n"
        
        if overall_score >= 80:
            result += "🎉 **Excellent Quality**: Dataset is ready for advanced ML modeling!"
            concise = f"🔍 Data quality score: {overall_score:.1f}/100 - Excellent! Ready for ML modeling."
        elif overall_score >= 60:
            result += "👍 **Good Quality**: Minor preprocessing needed before modeling."
            concise = f"🔍 Data quality score: {overall_score:.1f}/100 - Good. Minor preprocessing needed."
        else:
            result += "⚠️ **Needs Improvement**: Address data quality issues before modeling."
            concise = f"🔍 Data quality score: {overall_score:.1f}/100 - Needs improvement. Check main page."
            
        return {"concise": concise, "detailed": result}
            
    # Model recommendation questions
    if any(keyword in q_lower for keyword in ['model', 'algorithm', 'ml', 'machine learning', 'recommend']):
        numeric_count = len(eda_results.get('numericColumns', []))
        categorical_count = len(eda_results.get('objectColumns', []))
        rows = eda_results.get('shape', {}).get('rows', 0)
        
        result = "🤖 **ML Model Recommendations**:\n\n"
        
        if rows < 100:
            result += "⚠️ **Small Dataset** (<100 samples):\n• Simple models: Logistic Regression, Decision Trees\n• Avoid complex models to prevent overfitting\n\n"
        elif rows < 1000:
            result += "📊 **Medium Dataset** (100-1K samples):\n• **Best**: Random Forest, SVM, Gradient Boosting\n• **Avoid**: Deep Learning (insufficient data)\n\n"
        else:
            result += "🚀 **Large Dataset** (>1K samples):\n• **Excellent for**: XGBoost, LightGBM, Neural Networks\n• **Also consider**: Random Forest, SVM\n\n"
        
        if numeric_count > categorical_count * 3:
            result += "🔢 **Numeric-Heavy Data**:\n• **Ideal**: Linear/Polynomial Regression, SVM\n• **Tree-based**: Random Forest, XGBoost\n\n"
        elif categorical_count > numeric_count * 2:
            result += "🏷️ **Categorical-Heavy Data**:\n• **Perfect**: Decision Trees, Random Forest\n• **Preprocessing**: OneHot/Label encoding + Linear models\n\n"
        else:
            result += "⚖️ **Balanced Data Types**:\n• **Universal**: Random Forest, XGBoost\n• **Linear**: After proper encoding\n\n"
            
        detailed += "💡 **Pro Tip**: Start with Random Forest for baseline, then try XGBoost for optimization!"
        
        best_model = "XGBoost" if rows > 1000 else "Random Forest" if rows > 100 else "Logistic Regression"
        concise = f"🤖 Best model for your {rows:,} samples: {best_model}. See main page for complete recommendations."
        return {"concise": concise, "detailed": detailed}
        
    # Python code questions - detect specific sections
    if any(keyword in q_lower for keyword in ['python', 'code', 'implementation', 'script', 'pandas', 'show']):
        # Detect which specific code section user is asking about
        section_detected = None
        section_name = "Complete Python Code"
        
        # Check for specific sections
        if any(kw in q_lower for kw in ['missing', 'null', 'nan', 'imputation', 'fillna']):
            section_detected = 'missing_values'
            section_name = "Missing Values Handling"
        elif any(kw in q_lower for kw in ['correlation', 'correlate', 'heatmap', 'feature relationship']):
            section_detected = 'correlation'
            section_name = "Correlation Analysis"
        elif any(kw in q_lower for kw in ['visualization', 'visualize', 'plot', 'chart', 'graph', 'histogram']):
            section_detected = 'visualization'
            section_name = "Data Visualization"
        elif any(kw in q_lower for kw in ['model', 'train', 'machine learning', 'ml', 'algorithm', 'predict']):
            section_detected = 'model_training'
            section_name = "Model Training"
        elif any(kw in q_lower for kw in ['preprocess', 'preprocessing', 'clean', 'encode', 'scale']):
            section_detected = 'preprocessing'
            section_name = "Data Preprocessing"
        elif any(kw in q_lower for kw in ['eda', 'explore', 'exploration', 'exploratory']):
            section_detected = 'eda'
            section_name = "Exploratory Data Analysis"
        
        # Build response based on what section was detected
        if section_detected:
            concise = f"💻 Displaying **{section_name}** code on the main page. Scroll up to see the implementation!"
            detailed = f"""💻 **{section_name} - Python Implementation**

I'm displaying the relevant code section on the main page above. This includes:

"""
            # Add section-specific details
            if section_detected == 'missing_values':
                detailed += """✅ **Missing Values Code**:
• Detection of missing data
• Multiple imputation strategies (mean, median, mode)
• Forward/backward fill methods
• Dropping rows/columns with missing values
• Advanced techniques (KNN, iterative imputation)

📊 **You'll see**: Complete code for handling missing data in your dataset."""
            elif section_detected == 'correlation':
                detailed += """✅ **Correlation Analysis Code**:
• Correlation matrix calculation
• Correlation heatmap visualization
• High correlation pair detection
• Feature relationship analysis
• Multicollinearity identification

📊 **You'll see**: Full correlation analysis implementation with visualizations."""
            elif section_detected == 'visualization':
                detailed += """✅ **Visualization Code**:
• Distribution plots (histograms)
• Box plots for outlier detection
• Scatter plots for relationships
• Heatmaps for correlations
• Pair plots for multi-feature analysis

📊 **You'll see**: Complete visualization suite for your dataset."""
            elif section_detected == 'model_training':
                detailed += """✅ **Model Training Code**:
• Train-test data splitting
• Feature scaling & normalization
• Model selection and training
• Prediction and evaluation
• Performance metrics calculation
• Feature importance analysis

📊 **You'll see**: End-to-end ML model training pipeline."""
            elif section_detected == 'preprocessing':
                detailed += """✅ **Preprocessing Code**:
• Data cleaning and validation
• Missing value handling
• Feature encoding (label, one-hot)
• Feature scaling (standardization, normalization)
• Train-test split
• Data transformation pipeline

📊 **You'll see**: Complete preprocessing workflow for ML."""
            elif section_detected == 'eda':
                detailed += """✅ **EDA Code**:
• Dataset overview and structure
• Statistical summaries
• Data type analysis
• Distribution checks
• Initial quality assessment

📊 **You'll see**: Comprehensive exploratory data analysis code."""
            
            detailed += f"\n\n🚀 **Scroll to the code section on the main page** to see the full {section_name} implementation!"
            
            # Include section info for frontend to extract specific code
            return {
                "concise": concise, 
                "detailed": detailed,
                "code_section": section_detected  # Frontend can use this
            }
        else:
            # Show complete code
            concise = "💻 Displaying complete Python implementation on the main page. Scroll up to see all code!"
            detailed = """💻 **Complete Python Implementation**

I'm displaying the full Python code on the main page, including:

✅ **All Sections**:
• Library imports and setup
• Data loading and exploration
• Statistical analysis
• Missing value handling
• Correlation analysis
• Data visualization
• Preprocessing and cleaning
• Feature engineering
• ML model training
• Evaluation and metrics

📊 **Pro Tip**: Ask for specific sections:
• "show code for missing values"
• "show correlation code"
• "show visualization code"
• "show model training code"
• "show preprocessing code"

🚀 **Scroll to the code section** to see the complete implementation!"""
            
            return {"concise": concise, "detailed": detailed, "code_section": "full"}
        
    # General intelligent response for other questions
    rows = eda_results.get('shape', {}).get('rows', 0)
    cols = eda_results.get('shape', {}).get('columns', 0)
    
    concise = f"🤔 I analyzed your question. Your dataset has {rows:,} rows × {cols} cols. Check main page for detailed insights."
    
    detailed = f"""🤔 **Analyzing your question**: "{question}"

📊 **Based on your dataset**:
• **{rows:,} rows** × **{cols} features**
• **Data types**: {len(eda_results.get('numericColumns', []))} numeric, {len(eda_results.get('objectColumns', []))} categorical

💡 **I can help you with**:
• Dataset overview and quality assessment
• Missing data analysis and solutions
• Feature correlations and relationships
• Outlier detection and handling
• ML model recommendations
• Python code implementation

🔍 **Try asking**: "What's the data quality?", "Show correlations", "Any outliers?", "Best models?", "Python code?"

📈 **Pro tip**: Check the main validation page for comprehensive analysis results!"""
    
    return {"concise": concise, "detailed": detailed}
    
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

        response = _safe_chat_completion_call(
            messages=[
                {"role": "system", "content": "You are an expert data scientist providing actionable insights about datasets. Be concise, practical, and focus on what matters most for successful machine learning."},
                {"role": "user", "content": prompt}
            ],
            model=CHAT_MODEL,
            temperature=TEMPERATURE,
            max_completion_tokens=MAX_TOKENS,
            timeout=API_TIMEOUT
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

        response = _safe_chat_completion_call(
            messages=[
                {"role": "system", "content": "You are a senior ML engineer providing strategic guidance for machine learning projects. Focus on practical, implementable recommendations that drive success."},
                {"role": "user", "content": prompt}
            ],
            model=CHAT_MODEL,
            temperature=TEMPERATURE,
            max_completion_tokens=MAX_TOKENS,
            timeout=API_TIMEOUT
        )

        ai_insights = response.choices[0].message.content.strip()
        return f"🎯 **AI-Powered ML Strategy**:\n{ai_insights}"

    except Exception as e:
        return f"🤖 **ML AI Insights**: Error generating insights - {str(e)}. Using rule-based recommendations."


def perform_advanced_eda_from_csv_text(csv_text: str, goal: Dict[str, Any] = None) -> Dict[str, Any]:
    """Enhanced CSV/Excel analysis with intelligent insights and recommendations.

    Accepts either:
    - a CSV text string,
    - a bytes object containing an Excel file,
    - or a pandas DataFrame.
    """
    if goal is None:
        goal = {"type": "eda", "description": "Exploratory Data Analysis"}

    try:
        # If caller passed a DataFrame directly, use it
        if isinstance(csv_text, pd.DataFrame):
            df = csv_text.copy()
        # If raw bytes (e.g., uploaded Excel), try to read as Excel first
        elif isinstance(csv_text, (bytes, bytearray)):
            try:
                df = pd.read_excel(io.BytesIO(csv_text))
            except Exception:
                # Fallback: attempt to decode as UTF-8 CSV text
                try:
                    df = pd.read_csv(io.StringIO(csv_text.decode('utf-8', errors='ignore')))
                except Exception as e:
                    return {"error": f"Failed to parse bytes as Excel or CSV: {str(e)}", "isValid": False}
        else:
            # Assume string CSV
            df = pd.read_csv(io.StringIO(str(csv_text)))
    except Exception as e:
        return {"error": f"Failed to parse input as CSV/Excel: {str(e)}", "isValid": False}

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
        # Histogram/distribution (use 10 bins or 'auto' if enough samples)
        try:
            bins = 'auto' if len(series) > 50 else 10
            counts, bin_edges = np.histogram(series, bins=bins)
            distribution = {
                'bins': [round(float(b), 6) for b in bin_edges.tolist()],
                'counts': [int(c) for c in counts.tolist()]
            }
        except Exception:
            distribution = {'bins': [], 'counts': []}

        # canonical normal flag
        normal_flag = bool(is_normal)
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
            ,
            "distribution": distribution,
            "isNormal": normal_flag
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

    # Correlation pair ranking (top pairs by absolute correlation)
    correlation_pairs = []
    if correlation:
        seen = set()
        for c1, row in correlation.items():
            for c2, val in row.items():
                if c1 == c2: 
                    continue
                pair = tuple(sorted((c1, c2)))
                if pair in seen:
                    continue
                seen.add(pair)
                correlation_pairs.append({'pair': pair, 'value': float(val), 'abs': abs(float(val))})
        correlation_pairs = sorted(correlation_pairs, key=lambda x: x['abs'], reverse=True)

    correlation_insights = []
    for p in correlation_pairs[:10]:
        tag = 'strong' if p['abs'] > 0.7 else 'moderate' if p['abs'] > 0.4 else 'weak'
        correlation_insights.append({
            'columns': list(p['pair']),
            'correlation': round(p['value'], 4),
            'strength': tag
        })

    # Build concise numeric summary mapping for quick UI display
    summary = {}
    for col, info in numerical_summary.items():
        summary[col] = {
            'mean': info.get('mean'),
            'median': info.get('median'),
            'mode': info.get('mode'),
            'variance': info.get('variance'),
            'std': info.get('std'),
            'min': info.get('min'),
            'max': info.get('max'),
            'iqr': info.get('iqr'),
            'skewness': info.get('skewness'),
            'isNormal': info.get('isNormal'),
            'outliers': info.get('outliers'),
            'outliersPercentage': info.get('outliersPercentage')
        }

    # Dataset-level info
    try:
        dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
        memory_bytes = int(df.memory_usage(deep=True).sum())
    except Exception:
        dtypes = {col: str(df[col].dtype) for col in df.columns}
        memory_bytes = 0

    info = {
        'dtypes': dtypes,
        'memoryBytes': memory_bytes,
        'memoryReadable': f"{round(memory_bytes / 1024 / 1024, 3)} MB" if memory_bytes else 'N/A'
    }

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
        'correlationPairs': correlation_insights,
        'summary': summary,
        'info': info,
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
