import pandas as pd
import os
from typing import Dict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from .config import settings
from .state import AgentState

# Initialize OpenAI client if key is available
llm = None
if settings.has_openai_key:
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        openai_api_key=settings.OPENAI_API_KEY
    )

def load_dataset(state: AgentState) -> Dict:
    """Load CSV or XLSX file into DataFrame"""
    try:
        file_path = state["file_path"]
        ext = state["file_extension"]
        
        if ext == ".csv":
            df = pd.read_csv(file_path)
        elif ext in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path)
        else:
            return {
                "error": f"Unsupported file type: {ext}",
                "df": None
            }
        
        if df.empty:
            return {
                "error": "Dataset is empty",
                "df": None
            }
        
        return {"df": df, "error": None}
    
    except Exception as e:
        return {
            "error": f"Failed to load dataset: {str(e)}",
            "df": None
        }

def profile_dataset(state: AgentState) -> Dict:
    """Analyze dataset characteristics"""
    df = state["df"]
    
    if df is None:
        return {}
    
    # File size
    file_size_mb = os.path.getsize(state["file_path"]) / (1024 * 1024)
    
    # Column types
    column_types = {}
    for col in df.columns:
        dtype = str(df[col].dtype)
        if dtype.startswith('int'):
            column_types[col] = 'int'
        elif dtype.startswith('float'):
            column_types[col] = 'float'
        elif dtype == 'object':
            # Check if it's categorical
            if df[col].nunique() / len(df) < 0.5:
                column_types[col] = 'category'
            else:
                column_types[col] = 'string'
        elif dtype == 'bool':
            column_types[col] = 'boolean'
        elif 'datetime' in dtype:
            column_types[col] = 'datetime'
        else:
            column_types[col] = dtype
    
    # Missing percentages
    missing_percent = {}
    for col in df.columns:
        missing_pct = (df[col].isna().sum() / len(df)) * 100
        if missing_pct > 0:
            missing_percent[col] = round(missing_pct, 2)
    
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "file_size_mb": round(file_size_mb, 2),
        "column_types": column_types,
        "missing_percent": missing_percent
    }

def understand_goal(state: AgentState) -> Dict:
    """Understand user goal using LLM or rule-based heuristics"""
    goal = state["goal"].lower()
    df = state["df"]
    column_types = state.get("column_types", {})
    
    # Try LLM first if available
    if llm:
        try:
            system_prompt = """You are an ML task classifier. Analyze the user's goal and dataset columns to determine:
1. Task type: classification, regression, clustering, or unknown
2. Target column name (if applicable)
3. Confidence score (0.0 to 1.0)

Respond in JSON format:
{
  "task": "classification|regression|clustering|unknown",
  "target": "column_name or null",
  "confidence": 0.85
}"""
            
            columns_info = "\n".join([f"- {col}: {dtype}" for col, dtype in column_types.items()])
            user_prompt = f"""Goal: {state['goal']}

Dataset columns:
{columns_info}

Analyze this goal and determine the ML task."""
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = llm.invoke(messages)
            content = response.content.strip()
            
            # Parse JSON response
            import json
            if content.startswith("```json"):
                content = content.split("```json")[1].split("```")[0].strip()
            elif content.startswith("```"):
                content = content.split("```")[1].split("```")[0].strip()
            
            result = json.loads(content)
            
            return {
                "interpreted_task": result.get("task", "unknown"),
                "target_column_guess": result.get("target"),
                "goal_confidence": float(result.get("confidence", 0.5))
            }
        
        except Exception as e:
            print(f"LLM goal understanding failed: {e}, falling back to heuristics")
    
    # Fallback to rule-based heuristics
    interpreted_task = "unknown"
    target_guess = None
    confidence = 0.6
    
    # Classification keywords
    if any(kw in goal for kw in ["classify", "classification", "predict", "churn", "fraud", "category"]):
        interpreted_task = "classification"
        confidence = 0.75
        # Look for categorical target
        for col, dtype in column_types.items():
            if dtype in ['category', 'boolean'] and col.lower() in goal:
                target_guess = col
                confidence = 0.85
                break
    
    # Regression keywords
    elif any(kw in goal for kw in ["regression", "forecast", "predict", "sales", "revenue", "price"]):
        interpreted_task = "regression"
        confidence = 0.75
        # Look for numeric target
        for col, dtype in column_types.items():
            if dtype in ['int', 'float'] and col.lower() in goal:
                target_guess = col
                confidence = 0.85
                break
    
    # Clustering keywords
    elif any(kw in goal for kw in ["cluster", "segment", "group"]):
        interpreted_task = "clustering"
        confidence = 0.8
        target_guess = None  # Clustering is unsupervised
    
    return {
        "interpreted_task": interpreted_task,
        "target_column_guess": target_guess,
        "goal_confidence": confidence
    }

def validate_alignment(state: AgentState) -> Dict:
    """Validate dataset-goal alignment and identify issues"""
    df = state["df"]
    target = state.get("target_column_guess")
    task = state.get("interpreted_task")
    column_types = state.get("column_types", {})
    missing_percent = state.get("missing_percent", {})
    
    issues = []
    blocking_questions = []
    optional_questions = []
    
    # Check dataset size
    if state["rows"] < settings.MIN_ROWS_FOR_ML:
        issues.append(f"Dataset too small ({state['rows']} rows). Minimum {settings.MIN_ROWS_FOR_ML} rows recommended.")
    
    # Check target column
    if task in ["classification", "regression"]:
        if not target:
            issues.append("Could not identify target column from goal statement.")
            blocking_questions.append("Which column should be used as the target variable for prediction?")
        elif target not in df.columns:
            issues.append(f"Suggested target column '{target}' not found in dataset.")
            blocking_questions.append(f"The target column '{target}' was not found. Which column should be the target?")
        else:
            # Check target missing values
            if target in missing_percent and missing_percent[target] > 10:
                issues.append(f"Target column '{target}' has {missing_percent[target]}% missing values.")
            
            # Check task-type alignment
            target_type = column_types.get(target, "unknown")
            if task == "classification" and target_type not in ["category", "boolean", "string"]:
                issues.append(f"Classification task but target '{target}' appears to be numeric ({target_type}).")
            elif task == "regression" and target_type not in ["int", "float"]:
                issues.append(f"Regression task but target '{target}' appears to be categorical ({target_type}).")
            
            # Check class imbalance for classification
            if task == "classification" and target_type in ["category", "boolean"]:
                value_counts = df[target].value_counts()
                if len(value_counts) > 0:
                    imbalance_ratio = value_counts.max() / value_counts.min() if value_counts.min() > 0 else float('inf')
                    if imbalance_ratio > 10:
                        issues.append(f"Target column '{target}' is highly imbalanced (ratio: {imbalance_ratio:.1f}:1).")
    
    # Check excessive missing data
    high_missing_cols = [col for col, pct in missing_percent.items() if pct > settings.MAX_MISSING_PERCENT]
    if high_missing_cols:
        issues.append(f"{len(high_missing_cols)} columns have >50% missing data: {', '.join(high_missing_cols[:3])}{'...' if len(high_missing_cols) > 3 else ''}")
    
    # Check if goal is too ambiguous
    if state["goal_confidence"] < 0.5:
        issues.append("Goal statement is ambiguous or unclear.")
        blocking_questions.append("Could you clarify what you want to predict or analyze with this dataset?")
    
    # OPTIONAL questions (don't block PROCEED status)
    goal_lower = state["goal"].lower()
    
    # Time-based optional questions for forecasting/churn
    if "churn" in goal_lower or "forecast" in goal_lower or "time" in goal_lower:
        if "datetime" not in column_types.values():
            optional_questions.append("What is the time horizon for your prediction? (e.g., next 30 days, next quarter)")
    
    # Feature engineering suggestions
    if len(df.columns) > 20:
        optional_questions.append("Would you like recommendations on which features are most important?")
    
    return {
        "validation_issues": issues,
        "clarification_questions": blocking_questions,
        "optional_questions": optional_questions
    }

def compute_score(state: AgentState) -> Dict:
    """Compute satisfaction score based on validation results"""
    issues = state.get("validation_issues", [])
    confidence = state.get("goal_confidence", 0.5)
    
    # Start with base score
    score = 100
    
    # Deduct for each issue
    for issue in issues:
        if "too small" in issue.lower():
            score -= 30
        elif "not found" in issue.lower() or "could not identify" in issue.lower():
            score -= 20
        elif "missing" in issue.lower():
            if "target" in issue.lower():
                score -= 15
            else:
                score -= 10
        elif "imbalanced" in issue.lower():
            score -= 8
        elif "ambiguous" in issue.lower():
            score -= 15
        else:
            score -= 5
    
    # Factor in goal confidence
    score = int(score * (0.5 + 0.5 * confidence))
    
    # Clamp to 0-100
    score = max(0, min(100, score))
    
    return {"satisfaction_score": score}

def decide_outcome(state: AgentState) -> Dict:
    """Decide final status based on score and issues"""
    score = state["satisfaction_score"]
    issues = state.get("validation_issues", [])
    blocking_questions = state.get("clarification_questions", [])
    
    # Blocking issues that prevent ML
    blocking_keywords = ["too small", "not found", "could not identify"]
    has_blocking = any(any(kw in issue.lower() for kw in blocking_keywords) for issue in issues)
    
    # Decision logic
    if score >= 90 and not has_blocking and len(blocking_questions) == 0:
        status = "PROCEED"
    elif score < 70 or has_blocking or len(blocking_questions) > 0:
        status = "PAUSE" if score >= 70 and not has_blocking else "REJECT"
    else:
        status = "PAUSE"
    
    return {"status": status}

def generate_agent_answer(state: AgentState) -> Dict:
    """Generate formatted agent answer using LLM or template"""
    status = state["status"]
    score = state["satisfaction_score"]
    issues = state.get("validation_issues", [])
    blocking_questions = state.get("clarification_questions", [])
    optional_questions = state.get("optional_questions", [])
    task = state.get("interpreted_task", "unknown")
    target = state.get("target_column_guess")
    
    # Try LLM first
    if llm:
        try:
            system_prompt = """You are an ML assistant. Generate a concise, friendly response (2-3 sentences) about the validation result.
Use markdown formatting with **bold** for emphasis and \n for line breaks.
Be encouraging but honest about issues.
For PROCEED status, be clear they can move to next phase.
For optional questions, mention them as suggestions, not blockers."""
            
            user_prompt = f"""Status: {status}
Score: {score}/100
Task: {task}
Target: {target or 'unknown'}
Issues: {'; '.join(issues) if issues else 'None'}
Blocking Questions: {'; '.join(blocking_questions) if blocking_questions else 'None'}
Optional Questions: {'; '.join(optional_questions) if optional_questions else 'None'}

Generate a brief, friendly response."""
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = llm.invoke(messages)
            answer = response.content.strip()
            
            return {"agent_answer": answer}
        
        except Exception as e:
            print(f"LLM answer generation failed: {e}, falling back to template")
    
    # Fallback template
    if status == "PROCEED":
        answer = f"**Validation complete ({score}/100)** ✅ Your dataset is clean and fully suitable for **{task.title()}**.\n\n"
        answer += f"The target column **{target}** has been identified with high confidence. "
        answer += "You can proceed to the next step (Feature Understanding / Model Recommendation)."
        if optional_questions:
            answer += f"\n\n**Optional:** {optional_questions[0]}"
    
    elif status == "PAUSE":
        answer = f"**Almost ready!** Your dataset shows promise for {task} (score: {score}/100), but I need some clarification.\n\n"
        if blocking_questions:
            answer += f"**Question:** {blocking_questions[0]}"
        elif issues:
            answer += f"**Issue found:** {issues[0]}\n\nPlease address this to proceed."
    
    else:  # REJECT
        answer = f"**I found some significant issues** that need to be addressed before proceeding (score: {score}/100).\n\n"
        if issues:
            answer += f"**Main concern:** {issues[0]}\n\n"
        answer += "Please review the detailed report below and update your dataset or goal."
    
    return {"agent_answer": answer}

def build_response(state: AgentState) -> Dict:
    """Build detailed user-facing report"""
    status = state["status"]
    score = state["satisfaction_score"]
    issues = state.get("validation_issues", [])
    blocking_questions = state.get("clarification_questions", [])
    optional_questions = state.get("optional_questions", [])
    task = state.get("interpreted_task", "unknown")
    target = state.get("target_column_guess")
    
    # Status emoji
    status_emoji = "✅" if status == "PROCEED" else ("⚠️" if status == "PAUSE" else "❌")
    
    report = f"# Validation Report\n\n"
    report += f"## Overall Assessment\n"
    report += f"Status: **{status}** {status_emoji}\n"
    report += f"Satisfaction Score: **{score}/100**\n\n"
    
    report += f"## Goal Understanding\n"
    report += f"- Interpreted Task: **{task.title()}**\n"
    report += f"- Target Column: **{target or 'Not identified'}**\n"
    report += f"- Confidence: **{int(state.get('goal_confidence', 0) * 100)}%**\n\n"
    
    report += f"## Dataset Profile\n"
    report += f"- Rows: **{state['rows']:,}**\n"
    report += f"- Columns: **{state['columns']}**\n"
    report += f"- File Size: **{state['file_size_mb']} MB**\n\n"
    
    if issues:
        report += f"## Issues Detected\n"
        for i, issue in enumerate(issues, 1):
            report += f"{i}. {issue}\n"
        report += "\n"
    
    if blocking_questions:
        report += f"## Clarification Needed (Blocking)\n"
        for i, q in enumerate(blocking_questions, 1):
            report += f"{i}. {q}\n"
        report += "\n"
    
    if optional_questions:
        report += f"## Optional Questions\n"
        for i, q in enumerate(optional_questions, 1):
            report += f"{i}. {q}\n"
        report += "\n"
    
    if status == "PROCEED":
        report += f"## Next Steps\n"
        report += f"Your dataset is ready for ML model training. You may proceed to the next stage (Feature Engineering / Model Selection).\n"
        if optional_questions:
            report += f"\n**Note:** The optional questions above can help improve model performance but are not required to proceed.\n"
    elif status == "PAUSE":
        report += f"## Next Steps\n"
        report += f"Please answer the clarification questions above to help me better understand your requirements.\n"
    else:
        report += f"## Recommendations\n"
        report += f"Address the critical issues listed above before proceeding. Consider:\n"
        report += f"- Collecting more data if dataset is too small\n"
        report += f"- Clarifying your prediction goal\n"
        report += f"- Checking data quality and completeness\n"
    
    return {"user_view_report": report}