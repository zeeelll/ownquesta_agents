from typing import TypedDict, Optional, Dict, List, Any
import pandas as pd

class AgentState(TypedDict):
    # Input
    goal: str
    file_path: str
    file_extension: str
    
    # Dataset info
    df: Optional[pd.DataFrame]
    rows: int
    columns: int
    file_size_mb: float
    column_types: Dict[str, str]
    missing_percent: Dict[str, float]
    
    # Goal understanding
    interpreted_task: str
    target_column_guess: Optional[str]
    goal_confidence: float
    
    # Validation
    validation_issues: List[str]
    satisfaction_score: int
    status: str  # PROCEED, PAUSE, REJECT
    clarification_questions: List[str]
    optional_questions: List[str]
    
    # Output
    agent_answer: str
    user_view_report: str
    
    # Error handling
    error: Optional[str]