from typing import Dict, List, Optional, Literal
from pydantic import BaseModel, Field

class DatasetSummary(BaseModel):
    rows: int
    columns: int
    file_size_mb: float
    column_types: Dict[str, str]
    missing_percent: Dict[str, float]

class GoalUnderstanding(BaseModel):
    interpreted_task: Literal["classification", "regression", "clustering", "unknown"]
    target_column_guess: Optional[str]
    confidence: float = Field(ge=0.0, le=1.0)

class ValidationResponse(BaseModel):
    status: Literal["PROCEED", "PAUSE", "REJECT"]
    satisfaction_score: int = Field(ge=0, le=100)
    dataset_summary: DatasetSummary
    goal_understanding: GoalUnderstanding
    clarification_questions: List[str]
    optional_questions: List[str]
    agent_answer: str
    user_view_report: str