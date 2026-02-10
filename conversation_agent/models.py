from pydantic import BaseModel, Field
from typing import List


class ConversationMessage(BaseModel):
    """Model for a single conversation message"""
    role: str = Field(..., description="Role of the message sender (user or assistant)")
    content: str = Field(..., description="Content of the message")


class ConversationRequest(BaseModel):
    """Request model for conversation endpoint"""
    user_id: str = Field(..., description="Unique identifier for the user to track conversation history")
    message: str = Field(..., description="User's message/question about OwnQuesta")


class ConversationResponse(BaseModel):
    """Response model for conversation endpoint"""
    response: str = Field(..., description="Assistant's response")
    sources: List[str] = Field(
        default_factory=list,
        description="Source document IDs used to generate the response"
    )
    conversation_history: List[ConversationMessage] = Field(
        default_factory=list,
        description="Updated conversation history"
    )
