# endpoint.py
# Ownquesta AI Agent — FastAPI Chat Endpoint
# Place this file in: backend/questa/endpoint.py

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, validator
from typing import List, Literal
import openai
import logging

from .config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    MAX_TOKENS,
    TEMPERATURE,
    MAX_HISTORY,
    SYSTEM_PROMPT,
    AGENT_NAME,
    AGENT_ROLE,
    WELCOME_MESSAGE,
    SUGGESTED_QUESTIONS,
    get_local_answer,
    get_relevant_context,
)

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
logger = logging.getLogger("questa")

router = APIRouter(
    prefix="/questa",
    tags=["Questa AI Agent"],
)

# Initialize OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)


# ─────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────
class ChatMessage(BaseModel):
    role: Literal["user", "assistant"] = Field(..., description="Who sent this message")
    content: str = Field(..., min_length=1, max_length=4000, description="Message content")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="User's current message")
    history: List[ChatMessage] = Field(default=[], description="Previous conversation messages")

    @validator("message")
    def message_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Message cannot be empty or whitespace")
        return v.strip()


class ChatResponse(BaseModel):
    reply: str = Field(..., description="Questa's response")
    agent: str = Field(default=AGENT_NAME)
    model: str = Field(default=OPENAI_MODEL)


class AgentInfoResponse(BaseModel):
    name: str
    role: str
    model: str
    welcome_message: str
    suggested_questions: List[str]
    status: str


class HealthResponse(BaseModel):
    status: str
    agent: str
    api_key_configured: bool


# ─────────────────────────────────────────────
# HELPER — BUILD MESSAGES FOR OPENAI
# ─────────────────────────────────────────────
def build_messages(user_message: str, history: List[ChatMessage], context: str) -> List[dict]:
    """
    Constructs the full message list for OpenAI:
    System prompt + recent conversation history + current user message.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if context:
        messages.append({"role": "system", "content": f"Relevant Ownquesta context:\n{context}"})

    # Keep only the last MAX_HISTORY messages for context window efficiency
    recent_history = history[-MAX_HISTORY:]
    for msg in recent_history:
        messages.append({"role": msg.role, "content": msg.content})

    messages.append({"role": "user", "content": user_message})
    return messages


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check for Questa agent",
)
async def health_check():
    """Check if the Questa agent is running and API key is configured."""
    return HealthResponse(
        status="ok",
        agent=AGENT_NAME,
        api_key_configured=bool(OPENAI_API_KEY),
    )


@router.get(
    "/info",
    response_model=AgentInfoResponse,
    summary="Get Questa agent metadata",
)
async def get_agent_info():
    """Returns agent name, role, model, welcome message, and suggested questions."""
    return AgentInfoResponse(
        name=AGENT_NAME,
        role=AGENT_ROLE,
        model=OPENAI_MODEL,
        welcome_message=WELCOME_MESSAGE,
        suggested_questions=SUGGESTED_QUESTIONS,
        status="active",
    )


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Send a message to Questa",
    status_code=status.HTTP_200_OK,
)
async def chat(request: ChatRequest):
    """
    Main chat endpoint. Accepts user message + conversation history.
    Returns Questa's intelligent reply powered by OpenAI.
    """
    local_reply = get_local_answer(request.message)
    if local_reply:
        return ChatResponse(
            reply=local_reply,
            agent=AGENT_NAME,
            model=OPENAI_MODEL,
        )

    # Guard: API key must be set
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY is not configured in .env")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="OpenAI API key is not configured. Please set OPENAI_API_KEY in your .env file.",
        )

    try:
        context = get_relevant_context(request.message)
        messages = build_messages(request.message, request.history, context)

        logger.info(f"[Questa] Sending request to OpenAI | model={OPENAI_MODEL} | history_length={len(request.history)}")

        completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )

        reply = completion.choices[0].message.content

        if not reply or not reply.strip():
            reply = "I'm sorry, I didn't quite catch that. Could you rephrase your question?"

        logger.info(f"[Questa] Response generated successfully | tokens_used={completion.usage.total_tokens}")

        return ChatResponse(
            reply=reply.strip(),
            agent=AGENT_NAME,
            model=OPENAI_MODEL,
        )

    except openai.AuthenticationError:
        logger.error("[Questa] Invalid OpenAI API key")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid OpenAI API key. Please check your .env configuration.",
        )

    except openai.RateLimitError:
        logger.warning("[Questa] OpenAI rate limit reached")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit reached. Please wait a moment and try again.",
        )

    except openai.BadRequestError as e:
        logger.error(f"[Questa] Bad request to OpenAI: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid request sent to AI model. Please try again.",
        )

    except openai.APIConnectionError:
        logger.error("[Questa] Could not connect to OpenAI API")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not connect to AI service. Please check your internet connection.",
        )

    except Exception as e:
        logger.error(f"[Questa] Unexpected error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred. Please try again.",
        )