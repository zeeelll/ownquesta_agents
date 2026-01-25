from fastapi import APIRouter, HTTPException, status
from .models import ConversationRequest, ConversationResponse
from .rag_service import get_rag_service
from .vector_store import get_vector_store

router = APIRouter()

@router.post("/chat", response_model=ConversationResponse)
async def chat_with_ownquesta(request: ConversationRequest):
    """
    Chat with OwnQuesta AI assistant about the platform
    
    This endpoint provides a RAG-based conversational interface where users can
    ask questions about OwnQuesta's features, vision, and capabilities.
    """
    try:
        # Get RAG service
        rag_service = get_rag_service()
        
        # Generate response
        result = rag_service.chat(
            user_id=request.user_id,
            message=request.message
        )
        
        return ConversationResponse(**result)
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing conversation: {str(e)}"
        )

@router.delete("/history/{user_id}")
async def delete_conversation_history(user_id: str):
    """Delete all stored conversation history for a given user_id"""
    try:
        rag_service = get_rag_service()
        rag_service.clear_history(user_id)
        return {"status": "success", "message": f"History cleared for user_id={user_id}"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error clearing history: {str(e)}"
        )