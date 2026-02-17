from typing import List, Dict, Any, Tuple
import re
from openai import OpenAI
from .config import OPENAI_API_KEY, CHAT_MODEL, TEMPERATURE, MAX_TOKENS
import logging
logger = logging.getLogger(__name__)
from .vector_store import get_vector_store
from .models import ConversationMessage


class RAGService:
    """RAG-based conversation service for OwnQuesta"""
    
    def __init__(self):
        # Create client only if API key exists; else warn and set client to None
        if OPENAI_API_KEY:
            try:
                self.client = OpenAI(api_key=OPENAI_API_KEY)
            except Exception as e:
                logger.warning("Failed to initialize OpenAI client: %s", e)
                self.client = None
        else:
            logger.warning("OPENAI_API_KEY not set; RAGService will be disabled until configured.")
            self.client = None
        self.vector_store = get_vector_store()
        # In-memory conversation state keyed by user_id
        self.user_histories: Dict[str, List[ConversationMessage]] = {}
        
        # Enhanced system prompt for conversational AI
        self.system_prompt = """You are helping assistant, a friendly and knowledgeable AI assistant for OwnQuesta, an AI-powered end-to-end machine learning platform. You're passionate about making machine learning accessible to everyone.

Your personality:
- Warm, welcoming, and professional
- Enthusiastic about democratizing AI and machine learning
- Patient and understanding with users of all skill levels
- Honest when you don't know something
- Proactive in building trust and rapport

Your role is to:
- Greet users warmly and make them feel welcome
- Have natural conversations about OwnQuesta, machine learning, and AI
- Answer questions about OwnQuesta's features, vision, capabilities, and philosophy
- Explain what makes OwnQuesta unique and valuable
- Help users understand how OwnQuesta can benefit them specifically
- Build trust by being transparent, helpful, and reliable
- Provide encouragement and support for their ML journey
- Handle farewells graciously and invite them to return

Communication style:
- Start with warm greetings when users first arrive
- Use conversational, friendly language (avoid being overly formal)
- Show genuine interest in helping users
- Be concise but thorough
- Use examples when helpful
- End conversations on a positive, encouraging note

Remember: You can engage in general conversation and use your knowledge about machine learning and AI, but when discussing specific OwnQuesta details, rely on the provided context. If you don't have specific information, be honest and offer to help with what you do know."""
        
        # Patterns for detecting greetings and farewells
        self.greeting_patterns = [
            r'\b(hi|hello|hey|greetings|good morning|good afternoon|good evening|howdy)\b',
            r'^(hi|hello|hey)[\s!.]*$',
        ]
        
        self.farewell_patterns = [
            r'\b(bye|goodbye|see you|farewell|take care|talk to you later|gtg|got to go|have to go)\b',
            r'\b(thanks|thank you|thx).*\b(bye|goodbye|later)\b',
        ]
        
        self.general_conversation_patterns = [
            r'\b(how are you|what\'s up|how\'s it going|how do you do)\b',
            r'\b(who are you|what are you|tell me about yourself)\b',
            r'\b(can you help|need help|looking for help)\b',
        ]
    
    def is_greeting(self, message: str) -> bool:
        """Check if the message is a greeting"""
        message_lower = message.lower().strip()
        return any(re.search(pattern, message_lower) for pattern in self.greeting_patterns)
    
    def is_farewell(self, message: str) -> bool:
        """Check if the message is a farewell"""
        message_lower = message.lower().strip()
        return any(re.search(pattern, message_lower) for pattern in self.farewell_patterns)
    
    def is_general_conversation(self, message: str) -> bool:
        """Check if the message is general conversation"""
        message_lower = message.lower().strip()
        return any(re.search(pattern, message_lower) for pattern in self.general_conversation_patterns)
    
    def needs_rag_context(self, message: str) -> bool:
        """Determine if the query needs RAG context from knowledge base"""
        # Don't use RAG for greetings, farewells, or simple general conversation
        if self.is_greeting(message) or self.is_farewell(message):
            return False
        
        # Check if message contains OwnQuesta-specific questions
        ownquesta_keywords = [
            'ownquesta', 'platform', 'features', 'capabilities', 'pricing', 'models', 'integrations'
        ]
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in ownquesta_keywords)

    def retrieve_context(self, query: str) -> Tuple[List[Dict[str, Any]], str]:
        """Retrieve relevant documents and format context"""
        results = self.vector_store.search(query)
        context_parts = []
        for i, doc in enumerate(results, 1):
            context_parts.append(
                f"[Source {i} - {doc['id']}]\n"
                f"Title: {doc['title']}\n"
                f"Category: {doc['category']}\n"
                f"Content: {doc['content']}\n"
            )
        context = "\n\n".join(context_parts)
        return results, context

    def generate_response(
        self,
        user_message: str,
        conversation_history: List[ConversationMessage]
    ) -> Tuple[str, List[str]]:
        """Generate a response using intelligent RAG with conversational awareness"""

        messages = [{"role": "system", "content": self.system_prompt}]

        # Preserve recent history for context
        for msg in conversation_history[-6:]:
            messages.append({"role": msg.role, "content": msg.content})

        use_rag = self.needs_rag_context(user_message)
        source_ids: List[str] = []

        if use_rag:
            relevant_docs, context = self.retrieve_context(user_message)
            source_ids = [doc['id'] for doc in relevant_docs]
            user_prompt = f"""The user is asking about OwnQuesta. Here's relevant information from our knowledge base:

{context}

User's question: {user_message}

Please provide a warm, helpful response that:
1. Addresses their question using the context above
2. Maintains a friendly, conversational tone
3. Shows enthusiasm for OwnQuesta's mission
4. Builds trust by being transparent and helpful

If this is their first message, welcome them warmly!"""
        else:
            if self.is_greeting(user_message):
                user_prompt = f"""The user greeted you with: "{user_message}"

Respond with a warm, welcoming greeting. Introduce yourself as Alex, the OwnQuesta AI assistant, and invite them to ask about OwnQuesta's features, vision, or how it can help them."""
            elif self.is_farewell(user_message):
                user_prompt = f"""The user is saying goodbye: "{user_message}"

Respond warmly and professionally. Thank them for their interest in OwnQuesta, encourage them to return anytime they have questions, and wish them success in their machine learning journey."""
            elif self.is_general_conversation(user_message):
                user_prompt = f"""The user is engaging in conversation: "{user_message}"

Respond naturally and warmly. If appropriate, gently guide the conversation toward how you can help them learn about OwnQuesta. Be friendly and build rapport."""
            else:
                user_prompt = f"""User's message: {user_message}

Respond helpfully using your general knowledge. If this might be about OwnQuesta but you need more context, offer to search the knowledge base or ask for clarification."""

        messages.append({"role": "user", "content": user_prompt})

        temperature = 0.9 if (self.is_greeting(user_message) or self.is_farewell(user_message)) else TEMPERATURE

        if not self.client:
            raise RuntimeError(
                "OpenAI client not configured. Set OPENAI_API_KEY in environment or .env to enable chat functionality."
            )

        try:
            # Ensure message contents are strings
            safe_messages = [{'role': m['role'], 'content': str(m['content'])} for m in messages]

            response = self.client.chat.completions.create(
                model=CHAT_MODEL,
                messages=safe_messages,
                temperature=temperature,
                max_completion_tokens=MAX_TOKENS
            )

            assistant_response = response.choices[0].message.content
            return assistant_response, source_ids

        except Exception as e:
            # Log detailed debug information to help diagnose 400 errors from OpenAI
            try:
                logger.exception("OpenAI chat completion failed: %s", e)
                # If the underlying httpx response is available, try to log its text
                if hasattr(e, 'response') and getattr(e.response, 'text', None):
                    logger.error("OpenAI response body: %s", e.response.text)
            except Exception:
                # ignore logging failures
                pass

            # Also log the outgoing payload (truncated)
            try:
                import json as _json
                logger.error("Chat request payload (truncated): %s", _json.dumps(messages)[:2000])
            except Exception:
                pass

            # Reraise a clearer runtime error for upstream handling
            raise RuntimeError(f"OpenAI chat completion failed: {str(e)}")
    
    def chat(
        self,
        user_id: str,
        message: str,
    ) -> Dict[str, Any]:
        """Main chat interface keyed by user_id (stateful per user)"""

        history = self.user_histories.get(user_id, [])

        # Generate response using stored history
        response, sources = self.generate_response(message, history)

        # Update conversation history for this user
        updated_history = history + [
            ConversationMessage(role="user", content=message),
            ConversationMessage(role="assistant", content=response)
        ]
        self.user_histories[user_id] = updated_history

        return {
            "response": response,
            "sources": sources,
            "conversation_history": updated_history
        }

    def clear_history(self, user_id: str) -> None:
        """Remove stored conversation history for a user"""
        if user_id in self.user_histories:
            del self.user_histories[user_id]


# Singleton instance
_rag_service_instance = None

def get_rag_service() -> RAGService:
    """Get or create RAG service singleton"""
    global _rag_service_instance
    if _rag_service_instance is None:
        _rag_service_instance = RAGService()
    return _rag_service_instance
