import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Embedding Configuration
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# Vector Store Configuration
VECTOR_STORE_PATH = Path(__file__).parent.parent / "data" / "vector_store"
KNOWLEDGE_BASE_PATH = Path(__file__).parent.parent / "data" / "ownquesta_kb.json"

# RAG Configuration
TOP_K_RESULTS = 3  # Number of relevant documents to retrieve
TEMPERATURE = 0.7  # Model temperature for responses
MAX_TOKENS = 500  # Maximum tokens in response
