import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from project root
project_root = Path(__file__).parent.parent
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Validate API key is set; only warn at import time to avoid hard failures during testing
if not OPENAI_API_KEY:
    try:
        # Prefer logging if available in host app
        import logging

        logging.getLogger(__name__).warning(
            "OPENAI_API_KEY is not set. Set it in %s or via environment variable for full functionality.",
            env_path,
        )
    except Exception:
        # Fallback to a simple print if logging isn't configured yet
        print(
            f"Warning: OPENAI_API_KEY is not set. Set it in {env_path} or via environment variable for full functionality."
        )

# Embedding Configuration
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-5-mini"

# Vector Store Configuration
VECTOR_STORE_PATH = Path(__file__).parent.parent / "data" / "vector_store"
KNOWLEDGE_BASE_PATH = Path(__file__).parent.parent / "data" / "ownquesta_kb.json"

# RAG Configuration
TOP_K_RESULTS = 3  # Number of relevant documents to retrieve
TEMPERATURE = 0.3 # Model temperature for responses
MAX_TOKENS = 500  # Maximum tokens in response
