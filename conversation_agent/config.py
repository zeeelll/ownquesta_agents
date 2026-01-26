import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from project root
project_root = Path(__file__).parent.parent
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Validate API key is set
if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY is not set. Please ensure the .env file exists in the project root "
        f"({project_root}) with OPENAI_API_KEY=your-key-here, or set the environment variable."
    )

# Embedding Configuration
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# Vector Store Configuration
VECTOR_STORE_PATH = Path(__file__).parent.parent / "data" / "vector_store"
KNOWLEDGE_BASE_PATH = Path(__file__).parent.parent / "data" / "ownquesta_kb.json"

# RAG Configuration
TOP_K_RESULTS = 3  # Number of relevant documents to retrieve
TEMPERATURE = 0.3 # Model temperature for responses
MAX_TOKENS = 500  # Maximum tokens in response
