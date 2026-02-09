import os
from pathlib import Path
from dotenv import load_dotenv
import logging

# Load environment variables from project root
project_root = Path(__file__).parent.parent
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)

# Setup logging
logger = logging.getLogger(__name__)

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Validate API key is set; only warn at import time to avoid hard failures during testing
if not OPENAI_API_KEY:
    try:
        # Prefer logging if available in host app
        logger.warning(
            "OPENAI_API_KEY is not set. Set it in %s or via environment variable for full functionality.",
            env_path,
        )
    except Exception:
        # Fallback to a simple print if logging isn't configured yet
        print(
            f"Warning: OPENAI_API_KEY is not set. Set it in {env_path} or via environment variable for full functionality."
        )

# OpenAI Model Configuration
OPENAI_CHAT_MODEL = os.getenv("ML_ASSISTANT_CHAT_MODEL", "gpt-4o-mini")
OPENAI_TEMPERATURE = float(os.getenv("ML_ASSISTANT_TEMPERATURE", "0.3"))
OPENAI_MAX_TOKENS = int(os.getenv("ML_ASSISTANT_MAX_TOKENS", "1000"))

# File Upload Configuration
UPLOAD_DIR = Path(__file__).parent / "uploads"
MAX_FILE_SIZE_MB = int(os.getenv("ML_ASSISTANT_MAX_FILE_SIZE_MB", "100"))  # MB
ALLOWED_EXTENSIONS = os.getenv("ML_ASSISTANT_ALLOWED_EXTENSIONS", "csv,xlsx,json").split(',')

# ML Processing Configuration
DEFAULT_TEST_SIZE = float(os.getenv("ML_ASSISTANT_DEFAULT_TEST_SIZE", "0.2"))
DEFAULT_RANDOM_STATE = int(os.getenv("ML_ASSISTANT_RANDOM_STATE", "42"))
MAX_CATEGORICAL_UNIQUE = int(os.getenv("ML_ASSISTANT_MAX_CATEGORICAL_UNIQUE", "20"))

# EDA Configuration
DEFAULT_HEAD_COUNT = int(os.getenv("ML_ASSISTANT_DEFAULT_HEAD_COUNT", "5"))
DEFAULT_DESCRIBE_PERCENTILES = [float(x) for x in os.getenv("ML_ASSISTANT_DESCRIBE_PERCENTILES", "0.25,0.5,0.75").split(',')]

# Model Training Configuration
LOGISTIC_REGRESSION_MAX_ITER = int(os.getenv("ML_ASSISTANT_LOGISTIC_MAX_ITER", "1000"))
MODEL_TIMEOUT_SECONDS = int(os.getenv("ML_ASSISTANT_MODEL_TIMEOUT", "300"))

# Validation Configuration
MIN_SAMPLES_FOR_VALIDATION = int(os.getenv("ML_ASSISTANT_MIN_SAMPLES", "10"))
MIN_FEATURES_FOR_VALIDATION = int(os.getenv("ML_ASSISTANT_MIN_FEATURES", "1"))

# Logging Configuration
LOG_LEVEL = os.getenv("ML_ASSISTANT_LOG_LEVEL", "INFO")
ENABLE_REQUEST_LOGGING = os.getenv("ML_ASSISTANT_ENABLE_REQUEST_LOGGING", "false").lower() == "true"

# Performance Configuration
ENABLE_CACHING = os.getenv("ML_ASSISTANT_ENABLE_CACHING", "true").lower() == "true"
CACHE_SIZE = int(os.getenv("ML_ASSISTANT_CACHE_SIZE", "100"))

# API Configuration
CORS_ORIGINS = os.getenv("ML_ASSISTANT_CORS_ORIGINS", "*").split(',')
REQUEST_TIMEOUT_SECONDS = int(os.getenv("ML_ASSISTANT_REQUEST_TIMEOUT", "30"))

# Create upload directory if it doesn't exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Validate configuration
def validate_config():
    """Validate configuration settings and log any issues"""
    issues = []
    
    if MAX_FILE_SIZE_MB <= 0:
        issues.append("MAX_FILE_SIZE_MB must be positive")
    
    if DEFAULT_TEST_SIZE <= 0 or DEFAULT_TEST_SIZE >= 1:
        issues.append("DEFAULT_TEST_SIZE must be between 0 and 1")
    
    if MIN_SAMPLES_FOR_VALIDATION < 2:
        issues.append("MIN_SAMPLES_FOR_VALIDATION must be at least 2")
    
    if OPENAI_TEMPERATURE < 0 or OPENAI_TEMPERATURE > 2:
        issues.append("OPENAI_TEMPERATURE must be between 0 and 2")
    
    if OPENAI_MAX_TOKENS < 1:
        issues.append("OPENAI_MAX_TOKENS must be positive")
    
    if issues:
        for issue in issues:
            logger.error(f"Configuration issue: {issue}")
        raise ValueError(f"Invalid configuration: {'; '.join(issues)}")
    
    logger.info("ML Assistant configuration validated successfully")

# Run validation on import
try:
    validate_config()
except Exception as e:
    logger.error(f"Configuration validation failed: {e}")
    # Don't raise during import to allow graceful degradation