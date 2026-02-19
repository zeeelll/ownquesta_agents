"""Configuration for Gen (explain/generate) agent."""
from pathlib import Path
from dotenv import load_dotenv
import os
import logging

project_root = Path(__file__).parent.parent
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)

LOG_LEVEL = os.getenv('GEN_AGENT_LOG_LEVEL', os.getenv('LOG_LEVEL', 'INFO'))
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

logger = logging.getLogger('gen_agent')
logger.setLevel(LOG_LEVEL)
