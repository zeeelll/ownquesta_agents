"""Configuration for MOCO (model comparison) agent."""
from pathlib import Path
from dotenv import load_dotenv
import os
import logging

project_root = Path(__file__).parent.parent
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)

LOG_LEVEL = os.getenv('MOCO_AGENT_LOG_LEVEL', os.getenv('LOG_LEVEL', 'INFO'))

logger = logging.getLogger('moco_agent')
logger.setLevel(LOG_LEVEL)
