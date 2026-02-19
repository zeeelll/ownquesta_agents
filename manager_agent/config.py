"""Configuration for Manager agent."""
from pathlib import Path
from dotenv import load_dotenv
import os
import logging

project_root = Path(__file__).parent.parent
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)

LOG_LEVEL = os.getenv('MANAGER_AGENT_LOG_LEVEL', os.getenv('LOG_LEVEL', 'INFO'))

logger = logging.getLogger('manager_agent')
logger.setLevel(LOG_LEVEL)
