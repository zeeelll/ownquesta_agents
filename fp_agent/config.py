"""Configuration for FP (feature-processing) agent."""
from pathlib import Path
from dotenv import load_dotenv
import os
import logging

project_root = Path(__file__).parent.parent
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)

LOG_LEVEL = os.getenv('FP_AGENT_LOG_LEVEL', os.getenv('LOG_LEVEL', 'INFO'))
MODEL_STORE = Path(os.getenv('MODEL_STORE', str(project_root / 'models')))
MODEL_STORE.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger('fp_agent')
logger.setLevel(LOG_LEVEL)
