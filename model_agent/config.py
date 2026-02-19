"""Configuration for Model agent."""
from pathlib import Path
from dotenv import load_dotenv
import os
import logging
import pickle

project_root = Path(__file__).parent.parent
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)

LOG_LEVEL = os.getenv('MODEL_AGENT_LOG_LEVEL', os.getenv('LOG_LEVEL', 'INFO'))
MODEL_STORE = Path(os.getenv('MODEL_STORE', str(project_root / 'models')))
MODEL_STORE.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger('model_agent')
logger.setLevel(LOG_LEVEL)


def save_model_artifact(obj, name: str) -> str:
    """Save a pickled model artifact and return the path."""
    path = MODEL_STORE / f"{name}.pkl"
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    return str(path)
