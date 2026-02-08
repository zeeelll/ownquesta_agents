"""EDA agent package exports."""
from .config import create_agent, dataset_shape  # re-export common symbols

__all__ = ["create_agent", "dataset_shape"]
