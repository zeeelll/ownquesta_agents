"""
Centralized AI model registry for the lab-agent server.

To add a new model:
  1. Add its API key to .env  (e.g. MY_PROVIDER_API_KEY=sk-...)
  2. Add one entry to MODEL_REGISTRY below
  3. Done — all endpoints pick it up automatically via resolve_model()
"""

from __future__ import annotations
import os
from typing import Optional, TypedDict


# ── Model definition ──────────────────────────────────────────────────────────

class ModelConfig(TypedDict):
    display_name : str          # Human-readable label shown in the UI
    short_name   : str          # Short label for tight spaces
    provider     : str          # "openai" | "anthropic"
    model_name   : str          # Exact identifier sent to the provider API
    api_key_env  : str          # Name of the .env variable that enables this model
    free_quota   : Optional[int]# None = unlimited, int = max free sessions


# ── Registry: add new models here ─────────────────────────────────────────────

MODEL_REGISTRY: dict[str, ModelConfig] = {
    "gpt-4o-mini": {
        "display_name": "GPT-4o Mini",
        "short_name":   "GPT-4o-mini",
        "provider":     "openai",
        "model_name":   "gpt-4o-mini",
        "api_key_env":  "OPENAI_API_KEY",
        "free_quota":   None,           # unlimited
    },
    "gpt-4": {
        "display_name": "GPT-4",
        "short_name":   "GPT-4",
        "provider":     "openai",
        "model_name":   "gpt-4",
        "api_key_env":  "OPENAI_API_KEY",
        "free_quota":   None,           # unlimited
    },
    "gpt-5": {
        "display_name": "GPT-5",
        "short_name":   "GPT-5",
        "provider":     "openai",
        "model_name":   "gpt-5",
        "api_key_env":  "OPENAI_API_KEY",
        "free_quota":   None,           # unlimited
    },
    "claude-opus-4-6": {
        "display_name": "Claude Opus 4.6",
        "short_name":   "Opus 4.6",
        "provider":     "anthropic",
        "model_name":   "claude-opus-4-6",
        "api_key_env":  "ANTHROPIC_API_KEY",
        "free_quota":   1,
    },
    "claude-sonnet-4-6": {
        "display_name": "Claude Sonnet 4.6",
        "short_name":   "Sonnet 4.6",
        "provider":     "anthropic",
        "model_name":   "claude-sonnet-4-6",
        "api_key_env":  "ANTHROPIC_API_KEY",
        "free_quota":   1,
    },
    "claude-sonnet-4-5": {
        "display_name": "Claude Sonnet 4.5",
        "short_name":   "Sonnet 4.5",
        "provider":     "anthropic",
        "model_name":   "claude-sonnet-4-5-20251001",
        "api_key_env":  "ANTHROPIC_API_KEY",
        "free_quota":   1,
    },
    "claude-haiku-4-5": {
        "display_name": "Claude Haiku 4.5",
        "short_name":   "Haiku 4.5",
        "provider":     "anthropic",
        "model_name":   "claude-haiku-4-5-20251001",
        "api_key_env":  "ANTHROPIC_API_KEY",
        "free_quota":   3,
    },
    "gpt-5-codex": {
        "display_name": "GPT-5 Codex",
        "short_name":   "Codex",
        "provider":     "openai",
        "model_name":   "gpt-5-codex",
        "api_key_env":  "OPENAI_CODEX_API_KEY",
        "free_quota":   1,              # 1 free session
    },
}

DEFAULT_MODEL_ID = "gpt-4o-mini"


# ── Public helpers ────────────────────────────────────────────────────────────

def get_available_models() -> list[dict]:
    """Return metadata for every model whose API key is present in the env."""
    result = []
    for model_id, cfg in MODEL_REGISTRY.items():
        if os.getenv(cfg["api_key_env"], "").strip():
            result.append({
                "id":           model_id,
                "display_name": cfg["display_name"],
                "short_name":   cfg["short_name"],
                "provider":     cfg["provider"],
                "free_quota":   cfg["free_quota"],   # None means unlimited
            })
    return result


def resolve_model(model_id: Optional[str]) -> ModelConfig:
    """
    Return the ModelConfig for the requested model_id.
    Falls back to DEFAULT_MODEL_ID if the model isn't in the registry
    or its API key isn't configured.
    """
    if model_id and model_id in MODEL_REGISTRY:
        cfg = MODEL_REGISTRY[model_id]
        if os.getenv(cfg["api_key_env"], "").strip():
            return cfg
    # Graceful fallback
    return MODEL_REGISTRY[DEFAULT_MODEL_ID]


def get_api_key(model_id: Optional[str]) -> str:
    """Return the API key for the resolved model. Raises ValueError if missing."""
    cfg = resolve_model(model_id)
    key = os.getenv(cfg["api_key_env"], "").strip()
    if not key:
        raise ValueError(
            f"API key not configured for {cfg['display_name']} "
            f"(env var: {cfg['api_key_env']})"
        )
    return key
