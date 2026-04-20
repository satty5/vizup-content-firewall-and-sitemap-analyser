from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-5.4")

ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
ANTHROPIC_MODEL_OPUS: str = os.getenv("ANTHROPIC_MODEL_OPUS", "claude-opus-4-6")
ANTHROPIC_MODEL_SONNET: str = os.getenv("ANTHROPIC_MODEL_SONNET", "claude-sonnet-4-6")

DEFAULT_PROVIDER: str = os.getenv("DEFAULT_PROVIDER", "openai")
DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "gpt-5.4")

MAX_CONCURRENT_REVIEWERS: int = int(os.getenv("MAX_CONCURRENT_REVIEWERS", "6"))
REVIEW_TEMPERATURE: float = float(os.getenv("REVIEW_TEMPERATURE", "0.1"))

BLOCKER_THRESHOLD: int = 1
MAJOR_THRESHOLD: int = 3
MINOR_THRESHOLD: int = 8

SLOP_SCORE_FAIL: float = 0.4
ROBOTIC_DENSITY_FAIL: float = 0.35

MODEL_REGISTRY: list[dict] = [
    {
        "id": "gpt-5.4",
        "name": "GPT-5.4",
        "provider": "openai",
        "api_key_env": "OPENAI_API_KEY",
        "thinking": False,
    },
    {
        "id": "gpt-5.4-thinking",
        "name": "GPT-5.4 (Thinking)",
        "provider": "openai",
        "api_key_env": "OPENAI_API_KEY",
        "thinking": True,
        "base_model": "gpt-5.4",
        "reasoning_effort": "high",
    },
    {
        "id": "claude-opus-4-6",
        "name": "Claude Opus 4.6",
        "provider": "anthropic",
        "api_key_env": "ANTHROPIC_API_KEY",
        "thinking": False,
    },
    {
        "id": "claude-opus-4-6-thinking",
        "name": "Claude Opus 4.6 (Thinking)",
        "provider": "anthropic",
        "api_key_env": "ANTHROPIC_API_KEY",
        "thinking": True,
        "base_model": "claude-opus-4-6",
    },
    {
        "id": "claude-sonnet-4-6",
        "name": "Claude Sonnet 4.6",
        "provider": "anthropic",
        "api_key_env": "ANTHROPIC_API_KEY",
        "thinking": False,
    },
    {
        "id": "claude-sonnet-4-6-thinking",
        "name": "Claude Sonnet 4.6 (Thinking)",
        "provider": "anthropic",
        "api_key_env": "ANTHROPIC_API_KEY",
        "thinking": True,
        "base_model": "claude-sonnet-4-6",
    },
]
