"""
config.py
─────────
Centralised configuration for the Autonomous UI Bug-Finding Engine.

All values are loaded from environment variables (or a .env file via
python-dotenv). Pydantic-Settings validates them at startup so
misconfigurations fail fast.

LLM provider
────────────
Only OpenRouter is supported.  OpenRouter is an OpenAI-compatible proxy
that gives access to every public model (xAI Grok, Anthropic Claude,
OpenAI GPT, Meta Llama, Google Gemini, Mistral, etc.) through a single
API key.  Change MODEL_NAME in .env to switch models instantly.

Usage
─────
    from config import get_settings, build_llm

    settings = get_settings()   # validated singleton
    llm      = build_llm()      # ready-to-use ChatOpenAI → OpenRouter
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    """
    Type-safe, validated settings loaded from environment variables.

    Priority (highest → lowest):
        1. Real environment variables
        2. Variables in .env file
        3. Field default values below
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── OpenRouter ────────────────────────────────────────────────────────────

    openrouter_api_key: SecretStr = Field(
        ...,
        description="OpenRouter API key — get one at openrouter.ai/keys",
    )
    model_name: str = Field(
        default="x-ai/grok-4",
        description=(
            "Any OpenRouter model slug, e.g.:\n"
            "  x-ai/grok-4\n"
            "  anthropic/claude-3.5-sonnet\n"
            "  openai/gpt-4o\n"
            "  meta-llama/llama-3.3-70b-instruct\n"
            "  google/gemini-2.0-flash-001\n"
            "  mistralai/mistral-large\n"
            "See full list at openrouter.ai/models"
        ),
    )
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenRouter base URL (OpenAI-compatible).",
    )
    llm_temperature: float = Field(
        default=0.0, ge=0.0, le=2.0,
        description="Sampling temperature (0 = deterministic).",
    )
    llm_max_tokens: int = Field(
        default=4096, gt=0,
        description="Max tokens per LLM call.",
    )

    # ── OpenHands ─────────────────────────────────────────────────────────────

    openhands_api_url: str = Field(
        default="http://localhost:3000",
        description="OpenHands Docker service base URL.",
    )
    openhands_api_key: Optional[SecretStr] = Field(
        default=None,
        description="Optional bearer token for OpenHands auth.",
    )
    openhands_timeout_seconds: int = Field(
        default=600, gt=0,
        description="Seconds to wait for OpenHands task completion.",
    )

    # ── GitHub ────────────────────────────────────────────────────────────────

    github_token: SecretStr = Field(
        ...,
        description="GitHub PAT with Contents (rw) + Pull Requests (rw).",
    )
    repo_name: str = Field(
        ...,
        description='Target repo in "owner/repo" format.',
    )
    github_base_branch: str = Field(
        default="main",
        description="Base branch PRs will target.",
    )

    # ── QA / Browser ──────────────────────────────────────────────────────────

    target_urls: list[str] = Field(
        default_factory=list,
        description="Comma-separated URLs to audit.",
    )
    default_test_scenario: str = Field(
        default="Navigate the site as a new user and report any visual or functional bugs.",
        description="Fallback QA scenario.",
    )
    browser_max_steps: int = Field(
        default=25, gt=0,
        description="Maximum browser steps per scenario.",
    )

    # ── Logging ───────────────────────────────────────────────────────────────

    log_level: str = Field(
        default="INFO",
        description="Python log level: DEBUG | INFO | WARNING | ERROR | CRITICAL",
    )

    # ── Validators ────────────────────────────────────────────────────────────

    @field_validator("target_urls", mode="before")
    @classmethod
    def _parse_urls(cls, v: object) -> list[str]:
        if isinstance(v, str):
            return [u.strip() for u in v.split(",") if u.strip()]
        return v if isinstance(v, list) else []

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        u = v.upper()
        if u not in allowed:
            raise ValueError(f"log_level must be one of {allowed}")
        return u

    @model_validator(mode="after")
    def _configure_logging(self) -> "Settings":
        logging.basicConfig(
            level=getattr(logging, self.log_level),
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the validated Settings singleton (read once, cached forever)."""
    return Settings()  # type: ignore[call-arg]


def build_llm(
    *,
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> ChatOpenAI:
    """
    Build a ChatOpenAI instance pointed at OpenRouter.

    Any keyword argument overrides the corresponding .env setting.

    Examples
    --------
    >>> llm = build_llm()                                   # uses MODEL_NAME from .env
    >>> llm = build_llm(model_name="openai/gpt-4o")        # one-off override
    >>> llm = build_llm(model_name="google/gemini-2.0-flash-001", temperature=0.3)
    """
    cfg = get_settings()
    return ChatOpenAI(
        model=model_name or cfg.model_name,
        openai_api_key=cfg.openrouter_api_key.get_secret_value(),
        openai_api_base=cfg.openrouter_base_url,
        temperature=temperature if temperature is not None else cfg.llm_temperature,
        max_tokens=max_tokens or cfg.llm_max_tokens,
        default_headers={
            "HTTP-Referer": "https://github.com/KashinathTilagul/autonomous-engine",
            "X-Title": "Autonomous UI Bug Engine",
        },
    )
