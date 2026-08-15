"""
config.py
─────────
Centralised configuration for the Autonomous UI Bug-Finding Engine.

All values are read from environment variables (or a .env file via
python-dotenv).  Pydantic-Settings provides runtime validation and
type coercion so misconfigured deployments fail fast at startup.

Usage
-----
    from config import get_settings, build_llm

    settings = get_settings()          # cached singleton
    llm      = build_llm()             # ChatOpenAI instance → OpenRouter
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env file before Pydantic reads environment variables.
load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# Settings model
# ─────────────────────────────────────────────────────────────────────────────

class Settings(BaseSettings):
    """
    Validated, type-safe configuration loaded from environment variables.

    Priority (highest → lowest):
        1. Actual environment variables
        2. Variables in a .env file
        3. Field default values defined below
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",         # silently ignore unknown env vars
    )

    # ── LLM / OpenRouter ─────────────────────────────────────────────────────

    openrouter_api_key: SecretStr = Field(
        ...,
        description="OpenRouter API key (required).",
    )
    model_name: str = Field(
        default="x-ai/grok-4",
        description="OpenRouter model identifier (OpenAI-compatible).",
    )
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="Base URL for the OpenAI-compatible endpoint.",
    )
    llm_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0.0 = deterministic).",
    )
    llm_max_tokens: int = Field(
        default=4096,
        gt=0,
        description="Maximum tokens the LLM may generate per call.",
    )

    # ── OpenHands ("Hands") ──────────────────────────────────────────────────

    openhands_api_url: str = Field(
        default="http://localhost:3000",
        description="Base URL of the locally running OpenHands Docker service.",
    )
    openhands_api_key: Optional[SecretStr] = Field(
        default=None,
        description="Optional bearer token for OpenHands auth.",
    )
    openhands_timeout_seconds: int = Field(
        default=600,
        gt=0,
        description="Seconds to wait for OpenHands to complete a task.",
    )

    # ── GitHub ───────────────────────────────────────────────────────────────

    github_token: SecretStr = Field(
        ...,
        description="GitHub personal access token (required).",
    )
    repo_name: str = Field(
        ...,
        description='Target repository in "owner/repo" format (required).',
    )
    github_base_branch: str = Field(
        default="main",
        description="Base branch that PRs will target.",
    )

    # ── QA / Browser ─────────────────────────────────────────────────────────

    target_urls: list[str] = Field(
        default_factory=list,
        description="Comma-separated list of URLs to audit.",
    )
    default_test_scenario: str = Field(
        default=(
            "Navigate the site as a new user and report any visual or "
            "functional bugs you encounter."
        ),
        description="Fallback test scenario for the QA agent.",
    )
    browser_max_steps: int = Field(
        default=25,
        gt=0,
        description="Maximum browser interaction steps per test scenario.",
    )

    # ── Logging ──────────────────────────────────────────────────────────────

    log_level: str = Field(
        default="INFO",
        description="Python logging level: DEBUG | INFO | WARNING | ERROR | CRITICAL",
    )

    # ── Validators ───────────────────────────────────────────────────────────

    @field_validator("target_urls", mode="before")
    @classmethod
    def _parse_urls(cls, v: object) -> list[str]:
        """Accept a comma-separated string or a list from the environment."""
        if isinstance(v, str):
            return [u.strip() for u in v.split(",") if u.strip()]
        if isinstance(v, list):
            return v
        return []

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}; got {v!r}")
        return upper

    @model_validator(mode="after")
    def _configure_logging(self) -> "Settings":
        """Apply the configured log level to the root logger at startup."""
        logging.basicConfig(
            level=getattr(logging, self.log_level),
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Public helpers
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the validated Settings singleton.

    Uses ``lru_cache`` so the .env file and environment are read exactly once,
    even when ``get_settings()`` is called from multiple modules.

    Raises
    ------
    pydantic.ValidationError
        If any required field is missing or a value fails validation.
    """
    return Settings()  # type: ignore[call-arg]


def build_llm(
    *,
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> ChatOpenAI:
    """
    Construct and return a ``ChatOpenAI`` instance wired to OpenRouter.

    The function reads defaults from ``get_settings()``; any keyword argument
    passed explicitly overrides the corresponding setting.

    Parameters
    ----------
    model_name : str, optional
        Override the model identifier (e.g. ``"zhipu/glm-4-9b"``).
    temperature : float, optional
        Sampling temperature override.
    max_tokens : int, optional
        Max-token override.

    Returns
    -------
    ChatOpenAI
        A ready-to-use LangChain chat model pointed at OpenRouter.

    Example
    -------
    >>> llm = build_llm(model_name="openai/gpt-4o")
    >>> response = llm.invoke("Say hello.")
    """
    cfg = get_settings()

    return ChatOpenAI(
        model=model_name or cfg.model_name,
        openai_api_key=cfg.openrouter_api_key.get_secret_value(),
        openai_api_base=cfg.openrouter_base_url,
        temperature=temperature if temperature is not None else cfg.llm_temperature,
        max_tokens=max_tokens or cfg.llm_max_tokens,
        # Metadata forwarded in the X-* headers that OpenRouter reads.
        default_headers={
            "HTTP-Referer": "https://github.com/autonomous-engine",
            "X-Title": "Autonomous UI Bug Engine",
        },
    )
