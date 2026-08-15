"""
config.py
─────────
Centralised configuration for the Autonomous UI Bug-Finding Engine.

Configured for OmniRoute / OpenAI-compatible routing gateways with
free API tiers and seamless model switching.
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
    Type-safe settings for OmniRoute and autonomous QA engine.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── OmniRoute / Gateway ───────────────────────────────────────────────────

    omniroute_api_key: Optional[SecretStr] = Field(
        default=None,
        alias="OMNIROUTE_API_KEY",
        description="OmniRoute API key — entered via UI or .env",
    )
    model_name: str = Field(
        default="deepseek/deepseek-r1:free",
        alias="MODEL_NAME",
        description="Default best free model tier or any custom model slug",
    )
    omniroute_base_url: str = Field(
        default="https://api.omniroute.ai/v1",
        alias="OMNIROUTE_BASE_URL",
        description="OmniRoute base URL (OpenAI-compatible).",
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
        description="OpenHands service base URL.",
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

    github_token: Optional[SecretStr] = Field(
        default=None,
        description="GitHub PAT with Contents (rw) + Pull Requests (rw).",
    )
    repo_name: str = Field(
        default="KashinathTilagul/autonomous-engine",
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
    """Return the validated Settings singleton."""
    return Settings()


def build_llm(
    *,
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> ChatOpenAI:
    """
    Build a ChatOpenAI instance pointed at OmniRoute.
    """
    cfg = get_settings()
    api_key_val = (
        cfg.omniroute_api_key.get_secret_value()
        if cfg.omniroute_api_key
        else "free-key"
    )

    return ChatOpenAI(
        model=model_name or cfg.model_name,
        openai_api_key=api_key_val,
        openai_api_base=cfg.omniroute_base_url,
        temperature=temperature if temperature is not None else cfg.llm_temperature,
        max_tokens=max_tokens or cfg.llm_max_tokens,
        default_headers={
            "HTTP-Referer": "https://github.com/KashinathTilagul/autonomous-engine",
            "X-Title": "Autonomous UI Bug Engine",
        },
    )
