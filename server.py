"""
server.py
─────────
FastAPI web server — provides the Settings UI and pipeline REST API.

Endpoints
─────────
GET  /                        → serves index.html
GET  /api/settings            → returns current settings (secrets masked)
POST /api/settings            → writes new values to .env
POST /api/run                 → starts a pipeline run (streams SSE logs)
GET  /api/run/status          → returns last run result

Run
───
    uvicorn server:app --reload --port 8080
    # then open http://localhost:8080
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Any, AsyncGenerator

import uvicorn
from dotenv import dotenv_values, set_key
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ─── paths ───────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
ENV_FILE = BASE_DIR / ".env"
STATIC_DIR = BASE_DIR / "static"

# ─── app ─────────────────────────────────────────────────────────────────────
app = FastAPI(title="Autonomous UI Bug Engine", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# In-memory store for the last run result and live log lines
_run_state: dict[str, Any] = {"status": "idle", "log": [], "result": None}


# ─── helpers ─────────────────────────────────────────────────────────────────

_SENSITIVE_KEYS = {"OPENROUTER_API_KEY", "GITHUB_TOKEN", "OPENHANDS_API_KEY",
                   "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"}


def _mask(key: str, value: str) -> str:
    """Return masked value for sensitive keys (show last 4 chars only)."""
    if key in _SENSITIVE_KEYS and value and len(value) > 4:
        return "•" * (len(value) - 4) + value[-4:]
    return value


def _read_env() -> dict[str, str]:
    """Read .env if it exists, else return empty dict."""
    if ENV_FILE.exists():
        return {k: v or "" for k, v in dotenv_values(ENV_FILE).items()}
    return {}


def _write_env(updates: dict[str, str]) -> None:
    """Write / update keys in .env (creates file if missing)."""
    ENV_FILE.touch(exist_ok=True)
    for key, value in updates.items():
        if value:                         # only write non-empty values
            set_key(str(ENV_FILE), key, value)


# ─── routes ──────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_ui() -> FileResponse:
    """Serve the single-page UI."""
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/settings")
async def get_settings() -> dict:
    """Return current settings with sensitive values masked."""
    env = _read_env()
    masked = {k: _mask(k, v) for k, v in env.items()}
    # Add defaults for keys that may not exist yet
    defaults = {
        "LLM_PROVIDER": "openrouter",
        "MODEL_NAME": "x-ai/grok-4",
        "OPENROUTER_BASE_URL": "https://openrouter.ai/api/v1",
        "LLM_TEMPERATURE": "0.0",
        "LLM_MAX_TOKENS": "4096",
        "OPENHANDS_API_URL": "http://localhost:3000",
        "OPENHANDS_TIMEOUT_SECONDS": "600",
        "GITHUB_BASE_BRANCH": "main",
        "BROWSER_MAX_STEPS": "25",
        "LOG_LEVEL": "INFO",
        "AWS_REGION": "us-east-1",
    }
    return {**defaults, **masked}


class SettingsPayload(BaseModel):
    """Incoming settings from the UI form."""
    # Provider selection
    LLM_PROVIDER: str = "openrouter"        # "openrouter" | "bedrock"

    # OpenRouter fields
    OPENROUTER_API_KEY: str = ""
    MODEL_NAME: str = "x-ai/grok-4"
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

    # Amazon Bedrock fields
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_REGION: str = "us-east-1"
    BEDROCK_MODEL_ID: str = "anthropic.claude-3-5-sonnet-20241022-v2:0"

    # LLM tuning
    LLM_TEMPERATURE: str = "0.0"
    LLM_MAX_TOKENS: str = "4096"

    # OpenHands
    OPENHANDS_API_URL: str = "http://localhost:3000"
    OPENHANDS_API_KEY: str = ""
    OPENHANDS_TIMEOUT_SECONDS: str = "600"

    # GitHub
    GITHUB_TOKEN: str = ""
    REPO_NAME: str = ""
    GITHUB_BASE_BRANCH: str = "main"

    # QA
    BROWSER_MAX_STEPS: str = "25"
    LOG_LEVEL: str = "INFO"


_MASK_PATTERN = re.compile(r"^[•]+")


@app.post("/api/settings")
async def save_settings(payload: SettingsPayload) -> dict:
    """
    Persist settings to .env.
    Masked values (all bullets) are skipped so existing secrets are preserved.
    """
    updates: dict[str, str] = {}
    for field, value in payload.model_dump().items():
        if isinstance(value, str) and _MASK_PATTERN.fullmatch(value):
            continue           # unchanged masked field — don't overwrite
        updates[field] = value

    _write_env(updates)
    # Invalidate the settings cache so config.py picks up fresh values
    try:
        from config import get_settings
        get_settings.cache_clear()
    except Exception:
        pass

    return {"ok": True, "saved": list(updates.keys())}


class RunPayload(BaseModel):
    """Incoming run request from the UI."""
    url: str
    scenario: str = ""
    repo_path: str = "."
    verify: bool = True
    publish_pr: bool = False


@app.post("/api/run")
async def start_run(payload: RunPayload) -> dict:
    """
    Start the pipeline in a background task.
    Poll /api/run/status or use /api/run/stream for live logs.
    """
    if _run_state["status"] == "running":
        raise HTTPException(status_code=409, detail="A run is already in progress.")

    _run_state.update({"status": "running", "log": [], "result": None})
    asyncio.create_task(_execute_pipeline(payload))
    return {"ok": True, "message": "Pipeline started."}


@app.get("/api/run/status")
async def run_status() -> dict:
    """Return the current run state."""
    return {
        "status": _run_state["status"],
        "log": _run_state["log"][-100:],   # last 100 lines
        "result": _run_state["result"],
    }


@app.get("/api/run/stream")
async def run_stream() -> StreamingResponse:
    """Server-Sent Events stream — yields log lines as they appear."""
    return StreamingResponse(
        _sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ─── pipeline execution ───────────────────────────────────────────────────────

def _log(msg: str) -> None:
    """Append a timestamped log line to the shared run state."""
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    _run_state["log"].append(line)


async def _sse_generator() -> AsyncGenerator[str, None]:
    """Yield SSE events while the pipeline is running."""
    sent = 0
    while True:
        lines = _run_state["log"]
        while sent < len(lines):
            yield f"data: {json.dumps({'line': lines[sent]})}\n\n"
            sent += 1

        if _run_state["status"] != "running":
            yield f"data: {json.dumps({'done': True, 'result': _run_state['result']})}\n\n"
            break

        await asyncio.sleep(0.3)


async def _execute_pipeline(payload: RunPayload) -> None:
    """Run the full find→fix→verify→PR pipeline asynchronously."""
    # Reload settings after potential save
    try:
        from config import get_settings
        get_settings.cache_clear()
    except Exception:
        pass

    from main import run_pipeline_for_url

    _log(f"🚀 Starting pipeline for: {payload.url}")

    try:
        result = await run_pipeline_for_url(
            url=payload.url,
            scenario=payload.scenario or "Navigate the site as a new user and report any visual or functional bugs.",
            repo_path=payload.repo_path,
            model_name=None,
            verify=payload.verify,
            publish_pr=payload.publish_pr,
        )
        _run_state["result"] = result

        if result.get("bug_found"):
            _log(f"🐛 Bug found — severity: {result.get('severity', 'unknown')}")
        else:
            _log("✅ No bugs detected.")

        if result.get("fixed"):
            _log("🔧 Fix applied successfully.")
        if result.get("pr_url"):
            _log(f"📬 PR created: {result['pr_url']}")
        if result.get("error"):
            _log(f"❌ Error: {result['error']}")

        _log("🏁 Pipeline complete.")
        _run_state["status"] = "done"

    except Exception as exc:  # noqa: BLE001
        _log(f"💥 Unhandled exception: {exc}")
        _run_state["status"] = "error"
        _run_state["result"] = {"error": str(exc)}


# ─── entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=True)
