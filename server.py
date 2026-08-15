"""
server.py
─────────
FastAPI web server — Settings UI, pipeline REST API, scheduler, queue, history.

Endpoints
─────────
GET  /                          → serves index.html
GET  /api/settings              → current settings (secrets masked)
POST /api/settings              → save to .env
POST /api/run                   → start a single pipeline run
GET  /api/run/status            → last run state
GET  /api/run/stream            → SSE live log stream

GET  /api/queue                 → list queued URLs
POST /api/queue                 → add URL to queue
DELETE /api/queue/{index}       → remove item from queue
POST /api/queue/run             → run all queued URLs now

GET  /api/schedule              → get schedule config
POST /api/schedule              → set/update schedule (interval or cron)
DELETE /api/schedule            → disable schedule

GET  /api/history               → last N run results

Run
───
    uvicorn server:app --reload --port 8080
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

import uvicorn
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from dotenv import dotenv_values, set_key
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ─── paths ───────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
ENV_FILE    = BASE_DIR / ".env"
STATIC_DIR  = BASE_DIR / "static"
HISTORY_FILE = BASE_DIR / ".run_history.json"
QUEUE_FILE   = BASE_DIR / ".url_queue.json"
SCHEDULE_FILE = BASE_DIR / ".schedule.json"

# ─── app ─────────────────────────────────────────────────────────────────────
app = FastAPI(title="Autonomous UI Bug Engine", version="2.0.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ─── scheduler ───────────────────────────────────────────────────────────────
scheduler = AsyncIOScheduler()

# ─── shared state ────────────────────────────────────────────────────────────
_run_state: dict[str, Any] = {
    "status": "idle",
    "log": [],
    "result": None,
    "run_id": None,
}


# ─────────────────────────────────────────────────────────────────────────────
# Persistence helpers
# ─────────────────────────────────────────────────────────────────────────────

_SENSITIVE_KEYS = {
    "OPENROUTER_API_KEY", "GITHUB_TOKEN", "OPENHANDS_API_KEY",
    "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
}
_MASK_PATTERN = re.compile(r"^[•]+$")


def _mask(key: str, value: str) -> str:
    if key in _SENSITIVE_KEYS and value and len(value) > 4:
        return "•" * (len(value) - 4) + value[-4:]
    return value


def _read_env() -> dict[str, str]:
    if ENV_FILE.exists():
        return {k: v or "" for k, v in dotenv_values(ENV_FILE).items()}
    return {}


def _write_env(updates: dict[str, str]) -> None:
    ENV_FILE.touch(exist_ok=True)
    for key, value in updates.items():
        if value:
            set_key(str(ENV_FILE), key, value)


def _load_json(path: Path, default: Any) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return default


def _save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, default=str))


def _append_history(result: dict) -> None:
    history = _load_json(HISTORY_FILE, [])
    history.insert(0, {
        "run_id":    result.get("run_id", str(uuid.uuid4())[:8]),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "url":       result.get("url", ""),
        "bug_found": result.get("bug_found", False),
        "severity":  result.get("severity", "none"),
        "fixed":     result.get("fixed", False),
        "verified":  result.get("verified"),
        "pr_url":    result.get("pr_url"),
        "error":     result.get("error"),
        "summary":   result.get("summary", ""),
    })
    _save_json(HISTORY_FILE, history[:100])      # keep last 100


# ─────────────────────────────────────────────────────────────────────────────
# Routes – UI
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_ui() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


# ─────────────────────────────────────────────────────────────────────────────
# Routes – Settings
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/settings")
async def get_settings() -> dict:
    env = _read_env()
    masked = {k: _mask(k, v) for k, v in env.items()}
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
    LLM_PROVIDER: str = "openrouter"
    OPENROUTER_API_KEY: str = ""
    MODEL_NAME: str = "x-ai/grok-4"
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_REGION: str = "us-east-1"
    BEDROCK_MODEL_ID: str = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    LLM_TEMPERATURE: str = "0.0"
    LLM_MAX_TOKENS: str = "4096"
    OPENHANDS_API_URL: str = "http://localhost:3000"
    OPENHANDS_API_KEY: str = ""
    OPENHANDS_TIMEOUT_SECONDS: str = "600"
    GITHUB_TOKEN: str = ""
    REPO_NAME: str = ""
    GITHUB_BASE_BRANCH: str = "main"
    BROWSER_MAX_STEPS: str = "25"
    LOG_LEVEL: str = "INFO"


@app.post("/api/settings")
async def save_settings(payload: SettingsPayload) -> dict:
    updates: dict[str, str] = {}
    for field, value in payload.model_dump().items():
        if isinstance(value, str) and _MASK_PATTERN.fullmatch(value):
            continue
        updates[field] = value
    _write_env(updates)
    try:
        from config import get_settings as _gs
        _gs.cache_clear()
    except Exception:
        pass
    return {"ok": True, "saved": list(updates.keys())}


# ─────────────────────────────────────────────────────────────────────────────
# Routes – Single run
# ─────────────────────────────────────────────────────────────────────────────

class RunPayload(BaseModel):
    url: str
    scenario: str = ""
    repo_path: str = "."
    verify: bool = True
    publish_pr: bool = False


@app.post("/api/run")
async def start_run(payload: RunPayload) -> dict:
    if _run_state["status"] == "running":
        raise HTTPException(status_code=409, detail="A run is already in progress.")
    run_id = str(uuid.uuid4())[:8]
    _run_state.update({"status": "running", "log": [], "result": None, "run_id": run_id})
    asyncio.create_task(_execute_pipeline(payload, run_id))
    return {"ok": True, "run_id": run_id}


@app.get("/api/run/status")
async def run_status() -> dict:
    return {
        "status":  _run_state["status"],
        "run_id":  _run_state["run_id"],
        "log":     _run_state["log"][-200:],
        "result":  _run_state["result"],
    }


@app.get("/api/run/stream")
async def run_stream() -> StreamingResponse:
    return StreamingResponse(
        _sse_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Routes – URL Queue
# ─────────────────────────────────────────────────────────────────────────────

class QueueItem(BaseModel):
    url: str
    scenario: str = ""
    repo_path: str = "."
    verify: bool = True
    publish_pr: bool = False


@app.get("/api/queue")
async def get_queue() -> dict:
    return {"queue": _load_json(QUEUE_FILE, [])}


@app.post("/api/queue")
async def add_to_queue(item: QueueItem) -> dict:
    queue = _load_json(QUEUE_FILE, [])
    entry = item.model_dump()
    entry["id"] = str(uuid.uuid4())[:8]
    entry["added_at"] = datetime.now(timezone.utc).isoformat()
    queue.append(entry)
    _save_json(QUEUE_FILE, queue)
    return {"ok": True, "id": entry["id"], "position": len(queue)}


@app.delete("/api/queue/{item_id}")
async def remove_from_queue(item_id: str) -> dict:
    queue = _load_json(QUEUE_FILE, [])
    queue = [q for q in queue if q.get("id") != item_id]
    _save_json(QUEUE_FILE, queue)
    return {"ok": True, "remaining": len(queue)}


@app.post("/api/queue/run")
async def run_queue() -> dict:
    if _run_state["status"] == "running":
        raise HTTPException(status_code=409, detail="A run is already in progress.")
    queue = _load_json(QUEUE_FILE, [])
    if not queue:
        raise HTTPException(status_code=400, detail="Queue is empty.")
    asyncio.create_task(_run_queue_task(queue))
    return {"ok": True, "count": len(queue)}


# ─────────────────────────────────────────────────────────────────────────────
# Routes – Schedule
# ─────────────────────────────────────────────────────────────────────────────

class SchedulePayload(BaseModel):
    mode: str          # "interval" | "cron" | "disabled"
    # interval mode
    interval_minutes: Optional[int] = None
    # cron mode
    cron_expression: Optional[str] = None   # e.g. "0 9 * * 1-5"
    # run config
    url: str = ""
    scenario: str = ""
    repo_path: str = "."
    verify: bool = True
    publish_pr: bool = False


@app.get("/api/schedule")
async def get_schedule() -> dict:
    cfg = _load_json(SCHEDULE_FILE, {"mode": "disabled"})
    cfg["next_run"] = _get_next_run_time()
    return cfg


@app.post("/api/schedule")
async def set_schedule(payload: SchedulePayload) -> dict:
    _save_json(SCHEDULE_FILE, payload.model_dump())
    _apply_schedule(payload)
    return {"ok": True, "mode": payload.mode, "next_run": _get_next_run_time()}


@app.delete("/api/schedule")
async def disable_schedule() -> dict:
    cfg = _load_json(SCHEDULE_FILE, {})
    cfg["mode"] = "disabled"
    _save_json(SCHEDULE_FILE, cfg)
    _remove_scheduled_job()
    return {"ok": True}


# ─────────────────────────────────────────────────────────────────────────────
# Routes – History
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/history")
async def get_history(limit: int = 50) -> dict:
    history = _load_json(HISTORY_FILE, [])
    return {"history": history[:limit]}


@app.delete("/api/history")
async def clear_history() -> dict:
    _save_json(HISTORY_FILE, [])
    return {"ok": True}


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline execution
# ─────────────────────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    ts   = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    _run_state["log"].append(line)


async def _sse_generator() -> AsyncGenerator[str, None]:
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


async def _execute_pipeline(payload: RunPayload, run_id: str) -> None:
    try:
        from config import get_settings as _gs
        _gs.cache_clear()
    except Exception:
        pass

    from main import run_pipeline_for_url

    _log(f"🚀 [{run_id}] Starting: {payload.url}")

    try:
        result = await run_pipeline_for_url(
            url=payload.url,
            scenario=payload.scenario or (
                "Navigate the site as a new user and report any "
                "visual or functional bugs."
            ),
            repo_path=payload.repo_path,
            model_name=None,
            verify=payload.verify,
            publish_pr=payload.publish_pr,
        )
        result["run_id"] = run_id

        if result.get("bug_found"):
            _log(f"🐛 Bug found — severity: {result.get('severity', '?')}")
            _log(f"   {result.get('summary', '')}")
        else:
            _log("✅ No bugs detected.")

        if result.get("fixed"):
            _log("🔧 Fix applied successfully.")
        if result.get("verified") is True:
            _log("🔍 Verification passed.")
        elif result.get("verified") is False:
            _log("⚠️  Verification: residual issues remain.")
        if result.get("pr_url"):
            _log(f"📬 PR: {result['pr_url']}")
        if result.get("error"):
            _log(f"❌ {result['error']}")

        _run_state["result"] = result
        _run_state["status"] = "done"
        _append_history(result)
        _log("🏁 Done.")

    except Exception as exc:  # noqa: BLE001
        _log(f"💥 Exception: {exc}")
        _run_state["status"] = "error"
        _run_state["result"] = {"error": str(exc), "run_id": run_id}
        _append_history(_run_state["result"])


async def _run_queue_task(queue: list[dict]) -> None:
    """Process all items in the queue sequentially."""
    _log(f"📋 Queue run starting — {len(queue)} URLs")
    for i, item in enumerate(queue, 1):
        _log(f"\n── [{i}/{len(queue)}] {item['url']}")
        payload = RunPayload(**{k: item[k] for k in RunPayload.model_fields if k in item})
        run_id = str(uuid.uuid4())[:8]
        _run_state.update({"status": "running", "run_id": run_id})
        await _execute_pipeline(payload, run_id)
        await asyncio.sleep(2)     # brief pause between runs

    _save_json(QUEUE_FILE, [])     # clear queue after full run
    _log("✅ Queue complete — queue cleared.")
    _run_state["status"] = "done"


async def _scheduled_run() -> None:
    """Called by APScheduler on each tick."""
    cfg = _load_json(SCHEDULE_FILE, {})
    if not cfg.get("url"):
        return
    if _run_state["status"] == "running":
        return    # skip tick if a run is in progress

    payload = RunPayload(
        url=cfg["url"],
        scenario=cfg.get("scenario", ""),
        repo_path=cfg.get("repo_path", "."),
        verify=cfg.get("verify", True),
        publish_pr=cfg.get("publish_pr", False),
    )
    run_id = str(uuid.uuid4())[:8]
    _run_state.update({"status": "running", "log": [], "result": None, "run_id": run_id})
    _log(f"⏰ Scheduled run triggered (id={run_id})")
    await _execute_pipeline(payload, run_id)


# ─────────────────────────────────────────────────────────────────────────────
# Scheduler helpers
# ─────────────────────────────────────────────────────────────────────────────

_JOB_ID = "auto_qa_job"


def _apply_schedule(payload: SchedulePayload) -> None:
    _remove_scheduled_job()
    if payload.mode == "disabled":
        return
    if payload.mode == "interval" and payload.interval_minutes:
        trigger = IntervalTrigger(minutes=payload.interval_minutes)
    elif payload.mode == "cron" and payload.cron_expression:
        parts = payload.cron_expression.strip().split()
        if len(parts) != 5:
            raise ValueError("Cron expression must have 5 fields.")
        trigger = CronTrigger(
            minute=parts[0], hour=parts[1],
            day=parts[2], month=parts[3], day_of_week=parts[4],
        )
    else:
        return
    scheduler.add_job(_scheduled_run, trigger=trigger, id=_JOB_ID, replace_existing=True)


def _remove_scheduled_job() -> None:
    if scheduler.get_job(_JOB_ID):
        scheduler.remove_job(_JOB_ID)


def _get_next_run_time() -> Optional[str]:
    job = scheduler.get_job(_JOB_ID)
    if job and job.next_run_time:
        return job.next_run_time.isoformat()
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Startup / shutdown
# ─────────────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup() -> None:
    scheduler.start()
    # Re-apply schedule from disk on restart
    cfg = _load_json(SCHEDULE_FILE, {})
    if cfg.get("mode") and cfg["mode"] != "disabled":
        try:
            _apply_schedule(SchedulePayload(**cfg))
        except Exception:
            pass


@app.on_event("shutdown")
async def shutdown() -> None:
    scheduler.shutdown(wait=False)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=True)
