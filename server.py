"""
server.py  ·  Autonomous UI Bug Engine
─────────────────────────────────────────────────────────────────────────────
FastAPI web server – Settings, Pipeline, Scheduler, Queue, History,
Target URL Manager, and Repo Manager with inline file editor.
Optimized for local and serverless (Vercel) environments.
"""

from __future__ import annotations

import asyncio
import json
import mimetypes
import os
import re
import subprocess
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

import uvicorn
from dotenv import dotenv_values, set_key
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

# ── paths (serverless safe /tmp for writable state when on Vercel) ───────────
BASE_DIR      = Path(__file__).resolve().parent
STATIC_DIR    = BASE_DIR / "static"
INDEX_FILE    = STATIC_DIR / "index.html"

# If deployed in read-only serverless environment, use temp directory for writable state
def _get_writable_dir() -> Path:
    if os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
        tmp = Path(tempfile.gettempdir()) / "autonomous_engine"
        tmp.mkdir(parents=True, exist_ok=True)
        return tmp
    return BASE_DIR

STATE_DIR     = _get_writable_dir()
ENV_FILE      = BASE_DIR / ".env"
HISTORY_FILE  = STATE_DIR / ".run_history.json"
QUEUE_FILE    = STATE_DIR / ".url_queue.json"
SCHEDULE_FILE = STATE_DIR / ".schedule.json"
TARGETS_FILE  = STATE_DIR / ".targets.json"
REPOS_FILE    = STATE_DIR / ".repos.json"

# In-memory fallbacks for serverless environments
_mem_store: dict[str, Any] = {
    "settings": {},
    "targets": [],
    "repos": [],
    "queue": [],
    "history": [],
    "schedule": {"mode": "disabled"}
}

# ── app ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Autonomous UI Bug Engine", version="3.2.0")

_run_state: dict[str, Any] = {
    "status": "idle", "log": [], "result": None, "run_id": None,
}

# ── sensitive keys ────────────────────────────────────────────────────────────
_SENSITIVE = {
    "OMNIROUTE_API_KEY", "OPENROUTER_API_KEY", "GITHUB_TOKEN", "OPENHANDS_API_KEY",
}
_MASKED_RE = re.compile(r"^[•]+$")


# ─────────────────────────────────────────────────────────────────────────────
# Persistence helpers
# ─────────────────────────────────────────────────────────────────────────────

def _mask(key: str, val: str) -> str:
    if key in _SENSITIVE and val and len(val) > 4:
        return "•" * (len(val) - 4) + val[-4:]
    return val

def _read_env() -> dict[str, str]:
    data = {}
    if ENV_FILE.exists():
        try:
            data.update({k: v or "" for k, v in dotenv_values(ENV_FILE).items()})
        except Exception:
            pass
    data.update(_mem_store["settings"])
    return data

def _write_env(updates: dict[str, str]) -> None:
    _mem_store["settings"].update(updates)
    try:
        if not os.environ.get("VERCEL"):
            ENV_FILE.touch(exist_ok=True)
            for k, v in updates.items():
                if v:
                    set_key(str(ENV_FILE), k, v)
    except Exception:
        pass

def _load(path: Path, mem_key: str, default: Any = None) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return _mem_store.get(mem_key, default if default is not None else [])

def _save(path: Path, mem_key: str, data: Any) -> None:
    _mem_store[mem_key] = data
    try:
        path.write_text(json.dumps(data, indent=2, default=str))
    except Exception:
        pass

def _append_history(result: dict) -> None:
    h = _load(HISTORY_FILE, "history", [])
    h.insert(0, {
        "run_id":    result.get("run_id", ""),
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
    _save(HISTORY_FILE, "history", h[:100])


# ─────────────────────────────────────────────────────────────────────────────
# Routes – UI
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_ui() -> HTMLResponse:
    if INDEX_FILE.exists():
        return HTMLResponse(content=INDEX_FILE.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>Autonomous UI Bug Engine is running. Index file not found.</h2>")


# ─────────────────────────────────────────────────────────────────────────────
# Routes – Settings
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/settings")
async def get_settings() -> dict:
    env = _read_env()
    masked = {k: _mask(k, v) for k, v in env.items()}
    defaults = {
        "OMNIROUTE_API_KEY": "",
        "MODEL_NAME": "deepseek/deepseek-r1:free",
        "OMNIROUTE_BASE_URL": "https://api.omniroute.ai/v1",
        "LLM_TEMPERATURE": "0.0",
        "LLM_MAX_TOKENS": "4096",
        "OPENHANDS_API_URL": "http://localhost:3000",
        "OPENHANDS_API_KEY": "",
        "OPENHANDS_TIMEOUT_SECONDS": "600",
        "GITHUB_TOKEN": "",
        "REPO_NAME": "KashinathTilagul/autonomous-engine",
        "GITHUB_BASE_BRANCH": "main",
        "BROWSER_MAX_STEPS": "25",
        "LOG_LEVEL": "INFO",
    }
    return {**defaults, **masked}


class SettingsPayload(BaseModel):
    OMNIROUTE_API_KEY: str = ""
    MODEL_NAME: str = "deepseek/deepseek-r1:free"
    OMNIROUTE_BASE_URL: str = "https://api.omniroute.ai/v1"
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
    updates = {k: v for k, v in payload.model_dump().items()
               if not (isinstance(v, str) and _MASKED_RE.fullmatch(v))}
    _write_env(updates)
    try:
        from config import get_settings as _gs; _gs.cache_clear()
    except Exception:
        pass
    return {"ok": True, "saved": list(updates.keys())}


# ─────────────────────────────────────────────────────────────────────────────
# Routes – Run
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
        raise HTTPException(409, "A run is already in progress.")
    run_id = str(uuid.uuid4())[:8]
    _run_state.update({"status": "running", "log": [], "result": None, "run_id": run_id})
    asyncio.create_task(_execute_pipeline(payload, run_id))
    return {"ok": True, "run_id": run_id}


@app.get("/api/run/status")
async def run_status() -> dict:
    return {
        "status": _run_state["status"],
        "run_id": _run_state["run_id"],
        "log":    _run_state["log"][-200:],
        "result": _run_state["result"],
    }


@app.get("/api/run/stream")
async def run_stream() -> StreamingResponse:
    return StreamingResponse(
        _sse_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Routes – Queue
# ─────────────────────────────────────────────────────────────────────────────

class QueueItem(BaseModel):
    url: str
    scenario: str = ""
    repo_path: str = "."
    verify: bool = True
    publish_pr: bool = False


@app.get("/api/queue")
async def get_queue() -> dict:
    return {"queue": _load(QUEUE_FILE, "queue", [])}


@app.post("/api/queue")
async def add_to_queue(item: QueueItem) -> dict:
    q = _load(QUEUE_FILE, "queue", [])
    entry = {**item.model_dump(), "id": str(uuid.uuid4())[:8],
             "added_at": datetime.now(timezone.utc).isoformat()}
    q.append(entry)
    _save(QUEUE_FILE, "queue", q)
    return {"ok": True, "id": entry["id"], "position": len(q)}


@app.delete("/api/queue/{item_id}")
async def remove_from_queue(item_id: str) -> dict:
    q = [i for i in _load(QUEUE_FILE, "queue", []) if i.get("id") != item_id]
    _save(QUEUE_FILE, "queue", q)
    return {"ok": True}


@app.post("/api/queue/run")
async def run_queue() -> dict:
    if _run_state["status"] == "running":
        raise HTTPException(409, "A run is already in progress.")
    q = _load(QUEUE_FILE, "queue", [])
    if not q:
        raise HTTPException(400, "Queue is empty.")
    asyncio.create_task(_run_queue_task(q))
    return {"ok": True, "count": len(q)}


# ─────────────────────────────────────────────────────────────────────────────
# Routes – Schedule
# ─────────────────────────────────────────────────────────────────────────────

class SchedulePayload(BaseModel):
    mode: str
    interval_minutes: Optional[int] = None
    cron_expression: Optional[str] = None
    url: str = ""
    scenario: str = ""
    repo_path: str = "."
    verify: bool = True
    publish_pr: bool = False


@app.get("/api/schedule")
async def get_schedule() -> dict:
    cfg = _load(SCHEDULE_FILE, "schedule", {"mode": "disabled"})
    return cfg


@app.post("/api/schedule")
async def set_schedule(payload: SchedulePayload) -> dict:
    _save(SCHEDULE_FILE, "schedule", payload.model_dump())
    return {"ok": True, "mode": payload.mode}


@app.delete("/api/schedule")
async def delete_schedule() -> dict:
    cfg = _load(SCHEDULE_FILE, "schedule", {})
    cfg["mode"] = "disabled"
    _save(SCHEDULE_FILE, "schedule", cfg)
    return {"ok": True}


# ─────────────────────────────────────────────────────────────────────────────
# Routes – History
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/history")
async def get_history(limit: int = 50) -> dict:
    return {"history": _load(HISTORY_FILE, "history", [])[:limit]}


@app.delete("/api/history")
async def clear_history() -> dict:
    _save(HISTORY_FILE, "history", [])
    return {"ok": True}


# ─────────────────────────────────────────────────────────────────────────────
# Routes – Target URL Manager
# ─────────────────────────────────────────────────────────────────────────────

class TargetPayload(BaseModel):
    name: str = ""
    url: str
    scenario: str = ""
    notes: str = ""
    tags: list[str] = []


class TargetPatch(BaseModel):
    name: Optional[str] = None
    url: Optional[str] = None
    scenario: Optional[str] = None
    notes: Optional[str] = None
    tags: Optional[list[str]] = None


@app.get("/api/targets")
async def get_targets() -> dict:
    return {"targets": _load(TARGETS_FILE, "targets", [])}


@app.post("/api/targets")
async def add_target(payload: TargetPayload) -> dict:
    targets = _load(TARGETS_FILE, "targets", [])
    entry = {**payload.model_dump(), "id": str(uuid.uuid4())[:8],
             "created_at": datetime.now(timezone.utc).isoformat(),
             "last_run": None, "last_result": None}
    targets.append(entry)
    _save(TARGETS_FILE, "targets", targets)
    return {"ok": True, "id": entry["id"]}


@app.patch("/api/targets/{target_id}")
async def update_target(target_id: str, patch: TargetPatch) -> dict:
    targets = _load(TARGETS_FILE, "targets", [])
    for t in targets:
        if t["id"] == target_id:
            for k, v in patch.model_dump(exclude_none=True).items():
                t[k] = v
    _save(TARGETS_FILE, "targets", targets)
    return {"ok": True}


@app.delete("/api/targets/{target_id}")
async def delete_target(target_id: str) -> dict:
    targets = [t for t in _load(TARGETS_FILE, "targets", []) if t["id"] != target_id]
    _save(TARGETS_FILE, "targets", targets)
    return {"ok": True}


# ─────────────────────────────────────────────────────────────────────────────
# Routes – Repo Manager + File Editor
# ─────────────────────────────────────────────────────────────────────────────

_TEXT_EXTENSIONS = {
    ".py",".js",".ts",".tsx",".jsx",".html",".css",".scss",".json",
    ".yaml",".yml",".toml",".ini",".cfg",".env",".example",".md",
    ".txt",".sh",".bash",".zsh",".fish",".rs",".go",".rb",".php",
    ".java",".c",".cpp",".h",".hpp",".cs",".swift",".kt",".sql",
    ".graphql",".xml",".svg",".gitignore",".dockerignore","Makefile",
    "Dockerfile",".tf",".hcl",".lock",".prettierrc",".eslintrc",
}
_SKIP_DIRS = {".git",".venv","venv","node_modules","__pycache__",".mypy_cache",
              ".pytest_cache","dist","build",".next",".nuxt"}
_MAX_FILE_SIZE = 512 * 1024   # 512 KB read limit


class RepoPayload(BaseModel):
    name: str
    local_path: str
    github_url: str = ""
    branch: str = "main"
    description: str = ""


class RepoPatch(BaseModel):
    name: Optional[str] = None
    local_path: Optional[str] = None
    github_url: Optional[str] = None
    branch: Optional[str] = None
    description: Optional[str] = None


class FileWritePayload(BaseModel):
    path: str
    content: str


class GitOpPayload(BaseModel):
    op: str
    message: str = "chore: automated edit via Bug Engine UI"


@app.get("/api/repos")
async def get_repos() -> dict:
    return {"repos": _load(REPOS_FILE, "repos", [])}


@app.post("/api/repos")
async def add_repo(payload: RepoPayload) -> dict:
    repos = _load(REPOS_FILE, "repos", [])
    entry = {**payload.model_dump(), "id": str(uuid.uuid4())[:8],
             "created_at": datetime.now(timezone.utc).isoformat()}
    repos.append(entry)
    _save(REPOS_FILE, "repos", repos)
    return {"ok": True, "id": entry["id"]}


@app.patch("/api/repos/{repo_id}")
async def update_repo(repo_id: str, patch: RepoPatch) -> dict:
    repos = _load(REPOS_FILE, "repos", [])
    for r in repos:
        if r["id"] == repo_id:
            for k, v in patch.model_dump(exclude_none=True).items():
                r[k] = v
    _save(REPOS_FILE, "repos", repos)
    return {"ok": True}


@app.delete("/api/repos/{repo_id}")
async def delete_repo(repo_id: str) -> dict:
    repos = [r for r in _load(REPOS_FILE, "repos", []) if r["id"] != repo_id]
    _save(REPOS_FILE, "repos", repos)
    return {"ok": True}


def _get_repo_or_404(repo_id: str) -> dict:
    for r in _load(REPOS_FILE, "repos", []):
        if r["id"] == repo_id:
            return r
    raise HTTPException(404, "Repo not found.")


@app.get("/api/repos/{repo_id}/tree")
async def repo_tree(repo_id: str, path: str = Query(".")) -> dict:
    repo = _get_repo_or_404(repo_id)
    base = Path(repo["local_path"]).resolve()

    if not base.exists():
        return {"path": path, "entries": []}

    target = (base / path).resolve()
    if not str(target).startswith(str(base)):
        raise HTTPException(403, "Access denied.")
    if not target.exists():
        return {"path": path, "entries": []}

    entries = []
    if target.is_dir():
        for child in sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name)):
            if child.name.startswith(".") and child.name in _SKIP_DIRS:
                continue
            if child.is_dir() and child.name in _SKIP_DIRS:
                continue
            rel = child.relative_to(base)
            entries.append({
                "name":  child.name,
                "path":  str(rel),
                "type":  "dir" if child.is_dir() else "file",
                "size":  child.stat().st_size if child.is_file() else None,
                "ext":   child.suffix.lower() if child.is_file() else None,
            })
    return {"path": path, "entries": entries}


@app.get("/api/repos/{repo_id}/file")
async def read_file(repo_id: str, path: str = Query(...)) -> dict:
    repo = _get_repo_or_404(repo_id)
    base = Path(repo["local_path"]).resolve()
    target = (base / path).resolve()

    if not str(target).startswith(str(base)):
        raise HTTPException(403, "Access denied.")
    if not target.exists() or not target.is_file():
        raise HTTPException(404, "File not found.")

    size = target.stat().st_size
    ext  = target.suffix.lower()
    name = target.name

    is_text = (ext in _TEXT_EXTENSIONS or name in _TEXT_EXTENSIONS)

    if not is_text or size > _MAX_FILE_SIZE:
        return {"path": path, "content": None, "binary": True,
                "size": size, "message": "Binary or large file — not shown."}

    try:
        content = target.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        raise HTTPException(500, str(e))

    return {"path": path, "content": content, "binary": False,
            "size": size, "lines": content.count("\n") + 1}


@app.post("/api/repos/{repo_id}/file")
async def write_file(repo_id: str, payload: FileWritePayload) -> dict:
    repo = _get_repo_or_404(repo_id)
    base = Path(repo["local_path"]).resolve()
    target = (base / payload.path).resolve()

    if not str(target).startswith(str(base)):
        raise HTTPException(403, "Access denied.")

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(payload.content, encoding="utf-8")
    return {"ok": True, "path": payload.path,
            "size": target.stat().st_size, "saved_at": datetime.now(timezone.utc).isoformat()}


@app.post("/api/repos/{repo_id}/git")
async def git_op(repo_id: str, payload: GitOpPayload) -> dict:
    repo = _get_repo_or_404(repo_id)
    cwd  = repo["local_path"]

    cmd_map: dict[str, list[str]] = {
        "status": ["git", "status", "--short"],
        "diff":   ["git", "diff", "--stat"],
        "pull":   ["git", "pull"],
        "push":   ["git", "push"],
        "commit": ["git", "add", "--all"],
    }

    if payload.op not in cmd_map:
        raise HTTPException(400, f"Unknown op: {payload.op!r}")

    def _run(cmd: list[str]) -> str:
        try:
            r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
            return (r.stdout + r.stderr).strip()
        except Exception as e:
            return f"Error executing git command: {e}"

    if payload.op == "commit":
        out  = _run(["git", "add", "--all"])
        out += "\n" + _run(["git", "commit", "-m", payload.message])
    else:
        out = _run(cmd_map[payload.op])

    return {"ok": True, "op": payload.op, "output": out}


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline execution
# ─────────────────────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    _run_state["log"].append(f"[{time.strftime('%H:%M:%S')}] {msg}")


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
        from config import get_settings as _gs; _gs.cache_clear()
    except Exception:
        pass
    from main import run_pipeline_for_url

    _log(f"🚀 [{run_id}] {payload.url}")
    try:
        result = await run_pipeline_for_url(
            url=payload.url,
            scenario=payload.scenario or "Navigate the site as a new user and report any visual or functional bugs.",
            repo_path=payload.repo_path,
            model_name=None,
            verify=payload.verify,
            publish_pr=payload.publish_pr,
        )
        result["run_id"] = run_id
        if result.get("bug_found"):
            _log(f"🐛 Bug found ({result.get('severity','?')}) — {result.get('summary','')}")
        else:
            _log("✅ No bugs detected.")
        if result.get("fixed"):    _log("🔧 Fix applied.")
        if result.get("pr_url"):   _log(f"📬 PR: {result['pr_url']}")
        if result.get("error"):    _log(f"❌ {result['error']}")
        _run_state.update({"result": result, "status": "done"})
        _append_history(result)
        _update_target_last_run(payload.url, result)
        _log("🏁 Done.")
    except Exception as exc:
        _log(f"💥 {exc}")
        _run_state.update({"status": "error",
                           "result": {"error": str(exc), "run_id": run_id}})
        _append_history(_run_state["result"])


def _update_target_last_run(url: str, result: dict) -> None:
    targets = _load(TARGETS_FILE, "targets", [])
    for t in targets:
        if t["url"] == url:
            t["last_run"]    = datetime.now(timezone.utc).isoformat()
            t["last_result"] = result.get("severity", "none")
    _save(TARGETS_FILE, "targets", targets)


async def _run_queue_task(queue: list[dict]) -> None:
    _log(f"📋 Queue: {len(queue)} URLs")
    for i, item in enumerate(queue, 1):
        _log(f"\n── [{i}/{len(queue)}] {item['url']}")
        p = RunPayload(**{k: item[k] for k in RunPayload.model_fields if k in item})
        rid = str(uuid.uuid4())[:8]
        _run_state.update({"status": "running", "run_id": rid})
        await _execute_pipeline(p, rid)
        await asyncio.sleep(2)
    _save(QUEUE_FILE, "queue", [])
    _run_state["status"] = "done"
    _log("✅ Queue complete.")


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=True)
