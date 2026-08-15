"""
server.py  ·  Autonomous UI Bug Engine
─────────────────────────────────────────────────────────────────────────────
FastAPI web server with:
1. User Authentication (JWT + Session cookies, sign-up, sign-in, guest mode)
2. Beautiful Landing Page & Marketing Showcase
3. User-isolated Persistence (API keys, history, targets, repos per user)
4. Full Client-side LocalStorage sync + Serverless Database
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import mimetypes
import os
import re
import secrets
import subprocess
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

import uvicorn
from dotenv import dotenv_values, set_key
from fastapi import Cookie, Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

BASE_DIR   = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
INDEX_FILE = STATIC_DIR / "index.html"

def _get_writable_dir() -> Path:
    if os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
        tmp = Path(tempfile.gettempdir()) / "autonomous_engine"
        tmp.mkdir(parents=True, exist_ok=True)
        return tmp
    return BASE_DIR

STATE_DIR = _get_writable_dir()
DB_FILE   = STATE_DIR / ".db.json"

# In-memory primary database
_db: dict[str, Any] = {
    "users": {},         # email -> {id, name, email, password_hash, created_at}
    "sessions": {},      # token -> user_id
    "user_data": {},     # user_id -> {settings, history, targets, repos, queue}
}

def _load_db() -> None:
    global _db
    try:
        if DB_FILE.exists():
            data = json.loads(DB_FILE.read_text(encoding="utf-8"))
            _db.update(data)
    except Exception:
        pass

def _save_db() -> None:
    try:
        DB_FILE.write_text(json.dumps(_db, indent=2, default=str), encoding="utf-8")
    except Exception:
        pass

_load_db()

# Create default demo user if empty
if not _db["users"]:
    demo_id = "user_demo"
    _db["users"]["demo@example.com"] = {
        "id": demo_id,
        "name": "Developer Demo",
        "email": "demo@example.com",
        "password_hash": hashlib.sha256(b"password123").hexdigest(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _db["user_data"][demo_id] = {
        "settings": {
            "OMNIROUTE_API_KEY": "",
            "MODEL_NAME": "deepseek/deepseek-r1:free",
            "OMNIROUTE_BASE_URL": "https://api.omniroute.ai/v1",
            "LLM_TEMPERATURE": "0.0",
            "LLM_MAX_TOKENS": "4096",
            "GITHUB_TOKEN": "",
            "REPO_NAME": "KashinathTilagul/autonomous-engine",
            "GITHUB_BASE_BRANCH": "main",
            "BROWSER_MAX_STEPS": "25",
            "LOG_LEVEL": "INFO",
        },
        "history": [],
        "targets": [],
        "repos": [],
        "queue": [],
    }
    _save_db()

app = FastAPI(title="Autonomous UI Bug Engine", version="4.0.0")

_run_state: dict[str, Any] = {
    "status": "idle", "log": [], "result": None, "run_id": None, "user_id": None,
}


# ─────────────────────────────────────────────────────────────────────────────
# Auth Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode("utf-8")).hexdigest()

def get_current_user_id(
    authorization: Optional[str] = Header(None),
    auth_token: Optional[str] = Cookie(None),
) -> str:
    token = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization.replace("Bearer ", "").strip()
    elif auth_token:
        token = auth_token

    if token and token in _db["sessions"]:
        return _db["sessions"][token]
    
    # Return demo user ID as seamless fallback
    return "user_demo"


# ─────────────────────────────────────────────────────────────────────────────
# Routes – Auth API
# ─────────────────────────────────────────────────────────────────────────────

class SignUpPayload(BaseModel):
    name: str
    email: str
    password: str

class SignInPayload(BaseModel):
    email: str
    password: str

@app.post("/api/auth/signup")
async def signup(p: SignUpPayload, response: Response) -> dict:
    email = p.email.strip().lower()
    if not email or not p.password:
        raise HTTPException(400, "Email and password required.")
    if email in _db["users"]:
        raise HTTPException(400, "Account already exists with this email.")

    user_id = f"user_{secrets.token_hex(6)}"
    _db["users"][email] = {
        "id": user_id,
        "name": p.name.strip() or email.split("@")[0],
        "email": email,
        "password_hash": _hash_pw(p.password),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _db["user_data"][user_id] = {
        "settings": {
            "OMNIROUTE_API_KEY": "",
            "MODEL_NAME": "deepseek/deepseek-r1:free",
            "OMNIROUTE_BASE_URL": "https://api.omniroute.ai/v1",
            "LLM_TEMPERATURE": "0.0",
            "LLM_MAX_TOKENS": "4096",
            "GITHUB_TOKEN": "",
            "REPO_NAME": "KashinathTilagul/autonomous-engine",
            "GITHUB_BASE_BRANCH": "main",
            "BROWSER_MAX_STEPS": "25",
            "LOG_LEVEL": "INFO",
        },
        "history": [],
        "targets": [],
        "repos": [],
        "queue": [],
    }

    token = f"tok_{secrets.token_urlsafe(24)}"
    _db["sessions"][token] = user_id
    _save_db()

    response.set_cookie("auth_token", token, httponly=True, max_age=86400 * 30, samesite="lax")
    return {
        "ok": True,
        "token": token,
        "user": {"id": user_id, "name": _db["users"][email]["name"], "email": email},
    }

@app.post("/api/auth/signin")
async def signin(p: SignInPayload, response: Response) -> dict:
    email = p.email.strip().lower()
    user = _db["users"].get(email)
    if not user or user["password_hash"] != _hash_pw(p.password):
        raise HTTPException(401, "Invalid email or password.")

    token = f"tok_{secrets.token_urlsafe(24)}"
    _db["sessions"][token] = user["id"]
    _save_db()

    response.set_cookie("auth_token", token, httponly=True, max_age=86400 * 30, samesite="lax")
    return {
        "ok": True,
        "token": token,
        "user": {"id": user["id"], "name": user["name"], "email": user["email"]},
    }

@app.get("/api/auth/me")
async def get_me(user_id: str = Depends(get_current_user_id)) -> dict:
    for u in _db["users"].values():
        if u["id"] == user_id:
            return {"authenticated": True, "user": {"id": u["id"], "name": u["name"], "email": u["email"]}}
    return {"authenticated": False, "user": None}

@app.post("/api/auth/signout")
async def signout(response: Response, auth_token: Optional[str] = Cookie(None)) -> dict:
    if auth_token and auth_token in _db["sessions"]:
        del _db["sessions"][auth_token]
        _save_db()
    response.delete_cookie("auth_token")
    return {"ok": True}


# ─────────────────────────────────────────────────────────────────────────────
# Routes – UI
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_ui() -> HTMLResponse:
    if INDEX_FILE.exists():
        return HTMLResponse(content=INDEX_FILE.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>Autonomous UI Bug Engine. Please build static assets.</h2>")


# ─────────────────────────────────────────────────────────────────────────────
# Routes – User-Specific Settings
# ─────────────────────────────────────────────────────────────────────────────

_SENSITIVE = {"OMNIROUTE_API_KEY", "OPENROUTER_API_KEY", "GITHUB_TOKEN", "OPENHANDS_API_KEY"}
_MASKED_RE = re.compile(r"^[•]+$")

def _mask(key: str, val: str) -> str:
    if key in _SENSITIVE and val and len(val) > 4:
        return "•" * (len(val) - 4) + val[-4:]
    return val

@app.get("/api/settings")
async def get_settings(user_id: str = Depends(get_current_user_id)) -> dict:
    udata = _db["user_data"].setdefault(user_id, {})
    settings = udata.setdefault("settings", {
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
    })
    masked = {k: _mask(k, v) for k, v in settings.items()}
    return masked

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
async def save_settings(
    p: SettingsPayload,
    user_id: str = Depends(get_current_user_id),
) -> dict:
    udata = _db["user_data"].setdefault(user_id, {})
    current = udata.setdefault("settings", {})
    updates = {}
    for k, v in p.model_dump().items():
        if isinstance(v, str) and _MASKED_RE.fullmatch(v):
            continue
        current[k] = v
        updates[k] = v
    _save_db()
    return {"ok": True, "saved": list(updates.keys())}


# ─────────────────────────────────────────────────────────────────────────────
# Routes – Pipeline Run
# ─────────────────────────────────────────────────────────────────────────────

class RunPayload(BaseModel):
    url: str
    scenario: str = ""
    repo_path: str = "."
    verify: bool = True
    publish_pr: bool = False

@app.post("/api/run")
async def start_run(
    p: RunPayload,
    user_id: str = Depends(get_current_user_id),
) -> dict:
    if _run_state["status"] == "running":
        raise HTTPException(409, "A run is already in progress.")
    run_id = str(uuid.uuid4())[:8]
    _run_state.update({
        "status": "running", "log": [], "result": None, "run_id": run_id, "user_id": user_id,
    })
    asyncio.create_task(_execute_pipeline(p, run_id, user_id))
    return {"ok": True, "run_id": run_id}

@app.get("/api/run/status")
async def run_status() -> dict:
    return {
        "status": _run_state["status"],
        "run_id": _run_state["run_id"],
        "log": _run_state["log"][-200:],
        "result": _run_state["result"],
    }

@app.get("/api/run/stream")
async def run_stream() -> StreamingResponse:
    return StreamingResponse(
        _sse_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

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

async def _execute_pipeline(payload: RunPayload, run_id: str, user_id: str) -> None:
    from main import run_pipeline_for_url

    _log(f"🚀 [{run_id}] Initiating inspection for: {payload.url}")
    try:
        result = await run_pipeline_for_url(
            url=payload.url,
            scenario=payload.scenario or "Inspect user interface and functionality.",
            repo_path=payload.repo_path,
            model_name=None,
            verify=payload.verify,
            publish_pr=payload.publish_pr,
        )
        result["run_id"] = run_id
        if result.get("bug_found"):
            _log(f"🐛 Result: Bug Detected ({result.get('severity','?')})")
            _log(f"   Summary: {result.get('summary','')}")
            for de in result.get("dom_errors", []):
                _log(f"   ⚠️ Issue detail: {de}")
            for st in result.get("steps_taken", []):
                _log(f"      {st}")
        else:
            _log("✅ Result: No bugs detected.")
            _log(f"   Summary: {result.get('summary','')}")

        if result.get("fixed"):
            _log("🔧 Auto-fix plan prepared.")
        if result.get("pr_url"):
            _log(f"📬 GitHub PR created: {result['pr_url']}")
        if result.get("error"):
            _log(f"❌ Details: {result['error']}")

        _run_state.update({"result": result, "status": "done"})
        
        # Save to user history
        udata = _db["user_data"].setdefault(user_id, {})
        history = udata.setdefault("history", [])
        history.insert(0, {
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "url": payload.url,
            "bug_found": result.get("bug_found", False),
            "severity": result.get("severity", "none"),
            "fixed": result.get("fixed", False),
            "verified": result.get("verified"),
            "pr_url": result.get("pr_url"),
            "error": result.get("error"),
            "summary": result.get("summary", ""),
        })
        udata["history"] = history[:100]
        _save_db()
        _log("🏁 Audit Completed.")
    except Exception as exc:
        _log(f"💥 {exc}")
        err_res = {"error": str(exc), "run_id": run_id, "bug_found": True, "severity": "critical"}
        _run_state.update({"status": "error", "result": err_res})
        udata = _db["user_data"].setdefault(user_id, {})
        history = udata.setdefault("history", [])
        history.insert(0, {
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "url": payload.url,
            "bug_found": True,
            "severity": "critical",
            "error": str(exc),
            "summary": f"Audit failed: {exc}",
        })
        _save_db()


# ─────────────────────────────────────────────────────────────────────────────
# Routes – History, Targets, Repos, Queue
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/history")
async def get_history(user_id: str = Depends(get_current_user_id)) -> dict:
    udata = _db["user_data"].setdefault(user_id, {})
    return {"history": udata.get("history", [])}

@app.delete("/api/history")
async def clear_history(user_id: str = Depends(get_current_user_id)) -> dict:
    udata = _db["user_data"].setdefault(user_id, {})
    udata["history"] = []
    _save_db()
    return {"ok": True}

class TargetPayload(BaseModel):
    name: str = ""
    url: str
    scenario: str = ""
    notes: str = ""
    tags: list[str] = []

@app.get("/api/targets")
async def get_targets(user_id: str = Depends(get_current_user_id)) -> dict:
    udata = _db["user_data"].setdefault(user_id, {})
    return {"targets": udata.get("targets", [])}

@app.post("/api/targets")
async def add_target(p: TargetPayload, user_id: str = Depends(get_current_user_id)) -> dict:
    udata = _db["user_data"].setdefault(user_id, {})
    targets = udata.setdefault("targets", [])
    entry = {
        **p.model_dump(),
        "id": str(uuid.uuid4())[:8],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "last_run": None,
        "last_result": None,
    }
    targets.append(entry)
    _save_db()
    return {"ok": True, "id": entry["id"]}

@app.delete("/api/targets/{target_id}")
async def delete_target(target_id: str, user_id: str = Depends(get_current_user_id)) -> dict:
    udata = _db["user_data"].setdefault(user_id, {})
    udata["targets"] = [t for t in udata.get("targets", []) if t["id"] != target_id]
    _save_db()
    return {"ok": True}

@app.get("/api/repos")
async def get_repos(user_id: str = Depends(get_current_user_id)) -> dict:
    udata = _db["user_data"].setdefault(user_id, {})
    return {"repos": udata.get("repos", [])}

class RepoPayload(BaseModel):
    name: str
    local_path: str
    github_url: str = ""
    branch: str = "main"

@app.post("/api/repos")
async def add_repo(p: RepoPayload, user_id: str = Depends(get_current_user_id)) -> dict:
    udata = _db["user_data"].setdefault(user_id, {})
    repos = udata.setdefault("repos", [])
    entry = {**p.model_dump(), "id": str(uuid.uuid4())[:8]}
    repos.append(entry)
    _save_db()
    return {"ok": True, "id": entry["id"]}

@app.delete("/api/repos/{repo_id}")
async def delete_repo(repo_id: str, user_id: str = Depends(get_current_user_id)) -> dict:
    udata = _db["user_data"].setdefault(user_id, {})
    udata["repos"] = [r for r in udata.get("repos", []) if r["id"] != repo_id]
    _save_db()
    return {"ok": True}

@app.get("/api/queue")
async def get_queue(user_id: str = Depends(get_current_user_id)) -> dict:
    udata = _db["user_data"].setdefault(user_id, {})
    return {"queue": udata.get("queue", [])}

class QueueItem(BaseModel):
    url: str
    scenario: str = ""
    repo_path: str = "."
    verify: bool = True
    publish_pr: bool = False

@app.post("/api/queue")
async def add_queue(p: QueueItem, user_id: str = Depends(get_current_user_id)) -> dict:
    udata = _db["user_data"].setdefault(user_id, {})
    q = udata.setdefault("queue", [])
    entry = {**p.model_dump(), "id": str(uuid.uuid4())[:8], "added_at": datetime.now(timezone.utc).isoformat()}
    q.append(entry)
    _save_db()
    return {"ok": True, "id": entry["id"]}

@app.delete("/api/queue/{item_id}")
async def remove_queue(item_id: str, user_id: str = Depends(get_current_user_id)) -> dict:
    udata = _db["user_data"].setdefault(user_id, {})
    udata["queue"] = [i for i in udata.get("queue", []) if i.get("id") != item_id]
    _save_db()
    return {"ok": True}

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=True)
