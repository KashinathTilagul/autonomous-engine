# 🤖 Autonomous UI Bug-Finding & Fix Engine

> **Automatically detect UI bugs with headless browser AI, repair them in your
> codebase via OpenHands, and publish a GitHub Pull Request — all without human
> intervention.**

Powered by **[OpenRouter](https://openrouter.ai)**, allowing you to seamlessly use **any AI model** (`x-ai/grok-4`, `anthropic/claude-3.7-sonnet`, `openai/gpt-4o`, `deepseek/deepseek-r1`, `google/gemini-2.0-flash-001`, and hundreds more).

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Autonomous Engine                                │
│                                                                             │
│  ① UIAuditAgent                ② CodeRepairAgent         ③ GitPublisher     │
│  ──────────────────────        ───────────────────       ──────────────     │
│  browser-use + OpenRouter ──▶  OpenHands Docker   ──▶    PyGithub           │
│  (Headless Chromium Eyes)      REST API / Bash           Draft PR           │
│                                (Code Hands)                                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

| Component | Technology | Role |
|---|---|---|
| **LLM Brain** | `langchain-openai` → **OpenRouter** | Reasoning for QA & repair prompts across any model |
| **Eyes** | `browser-use` + Playwright | Navigate, interact, observe the DOM/UI |
| **Hands** | OpenHands REST API (Docker) | Read/write source code, run unit/integration tests |
| **Version Control** | `PyGithub` + `subprocess git` | Branch, commit, push, draft PR |
| **Control UI** | FastAPI + Vanilla JS | Target URLs, multi-repo code editor, live SSE logs, scheduler |

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.11+ | `python --version` |
| Playwright browsers | Installed via `playwright install chromium` |
| Docker | Required to run OpenHands locally |
| OpenRouter API key | From [openrouter.ai/keys](https://openrouter.ai/keys) |
| GitHub PAT | Fine-grained token with **Contents** (rw) + **Pull requests** (rw) |

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/KashinathTilagul/autonomous-engine.git
cd autonomous-engine

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
playwright install chromium      # download browser binaries
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env or configure directly in the Web UI!
```

Key variables:

```dotenv
OPENROUTER_API_KEY=sk-or-v1-...
MODEL_NAME=x-ai/grok-4          # Any OpenRouter model slug
GITHUB_TOKEN=github_pat_...
REPO_NAME=KashinathTilagul/autonomous-engine
```

### 3. Start the Web Control Dashboard

```bash
python server.py
# Open http://localhost:8080 in your browser
```

The Web Dashboard features:
- **▶ Run:** One-click automated test runs with real-time log streaming.
- **🎯 Targets:** Save, manage, and label staging/production URLs with audit history.
- **📁 Repos:** Multi-repo workspace with a built-in file tree, code editor, and git operations (`status`, `diff`, `commit`, `push`, `pull`).
- **⏰ Automate:** Cron or interval scheduling with desktop browser notifications.
- **📋 Queue:** Batch multi-URL audits.
- **🕘 History:** Persistent database of all runs, bug severities, and PR links.

---

## Switching Models with OpenRouter

You can use **any model** hosted on OpenRouter. Simply type or select the model ID in the UI settings or update `MODEL_NAME` in `.env`:

```bash
# Grok 4
MODEL_NAME=x-ai/grok-4

# Claude 3.7 Sonnet
MODEL_NAME=anthropic/claude-3.7-sonnet

# GPT-4o
MODEL_NAME=openai/gpt-4o

# DeepSeek R1
MODEL_NAME=deepseek/deepseek-r1

# Gemini 2.0 Flash
MODEL_NAME=google/gemini-2.0-flash-001
```

---

## Running OpenHands ("Hands")

The engine communicates with OpenHands via its REST API to perform code edits and test validation inside Docker:

```bash
docker run -d \
  --name openhands \
  -p 3000:3000 \
  -e SANDBOX_RUNTIME_CONTAINER_IMAGE=ghcr.io/all-hands-ai/runtime:latest \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v "$HOME/.openhands-state:/.openhands-state" \
  -v /absolute/path/to/your/repo:/workspace \
  --add-host host.docker.internal:host-gateway \
  ghcr.io/all-hands-ai/openhands:latest
```

---

## CLI Orchestrator (Headless Mode)

You can also run directly from the command line without the web server:

```bash
# Single URL audit
python main.py --url https://staging.example.com

# Custom scenario & model override
python main.py \
  --url https://staging.example.com \
  --scenario "Test account creation and payment flow" \
  --model anthropic/claude-3.7-sonnet

# Dry run (no PR creation)
python main.py --url https://staging.example.com --no-pr
```

---

## License

MIT
