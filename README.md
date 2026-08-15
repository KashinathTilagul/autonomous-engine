# 🤖 Autonomous UI Bug-Finding & Fix Engine

> **Automatically detect UI bugs with headless browser AI, repair them in your
> codebase via OpenHands, and publish a GitHub Pull Request — all without human
> intervention.**

---

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────────┐
│                       main.py  (Orchestrator)                      │
│                                                                    │
│  ① UIAuditAgent           ② CodeRepairAgent        ③ GitPublisher │
│  ─────────────────         ──────────────────        ─────────────│
│  browser-use + LLM  ──▶   OpenHands Docker    ──▶   PyGithub     │
│  Headless Chromium         REST API / Bash           Draft PR     │
│  ("Eyes")                  ("Hands")                              │
└────────────────────────────────────────────────────────────────────┘
```

| Component | Technology | Role |
|---|---|---|
| **LLM Brain** | `langchain-openai` → OpenRouter | Reasoning for QA & repair prompts |
| **Eyes** | `browser-use` + Playwright | Navigate, interact, observe the UI |
| **Hands** | OpenHands REST API (Docker) | Read/write source code, run tests |
| **Version Control** | `PyGithub` + `subprocess git` | Branch, commit, push, PR |

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.11+ | `python --version` |
| Playwright browsers | Installed automatically via `playwright install` |
| Docker | Required to run OpenHands locally |
| OpenHands container | See [Running OpenHands](#running-openhands) |
| OpenRouter API key | From [openrouter.ai/keys](https://openrouter.ai/keys) |
| GitHub PAT | Fine-grained token with **Contents** (rw) + **Pull requests** (rw) |

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/your-org/autonomous-engine.git
cd autonomous-engine

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
playwright install chromium      # download headless browser binary
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your real API keys and settings
```

Key variables:

```dotenv
OPENROUTER_API_KEY=sk-or-v1-...
MODEL_NAME=x-ai/grok-4          # any OpenRouter model slug
GITHUB_TOKEN=github_pat_...
REPO_NAME=your-org/your-repo
TARGET_URLS=https://staging.example.com
```

### 3. Start OpenHands (Docker)

```bash
# Pull and run the OpenHands image (exposes REST API on port 3000)
docker run -d \
  --name openhands \
  -p 3000:3000 \
  -v /path/to/your/repo:/workspace \
  ghcr.io/all-hands-ai/openhands:latest
```

> **Bind-mount tip:** the path you pass as `--repo-path` on the CLI must match
> the directory mounted at `/workspace` inside the container.

### 4. Run the engine

```bash
# Single URL, default scenario from .env
python main.py --url https://staging.example.com

# Custom scenario + override model
python main.py \
  --url https://staging.example.com \
  --scenario "Try to reset the password and check the email arrives" \
  --model openai/gpt-4o

# Audit multiple URLs (set TARGET_URLS in .env), skip PR
python main.py --no-pr

# Dry run — no git operations, no PR
python main.py --url https://example.com --no-pr --no-verify
```

---

## CLI Reference

```
usage: autonomous-engine [-h] [--url URL] [--scenario TEXT]
                         [--repo-path PATH] [--model MODEL]
                         [--no-verify] [--no-pr]

options:
  --url URL          Target URL to audit. Overrides TARGET_URLS in .env.
  --scenario TEXT    Test scenario. Overrides DEFAULT_TEST_SCENARIO in .env.
  --repo-path PATH   Absolute path to the local source repo (default: .)
  --model MODEL      Override LLM model (e.g. openai/gpt-4o)
  --no-verify        Skip the post-fix verification QA pass.
  --no-pr            Skip GitHub PR creation (dry-run mode).
```

---

## Project Structure

```
autonomous-engine/
├── .env.example              ← Template for secrets and settings
├── requirements.txt          ← Python dependencies
├── README.md                 ← This file
├── config.py                 ← Pydantic-settings config + LLM builder
├── agents/
│   ├── __init__.py
│   ├── qa_agent.py           ← UIAuditAgent  (browser-use "Eyes")
│   └── coder_agent.py        ← CodeRepairAgent (OpenHands "Hands")
├── utils/
│   ├── __init__.py
│   └── github_publisher.py  ← GitPublisher (branch + PR)
└── main.py                   ← Orchestrator + CLI entry point
```

---

## Pipeline Walkthrough

```
1. UIAuditAgent.run_test_flow(url, scenario)
   │
   ├── Spin up headless Chromium via browser-use
   ├── LLM drives navigation following the scenario
   ├── Capture DOM errors, console logs, visual anomalies
   └── Return structured BugReport dict
          │
          ▼ (has_bug == True)
2. CodeRepairAgent.trigger_fix(bug_report, repo_path)
   │
   ├── Build detailed repair prompt (bug context + file path + test cmds)
   ├── POST to OpenHands /api/v1/workspaces/default/tasks
   ├── Poll until status == "success" | "failed" | timeout
   └── Return repair result dict
          │
          ▼ (success == True)
3. UIAuditAgent.run_test_flow(url, scenario)   [optional --verify]
   └── Confirm has_bug == False after patch
          │
          ▼
4. GitPublisher.create_pull_request(branch, title, body)
   ├── git fetch → git checkout -b fix/...
   ├── git add --all && git commit
   ├── git push --set-upstream origin fix/...
   └── GitHub API: create draft PR with full bug report body
```

---

## Switching LLM Models

Edit `MODEL_NAME` in `.env` or pass `--model` on the CLI:

```bash
# Use Zhipu GLM
MODEL_NAME=zhipu/glm-4-9b python main.py --url https://example.com

# Or via CLI flag
python main.py --url https://example.com --model anthropic/claude-3.5-sonnet
```

All OpenRouter-supported models work as-is because `config.py` points
`ChatOpenAI` at the OpenRouter base URL.

---

## Environment Variables Reference

| Variable | Default | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | **required** | OpenRouter API key |
| `MODEL_NAME` | `x-ai/grok-4` | LLM model slug |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | OpenAI-compatible base URL |
| `LLM_TEMPERATURE` | `0.0` | Sampling temperature |
| `LLM_MAX_TOKENS` | `4096` | Max tokens per LLM call |
| `OPENHANDS_API_URL` | `http://localhost:3000` | OpenHands Docker service URL |
| `OPENHANDS_API_KEY` | *(empty)* | Bearer token for OpenHands auth |
| `OPENHANDS_TIMEOUT_SECONDS` | `600` | Task polling timeout |
| `GITHUB_TOKEN` | **required** | GitHub PAT |
| `REPO_NAME` | **required** | `owner/repo` format |
| `GITHUB_BASE_BRANCH` | `main` | PR target branch |
| `TARGET_URLS` | *(empty)* | Comma-separated URLs to audit |
| `DEFAULT_TEST_SCENARIO` | *(see .env.example)* | Fallback QA scenario |
| `BROWSER_MAX_STEPS` | `25` | Max browser steps per scenario |
| `LOG_LEVEL` | `INFO` | `DEBUG`/`INFO`/`WARNING`/`ERROR` |

---

## Running OpenHands

The engine communicates with OpenHands via its REST API.  The quickest way to
get it running locally:

```bash
# Latest stable image
docker pull ghcr.io/all-hands-ai/openhands:latest

docker run -d \
  --name openhands \
  -p 3000:3000 \
  -e SANDBOX_RUNTIME_CONTAINER_IMAGE=ghcr.io/all-hands-ai/runtime:latest \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v "$HOME/.openhands-state:/.openhands-state" \
  -v /absolute/path/to/repo:/workspace \
  --add-host host.docker.internal:host-gateway \
  ghcr.io/all-hands-ai/openhands:latest
```

Verify it is running:

```bash
curl http://localhost:3000/api/health
# → {"status":"ok"}
```

---

## Development

### Running tests (when you add them)

```bash
pip install pytest pytest-asyncio
pytest tests/ -v
```

### Linting

```bash
pip install ruff
ruff check .
```

### Type checking

```bash
pip install mypy
mypy . --ignore-missing-imports
```

---

## Contributing

1. Fork and create a feature branch.
2. Make changes, run `ruff check .` and `mypy .`.
3. Open a PR — the engine might even fix its own bugs! 🤖

---

## License

MIT — see [LICENSE](LICENSE) for details.
