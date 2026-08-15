"""
agents/coder_agent.py
─────────────────────
CodeRepairAgent – the "Hands" of the autonomous engine.

Responsibilities
────────────────
* Accept a structured bug report produced by UIAuditAgent.
* Build a rich natural-language prompt that describes the bug, points to the
  relevant source file context, and asks OpenHands to:
    1. Locate the defective code.
    2. Apply a minimal, test-passing fix.
    3. Run the project's unit/integration tests via Bash.
    4. Confirm success before marking the task as done.
* Poll the OpenHands REST API until the task reaches a terminal state.
* Return a structured result dict so the main pipeline can branch on success
  or failure.

OpenHands API contract (v0.x)
──────────────────────────────
POST /api/v1/workspaces/default/tasks
  Body: { "prompt": str, "repo_path": str }
  Response: { "task_id": str, ... }

GET /api/v1/workspaces/default/tasks/{task_id}
  Response: { "status": "pending"|"running"|"success"|"failed", ... }

These endpoints match the OpenHands Docker image's default REST server.
Adjust ``_TASK_PATH`` and ``_STATUS_PATH`` if your version differs.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import httpx
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config import get_settings

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# OpenHands API path constants
# ─────────────────────────────────────────────────────────────────────────────

_TASK_PATH = "/api/v1/workspaces/default/tasks"
_POLL_INTERVAL_SECONDS = 10
_TERMINAL_STATUSES = {"success", "failed", "error", "cancelled"}


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder
# ─────────────────────────────────────────────────────────────────────────────

_REPAIR_PROMPT_TEMPLATE = """\
## Autonomous Code Repair Task

You are an expert software engineer. A UI QA audit discovered the following bug.
Your job is to locate, fix, and verify the defect in the local repository.

─────────────────────────────────────────────────────────
### Bug Report
─────────────────────────────────────────────────────────
Target URL  : {target_url}
Severity    : {severity}
Summary     : {summary}

Steps Taken by QA Agent:
{steps_taken}

DOM Errors Captured:
{dom_errors}

Visual Anomalies:
{visual_anomalies}

─────────────────────────────────────────────────────────
### Repository Context
─────────────────────────────────────────────────────────
Local repository path: {local_repo_path}

─────────────────────────────────────────────────────────
### Instructions
─────────────────────────────────────────────────────────
1. Read the relevant source files to understand the codebase structure.
2. Identify the root cause of the bug described above.
3. Apply the minimal change necessary to fix the bug without introducing
   regressions.
4. After editing, run the project's test suite using the Bash tool:
      cd {local_repo_path} && <your test command>
   (Detect the test runner from package.json, pyproject.toml, Makefile, etc.)
5. If tests pass, output a summary of what you changed and why.
6. If tests fail, attempt to fix the failures before declaring success.
7. Do NOT commit or push changes – the orchestrator handles git operations.

Begin now.
"""


def _format_list(items: list[str]) -> str:
    """Convert a list of strings to a numbered text block."""
    if not items:
        return "  (none)"
    return "\n".join(f"  {i + 1}. {item}" for i, item in enumerate(items))


# ─────────────────────────────────────────────────────────────────────────────
# Agent class
# ─────────────────────────────────────────────────────────────────────────────

class CodeRepairAgent:
    """
    Dispatches bug-fix tasks to a locally running OpenHands Docker service.

    OpenHands acts as an agentic code editor with Bash, file-read/write, and
    test-running capabilities.  This class is responsible only for:

    * Building the repair prompt.
    * Submitting the task via HTTP POST.
    * Polling until completion.
    * Returning a structured result to the pipeline.

    Parameters
    ----------
    openhands_api_url : str, optional
        Override the OpenHands base URL (default: read from settings).
    """

    def __init__(
        self,
        *,
        openhands_api_url: Optional[str] = None,
    ) -> None:
        cfg = get_settings()
        self._base_url = (openhands_api_url or cfg.openhands_api_url).rstrip("/")
        self._timeout = cfg.openhands_timeout_seconds

        # Build optional auth header.
        self._headers: dict[str, str] = {"Content-Type": "application/json"}
        if cfg.openhands_api_key:
            self._headers["Authorization"] = (
                f"Bearer {cfg.openhands_api_key.get_secret_value()}"
            )

        logger.info("CodeRepairAgent initialised", extra={"base_url": self._base_url})

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def trigger_fix(
        self,
        bug_report: dict[str, Any],
        local_repo_path: str,
    ) -> dict[str, Any]:
        """
        Submit a repair task to OpenHands and block until it finishes.

        Parameters
        ----------
        bug_report : dict
            The ``BugReport.to_dict()`` payload from UIAuditAgent.
        local_repo_path : str
            Absolute path to the repository on the host that OpenHands can
            access (bind-mounted into the container).

        Returns
        -------
        dict
            Keys:
            * ``success`` (bool) – True when OpenHands reports "success".
            * ``task_id`` (str)  – OpenHands task identifier.
            * ``status`` (str)   – Terminal status string.
            * ``details`` (dict) – Raw OpenHands task-status payload.
            * ``error`` (str | None) – Error description if the call failed.
        """
        prompt = self._build_prompt(bug_report, local_repo_path)

        try:
            task_id = self._submit_task(prompt, local_repo_path)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to submit task to OpenHands")
            return {
                "success": False,
                "task_id": None,
                "status": "submission_failed",
                "details": {},
                "error": str(exc),
            }

        logger.info("OpenHands task submitted", extra={"task_id": task_id})

        try:
            final_status = self._poll_until_done(task_id)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Error while polling OpenHands task %s", task_id)
            return {
                "success": False,
                "task_id": task_id,
                "status": "polling_failed",
                "details": {},
                "error": str(exc),
            }

        success = final_status.get("status") == "success"
        logger.info(
            "OpenHands task finished",
            extra={"task_id": task_id, "status": final_status.get("status")},
        )
        return {
            "success": success,
            "task_id": task_id,
            "status": final_status.get("status", "unknown"),
            "details": final_status,
            "error": None if success else final_status.get("error"),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_prompt(bug_report: dict[str, Any], local_repo_path: str) -> str:
        """Render the repair prompt from the bug report and repo path."""
        return _REPAIR_PROMPT_TEMPLATE.format(
            target_url=bug_report.get("target_url", "N/A"),
            severity=bug_report.get("severity", "unknown"),
            summary=bug_report.get("summary", "No summary provided."),
            steps_taken=_format_list(bug_report.get("steps_taken", [])),
            dom_errors=_format_list(bug_report.get("dom_errors", [])),
            visual_anomalies=_format_list(bug_report.get("visual_anomalies", [])),
            local_repo_path=local_repo_path,
        )

    @retry(
        retry=retry_if_exception_type(httpx.TransportError),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _submit_task(self, prompt: str, repo_path: str) -> str:
        """
        POST the repair task to OpenHands.

        Returns the ``task_id`` string from the response.

        Retries up to 5 times on transient network errors with exponential
        back-off (2 s → 4 s → 8 s → 16 s → 30 s).
        """
        url = f"{self._base_url}{_TASK_PATH}"
        payload = {
            "prompt": prompt,
            "repo_path": repo_path,
        }

        with httpx.Client(timeout=30) as client:
            response = client.post(url, json=payload, headers=self._headers)

        if response.status_code not in (200, 201, 202):
            raise RuntimeError(
                f"OpenHands returned HTTP {response.status_code}: {response.text}"
            )

        data = response.json()
        task_id: str = data.get("task_id") or data.get("id") or data["task_id"]
        return task_id

    def _poll_until_done(self, task_id: str) -> dict[str, Any]:
        """
        Poll the OpenHands task-status endpoint every ``_POLL_INTERVAL_SECONDS``
        seconds until the task reaches a terminal state or the global timeout
        is exceeded.

        Returns
        -------
        dict
            The final task-status payload from OpenHands.

        Raises
        ------
        TimeoutError
            If the task has not completed within ``self._timeout`` seconds.
        """
        url = f"{self._base_url}{_TASK_PATH}/{task_id}"
        deadline = time.monotonic() + self._timeout
        attempt = 0

        with httpx.Client(timeout=15) as client:
            while time.monotonic() < deadline:
                attempt += 1
                try:
                    response = client.get(url, headers=self._headers)
                    response.raise_for_status()
                    data: dict[str, Any] = response.json()
                except httpx.HTTPStatusError as exc:
                    logger.warning(
                        "HTTP error polling task %s (attempt %d): %s",
                        task_id,
                        attempt,
                        exc,
                    )
                    time.sleep(_POLL_INTERVAL_SECONDS)
                    continue
                except httpx.TransportError as exc:
                    logger.warning(
                        "Transport error polling task %s (attempt %d): %s",
                        task_id,
                        attempt,
                        exc,
                    )
                    time.sleep(_POLL_INTERVAL_SECONDS)
                    continue

                status = data.get("status", "")
                logger.debug(
                    "Task %s status=%s (attempt %d)", task_id, status, attempt
                )

                if status in _TERMINAL_STATUSES:
                    return data

                time.sleep(_POLL_INTERVAL_SECONDS)

        raise TimeoutError(
            f"OpenHands task {task_id!r} did not complete within "
            f"{self._timeout} seconds."
        )
