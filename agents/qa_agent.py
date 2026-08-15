"""
agents/qa_agent.py
──────────────────
UIAuditAgent – the "Eyes" of the autonomous engine.

Responsibilities
────────────────
* Accept a target URL and a plain-English test scenario.
* Use browser-use's ``Agent`` class to drive a headless Chromium session.
* Collect step-by-step browser actions and observe the DOM / visual state.
* On completion (pass or fail), return a structured ``BugReport`` dict so the
  rest of the pipeline can decide whether a fix is needed.

Design decisions
────────────────
* The class is fully asynchronous to avoid blocking the event loop while the
  browser performs navigation.
* All browser-use configuration (LLM, step limit) is read from ``config.py``
  so no secrets are hard-coded here.
* Exceptions are caught, structured, and surfaced as a bug report rather than
  raw stack traces – this keeps the main pipeline logic simple.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Optional

# browser-use public API  (>=0.1.40)
from browser_use import Agent as BrowserAgent
from browser_use import BrowserConfig, BrowserContextConfig

from config import build_llm, get_settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BugReport:
    """
    Structured representation of a QA finding.

    All fields are JSON-serialisable so the report can be stored, logged, and
    forwarded to the CodeRepairAgent without further transformation.
    """

    # ── Classification ────────────────────────────────────────────────────────
    has_bug: bool
    """True if the agent detected at least one defect."""

    severity: str
    """One of: 'critical', 'high', 'medium', 'low', 'none'."""

    # ── Context ───────────────────────────────────────────────────────────────
    target_url: str
    """The URL that was tested."""

    test_scenario: str
    """The natural-language scenario that was exercised."""

    # ── Evidence ──────────────────────────────────────────────────────────────
    summary: str
    """One-sentence description of the defect (or 'No bugs found.')."""

    steps_taken: list[str] = field(default_factory=list)
    """Ordered list of browser actions the agent performed."""

    dom_errors: list[str] = field(default_factory=list)
    """Console errors, broken selectors, or missing elements captured."""

    visual_anomalies: list[str] = field(default_factory=list)
    """Description of layout shifts, overlapping elements, etc."""

    screenshot_paths: list[str] = field(default_factory=list)
    """Absolute paths to screenshots captured during the session."""

    raw_agent_output: Optional[str] = None
    """Full text output from browser-use (for debugging)."""

    # ── Meta ──────────────────────────────────────────────────────────────────
    elapsed_seconds: float = 0.0
    error_message: Optional[str] = None
    """Set when the QA run itself threw an exception."""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "has_bug": self.has_bug,
            "severity": self.severity,
            "target_url": self.target_url,
            "test_scenario": self.test_scenario,
            "summary": self.summary,
            "steps_taken": self.steps_taken,
            "dom_errors": self.dom_errors,
            "visual_anomalies": self.visual_anomalies,
            "screenshot_paths": self.screenshot_paths,
            "raw_agent_output": self.raw_agent_output,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "error_message": self.error_message,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Prompt templates
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an expert QA engineer performing automated UI testing.
Your goal is to exercise the provided scenario, observe the application's
behaviour, and produce a precise bug report.

## Rules
1. Follow the test scenario step-by-step.
2. After completing or failing each step, describe what you observed.
3. At the very end, output a JSON block wrapped in triple back-ticks with the
   key "bug_report" containing these fields:
      - has_bug (bool)
      - severity ("critical"|"high"|"medium"|"low"|"none")
      - summary (str, ≤ 200 chars)
      - steps_taken (list[str])
      - dom_errors (list[str])
      - visual_anomalies (list[str])

4. Do NOT invent bugs that you did not observe.
5. If unsure whether something is a bug, mark severity as "low" and describe it.
"""

_USER_PROMPT_TEMPLATE = """\
Target URL: {target_url}

Test Scenario:
{test_scenario}

Begin testing now.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Agent class
# ─────────────────────────────────────────────────────────────────────────────

class UIAuditAgent:
    """
    Drives a headless browser to detect UI bugs in a web application.

    Parameters
    ----------
    model_name : str, optional
        Override the LLM model for this specific agent instance.
    headless : bool
        Run Chromium in headless mode (default: True).
    """

    def __init__(
        self,
        *,
        model_name: Optional[str] = None,
        headless: bool = True,
    ) -> None:
        self._settings = get_settings()
        self._llm = build_llm(model_name=model_name)
        self._headless = headless
        logger.info(
            "UIAuditAgent initialised",
            extra={"model": model_name or self._settings.model_name},
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    async def run_test_flow(
        self,
        target_url: str,
        user_scenario: str,
    ) -> dict[str, Any]:
        """
        Navigate *target_url*, exercise *user_scenario*, and return a bug report.

        Parameters
        ----------
        target_url : str
            The fully-qualified URL to test (e.g. ``"https://example.com"``).
        user_scenario : str
            Plain-English description of the test case to execute.

        Returns
        -------
        dict
            A ``BugReport.to_dict()`` payload.  ``has_bug`` is ``False`` when
            no defects are detected.  ``error_message`` is set when the agent
            itself fails (network error, LLM timeout, etc.).
        """
        logger.info("Starting QA audit", extra={"url": target_url})
        start = time.monotonic()

        try:
            raw_output = await self._run_browser_agent(target_url, user_scenario)
            report = self._parse_agent_output(raw_output, target_url, user_scenario)
        except asyncio.TimeoutError:
            logger.error("Browser agent timed out for %s", target_url)
            report = self._error_report(
                target_url=target_url,
                test_scenario=user_scenario,
                error="Browser agent timed out before completing the scenario.",
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Browser agent raised an unexpected exception")
            report = self._error_report(
                target_url=target_url,
                test_scenario=user_scenario,
                error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            )

        report.elapsed_seconds = time.monotonic() - start
        result = report.to_dict()
        logger.info(
            "QA audit complete",
            extra={
                "url": target_url,
                "has_bug": result["has_bug"],
                "severity": result["severity"],
                "elapsed_s": result["elapsed_seconds"],
            },
        )
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    async def _run_browser_agent(
        self,
        target_url: str,
        user_scenario: str,
    ) -> str:
        """
        Instantiate a browser-use Agent, run the scenario, and return its
        full text output.
        """
        task = _USER_PROMPT_TEMPLATE.format(
            target_url=target_url,
            test_scenario=user_scenario,
        )

        browser_config = BrowserConfig(headless=self._headless)
        context_config = BrowserContextConfig(
            wait_for_network_idle_page_load_time=3.0,
        )

        agent = BrowserAgent(
            task=task,
            llm=self._llm,
            max_actions_per_step=5,
            browser_config=browser_config,
            browser_context_config=context_config,
            system_prompt_class=None,   # use default browser-use system prompt
        )

        # Run with a hard timeout to avoid hanging indefinitely.
        history = await asyncio.wait_for(
            agent.run(max_steps=self._settings.browser_max_steps),
            timeout=self._settings.openhands_timeout_seconds,
        )

        # ``history.final_result()`` returns the agent's last message string.
        return history.final_result() or ""

    @staticmethod
    def _parse_agent_output(
        raw: str,
        target_url: str,
        test_scenario: str,
    ) -> BugReport:
        """
        Extract the structured JSON block from the agent's free-form output.

        The agent is instructed to embed a JSON block like::

            ```json
            {
                "bug_report": { ... }
            }
            ```

        If no such block is found, the entire output is treated as a summary
        and the report is conservatively marked as having a low-severity bug.
        """
        # Attempt to locate the JSON fence.
        import re

        json_match = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```",
            raw,
            flags=re.DOTALL | re.IGNORECASE,
        )

        if json_match:
            try:
                payload = json.loads(json_match.group(1))
                data: dict = payload.get("bug_report", payload)
                return BugReport(
                    has_bug=bool(data.get("has_bug", False)),
                    severity=data.get("severity", "low"),
                    target_url=target_url,
                    test_scenario=test_scenario,
                    summary=data.get("summary", "See raw output."),
                    steps_taken=data.get("steps_taken", []),
                    dom_errors=data.get("dom_errors", []),
                    visual_anomalies=data.get("visual_anomalies", []),
                    raw_agent_output=raw,
                )
            except json.JSONDecodeError as exc:
                logger.warning("Failed to parse JSON block from agent output: %s", exc)

        # Fallback: treat entire output as a human-readable summary.
        has_bug = any(
            kw in raw.lower()
            for kw in ("error", "bug", "broken", "fail", "missing", "not found", "404")
        )
        return BugReport(
            has_bug=has_bug,
            severity="low" if has_bug else "none",
            target_url=target_url,
            test_scenario=test_scenario,
            summary=raw[:200] if raw else "Agent produced no output.",
            raw_agent_output=raw,
        )

    @staticmethod
    def _error_report(
        *,
        target_url: str,
        test_scenario: str,
        error: str,
    ) -> BugReport:
        """Build a BugReport that signals an infrastructure-level failure."""
        return BugReport(
            has_bug=True,
            severity="critical",
            target_url=target_url,
            test_scenario=test_scenario,
            summary="QA agent encountered an infrastructure error.",
            error_message=error,
        )
