"""
agents/qa_agent.py
──────────────────
UIAuditAgent – Intelligent, Cloud & Local Adaptive Web QA Engine.

Runs autonomously on both cloud deployments (Vercel) and local machines:
1. Cloud Mode (HTTP DOM & Visual Semantic Audit): Fetches live page content,
   DOM nodes, forms, links, and runs comprehensive LLM-driven UX/functional audits.
2. Headless Browser Mode (Playwright/browser-use): If Chromium/Playwright is
   available, drives full browser sessions with real-time UI interaction.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx
from config import build_llm, get_settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BugReport:
    has_bug: bool
    severity: str  # 'critical', 'high', 'medium', 'low', 'none'
    target_url: str
    test_scenario: str
    summary: str
    steps_taken: list[str] = field(default_factory=list)
    dom_errors: list[str] = field(default_factory=list)
    visual_anomalies: list[str] = field(default_factory=list)
    screenshot_paths: list[str] = field(default_factory=list)
    raw_agent_output: Optional[str] = None
    elapsed_seconds: float = 0.0
    error_message: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
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
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

_AUDIT_PROMPT_TEMPLATE = """\
You are an elite QA Engineer and UI/UX Bug Auditor inspecting the target webpage.

Target URL: {target_url}
Test Scenario: {test_scenario}

Webpage HTTP Status: {status_code}
Page Title: {title}
DOM / Content Snapshot:
```html
{content_snippet}
```

Instructions:
1. Carefully evaluate if there are UI bugs, console errors, 404/broken assets, broken navigation, missing form fields, layout inconsistencies, or broken scenario expectations.
2. Return your structured findings strictly in JSON format inside triple backticks:
```json
{{
  "bug_report": {{
    "has_bug": true,
    "severity": "critical" | "high" | "medium" | "low" | "none",
    "summary": "Concise summary of the bug detected or 'No bugs found'",
    "steps_taken": ["Step 1...", "Step 2..."],
    "dom_errors": ["Error detail 1..."],
    "visual_anomalies": ["Visual/layout anomaly 1..."]
  }}
}}
```
"""


# ─────────────────────────────────────────────────────────────────────────────
# Agent class
# ─────────────────────────────────────────────────────────────────────────────

class UIAuditAgent:
    """
    Cloud-native & local adaptive UI QA agent.
    Works seamlessly on serverless platforms (Vercel) as well as local machines.
    """

    def __init__(
        self,
        *,
        model_name: Optional[str] = None,
        headless: bool = True,
    ) -> None:
        self._settings = get_settings()
        self._model_name = model_name
        self._headless = headless

    async def run_test_flow(
        self,
        target_url: str,
        user_scenario: str,
    ) -> dict[str, Any]:
        start = time.monotonic()
        logger.info("Starting QA audit for: %s", target_url)

        # 1. Attempt full browser-use automation if Playwright is locally available
        try:
            from browser_use import Agent as BrowserAgent
            from browser_use import BrowserConfig, BrowserContextConfig

            llm = build_llm(model_name=self._model_name)
            agent = BrowserAgent(
                task=f"Navigate to {target_url} and execute: {user_scenario}",
                llm=llm,
                browser_config=BrowserConfig(headless=self._headless),
                browser_context_config=BrowserContextConfig(wait_for_network_idle_page_load_time=2.0),
            )
            history = await asyncio.wait_for(
                agent.run(max_steps=self._settings.browser_max_steps),
                timeout=self._settings.openhands_timeout_seconds,
            )
            raw = history.final_result() or ""
            report = self._parse_agent_output(raw, target_url, user_scenario)
            report.elapsed_seconds = time.monotonic() - start
            return report.to_dict()

        except Exception as browser_exc:
            logger.info("Native browser engine fallback to cloud HTTP inspector: %s", browser_exc)

        # 2. Serverless / Cloud-native HTTP DOM & Semantic QA Inspector
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=25.0) as client:
                resp = await client.get(target_url, headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
                })

            status_code = resp.status_code
            html_text = resp.text

            # Extract title
            title_match = re.search(r"<title>(.*?)</title>", html_text, re.IGNORECASE | re.DOTALL)
            title = title_match.group(1).strip() if title_match else "No title found"

            # Clean and truncate HTML for LLM context
            clean_snippet = re.sub(r"<script.*?</script>", "", html_text, flags=re.DOTALL | re.IGNORECASE)
            clean_snippet = re.sub(r"<style.*?</style>", "", clean_snippet, flags=re.DOTALL | re.IGNORECASE)
            clean_snippet = re.sub(r"\s+", " ", clean_snippet)[:10000]

            prompt = _AUDIT_PROMPT_TEMPLATE.format(
                target_url=target_url,
                test_scenario=user_scenario,
                status_code=status_code,
                title=title,
                content_snippet=clean_snippet,
            )

            llm = build_llm(model_name=self._model_name)
            response = await llm.ainvoke(prompt)
            raw_output = response.content if hasattr(response, "content") else str(response)

            report = self._parse_agent_output(raw_output, target_url, user_scenario)
            report.elapsed_seconds = time.monotonic() - start
            return report.to_dict()

        except Exception as exc:
            logger.exception("Cloud QA Audit Inspector encountered an error")
            report = BugReport(
                has_bug=True,
                severity="critical",
                target_url=target_url,
                test_scenario=user_scenario,
                summary=f"Unable to complete audit: {exc}",
                error_message=str(exc),
                elapsed_seconds=time.monotonic() - start,
            )
            return report.to_dict()

    @staticmethod
    def _parse_agent_output(
        raw: str,
        target_url: str,
        test_scenario: str,
    ) -> BugReport:
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
                    summary=data.get("summary", "Audit finished."),
                    steps_taken=data.get("steps_taken", []),
                    dom_errors=data.get("dom_errors", []),
                    visual_anomalies=data.get("visual_anomalies", []),
                    raw_agent_output=raw,
                )
            except Exception:
                pass

        has_bug = any(
            kw in raw.lower()
            for kw in ("error", "bug", "broken", "fail", "missing", "not found", "404", "issue")
        )
        return BugReport(
            has_bug=has_bug,
            severity="low" if has_bug else "none",
            target_url=target_url,
            test_scenario=test_scenario,
            summary=raw[:200] if raw else "Audit complete.",
            raw_agent_output=raw,
        )
