"""
agents/qa_agent.py
──────────────────
UIAuditAgent – Resilient Cloud QA Inspector with Multiple Fetch Strategies & Direct DOM Inspection.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import urllib.request
import ssl
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx
from config import build_llm, get_settings

logger = logging.getLogger(__name__)


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


_AUDIT_PROMPT_TEMPLATE = """\
You are an expert QA Engineer and Web Application Auditor inspecting the following target page:

Target URL: {target_url}
Test Scenario: {test_scenario}
HTTP Response Code: {status_code}
Page Title: {title}

Page HTML / DOM Snippet:
```html
{content_snippet}
```

Instructions:
1. Examine if the page successfully loaded according to the scenario, or if there are broken components, missing login fields, broken buttons, 4xx/5xx errors, or UI anomalies.
2. If the page loaded normally and matches the scenario with no errors, mark "has_bug": false and "severity": "none".
3. Return your structured findings strictly in JSON format inside triple backticks:
```json
{{
  "bug_report": {{
    "has_bug": false,
    "severity": "none",
    "summary": "Page loaded successfully with all expected elements.",
    "steps_taken": [
      "1. Fetched and verified HTTP status code (200 OK)",
      "2. Analyzed page title and meta elements",
      "3. Evaluated form inputs and buttons for the scenario"
    ],
    "dom_errors": [],
    "visual_anomalies": []
  }}
}}
```
"""


class UIAuditAgent:
    """
    Detailed, cloud-native UI QA Auditor.
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

    def _normalize_url(self, url: str) -> str:
        url = url.strip()
        if not url.startswith("http://") and not url.startswith("https://"):
            url = "https://" + url
        return url

    async def run_test_flow(
        self,
        target_url: str,
        user_scenario: str,
    ) -> dict[str, Any]:
        start = time.monotonic()
        target_url = self._normalize_url(target_url)
        logger.info("Starting QA audit for: %s", target_url)

        # 1. Attempt Playwright browser-use automation if available
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
            logger.info("Playwright not present or failed, falling back to HTTP inspector: %s", browser_exc)

        # 2. Resilient Cloud HTTP Web DOM & Content Inspector
        html_text = ""
        status_code = 200

        # Strategy A: httpx with browser-grade headers and relaxed SSL
        try:
            async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=15.0,
                verify=False,
                http2=True,
            ) as client:
                resp = await client.get(target_url, headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.9",
                })
                status_code = resp.status_code
                html_text = resp.text
        except Exception as e_httpx:
            logger.info("httpx attempt error (%s), trying standard urllib...", e_httpx)
            # Strategy B: urllib fallback
            try:
                ctx = ssl.create_default_context()
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
                req = urllib.request.Request(
                    target_url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                    }
                )
                with urllib.request.urlopen(req, context=ctx, timeout=15) as u_resp:
                    status_code = u_resp.status
                    html_text = u_resp.read().decode("utf-8", errors="replace")
            except Exception as e_urllib:
                err_str = str(e_urllib)
                logger.exception("Both fetch strategies failed")
                return BugReport(
                    has_bug=True,
                    severity="critical",
                    target_url=target_url,
                    test_scenario=user_scenario,
                    summary=f"Unable to reach host: {err_str}",
                    steps_taken=[
                        f"1. Attempted HTTP/HTTPS connection to {target_url}",
                        "2. Host connection failed or timed out",
                    ],
                    dom_errors=[f"Network Error: {err_str}"],
                    error_message=err_str,
                    elapsed_seconds=time.monotonic() - start,
                ).to_dict()

        # Parse page DOM
        title_match = re.search(r"<title>(.*?)</title>", html_text, re.IGNORECASE | re.DOTALL)
        title = title_match.group(1).strip() if title_match else "Untitled"

        clean_snippet = re.sub(r"<script.*?</script>", "", html_text, flags=re.DOTALL | re.IGNORECASE)
        clean_snippet = re.sub(r"<style.*?</style>", "", clean_snippet, flags=re.DOTALL | re.IGNORECASE)
        clean_snippet = re.sub(r"\s+", " ", clean_snippet)[:8000]

        try:
            prompt = _AUDIT_PROMPT_TEMPLATE.format(
                target_url=target_url,
                test_scenario=user_scenario or "Inspect page UI and functionality",
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
            logger.info("LLM reasoning fallback: %s", exc)
            # Local evaluation without LLM if API key / gateway is slow
            has_error = status_code >= 400 or "error" in html_text.lower()
            return BugReport(
                has_bug=has_error,
                severity="high" if status_code >= 400 else ("low" if has_error else "none"),
                target_url=target_url,
                test_scenario=user_scenario,
                summary=f"HTTP {status_code} - Page '{title}' inspected successfully.",
                steps_taken=[
                    f"1. Established HTTPS connection to {target_url}",
                    f"2. Verified HTTP response code: {status_code}",
                    f"3. Parsed document structure ({len(html_text)} bytes)",
                    f"4. Checked page title: '{title}'",
                ],
                dom_errors=[f"HTTP Status: {status_code}"] if status_code >= 400 else [],
                elapsed_seconds=time.monotonic() - start,
            ).to_dict()

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
                    severity=data.get("severity", "none"),
                    target_url=target_url,
                    test_scenario=test_scenario,
                    summary=data.get("summary", "Audit complete."),
                    steps_taken=data.get("steps_taken", []),
                    dom_errors=data.get("dom_errors", []),
                    visual_anomalies=data.get("visual_anomalies", []),
                    raw_agent_output=raw,
                )
            except Exception:
                pass

        has_bug = any(
            kw in raw.lower()
            for kw in ("error", "broken", "failed", "missing element", "404 not found")
        )
        return BugReport(
            has_bug=has_bug,
            severity="medium" if has_bug else "none",
            target_url=target_url,
            test_scenario=test_scenario,
            summary=raw[:200] if raw else "Audit complete.",
            raw_agent_output=raw,
        )
