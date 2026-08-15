"""
main.py
───────
Autonomous UI Bug-Finding & Fix Engine — Core Orchestrator.
Maintains full pipeline transparency with step-by-step progress tracking.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import sys
from datetime import datetime, timezone
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agents.coder_agent import CodeRepairAgent
from agents.qa_agent import UIAuditAgent
from config import get_settings
from utils.github_publisher import GitPublisher

logger = logging.getLogger("autonomous_engine")
console = Console()


def _slugify(text: str, max_len: int = 50) -> str:
    slug = re.sub(r"[^\w\s-]", "", text).strip().lower()
    slug = re.sub(r"[-\s]+", "-", slug)
    return slug[:max_len].rstrip("-")


async def run_pipeline_for_url(
    *,
    url: str,
    scenario: str,
    repo_path: str,
    model_name: Optional[str],
    verify: bool,
    publish_pr: bool,
) -> dict:
    result: dict = {
        "url": url,
        "bug_found": False,
        "severity": "none",
        "summary": "",
        "steps_taken": [],
        "dom_errors": [],
        "fixed": False,
        "verified": None,
        "pr_url": None,
        "error": None,
    }

    # ── Step 1: QA Audit ─────────────────────────────────────────────────────
    console.rule(f"[bold cyan]🔍  Step 1: QA Audit → {url}")
    qa_agent = UIAuditAgent(model_name=model_name)
    bug_report = await qa_agent.run_test_flow(url, scenario)

    result["summary"] = bug_report.get("summary", "")
    result["steps_taken"] = bug_report.get("steps_taken", [])
    result["dom_errors"] = bug_report.get("dom_errors", [])
    result["severity"] = bug_report.get("severity", "none")

    if not bug_report.get("has_bug"):
        console.print(Panel("[green]✓ Audit Passed: No bugs detected.[/green]", expand=False))
        return result

    result["bug_found"] = True
    console.print(
        Panel(
            f"[red]⚠ Bug detected![/red]\n"
            f"Severity : {bug_report['severity']}\n"
            f"Summary  : {bug_report['summary']}",
            expand=False,
        )
    )

    # ── Step 2: Code Repair ──────────────────────────────────────────────────
    console.rule("[bold yellow]🔧  Step 2: AI Code Repair")
    coder = CodeRepairAgent()
    repair_result = coder.trigger_fix(bug_report, repo_path)

    if not repair_result.get("success"):
        error_msg = repair_result.get("error", "Code repair step failed")
        result["error"] = error_msg
        return result

    result["fixed"] = True
    result["repair_details"] = repair_result.get("details", {})
    console.print(Panel("[green]✓ AI Code Repair solution planned[/green]", expand=False))

    # ── Step 3: Verification ─────────────────────────────────────────────────
    if verify:
        console.rule("[bold blue]🔍  Step 3: Post-Fix Verification")
        verification_report = await qa_agent.run_test_flow(url, scenario)
        still_broken = verification_report.get("has_bug", False)
        result["verified"] = not still_broken

    # ── Step 4: Publish PR ───────────────────────────────────────────────────
    if publish_pr and result.get("fixed"):
        console.rule("[bold magenta]📬  Step 4: Publishing Pull Request")
        try:
            branch_name = f"fix/auto-{_slugify(bug_report.get('summary', url))}"
            pr_title = f"fix: auto-repair {bug_report.get('severity', 'unknown')} bug on {url}"
            pr_body = f"## 🤖 Automated Bug Fix\n- **Target:** {url}\n- **Summary:** {bug_report.get('summary')}"
            publisher = GitPublisher(local_repo_path=repo_path)
            pr_url = publisher.create_pull_request(
                branch_name=branch_name,
                title=pr_title,
                body=pr_body,
                labels=["automated", "bug"],
            )
            result["pr_url"] = pr_url
        except Exception as exc:
            result["error"] = f"PR creation failed: {exc}"

    return result
