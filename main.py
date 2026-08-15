"""
main.py
───────
Autonomous UI Bug-Finding & Fix Engine – Main Orchestrator

Pipeline
────────
  ┌─────────────────────────────────────────────────────────────────────┐
  │  For each (URL, scenario) pair in config:                           │
  │                                                                     │
  │  1. UIAuditAgent  →  run_test_flow()   →  bug_report               │
  │  2. If no bug detected → log & continue to next URL                 │
  │  3. CodeRepairAgent → trigger_fix()    →  repair_result             │
  │  4. If repair failed → log & continue (no broken PR)                │
  │  5. UIAuditAgent  →  run_test_flow()   →  verification_report       │
  │     (optional regression check after the fix)                       │
  │  6. GitPublisher  →  create_pull_request() → PR URL                 │
  └─────────────────────────────────────────────────────────────────────┘

Running the engine
──────────────────
  # Single URL, default scenario
  python main.py --url https://example.com

  # Multiple URLs from .env, skip verification pass
  python main.py --no-verify

  # Override model for this run
  python main.py --url https://example.com --model openai/gpt-4o

Environment
───────────
  All secrets and defaults are read from .env (see .env.example).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import textwrap
from datetime import datetime, timezone
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agents.coder_agent import CodeRepairAgent
from agents.qa_agent import UIAuditAgent
from config import get_settings
from utils.github_publisher import GitPublisher

# ─────────────────────────────────────────────────────────────────────────────
# Console (rich) – pretty output for humans running the engine interactively
# ─────────────────────────────────────────────────────────────────────────────

console = Console()
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CLI argument parser
# ─────────────────────────────────────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autonomous-engine",
        description="Autonomous UI bug-finding and auto-fix pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Examples:
              python main.py --url https://staging.example.com
              python main.py --url https://staging.example.com --scenario "Try logging in with wrong password"
              python main.py --no-verify --repo-path /src/my-app
            """
        ),
    )
    parser.add_argument(
        "--url",
        metavar="URL",
        help="Target URL to audit.  Overrides TARGET_URLS in .env.",
    )
    parser.add_argument(
        "--scenario",
        metavar="TEXT",
        help="Test scenario.  Overrides DEFAULT_TEST_SCENARIO in .env.",
    )
    parser.add_argument(
        "--repo-path",
        metavar="PATH",
        default=".",
        help="Absolute path to the local source repo (default: current dir).",
    )
    parser.add_argument(
        "--model",
        metavar="MODEL",
        help="Override LLM model for this run (e.g. openai/gpt-4o).",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip the post-fix verification QA pass.",
    )
    parser.add_argument(
        "--no-pr",
        action="store_true",
        help="Skip GitHub PR creation (dry-run mode).",
    )
    return parser


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_pr_body(
    bug_report: dict,
    repair_result: dict,
    verification_report: Optional[dict],
) -> str:
    """Render the PR description from pipeline results."""
    lines: list[str] = [
        "## 🤖 Automated Bug Fix",
        "",
        "This Pull Request was opened automatically by the Autonomous UI Bug Engine.",
        "",
        "---",
        "",
        "### 🐛 Bug Detected",
        f"- **URL:** {bug_report['target_url']}",
        f"- **Severity:** `{bug_report['severity']}`",
        f"- **Summary:** {bug_report['summary']}",
        "",
    ]

    if bug_report.get("dom_errors"):
        lines += ["**DOM Errors:**"]
        lines += [f"  - {e}" for e in bug_report["dom_errors"]]
        lines.append("")

    if bug_report.get("visual_anomalies"):
        lines += ["**Visual Anomalies:**"]
        lines += [f"  - {a}" for a in bug_report["visual_anomalies"]]
        lines.append("")

    lines += [
        "---",
        "",
        "### 🔧 Repair",
        f"- **OpenHands Task ID:** `{repair_result.get('task_id', 'N/A')}`",
        f"- **Status:** `{repair_result.get('status', 'N/A')}`",
        "",
    ]

    if verification_report:
        verified = not verification_report.get("has_bug", True)
        status_emoji = "✅" if verified else "⚠️"
        lines += [
            "---",
            "",
            "### 🔍 Post-Fix Verification",
            f"{status_emoji} QA re-run result: "
            f"{'No bugs detected.' if verified else verification_report.get('summary', 'Bugs still present.')}",
            "",
        ]

    lines += [
        "---",
        "",
        f"*Generated at {datetime.now(timezone.utc).isoformat()} by autonomous-engine.*",
    ]

    return "\n".join(lines)


def _slugify(text: str, max_len: int = 50) -> str:
    """Convert free-form text to a git-safe slug."""
    import re
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len]


# ─────────────────────────────────────────────────────────────────────────────
# Core pipeline (single URL)
# ─────────────────────────────────────────────────────────────────────────────

async def run_pipeline_for_url(
    *,
    url: str,
    scenario: str,
    repo_path: str,
    model_name: Optional[str],
    verify: bool,
    publish_pr: bool,
) -> dict:
    """
    Execute the full find-fix-verify-publish pipeline for a single URL.

    Returns
    -------
    dict
        Summary of the pipeline run with keys:
        ``url``, ``bug_found``, ``fixed``, ``verified``, ``pr_url``, ``error``.
    """
    result: dict = {
        "url": url,
        "bug_found": False,
        "fixed": False,
        "verified": None,
        "pr_url": None,
        "error": None,
    }

    # ── Step 1: QA Audit ─────────────────────────────────────────────────────
    console.rule(f"[bold cyan]🔍  QA Audit → {url}")
    qa_agent = UIAuditAgent(model_name=model_name)
    bug_report = await qa_agent.run_test_flow(url, scenario)

    if not bug_report.get("has_bug"):
        console.print(
            Panel("[green]✓ No bugs detected.[/green]", expand=False)
        )
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
    logger.info("Bug report: %s", bug_report)

    # ── Step 2: Code Repair ──────────────────────────────────────────────────
    console.rule("[bold yellow]🔧  Code Repair via OpenHands")
    coder = CodeRepairAgent()
    repair_result = coder.trigger_fix(bug_report, repo_path)

    if not repair_result.get("success"):
        error_msg = repair_result.get("error", "Unknown error from OpenHands")
        console.print(
            Panel(f"[red]✗ Repair failed:[/red] {error_msg}", expand=False)
        )
        result["error"] = error_msg
        return result

    result["fixed"] = True
    console.print(
        Panel(
            f"[green]✓ Repair succeeded[/green]\n"
            f"Task ID : {repair_result.get('task_id')}",
            expand=False,
        )
    )
    logger.info("Repair result: %s", repair_result)

    # ── Step 3: Verification (optional) ─────────────────────────────────────
    verification_report: Optional[dict] = None
    if verify:
        console.rule("[bold blue]🔍  Post-Fix Verification")
        verification_report = await qa_agent.run_test_flow(url, scenario)
        still_broken = verification_report.get("has_bug", False)

        if still_broken:
            console.print(
                Panel(
                    "[yellow]⚠ Verification detected residual issues.[/yellow]\n"
                    f"{verification_report.get('summary', '')}",
                    expand=False,
                )
            )
        else:
            console.print(
                Panel("[green]✓ Verification passed – fix confirmed.[/green]", expand=False)
            )

        result["verified"] = not still_broken

    # ── Step 4: Publish PR ───────────────────────────────────────────────────
    if not publish_pr:
        console.print("[dim]--no-pr flag set; skipping Pull Request creation.[/dim]")
        return result

    console.rule("[bold magenta]📬  Publishing Pull Request")
    try:
        branch_name = f"fix/auto-{_slugify(bug_report.get('summary', url))}"
        pr_title = f"fix: auto-repair {bug_report.get('severity', 'unknown')} bug on {url}"
        pr_body = _build_pr_body(bug_report, repair_result, verification_report)

        publisher = GitPublisher(local_repo_path=repo_path)
        pr_url = publisher.create_pull_request(
            branch_name=branch_name,
            title=pr_title,
            body=pr_body,
            labels=["automated", "bug", f"severity:{bug_report.get('severity', 'unknown')}"],
        )

        result["pr_url"] = pr_url
        console.print(
            Panel(
                f"[green]✓ Draft PR created[/green]\n{pr_url}",
                expand=False,
            )
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to create Pull Request")
        result["error"] = f"PR creation failed: {exc}"
        console.print(f"[red]✗ PR creation failed:[/red] {exc}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

async def main(argv: Optional[list[str]] = None) -> int:
    """
    Parse CLI arguments, load settings, and run the pipeline for every URL.

    Returns
    -------
    int
        Exit code: 0 for clean run, 1 if any URL produced an error.
    """
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    cfg = get_settings()

    # Determine the list of URLs to process.
    urls: list[str] = [args.url] if args.url else cfg.target_urls
    if not urls:
        console.print(
            "[red]Error:[/red] No target URLs provided.  "
            "Use --url or set TARGET_URLS in .env."
        )
        return 1

    scenario = args.scenario or cfg.default_test_scenario
    repo_path = args.repo_path
    model_name: Optional[str] = args.model

    console.print(
        Panel(
            f"[bold]Autonomous UI Bug Engine[/bold]\n"
            f"URLs     : {len(urls)}\n"
            f"Model    : {model_name or cfg.model_name}\n"
            f"Repo     : {repo_path}\n"
            f"Verify   : {not args.no_verify}\n"
            f"Publish  : {not args.no_pr}",
            title="🚀 Starting",
            expand=False,
        )
    )

    all_results: list[dict] = []
    had_error = False

    for url in urls:
        try:
            run_result = await run_pipeline_for_url(
                url=url,
                scenario=scenario,
                repo_path=repo_path,
                model_name=model_name,
                verify=not args.no_verify,
                publish_pr=not args.no_pr,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unhandled exception in pipeline for %s", url)
            run_result = {
                "url": url,
                "bug_found": False,
                "fixed": False,
                "verified": None,
                "pr_url": None,
                "error": str(exc),
            }

        all_results.append(run_result)
        if run_result.get("error"):
            had_error = True

    # ── Summary table ────────────────────────────────────────────────────────
    console.rule("[bold]📋  Pipeline Summary")
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("URL", no_wrap=False, max_width=50)
    table.add_column("Bug?", justify="center")
    table.add_column("Fixed?", justify="center")
    table.add_column("Verified?", justify="center")
    table.add_column("PR", no_wrap=False, max_width=40)
    table.add_column("Error", no_wrap=False, max_width=30)

    for r in all_results:
        table.add_row(
            r["url"],
            "🐛" if r["bug_found"] else "✅",
            "✅" if r["fixed"] else ("—" if not r["bug_found"] else "❌"),
            (
                "✅" if r["verified"] is True
                else ("❌" if r["verified"] is False else "—")
            ),
            r["pr_url"] or "—",
            (r["error"] or "")[:30],
        )

    console.print(table)
    return 1 if had_error else 0


if __name__ == "__main__":
    exit_code = asyncio.run(main(sys.argv[1:]))
    sys.exit(exit_code)
