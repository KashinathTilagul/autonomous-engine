"""
utils/github_publisher.py
──────────────────────────
GitPublisher – bridges the local git repo and GitHub.

Responsibilities
────────────────
* Stage and commit all modified files in the working tree.
* Push the commit to a new remote branch.
* Open a **draft** Pull Request on GitHub summarising the automated fix.

Design choices
──────────────
* Uses the PyGithub library for all GitHub API interactions (branch
  protection, PR creation) while delegating the actual git operations
  to the ``subprocess`` module so that SSH/HTTPS credentials configured
  in the local git environment are reused transparently.
* Commits are always made on a fresh branch derived from the configured
  base branch, so there is never a risk of force-pushing to ``main``.
* The PR is opened as a draft so a human engineer can review and merge it.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from datetime import datetime, timezone
from typing import Optional

from github import Github, GithubException

from config import get_settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _run_git(args: list[str], cwd: str) -> str:
    """
    Run a git sub-command in *cwd* and return combined stdout+stderr.

    Raises
    ------
    subprocess.CalledProcessError
        If the command exits with a non-zero status.
    """
    cmd = ["git"] + args
    logger.debug("git %s (cwd=%s)", " ".join(args), cwd)
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )
    return (result.stdout + result.stderr).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Publisher class
# ─────────────────────────────────────────────────────────────────────────────

class GitPublisher:
    """
    Commits local changes and opens a GitHub draft Pull Request.

    Parameters
    ----------
    local_repo_path : str
        Absolute path to the git working tree on the local filesystem.
    repo_name : str, optional
        GitHub repository in ``"owner/repo"`` format.  Defaults to the value
        in settings.
    """

    def __init__(
        self,
        local_repo_path: str,
        *,
        repo_name: Optional[str] = None,
    ) -> None:
        cfg = get_settings()
        self._repo_path = local_repo_path
        self._repo_name = repo_name or cfg.repo_name
        self._base_branch = cfg.github_base_branch
        self._gh = Github(cfg.github_token.get_secret_value())
        self._gh_repo = self._gh.get_repo(self._repo_name)

        logger.info(
            "GitPublisher initialised",
            extra={"repo": self._repo_name, "base_branch": self._base_branch},
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def create_pull_request(
        self,
        branch_name: str,
        title: str,
        body: str,
        *,
        commit_message: Optional[str] = None,
        labels: Optional[list[str]] = None,
    ) -> str:
        """
        Stage all changes, commit, push, and open a draft PR.

        Parameters
        ----------
        branch_name : str
            Name of the new branch to create (e.g. ``"fix/login-button-404"``).
            A timestamp suffix is appended automatically if the branch already
            exists on the remote.
        title : str
            PR title shown on GitHub.
        body : str
            Markdown body of the PR description.
        commit_message : str, optional
            Git commit message.  Defaults to *title*.
        labels : list[str], optional
            GitHub label names to apply to the PR.

        Returns
        -------
        str
            The HTML URL of the created Pull Request.

        Raises
        ------
        subprocess.CalledProcessError
            If any git operation fails.
        GithubException
            If the GitHub API call fails.
        """
        branch_name = self._safe_branch_name(branch_name)
        commit_msg = commit_message or title

        self._checkout_new_branch(branch_name)
        self._stage_and_commit(commit_msg)
        self._push_branch(branch_name)

        pr_url = self._open_draft_pr(
            branch_name=branch_name,
            title=title,
            body=body,
            labels=labels or [],
        )

        logger.info("Pull Request opened", extra={"url": pr_url})
        return pr_url

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers – git operations
    # ─────────────────────────────────────────────────────────────────────────

    def _safe_branch_name(self, name: str) -> str:
        """
        Ensure *name* does not collide with an existing remote branch by
        appending a compact UTC timestamp when necessary.
        """
        # Sanitise: replace spaces with hyphens, lowercase
        sanitised = name.replace(" ", "-").lower()[:80]

        try:
            self._gh_repo.get_branch(sanitised)
            # Branch already exists – append timestamp
            ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
            sanitised = f"{sanitised}-{ts}"
            logger.debug("Branch name collision resolved to: %s", sanitised)
        except GithubException:
            # Branch does not exist – use as-is
            pass

        return sanitised

    def _checkout_new_branch(self, branch_name: str) -> None:
        """Create and check out *branch_name* from the current HEAD."""
        # Fetch latest from remote to stay up-to-date
        try:
            _run_git(["fetch", "origin", self._base_branch], cwd=self._repo_path)
            _run_git(
                ["checkout", "-b", branch_name, f"origin/{self._base_branch}"],
                cwd=self._repo_path,
            )
        except subprocess.CalledProcessError:
            # Fall back to creating branch from local HEAD
            logger.warning(
                "Could not base branch on origin/%s; using local HEAD.",
                self._base_branch,
            )
            _run_git(["checkout", "-b", branch_name], cwd=self._repo_path)

    def _stage_and_commit(self, commit_message: str) -> None:
        """Stage all changes (tracked + untracked) and create a commit."""
        _run_git(["add", "--all"], cwd=self._repo_path)

        # Check if there is anything to commit.
        status_output = _run_git(
            ["status", "--porcelain"], cwd=self._repo_path
        )
        if not status_output.strip():
            logger.info("No changes to commit; skipping git commit step.")
            return

        _run_git(
            [
                "commit",
                "--message", commit_message,
                "--author", "Autonomous Engine <bot@autonomous-engine>",
            ],
            cwd=self._repo_path,
        )

    def _push_branch(self, branch_name: str) -> None:
        """Push the branch to the ``origin`` remote."""
        _run_git(
            ["push", "--set-upstream", "origin", branch_name],
            cwd=self._repo_path,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers – GitHub API
    # ─────────────────────────────────────────────────────────────────────────

    def _open_draft_pr(
        self,
        *,
        branch_name: str,
        title: str,
        body: str,
        labels: list[str],
    ) -> str:
        """
        Create a draft Pull Request on GitHub and optionally apply labels.

        Returns the PR's HTML URL.
        """
        pr = self._gh_repo.create_pull(
            title=title,
            body=body,
            head=branch_name,
            base=self._base_branch,
            draft=True,
        )

        # Apply labels if they exist in the repository.
        if labels:
            existing_labels = {lbl.name for lbl in self._gh_repo.get_labels()}
            valid_labels = [lbl for lbl in labels if lbl in existing_labels]
            if valid_labels:
                pr.add_to_labels(*valid_labels)
            skipped = set(labels) - set(valid_labels)
            if skipped:
                logger.warning(
                    "Labels not found in repo, skipping: %s", skipped
                )

        return pr.html_url
