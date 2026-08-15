"""
agents/__init__.py
──────────────────
Package marker for the agents sub-package.
Exports the two primary agent classes for convenient top-level imports.
"""

from .coder_agent import CodeRepairAgent
from .qa_agent import UIAuditAgent

__all__ = ["UIAuditAgent", "CodeRepairAgent"]
