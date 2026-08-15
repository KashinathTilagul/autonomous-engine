"""
api/index.py
────────────
Vercel Serverless entry point for FastAPI.
Exposes the `app` instance from `server.py`.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from server import app
