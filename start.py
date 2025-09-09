#!/usr/bin/env python3
"""Deprecated startup script.
Use the unified CLI instead: ``python wan-cli``.
This stub will be removed in a future release.
"""
import warnings
from cli.main import app

warnings.warn(
    "start.py is deprecated; use 'python wan-cli' instead.",
    DeprecationWarning,
)

if __name__ == "__main__":
    app()
