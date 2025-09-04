"""
Code quality enforcement module.

This module provides automated quality enforcement through:
- Pre-commit hooks for local development
- CI/CD integration for automated checking
- Quality metrics tracking and reporting
"""

from tools..pre_commit_hooks import PreCommitHookManager
from tools..ci_integration import CIIntegration

__all__ = ['PreCommitHookManager', 'CIIntegration']