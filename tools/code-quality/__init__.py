"""
Code Quality Checking System

A comprehensive system for enforcing code quality standards including
formatting, style, documentation, type hints, and complexity analysis.
"""

from .quality_checker import QualityChecker
from .models import QualityReport, QualityIssue, QualityMetrics

__version__ = "1.0.0"
__all__ = ["QualityChecker", "QualityReport", "QualityIssue", "QualityMetrics"]