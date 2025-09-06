"""
Local Testing Framework for Wan2.2 UI Variant

A comprehensive testing and validation system that automates the verification 
of performance optimizations, system functionality, and deployment readiness.
"""

__version__ = "1.0.0"
__author__ = "Wan2.2 Development Team"

from .environment_validator import EnvironmentValidator
from .models.test_results import ValidationResult, ValidationStatus

__all__ = [
    "EnvironmentValidator",
    "ValidationResult", 
    "ValidationStatus"
]