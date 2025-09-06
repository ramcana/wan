"""
Data models for the Local Testing Framework
"""

from .test_results import (
    ValidationResult,
    ValidationStatus,
    EnvironmentValidationResults,
    TestResults,
    TestStatus
)

from .configuration import (
    LocalTestConfiguration,
    PerformanceTargets,
    EnvironmentRequirements
)

__all__ = [
    "ValidationResult",
    "ValidationStatus", 
    "EnvironmentValidationResults",
    "TestResults",
    "TestStatus",
    "LocalTestConfiguration",
    "PerformanceTargets",
    "EnvironmentRequirements"
]