"""
Test Quality Improvement Tools

Comprehensive test coverage analysis system that provides:
- Detailed coverage analysis with untested code path detection
- Coverage threshold enforcement for new code
- Trend tracking and historical analysis
- Actionable recommendations for coverage improvement
- Multiple report formats (JSON, HTML, Markdown)
"""

from tools..coverage_system import (
    ComprehensiveCoverageSystem,
    CoverageTrendTracker,
    NewCodeCoverageAnalyzer,
    CoverageThresholdEnforcer,
    DetailedCoverageReporter,
    CoverageTrend,
    NewCodeCoverage,
    CoverageThresholdResult
)

__version__ = "1.0.0"
__author__ = "WAN22 Project"

__all__ = [
    'ComprehensiveCoverageSystem',
    'CoverageTrendTracker',
    'NewCodeCoverageAnalyzer',
    'CoverageThresholdEnforcer',
    'DetailedCoverageReporter',
    'CoverageTrend',
    'NewCodeCoverage',
    'CoverageThresholdResult'
]
