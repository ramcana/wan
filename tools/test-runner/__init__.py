"""
Test Runner Module - Comprehensive test orchestration and execution
"""

from tools..orchestrator import (
    TestSuiteOrchestrator,
    TestCategory,
    TestStatus,
    TestResults,
    CategoryResults,
    TestDetail,
    TestSummary,
    TestConfig,
    ResourceManager
)

from tools..runner_engine import (
    TestRunnerEngine,
    TestDiscovery,
    TestExecutionContext,
    ExecutionProgress,
    ProgressMonitor,
    TimeoutManager
)

from tools..coverage_analyzer import (
    CoverageAnalyzer,
    CoverageReport,
    FileCoverage,
    ModuleCoverage,
    CoverageTrend,
    CoverageHistory,
    CoverageThresholdValidator
)

from tools..test_auditor import (
    TestAuditor,
    TestFileAudit,
    AuditReport,
    TestIssue,
    TestIssueType,
    TestCodeAnalyzer
)

__all__ = [
    # Orchestrator
    'TestSuiteOrchestrator',
    'TestCategory',
    'TestStatus', 
    'TestResults',
    'CategoryResults',
    'TestDetail',
    'TestSummary',
    'TestConfig',
    'ResourceManager',
    
    # Runner Engine
    'TestRunnerEngine',
    'TestDiscovery',
    'TestExecutionContext',
    'ExecutionProgress',
    'ProgressMonitor',
    'TimeoutManager',
    
    # Coverage Analyzer
    'CoverageAnalyzer',
    'CoverageReport',
    'FileCoverage',
    'ModuleCoverage',
    'CoverageTrend',
    'CoverageHistory',
    'CoverageThresholdValidator',
    
    # Test Auditor
    'TestAuditor',
    'TestFileAudit',
    'AuditReport',
    'TestIssue',
    'TestIssueType',
    'TestCodeAnalyzer'
]

__version__ = '1.0.0'