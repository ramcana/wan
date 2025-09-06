"""
Test Suite Auditor Package

A comprehensive test suite analysis and auditing system that provides:
- Test discovery and categorization
- Syntax and structure analysis
- Performance profiling and monitoring
- Coverage analysis and gap identification
- Health scoring and improvement recommendations
- Automated action planning

Main Components:
- TestAuditor: Core auditing functionality
- TestRunner: Advanced test execution with monitoring
- CoverageAnalyzer: Comprehensive coverage analysis
- TestSuiteOrchestrator: Coordinates all components

Usage:
    from tools.test_auditor import TestSuiteOrchestrator
    
    orchestrator = TestSuiteOrchestrator(project_root)
    analysis = orchestrator.run_comprehensive_analysis()
    print(f"Health Score: {analysis.health_score}")
"""

from tools..test_auditor import (
    TestAuditor,
    TestDiscoveryEngine,
    TestDependencyAnalyzer,
    TestPerformanceProfiler,
    TestSuiteAuditReport,
    TestFileAnalysis,
    TestIssue
)

from tools..test_runner import (
    ParallelTestRunner,
    TestExecutor,
    TestExecutionResult,
    TestSuiteExecutionReport,
    TestIsolationManager,
    TestTimeoutManager,
    TestRetryManager
)

from tools..coverage_analyzer import (
    CoverageAnalyzer,
    CoverageReport,
    FileCoverage,
    CoverageGap,
    CoverageThresholdManager
)

from tools..orchestrator import (
    TestSuiteOrchestrator,
    ComprehensiveTestAnalysis,
    TestSuiteHealthScorer,
    ActionPlanGenerator
)

__version__ = "1.0.0"
__author__ = "WAN22 Development Team"

__all__ = [
    # Main orchestrator
    'TestSuiteOrchestrator',
    'ComprehensiveTestAnalysis',
    
    # Core auditing
    'TestAuditor',
    'TestDiscoveryEngine',
    'TestDependencyAnalyzer',
    'TestPerformanceProfiler',
    'TestSuiteAuditReport',
    'TestFileAnalysis',
    'TestIssue',
    
    # Test execution
    'ParallelTestRunner',
    'TestExecutor',
    'TestExecutionResult',
    'TestSuiteExecutionReport',
    'TestIsolationManager',
    'TestTimeoutManager',
    'TestRetryManager',
    
    # Coverage analysis
    'CoverageAnalyzer',
    'CoverageReport',
    'FileCoverage',
    'CoverageGap',
    'CoverageThresholdManager',
    
    # Health and planning
    'TestSuiteHealthScorer',
    'ActionPlanGenerator',
]