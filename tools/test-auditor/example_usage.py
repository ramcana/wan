#!/usr/bin/env python3
"""
Test Auditor Example Usage

This script demonstrates how to use the comprehensive test auditor system
to analyze test suite health and generate actionable improvement plans.
"""

import json
import sys
from pathlib import Path

# Add the test auditor to the path
sys.path.insert(0, str(Path(__file__).parent))

from orchestrator import TestSuiteOrchestrator
from test_auditor import TestAuditor
from coverage_analyzer import CoverageAnalyzer
from test_runner import ParallelTestRunner


def example_basic_audit():
    """Example: Basic test suite audit"""
    print("="*60)
    print("EXAMPLE 1: Basic Test Suite Audit")
    print("="*60)
    
    project_root = Path.cwd()
    auditor = TestAuditor(project_root)
    
    print("Running basic audit...")
    report = auditor.audit_test_suite()
    
    print(f"\nResults:")
    print(f"  Total files: {report.total_files}")
    print(f"  Total tests: {report.total_tests}")
    print(f"  Passing tests: {report.passing_tests}")
    print(f"  Failing tests: {report.failing_tests}")
    print(f"  Broken files: {len(report.broken_files)}")
    print(f"  Critical issues: {len(report.critical_issues)}")
    
    if report.recommendations:
        print(f"\nTop 3 Recommendations:")
        for i, rec in enumerate(report.recommendations[:3], 1):
            print(f"  {i}. {rec}")
    
    return report


def example_coverage_analysis():
    """Example: Coverage analysis"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Coverage Analysis")
    print("="*60)
    
    project_root = Path.cwd()
    analyzer = CoverageAnalyzer(project_root)
    
    # Discover test files
    from test_auditor import TestDiscoveryEngine
    discovery = TestDiscoveryEngine(project_root)
    test_files = discovery.discover_test_files()
    
    print(f"Analyzing coverage for {len(test_files)} test files...")
    
    # Set custom thresholds
    analyzer.threshold_manager.set_threshold('overall', 75.0)
    analyzer.threshold_manager.set_threshold('file', 60.0)
    
    report = analyzer.analyze_coverage(test_files[:5])  # Limit for demo
    
    print(f"\nCoverage Results:")
    print(f"  Overall coverage: {report.overall_percentage:.1f}%")
    print(f"  Files covered: {report.covered_files}/{report.total_files}")
    print(f"  Lines covered: {report.covered_lines}/{report.total_lines}")
    print(f"  Coverage gaps: {len(report.coverage_gaps)}")
    print(f"  Threshold violations: {len(report.threshold_violations)}")
    
    if report.coverage_gaps:
        print(f"\nTop Coverage Gaps:")
        for gap in report.coverage_gaps[:3]:
            print(f"  - {gap.gap_type} '{gap.name}' in {gap.file_path} (line {gap.line_start})")
    
    return report


def example_test_execution():
    """Example: Test execution with monitoring"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Test Execution with Monitoring")
    print("="*60)
    
    project_root = Path.cwd()
    runner = ParallelTestRunner(max_workers=2)  # Limit workers for demo
    
    # Discover test files
    from test_auditor import TestDiscoveryEngine
    discovery = TestDiscoveryEngine(project_root)
    test_files = discovery.discover_test_files()
    
    # Run a subset for demo
    demo_files = test_files[:3] if test_files else []
    
    if not demo_files:
        print("No test files found for execution demo")
        return None
    
    print(f"Executing {len(demo_files)} test files with monitoring...")
    
    report = runner.run_tests_parallel(demo_files, project_root)
    
    print(f"\nExecution Results:")
    print(f"  Total files: {report.total_files}")
    print(f"  Successful: {report.successful_files}")
    print(f"  Failed: {report.failed_files}")
    print(f"  Total time: {report.total_execution_time:.2f}s")
    
    if report.performance_summary:
        avg_time = report.performance_summary.get('average_time', 0)
        max_time = report.performance_summary.get('max_time', 0)
        print(f"  Average time: {avg_time:.2f}s")
        print(f"  Max time: {max_time:.2f}s")
    
    return report


def example_comprehensive_analysis():
    """Example: Comprehensive analysis with all components"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Comprehensive Analysis")
    print("="*60)
    
    project_root = Path.cwd()
    orchestrator = TestSuiteOrchestrator(project_root)
    
    print("Running comprehensive analysis (this may take a while)...")
    
    try:
        analysis = orchestrator.run_comprehensive_analysis()
        
        print(f"\nComprehensive Analysis Results:")
        print(f"  Health Score: {analysis.health_score:.1f}/100")
        
        # Analysis summary
        summary = analysis.analysis_summary
        print(f"\nTest Discovery:")
        print(f"  Files: {summary['test_discovery']['total_files']}")
        print(f"  Tests: {summary['test_discovery']['total_tests']}")
        
        print(f"\nQuality Metrics:")
        quality = summary['quality_metrics']
        print(f"  Syntax Health: {'✓' if quality['syntax_health'] else '✗'}")
        print(f"  Import Health: {'✓' if quality['import_health'] else '✗'}")
        print(f"  Assertion Completeness: {'✓' if quality['assertion_completeness'] else '✗'}")
        
        print(f"\nExecution Metrics:")
        execution = summary['execution_metrics']
        print(f"  Success Rate: {execution['success_rate']:.1%}")
        print(f"  Average Time: {execution['average_execution_time']:.2f}s")
        print(f"  Timeout Rate: {execution['timeout_rate']:.1%}")
        
        print(f"\nCoverage Metrics:")
        coverage = summary['coverage_metrics']
        print(f"  Overall: {coverage['overall_percentage']:.1f}%")
        print(f"  Files with Coverage: {coverage['files_with_coverage']}")
        print(f"  Critical Gaps: {coverage['critical_gaps']}")
        
        # Action plan
        if analysis.action_plan:
            print(f"\nTop Priority Actions:")
            for i, action in enumerate(analysis.action_plan[:5], 1):
                priority_icon = {
                    'critical': '🔴',
                    'high': '🟡',
                    'medium': '🟢',
                    'low': '🔵'
                }.get(action['priority'], '⚪')
                
                print(f"  {i}. {priority_icon} [{action['priority'].upper()}] {action['title']}")
                print(f"     {action['description']}")
        
        # Recommendations
        if analysis.recommendations:
            print(f"\nKey Recommendations:")
            for i, rec in enumerate(analysis.recommendations[:5], 1):
                print(f"  {i}. {rec}")
        
        return analysis
        
    except Exception as e:
        print(f"Error during comprehensive analysis: {e}")
        print("This might be due to missing dependencies or test execution issues.")
        return None


def example_custom_configuration():
    """Example: Custom configuration and thresholds"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Custom Configuration")
    print("="*60)
    
    project_root = Path.cwd()
    
    # Create orchestrator with custom configuration
    orchestrator = TestSuiteOrchestrator(project_root)
    
    # Customize coverage thresholds
    orchestrator.coverage_analyzer.threshold_manager.set_threshold('overall', 85.0)
    orchestrator.coverage_analyzer.threshold_manager.set_threshold('file', 75.0)
    
    # Add critical files that need high coverage
    orchestrator.coverage_analyzer.threshold_manager.add_critical_file('core/')
    orchestrator.coverage_analyzer.threshold_manager.add_critical_file('api/')
    
    # Exclude test files from coverage requirements
    orchestrator.coverage_analyzer.threshold_manager.exclude_file('test_')
    orchestrator.coverage_analyzer.threshold_manager.exclude_file('conftest.py')
    
    # Customize timeout settings
    orchestrator.runner.executor.timeout_manager.set_timeout_override('integration', 90)
    orchestrator.runner.executor.timeout_manager.set_timeout_override('e2e', 180)
    
    # Customize health score weights
    orchestrator.health_scorer.weights['coverage'] = 0.30  # Increase coverage importance
    orchestrator.health_scorer.weights['execution_success'] = 0.20
    
    print("Configuration applied:")
    print(f"  Overall coverage threshold: 85%")
    print(f"  File coverage threshold: 75%")
    print(f"  Critical file patterns: core/, api/")
    print(f"  Integration test timeout: 90s")
    print(f"  E2E test timeout: 180s")
    print(f"  Coverage weight in health score: 30%")
    
    return orchestrator


def example_save_and_load_results():
    """Example: Saving and loading analysis results"""
    print("\n" + "="*60)
    print("EXAMPLE 6: Save and Load Results")
    print("="*60)
    
    # Run basic audit
    project_root = Path.cwd()
    auditor = TestAuditor(project_root)
    report = auditor.audit_test_suite()
    
    # Save to JSON
    output_file = Path('demo_audit_report.json')
    
    from dataclasses import asdict
    with open(output_file, 'w') as f:
        json.dump(asdict(report), f, indent=2, default=str)
    
    print(f"Saved audit report to {output_file}")
    
    # Load and display
    with open(output_file, 'r') as f:
        loaded_data = json.load(f)
    
    print(f"Loaded report:")
    print(f"  Total files: {loaded_data['total_files']}")
    print(f"  Total tests: {loaded_data['total_tests']}")
    print(f"  Recommendations: {len(loaded_data['recommendations'])}")
    
    # Clean up
    output_file.unlink()
    print(f"Cleaned up {output_file}")


def main():
    """Run all examples"""
    print("Test Auditor Example Usage")
    print("This script demonstrates various features of the test auditor system.")
    print()
    
    try:
        # Example 1: Basic audit
        example_basic_audit()
        
        # Example 2: Coverage analysis (may fail if no tests found)
        try:
            example_coverage_analysis()
        except Exception as e:
            print(f"Coverage analysis example failed: {e}")
        
        # Example 3: Test execution (may fail if no tests found)
        try:
            example_test_execution()
        except Exception as e:
            print(f"Test execution example failed: {e}")
        
        # Example 4: Comprehensive analysis (may take time)
        try:
            example_comprehensive_analysis()
        except Exception as e:
            print(f"Comprehensive analysis example failed: {e}")
        
        # Example 5: Custom configuration
        example_custom_configuration()
        
        # Example 6: Save and load
        example_save_and_load_results()
        
        print("\n" + "="*60)
        print("EXAMPLES COMPLETE")
        print("="*60)
        print()
        print("To run the test auditor on your project:")
        print("  python tools/test-auditor/orchestrator.py")
        print()
        print("For more options:")
        print("  python tools/test-auditor/cli.py --help")
        
    except KeyboardInterrupt:
        print("\nExamples interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()