#!/usr/bin/env python3
"""
Example usage of the complete Test Suite Infrastructure and Orchestration system
"""

import asyncio
import logging
from pathlib import Path

from orchestrator import TestSuiteOrchestrator, TestCategory, TestConfig
from runner_engine import TestRunnerEngine, TestExecutionContext
from coverage_analyzer import CoverageAnalyzer
from test_auditor import TestAuditor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demonstrate_test_infrastructure():
    """Demonstrate the complete test infrastructure system"""
    
    print("🚀 Test Suite Infrastructure and Orchestration Demo")
    print("=" * 60)
    
    # 1. Load configuration
    config_path = Path("../../tests/config/test-config.yaml")
    config = TestConfig.load_from_file(config_path)
    print(f"✅ Loaded configuration from {config_path}")
    
    # 2. Test Auditing
    print("\n📋 Step 1: Test Auditing")
    print("-" * 30)
    
    auditor = TestAuditor(config)
    audit_report = auditor.audit_all_tests()
    
    print(f"Total test files: {audit_report.summary['total_files']}")
    print(f"Broken files: {audit_report.summary['broken_files']}")
    print(f"Healthy files: {audit_report.summary['healthy_files']}")
    print(f"Issues found: {audit_report.summary['issues_found']}")
    
    # Generate audit report
    audit_report_path = Path("../../test_results/demo_audit_report.md")
    auditor.generate_audit_report(audit_report, audit_report_path)
    print(f"📄 Audit report saved to: {audit_report_path}")
    
    # 3. Test Orchestration
    print("\n🎯 Step 2: Test Orchestration")
    print("-" * 30)
    
    orchestrator = TestSuiteOrchestrator(config_path)
    
    # Run specific categories
    categories_to_run = [TestCategory.UNIT, TestCategory.INTEGRATION]
    
    print(f"Running test categories: {[c.value for c in categories_to_run]}")
    
    # Add progress callback
    def progress_callback(progress):
        print(f"Progress: {progress.progress_percentage:.1f}% "
              f"({progress.completed_tests}/{progress.total_tests})")
    
    results = await orchestrator.run_full_suite(
        categories=categories_to_run,
        parallel=True
    )
    
    print(f"✅ Test execution completed!")
    print(f"Success rate: {results.overall_summary.success_rate:.1f}%")
    print(f"Total tests: {results.overall_summary.total_tests}")
    print(f"Passed: {results.overall_summary.passed_tests}")
    print(f"Failed: {results.overall_summary.failed_tests}")
    
    # Export results
    results_path = Path("../../test_results/demo_test_results.json")
    orchestrator.export_results(results, results_path)
    print(f"📊 Test results saved to: {results_path}")
    
    # 4. Coverage Analysis
    print("\n📈 Step 3: Coverage Analysis")
    print("-" * 30)
    
    coverage_analyzer = CoverageAnalyzer(config)
    
    # Measure coverage for the categories we tested
    coverage_report = coverage_analyzer.measure_coverage(categories_to_run)
    
    print(f"Coverage: {coverage_report.coverage_percentage:.1f}%")
    print(f"Total lines: {coverage_report.total_lines:,}")
    print(f"Covered lines: {coverage_report.covered_lines:,}")
    print(f"Threshold met: {coverage_report.threshold_met}")
    
    # Validate coverage thresholds
    validation_results = coverage_analyzer.validate_coverage_thresholds(coverage_report)
    print(f"Threshold validation: {'✅ PASS' if validation_results['global_threshold_met'] else '❌ FAIL'}")
    
    if validation_results['violations']:
        print("Coverage violations:")
        for violation in validation_results['violations']:
            print(f"  - {violation['message']}")
    
    if validation_results['recommendations']:
        print("Recommendations:")
        for rec in validation_results['recommendations']:
            print(f"  - {rec['message']}")
    
    # Generate coverage reports
    coverage_html_path = Path("../../test_results/demo_coverage_report.html")
    coverage_md_path = Path("../../test_results/demo_coverage_report.md")
    
    coverage_analyzer.generate_coverage_report(coverage_report, coverage_html_path, "html")
    coverage_analyzer.generate_coverage_report(coverage_report, coverage_md_path, "markdown")
    
    print(f"📄 Coverage reports saved:")
    print(f"  - HTML: {coverage_html_path}")
    print(f"  - Markdown: {coverage_md_path}")
    
    # 5. Test Runner Engine Demo
    print("\n⚙️  Step 4: Test Runner Engine")
    print("-" * 30)
    
    runner_engine = TestRunnerEngine(config)
    
    # Add progress monitoring
    def runner_progress_callback(progress):
        if progress.current_test:
            print(f"Running: {Path(progress.current_test).name}")
    
    runner_engine.add_progress_callback(runner_progress_callback)
    
    # Execute a specific category with detailed monitoring
    execution_context = TestExecutionContext(
        category=TestCategory.UNIT,
        test_files=[],  # Will be discovered automatically
        timeout=60,
        parallel=False  # Sequential for demo
    )
    
    detailed_results = await runner_engine.execute_category_tests(execution_context)
    
    print(f"Detailed execution results:")
    print(f"  - Tests executed: {len(detailed_results)}")
    
    for result in detailed_results[:5]:  # Show first 5 results
        status_icon = "✅" if result.status.value == "passed" else "❌"
        print(f"  {status_icon} {result.name}: {result.duration:.2f}s")
    
    if len(detailed_results) > 5:
        print(f"  ... and {len(detailed_results) - 5} more tests")
    
    # 6. Summary
    print("\n🎉 Demo Complete!")
    print("=" * 60)
    print("The Test Suite Infrastructure provides:")
    print("✅ Comprehensive test auditing and categorization")
    print("✅ Orchestrated test execution with parallel support")
    print("✅ Detailed coverage analysis and reporting")
    print("✅ Progress monitoring and timeout handling")
    print("✅ Automated test discovery and validation")
    print("✅ Multiple report formats (JSON, HTML, Markdown)")
    
    print(f"\n📁 All reports saved to: {Path('../../test_results').absolute()}")
    
    return {
        'audit_report': audit_report,
        'test_results': results,
        'coverage_report': coverage_report,
        'detailed_results': detailed_results
    }


if __name__ == "__main__":
    # Run the demonstration
    demo_results = asyncio.run(demonstrate_test_infrastructure())
    
    print("\n🔍 Demo Results Summary:")
    print(f"Files audited: {demo_results['audit_report'].summary['total_files']}")
    print(f"Tests executed: {demo_results['test_results'].overall_summary.total_tests}")
    print(f"Coverage achieved: {demo_results['coverage_report'].coverage_percentage:.1f}%")
    print(f"Detailed test results: {len(demo_results['detailed_results'])}")