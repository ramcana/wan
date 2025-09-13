#!/usr/bin/env python3
"""
Comprehensive Test Suite Runner

This module provides a unified runner for all comprehensive testing and validation
of cleanup and quality improvement tools.

Requirements covered: 1.1, 1.6, 2.6, 3.6, 4.6, 5.6, 6.6
"""

import pytest
import sys
import time
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))


@dataclass
class TestSuiteResult:
    """Results from running a test suite"""
    suite_name: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    tests_run: int
    tests_passed: int
    tests_failed: int
    tests_skipped: int
    success_rate: float
    errors: List[str]
    warnings: List[str]
    
    def __post_init__(self):
        if self.tests_run > 0:
            self.success_rate = self.tests_passed / self.tests_run * 100
        else:
            self.success_rate = 0.0


@dataclass
class ComprehensiveTestReport:
    """Comprehensive report of all test suite results"""
    execution_time: datetime
    total_duration_seconds: float
    suite_results: List[TestSuiteResult]
    overall_success_rate: float
    total_tests_run: int
    total_tests_passed: int
    total_tests_failed: int
    summary: Dict[str, Any]
    recommendations: List[str]
    
    def __post_init__(self):
        self.total_tests_run = sum(r.tests_run for r in self.suite_results)
        self.total_tests_passed = sum(r.tests_passed for r in self.suite_results)
        self.total_tests_failed = sum(r.tests_failed for r in self.suite_results)
        
        if self.total_tests_run > 0:
            self.overall_success_rate = self.total_tests_passed / self.total_tests_run * 100
        else:
            self.overall_success_rate = 0.0


class ComprehensiveTestRunner:
    """Runner for comprehensive test suites"""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path(__file__).parent.parent.parent
        self.test_suites = {
            "e2e": "tests/comprehensive/test_e2e_cleanup_quality_suite.py",
            "integration": "tests/integration/test_tool_interactions.py", 
            "performance": "tests/performance/test_tool_performance.py",
            "acceptance": "tests/acceptance/test_user_acceptance_scenarios.py"
        }
        self.results = []
    
    def run_single_suite(self, suite_name: str, suite_path: str) -> TestSuiteResult:
        """Run a single test suite and return results"""
        print(f"\n{'='*60}")
        print(f"Running {suite_name.upper()} Test Suite")
        print(f"{'='*60}")
        
        start_time = datetime.now()
        
        try:
            # Run pytest with JSON output
            cmd = [
                sys.executable, "-m", "pytest", 
                str(self.project_root / suite_path),
                "-v", "--tb=short", "--json-report", 
                f"--json-report-file={self.project_root}/test_results_{suite_name}.json"
            ]
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=300  # 5 minute timeout per suite
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Parse results
            json_report_path = self.project_root / f"test_results_{suite_name}.json"
            if json_report_path.exists():
                with open(json_report_path) as f:
                    json_data = json.load(f)
                
                summary = json_data.get("summary", {})
                tests_run = summary.get("total", 0)
                tests_passed = summary.get("passed", 0)
                tests_failed = summary.get("failed", 0)
                tests_skipped = summary.get("skipped", 0)
                
                # Clean up JSON report
                json_report_path.unlink()
            else:
                # Fallback parsing from stdout
                tests_run = result.stdout.count("PASSED") + result.stdout.count("FAILED")
                tests_passed = result.stdout.count("PASSED")
                tests_failed = result.stdout.count("FAILED")
                tests_skipped = result.stdout.count("SKIPPED")
            
            errors = []
            warnings = []
            
            if result.returncode != 0:
                errors.append(f"Test suite exited with code {result.returncode}")
                if result.stderr:
                    errors.append(result.stderr)
            
            if "warning" in result.stdout.lower():
                warnings.append("Warnings detected in test output")
            
            suite_result = TestSuiteResult(
                suite_name=suite_name,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                tests_run=tests_run,
                tests_passed=tests_passed,
                tests_failed=tests_failed,
                tests_skipped=tests_skipped,
                success_rate=0.0,  # Will be calculated in __post_init__
                errors=errors,
                warnings=warnings
            )
            
            print(f"✓ {suite_name} suite completed in {duration:.2f}s")
            print(f"  Tests: {tests_run}, Passed: {tests_passed}, Failed: {tests_failed}")
            
            return suite_result
            
        except subprocess.TimeoutExpired:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            return TestSuiteResult(
                suite_name=suite_name,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                tests_run=0,
                tests_passed=0,
                tests_failed=1,
                tests_skipped=0,
                success_rate=0.0,
                errors=[f"Test suite timed out after {duration:.2f} seconds"],
                warnings=[]
            )
        
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            return TestSuiteResult(
                suite_name=suite_name,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                tests_run=0,
                tests_passed=0,
                tests_failed=1,
                tests_skipped=0,
                success_rate=0.0,
                errors=[f"Test suite failed with exception: {str(e)}"],
                warnings=[]
            )
    
    def run_all_suites(self) -> ComprehensiveTestReport:
        """Run all test suites and generate comprehensive report"""
        print("Starting Comprehensive Test Suite Execution")
        print(f"Project Root: {self.project_root}")
        print(f"Test Suites: {list(self.test_suites.keys())}")
        
        execution_start = datetime.now()
        suite_results = []
        
        for suite_name, suite_path in self.test_suites.items():
            suite_result = self.run_single_suite(suite_name, suite_path)
            suite_results.append(suite_result)
        
        execution_end = datetime.now()
        total_duration = (execution_end - execution_start).total_seconds()
        
        # Generate comprehensive report
        report = ComprehensiveTestReport(
            execution_time=execution_start,
            total_duration_seconds=total_duration,
            suite_results=suite_results,
            overall_success_rate=0.0,  # Will be calculated in __post_init__
            total_tests_run=0,  # Will be calculated in __post_init__
            total_tests_passed=0,  # Will be calculated in __post_init__
            total_tests_failed=0,  # Will be calculated in __post_init__
            summary={},
            recommendations=[]
        )
        
        # Generate summary and recommendations
        report.summary = self._generate_summary(report)
        report.recommendations = self._generate_recommendations(report)
        
        return report
    
    def _generate_summary(self, report: ComprehensiveTestReport) -> Dict[str, Any]:
        """Generate summary statistics"""
        return {
            "execution_date": report.execution_time.isoformat(),
            "total_duration_minutes": round(report.total_duration_seconds / 60, 2),
            "suites_executed": len(report.suite_results),
            "suites_passed": len([r for r in report.suite_results if r.tests_failed == 0]),
            "suites_failed": len([r for r in report.suite_results if r.tests_failed > 0]),
            "overall_success_rate": round(report.overall_success_rate, 2),
            "performance_metrics": {
                "fastest_suite": min(report.suite_results, key=lambda r: r.duration_seconds).suite_name,
                "slowest_suite": max(report.suite_results, key=lambda r: r.duration_seconds).suite_name,
                "average_duration": round(sum(r.duration_seconds for r in report.suite_results) / len(report.suite_results), 2)
            }
        }
    
    def _generate_recommendations(self, report: ComprehensiveTestReport) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Overall success rate recommendations
        if report.overall_success_rate < 80:
            recommendations.append("Overall test success rate is below 80%. Focus on fixing failing tests.")
        elif report.overall_success_rate < 95:
            recommendations.append("Test success rate is good but could be improved. Review failing tests.")
        
        # Performance recommendations
        slow_suites = [r for r in report.suite_results if r.duration_seconds > 120]  # 2 minutes
        if slow_suites:
            suite_names = [r.suite_name for r in slow_suites]
            recommendations.append(f"Slow test suites detected: {', '.join(suite_names)}. Consider optimization.")
        
        # Error analysis recommendations
        suites_with_errors = [r for r in report.suite_results if r.errors]
        if suites_with_errors:
            recommendations.append("Some test suites have errors. Review error logs for details.")
        
        # Warning analysis recommendations
        suites_with_warnings = [r for r in report.suite_results if r.warnings]
        if suites_with_warnings:
            recommendations.append("Some test suites have warnings. Review for potential issues.")
        
        # Coverage recommendations
        if report.total_tests_run < 50:
            recommendations.append("Test coverage appears low. Consider adding more comprehensive tests.")
        
        return recommendations
    
    def save_report(self, report: ComprehensiveTestReport, output_path: Optional[Path] = None) -> Path:
        """Save comprehensive test report to file"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.project_root / f"comprehensive_test_report_{timestamp}.json"
        
        # Convert report to dictionary for JSON serialization
        report_dict = asdict(report)
        
        # Convert datetime objects to ISO strings
        for suite_result in report_dict["suite_results"]:
            suite_result["start_time"] = suite_result["start_time"].isoformat() if isinstance(suite_result["start_time"], datetime) else suite_result["start_time"]
            suite_result["end_time"] = suite_result["end_time"].isoformat() if isinstance(suite_result["end_time"], datetime) else suite_result["end_time"]
        
        report_dict["execution_time"] = report_dict["execution_time"].isoformat() if isinstance(report_dict["execution_time"], datetime) else report_dict["execution_time"]
        
        with open(output_path, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        return output_path
    
    def print_report(self, report: ComprehensiveTestReport):
        """Print comprehensive test report to console"""
        print(f"\n{'='*80}")
        print("COMPREHENSIVE TEST SUITE REPORT")
        print(f"{'='*80}")
        
        print(f"Execution Time: {report.execution_time}")
        print(f"Total Duration: {report.total_duration_seconds:.2f} seconds")
        print(f"Overall Success Rate: {report.overall_success_rate:.2f}%")
        print(f"Total Tests: {report.total_tests_run} (Passed: {report.total_tests_passed}, Failed: {report.total_tests_failed})")
        
        print(f"\n{'='*60}")
        print("SUITE RESULTS")
        print(f"{'='*60}")
        
        for result in report.suite_results:
            status = "✓ PASS" if result.tests_failed == 0 else "✗ FAIL"
            print(f"{status} {result.suite_name.upper()}")
            print(f"  Duration: {result.duration_seconds:.2f}s")
            print(f"  Tests: {result.tests_run} (Passed: {result.tests_passed}, Failed: {result.tests_failed})")
            print(f"  Success Rate: {result.success_rate:.2f}%")
            
            if result.errors:
                print(f"  Errors: {len(result.errors)}")
                for error in result.errors[:3]:  # Show first 3 errors
                    print(f"    - {error}")
            
            if result.warnings:
                print(f"  Warnings: {len(result.warnings)}")
            
            print()
        
        print(f"{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        
        for key, value in report.summary.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for sub_key, sub_value in value.items():
                    print(f"  {sub_key}: {sub_value}")
            else:
                print(f"{key}: {value}")
        
        if report.recommendations:
            print(f"\n{'='*60}")
            print("RECOMMENDATIONS")
            print(f"{'='*60}")
            
            for i, recommendation in enumerate(report.recommendations, 1):
                print(f"{i}. {recommendation}")
        
        print(f"\n{'='*80}")


def main():
    """Main entry point for comprehensive test runner"""
    import argparse

    parser = argparse.ArgumentParser(description="Run comprehensive test suite")
    parser.add_argument("--suite", choices=["e2e", "integration", "performance", "acceptance", "all"], 
                       default="all", help="Test suite to run")
    parser.add_argument("--output", type=Path, help="Output file for test report")
    parser.add_argument("--quiet", action="store_true", help="Suppress detailed output")
    
    args = parser.parse_args()
    
    runner = ComprehensiveTestRunner()
    
    if args.suite == "all":
        report = runner.run_all_suites()
    else:
        suite_path = runner.test_suites[args.suite]
        suite_result = runner.run_single_suite(args.suite, suite_path)
        
        # Create minimal report for single suite
        report = ComprehensiveTestReport(
            execution_time=suite_result.start_time,
            total_duration_seconds=suite_result.duration_seconds,
            suite_results=[suite_result],
            overall_success_rate=0.0,
            total_tests_run=0,
            total_tests_passed=0,
            total_tests_failed=0,
            summary={},
            recommendations=[]
        )
        report.summary = runner._generate_summary(report)
        report.recommendations = runner._generate_recommendations(report)
    
    # Save report
    output_path = runner.save_report(report, args.output)
    
    if not args.quiet:
        runner.print_report(report)
    
    print(f"\nReport saved to: {output_path}")
    
    # Exit with appropriate code
    if report.total_tests_failed > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
