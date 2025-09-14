#!/usr/bin/env python3
"""
Comprehensive test runner for Model Orchestrator.

This script runs all test suites and generates comprehensive reports
for end-to-end validation of the Model Orchestrator system.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import pytest


class TestRunner:
    """Comprehensive test runner for Model Orchestrator."""

    def __init__(self, verbose: bool = False, coverage: bool = True):
        self.verbose = verbose
        self.coverage = coverage
        self.results = {}
        self.start_time = time.time()

    def run_test_suite(self, suite_name: str, test_path: str, 
                      markers: Optional[List[str]] = None) -> Dict:
        """Run a specific test suite and return results."""
        print(f"\n{'='*60}")
        print(f"Running {suite_name}")
        print(f"{'='*60}")

        cmd = ["python", "-m", "pytest", test_path]
        
        if self.verbose:
            cmd.append("-v")
        
        if self.coverage:
            cmd.extend([
                "--cov=backend.core.model_orchestrator",
                "--cov-report=term-missing",
                "--cov-append"
            ])
        
        if markers:
            for marker in markers:
                cmd.extend(["-m", marker])
        
        cmd.extend([
            "--tb=short"
        ])
        
        # Add JSON report if plugin is available
        try:
            import pytest_json_report
            cmd.extend([
                "--json-report",
                f"--json-report-file=/tmp/{suite_name}_report.json"
            ])
        except ImportError:
            # JSON report plugin not available, skip it
            pass

        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent.parent.parent
            )
            
            duration = time.time() - start_time
            
            # Load JSON report if available
            report_file = f"/tmp/{suite_name}_report.json"
            test_details = {}
            if Path(report_file).exists():
                try:
                    with open(report_file) as f:
                        test_details = json.load(f)
                except json.JSONDecodeError:
                    pass
            
            suite_result = {
                'name': suite_name,
                'duration': duration,
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'success': result.returncode == 0,
                'details': test_details
            }
            
            if self.verbose:
                print(f"STDOUT:\n{result.stdout}")
                if result.stderr:
                    print(f"STDERR:\n{result.stderr}")
            
            print(f"Suite {suite_name}: {'PASSED' if suite_result['success'] else 'FAILED'} "
                  f"({duration:.2f}s)")
            
            return suite_result
            
        except Exception as e:
            return {
                'name': suite_name,
                'duration': time.time() - start_time,
                'return_code': -1,
                'success': False,
                'error': str(e)
            }

    def run_all_tests(self) -> Dict:
        """Run all test suites."""
        print("Starting comprehensive Model Orchestrator test suite")
        print(f"Platform: {platform.system()} {platform.release()}")
        print(f"Python: {sys.version}")
        
        # Define test suites
        test_suites = [
            {
                'name': 'unit_tests',
                'path': 'backend/core/model_orchestrator/test_*.py',
                'description': 'Unit tests for core components'
            },
            {
                'name': 'end_to_end',
                'path': 'backend/core/model_orchestrator/tests/test_end_to_end_workflows.py',
                'description': 'End-to-end workflow tests'
            },
            {
                'name': 'cross_platform',
                'path': 'backend/core/model_orchestrator/tests/test_cross_platform_compatibility.py',
                'description': 'Cross-platform compatibility tests'
            },
            {
                'name': 'performance_load',
                'path': 'backend/core/model_orchestrator/tests/test_performance_load.py',
                'description': 'Performance and load tests'
            }
        ]

        # Add platform-specific tests
        if platform.system() == "Windows":
            test_suites.append({
                'name': 'windows_specific',
                'path': 'backend/core/model_orchestrator/tests/test_cross_platform_compatibility.py',
                'description': 'Windows-specific tests',
                'markers': ['windows']
            })
        elif platform.system() == "Linux":
            test_suites.append({
                'name': 'linux_specific',
                'path': 'backend/core/model_orchestrator/tests/test_cross_platform_compatibility.py',
                'description': 'Linux-specific tests',
                'markers': ['linux']
            })

        # Run each test suite
        for suite_config in test_suites:
            suite_result = self.run_test_suite(
                suite_config['name'],
                suite_config['path'],
                suite_config.get('markers')
            )
            self.results[suite_config['name']] = suite_result

        return self.results

    def generate_report(self, output_file: Optional[str] = None) -> str:
        """Generate comprehensive test report."""
        total_duration = time.time() - self.start_time
        
        # Calculate summary statistics
        total_suites = len(self.results)
        passed_suites = sum(1 for r in self.results.values() if r['success'])
        failed_suites = total_suites - passed_suites
        
        # Generate report
        report = []
        report.append("=" * 80)
        report.append("MODEL ORCHESTRATOR COMPREHENSIVE TEST REPORT")
        report.append("=" * 80)
        report.append(f"Platform: {platform.system()} {platform.release()}")
        report.append(f"Python Version: {sys.version}")
        report.append(f"Test Duration: {total_duration:.2f} seconds")
        report.append(f"Total Test Suites: {total_suites}")
        report.append(f"Passed: {passed_suites}")
        report.append(f"Failed: {failed_suites}")
        report.append("")

        # Suite details
        for suite_name, result in self.results.items():
            status = "PASSED" if result['success'] else "FAILED"
            report.append(f"{suite_name}: {status} ({result['duration']:.2f}s)")
            
            if not result['success']:
                report.append(f"  Error: {result.get('error', 'Test failures')}")
                if 'stderr' in result and result['stderr']:
                    report.append(f"  Details: {result['stderr'][:200]}...")
            
            # Add test details if available
            if 'details' in result and 'summary' in result['details']:
                summary = result['details']['summary']
                report.append(f"  Tests: {summary.get('total', 0)} "
                             f"(passed: {summary.get('passed', 0)}, "
                             f"failed: {summary.get('failed', 0)}, "
                             f"skipped: {summary.get('skipped', 0)})")
        
        report.append("")
        report.append("=" * 80)
        
        overall_status = "PASSED" if failed_suites == 0 else "FAILED"
        report.append(f"OVERALL STATUS: {overall_status}")
        report.append("=" * 80)

        report_text = "\n".join(report)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"Report saved to: {output_file}")
        
        return report_text

    def run_coverage_report(self):
        """Generate coverage report."""
        if not self.coverage:
            return
        
        print("\nGenerating coverage report...")
        
        # Generate HTML coverage report
        subprocess.run([
            "python", "-m", "coverage", "html",
            "--directory", "coverage_html_report"
        ])
        
        # Generate coverage summary
        result = subprocess.run([
            "python", "-m", "coverage", "report"
        ], capture_output=True, text=True)
        
        print("Coverage Summary:")
        print(result.stdout)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive Model Orchestrator tests"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--no-coverage",
        action="store_true",
        help="Disable coverage reporting"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file for test report"
    )
    parser.add_argument(
        "--suite",
        choices=['unit', 'e2e', 'platform', 'performance', 'all'],
        default='all',
        help="Test suite to run"
    )

    args = parser.parse_args()

    # Initialize test runner
    runner = TestRunner(
        verbose=args.verbose,
        coverage=not args.no_coverage
    )

    try:
        # Run tests
        if args.suite == 'all':
            results = runner.run_all_tests()
        else:
            # Run specific suite
            suite_map = {
                'unit': ('unit_tests', 'backend/core/model_orchestrator/test_*.py'),
                'e2e': ('end_to_end', 'backend/core/model_orchestrator/tests/test_end_to_end_workflows.py'),
                'platform': ('cross_platform', 'backend/core/model_orchestrator/tests/test_cross_platform_compatibility.py'),
                'performance': ('performance_load', 'backend/core/model_orchestrator/tests/test_performance_load.py')
            }
            
            suite_name, suite_path = suite_map[args.suite]
            result = runner.run_test_suite(suite_name, suite_path)
            results = {suite_name: result}
            runner.results = results

        # Generate report
        report = runner.generate_report(args.output)
        print("\n" + report)

        # Generate coverage report
        runner.run_coverage_report()

        # Exit with appropriate code
        failed_suites = sum(1 for r in results.values() if not r['success'])
        sys.exit(0 if failed_suites == 0 else 1)

    except KeyboardInterrupt:
        print("\nTest run interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"Error running tests: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()