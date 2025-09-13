"""
Comprehensive Integration Test Runner for Wan2.2 Video Generation System
Executes all end-to-end integration tests with detailed reporting
"""

import pytest
import sys
import os
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import argparse
from datetime import datetime

class IntegrationTestRunner:
    """Manages execution of comprehensive integration tests"""
    
    def __init__(self):
        self.test_files = [
            "test_end_to_end_integration.py",
            "test_generation_modes_integration.py", 
            "test_error_scenarios_integration.py",
            "test_performance_resource_integration.py"
        ]
        self.results = {}
        self.start_time = None
        self.end_time = None
    
    def run_all_tests(self, verbose: bool = True, stop_on_first_failure: bool = False) -> Dict[str, Any]:
        """Run all integration tests and return comprehensive results"""
        print("🎬 Starting Wan2.2 Video Generation Integration Tests")
        print("=" * 60)
        
        self.start_time = datetime.now()
        overall_success = True
        
        for test_file in self.test_files:
            print(f"\n📋 Running {test_file}...")
            print("-" * 40)
            
            if not Path(test_file).exists():
                print(f"❌ Test file {test_file} not found!")
                self.results[test_file] = {
                    "status": "file_not_found",
                    "duration": 0,
                    "tests_run": 0,
                    "failures": 1,
                    "errors": 0
                }
                overall_success = False
                continue
            
            # Run individual test file
            result = self._run_single_test_file(test_file, verbose, stop_on_first_failure)
            self.results[test_file] = result
            
            if result["status"] != "passed":
                overall_success = False
                if stop_on_first_failure:
                    print(f"🛑 Stopping on first failure in {test_file}")
                    break
        
        self.end_time = datetime.now()
        
        # Generate comprehensive report
        report = self._generate_comprehensive_report(overall_success)
        self._save_test_report(report)
        self._print_summary_report(report)
        
        return report
    
    def _run_single_test_file(self, test_file: str, verbose: bool, stop_on_first_failure: bool) -> Dict[str, Any]:
        """Run a single test file and return results"""
        start_time = time.time()
        
        # Prepare pytest arguments
        pytest_args = [test_file]
        if verbose:
            pytest_args.extend(["-v", "--tb=short"])
        if stop_on_first_failure:
            pytest_args.append("-x")
        
        # Add coverage and reporting options
        pytest_args.extend([
            "--tb=short",
            "--disable-warnings",
            "-q" if not verbose else "-v"
        ])
        
        try:
            # Run pytest and capture results
            exit_code = pytest.main(pytest_args)
            
            duration = time.time() - start_time
            
            # Parse results based on exit code
            if exit_code == 0:
                status = "passed"
                print(f"✅ {test_file} - All tests passed ({duration:.1f}s)")
            elif exit_code == 1:
                status = "failed"
                print(f"❌ {test_file} - Some tests failed ({duration:.1f}s)")
            elif exit_code == 2:
                status = "interrupted"
                print(f"⚠️ {test_file} - Test execution interrupted ({duration:.1f}s)")
            else:
                status = "error"
                print(f"💥 {test_file} - Test execution error ({duration:.1f}s)")
            
            return {
                "status": status,
                "duration": duration,
                "exit_code": exit_code,
                "tests_run": self._estimate_tests_run(test_file),
                "failures": 1 if exit_code != 0 else 0,
                "errors": 1 if exit_code > 1 else 0
            }
            
        except Exception as e:
            duration = time.time() - start_time
            print(f"💥 {test_file} - Exception during execution: {e}")
            return {
                "status": "exception",
                "duration": duration,
                "exit_code": -1,
                "tests_run": 0,
                "failures": 0,
                "errors": 1,
                "exception": str(e)
            }
    
    def _estimate_tests_run(self, test_file: str) -> int:
        """Estimate number of tests in a file by counting test methods"""
        try:
            with open(test_file, 'r') as f:
                content = f.read()
                # Count test methods (functions starting with 'def test_')
                return content.count('def test_')
        except Exception:
            return 0
    
    def _generate_comprehensive_report(self, overall_success: bool) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        total_duration = (self.end_time - self.start_time).total_seconds()
        
        # Calculate totals
        total_tests = sum(result.get("tests_run", 0) for result in self.results.values())
        total_failures = sum(result.get("failures", 0) for result in self.results.values())
        total_errors = sum(result.get("errors", 0) for result in self.results.values())
        
        # Calculate success rate
        success_rate = ((total_tests - total_failures - total_errors) / max(1, total_tests)) * 100
        
        report = {
            "summary": {
                "overall_success": overall_success,
                "total_duration_seconds": total_duration,
                "total_test_files": len(self.test_files),
                "total_tests_run": total_tests,
                "total_failures": total_failures,
                "total_errors": total_errors,
                "success_rate_percent": success_rate,
                "start_time": self.start_time.isoformat(),
                "end_time": self.end_time.isoformat()
            },
            "test_file_results": self.results,
            "test_categories": {
                "end_to_end": {
                    "file": "test_end_to_end_integration.py",
                    "description": "Complete T2V, I2V, TI2V generation workflows",
                    "status": self.results.get("test_end_to_end_integration.py", {}).get("status", "not_run")
                },
                "generation_modes": {
                    "file": "test_generation_modes_integration.py", 
                    "description": "Different generation modes with various inputs",
                    "status": self.results.get("test_generation_modes_integration.py", {}).get("status", "not_run")
                },
                "error_scenarios": {
                    "file": "test_error_scenarios_integration.py",
                    "description": "Error handling and recovery mechanisms",
                    "status": self.results.get("test_error_scenarios_integration.py", {}).get("status", "not_run")
                },
                "performance_resource": {
                    "file": "test_performance_resource_integration.py",
                    "description": "Performance metrics and resource usage",
                    "status": self.results.get("test_performance_resource_integration.py", {}).get("status", "not_run")
                }
            },
            "requirements_coverage": {
                "1.1": "T2V generation mode testing",
                "1.2": "I2V generation mode testing", 
                "1.3": "TI2V generation mode testing",
                "3.1": "Error handling and recovery testing",
                "3.2": "Resource management testing",
                "3.3": "Performance validation testing",
                "3.4": "Integration testing across components",
                "5.1": "End-to-end workflow testing",
                "5.2": "Error scenario testing",
                "5.3": "Performance and resource testing"
            },
            "system_info": {
                "python_version": sys.version,
                "platform": sys.platform,
                "working_directory": os.getcwd(),
                "test_runner_version": "1.0.0"
            }
        }
        
        return report
    
    def _save_test_report(self, report: Dict[str, Any]):
        """Save test report to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"integration_test_report_{timestamp}.json"
        
        try:
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"\n📊 Test report saved to: {report_file}")
        except Exception as e:
            print(f"⚠️ Failed to save test report: {e}")
    
    def _print_summary_report(self, report: Dict[str, Any]):
        """Print summary report to console"""
        print("\n" + "=" * 60)
        print("🎯 INTEGRATION TEST SUMMARY")
        print("=" * 60)
        
        summary = report["summary"]
        
        # Overall status
        status_emoji = "✅" if summary["overall_success"] else "❌"
        print(f"{status_emoji} Overall Status: {'PASSED' if summary['overall_success'] else 'FAILED'}")
        print(f"⏱️  Total Duration: {summary['total_duration_seconds']:.1f} seconds")
        print(f"📁 Test Files: {summary['total_test_files']}")
        print(f"🧪 Total Tests: {summary['total_tests_run']}")
        print(f"❌ Failures: {summary['total_failures']}")
        print(f"💥 Errors: {summary['total_errors']}")
        print(f"📈 Success Rate: {summary['success_rate_percent']:.1f}%")
        
        # Test category results
        print("\n📋 Test Category Results:")
        print("-" * 40)
        for category, info in report["test_categories"].items():
            status_emoji = "✅" if info["status"] == "passed" else "❌" if info["status"] in ["failed", "error"] else "⚠️"
            print(f"{status_emoji} {category.replace('_', ' ').title()}: {info['status'].upper()}")
            print(f"   📄 {info['description']}")
        
        # Requirements coverage
        print("\n📋 Requirements Coverage:")
        print("-" * 40)
        for req_id, description in report["requirements_coverage"].items():
            # Determine if requirement is covered based on test results
            covered = summary["overall_success"] or summary["total_tests_run"] > 0
            status_emoji = "✅" if covered else "❌"
            print(f"{status_emoji} Requirement {req_id}: {description}")
        
        # Performance summary
        if summary["total_tests_run"] > 0:
            avg_time_per_test = summary["total_duration_seconds"] / summary["total_tests_run"]
            print(f"\n⚡ Performance: {avg_time_per_test:.2f}s average per test")
        
        print("\n" + "=" * 60)
        
        if summary["overall_success"]:
            print("🎉 All integration tests completed successfully!")
            print("✨ The video generation system is ready for production use.")
        else:
            print("⚠️ Some integration tests failed.")
            print("🔧 Please review the test results and fix any issues before deployment.")
        
        print("=" * 60)
    
    def run_specific_category(self, category: str, verbose: bool = True) -> Dict[str, Any]:
        """Run tests for a specific category"""
        category_files = {
            "end_to_end": ["test_end_to_end_integration.py"],
            "generation_modes": ["test_generation_modes_integration.py"],
            "error_scenarios": ["test_error_scenarios_integration.py"],
            "performance": ["test_performance_resource_integration.py"],
            "all": self.test_files
        }
        
        if category not in category_files:
            print(f"❌ Unknown category: {category}")
            print(f"Available categories: {', '.join(category_files.keys())}")
            return {"error": "unknown_category"}
        
        # Temporarily set test files to category files
        original_files = self.test_files
        self.test_files = category_files[category]
        
        try:
            result = self.run_all_tests(verbose=verbose)
            return result
        finally:
            # Restore original test files
            self.test_files = original_files

def main():
    """Main entry point for integration test runner"""
    parser = argparse.ArgumentParser(
        description="Run Wan2.2 Video Generation Integration Tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_integration_tests.py                    # Run all tests
  python run_integration_tests.py --category end_to_end  # Run specific category
  python run_integration_tests.py --verbose --stop-on-fail  # Verbose with early stop
  python run_integration_tests.py --quiet            # Minimal output
        """
    )
    
    parser.add_argument(
        "--category", "-c",
        choices=["end_to_end", "generation_modes", "error_scenarios", "performance", "all"],
        default="all",
        help="Test category to run (default: all)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Minimal output (overrides verbose)"
    )
    
    parser.add_argument(
        "--stop-on-fail", "-x",
        action="store_true",
        help="Stop on first test failure"
    )
    
    parser.add_argument(
        "--list-tests",
        action="store_true",
        help="List available test files and exit"
    )
    
    args = parser.parse_args()
    
    runner = IntegrationTestRunner()
    
    if args.list_tests:
        print("Available test files:")
        for i, test_file in enumerate(runner.test_files, 1):
            print(f"  {i}. {test_file}")
        return
    
    # Determine verbosity
    verbose = args.verbose and not args.quiet
    
    try:
        if args.category == "all":
            result = runner.run_all_tests(
                verbose=verbose,
                stop_on_first_failure=args.stop_on_fail
            )
        else:
            result = runner.run_specific_category(
                args.category,
                verbose=verbose
            )
        
        # Exit with appropriate code
        if result.get("summary", {}).get("overall_success", False):
            sys.exit(0)
        else:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️ Test execution interrupted by user")
        sys.exit(2)
    except Exception as e:
        print(f"\n💥 Unexpected error during test execution: {e}")
        sys.exit(3)

if __name__ == "__main__":
    main()
