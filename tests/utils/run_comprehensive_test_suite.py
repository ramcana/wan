"""
Comprehensive test suite runner for the Server Startup Management System.

Runs all unit tests, integration tests, and performance benchmarks,
generating detailed reports and coverage analysis.
"""

import pytest
import sys
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestSuiteRunner:
    """Comprehensive test suite runner with reporting."""
    
    def __init__(self):
        self.test_dir = Path(__file__).parent
        self.project_root = self.test_dir.parent
        self.results = {}
        self.start_time = None
        self.end_time = None
    
    def parse_pytest_output(self, output: str) -> Dict[str, Any]:
        """Parse pytest output to extract test results."""
        lines = output.split('\n')
        
        # Look for the summary line
        summary = {"total": 0, "passed": 0, "failed": 0, "skipped": 0}
        
        for line in lines:
            if "passed" in line or "failed" in line or "error" in line:
                # Try to extract numbers from lines like "5 passed, 2 failed in 1.23s"
                import re
                
                passed_match = re.search(r'(\d+) passed', line)
                if passed_match:
                    summary["passed"] = int(passed_match.group(1))
                
                failed_match = re.search(r'(\d+) failed', line)
                if failed_match:
                    summary["failed"] = int(failed_match.group(1))
                
                error_match = re.search(r'(\d+) error', line)
                if error_match:
                    summary["failed"] += int(error_match.group(1))
                
                skipped_match = re.search(r'(\d+) skipped', line)
                if skipped_match:
                    summary["skipped"] = int(skipped_match.group(1))
        
        summary["total"] = summary["passed"] + summary["failed"] + summary["skipped"]
        
        return {"summary": summary}
    
    def run_test_category(self, category: str, test_files: List[str], markers: str = None) -> Dict[str, Any]:
        """Run a category of tests and return results."""
        print(f"\n{'='*60}")
        print(f"Running {category} Tests")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Build pytest command
        cmd = ["python", "-m", "pytest", "-v", "--tb=short"]
        
        if markers:
            cmd.extend(["-m", markers])
        
        # Add test files
        for test_file in test_files:
            test_path = self.test_dir / test_file
            if test_path.exists():
                cmd.append(str(test_path))
        
        try:
            # Run tests
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per category
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Parse test results from stdout
            test_results = self.parse_pytest_output(result.stdout)
            
            return {
                "category": category,
                "duration": duration,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "test_results": test_results,
                "success": result.returncode == 0
            }
            
        except subprocess.TimeoutExpired:
            return {
                "category": category,
                "duration": 300,
                "return_code": -1,
                "stdout": "",
                "stderr": "Test category timed out after 5 minutes",
                "test_results": {},
                "success": False
            }
        except Exception as e:
            return {
                "category": category,
                "duration": time.time() - start_time,
                "return_code": -1,
                "stdout": "",
                "stderr": f"Error running tests: {str(e)}",
                "test_results": {},
                "success": False
            }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all test categories."""
        self.start_time = datetime.now()
        
        print("Starting Comprehensive Test Suite")
        print(f"Start time: {self.start_time}")
        print(f"Test directory: {self.test_dir}")
        print(f"Project root: {self.project_root}")
        
        # Define test categories
        test_categories = [
            {
                "name": "Unit Tests - Environment Validator",
                "files": ["test_environment_validator.py"],
                "markers": None
            },
            {
                "name": "Unit Tests - Port Manager", 
                "files": ["test_port_manager.py"],
                "markers": None
            },
            {
                "name": "Unit Tests - Process Manager",
                "files": ["test_process_manager.py"],
                "markers": None
            },
            {
                "name": "Unit Tests - Recovery Engine",
                "files": ["test_recovery_engine.py"],
                "markers": None
            },
            {
                "name": "Unit Tests - Configuration",
                "files": ["test_startup_manager_config.py", "test_startup_manager_utils.py"],
                "markers": None
            },
            {
                "name": "Unit Tests - CLI and Logging",
                "files": ["test_cli_interface.py", "test_logger.py"],
                "markers": None
            },
            {
                "name": "Integration Tests - Port Management",
                "files": ["test_port_manager_integration.py"],
                "markers": None
            },
            {
                "name": "Integration Tests - Process Lifecycle",
                "files": ["test_process_lifecycle_integration.py"],
                "markers": None
            },
            {
                "name": "Integration Tests - Recovery Engine",
                "files": ["test_recovery_engine_integration.py"],
                "markers": None
            },
            {
                "name": "Integration Tests - Configuration",
                "files": ["test_configuration_integration.py"],
                "markers": None
            },
            {
                "name": "Integration Tests - Error Handling",
                "files": ["test_error_handling_integration.py"],
                "markers": None
            },
            {
                "name": "Integration Tests - Windows Specific",
                "files": ["test_windows_integration.py"],
                "markers": None
            },
            {
                "name": "Comprehensive Integration Tests",
                "files": ["test_comprehensive_integration.py"],
                "markers": None
            },
            {
                "name": "Performance Benchmarks",
                "files": ["test_performance_benchmarks.py"],
                "markers": None
            }
        ]
        
        # Run each test category
        for category_config in test_categories:
            result = self.run_test_category(
                category_config["name"],
                category_config["files"],
                category_config["markers"]
            )
            self.results[category_config["name"]] = result
            
            # Print summary for this category
            if result["success"]:
                print(f"✅ {category_config['name']}: PASSED ({result['duration']:.1f}s)")
            else:
                print(f"❌ {category_config['name']}: FAILED ({result['duration']:.1f}s)")
                if result["stderr"]:
                    print(f"   Error: {result['stderr'][:200]}...")
        
        self.end_time = datetime.now()
        
        return self.generate_final_report()
    
    def generate_final_report(self) -> Dict[str, Any]:
        """Generate final comprehensive report."""
        total_duration = (self.end_time - self.start_time).total_seconds()
        
        # Calculate overall statistics
        total_categories = len(self.results)
        passed_categories = sum(1 for r in self.results.values() if r["success"])
        failed_categories = total_categories - passed_categories
        
        # Aggregate test statistics
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        skipped_tests = 0
        
        for result in self.results.values():
            if "test_results" in result and "summary" in result["test_results"]:
                summary = result["test_results"]["summary"]
                total_tests += summary.get("total", 0)
                passed_tests += summary.get("passed", 0)
                failed_tests += summary.get("failed", 0)
                skipped_tests += summary.get("skipped", 0)
        
        # Performance statistics
        performance_results = self.results.get("Performance Benchmarks", {})
        performance_success = performance_results.get("success", False)
        
        # Generate report
        report = {
            "test_run_info": {
                "start_time": self.start_time.isoformat(),
                "end_time": self.end_time.isoformat(),
                "total_duration": total_duration,
                "python_version": sys.version,
                "platform": sys.platform
            },
            "summary": {
                "total_categories": total_categories,
                "passed_categories": passed_categories,
                "failed_categories": failed_categories,
                "category_success_rate": passed_categories / total_categories if total_categories > 0 else 0,
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "skipped_tests": skipped_tests,
                "test_success_rate": passed_tests / total_tests if total_tests > 0 else 0,
                "performance_tests_passed": performance_success
            },
            "category_results": self.results,
            "recommendations": self.generate_recommendations()
        }
        
        return report
    
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        # Check for failed categories
        failed_categories = [name for name, result in self.results.items() if not result["success"]]
        
        if failed_categories:
            recommendations.append(
                f"Address failures in: {', '.join(failed_categories)}"
            )
        
        # Check performance
        perf_result = self.results.get("Performance Benchmarks", {})
        if not perf_result.get("success", False):
            recommendations.append(
                "Performance benchmarks failed - investigate startup time and resource usage"
            )
        
        # Check integration tests
        integration_failures = [
            name for name in failed_categories 
            if "Integration" in name
        ]
        if integration_failures:
            recommendations.append(
                "Integration test failures indicate potential system-level issues"
            )
        
        # Check unit test coverage
        unit_failures = [
            name for name in failed_categories 
            if "Unit Tests" in name
        ]
        if unit_failures:
            recommendations.append(
                "Unit test failures indicate component-level issues that should be fixed first"
            )
        
        if not recommendations:
            recommendations.append("All tests passed! System is ready for deployment.")
        
        return recommendations
    
    def save_report(self, report: Dict[str, Any], filename: str = None):
        """Save the test report to a file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_report_{timestamp}.json"
        
        report_path = self.test_dir / filename
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\nDetailed report saved to: {report_path}")
        return report_path
    
    def print_summary(self, report: Dict[str, Any]):
        """Print a summary of the test results."""
        print(f"\n{'='*80}")
        print("COMPREHENSIVE TEST SUITE SUMMARY")
        print(f"{'='*80}")
        
        summary = report["summary"]
        
        print(f"Total Duration: {report['test_run_info']['total_duration']:.1f} seconds")
        print(f"Categories: {summary['passed_categories']}/{summary['total_categories']} passed "
              f"({summary['category_success_rate']:.1%})")
        print(f"Tests: {summary['passed_tests']}/{summary['total_tests']} passed "
              f"({summary['test_success_rate']:.1%})")
        
        if summary['failed_tests'] > 0:
            print(f"Failed Tests: {summary['failed_tests']}")
        
        if summary['skipped_tests'] > 0:
            print(f"Skipped Tests: {summary['skipped_tests']}")
        
        print(f"Performance Tests: {'✅ PASSED' if summary['performance_tests_passed'] else '❌ FAILED'}")
        
        print(f"\nRecommendations:")
        for i, rec in enumerate(report["recommendations"], 1):
            print(f"  {i}. {rec}")
        
        # Overall status
        overall_success = (
            summary['category_success_rate'] >= 0.9 and 
            summary['test_success_rate'] >= 0.95 and
            summary['performance_tests_passed']
        )
        
        print(f"\n{'='*80}")
        if overall_success:
            print("🎉 OVERALL STATUS: SUCCESS - System ready for deployment!")
        else:
            print("⚠️  OVERALL STATUS: NEEDS ATTENTION - Address failures before deployment")
        print(f"{'='*80}")


def main():
    """Main entry point for the test suite runner."""
    runner = TestSuiteRunner()
    
    try:
        # Run all tests
        report = runner.run_all_tests()
        
        # Save and display results
        report_path = runner.save_report(report)
        runner.print_summary(report)
        
        # Exit with appropriate code
        if report["summary"]["category_success_rate"] >= 0.9:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\nTest suite interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\nError running test suite: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
