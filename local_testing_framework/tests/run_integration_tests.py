#!/usr/bin/env python3
"""
Integration and End-to-End Test Runner
Runs comprehensive integration tests for the Local Testing Framework
"""

import sys
import unittest
import time
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class IntegrationTestResult:
    """Custom test result class for integration tests"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.end_time = None
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.error_tests = 0
        self.skipped_tests = 0
        self.failures = []
        self.errors = []
        self.test_details = []
    
    def add_test_result(self, test_name, status, duration, details=None):
        """Add a test result"""
        self.total_tests += 1
        
        if status == "PASS":
            self.passed_tests += 1
        elif status == "FAIL":
            self.failed_tests += 1
            if details:
                self.failures.append((test_name, details))
        elif status == "ERROR":
            self.error_tests += 1
            if details:
                self.errors.append((test_name, details))
        elif status == "SKIP":
            self.skipped_tests += 1
        
        self.test_details.append({
            "name": test_name,
            "status": status,
            "duration": duration,
            "details": details
        })
    
    def finalize(self):
        """Finalize the test results"""
        self.end_time = datetime.now()
    
    def get_summary(self):
        """Get test summary"""
        duration = (self.end_time - self.start_time).total_seconds()
        
        return {
            "total_duration": duration,
            "total_tests": self.total_tests,
            "passed": self.passed_tests,
            "failed": self.failed_tests,
            "errors": self.error_tests,
            "skipped": self.skipped_tests,
            "success_rate": (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0
        }
    
    def print_summary(self):
        """Print test summary"""
        summary = self.get_summary()
        
        print("\n" + "="*80)
        print("INTEGRATION TEST SUMMARY")
        print("="*80)
        print(f"Total Duration: {summary['total_duration']:.2f} seconds")
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Errors: {summary['errors']}")
        print(f"Skipped: {summary['skipped']}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        
        if self.failures:
            print("\nFAILURES:")
            for test_name, details in self.failures:
                print(f"  - {test_name}: {details}")
        
        if self.errors:
            print("\nERRORS:")
            for test_name, details in self.errors:
                print(f"  - {test_name}: {details}")
        
        print("="*80)


def run_test_suite(test_module_name, description):
    """Run a specific test suite"""
    print(f"\n{'='*60}")
    print(f"Running {description}")
    print(f"Module: {test_module_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Import and run the test module
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromName(test_module_name)
        
        runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
        result = runner.run(suite)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n{description} completed in {duration:.2f} seconds")
        print(f"Tests run: {result.testsRun}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
        
        return {
            "module": test_module_name,
            "description": description,
            "duration": duration,
            "tests_run": result.testsRun,
            "failures": len(result.failures),
            "errors": len(result.errors),
            "skipped": len(result.skipped) if hasattr(result, 'skipped') else 0,
            "success": len(result.failures) == 0 and len(result.errors) == 0
        }
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\nERROR running {description}: {e}")
        
        return {
            "module": test_module_name,
            "description": description,
            "duration": duration,
            "tests_run": 0,
            "failures": 0,
            "errors": 1,
            "skipped": 0,
            "success": False,
            "error": str(e)
        }


def run_integration_tests():
    """Run all integration and end-to-end tests"""
    print("Starting Integration and End-to-End Test Suite")
    print(f"Start time: {datetime.now()}")
    
    overall_start = time.time()
    results = []
    
    # Define test suites to run
    test_suites = [
        ("local_testing_framework.tests.test_integration_workflows", "Component Integration Tests"),
        ("local_testing_framework.tests.test_end_to_end", "End-to-End Workflow Tests"),
        ("local_testing_framework.tests.test_cross_platform", "Cross-Platform Compatibility Tests"),
    ]
    
    # Run each test suite
    for module_name, description in test_suites:
        try:
            result = run_test_suite(module_name, description)
            results.append(result)
        except KeyboardInterrupt:
            print("\nTest execution interrupted by user")
            break
        except Exception as e:
            print(f"\nUnexpected error running {description}: {e}")
            results.append({
                "module": module_name,
                "description": description,
                "duration": 0,
                "tests_run": 0,
                "failures": 0,
                "errors": 1,
                "skipped": 0,
                "success": False,
                "error": str(e)
            })
    
    overall_end = time.time()
    overall_duration = overall_end - overall_start
    
    # Print overall summary
    print("\n" + "="*80)
    print("OVERALL INTEGRATION TEST SUMMARY")
    print("="*80)
    print(f"Total Duration: {overall_duration:.2f} seconds")
    print(f"Test Suites Run: {len(results)}")
    
    total_tests = sum(r["tests_run"] for r in results)
    total_failures = sum(r["failures"] for r in results)
    total_errors = sum(r["errors"] for r in results)
    total_skipped = sum(r["skipped"] for r in results)
    successful_suites = sum(1 for r in results if r["success"])
    
    print(f"Total Tests: {total_tests}")
    print(f"Total Failures: {total_failures}")
    print(f"Total Errors: {total_errors}")
    print(f"Total Skipped: {total_skipped}")
    print(f"Successful Suites: {successful_suites}/{len(results)}")
    
    if total_tests > 0:
        success_rate = ((total_tests - total_failures - total_errors) / total_tests) * 100
        print(f"Overall Success Rate: {success_rate:.1f}%")
    
    print("\nSUITE DETAILS:")
    for result in results:
        status = "✓ PASS" if result["success"] else "✗ FAIL"
        print(f"  {status} {result['description']} ({result['duration']:.1f}s)")
        if not result["success"]:
            if "error" in result:
                print(f"    Error: {result['error']}")
            else:
                print(f"    Failures: {result['failures']}, Errors: {result['errors']}")
    
    print("="*80)
    
    # Return exit code
    if total_failures > 0 or total_errors > 0:
        return 1
    else:
        return 0


def run_specific_test_class(module_name, class_name):
    """Run a specific test class"""
    print(f"Running specific test class: {class_name} from {module_name}")
    
    try:
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromName(f"{module_name}.{class_name}")
        
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        return 0 if result.wasSuccessful() else 1
        
    except Exception as e:
        print(f"Error running test class {class_name}: {e}")
        return 1


def run_specific_test_method(module_name, class_name, method_name):
    """Run a specific test method"""
    print(f"Running specific test method: {method_name} from {class_name} in {module_name}")
    
    try:
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromName(f"{module_name}.{class_name}.{method_name}")
        
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        return 0 if result.wasSuccessful() else 1
        
    except Exception as e:
        print(f"Error running test method {method_name}: {e}")
        return 1


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Integration and End-to-End Test Runner")
    parser.add_argument("--suite", help="Run specific test suite (integration, end-to-end, cross-platform)")
    parser.add_argument("--class", dest="test_class", help="Run specific test class")
    parser.add_argument("--method", help="Run specific test method")
    parser.add_argument("--list", action="store_true", help="List available test suites")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.list:
        print("Available test suites:")
        print("  integration - Component integration tests")
        print("  end-to-end - End-to-end workflow tests")
        print("  cross-platform - Cross-platform compatibility tests")
        return 0
    
    if args.method and args.test_class:
        # Run specific test method
        if args.suite == "integration":
            module = "local_testing_framework.tests.test_integration_workflows"
        elif args.suite == "end-to-end":
            module = "local_testing_framework.tests.test_end_to_end"
        elif args.suite == "cross-platform":
            module = "local_testing_framework.tests.test_cross_platform"
        else:
            print("Error: --suite required when specifying --class and --method")
            return 1
        
        return run_specific_test_method(module, args.test_class, args.method)
    
    elif args.test_class:
        # Run specific test class
        if args.suite == "integration":
            module = "local_testing_framework.tests.test_integration_workflows"
        elif args.suite == "end-to-end":
            module = "local_testing_framework.tests.test_end_to_end"
        elif args.suite == "cross-platform":
            module = "local_testing_framework.tests.test_cross_platform"
        else:
            print("Error: --suite required when specifying --class")
            return 1
        
        return run_specific_test_class(module, args.test_class)
    
    elif args.suite:
        # Run specific test suite
        if args.suite == "integration":
            return run_test_suite("local_testing_framework.tests.test_integration_workflows", 
                                "Component Integration Tests")["success"]
        elif args.suite == "end-to-end":
            return run_test_suite("local_testing_framework.tests.test_end_to_end", 
                                "End-to-End Workflow Tests")["success"]
        elif args.suite == "cross-platform":
            return run_test_suite("local_testing_framework.tests.test_cross_platform", 
                                "Cross-Platform Compatibility Tests")["success"]
        else:
            print(f"Unknown test suite: {args.suite}")
            return 1
    
    else:
        # Run all integration tests
        return run_integration_tests()


if __name__ == '__main__':
    sys.exit(main())