#!/usr/bin/env python3
"""
Comprehensive test runner for WAN22 System Optimization
Runs all unit tests, integration tests, stress tests, and validation tests
"""

import unittest
import sys
import time
from pathlib import Path
import argparse


def discover_and_run_tests(test_pattern="test_*.py", verbosity=2):
    """Discover and run all tests matching the pattern"""
    
    # Get the directory containing this script
    test_dir = Path(__file__).parent
    
    # Discover tests
    loader = unittest.TestLoader()
    suite = loader.discover(str(test_dir), pattern=test_pattern)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbosity, buffer=True)
    result = runner.run(suite)
    
    return result


def run_test_suite(suite_name, pattern, description):
    """Run a specific test suite"""
    print(f"\n{'='*60}")
    print(f"Running {description}")
    print(f"{'='*60}")
    
    start_time = time.time()
    result = discover_and_run_tests(pattern)
    end_time = time.time()
    
    print(f"\n{description} completed in {end_time - start_time:.2f} seconds")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.failures:
        print(f"\nFailures in {suite_name}:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print(f"\nErrors in {suite_name}:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Exception:')[-1].strip()}")
    
    return result


def main():
    """Main test runner function"""
    parser = argparse.ArgumentParser(description="WAN22 System Optimization Test Runner")
    parser.add_argument("--suite", choices=["unit", "integration", "stress", "validation", "all"], 
                       default="all", help="Test suite to run")
    parser.add_argument("--verbose", "-v", action="count", default=1, 
                       help="Increase verbosity (use -v, -vv, or -vvv)")
    parser.add_argument("--fast", action="store_true", 
                       help="Skip slow tests (stress tests)")
    
    args = parser.parse_args()
    
    # Configure verbosity
    verbosity = min(args.verbose, 3)
    
    print("WAN22 System Optimization - Comprehensive Test Suite")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"Test directory: {Path(__file__).parent}")
    print(f"Verbosity level: {verbosity}")
    
    # Track overall results
    total_tests = 0
    total_failures = 0
    total_errors = 0
    total_skipped = 0
    
    start_time = time.time()
    
    # Define test suites
    test_suites = []
    
    if args.suite in ["unit", "all"]:
        test_suites.append(("Unit Tests", "test_*_unit.py", "Unit Tests for Individual Components"))
    
    if args.suite in ["integration", "all"]:
        test_suites.append(("Integration Tests", "test_*_integration.py", "Integration Tests for Component Interactions"))
    
    if args.suite in ["stress", "all"] and not args.fast:
        test_suites.append(("Stress Tests", "test_stress_*.py", "Stress Tests for High Load Conditions"))
    
    if args.suite in ["validation", "all"]:
        test_suites.append(("Validation Tests", "test_validation_*.py", "Validation Tests for Accuracy and Reliability"))
    
    # Run test suites
    for suite_name, pattern, description in test_suites:
        result = run_test_suite(suite_name, pattern, description)
        
        total_tests += result.testsRun
        total_failures += len(result.failures)
        total_errors += len(result.errors)
        total_skipped += len(result.skipped)
    
    # Overall summary
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n{'='*60}")
    print("OVERALL TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Total tests run: {total_tests}")
    print(f"Total failures: {total_failures}")
    print(f"Total errors: {total_errors}")
    print(f"Total skipped: {total_skipped}")
    
    # Calculate success rate
    if total_tests > 0:
        success_rate = ((total_tests - total_failures - total_errors) / total_tests) * 100
        print(f"Success rate: {success_rate:.1f}%")
    
    # Exit with appropriate code
    if total_failures > 0 or total_errors > 0:
        print(f"\n❌ Tests FAILED - {total_failures} failures, {total_errors} errors")
        sys.exit(1)
    else:
        print(f"\n✅ All tests PASSED - {total_tests} tests completed successfully")
        sys.exit(0)


if __name__ == "__main__":
    main()