#!/usr/bin/env python3
"""
Example usage of the Test Execution Engine

This script demonstrates how to use the test execution engine
with various configurations and features.
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.test_execution_engine import TestExecutionEngine, TestConfig, TestCategory


def example_basic_usage():
    """Example of basic test execution"""
    print("=== Basic Usage Example ===")
    
    # Create default configuration
    config = TestConfig()
    
    # Create engine
    engine = TestExecutionEngine(config)
    
    # Discover tests
    test_files = engine.discover_tests("tests")
    print(f"Found {len(test_files)} test files")
    
    # Run a subset of tests for demo
    if test_files:
        demo_tests = test_files[:3]  # Run first 3 tests
        print(f"Running {len(demo_tests)} tests...")
        
        result = engine.run_tests(demo_tests)
        
        # Print summary
        print(f"Results: {result.passed} passed, {result.failed} failed, {result.timeout} timeout")
        print(f"Total duration: {result.total_duration:.2f}s")
        
        if result.flaky_tests:
            print(f"Flaky tests detected: {list(result.flaky_tests)}")


def example_custom_configuration():
    """Example with custom configuration"""
    print("\n=== Custom Configuration Example ===")
    
    # Create custom configuration
    config = TestConfig(
        timeouts={
            TestCategory.UNIT: 15,        # 15 seconds for unit tests
            TestCategory.INTEGRATION: 60, # 1 minute for integration tests
            TestCategory.E2E: 180,        # 3 minutes for e2e tests
        },
        max_retries=2,                    # Retry failed tests twice
        max_workers=4,                    # Use 4 parallel workers
        retry_delay_base=2.0,             # Start with 2 second delay
        cpu_threshold=0.7,                # Throttle at 70% CPU usage
        memory_threshold=0.8              # Throttle at 80% memory usage
    )
    
    # Create engine with custom config
    engine = TestExecutionEngine(config)
    
    print(f"Configuration:")
    print(f"  Unit test timeout: {config.timeouts[TestCategory.UNIT]}s")
    print(f"  Max retries: {config.max_retries}")
    print(f"  Max workers: {config.max_workers}")
    print(f"  CPU threshold: {config.cpu_threshold * 100}%")


def example_category_filtering():
    """Example of running tests by category"""
    print("\n=== Category Filtering Example ===")
    
    engine = TestExecutionEngine()
    
    # Discover all tests
    all_tests = engine.discover_tests("tests")
    
    # Categorize tests
    categories = {}
    for test_file in all_tests:
        category = engine.categorize_test(test_file)
        if category not in categories:
            categories[category] = []
        categories[category].append(test_file)
    
    # Print categorization
    for category, tests in categories.items():
        print(f"{category.value}: {len(tests)} tests")
        for test in tests[:3]:  # Show first 3 tests
            print(f"  - {test}")
        if len(tests) > 3:
            print(f"  ... and {len(tests) - 3} more")


def example_with_reporting():
    """Example with detailed reporting"""
    print("\n=== Reporting Example ===")
    
    engine = TestExecutionEngine()
    
    # Run tests (use a small subset for demo)
    test_files = engine.discover_tests("tests")
    if test_files:
        demo_tests = test_files[:2]  # Run first 2 tests
        
        print(f"Running {len(demo_tests)} tests with reporting...")
        result = engine.run_tests(demo_tests)
        
        # Generate report
        report = engine.generate_report(result, "test_report.txt")
        print("Report generated:")
        print(report[:500] + "..." if len(report) > 500 else report)


def example_error_handling():
    """Example of error handling"""
    print("\n=== Error Handling Example ===")
    
    engine = TestExecutionEngine()
    
    try:
        # Try to run non-existent test
        result = engine.run_tests(["non_existent_test.py"])
        print(f"Result: {result.total_tests} tests, {result.error} errors")
        
    except Exception as e:
        print(f"Caught exception: {e}")


def main():
    """Run all examples"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Test Execution Engine Examples")
    print("=" * 40)
    
    try:
        example_basic_usage()
        example_custom_configuration()
        example_category_filtering()
        example_with_reporting()
        example_error_handling()
        
        print("\n=== Examples Complete ===")
        print("Check test_report.txt for detailed report output")
        
    except KeyboardInterrupt:
        print("\nExamples interrupted by user")
    except Exception as e:
        print(f"Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()