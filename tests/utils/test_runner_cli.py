#!/usr/bin/env python3
"""
CLI wrapper for the Test Execution Engine

Provides easy-to-use command line interface for running tests with
timeout handling, retry logic, and parallel execution.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.test_execution_engine import TestExecutionEngine, TestConfig, TestCategory


def load_config(config_file: Optional[str] = None) -> TestConfig:
    """Load configuration from YAML file"""
    if config_file is None:
        config_file = Path(__file__).parent.parent / "config" / "execution_config.yaml"
    
    config = TestConfig()
    
    if Path(config_file).exists():
        with open(config_file, 'r') as f:
            yaml_config = yaml.safe_load(f)
            
        # Update timeouts
        if 'timeouts' in yaml_config:
            for category_name, timeout in yaml_config['timeouts'].items():
                try:
                    category = TestCategory(category_name)
                    config.timeouts[category] = timeout
                except ValueError:
                    logging.warning(f"Unknown test category: {category_name}")
                    
        # Update retry settings
        if 'retry' in yaml_config:
            retry_config = yaml_config['retry']
            config.max_retries = retry_config.get('max_retries', config.max_retries)
            config.retry_delay_base = retry_config.get('delay_base', config.retry_delay_base)
            config.retry_delay_max = retry_config.get('delay_max', config.retry_delay_max)
            
        # Update parallel settings
        if 'parallel' in yaml_config:
            parallel_config = yaml_config['parallel']
            config.max_workers = parallel_config.get('max_workers', config.max_workers)
            config.memory_limit_mb = parallel_config.get('memory_limit_mb', config.memory_limit_mb)
            
        # Update resource thresholds
        if 'resources' in yaml_config:
            resource_config = yaml_config['resources']
            config.cpu_threshold = resource_config.get('cpu_threshold', config.cpu_threshold)
            config.memory_threshold = resource_config.get('memory_threshold', config.memory_threshold)
            
        # Update flaky detection settings
        if 'flaky_detection' in yaml_config:
            flaky_config = yaml_config['flaky_detection']
            config.flaky_threshold = flaky_config.get('failure_threshold', config.flaky_threshold)
            config.flaky_success_rate = flaky_config.get('success_rate_threshold', config.flaky_success_rate)
            
    return config


def filter_tests_by_category(test_files: List[str], categories: List[str]) -> List[str]:
    """Filter test files by category"""
    if not categories:
        return test_files
        
    filtered_tests = []
    for test_file in test_files:
        test_path = test_file.lower()
        for category in categories:
            category_lower = category.lower()
            if (f"/{category_lower}/" in test_path or 
                f"test_{category_lower}_" in test_path or
                f"_{category_lower}_test.py" in test_path):
                filtered_tests.append(test_file)
                break
                
    return filtered_tests


def filter_tests_by_pattern(test_files: List[str], patterns: List[str]) -> List[str]:
    """Filter test files by pattern"""
    if not patterns:
        return test_files
        
    import fnmatch

    filtered_tests = []
    for test_file in test_files:
        for pattern in patterns:
            if fnmatch.fnmatch(test_file, pattern):
                filtered_tests.append(test_file)
                break
                
    return filtered_tests


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Test Execution Engine with Timeout Handling and Retry Logic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests
  python test_runner_cli.py
  
  # Run only unit tests
  python test_runner_cli.py --category unit
  
  # Run tests with custom timeout
  python test_runner_cli.py --timeout-unit 60
  
  # Run tests matching pattern
  python test_runner_cli.py --pattern "*integration*"
  
  # Run with more workers
  python test_runner_cli.py --workers 8
  
  # Generate detailed report
  python test_runner_cli.py --output test_report.txt --verbose
        """
    )
    
    # Test selection
    parser.add_argument(
        "--test-dir", 
        default="tests", 
        help="Test directory to scan (default: tests)"
    )
    parser.add_argument(
        "--category", 
        action="append", 
        choices=["unit", "integration", "e2e", "performance", "reliability"],
        help="Run tests from specific categories (can be used multiple times)"
    )
    parser.add_argument(
        "--pattern", 
        action="append",
        help="Run tests matching pattern (can be used multiple times)"
    )
    parser.add_argument(
        "--file", 
        action="append",
        help="Run specific test files (can be used multiple times)"
    )
    
    # Execution settings
    parser.add_argument(
        "--workers", 
        type=int, 
        help="Maximum parallel workers"
    )
    parser.add_argument(
        "--timeout-unit", 
        type=int, 
        help="Timeout for unit tests (seconds)"
    )
    parser.add_argument(
        "--timeout-integration", 
        type=int, 
        help="Timeout for integration tests (seconds)"
    )
    parser.add_argument(
        "--timeout-e2e", 
        type=int, 
        help="Timeout for e2e tests (seconds)"
    )
    parser.add_argument(
        "--timeout-performance", 
        type=int, 
        help="Timeout for performance tests (seconds)"
    )
    parser.add_argument(
        "--timeout-reliability", 
        type=int, 
        help="Timeout for reliability tests (seconds)"
    )
    
    # Retry settings
    parser.add_argument(
        "--max-retries", 
        type=int, 
        help="Maximum retry attempts for failed tests"
    )
    parser.add_argument(
        "--no-retry", 
        action="store_true", 
        help="Disable retry logic"
    )
    
    # Resource management
    parser.add_argument(
        "--cpu-threshold", 
        type=float, 
        help="CPU usage threshold for throttling (0.0-1.0)"
    )
    parser.add_argument(
        "--memory-threshold", 
        type=float, 
        help="Memory usage threshold for throttling (0.0-1.0)"
    )
    
    # Output and reporting
    parser.add_argument(
        "--output", 
        help="Output file for test report"
    )
    parser.add_argument(
        "--config", 
        help="Configuration file path"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true", 
        help="Verbose logging"
    )
    parser.add_argument(
        "--quiet", "-q", 
        action="store_true", 
        help="Quiet mode (minimal output)"
    )
    parser.add_argument(
        "--json-output", 
        help="Output results as JSON to specified file"
    )
    
    # Special modes
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Show what tests would be run without executing them"
    )
    parser.add_argument(
        "--list-categories", 
        action="store_true", 
        help="List available test categories and exit"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    if args.quiet:
        log_level = logging.ERROR
    elif args.verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
        
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    # Handle special modes
    if args.list_categories:
        print("Available test categories:")
        for category in TestCategory:
            print(f"  {category.value}")
        return 0
    
    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1
    
    # Override config with command line arguments
    if args.workers:
        config.max_workers = args.workers
    if args.max_retries is not None:
        config.max_retries = args.max_retries
    if args.no_retry:
        config.max_retries = 0
    if args.cpu_threshold:
        config.cpu_threshold = args.cpu_threshold
    if args.memory_threshold:
        config.memory_threshold = args.memory_threshold
        
    # Override timeouts
    if args.timeout_unit:
        config.timeouts[TestCategory.UNIT] = args.timeout_unit
    if args.timeout_integration:
        config.timeouts[TestCategory.INTEGRATION] = args.timeout_integration
    if args.timeout_e2e:
        config.timeouts[TestCategory.E2E] = args.timeout_e2e
    if args.timeout_performance:
        config.timeouts[TestCategory.PERFORMANCE] = args.timeout_performance
    if args.timeout_reliability:
        config.timeouts[TestCategory.RELIABILITY] = args.timeout_reliability
    
    # Create test execution engine
    engine = TestExecutionEngine(config)
    
    # Discover or select test files
    if args.file:
        test_files = args.file
    else:
        test_files = engine.discover_tests(args.test_dir)
        
        # Filter by category
        if args.category:
            test_files = filter_tests_by_category(test_files, args.category)
            
        # Filter by pattern
        if args.pattern:
            test_files = filter_tests_by_pattern(test_files, args.pattern)
    
    if not test_files:
        logger.error("No test files found matching criteria")
        return 1
    
    logger.info(f"Found {len(test_files)} test files")
    
    # Dry run mode
    if args.dry_run:
        print(f"Would run {len(test_files)} test files:")
        for test_file in sorted(test_files):
            category = engine.categorize_test(test_file)
            timeout = config.timeouts[category]
            print(f"  {test_file} ({category.value}, timeout: {timeout}s)")
        return 0
    
    # Run tests
    try:
        logger.info(f"Running {len(test_files)} tests with {config.max_workers} workers")
        result = engine.run_tests(test_files)
        
        # Generate report
        report_text = engine.generate_report(result, args.output)
        
        # Output results
        if not args.quiet:
            print(report_text)
        
        # JSON output
        if args.json_output:
            report_data = {
                'summary': {
                    'total_tests': result.total_tests,
                    'passed': result.passed,
                    'failed': result.failed,
                    'timeout': result.timeout,
                    'error': result.error,
                    'skipped': result.skipped,
                    'success_rate': result.passed / result.total_tests if result.total_tests > 0 else 0,
                    'total_duration': result.total_duration,
                    'start_time': result.start_time.isoformat(),
                    'end_time': result.end_time.isoformat()
                },
                'resource_usage': result.resource_usage,
                'flaky_tests': list(result.flaky_tests),
                'test_results': [
                    {
                        'test_id': r.test_id,
                        'category': r.category.value,
                        'status': r.status.value,
                        'duration': r.duration,
                        'retry_count': r.retry_count,
                        'error_message': r.error_message
                    }
                    for r in result.results
                ]
            }
            
            with open(args.json_output, 'w') as f:
                json.dump(report_data, f, indent=2)
            logger.info(f"JSON report saved to {args.json_output}")
        
        # Exit with appropriate code
        if result.failed > 0 or result.timeout > 0 or result.error > 0:
            return 1
        else:
            return 0
            
    except KeyboardInterrupt:
        logger.info("Test execution cancelled by user")
        return 130
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        if args.verbose:
            import traceback
traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())