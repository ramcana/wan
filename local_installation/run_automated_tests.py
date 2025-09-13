#!/usr/bin/env python3
"""
Main test runner for the WAN2.2 local installation automated test framework.
Runs comprehensive tests including unit tests, integration tests, and hardware simulation.
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Add tests directory to path
sys.path.insert(0, str(Path(__file__).parent / "tests"))

from tests.test_runner import AutomatedTestFramework
from tests.test_config import setup_test_environment, cleanup_test_environment, TestReportGenerator


def main():
    """Main entry point for the automated test framework."""
    parser = argparse.ArgumentParser(
        description="WAN2.2 Local Installation Automated Test Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_automated_tests.py                    # Run all tests
  python run_automated_tests.py --suite unit       # Run only unit tests
  python run_automated_tests.py --suite integration # Run only integration tests
  python run_automated_tests.py --suite hardware   # Run only hardware simulation tests
  python run_automated_tests.py --verbose          # Run with verbose output
  python run_automated_tests.py --no-cleanup       # Don't cleanup test files
  python run_automated_tests.py --html-report      # Generate HTML report
        """
    )
    
    parser.add_argument(
        "--suite",
        choices=["unit", "integration", "hardware", "all"],
        default="all",
        help="Test suite to run (default: all)"
    )
    
    parser.add_argument(
        "--installation-path",
        default=".",
        help="Installation path to test (default: current directory)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Don't cleanup temporary test files"
    )
    
    parser.add_argument(
        "--html-report",
        action="store_true",
        help="Generate HTML test report"
    )
    
    parser.add_argument(
        "--output-dir",
        default="test_results",
        help="Output directory for test reports (default: test_results)"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Total test timeout in seconds (default: 1800)"
    )
    
    args = parser.parse_args()
    
    # Print header
    print("=" * 80)
    print("WAN2.2 Local Installation Automated Test Framework")
    print("=" * 80)
    print(f"Test Suite: {args.suite}")
    print(f"Installation Path: {args.installation_path}")
    print(f"Verbose Output: {args.verbose}")
    print(f"Cleanup: {not args.no_cleanup}")
    print(f"Timeout: {args.timeout}s")
    print("=" * 80)
    
    # Set up test environment
    test_config = {
        "verbose_output": args.verbose,
        "cleanup_after_tests": not args.no_cleanup,
        "total_timeout": args.timeout
    }
    
    test_env = setup_test_environment(test_config)
    
    try:
        # Initialize test framework
        framework = AutomatedTestFramework(args.installation_path)
        
        # Run tests based on suite selection
        start_time = time.time()
        
        if args.suite == "all":
            print("Running all test suites...")
            report = framework.run_all_tests()
        else:
            # Run specific suite
            suite_map = {
                "unit": "unit_tests",
                "integration": "integration_tests",
                "hardware": "hardware_simulation_tests"
            }
            
            suite_name = suite_map[args.suite]
            suite = framework.test_suites[suite_name]
            
            print(f"Running {suite.name}...")
            suite_result = framework._run_test_suite(suite)
            
            # Create report structure
            report = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_duration": time.time() - start_time,
                "suite_results": {suite_name: suite_result},
                "overall_stats": {
                    "total_tests": suite_result["total"],
                    "total_passed": suite_result["passed"],
                    "total_failed": suite_result["total"] - suite_result["passed"],
                    "success_rate": suite_result["success_rate"]
                },
                "summary": {
                    "status": "PASSED" if suite_result["success_rate"] >= 0.9 else "FAILED",
                    "critical_failures": [],
                    "recommendations": []
                }
            }
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save JSON report
        json_report_file = output_dir / "test_report.json"
        TestReportGenerator.generate_test_summary(report, json_report_file)
        print(f"\n📊 JSON report saved to: {json_report_file}")
        
        # Generate HTML report if requested
        if args.html_report:
            html_report_file = output_dir / "test_report.html"
            TestReportGenerator.generate_html_report(report, html_report_file)
            print(f"🌐 HTML report saved to: {html_report_file}")
        
        # Print final summary
        print("\n" + "=" * 80)
        print("FINAL TEST SUMMARY")
        print("=" * 80)
        
        overall_stats = report["overall_stats"]
        print(f"Total Tests: {overall_stats['total_tests']}")
        print(f"Passed: {overall_stats['total_passed']}")
        print(f"Failed: {overall_stats['total_failed']}")
        print(f"Success Rate: {overall_stats['success_rate']:.1%}")
        print(f"Duration: {report['total_duration']:.2f}s")
        print(f"Status: {report['summary']['status']}")
        
        # Print critical failures
        if report['summary']['critical_failures']:
            print("\nCritical Failures:")
            for failure in report['summary']['critical_failures']:
                print(f"  ❌ {failure}")
        
        # Print recommendations
        if report['summary']['recommendations']:
            print("\nRecommendations:")
            for rec in report['summary']['recommendations']:
                print(f"  💡 {rec}")
        
        # Determine exit code
        success_rate = overall_stats['success_rate']
        if success_rate >= 0.9:
            print(f"\n🎉 All tests completed successfully!")
            exit_code = 0
        elif success_rate >= 0.7:
            print(f"\n⚠️  Some tests failed, but core functionality appears to work.")
            exit_code = 1
        else:
            print(f"\n❌ Many tests failed. Please review and fix issues before deployment.")
            exit_code = 2
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Test execution interrupted by user.")
        exit_code = 130
        
    except Exception as e:
        print(f"\n\n❌ Test framework error: {e}")
        import traceback
        traceback.print_exc()
        exit_code = 3
        
    finally:
        # Cleanup test environment
        if not args.no_cleanup:
            cleanup_test_environment(test_env)
            print(f"\n🧹 Test environment cleaned up.")
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
