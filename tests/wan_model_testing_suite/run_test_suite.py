#!/usr/bin/env python3
"""
WAN Model Testing Suite Runner

This script runs the complete WAN model testing suite with configurable options.
"""

import argparse
import sys
import os
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def run_command(cmd: List[str], cwd: Optional[Path] = None) -> tuple[int, str, str]:
    """Run a command and return exit code, stdout, stderr"""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except Exception as e:
        return -1, "", str(e)


def run_unit_tests(verbose: bool = False, coverage: bool = False) -> bool:
    """Run unit tests"""
    print("Running unit tests...")
    
    cmd = ["python", "-m", "pytest", "tests/wan_model_testing_suite/unit/", "-v" if verbose else "-q"]
    
    if coverage:
        cmd.extend(["--cov=core.models.wan_models", "--cov-report=html", "--cov-report=term"])
    
    exit_code, stdout, stderr = run_command(cmd)
    
    if verbose:
        print(stdout)
    if stderr:
        print(f"STDERR: {stderr}")
    
    success = exit_code == 0
    print(f"Unit tests: {'PASSED' if success else 'FAILED'}")
    return success


def run_integration_tests(verbose: bool = False) -> bool:
    """Run integration tests"""
    print("Running integration tests...")
    
    cmd = ["python", "-m", "pytest", "tests/wan_model_testing_suite/integration/", "-v" if verbose else "-q"]
    
    exit_code, stdout, stderr = run_command(cmd)
    
    if verbose:
        print(stdout)
    if stderr:
        print(f"STDERR: {stderr}")
    
    success = exit_code == 0
    print(f"Integration tests: {'PASSED' if success else 'FAILED'}")
    return success


def run_performance_tests(verbose: bool = False, iterations: int = 10) -> bool:
    """Run performance tests"""
    print(f"Running performance tests ({iterations} iterations)...")
    
    # Set environment variable for benchmark iterations
    env = os.environ.copy()
    env["WAN_BENCHMARK_ITERATIONS"] = str(iterations)
    
    cmd = ["python", "-m", "pytest", "tests/wan_model_testing_suite/performance/", "-v" if verbose else "-q", "-m", "performance"]
    
    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=7200  # 2 hour timeout for performance tests
        )
        exit_code, stdout, stderr = result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        exit_code, stdout, stderr = -1, "", "Performance tests timed out"
    
    if verbose:
        print(stdout)
    if stderr:
        print(f"STDERR: {stderr}")
    
    success = exit_code == 0
    print(f"Performance tests: {'PASSED' if success else 'FAILED'}")
    return success


def run_hardware_tests(verbose: bool = False, hardware_type: str = "all") -> bool:
    """Run hardware compatibility tests"""
    print(f"Running hardware tests ({hardware_type})...")
    
    # Set environment variable to enable hardware tests
    env = os.environ.copy()
    env["WAN_TEST_HARDWARE"] = "true"
    
    cmd = ["python", "-m", "pytest", "tests/wan_model_testing_suite/hardware/", "-v" if verbose else "-q", "-m", "hardware"]
    
    # Filter by hardware type if specified
    if hardware_type != "all":
        if hardware_type == "rtx4080":
            cmd.extend(["-k", "rtx4080"])
        elif hardware_type == "threadripper":
            cmd.extend(["-k", "threadripper"])
    
    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        exit_code, stdout, stderr = result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        exit_code, stdout, stderr = -1, "", "Hardware tests timed out"
    
    if verbose:
        print(stdout)
    if stderr:
        print(f"STDERR: {stderr}")
    
    success = exit_code == 0
    print(f"Hardware tests: {'PASSED' if success else 'FAILED'}")
    return success


def run_specific_model_tests(model_type: str, verbose: bool = False) -> bool:
    """Run tests for a specific model type"""
    print(f"Running tests for {model_type} model...")
    
    # Map model types to test files
    model_test_files = {
        "t2v": "tests/wan_model_testing_suite/unit/test_wan_t2v_a14b.py",
        "i2v": "tests/wan_model_testing_suite/unit/test_wan_i2v_a14b.py",
        "ti2v": "tests/wan_model_testing_suite/unit/test_wan_ti2v_5b.py"
    }
    
    if model_type not in model_test_files:
        print(f"Unknown model type: {model_type}")
        return False
    
    cmd = ["python", "-m", "pytest", model_test_files[model_type], "-v" if verbose else "-q"]
    
    exit_code, stdout, stderr = run_command(cmd)
    
    if verbose:
        print(stdout)
    if stderr:
        print(f"STDERR: {stderr}")
    
    success = exit_code == 0
    print(f"{model_type.upper()} model tests: {'PASSED' if success else 'FAILED'}")
    return success


def generate_test_report(results: Dict[str, bool], output_file: Optional[str] = None) -> None:
    """Generate a test report"""
    report = {
        "timestamp": time.time(),
        "test_results": results,
        "summary": {
            "total_test_suites": len(results),
            "passed_suites": sum(1 for passed in results.values() if passed),
            "failed_suites": sum(1 for passed in results.values() if not passed),
            "success_rate": sum(1 for passed in results.values() if passed) / len(results) * 100 if results else 0
        }
    }
    
    if output_file:
        import json
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Test report saved to: {output_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUITE SUMMARY")
    print("="*60)
    
    for suite_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"{suite_name:30} {status}")
    
    print("-"*60)
    print(f"Total test suites: {report['summary']['total_test_suites']}")
    print(f"Passed: {report['summary']['passed_suites']}")
    print(f"Failed: {report['summary']['failed_suites']}")
    print(f"Success rate: {report['summary']['success_rate']:.1f}%")
    print("="*60)


def main():
    """Main test runner"""
    parser = argparse.ArgumentParser(description="WAN Model Testing Suite Runner")
    
    # Test suite selection
    parser.add_argument("--unit", action="store_true", help="Run unit tests")
    parser.add_argument("--integration", action="store_true", help="Run integration tests")
    parser.add_argument("--performance", action="store_true", help="Run performance tests")
    parser.add_argument("--hardware", action="store_true", help="Run hardware tests")
    parser.add_argument("--all", action="store_true", help="Run all test suites")
    
    # Specific model tests
    parser.add_argument("--model", choices=["t2v", "i2v", "ti2v"], help="Run tests for specific model")
    
    # Hardware-specific tests
    parser.add_argument("--hardware-type", choices=["all", "rtx4080", "threadripper"], 
                       default="all", help="Hardware type for hardware tests")
    
    # Test configuration
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--iterations", type=int, default=10, help="Benchmark iterations")
    parser.add_argument("--report", help="Output file for test report")
    
    # Quick test options
    parser.add_argument("--quick", action="store_true", help="Run quick tests only (fewer iterations)")
    parser.add_argument("--smoke", action="store_true", help="Run smoke tests only")
    
    args = parser.parse_args()
    
    # Adjust iterations for quick tests
    if args.quick:
        args.iterations = 3
    
    # If no specific tests selected, run all
    if not any([args.unit, args.integration, args.performance, args.hardware, args.model, args.smoke]):
        args.all = True
    
    results = {}
    
    try:
        # Run smoke tests (basic functionality)
        if args.smoke:
            print("Running smoke tests...")
            results["smoke_unit"] = run_unit_tests(args.verbose, False)
            results["smoke_integration"] = run_integration_tests(args.verbose)
        
        # Run specific model tests
        elif args.model:
            results[f"{args.model}_model"] = run_specific_model_tests(args.model, args.verbose)
        
        # Run selected test suites
        else:
            if args.unit or args.all:
                results["unit_tests"] = run_unit_tests(args.verbose, args.coverage)
            
            if args.integration or args.all:
                results["integration_tests"] = run_integration_tests(args.verbose)
            
            if args.performance or args.all:
                results["performance_tests"] = run_performance_tests(args.verbose, args.iterations)
            
            if args.hardware or args.all:
                results["hardware_tests"] = run_hardware_tests(args.verbose, args.hardware_type)
    
    except KeyboardInterrupt:
        print("\nTest execution interrupted by user")
        return 1
    
    except Exception as e:
        print(f"Error during test execution: {e}")
        return 1
    
    # Generate report
    generate_test_report(results, args.report)
    
    # Return exit code based on results
    if all(results.values()):
        print("\nAll tests passed! ✅")
        return 0
    else:
        print("\nSome tests failed! ❌")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
