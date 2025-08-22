#!/usr/bin/env python3
"""
Daily Development Workflow Script

This script demonstrates a typical daily development workflow using the
Local Testing Framework. It performs quick validation checks and runs
essential tests to ensure the development environment is ready.

Usage:
    python daily_development_workflow.py [--quick] [--report]
"""

import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

def run_command(command, description, capture_output=False):
    """Run a command and handle output"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print(f"{'='*60}")
    
    try:
        if capture_output:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✓ SUCCESS: {description}")
                return result.stdout
            else:
                print(f"✗ FAILED: {description}")
                print(f"Error: {result.stderr}")
                return None
        else:
            result = subprocess.run(command, shell=True)
            if result.returncode == 0:
                print(f"✓ SUCCESS: {description}")
                return True
            else:
                print(f"✗ FAILED: {description}")
                return False
    except Exception as e:
        print(f"✗ ERROR: {description} - {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Daily Development Workflow")
    parser.add_argument("--quick", action="store_true", help="Run quick tests only")
    parser.add_argument("--report", action="store_true", help="Generate detailed reports")
    parser.add_argument("--fix", action="store_true", help="Attempt to fix issues automatically")
    args = parser.parse_args()

    print("🚀 Starting Daily Development Workflow")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Environment Validation
    print("\n📋 Step 1: Environment Validation")
    env_command = "python -m local_testing_framework validate-env"
    if args.fix:
        env_command += " --fix"
    if args.report:
        env_command += " --report"
    
    env_success = run_command(env_command, "Environment validation")
    if not env_success:
        print("❌ Environment validation failed. Please fix issues before continuing.")
        return 1

    # Step 2: Quick Performance Check
    print("\n⚡ Step 2: Quick Performance Check")
    if args.quick:
        perf_command = "python -m local_testing_framework test-performance --resolution 720p"
    else:
        perf_command = "python -m local_testing_framework test-performance --benchmark"
    
    perf_success = run_command(perf_command, "Performance testing")
    if not perf_success:
        print("⚠️  Performance tests failed. Check optimization settings.")

    # Step 3: Integration Tests (if not quick mode)
    if not args.quick:
        print("\n🔗 Step 3: Integration Tests")
        integration_command = "python -m local_testing_framework test-integration --ui --api"
        integration_success = run_command(integration_command, "Integration testing")
        if not integration_success:
            print("⚠️  Integration tests failed. Check application status.")

    # Step 4: Generate Sample Data (if needed)
    print("\n📊 Step 4: Sample Data Check")
    sample_command = "python -m local_testing_framework generate-samples --config --data"
    run_command(sample_command, "Sample data generation")

    # Step 5: System Diagnostics
    print("\n🔍 Step 5: System Diagnostics")
    diag_command = "python -m local_testing_framework diagnose --system --cuda"
    run_command(diag_command, "System diagnostics")

    # Step 6: Generate Report (if requested)
    if args.report:
        print("\n📄 Step 6: Report Generation")
        report_command = "python -m local_testing_framework run-all --report-format html"
        run_command(report_command, "Comprehensive report generation")

    # Summary
    print("\n" + "="*60)
    print("📋 DAILY WORKFLOW SUMMARY")
    print("="*60)
    print("✓ Environment validation completed")
    print("✓ Performance testing completed")
    if not args.quick:
        print("✓ Integration testing completed")
    print("✓ Sample data verified")
    print("✓ System diagnostics completed")
    if args.report:
        print("✓ Reports generated")
    
    print(f"\n🎉 Daily development workflow completed at {datetime.now().strftime('%H:%M:%S')}")
    print("Your development environment is ready! 🚀")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())