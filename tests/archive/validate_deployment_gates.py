#!/usr/bin/env python3
"""
Validate deployment gates workflow locally
This script tests the deployment gates components to ensure they work properly
"""

import subprocess
import sys
import json
from pathlib import Path


def test_health_checker():
    """Test the health checker script"""
    print("Testing health checker...")
    try:
        result = subprocess.run([
            sys.executable, "tools/health-checker/simple_health_check.py",
            "--comprehensive=true",
            "--output-format=json",
            "--output-file=test-health-report.json",
            "--include-trends=true",
            "--deployment-gate=true"
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Health checker completed successfully")
            
            # Check if report was generated
            if Path("test-health-report.json").exists():
                with open("test-health-report.json") as f:
                    report = json.load(f)
                print(f"   Health score: {report.get('overall_score', 'N/A')}")
                print(f"   Critical issues: {report.get('critical_issues', 'N/A')}")
                return True
            else:
                print("❌ Health report not generated")
                return False
        else:
            print(f"❌ Health checker failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Health checker error: {e}")
        return False


def test_test_runner():
    """Test the test runner script"""
    print("Testing test runner...")
    try:
        result = subprocess.run([
            sys.executable, "tools/deployment-gates/simple_test_runner.py"
        ], capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("✅ Test runner completed successfully")
            
            # Check if coverage report was generated
            if Path("coverage.xml").exists():
                print("   Coverage report generated")
                return True
            else:
                print("   No coverage report, but tests passed")
                return True
        else:
            print(f"❌ Test runner failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Test runner error: {e}")
        return False


def check_workflow_dependencies():
    """Check if workflow dependencies are available"""
    print("Checking workflow dependencies...")
    
    # Map package names to import names
    dependencies = {
        "pyyaml": "yaml",
        "jsonschema": "jsonschema", 
        "requests": "requests",
        "beautifulsoup4": "bs4",
        "pytest": "pytest",
        "pytest-cov": "pytest_cov",
        "pytest-asyncio": "pytest_asyncio",
        "pytest-mock": "pytest_mock"
    }
    
    missing = []
    for package, import_name in dependencies.items():
        try:
            result = subprocess.run([
                sys.executable, "-c", f"import {import_name}"
            ], capture_output=True, text=True)
            if result.returncode != 0:
                missing.append(package)
        except:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing dependencies: {', '.join(missing)}")
        print("   Install with: pip install " + " ".join(missing))
        return False
    else:
        print("✅ All workflow dependencies available")
        return True


def main():
    print("Validating deployment gates workflow components...\n")
    
    results = []
    
    # Test components
    results.append(("Dependencies", check_workflow_dependencies()))
    results.append(("Health Checker", test_health_checker()))
    results.append(("Test Runner", test_test_runner()))
    
    # Summary
    print("\n" + "="*50)
    print("VALIDATION SUMMARY")
    print("="*50)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:20} {status}")
        if not passed:
            all_passed = False
    
    print("="*50)
    if all_passed:
        print("🎉 All deployment gate components are working!")
        print("The workflow should run successfully.")
    else:
        print("⚠️  Some components failed validation.")
        print("Fix the issues above before running the workflow.")
    
    # Cleanup test files
    for test_file in ["test-health-report.json", "reports/coverage/coverage.xml", "reports/tests/test-results.xml"]:
        if Path(test_file).exists():
            Path(test_file).unlink()
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())