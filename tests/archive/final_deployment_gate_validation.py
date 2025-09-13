#!/usr/bin/env python3
"""
Final Deployment Gate Validation
Validates that all deployment gate components are working after autofix
"""

import subprocess
import sys
from pathlib import Path


def test_component(name, command, expected_exit_code=0):
    """Test a deployment gate component"""
    print(f"\n🔍 Testing {name}")
    print(f"Command: {' '.join(command)}")
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        
        if result.returncode == expected_exit_code:
            print("✅ SUCCESS")
            return True
        else:
            print(f"❌ FAILED (exit code: {result.returncode})")
            if result.stderr:
                print(f"Error: {result.stderr.strip()[:200]}")
            return False
            
    except Exception as e:
        print(f"💥 EXCEPTION: {e}")
        return False


def main():
    print("🚀 Final Deployment Gate Validation")
    print("=" * 60)
    
    # Test components in their new locations
    tests = [
        {
            "name": "Health Check (Original Location)",
            "command": [sys.executable, "tools/health-checker/simple_health_check.py", 
                       "--output-file=validation-health.json"]
        },
        {
            "name": "Test Runner (New Location)", 
            "command": [sys.executable, "tools/deployment-gates/simple_test_runner.py"]
        },
        {
            "name": "Deployment Status (New Location)",
            "command": [sys.executable, "tools/deployment-gates/deployment_gate_status.py",
                       "--output-file=validation-deployment.json"]
        }
    ]
    
    results = []
    for test in tests:
        success = test_component(test["name"], test["command"])
        results.append(success)
    
    print("\n" + "=" * 60)
    print("📊 FINAL VALIDATION RESULTS")
    print("=" * 60)
    
    for i, (test, success) in enumerate(zip(tests, results)):
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{i+1}. {test['name']}: {status}")
    
    total_tests = len(results)
    passed_tests = sum(results)
    
    print(f"\nOverall: {passed_tests}/{total_tests} components working")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL DEPLOYMENT GATE COMPONENTS ARE WORKING!")
        print("✅ The deployment gates are ready for CI/CD use")
        return 0
    else:
        print("\n⚠️ Some components need attention")
        return 1


if __name__ == "__main__":
    sys.exit(main())
