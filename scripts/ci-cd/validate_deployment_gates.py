#!/usr/bin/env python3
"""
Validate Deployment Gates
Quick validation script to ensure all deployment gate components are working
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and report results"""
    print(f"\n🔍 {description}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ SUCCESS")
            if result.stdout:
                print(f"Output: {result.stdout.strip()[:200]}...")
        else:
            print("❌ FAILED")
            print(f"Error: {result.stderr.strip()[:200]}...")
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("⏰ TIMEOUT")
        return False
    except Exception as e:
        print(f"💥 EXCEPTION: {e}")
        return False


def main():
    print("🚀 Validating Deployment Gates")
    print("=" * 50)
    
    tests = [
        {
            "cmd": [sys.executable, "tools/health-checker/simple_health_check.py", 
                   "--output-file=test-health.json"],
            "description": "Testing Simple Health Check"
        },
        {
            "cmd": [sys.executable, "tools/deployment-gates/simple_test_runner.py"],
            "description": "Testing Simple Test Runner"
        },
        {
            "cmd": [sys.executable, "tools/deployment-gates/deployment_gate_status.py", 
                   "--output-file=test-deployment.json", "--create-badge"],
            "description": "Testing Deployment Gate Status"
        }
    ]
    
    results = []
    
    for test in tests:
        success = run_command(test["cmd"], test["description"])
        results.append(success)
    
    print("\n" + "=" * 50)
    print("📊 VALIDATION SUMMARY")
    print("=" * 50)
    
    for i, (test, success) in enumerate(zip(tests, results)):
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{i+1}. {test['description']}: {status}")
    
    total_tests = len(results)
    passed_tests = sum(results)
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL DEPLOYMENT GATE COMPONENTS ARE WORKING!")
        return 0
    else:
        print("⚠️  Some deployment gate components need attention")
        return 1


if __name__ == "__main__":
    sys.exit(main())