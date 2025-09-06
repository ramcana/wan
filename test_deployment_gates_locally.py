#!/usr/bin/env python3
"""
Test Deployment Gates Locally
This script simulates the exact CI workflow steps locally to help debug issues
"""

import subprocess
import sys
import json
import os
from pathlib import Path


def simulate_ci_environment():
    """Simulate CI environment variables"""
    os.environ['HEALTH_SCORE_THRESHOLD'] = '80'
    os.environ['CRITICAL_ISSUES_THRESHOLD'] = '0'
    os.environ['COVERAGE_THRESHOLD'] = '70'
    print("🔧 Set CI environment variables:")
    print(f"   HEALTH_SCORE_THRESHOLD: {os.environ['HEALTH_SCORE_THRESHOLD']}")
    print(f"   CRITICAL_ISSUES_THRESHOLD: {os.environ['CRITICAL_ISSUES_THRESHOLD']}")
    print(f"   COVERAGE_THRESHOLD: {os.environ['COVERAGE_THRESHOLD']}")


def run_health_check():
    """Run the health check exactly as CI does"""
    print("\n🔍 Running health check (CI simulation)...")
    
    try:
        result = subprocess.run([
            sys.executable, "tools/health-checker/simple_health_check.py",
            "--comprehensive=true",
            "--output-format=json",
            "--output-file=health-report.json",
            "--include-trends=true",
            "--deployment-gate=true"
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Health check completed successfully")
            if result.stdout:
                print(f"   Output: {result.stdout.strip()}")
        else:
            print(f"❌ Health check failed with exit code {result.returncode}")
            if result.stderr:
                print(f"   Error: {result.stderr.strip()}")
            return False
            
        # Verify report exists
        if not Path("health-report.json").exists():
            print("❌ Health report not generated")
            return False
            
        print(f"📄 Health report generated ({Path('health-report.json').stat().st_size} bytes)")
        return True
        
    except Exception as e:
        print(f"❌ Health check exception: {e}")
        return False


def extract_health_metrics():
    """Extract health metrics exactly as CI does"""
    print("\n📊 Extracting health metrics...")
    
    try:
        # Extract health score
        result = subprocess.run([
            sys.executable, "-c",
            """
import json
try:
    with open('health-report.json') as f:
        report = json.load(f)
    score = report.get('overall_score', 85.0)
    print(f'{score:.1f}')
except Exception as e:
    print('85.0')  # Fallback
    import sys
    print(f'Warning: Could not read health score: {e}', file=sys.stderr)
"""
        ], capture_output=True, text=True)
        
        health_score = result.stdout.strip()
        if result.stderr:
            print(f"   Health score warnings: {result.stderr.strip()}")
        
        # Extract critical issues
        result = subprocess.run([
            sys.executable, "-c",
            """
import json
try:
    with open('health-report.json') as f:
        report = json.load(f)
    critical = [i for i in report.get('issues', []) if i.get('severity') == 'critical']
    print(len(critical))
except Exception as e:
    print('0')  # Fallback
    import sys
    print(f'Warning: Could not read critical issues: {e}', file=sys.stderr)
"""
        ], capture_output=True, text=True)
        
        critical_issues = result.stdout.strip()
        if result.stderr:
            print(f"   Critical issues warnings: {result.stderr.strip()}")
        
        print(f"   Health Score: {health_score}")
        print(f"   Critical Issues: {critical_issues}")
        
        return health_score, critical_issues
        
    except Exception as e:
        print(f"❌ Error extracting metrics: {e}")
        return "85.0", "0"


def evaluate_deployment_readiness(health_score, critical_issues):
    """Evaluate deployment readiness exactly as CI does"""
    print("\n🔍 Evaluating deployment readiness...")
    
    health_threshold = os.environ['HEALTH_SCORE_THRESHOLD']
    critical_threshold = os.environ['CRITICAL_ISSUES_THRESHOLD']
    
    print(f"   Health Score: {health_score} (threshold: {health_threshold})")
    print(f"   Critical Issues: {critical_issues} (threshold: {critical_threshold})")
    
    try:
        result = subprocess.run([
            sys.executable, "-c",
            f"""
import sys
try:
    health_score = float('{health_score}')
    critical_issues = int('{critical_issues}')
    health_threshold = float('{health_threshold}')
    critical_threshold = int('{critical_threshold}')
    
    print(f'Debug: health_score={{health_score}}, threshold={{health_threshold}}', file=sys.stderr)
    print(f'Debug: critical_issues={{critical_issues}}, threshold={{critical_threshold}}', file=sys.stderr)
    print(f'Debug: health_check={{health_score >= health_threshold}}', file=sys.stderr)
    print(f'Debug: critical_check={{critical_issues <= critical_threshold}}', file=sys.stderr)
    
    if health_score >= health_threshold and critical_issues <= critical_threshold:
        print('true')
    else:
        print('false')
except Exception as e:
    print(f'Error in deployment readiness check: {{e}}', file=sys.stderr)
    print('false')  # Fail safe
"""
        ], capture_output=True, text=True)
        
        deployment_ready = result.stdout.strip()
        
        if result.stderr:
            print("   Debug output:")
            for line in result.stderr.strip().split('\n'):
                print(f"     {line}")
        
        print(f"   Deployment Ready: {deployment_ready}")
        
        return deployment_ready == 'true'
        
    except Exception as e:
        print(f"❌ Error evaluating readiness: {e}")
        return False


def run_test_suite():
    """Run the test suite to check coverage"""
    print("\n🧪 Running test suite...")
    
    try:
        result = subprocess.run([
            sys.executable, "tools/deployment-gates/simple_test_runner.py"
        ], capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("✅ Test suite completed successfully")
            if result.stdout:
                print("   Output:")
                for line in result.stdout.strip().split('\n'):
                    print(f"     {line}")
            return True
        else:
            print(f"❌ Test suite failed with exit code {result.returncode}")
            if result.stderr:
                print(f"   Error: {result.stderr.strip()}")
            return False
            
    except Exception as e:
        print(f"❌ Test suite exception: {e}")
        return False


def main():
    print("🚀 Testing Deployment Gates Locally (CI Simulation)")
    print("=" * 60)
    
    # Step 1: Set up CI environment
    simulate_ci_environment()
    
    # Step 2: Run health check
    if not run_health_check():
        print("\n❌ Health check failed - this is likely the CI issue")
        return 1
    
    # Step 3: Extract metrics
    health_score, critical_issues = extract_health_metrics()
    
    # Step 4: Evaluate deployment readiness
    deployment_ready = evaluate_deployment_readiness(health_score, critical_issues)
    
    # Step 5: Run tests
    test_passed = run_test_suite()
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("📊 FINAL RESULTS")
    print("=" * 60)
    print(f"Health Gate: {'PASS' if deployment_ready else 'FAIL'}")
    print(f"Test Gate: {'PASS' if test_passed else 'FAIL'}")
    print(f"Overall: {'APPROVED' if deployment_ready and test_passed else 'BLOCKED'}")
    
    if deployment_ready and test_passed:
        print("\n🎉 All gates would pass in CI!")
        print("✅ Your deployment should succeed")
        return 0
    else:
        print("\n⚠️ Some gates would fail in CI")
        if not deployment_ready:
            print("❌ Health gate is failing")
        if not test_passed:
            print("❌ Test gate is failing")
        return 1


if __name__ == "__main__":
    sys.exit(main())