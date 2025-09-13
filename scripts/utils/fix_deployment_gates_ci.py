#!/usr/bin/env python3
"""
Fix Deployment Gates CI Issues
This script ensures deployment gates work correctly in CI environments
"""

import subprocess
import sys
import json
from pathlib import Path


def update_health_status():
    """Update health status to current values"""
    print("🔄 Updating health status...")
    
    try:
        # Run health check to get current status
        result = subprocess.run([
            sys.executable, "tools/health-checker/simple_health_check.py",
            "--output-file=validation-health.json"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Health check updated successfully")
            
            # Read the updated health data
            with open("validation-health.json", 'r') as f:
                health_data = json.load(f)
            
            print(f"   Health Score: {health_data.get('overall_score', 'N/A')}")
            print(f"   Critical Issues: {health_data.get('critical_issues', 'N/A')}")
            print(f"   Status: {health_data.get('status', 'N/A')}")
            
            return True
        else:
            print(f"❌ Health check failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error updating health status: {e}")
        return False


def update_deployment_status():
    """Update deployment status based on current health"""
    print("🔄 Updating deployment status...")
    
    try:
        # Run deployment gate status check
        result = subprocess.run([
            sys.executable, "tools/deployment-gates/deployment_gate_status.py",
            "--output-file=deployment-status.json",
            "--create-badge"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Deployment status updated successfully")
            
            # Also update validation deployment status
            subprocess.run([
                sys.executable, "tools/deployment-gates/deployment_gate_status.py",
                "--output-file=validation-deployment.json"
            ], capture_output=True, text=True)
            
            return True
        else:
            print(f"❌ Deployment status update failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error updating deployment status: {e}")
        return False


def verify_ci_compatibility():
    """Verify that the deployment gates work in CI-like conditions"""
    print("🔍 Verifying CI compatibility...")
    
    # Check if all required files exist
    required_files = [
        "tools/health-checker/simple_health_check.py",
        "tools/deployment-gates/deployment_gate_status.py",
        "config.json",
        "requirements.txt"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        return False
    
    # Test health check with CI-like parameters
    try:
        result = subprocess.run([
            sys.executable, "tools/health-checker/simple_health_check.py",
            "--comprehensive=true",
            "--output-format=json", 
            "--output-file=ci-test-health.json",
            "--include-trends=true",
            "--deployment-gate=true"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ CI-style health check works")
            
            # Clean up test file
            if Path("ci-test-health.json").exists():
                Path("ci-test-health.json").unlink()
            
            return True
        else:
            print(f"❌ CI-style health check failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ CI compatibility test failed: {e}")
        return False


def main():
    print("🚀 Fixing Deployment Gates CI Issues")
    print("=" * 50)
    
    success_count = 0
    total_steps = 3
    
    # Step 1: Update health status
    if update_health_status():
        success_count += 1
    
    # Step 2: Update deployment status  
    if update_deployment_status():
        success_count += 1
    
    # Step 3: Verify CI compatibility
    if verify_ci_compatibility():
        success_count += 1
    
    print("\n" + "=" * 50)
    print("📊 RESULTS")
    print("=" * 50)
    print(f"Steps completed: {success_count}/{total_steps}")
    
    if success_count == total_steps:
        print("\n🎉 All deployment gate issues fixed!")
        print("✅ Your CI pipeline should now pass the health gate")
        print("\nNext steps:")
        print("1. Commit and push these changes")
        print("2. The CI pipeline should now show HEALTH_READY='true'")
        return 0
    else:
        print("\n⚠️ Some issues remain")
        print("Please check the errors above and fix them manually")
        return 1


if __name__ == "__main__":
    sys.exit(main())
