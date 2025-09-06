#!/usr/bin/env python3
"""
Test the monitoring service with a single run.
"""

import json
from datetime import datetime
from pathlib import Path

def run_single_health_check():
    """Run a single health check to test the monitoring system."""
    
    # Load baseline
    baseline_file = Path("baseline_metrics.json")
    if not baseline_file.exists():
        print("ERROR: No baseline found. Run baseline establishment first.")
        return False
    
    with open(baseline_file, 'r') as f:
        baseline = json.load(f)
    
    # Simple health check
    current_score = baseline["baseline_metrics"]["overall_score"]["current"]
    
    print(f"Health Check - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Current Health Score: {current_score:.1f}")
    
    # Check against thresholds
    if current_score < 50:
        print("   CRITICAL: Health score is critically low")
        status = "critical"
    elif current_score < 70:
        print("   WARNING: Health score is below target")
        status = "warning"
    else:
        print("   HEALTHY: Health score is acceptable")
        status = "healthy"
    
    # Log to file
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "health_score": current_score,
        "status": status
    }
    
    log_file = Path("health_monitoring.log")
    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry) + "\n")
    
    print(f"   Log entry written to: {log_file}")
    return True

if __name__ == "__main__":
    print("Testing health monitoring system...")
    success = run_single_health_check()
    if success:
        print("✅ Monitoring test completed successfully")
    else:
        print("❌ Monitoring test failed")