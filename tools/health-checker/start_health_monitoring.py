#!/usr/bin/env python3
"""
Simple health monitoring service.
"""

import json
import time
from datetime import datetime
from pathlib import Path

def run_health_check():
    """Run a simple health check."""
    
    # Load baseline
    baseline_file = Path("baseline_metrics.json")
    if not baseline_file.exists():
        print("ERROR: No baseline found. Run baseline establishment first.")
        return
    
    with open(baseline_file, 'r') as f:
        baseline = json.load(f)
    
    # Simple health check
    current_score = baseline["baseline_metrics"]["overall_score"]["current"]
    
    print(f"Health Check - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Current Health Score: {current_score:.1f}")
    
    # Check against thresholds
    if current_score < 50:
        print("   CRITICAL: Health score is critically low")
    elif current_score < 70:
        print("   WARNING: Health score is below target")
    else:
        print("   HEALTHY: Health score is acceptable")
    
    # Log to file
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "health_score": current_score,
        "status": "critical" if current_score < 50 else "warning" if current_score < 70 else "healthy"
    }
    
    log_file = Path("health_monitoring.log")
    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry) + "\n")

def main():
    """Main monitoring loop."""
    
    print("Starting health monitoring service...")
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            run_health_check()
            time.sleep(3600)  # Check every hour
    except KeyboardInterrupt:
        print("\nMonitoring stopped")

if __name__ == "__main__":
    main()
