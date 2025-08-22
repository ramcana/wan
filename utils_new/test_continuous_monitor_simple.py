#!/usr/bin/env python3
"""
Simple test script for continuous monitoring system
"""

import time
import sys
import os

# Add the local_testing_framework to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from local_testing_framework.continuous_monitor import ContinuousMonitor, Alert, ProgressInfo


def alert_handler(alert: Alert):
    """Handle alerts from monitoring system"""
    print(f"🚨 ALERT [{alert.level.value.upper()}]: {alert.message}")
    print(f"   Metric: {alert.metric_name} = {alert.current_value:.1f} (threshold: {alert.threshold_value:.1f})")


def progress_handler(progress: ProgressInfo):
    """Handle progress updates"""
    eta_str = f" (ETA: {progress.eta_seconds:.1f}s)" if progress.eta_seconds else ""
    print(f"📊 Progress: {progress.percentage:.1f}% ({progress.current_step}/{progress.total_steps}){eta_str}")


def main():
    """Test continuous monitoring system"""
    print("🔍 Testing Continuous Monitoring System")
    print("=" * 50)
    
    # Initialize monitor
    monitor = ContinuousMonitor()
    monitor.refresh_interval = 2  # 2-second intervals for demo
    monitor.stability_check_interval = 10  # Check stability every 10 seconds
    
    # Register callbacks
    monitor.add_alert_callback(alert_handler)
    monitor.add_progress_callback(progress_handler)
    
    try:
        # Start monitoring session
        print("🚀 Starting monitoring session...")
        session = monitor.start_monitoring("demo_session")
        print(f"   Session ID: {session.session_id}")
        print(f"   Started at: {session.start_time}")
        
        # Simulate some work with progress updates
        print("\n📈 Simulating work with progress tracking...")
        total_steps = 10
        for step in range(1, total_steps + 1):
            monitor.update_progress(step, total_steps, session.start_time)
            time.sleep(1)
        
        # Let monitoring run for a bit
        print("\n⏱️  Monitoring system resources for 15 seconds...")
        time.sleep(15)
        
        # Force a diagnostic snapshot
        print("\n📸 Capturing diagnostic snapshot...")
        snapshot = monitor.force_diagnostic_snapshot()
        print(f"   Snapshot captured at: {snapshot.timestamp}")
        print(f"   GPU Memory State: {snapshot.gpu_memory_state}")
        print(f"   Top processes: {len(snapshot.system_processes)}")
        print(f"   System logs: {len(snapshot.system_logs)}")
        
        # Trigger recovery procedures
        print("\n🔧 Triggering recovery procedures...")
        recovery_actions = monitor.trigger_recovery_procedures()
        for action in recovery_actions:
            status = "✅" if action.success else "❌"
            print(f"   {status} {action.action_name}: {action.message}")
        
        # Generate monitoring report
        print("\n📋 Generating monitoring report...")
        report = monitor.generate_monitoring_report()
        print(f"   Timeline entries: {len(report['timeline'])}")
        print(f"   Threshold violations: {len(report['threshold_violations'])}")
        print(f"   Stability events: {len(report['stability_events'])}")
        print(f"   Recovery actions: {sum(r['attempts'] for r in report['recovery_summary'].values())}")
        
        # Get session summary
        print("\n📊 Session summary:")
        summary = monitor.get_session_summary()
        if summary:
            print(f"   Metrics collected: {summary['metrics_count']}")
            print(f"   Alerts generated: {summary['alerts_count']}")
            if 'latest_metrics' in summary:
                latest = summary['latest_metrics']
                print(f"   Latest CPU: {latest['cpu_percent']:.1f}%")
                print(f"   Latest Memory: {latest['memory_percent']:.1f}%")
                print(f"   Latest VRAM: {latest['vram_percent']:.1f}%")
            if 'averages' in summary:
                avg = summary['averages']
                print(f"   Average CPU: {avg['cpu_percent']:.1f}%")
                print(f"   Average Memory: {avg['memory_percent']:.1f}%")
                print(f"   Average VRAM: {avg['vram_percent']:.1f}%")
        
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        # Stop monitoring
        print("\n🛑 Stopping monitoring session...")
        final_session = monitor.stop_monitoring_session()
        if final_session:
            duration = (final_session.end_time - final_session.start_time).total_seconds()
            print(f"   Session duration: {duration:.1f} seconds")
            print(f"   Final metrics count: {len(final_session.metrics_history)}")
            print(f"   Final alerts count: {len(final_session.alerts_history)}")
            print(f"   Diagnostic snapshots: {len(final_session.diagnostic_snapshots)}")
            print(f"   Recovery actions: {len(final_session.recovery_actions)}")
        
        # Cleanup
        monitor.cleanup_resources()
        print("✅ Monitoring system cleaned up")


if __name__ == "__main__":
    main()