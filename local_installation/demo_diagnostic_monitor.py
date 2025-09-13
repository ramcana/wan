#!/usr/bin/env python3
"""
Demo script for the comprehensive diagnostic monitoring system
Demonstrates real-time monitoring, alerting, and predictive analysis capabilities
"""

import time
import json
import threading
from datetime import datetime, timedelta
from scripts.diagnostic_monitor import DiagnosticMonitor, Alert, HealthReport


def alert_handler(alert: Alert):
    """Handle alerts from the diagnostic monitor"""
    print(f"\n🚨 ALERT [{alert.level.value.upper()}] - {alert.component}")
    print(f"   Message: {alert.message}")
    print(f"   Current: {alert.current_value:.1f}, Threshold: {alert.threshold_value:.1f}")
    print(f"   Time: {alert.timestamp.strftime('%H:%M:%S')}")


def health_report_handler(report: HealthReport):
    """Handle health reports from the diagnostic monitor"""
    print(f"\n📊 Health Report - Overall Status: {report.overall_health.value.upper()}")
    print(f"   CPU: {report.resource_metrics.cpu_percent:.1f}%")
    print(f"   Memory: {report.resource_metrics.memory_percent:.1f}%")
    print(f"   VRAM: {report.resource_metrics.vram_percent:.1f}%")
    print(f"   Disk: {report.resource_metrics.disk_usage_percent:.1f}%")
    
    if report.active_alerts:
        print(f"   Active Alerts: {len(report.active_alerts)}")
    
    if report.potential_issues:
        print(f"   Potential Issues: {len(report.potential_issues)}")
        for issue in report.potential_issues[:2]:  # Show first 2
            print(f"     - {issue.issue_type}: {issue.probability:.1%} probability")


def simulate_system_load():
    """Simulate system load to trigger monitoring alerts"""
    print("\n🔄 Simulating system load to demonstrate monitoring...")
    
    # Simulate CPU-intensive task using threading instead of multiprocessing for Windows compatibility
    import threading
    import psutil
    
    def cpu_intensive_task():
        """CPU intensive task for simulation"""
        end_time = time.time() + 8  # Run for 8 seconds
        while time.time() < end_time:
            # Simple CPU-intensive calculation
            sum(i * i for i in range(10000))
    
    # Start multiple CPU-intensive threads
    threads = []
    thread_count = min(8, threading.active_count() * 2)  # Reasonable number of threads
    
    print(f"   Starting {thread_count} CPU-intensive threads...")
    for _ in range(thread_count):
        thread = threading.Thread(target=cpu_intensive_task)
        thread.daemon = True
        thread.start()
        threads.append(thread)
    
    # Let them run for a bit
    time.sleep(5)
    
    # Wait for threads to complete
    for thread in threads:
        thread.join(timeout=1)
    
    print("   System load simulation completed")


def demonstrate_error_pattern_detection(monitor: DiagnosticMonitor):
    """Demonstrate error pattern detection"""
    print("\n🔍 Demonstrating error pattern detection...")
    
    # Simulate recurring errors
    components = ["model_downloader", "dependency_manager", "config_validator"]
    
    for i in range(15):
        component = components[i % len(components)]
        error_msg = f"Simulated error {i+1} in {component}"
        monitor.record_error(component, error_msg)
        
        if i % 3 == 0:  # Add some delay every 3 errors
            time.sleep(0.5)
    
    print("   Recorded 15 simulated errors across components")
    
    # Wait a moment for pattern detection
    time.sleep(2)
    
    # Generate health report to see detected patterns
    report = monitor.generate_health_report()
    
    if report.error_patterns:
        print(f"   Detected {len(report.error_patterns)} error patterns:")
        for pattern in report.error_patterns:
            print(f"     - {pattern.pattern_type}: {pattern.frequency} occurrences, trend: {pattern.severity_trend}")
    else:
        print("   No error patterns detected yet (may need more time)")


def demonstrate_predictive_analysis(monitor: DiagnosticMonitor):
    """Demonstrate predictive failure analysis"""
    print("\n🔮 Demonstrating predictive failure analysis...")
    
    # Let the monitor collect some baseline data
    print("   Collecting baseline metrics...")
    time.sleep(10)
    
    # Generate health report to see potential issues
    report = monitor.generate_health_report()
    
    if report.potential_issues:
        print(f"   Detected {len(report.potential_issues)} potential issues:")
        for issue in report.potential_issues:
            print(f"     - {issue.issue_type}:")
            print(f"       Probability: {issue.probability:.1%}")
            if issue.estimated_time_to_failure:
                print(f"       ETA to failure: {issue.estimated_time_to_failure}")
            print(f"       Affected components: {', '.join(issue.affected_components)}")
            print(f"       Recommended actions: {', '.join(issue.recommended_actions[:2])}")
    else:
        print("   No immediate potential issues detected")
    
    if report.recommendations:
        print(f"   System recommendations:")
        for rec in report.recommendations[:3]:  # Show first 3
            print(f"     - {rec}")


def demonstrate_component_health_monitoring(monitor: DiagnosticMonitor):
    """Demonstrate component health monitoring"""
    print("\n🏥 Demonstrating component health monitoring...")
    
    components = ["model_downloader", "dependency_manager", "config_validator", "error_handler"]
    
    for component in components:
        health = monitor.check_component_health(component)
        print(f"   {component}:")
        print(f"     Status: {health.status.value}")
        print(f"     Response Time: {health.response_time_ms:.1f}ms")
        print(f"     Performance Score: {health.performance_score:.1f}/100")
        print(f"     Error Count (30min): {health.error_count}")


def main():
    """Main demo function"""
    print("🚀 Diagnostic Monitor Demo")
    print("=" * 50)
    
    # Initialize monitor
    print("\n📋 Initializing Diagnostic Monitor...")
    monitor = DiagnosticMonitor()
    
    # Register callbacks
    monitor.add_alert_callback(alert_handler)
    monitor.add_health_callback(health_report_handler)
    
    # Start monitoring
    print("   Starting continuous monitoring...")
    monitor.start_monitoring()
    
    try:
        # Let initial monitoring stabilize
        print("   Allowing monitoring to stabilize...")
        time.sleep(5)
        
        # Demonstrate component health monitoring
        demonstrate_component_health_monitoring(monitor)
        
        # Demonstrate error pattern detection
        demonstrate_error_pattern_detection(monitor)
        
        # Simulate system load to trigger alerts
        simulate_system_load()
        
        # Wait for alerts to be processed
        time.sleep(3)
        
        # Demonstrate predictive analysis
        demonstrate_predictive_analysis(monitor)
        
        # Show monitoring status
        print("\n📈 Current Monitoring Status:")
        status = monitor.get_monitoring_status()
        for key, value in status.items():
            print(f"   {key}: {value}")
        
        # Generate final comprehensive report
        print("\n📋 Final Comprehensive Health Report:")
        final_report = monitor.generate_health_report()
        
        print(f"   Overall Health: {final_report.overall_health.value.upper()}")
        print(f"   Active Alerts: {len(final_report.active_alerts)}")
        print(f"   Error Patterns: {len(final_report.error_patterns)}")
        print(f"   Potential Issues: {len(final_report.potential_issues)}")
        print(f"   Component Health Checks: {len(final_report.component_health)}")
        
        # Show performance trends if available
        if final_report.performance_trends:
            print("   Performance Trends Available:")
            for metric, values in final_report.performance_trends.items():
                if values:
                    avg_value = sum(values) / len(values)
                    print(f"     {metric}: avg {avg_value:.1f}% (last {len(values)} measurements)")
        
        # Show recommendations
        if final_report.recommendations:
            print("   System Recommendations:")
            for i, rec in enumerate(final_report.recommendations[:5], 1):
                print(f"     {i}. {rec}")
        
        print("\n✅ Demo completed successfully!")
        print("   The diagnostic monitor demonstrated:")
        print("   - Real-time resource monitoring")
        print("   - Alert generation and callbacks")
        print("   - Component health tracking")
        print("   - Error pattern detection")
        print("   - Predictive failure analysis")
        print("   - Comprehensive health reporting")
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Stop monitoring
        print("\n🛑 Stopping diagnostic monitor...")
        monitor.stop_monitoring()
        print("   Monitor stopped")


if __name__ == "__main__":
    main()
