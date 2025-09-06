"""
Simple demonstration of Health Reporter functionality
"""

import sys
import json
from pathlib import Path

# Add scripts to path
sys.path.append('scripts')

from health_reporter import HealthReporter, InstallationStatus
from interfaces import InstallationPhase, ErrorCategory

def demonstrate_health_reporter():
    """Demonstrate health reporter functionality."""
    print("="*60)
    print("Health Reporter Demonstration")
    print("="*60)
    
    # Initialize health reporter in current directory
    health_reporter = HealthReporter(installation_path=".")
    print("✓ Health Reporter initialized")
    
    # Generate sample installation reports
    print("\n1. Generating Sample Installation Reports...")
    
    # Successful installation
    report1 = health_reporter.generate_installation_report(
        installation_id="demo-success-001",
        status=InstallationStatus.SUCCESS,
        duration_seconds=1200.0,
        phases_completed=[
            InstallationPhase.DETECTION,
            InstallationPhase.DEPENDENCIES,
            InstallationPhase.MODELS,
            InstallationPhase.CONFIGURATION,
            InstallationPhase.VALIDATION
        ],
        errors_encountered=[],
        warnings_generated=["Low disk space warning"],
        recovery_attempts=[],
        successful_recoveries=[],
        resource_usage_peak={"cpu_percent": 45.2, "memory_percent": 67.8},
        performance_metrics={"download_speed_mbps": 25.4, "install_time_s": 1200},
        user_interventions=0
    )
    print(f"  ✓ Success Report: {report1.installation_id} (Health Score: {report1.final_health_score:.1f})")
    
    # Installation with minor issues
    report2 = health_reporter.generate_installation_report(
        installation_id="demo-minor-issues-002",
        status=InstallationStatus.SUCCESS,
        duration_seconds=1800.0,
        phases_completed=[
            InstallationPhase.DETECTION,
            InstallationPhase.DEPENDENCIES,
            InstallationPhase.MODELS,
            InstallationPhase.CONFIGURATION
        ],
        errors_encountered=[
            {"type": "network_timeout", "category": "network", "message": "Download timeout", "recovery_method": "retry_with_backoff", "recovery_success": True}
        ],
        warnings_generated=["Slow network connection", "High memory usage"],
        recovery_attempts=[
            {"method": "retry_with_backoff", "success": True, "execution_time": 15.0, "error_types_handled": ["network_timeout"]}
        ],
        successful_recoveries=[
            {"method": "retry_with_backoff", "success": True, "execution_time": 15.0}
        ],
        resource_usage_peak={"cpu_percent": 78.5, "memory_percent": 89.2},
        performance_metrics={"download_speed_mbps": 8.2, "install_time_s": 1800},
        user_interventions=1
    )
    print(f"  ✓ Minor Issues Report: {report2.installation_id} (Health Score: {report2.final_health_score:.1f})")
    
    # Failed installation
    report3 = health_reporter.generate_installation_report(
        installation_id="demo-failure-003",
        status=InstallationStatus.FAILURE,
        duration_seconds=900.0,
        phases_completed=[InstallationPhase.DETECTION, InstallationPhase.DEPENDENCIES],
        errors_encountered=[
            {"type": "permission_denied", "category": "permission", "message": "Access denied", "recovery_method": "elevate_privileges", "recovery_success": False},
            {"type": "disk_space_insufficient", "category": "system", "message": "Not enough space", "recovery_method": "cleanup_temp", "recovery_success": False}
        ],
        warnings_generated=["Low disk space", "Permission issues"],
        recovery_attempts=[
            {"method": "elevate_privileges", "success": False, "execution_time": 5.0, "error_types_handled": ["permission_denied"]},
            {"method": "cleanup_temp", "success": False, "execution_time": 30.0, "error_types_handled": ["disk_space_insufficient"]}
        ],
        successful_recoveries=[],
        resource_usage_peak={"cpu_percent": 25.1, "memory_percent": 45.3},
        performance_metrics={"download_speed_mbps": 0.0, "install_time_s": 900},
        user_interventions=3
    )
    print(f"  ✓ Failure Report: {report3.installation_id} (Health Score: {report3.final_health_score:.1f})")
    
    # Track additional error patterns
    print("\n2. Tracking Error Patterns...")
    additional_errors = [
        {"type": "network_timeout", "category": "network", "installation_id": "demo-001"},
        {"type": "model_download_failed", "category": "network", "installation_id": "demo-002"},
        {"type": "dependency_conflict", "category": "dependency", "installation_id": "demo-003"},
        {"type": "network_timeout", "category": "network", "installation_id": "demo-004"}  # Repeat for pattern
    ]
    health_reporter.track_error_patterns(additional_errors)
    print(f"  ✓ Tracked {len(additional_errors)} additional error patterns")
    
    # Generate trend analysis
    print("\n3. Generating Trend Analysis...")
    trend_analysis = health_reporter.generate_trend_analysis()
    print(f"  ✓ Analysis Period: {trend_analysis.analysis_period.days} days")
    print(f"  ✓ Total Installations: {trend_analysis.total_installations}")
    print(f"  ✓ Success Rate: {trend_analysis.success_rate:.1%}")
    print(f"  ✓ Average Installation Time: {trend_analysis.average_installation_time:.0f} seconds")
    
    if trend_analysis.most_common_errors:
        print(f"  ✓ Most Common Errors:")
        for i, error in enumerate(trend_analysis.most_common_errors[:3], 1):
            print(f"    {i}. {error.error_type} ({error.occurrence_count} times, trend: {error.frequency_trend})")
    
    print(f"  ✓ Recovery Methods Analyzed: {len(trend_analysis.recovery_effectiveness)}")
    
    if trend_analysis.recommendations:
        print(f"  ✓ Recommendations:")
        for i, rec in enumerate(trend_analysis.recommendations[:3], 1):
            print(f"    {i}. {rec}")
    
    # Generate dashboard data
    print("\n4. Generating Dashboard Data...")
    dashboard_data = health_reporter.get_dashboard_data()
    print(f"  ✓ Overall Health Status: {dashboard_data['health_status']}")
    print(f"  ✓ Recent Success Rate: {dashboard_data['summary']['recent_success_rate']:.1%}")
    print(f"  ✓ Active Error Patterns: {dashboard_data['summary']['active_error_patterns']}")
    
    # Export metrics
    print("\n5. Exporting Metrics...")
    
    # JSON export
    json_export = health_reporter.export_metrics(format="json")
    json_file = Path("health_metrics_demo.json")
    with open(json_file, 'w') as f:
        f.write(json_export)
    print(f"  ✓ JSON export saved to: {json_file}")
    
    # CSV export
    csv_export = health_reporter.export_metrics(format="csv")
    csv_file = Path("health_metrics_demo.csv")
    with open(csv_file, 'w') as f:
        f.write(csv_export)
    print(f"  ✓ CSV export saved to: {csv_file}")
    
    # Recovery method statistics
    print("\n6. Recovery Method Statistics...")
    recovery_stats = health_reporter.get_recovery_method_stats()
    if recovery_stats:
        print(f"  ✓ Recovery Methods Tracked: {len(recovery_stats)}")
        for stats in recovery_stats:
            print(f"    • {stats.method_name}: {stats.success_rate:.1%} success rate "
                  f"({stats.successful_attempts}/{stats.total_attempts} attempts)")
    
    # Installation history
    print("\n7. Installation History...")
    history = health_reporter.get_installation_history(limit=10)
    print(f"  ✓ Retrieved {len(history)} recent installations")
    for install in history[:3]:  # Show first 3
        print(f"    • {install.installation_id}: {install.status.value} "
              f"(score: {install.final_health_score:.1f}, duration: {install.duration_seconds:.0f}s)")
    
    print("\n" + "="*60)
    print("Health Reporter Demonstration Complete!")
    print("="*60)
    print("\nGenerated Files:")
    print(f"  • Database: {health_reporter.db_path}")
    print(f"  • JSON Export: {json_file}")
    print(f"  • CSV Export: {csv_file}")
    
    print("\nKey Features Demonstrated:")
    print("  ✓ Comprehensive installation reporting")
    print("  ✓ Error pattern tracking and analysis")
    print("  ✓ Recovery method effectiveness logging")
    print("  ✓ Trend analysis across installations")
    print("  ✓ Dashboard data generation")
    print("  ✓ Metrics export in multiple formats")
    print("  ✓ Health score calculation")
    print("  ✓ Installation history tracking")
    
    return health_reporter

if __name__ == "__main__":
    try:
        health_reporter = demonstrate_health_reporter()
        print(f"\n🎉 Demonstration completed successfully!")
        print(f"📊 Health Reporter ready for production use")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed with error: {e}")
        import traceback
        traceback.print_exc()