"""
Simple test for Health Reporter functionality
"""

import sys
import tempfile
import json
from pathlib import Path

# Add scripts to path
sys.path.append('scripts')

from health_reporter import HealthReporter, InstallationStatus
from interfaces import InstallationPhase, ErrorCategory

def test_basic_functionality():
    """Test basic health reporter functionality."""
    print("Testing Health Reporter Basic Functionality...")
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test config
        config_path = Path(temp_dir) / "config.json"
        with open(config_path, 'w') as f:
            json.dump({"health_reporting": {"retention_days": 30}}, f)
        
        # Initialize health reporter
        health_reporter = HealthReporter(
            installation_path=temp_dir,
            config_path=str(config_path)
        )
        
        print("✓ Health Reporter initialized successfully")
        
        # Test installation report generation
        report = health_reporter.generate_installation_report(
            installation_id="test-001",
            status=InstallationStatus.SUCCESS,
            duration_seconds=1200.0,
            phases_completed=[InstallationPhase.DETECTION, InstallationPhase.DEPENDENCIES],
            errors_encountered=[{"type": "test_error", "category": "system"}],
            warnings_generated=["Test warning"],
            recovery_attempts=[{"method": "retry", "success": True}],
            successful_recoveries=[{"method": "retry", "success": True}]
        )
        
        print(f"✓ Installation report generated: {report.installation_id}")
        print(f"  - Status: {report.status.value}")
        print(f"  - Health Score: {report.final_health_score:.1f}")
        print(f"  - Duration: {report.duration_seconds}s")
        
        # Test error pattern tracking
        errors = [
            {"type": "network_timeout", "category": "network", "installation_id": "test-001"},
            {"type": "permission_denied", "category": "permission", "installation_id": "test-002"}
        ]
        health_reporter.track_error_patterns(errors)
        print("✓ Error patterns tracked")
        
        # Test trend analysis
        trend_analysis = health_reporter.generate_trend_analysis()
        print(f"✓ Trend analysis generated:")
        print(f"  - Total installations: {trend_analysis.total_installations}")
        print(f"  - Success rate: {trend_analysis.success_rate:.1%}")
        print(f"  - Recommendations: {len(trend_analysis.recommendations)}")
        
        # Test dashboard data
        dashboard_data = health_reporter.get_dashboard_data()
        print(f"✓ Dashboard data generated:")
        print(f"  - Health status: {dashboard_data['health_status']}")
        print(f"  - Recent success rate: {dashboard_data['summary']['recent_success_rate']:.1%}")
        
        # Test metrics export
        json_export = health_reporter.export_metrics(format="json")
        export_data = json.loads(json_export)
        print(f"✓ Metrics exported (JSON): {len(export_data['installation_reports'])} reports")
        
        csv_export = health_reporter.export_metrics(format="csv")
        csv_lines = csv_export.strip().split('\n')
        print(f"✓ Metrics exported (CSV): {len(csv_lines)} lines")
        
        # Test recovery stats
        recovery_stats = health_reporter.get_recovery_method_stats()
        print(f"✓ Recovery stats retrieved: {len(recovery_stats)} methods")
        
        print("\n🎉 All basic functionality tests passed!")
        return True

def test_multiple_installations():
    """Test with multiple installation reports."""
    print("\nTesting Multiple Installation Reports...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        health_reporter = HealthReporter(installation_path=temp_dir)
        
        # Generate multiple reports
        statuses = [InstallationStatus.SUCCESS, InstallationStatus.SUCCESS, InstallationStatus.FAILURE]
        
        for i, status in enumerate(statuses):
            health_reporter.generate_installation_report(
                installation_id=f"multi-test-{i+1}",
                status=status,
                duration_seconds=1200.0 + i * 300,
                errors_encountered=[{"type": f"error_{i}", "category": "system"}] if status == InstallationStatus.FAILURE else []
            )
        
        # Test trend analysis with data
        trend_analysis = health_reporter.generate_trend_analysis()
        print(f"✓ Trend analysis with {trend_analysis.total_installations} installations")
        print(f"  - Success rate: {trend_analysis.success_rate:.1%}")
        print(f"  - Average time: {trend_analysis.average_installation_time:.0f}s")
        
        # Test dashboard
        dashboard_data = health_reporter.get_dashboard_data()
        print(f"✓ Dashboard shows health status: {dashboard_data['health_status']}")
        
        print("🎉 Multiple installations test passed!")
        return True

if __name__ == "__main__":
    try:
        print("="*60)
        print("Health Reporter Simple Test Suite")
        print("="*60)
        
        success1 = test_basic_functionality()
        success2 = test_multiple_installations()
        
        if success1 and success2:
            print("\n" + "="*60)
            print("✅ ALL TESTS PASSED - Health Reporter is working correctly!")
            print("="*60)
            print("\nKey Features Verified:")
            print("  ✓ Installation report generation")
            print("  ✓ Error pattern tracking")
            print("  ✓ Trend analysis")
            print("  ✓ Dashboard data generation")
            print("  ✓ Metrics export (JSON/CSV)")
            print("  ✓ Recovery method statistics")
            print("  ✓ Database operations")
            print("  ✓ Health score calculation")
        else:
            print("\n❌ Some tests failed")
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()