"""
Example usage of the health monitoring system
"""

import asyncio
from pathlib import Path

from health_checker import ProjectHealthChecker
from health_reporter import HealthReporter
from health_notifier import HealthNotifier
from recommendation_engine import RecommendationEngine
from health_analytics import HealthAnalytics
from health_models import HealthConfig


async def main():
    """Example usage of the health monitoring system"""
    
    print("🏥 Project Health Monitoring System - Example Usage")
    print("=" * 60)
    
    # Initialize configuration
    config = HealthConfig(
        project_root=Path("."),
        test_directory=Path("tests"),
        docs_directory=Path("docs"),
        config_directory=Path("config")
    )
    
    # Initialize components
    health_checker = ProjectHealthChecker(config)
    health_reporter = HealthReporter(config)
    health_notifier = HealthNotifier(config)
    recommendation_engine = RecommendationEngine(config)
    health_analytics = HealthAnalytics()
    
    try:
        # 1. Run comprehensive health check
        print("\n1. Running comprehensive health check...")
        report = await health_checker.run_health_check()
        
        print(f"   Overall Health Score: {report.overall_score:.1f}/100")
        print(f"   Components Checked: {len(report.component_scores)}")
        print(f"   Issues Found: {len(report.issues)}")
        print(f"   Critical Issues: {len(report.get_critical_issues())}")
        
        # 2. Generate recommendations
        print("\n2. Generating actionable recommendations...")
        recommendations = recommendation_engine.generate_recommendations(report)
        report.recommendations = recommendations
        
        print(f"   Recommendations Generated: {len(recommendations)}")
        if recommendations:
            print(f"   Top Priority: {recommendations[0].title}")
        
        # 3. Generate reports
        print("\n3. Generating health reports...")
        
        # Console report
        print("\n   Console Report:")
        health_reporter._print_console_report(report)
        
        # HTML report
        html_file = health_reporter.generate_report(report, "html")
        print(f"\n   HTML Report: {html_file}")
        
        # JSON report
        json_file = health_reporter.generate_report(report, "json")
        print(f"   JSON Report: {json_file}")
        
        # 4. Test notifications
        print("\n4. Testing notification channels...")
        notification_results = health_notifier.test_notifications()
        
        for channel, success in notification_results.items():
            status = "✅ Working" if success else "❌ Failed"
            print(f"   {channel}: {status}")
        
        # 5. Process health report for notifications
        print("\n5. Processing health report for alerts...")
        notifications_sent = await health_notifier.process_health_report(report)
        
        if notifications_sent:
            print(f"   Notifications sent: {len(notifications_sent)}")
        else:
            print("   No notifications triggered")
        
        # 6. Analyze trends (if history exists)
        print("\n6. Analyzing health trends...")
        try:
            trends = health_analytics.analyze_health_trends(30)
            if "error" not in trends:
                print(f"   Trend Direction: {trends.get('trend_analysis', {}).get('trend_direction', 'unknown')}")
                print(f"   Volatility: {trends.get('volatility_analysis', {}).get('volatility_level', 'unknown')}")
            else:
                print(f"   {trends['error']}")
        except Exception as e:
            print(f"   Trend analysis not available: {e}")
        
        # 7. Generate implementation plan
        if recommendations:
            print("\n7. Generating implementation plan...")
            implementation_plan = recommendation_engine.generate_implementation_plan(recommendations)
            
            print(f"   Implementation Phases: {len(implementation_plan['phases'])}")
            print(f"   Estimated Duration: {implementation_plan['estimated_total_duration']}")
            
            # Show first phase
            if implementation_plan['phases']:
                first_phase = implementation_plan['phases'][0]
                print(f"   First Phase: {first_phase['name']} ({first_phase['estimated_duration']})")
        
        # 8. Dashboard data
        print("\n8. Generating dashboard data...")
        dashboard_data = health_reporter.generate_dashboard_data(report)
        
        print(f"   Dashboard Status: {dashboard_data['status']}")
        print(f"   Components: {len(dashboard_data['components'])}")
        print(f"   Recent Trends: {len(dashboard_data['trends']['recent_scores'])} data points")
        
        print("\n" + "=" * 60)
        print("✅ Health monitoring system demonstration completed!")
        print("\nNext steps:")
        print("- Review the generated HTML report for detailed analysis")
        print("- Implement the top-priority recommendations")
        print("- Set up automated health checks in your CI/CD pipeline")
        print("- Configure notification channels for your team")
        print("- Run the dashboard for real-time monitoring")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during health monitoring: {e}")
        return 1


def run_example():
    """Run the example"""
    return asyncio.run(main())


if __name__ == "__main__":
    import sys
    sys.exit(run_example())
