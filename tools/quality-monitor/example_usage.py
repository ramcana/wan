"""
Example usage of the quality monitoring and alerting system.
"""

import time
from datetime import datetime
from tools.quality_monitor import QualityMonitor


def basic_usage_example():
    """Basic usage example of the quality monitoring system."""
    print("=== Quality Monitoring System - Basic Usage ===\n")
    
    # Initialize the monitoring system
    monitor = QualityMonitor(project_root=".")
    
    # Collect current quality metrics
    print("1. Collecting quality metrics...")
    metrics = monitor.collect_metrics()
    
    print(f"Collected {len(metrics)} quality metrics:")
    for metric in metrics:
        print(f"  • {metric.metric_type.value.replace('_', ' ').title()}: {metric.value:.2f}")
        if 'error' in metric.details:
            print(f"    (Error: {metric.details['error']})")
    
    print()
    
    # Analyze quality trends
    print("2. Analyzing quality trends...")
    trends = monitor.analyze_trends(days=30)
    
    if trends:
        print(f"Found {len(trends)} quality trends:")
        for trend in trends:
            direction_symbol = {
                'improving': '↗️',
                'stable': '→',
                'degrading': '↘️',
                'unknown': '?'
            }.get(trend.direction.value, '?')
            
            print(f"  • {trend.metric_type.value.replace('_', ' ').title()}: {direction_symbol} {trend.direction.value}")
            print(f"    Change rate: {trend.change_rate:+.2f}% per day (confidence: {trend.confidence:.2f})")
    else:
        print("  No trend data available. Collect metrics over time to see trends.")
    
    print()
    
    # Check for quality alerts
    print("3. Checking for quality alerts...")
    alerts = monitor.check_alerts()
    
    if alerts:
        print(f"Found {len(alerts)} active alerts:")
        for alert in alerts:
            severity_symbol = {
                'critical': '🔴',
                'high': '🟠',
                'medium': '🟡',
                'low': '🟢'
            }.get(alert.severity.value, '⚪')
            
            print(f"  {severity_symbol} {alert.severity.value.upper()}: {alert.message}")
            print(f"    {alert.description}")
            if alert.recommendations:
                print(f"    Recommendation: {alert.recommendations[0]}")
    else:
        print("  No active alerts.")
    
    print()
    
    # Get improvement recommendations
    print("4. Getting improvement recommendations...")
    recommendations = monitor.get_recommendations()
    
    if recommendations:
        print(f"Found {len(recommendations)} recommendations:")
        # Sort by priority and impact
        priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        recommendations.sort(key=lambda r: (priority_order.get(r.priority.value, 4), -r.estimated_impact))
        
        for i, rec in enumerate(recommendations[:5], 1):  # Show top 5
            priority_symbol = {
                'critical': '🔴',
                'high': '🟠',
                'medium': '🟡',
                'low': '🟢'
            }.get(rec.priority.value, '⚪')
            
            print(f"  {i}. {priority_symbol} {rec.title}")
            print(f"     Priority: {rec.priority.value.upper()} | Impact: {rec.estimated_impact:.1f}% | Effort: {rec.estimated_effort}")
            print(f"     {rec.description}")
            if rec.actions:
                print(f"     Next step: {rec.actions[0]}")
    else:
        print("  No recommendations available.")
    
    print()


def dashboard_example():
    """Example of using the web dashboard."""
    print("=== Quality Monitoring Dashboard Example ===\n")
    
    monitor = QualityMonitor(project_root=".")
    
    print("Starting quality monitoring dashboard...")
    print("Dashboard will be available at http://localhost:8080")
    print("Features:")
    print("  • Real-time quality metrics display")
    print("  • Interactive trend charts")
    print("  • Alert management and resolution")
    print("  • Recommendation tracking")
    print("  • Auto-refresh every 5 minutes")
    print("\nPress Ctrl+C to stop the dashboard")
    
    try:
        monitor.start_dashboard(host="localhost", port=8080)
    except KeyboardInterrupt:
        print("\nDashboard stopped.")


def continuous_monitoring_example():
    """Example of continuous quality monitoring."""
    print("=== Continuous Quality Monitoring Example ===\n")
    
    monitor = QualityMonitor(project_root=".")
    
    print("Starting continuous monitoring (collecting metrics every 5 minutes)...")
    print("This would typically run as a background service.")
    print("Press Ctrl+C to stop monitoring")
    
    try:
        iteration = 0
        while True:
            iteration += 1
            print(f"\n--- Monitoring Iteration {iteration} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
            
            # Refresh all monitoring data
            monitor.refresh_data()
            
            # Get dashboard summary
            dashboard_data = monitor.get_dashboard_data()
            
            print(f"Metrics: {len(dashboard_data.metrics)}")
            print(f"Active alerts: {len(dashboard_data.alerts)}")
            print(f"Recommendations: {len(dashboard_data.recommendations)}")
            
            # Show any critical alerts
            critical_alerts = [a for a in dashboard_data.alerts if a.severity.value == 'critical']
            if critical_alerts:
                print(f"🔴 CRITICAL: {len(critical_alerts)} critical alert(s) require immediate attention!")
                for alert in critical_alerts:
                    print(f"  • {alert.message}: {alert.description}")
            
            # Show degrading trends
            degrading_trends = [t for t in dashboard_data.trends if t.direction.value == 'degrading']
            if degrading_trends:
                print(f"📉 WARNING: {len(degrading_trends)} metric(s) showing degrading trends")
                for trend in degrading_trends:
                    print(f"  • {trend.metric_type.value}: {trend.change_rate:+.2f}% per day")
            
            # Wait for next iteration (5 minutes in real usage)
            print("Waiting 30 seconds for next check... (would be 5 minutes in production)")
            time.sleep(30)
            
    except KeyboardInterrupt:
        print("\nContinuous monitoring stopped.")


def integration_example():
    """Example of integrating quality monitoring into development workflow."""
    print("=== Development Workflow Integration Example ===\n")
    
    monitor = QualityMonitor(project_root=".")
    
    # Simulate a development workflow
    print("1. Pre-commit quality check...")
    
    # Collect current metrics
    metrics = monitor.collect_metrics()
    
    # Check for critical issues
    alerts = monitor.check_alerts()
    critical_alerts = [a for a in alerts if a.severity.value == 'critical']
    
    if critical_alerts:
        print("❌ COMMIT BLOCKED: Critical quality issues found!")
        for alert in critical_alerts:
            print(f"  • {alert.message}: {alert.description}")
        print("Please fix these issues before committing.")
        return False
    
    print("✅ Pre-commit quality check passed.")
    
    print("\n2. Post-commit quality analysis...")
    
    # Get recommendations for improvement
    recommendations = monitor.get_recommendations()
    high_priority_recs = [r for r in recommendations if r.priority.value in ['critical', 'high']]
    
    if high_priority_recs:
        print(f"📋 Found {len(high_priority_recs)} high-priority improvement opportunities:")
        for rec in high_priority_recs[:3]:
            print(f"  • {rec.title} (Impact: {rec.estimated_impact:.1f}%)")
    
    print("\n3. Quality trend monitoring...")
    
    # Check for concerning trends
    trends = monitor.analyze_trends(days=7)  # Last week
    concerning_trends = [t for t in trends if t.direction.value == 'degrading' and abs(t.change_rate) > 2.0]
    
    if concerning_trends:
        print(f"⚠️  Warning: {len(concerning_trends)} metric(s) showing concerning trends:")
        for trend in concerning_trends:
            print(f"  • {trend.metric_type.value}: {trend.change_rate:+.2f}% per day")
        print("Consider addressing these trends in the next sprint.")
    else:
        print("✅ No concerning quality trends detected.")
    
    return True


def custom_metrics_example():
    """Example of working with individual quality components."""
    print("=== Custom Metrics Collection Example ===\n")
    
    from tools.quality_monitor import MetricsCollector, TrendAnalyzer, AlertSystem, RecommendationEngine
    
    # Use individual components for custom workflows
    collector = MetricsCollector(project_root=".")
    
    print("1. Collecting specific metrics...")
    
    # Collect individual metrics
    test_coverage = collector.collect_test_coverage()
    print(f"Test Coverage: {test_coverage.value:.2f}%")
    if test_coverage.details:
        print(f"  Lines covered: {test_coverage.details.get('lines_covered', 'N/A')}")
        print(f"  Total lines: {test_coverage.details.get('total_lines', 'N/A')}")
    
    code_complexity = collector.collect_code_complexity()
    print(f"Average Code Complexity: {code_complexity.value:.2f}")
    if code_complexity.details:
        print(f"  Functions analyzed: {code_complexity.details.get('function_count', 'N/A')}")
    
    doc_coverage = collector.collect_documentation_coverage()
    print(f"Documentation Coverage: {doc_coverage.value:.2f}%")
    
    print("\n2. Custom alert checking...")
    
    # Set up custom alert system
    alert_system = AlertSystem()
    
    # Check specific metrics against thresholds
    metrics = [test_coverage, code_complexity, doc_coverage]
    new_alerts = alert_system.check_metric_alerts(metrics)
    
    if new_alerts:
        print(f"Generated {len(new_alerts)} new alerts:")
        for alert in new_alerts:
            print(f"  • {alert.severity.value.upper()}: {alert.message}")
    else:
        print("No new alerts generated.")
    
    print("\n3. Custom recommendations...")
    
    # Generate targeted recommendations
    rec_engine = RecommendationEngine()
    recommendations = rec_engine.generate_metric_recommendations(metrics)
    
    if recommendations:
        print(f"Generated {len(recommendations)} recommendations:")
        for rec in recommendations:
            print(f"  • {rec.title} (Priority: {rec.priority.value})")
            print(f"    Estimated impact: {rec.estimated_impact:.1f}%")
    else:
        print("No recommendations generated.")


def main():
    """Run all examples."""
    examples = [
        ("Basic Usage", basic_usage_example),
        ("Continuous Monitoring", continuous_monitoring_example),
        ("Development Workflow Integration", integration_example),
        ("Custom Metrics Collection", custom_metrics_example),
    ]
    
    print("Quality Monitoring System - Example Usage\n")
    print("Available examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    print(f"  {len(examples) + 1}. Web Dashboard")
    print(f"  {len(examples) + 2}. Run all examples")
    
    try:
        choice = input(f"\nSelect an example (1-{len(examples) + 2}): ").strip()
        
        if choice == str(len(examples) + 1):
            dashboard_example()
        elif choice == str(len(examples) + 2):
            for name, func in examples:
                print(f"\n{'=' * 60}")
                print(f"Running: {name}")
                print('=' * 60)
                func()
                input("\nPress Enter to continue to next example...")
        else:
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(examples):
                name, func = examples[choice_idx]
                print(f"\n{'=' * 60}")
                print(f"Running: {name}")
                print('=' * 60)
                func()
            else:
                print("Invalid choice.")
    
    except (ValueError, KeyboardInterrupt):
        print("\nExiting...")


if __name__ == '__main__':
    main()