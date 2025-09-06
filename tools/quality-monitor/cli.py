"""
Quality monitoring command-line interface.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

try:
from tools..models import MetricType, AlertSeverity
from tools..metrics_collector import MetricsCollector
from tools..trend_analyzer import TrendAnalyzer
from tools..alert_system import AlertSystem
from tools..recommendation_engine import RecommendationEngine
from tools..dashboard import DashboardManager
except ImportError:
    from models import MetricType, AlertSeverity
    from metrics_collector import MetricsCollector
    from trend_analyzer import TrendAnalyzer
    from alert_system import AlertSystem
    from recommendation_engine import RecommendationEngine
    from dashboard import DashboardManager


def cmd_collect_metrics(args):
    """Collect quality metrics."""
    collector = MetricsCollector(args.project_root)
    metrics = collector.collect_all_metrics()
    
    if args.output:
        output_file = Path(args.output)
        data = {
            'timestamp': datetime.now().isoformat(),
            'metrics': [metric.to_dict() for metric in metrics]
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Metrics saved to {output_file}")
    else:
        print("Quality Metrics:")
        print("-" * 50)
        
        for metric in metrics:
            print(f"{metric.metric_type.value.replace('_', ' ').title()}: {metric.value:.2f}")
            if metric.details:
                for key, value in metric.details.items():
                    if key != 'error':
                        print(f"  {key}: {value}")
        
        print(f"\nCollected {len(metrics)} metrics at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def cmd_analyze_trends(args):
    """Analyze quality trends."""
    analyzer = TrendAnalyzer()
    trends = analyzer.analyze_all_trends(args.days)
    
    if args.output:
        summary = analyzer.get_trend_summary(args.days)
        
        with open(args.output, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Trend analysis saved to {args.output}")
    else:
        print(f"Quality Trends (last {args.days} days):")
        print("-" * 50)
        
        if not trends:
            print("No trend data available. Collect metrics over time to see trends.")
            return
        
        for trend in trends:
            direction_symbol = {
                'improving': '↗️',
                'stable': '→',
                'degrading': '↘️',
                'unknown': '?'
            }.get(trend.direction.value, '?')
            
            print(f"{trend.metric_type.value.replace('_', ' ').title()}: {direction_symbol} {trend.direction.value}")
            print(f"  Change rate: {trend.change_rate:+.2f}% per day")
            print(f"  Confidence: {trend.confidence:.2f}")
            print(f"  Current: {trend.current_value:.2f}, Previous: {trend.previous_value:.2f}")
            print()


def cmd_check_alerts(args):
    """Check for quality alerts."""
    alert_system = AlertSystem()
    
    # Collect current metrics
    collector = MetricsCollector(args.project_root)
    metrics = collector.collect_all_metrics()
    
    # Check for metric alerts
    new_alerts = alert_system.check_metric_alerts(metrics)
    
    # Check for trend alerts if requested
    if args.check_trends:
        analyzer = TrendAnalyzer()
        trends = analyzer.analyze_all_trends()
        trend_alerts = alert_system.check_trend_alerts(trends)
        new_alerts.extend(trend_alerts)
    
    # Get all active alerts
    active_alerts = alert_system.get_active_alerts()
    
    if args.output:
        summary = alert_system.get_alert_summary()
        
        with open(args.output, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Alert summary saved to {args.output}")
    else:
        print("Quality Alerts:")
        print("-" * 50)
        
        if not active_alerts:
            print("No active alerts.")
            return
        
        for alert in active_alerts:
            severity_symbol = {
                'critical': '🔴',
                'high': '🟠',
                'medium': '🟡',
                'low': '🟢'
            }.get(alert.severity.value, '⚪')
            
            print(f"{severity_symbol} {alert.severity.value.upper()}: {alert.message}")
            print(f"  {alert.description}")
            print(f"  Component: {alert.component or 'Global'}")
            print(f"  Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            
            if alert.recommendations:
                print("  Recommendations:")
                for rec in alert.recommendations[:3]:  # Show first 3
                    print(f"    • {rec}")
            print()
        
        if new_alerts:
            print(f"\n{len(new_alerts)} new alert(s) generated.")


def cmd_get_recommendations(args):
    """Get quality improvement recommendations."""
    rec_engine = RecommendationEngine()
    
    # Collect current metrics
    collector = MetricsCollector(args.project_root)
    metrics = collector.collect_all_metrics()
    
    # Generate recommendations
    metric_recs = rec_engine.generate_metric_recommendations(metrics)
    
    # Get trend recommendations if requested
    trend_recs = []
    if args.include_trends:
        analyzer = TrendAnalyzer()
        trends = analyzer.analyze_all_trends()
        trend_recs = rec_engine.generate_trend_recommendations(trends)
    
    # Get proactive recommendations
    proactive_recs = rec_engine.generate_proactive_recommendations(metrics, [])
    
    all_recs = metric_recs + trend_recs + proactive_recs
    
    # Filter by priority if specified
    if args.priority:
        priority = AlertSeverity(args.priority.lower())
        all_recs = [r for r in all_recs if r.priority == priority]
    
    if args.output:
        summary = rec_engine.get_recommendations_summary()
        
        with open(args.output, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Recommendations saved to {args.output}")
    else:
        print("Quality Improvement Recommendations:")
        print("-" * 50)
        
        if not all_recs:
            print("No recommendations available.")
            return
        
        # Sort by priority and impact
        priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        all_recs.sort(key=lambda r: (priority_order.get(r.priority.value, 4), -r.estimated_impact))
        
        for rec in all_recs[:args.limit]:
            priority_symbol = {
                'critical': '🔴',
                'high': '🟠',
                'medium': '🟡',
                'low': '🟢'
            }.get(rec.priority.value, '⚪')
            
            print(f"{priority_symbol} {rec.title}")
            print(f"  Priority: {rec.priority.value.upper()}")
            print(f"  Impact: {rec.estimated_impact:.1f}% | Effort: {rec.estimated_effort}")
            print(f"  Description: {rec.description}")
            
            if rec.actions:
                print("  Actions:")
                for action in rec.actions[:3]:  # Show first 3 actions
                    print(f"    • {action}")
            print()


def cmd_start_dashboard(args):
    """Start the quality monitoring dashboard."""
    dashboard = DashboardManager(args.project_root)
    
    print(f"Starting quality monitoring dashboard...")
    print(f"Dashboard will be available at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop the server")
    
    try:
        dashboard.start_server(args.host, args.port)
    except KeyboardInterrupt:
        print("\nDashboard stopped.")


def cmd_cleanup(args):
    """Clean up old data files."""
    total_cleaned = 0
    
    if args.metrics or args.all:
        analyzer = TrendAnalyzer()
        cleaned = analyzer.cleanup_old_data(args.days)
        total_cleaned += cleaned
        print(f"Cleaned up {cleaned} old metric files")
    
    if args.alerts or args.all:
        alert_system = AlertSystem()
        cleaned = alert_system.cleanup_resolved_alerts(args.days)
        total_cleaned += cleaned
        print(f"Cleaned up {cleaned} old resolved alerts")
    
    if args.recommendations or args.all:
        rec_engine = RecommendationEngine()
        cleaned = rec_engine.cleanup_old_recommendations(args.days)
        total_cleaned += cleaned
        print(f"Cleaned up {cleaned} old recommendations")
    
    print(f"Total items cleaned up: {total_cleaned}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Quality monitoring and alerting system",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--project-root',
        default='.',
        help='Project root directory (default: current directory)'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Collect metrics command
    metrics_parser = subparsers.add_parser('metrics', help='Collect quality metrics')
    metrics_parser.add_argument('--output', '-o', help='Output file for metrics data')
    metrics_parser.set_defaults(func=cmd_collect_metrics)
    
    # Analyze trends command
    trends_parser = subparsers.add_parser('trends', help='Analyze quality trends')
    trends_parser.add_argument('--days', '-d', type=int, default=30, help='Number of days to analyze')
    trends_parser.add_argument('--output', '-o', help='Output file for trend analysis')
    trends_parser.set_defaults(func=cmd_analyze_trends)
    
    # Check alerts command
    alerts_parser = subparsers.add_parser('alerts', help='Check for quality alerts')
    alerts_parser.add_argument('--check-trends', action='store_true', help='Also check trend alerts')
    alerts_parser.add_argument('--output', '-o', help='Output file for alert summary')
    alerts_parser.set_defaults(func=cmd_check_alerts)
    
    # Get recommendations command
    recs_parser = subparsers.add_parser('recommendations', help='Get improvement recommendations')
    recs_parser.add_argument('--priority', choices=['critical', 'high', 'medium', 'low'], 
                            help='Filter by priority level')
    recs_parser.add_argument('--include-trends', action='store_true', 
                            help='Include trend-based recommendations')
    recs_parser.add_argument('--limit', type=int, default=10, help='Maximum number of recommendations')
    recs_parser.add_argument('--output', '-o', help='Output file for recommendations')
    recs_parser.set_defaults(func=cmd_get_recommendations)
    
    # Start dashboard command
    dashboard_parser = subparsers.add_parser('dashboard', help='Start monitoring dashboard')
    dashboard_parser.add_argument('--host', default='localhost', help='Dashboard host')
    dashboard_parser.add_argument('--port', type=int, default=8080, help='Dashboard port')
    dashboard_parser.set_defaults(func=cmd_start_dashboard)
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up old data files')
    cleanup_parser.add_argument('--days', type=int, default=90, help='Keep data newer than N days')
    cleanup_parser.add_argument('--metrics', action='store_true', help='Clean up metric files')
    cleanup_parser.add_argument('--alerts', action='store_true', help='Clean up resolved alerts')
    cleanup_parser.add_argument('--recommendations', action='store_true', help='Clean up old recommendations')
    cleanup_parser.add_argument('--all', action='store_true', help='Clean up all data types')
    cleanup_parser.set_defaults(func=cmd_cleanup)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        args.func(args)
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())