"""
Command-line interface for the health monitoring system
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

from health_checker import ProjectHealthChecker
from health_reporter import HealthReporter
from health_notifier import HealthNotifier
from recommendation_engine import RecommendationEngine
from health_analytics import HealthAnalytics
from health_models import HealthConfig, Severity


class HealthMonitorCLI:
    """Command-line interface for health monitoring"""
    
    def __init__(self):
        self.config = HealthConfig()
        self.health_checker = ProjectHealthChecker(self.config)
        self.health_reporter = HealthReporter(self.config)
        self.health_notifier = HealthNotifier(self.config)
        self.recommendation_engine = RecommendationEngine(self.config)
        self.health_analytics = HealthAnalytics()
    
    async def run_health_check(self, args) -> int:
        """Run a health check"""
        try:
            print("Running project health check...")
            
            # Run health check
            categories = None
            if args.categories:
                from health_models import HealthCategory
                categories = [HealthCategory(cat) for cat in args.categories.split(',')]
            
            report = await self.health_checker.run_health_check(categories)
            
            # Generate recommendations
            if not args.no_recommendations:
                recommendations = self.recommendation_engine.generate_recommendations(report)
                report.recommendations = recommendations
            
            # Generate report
            if args.format == "console":
                self.health_reporter._print_console_report(report)
            else:
                output_file = self.health_reporter.generate_report(report, args.format)
                print(f"Report generated: {output_file}")
            
            # Send notifications if enabled
            if args.notify:
                notifications = await self.health_notifier.process_health_report(report)
                if notifications:
                    print(f"Sent {len(notifications)} notifications")
            
            # Set exit code based on health score
            if args.exit_code_threshold:
                return 0 if report.overall_score >= args.exit_code_threshold else 1
            
            return 0
            
        except Exception as e:
            print(f"Error running health check: {e}")
            return 1
    
    def generate_recommendations(self, args) -> int:
        """Generate recommendations from latest health report"""
        try:
            # Load latest health report
            history_file = Path("tools/health-checker/health_history.json")
            if not history_file.exists():
                print("No health history found. Run a health check first.")
                return 1
            
            with open(history_file, 'r') as f:
                history_data = json.load(f)
            
            score_history = history_data.get('score_history', [])
            if not score_history:
                print("No health data found in history.")
                return 1
            
            # Create a mock report from latest data (simplified)
            from health_models import HealthReport, HealthTrends
            from datetime import datetime
            
            latest_score = score_history[-1][1]
            trends = HealthTrends(
                score_history=[(datetime.fromisoformat(ts), score) for ts, score in score_history],
                improvement_rate=history_data.get('improvement_rate', 0.0)
            )
            
            # This is a simplified approach - in practice, you'd want to load a full report
            print("Generating recommendations based on latest health data...")
            print(f"Latest health score: {latest_score:.1f}")
            
            # For now, just show the recommendation engine capabilities
            print("\nRecommendation engine is ready. Run a full health check to get specific recommendations.")
            
            return 0
            
        except Exception as e:
            print(f"Error generating recommendations: {e}")
            return 1
    
    def analyze_trends(self, args) -> int:
        """Analyze health trends"""
        try:
            print(f"Analyzing health trends for the last {args.days} days...")
            
            analysis = self.health_analytics.analyze_health_trends(args.days)
            
            if "error" in analysis:
                print(f"Error: {analysis['error']}")
                return 1
            
            # Display analysis results
            print(f"\nTrend Analysis ({args.days} days):")
            print(f"Data Points: {analysis['time_period']['data_points']}")
            
            score_analysis = analysis.get('score_analysis', {})
            print(f"Current Score: {score_analysis.get('current_score', 0):.1f}")
            print(f"Average Score: {score_analysis.get('average_score', 0):.1f}")
            print(f"Score Range: {score_analysis.get('score_range', 0):.1f}")
            
            trend_analysis = analysis.get('trend_analysis', {})
            print(f"Trend Direction: {trend_analysis.get('trend_direction', 'unknown')}")
            print(f"Trend Strength: {trend_analysis.get('trend_strength', 'unknown')}")
            
            volatility = analysis.get('volatility_analysis', {})
            print(f"Volatility Level: {volatility.get('volatility_level', 'unknown')}")
            
            recommendations = analysis.get('recommendations', [])
            if recommendations:
                print("\nRecommendations:")
                for rec in recommendations:
                    print(f"- {rec}")
            
            return 0
            
        except Exception as e:
            print(f"Error analyzing trends: {e}")
            return 1
    
    def test_notifications(self, args) -> int:
        """Test notification channels"""
        try:
            print("Testing notification channels...")
            
            results = self.health_notifier.test_notifications()
            
            print("\nNotification Test Results:")
            for channel, success in results.items():
                status = "✓ PASS" if success else "✗ FAIL"
                print(f"  {channel}: {status}")
            
            total_channels = len(results)
            successful_channels = sum(results.values())
            
            print(f"\nSummary: {successful_channels}/{total_channels} channels working")
            
            return 0 if successful_channels > 0 else 1
            
        except Exception as e:
            print(f"Error testing notifications: {e}")
            return 1
    
    def run_dashboard(self, args) -> int:
        """Run the health dashboard server"""
        try:
            from dashboard_server import HealthDashboard
            
            dashboard = HealthDashboard(self.config)
            
            if not dashboard.app:
                print("Dashboard server not available. Install FastAPI and uvicorn.")
                return 1
            
            print(f"Starting health dashboard at http://{args.host}:{args.port}")
            dashboard.run_server(args.host, args.port)
            
            return 0
            
        except Exception as e:
            print(f"Error running dashboard: {e}")
            return 1
    
    def export_config(self, args) -> int:
        """Export current configuration"""
        try:
            config_data = {
                "test_weight": self.config.test_weight,
                "documentation_weight": self.config.documentation_weight,
                "configuration_weight": self.config.configuration_weight,
                "code_quality_weight": self.config.code_quality_weight,
                "performance_weight": self.config.performance_weight,
                "security_weight": self.config.security_weight,
                "critical_threshold": self.config.critical_threshold,
                "warning_threshold": self.config.warning_threshold,
                "full_check_interval": self.config.full_check_interval,
                "quick_check_interval": self.config.quick_check_interval,
                "enable_notifications": self.config.enable_notifications,
                "notification_channels": self.config.notification_channels,
                "max_check_duration": self.config.max_check_duration,
                "parallel_checks": self.config.parallel_checks,
                "max_workers": self.config.max_workers
            }
            
            output_file = Path(args.output)
            with open(output_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            print(f"Configuration exported to {output_file}")
            return 0
            
        except Exception as e:
            print(f"Error exporting configuration: {e}")
            return 1


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser"""
    parser = argparse.ArgumentParser(
        description="Project Health Monitoring System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s check                          # Run full health check
  %(prog)s check --format html            # Generate HTML report
  %(prog)s check --categories tests,docs # Check specific categories
  %(prog)s check --notify                 # Send notifications
  %(prog)s trends --days 30               # Analyze 30-day trends
  %(prog)s test-notifications             # Test notification channels
  %(prog)s dashboard                      # Run web dashboard
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Health check command
    check_parser = subparsers.add_parser('check', help='Run health check')
    check_parser.add_argument(
        '--format', 
        choices=['console', 'html', 'json', 'markdown'],
        default='console',
        help='Output format (default: console)'
    )
    check_parser.add_argument(
        '--categories',
        help='Comma-separated list of categories to check (tests,documentation,configuration,code_quality)'
    )
    check_parser.add_argument(
        '--no-recommendations',
        action='store_true',
        help='Skip generating recommendations'
    )
    check_parser.add_argument(
        '--notify',
        action='store_true',
        help='Send notifications based on results'
    )
    check_parser.add_argument(
        '--exit-code-threshold',
        type=float,
        help='Set exit code based on health score threshold'
    )
    
    # Recommendations command
    rec_parser = subparsers.add_parser('recommendations', help='Generate recommendations')
    rec_parser.add_argument(
        '--format',
        choices=['console', 'json', 'markdown'],
        default='console',
        help='Output format'
    )
    
    # Trends analysis command
    trends_parser = subparsers.add_parser('trends', help='Analyze health trends')
    trends_parser.add_argument(
        '--days',
        type=int,
        default=30,
        help='Number of days to analyze (default: 30)'
    )
    
    # Test notifications command
    subparsers.add_parser('test-notifications', help='Test notification channels')
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser('dashboard', help='Run web dashboard')
    dashboard_parser.add_argument(
        '--host',
        default='127.0.0.1',
        help='Host to bind to (default: 127.0.0.1)'
    )
    dashboard_parser.add_argument(
        '--port',
        type=int,
        default=8080,
        help='Port to bind to (default: 8080)'
    )
    
    # Export config command
    export_parser = subparsers.add_parser('export-config', help='Export configuration')
    export_parser.add_argument(
        '--output',
        default='health_config.json',
        help='Output file (default: health_config.json)'
    )
    
    return parser


async def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    cli = HealthMonitorCLI()
    
    # Route to appropriate command handler
    if args.command == 'check':
        return await cli.run_health_check(args)
    elif args.command == 'recommendations':
        return cli.generate_recommendations(args)
    elif args.command == 'trends':
        return cli.analyze_trends(args)
    elif args.command == 'test-notifications':
        return cli.test_notifications(args)
    elif args.command == 'dashboard':
        return cli.run_dashboard(args)
    elif args.command == 'export-config':
        return cli.export_config(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


def cli_main():
    """Synchronous entry point for CLI"""
    try:
        return asyncio.run(main())
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(cli_main())