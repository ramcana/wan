#!/usr/bin/env python3
"""
Comprehensive maintenance reporting CLI tool.
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

try:
    from tools.maintenance-reporter.operation_logger import OperationLogger
    from tools.maintenance-reporter.impact_analyzer import ImpactAnalyzer
    from tools.maintenance-reporter.recommendation_engine import MaintenanceRecommendationEngine
    from tools.maintenance-reporter.report_generator import MaintenanceReportGenerator
    from tools.maintenance-reporter.models import MaintenanceOperationType, ImpactLevel
except ImportError:
    from operation_logger import OperationLogger
    from impact_analyzer import ImpactAnalyzer
    from recommendation_engine import MaintenanceRecommendationEngine
    from report_generator import MaintenanceReportGenerator
    from models import MaintenanceOperationType, ImpactLevel


class MaintenanceReporterCLI:
    """Command-line interface for the maintenance reporting system."""
    
    def __init__(self):
        self.operation_logger = OperationLogger()
        self.impact_analyzer = ImpactAnalyzer()
        self.recommendation_engine = MaintenanceRecommendationEngine()
        self.report_generator = MaintenanceReportGenerator()
    
    def start_operation(self, args):
        """Start a new maintenance operation."""
        try:
            operation_type = MaintenanceOperationType(args.type)
            impact_level = ImpactLevel(args.impact) if args.impact else ImpactLevel.MEDIUM
            
            operation_id = self.operation_logger.start_operation(
                operation_type=operation_type,
                title=args.title,
                description=args.description,
                impact_level=impact_level,
                files_affected=args.files.split(',') if args.files else None,
                components_affected=args.components.split(',') if args.components else None
            )
            
            print(f"Started operation: {operation_id}")
            print(f"Type: {operation_type.value}")
            print(f"Title: {args.title}")
            print(f"Impact Level: {impact_level.value}")
            
        except Exception as e:
            print(f"Error starting operation: {e}")
            sys.exit(1)
    
    def complete_operation(self, args):
        """Complete a maintenance operation."""
        try:
            success_metrics = {}
            if args.metrics:
                # Parse metrics from JSON string or key=value pairs
                if args.metrics.startswith('{'):
                    success_metrics = json.loads(args.metrics)
                else:
                    for pair in args.metrics.split(','):
                        key, value = pair.split('=')
                        success_metrics[key.strip()] = float(value.strip())
            
            success = self.operation_logger.complete_operation(
                operation_id=args.operation_id,
                success_metrics=success_metrics,
                files_affected=args.files.split(',') if args.files else None,
                components_affected=args.components.split(',') if args.components else None
            )
            
            if success:
                print(f"Operation {args.operation_id} completed successfully")
                
                # Generate impact analysis if before/after metrics provided
                if args.before_metrics and args.after_metrics:
                    before_metrics = json.loads(args.before_metrics)
                    after_metrics = json.loads(args.after_metrics)
                    
                    analysis = self.impact_analyzer.analyze_operation_impact(
                        args.operation_id, before_metrics, after_metrics
                    )
                    
                    print(f"Impact Analysis:")
                    print(f"  Overall Score: {analysis.overall_impact_score:.1f}")
                    print(f"  Summary: {analysis.impact_summary}")
            else:
                print(f"Operation {args.operation_id} not found")
                sys.exit(1)
                
        except Exception as e:
            print(f"Error completing operation: {e}")
            sys.exit(1)
    
    def fail_operation(self, args):
        """Mark an operation as failed."""
        try:
            rollback_info = {}
            if args.rollback_info:
                rollback_info = json.loads(args.rollback_info)
            
            success = self.operation_logger.fail_operation(
                operation_id=args.operation_id,
                error_details=args.error,
                rollback_info=rollback_info
            )
            
            if success:
                print(f"Operation {args.operation_id} marked as failed")
                print(f"Error: {args.error}")
            else:
                print(f"Operation {args.operation_id} not found")
                sys.exit(1)
                
        except Exception as e:
            print(f"Error failing operation: {e}")
            sys.exit(1)
    
    def list_operations(self, args):
        """List maintenance operations."""
        try:
            if args.status:
                from models import MaintenanceStatus
                status = MaintenanceStatus(args.status)
                operations = self.operation_logger.get_operations_by_status(status)
            elif args.type:
                operation_type = MaintenanceOperationType(args.type)
                operations = self.operation_logger.get_operations_by_type(operation_type)
            elif args.days:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=args.days)
                operations = self.operation_logger.get_operations_in_period(start_date, end_date)
            else:
                # Get all operations from the last 30 days
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)
                operations = self.operation_logger.get_operations_in_period(start_date, end_date)
            
            if not operations:
                print("No operations found matching criteria")
                return
            
            print(f"Found {len(operations)} operations:")
            print("-" * 80)
            
            for operation in operations:
                duration_text = ""
                if operation.duration_seconds:
                    duration_text = f" ({operation.duration_seconds // 60} min)"
                
                print(f"ID: {operation.id}")
                print(f"Title: {operation.title}")
                print(f"Type: {operation.operation_type.value}")
                print(f"Status: {operation.status.value}")
                print(f"Impact: {operation.impact_level.value}")
                print(f"Started: {operation.started_at.strftime('%Y-%m-%d %H:%M')}{duration_text}")
                
                if operation.files_affected:
                    print(f"Files: {len(operation.files_affected)} affected")
                
                if operation.components_affected:
                    print(f"Components: {', '.join(operation.components_affected)}")
                
                print("-" * 80)
                
        except Exception as e:
            print(f"Error listing operations: {e}")
            sys.exit(1)
    
    def generate_report(self, args):
        """Generate a maintenance report."""
        try:
            if args.type == 'daily':
                target_date = datetime.fromisoformat(args.date) if args.date else None
                report = self.report_generator.generate_daily_report(
                    self.operation_logger, self.impact_analyzer, target_date
                )
            elif args.type == 'weekly':
                target_date = datetime.fromisoformat(args.date) if args.date else None
                report = self.report_generator.generate_weekly_report(
                    self.operation_logger, self.impact_analyzer, target_date
                )
            elif args.type == 'monthly':
                target_date = datetime.fromisoformat(args.date) if args.date else None
                report = self.report_generator.generate_monthly_report(
                    self.operation_logger, self.impact_analyzer, target_date
                )
            elif args.type == 'comprehensive':
                period_days = args.days if args.days else 30
                report = self.report_generator.generate_comprehensive_report(
                    self.operation_logger, self.impact_analyzer, 
                    self.recommendation_engine, period_days
                )
            elif args.type == 'operation':
                if not args.operation_id:
                    print("Operation ID required for operation report")
                    sys.exit(1)
                report = self.report_generator.generate_operation_summary_report(
                    args.operation_id, self.operation_logger, self.impact_analyzer
                )
            else:
                print(f"Unknown report type: {args.type}")
                sys.exit(1)
            
            # Output report
            if args.format == 'json':
                if args.output:
                    with open(args.output, 'w') as f:
                        f.write(report.to_json())
                    print(f"Report saved to {args.output}")
                else:
                    print(report.to_json())
            
            elif args.format == 'html':
                output_path = Path(args.output) if args.output else None
                html_path = self.report_generator.export_report_to_html(report, output_path)
                print(f"HTML report saved to {html_path}")
            
            else:
                # Text summary
                self._print_report_summary(report)
                
        except Exception as e:
            print(f"Error generating report: {e}")
            sys.exit(1)
    
    def _print_report_summary(self, report):
        """Print a text summary of the report."""
        print(f"Maintenance Report - {report.report_type.title()}")
        print("=" * 60)
        print(f"Period: {report.period_start.strftime('%Y-%m-%d')} to {report.period_end.strftime('%Y-%m-%d')}")
        print(f"Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Summary statistics
        stats = report.summary_statistics
        if stats:
            op_stats = stats.get('operations', {})
            print("Summary Statistics:")
            print(f"  Total Operations: {op_stats.get('total', 0)}")
            print(f"  Success Rate: {op_stats.get('success_rate', 0):.1f}%")
            
            impact_stats = stats.get('impact', {})
            if impact_stats:
                print(f"  Average Impact Score: {impact_stats.get('average_impact_score', 0):.1f}")
                print(f"  Positive Impacts: {impact_stats.get('positive_impacts', 0)}")
            
            rec_stats = stats.get('recommendations', {})
            print(f"  Active Recommendations: {rec_stats.get('total', 0)}")
            print()
        
        # Operations
        if report.operations:
            print(f"Operations ({len(report.operations)}):")
            for operation in report.operations[:10]:  # Show first 10
                print(f"  • {operation.title} ({operation.status.value})")
            
            if len(report.operations) > 10:
                print(f"  ... and {len(report.operations) - 10} more")
            print()
        
        # Recommendations
        if report.recommendations:
            print(f"Recommendations ({len(report.recommendations)}):")
            for rec in report.recommendations[:5]:  # Show first 5
                print(f"  • {rec.title} ({rec.priority.value} priority)")
            
            if len(report.recommendations) > 5:
                print(f"  ... and {len(report.recommendations) - 5} more")
            print()
    
    def generate_recommendations(self, args):
        """Generate maintenance recommendations."""
        try:
            # Get project metrics (would normally come from project analysis)
            project_metrics = {
                'test_coverage': args.test_coverage if args.test_coverage else 75.0,
                'code_complexity': args.code_complexity if args.code_complexity else 8.0,
                'documentation_coverage': args.doc_coverage if args.doc_coverage else 65.0,
                'duplicate_code': args.duplicate_code if args.duplicate_code else 10.0,
                'style_violations': args.style_violations if args.style_violations else 25.0
            }
            
            recommendations = self.recommendation_engine.generate_recommendations_from_analysis(
                self.operation_logger, self.impact_analyzer, project_metrics
            )
            
            if not recommendations:
                print("No new recommendations generated")
                return
            
            print(f"Generated {len(recommendations)} recommendations:")
            print("-" * 80)
            
            for rec in recommendations:
                print(f"Title: {rec.title}")
                print(f"Priority: {rec.priority.value}")
                print(f"Type: {rec.operation_type.value}")
                print(f"Effort: {rec.estimated_effort_hours:.1f} hours")
                print(f"Impact: {rec.estimated_impact_score:.1f}")
                print(f"Description: {rec.description}")
                
                if rec.suggested_schedule:
                    print(f"Schedule: {rec.suggested_schedule}")
                
                print("-" * 80)
                
        except Exception as e:
            print(f"Error generating recommendations: {e}")
            sys.exit(1)
    
    def optimize_schedule(self, args):
        """Optimize maintenance schedule."""
        try:
            recommendations = self.recommendation_engine.get_active_recommendations(args.days)
            
            if not recommendations:
                print("No active recommendations found")
                return
            
            schedule = self.recommendation_engine.optimize_maintenance_schedule(
                recommendations,
                available_hours_per_week=args.hours_per_week,
                max_concurrent_operations=args.max_concurrent
            )
            
            print("Optimized Maintenance Schedule:")
            print("=" * 60)
            print(f"Total Duration: {schedule.estimated_total_duration_hours:.1f} hours")
            print(f"Resource Requirements:")
            
            for resource, hours in schedule.resource_requirements.items():
                print(f"  {resource.replace('_', ' ').title()}: {hours:.1f} hours")
            
            print()
            print("Recommended Schedule:")
            
            for i, op_id in enumerate(schedule.recommended_schedule, 1):
                rec = next((r for r in recommendations if r.id == op_id), None)
                if rec:
                    rationale = schedule.scheduling_rationale.get(op_id, "")
                    print(f"  {i}. {rec.title} - {rationale}")
            
            print()
            print("Risk Mitigation Plan:")
            for risk in schedule.risk_mitigation_plan:
                print(f"  • {risk}")
                
        except Exception as e:
            print(f"Error optimizing schedule: {e}")
            sys.exit(1)
    
    def show_statistics(self, args):
        """Show maintenance statistics."""
        try:
            stats = self.operation_logger.get_operation_statistics()
            
            print("Maintenance Operation Statistics:")
            print("=" * 50)
            print(f"Total Operations: {stats['total_operations']}")
            print(f"Success Rate: {stats['success_rate']:.1f}%")
            print(f"Average Duration: {stats['average_duration_minutes']:.1f} minutes")
            print()
            
            print("By Status:")
            for status, count in stats['by_status'].items():
                print(f"  {status.replace('_', ' ').title()}: {count}")
            
            print()
            print("By Type:")
            for op_type, count in stats['by_type'].items():
                print(f"  {op_type.replace('_', ' ').title()}: {count}")
            
            print()
            print("By Impact Level:")
            for impact, count in stats['by_impact_level'].items():
                print(f"  {impact.title()}: {count}")
            
            # Impact analysis statistics
            opportunities = self.impact_analyzer.identify_improvement_opportunities(
                self.operation_logger, args.days
            )
            
            if opportunities:
                print()
                print("Improvement Opportunities:")
                for opp in opportunities:
                    print(f"  • {opp['recommendation']} ({opp['priority']} priority)")
                    
        except Exception as e:
            print(f"Error showing statistics: {e}")
            sys.exit(1)
    
    def cleanup(self, args):
        """Clean up old data."""
        try:
            days = args.days if args.days else 90
            
            operations_cleaned = self.operation_logger.cleanup_old_operations(days)
            analyses_cleaned = self.impact_analyzer.cleanup_old_analyses(days)
            recommendations_cleaned = self.recommendation_engine.cleanup_old_recommendations(days)
            reports_cleaned = self.report_generator.cleanup_old_reports(days)
            
            total_cleaned = operations_cleaned + analyses_cleaned + recommendations_cleaned + reports_cleaned
            
            print(f"Cleanup completed (older than {days} days):")
            print(f"  Operations: {operations_cleaned}")
            print(f"  Impact Analyses: {analyses_cleaned}")
            print(f"  Recommendations: {recommendations_cleaned}")
            print(f"  Reports: {reports_cleaned}")
            print(f"  Total Items Cleaned: {total_cleaned}")
            
        except Exception as e:
            print(f"Error during cleanup: {e}")
            sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Comprehensive Maintenance Reporting System")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Start operation command
    start_parser = subparsers.add_parser('start', help='Start a maintenance operation')
    start_parser.add_argument('--type', required=True, 
                             choices=[t.value for t in MaintenanceOperationType],
                             help='Operation type')
    start_parser.add_argument('--title', required=True, help='Operation title')
    start_parser.add_argument('--description', required=True, help='Operation description')
    start_parser.add_argument('--impact', choices=[i.value for i in ImpactLevel],
                             help='Impact level (default: medium)')
    start_parser.add_argument('--files', help='Comma-separated list of affected files')
    start_parser.add_argument('--components', help='Comma-separated list of affected components')
    start_parser.set_defaults(func=lambda cli, args: cli.start_operation(args))
    
    # Complete operation command
    complete_parser = subparsers.add_parser('complete', help='Complete a maintenance operation')
    complete_parser.add_argument('operation_id', help='Operation ID')
    complete_parser.add_argument('--metrics', help='Success metrics (JSON or key=value pairs)')
    complete_parser.add_argument('--files', help='Additional affected files')
    complete_parser.add_argument('--components', help='Additional affected components')
    complete_parser.add_argument('--before-metrics', help='Before metrics (JSON)')
    complete_parser.add_argument('--after-metrics', help='After metrics (JSON)')
    complete_parser.set_defaults(func=lambda cli, args: cli.complete_operation(args))
    
    # Fail operation command
    fail_parser = subparsers.add_parser('fail', help='Mark an operation as failed')
    fail_parser.add_argument('operation_id', help='Operation ID')
    fail_parser.add_argument('--error', required=True, help='Error description')
    fail_parser.add_argument('--rollback-info', help='Rollback information (JSON)')
    fail_parser.set_defaults(func=lambda cli, args: cli.fail_operation(args))
    
    # List operations command
    list_parser = subparsers.add_parser('list', help='List maintenance operations')
    list_parser.add_argument('--status', choices=['scheduled', 'in_progress', 'completed', 'failed', 'cancelled'],
                            help='Filter by status')
    list_parser.add_argument('--type', choices=[t.value for t in MaintenanceOperationType],
                            help='Filter by operation type')
    list_parser.add_argument('--days', type=int, help='Show operations from last N days')
    list_parser.set_defaults(func=lambda cli, args: cli.list_operations(args))
    
    # Generate report command
    report_parser = subparsers.add_parser('report', help='Generate maintenance report')
    report_parser.add_argument('--type', required=True,
                              choices=['daily', 'weekly', 'monthly', 'comprehensive', 'operation'],
                              help='Report type')
    report_parser.add_argument('--date', help='Target date (YYYY-MM-DD)')
    report_parser.add_argument('--days', type=int, help='Period in days (for comprehensive reports)')
    report_parser.add_argument('--operation-id', help='Operation ID (for operation reports)')
    report_parser.add_argument('--format', choices=['text', 'json', 'html'], default='text',
                              help='Output format')
    report_parser.add_argument('--output', help='Output file path')
    report_parser.set_defaults(func=lambda cli, args: cli.generate_report(args))
    
    # Generate recommendations command
    rec_parser = subparsers.add_parser('recommend', help='Generate maintenance recommendations')
    rec_parser.add_argument('--test-coverage', type=float, help='Current test coverage percentage')
    rec_parser.add_argument('--code-complexity', type=float, help='Current code complexity score')
    rec_parser.add_argument('--doc-coverage', type=float, help='Current documentation coverage')
    rec_parser.add_argument('--duplicate-code', type=float, help='Current duplicate code percentage')
    rec_parser.add_argument('--style-violations', type=float, help='Style violations per KLOC')
    rec_parser.set_defaults(func=lambda cli, args: cli.generate_recommendations(args))
    
    # Optimize schedule command
    schedule_parser = subparsers.add_parser('schedule', help='Optimize maintenance schedule')
    schedule_parser.add_argument('--days', type=int, default=30,
                                help='Consider recommendations from last N days')
    schedule_parser.add_argument('--hours-per-week', type=float, default=40,
                                help='Available hours per week')
    schedule_parser.add_argument('--max-concurrent', type=int, default=3,
                                help='Maximum concurrent operations')
    schedule_parser.set_defaults(func=lambda cli, args: cli.optimize_schedule(args))
    
    # Statistics command
    stats_parser = subparsers.add_parser('stats', help='Show maintenance statistics')
    stats_parser.add_argument('--days', type=int, default=30,
                             help='Analysis period in days')
    stats_parser.set_defaults(func=lambda cli, args: cli.show_statistics(args))
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up old maintenance data')
    cleanup_parser.add_argument('--days', type=int, default=90,
                               help='Remove data older than N days')
    cleanup_parser.set_defaults(func=lambda cli, args: cli.cleanup(args))
    
    # Parse arguments and execute
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    cli = MaintenanceReporterCLI()
    args.func(cli, args)


if __name__ == '__main__':
    main()