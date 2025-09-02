#!/usr/bin/env python3
"""
Example usage of the comprehensive maintenance reporting system.
Demonstrates all major features and workflows.
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path

from operation_logger import OperationLogger
from impact_analyzer import ImpactAnalyzer
from recommendation_engine import MaintenanceRecommendationEngine
from report_generator import MaintenanceReportGenerator
from models import MaintenanceOperationType, ImpactLevel


def demonstrate_operation_lifecycle():
    """Demonstrate complete operation lifecycle with audit trails."""
    print("=== Operation Lifecycle Demonstration ===")
    
    logger = OperationLogger()
    impact_analyzer = ImpactAnalyzer()
    
    # Start a test repair operation
    print("1. Starting test repair operation...")
    operation_id = logger.start_operation(
        operation_type=MaintenanceOperationType.TEST_REPAIR,
        title="Fix broken unit tests",
        description="Repair test imports and update fixtures after recent refactoring",
        impact_level=ImpactLevel.MEDIUM,
        files_affected=["tests/test_models.py", "tests/fixtures/test_data.py"],
        components_affected=["test_suite", "fixtures", "models"]
    )
    print(f"   Started operation: {operation_id}")
    
    # Simulate some progress updates
    print("2. Updating operation progress...")
    logger.update_operation_progress(operation_id, {
        "tests_analyzed": 25,
        "broken_imports_found": 8,
        "fixtures_updated": 3
    })
    
    # Simulate work time
    time.sleep(1)
    
    # Complete the operation with success metrics
    print("3. Completing operation with success metrics...")
    success_metrics = {
        "tests_fixed": 8,
        "fixtures_updated": 3,
        "coverage_improvement": 2.5,
        "execution_time_improvement": 15.2
    }
    
    logger.complete_operation(
        operation_id=operation_id,
        success_metrics=success_metrics,
        files_affected=["tests/conftest.py"],  # Additional file modified
        components_affected=["test_configuration"]  # Additional component
    )
    
    # Perform impact analysis
    print("4. Analyzing operation impact...")
    before_metrics = {
        "test_coverage": 72.5,
        "failing_tests": 8,
        "test_execution_time": 45.2,
        "flaky_tests": 3
    }
    
    after_metrics = {
        "test_coverage": 75.0,
        "failing_tests": 0,
        "test_execution_time": 38.3,
        "flaky_tests": 1
    }
    
    impact_analysis = impact_analyzer.analyze_operation_impact(
        operation_id, before_metrics, after_metrics
    )
    
    print(f"   Impact Score: {impact_analysis.overall_impact_score:.1f}")
    print(f"   Summary: {impact_analysis.impact_summary}")
    
    # Show audit trail
    print("5. Audit trail for operation:")
    audit_trail = logger.get_audit_trail_for_operation(operation_id)
    for entry in audit_trail:
        print(f"   {entry.timestamp.strftime('%H:%M:%S')} - {entry.action} by {entry.user}")
    
    return operation_id, impact_analysis


def demonstrate_failed_operation_with_rollback():
    """Demonstrate handling of failed operations with rollback."""
    print("\n=== Failed Operation with Rollback Demonstration ===")
    
    logger = OperationLogger()
    
    # Start a configuration consolidation operation
    print("1. Starting configuration consolidation operation...")
    operation_id = logger.start_operation(
        operation_type=MaintenanceOperationType.CONFIGURATION_CONSOLIDATION,
        title="Consolidate scattered config files",
        description="Merge multiple config files into unified configuration system",
        impact_level=ImpactLevel.HIGH,
        files_affected=["config/app.json", "config/db.yaml", "config/cache.ini"],
        components_affected=["configuration", "database", "cache"]
    )
    print(f"   Started operation: {operation_id}")
    
    # Simulate failure
    print("2. Operation encounters critical error...")
    rollback_info = {
        "backup_location": "/backups/config_20240101_143022",
        "rollback_steps": [
            "Restore original config files from backup",
            "Restart application services",
            "Verify service connectivity",
            "Clear configuration cache"
        ],
        "affected_services": ["web_server", "database", "cache_server"],
        "rollback_completed": True,
        "rollback_duration_seconds": 180
    }
    
    logger.fail_operation(
        operation_id=operation_id,
        error_details="Configuration validation failed: circular dependency detected between database and cache configurations",
        rollback_info=rollback_info
    )
    
    print("3. Operation failed and rolled back successfully")
    
    # Show the failed operation details
    operation = logger.get_operation(operation_id)
    print(f"   Status: {operation.status.value}")
    print(f"   Error: {operation.error_details}")
    print(f"   Rollback completed: {operation.rollback_info['rollback_completed']}")
    
    return operation_id


def demonstrate_recommendation_generation():
    """Demonstrate intelligent recommendation generation."""
    print("\n=== Recommendation Generation Demonstration ===")
    
    logger = OperationLogger()
    impact_analyzer = ImpactAnalyzer()
    recommendation_engine = MaintenanceRecommendationEngine()
    
    # Simulate current project metrics
    print("1. Analyzing current project metrics...")
    project_metrics = {
        "test_coverage": 68.5,  # Below recommended 80%
        "code_complexity": 12.8,  # Above recommended 10
        "documentation_coverage": 45.2,  # Below recommended 60%
        "duplicate_code": 18.7,  # Above recommended 15%
        "style_violations": 67.3,  # Above recommended 50
        "performance_score": 78.2,
        "security_score": 85.1
    }
    
    for metric, value in project_metrics.items():
        print(f"   {metric.replace('_', ' ').title()}: {value}")
    
    # Generate recommendations
    print("2. Generating maintenance recommendations...")
    recommendations = recommendation_engine.generate_recommendations_from_analysis(
        logger, impact_analyzer, project_metrics
    )
    
    print(f"   Generated {len(recommendations)} recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec.title}")
        print(f"      Priority: {rec.priority.value}")
        print(f"      Effort: {rec.estimated_effort_hours:.1f} hours")
        print(f"      Impact: {rec.estimated_impact_score:.1f}")
        print(f"      Type: {rec.operation_type.value.replace('_', ' ').title()}")
        if rec.suggested_schedule:
            print(f"      Schedule: {rec.suggested_schedule}")
        print()
    
    return recommendations


def demonstrate_schedule_optimization(recommendations):
    """Demonstrate maintenance schedule optimization."""
    print("=== Schedule Optimization Demonstration ===")
    
    recommendation_engine = MaintenanceRecommendationEngine()
    
    print("1. Optimizing maintenance schedule...")
    print(f"   Input: {len(recommendations)} recommendations")
    print("   Constraints: 40 hours/week, max 3 concurrent operations")
    
    schedule = recommendation_engine.optimize_maintenance_schedule(
        recommendations,
        available_hours_per_week=40,
        max_concurrent_operations=3
    )
    
    print("2. Optimized Schedule:")
    print(f"   Total Duration: {schedule.estimated_total_duration_hours:.1f} hours")
    print(f"   Estimated Weeks: {schedule.estimated_total_duration_hours / 40:.1f}")
    
    print("\n   Resource Requirements:")
    for resource, hours in schedule.resource_requirements.items():
        print(f"     {resource.replace('_', ' ').title()}: {hours:.1f} hours")
    
    print("\n   Recommended Execution Order:")
    for i, op_id in enumerate(schedule.recommended_schedule, 1):
        rec = next((r for r in recommendations if r.id == op_id), None)
        if rec:
            rationale = schedule.scheduling_rationale.get(op_id, "")
            print(f"     {i}. {rec.title}")
            print(f"        Rationale: {rationale}")
    
    print("\n   Risk Mitigation Plan:")
    for risk in schedule.risk_mitigation_plan:
        print(f"     • {risk}")
    
    return schedule


def demonstrate_comprehensive_reporting():
    """Demonstrate comprehensive report generation."""
    print("\n=== Comprehensive Reporting Demonstration ===")
    
    logger = OperationLogger()
    impact_analyzer = ImpactAnalyzer()
    recommendation_engine = MaintenanceRecommendationEngine()
    report_generator = MaintenanceReportGenerator()
    
    print("1. Generating comprehensive maintenance report...")
    
    # Generate a comprehensive report for the last 30 days
    report = report_generator.generate_comprehensive_report(
        operation_logger=logger,
        impact_analyzer=impact_analyzer,
        recommendation_engine=recommendation_engine,
        period_days=30
    )
    
    print(f"   Report ID: {report.report_id}")
    print(f"   Report Type: {report.report_type}")
    print(f"   Period: {report.period_start.strftime('%Y-%m-%d')} to {report.period_end.strftime('%Y-%m-%d')}")
    
    # Show summary statistics
    stats = report.summary_statistics
    if stats:
        op_stats = stats.get('operations', {})
        print(f"   Total Operations: {op_stats.get('total', 0)}")
        print(f"   Success Rate: {op_stats.get('success_rate', 0):.1f}%")
        
        impact_stats = stats.get('impact', {})
        if impact_stats:
            print(f"   Average Impact Score: {impact_stats.get('average_impact_score', 0):.1f}")
        
        rec_stats = stats.get('recommendations', {})
        print(f"   Active Recommendations: {rec_stats.get('total', 0)}")
    
    # Export to HTML
    print("2. Exporting report to HTML...")
    html_path = report_generator.export_report_to_html(report)
    print(f"   HTML report saved to: {html_path}")
    
    # Export to JSON
    print("3. Exporting report to JSON...")
    json_path = Path("data/maintenance-reports") / f"report_{report.report_id}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(json_path, 'w') as f:
        f.write(report.to_json())
    print(f"   JSON report saved to: {json_path}")
    
    return report


def demonstrate_statistics_and_analysis():
    """Demonstrate statistics and improvement opportunity analysis."""
    print("\n=== Statistics and Analysis Demonstration ===")
    
    logger = OperationLogger()
    impact_analyzer = ImpactAnalyzer()
    
    print("1. Operation Statistics:")
    stats = logger.get_operation_statistics()
    
    print(f"   Total Operations: {stats['total_operations']}")
    print(f"   Success Rate: {stats['success_rate']:.1f}%")
    print(f"   Average Duration: {stats['average_duration_minutes']:.1f} minutes")
    
    print("\n   By Status:")
    for status, count in stats['by_status'].items():
        print(f"     {status.replace('_', ' ').title()}: {count}")
    
    print("\n   By Type:")
    for op_type, count in stats['by_type'].items():
        if count > 0:
            print(f"     {op_type.replace('_', ' ').title()}: {count}")
    
    # Improvement opportunities
    print("\n2. Improvement Opportunities:")
    opportunities = impact_analyzer.identify_improvement_opportunities(logger, days=30)
    
    if opportunities:
        for opp in opportunities:
            print(f"   • {opp['recommendation']}")
            print(f"     Type: {opp['type']}")
            print(f"     Priority: {opp['priority']}")
            if 'failure_rate' in opp:
                print(f"     Failure Rate: {opp['failure_rate']:.1f}%")
            print()
    else:
        print("   No improvement opportunities identified")


def demonstrate_data_cleanup():
    """Demonstrate data cleanup and maintenance."""
    print("\n=== Data Cleanup Demonstration ===")
    
    logger = OperationLogger()
    impact_analyzer = ImpactAnalyzer()
    recommendation_engine = MaintenanceRecommendationEngine()
    report_generator = MaintenanceReportGenerator()
    
    print("1. Current data status:")
    stats = logger.get_operation_statistics()
    print(f"   Operations: {stats['total_operations']}")
    print(f"   Audit Entries: {stats['total_audit_entries']}")
    
    print("2. Performing cleanup (simulated - using 1 day for demo)...")
    
    # Cleanup old data (using 1 day for demonstration)
    operations_cleaned = logger.cleanup_old_operations(days=1)
    analyses_cleaned = impact_analyzer.cleanup_old_analyses(days=1)
    recommendations_cleaned = recommendation_engine.cleanup_old_recommendations(days=1)
    reports_cleaned = report_generator.cleanup_old_reports(days=1)
    
    total_cleaned = operations_cleaned + analyses_cleaned + recommendations_cleaned + reports_cleaned
    
    print(f"   Operations cleaned: {operations_cleaned}")
    print(f"   Impact analyses cleaned: {analyses_cleaned}")
    print(f"   Recommendations cleaned: {recommendations_cleaned}")
    print(f"   Reports cleaned: {reports_cleaned}")
    print(f"   Total items cleaned: {total_cleaned}")


def main():
    """Run comprehensive demonstration of the maintenance reporting system."""
    print("Comprehensive Maintenance Reporting System Demonstration")
    print("=" * 60)
    
    try:
        # Ensure data directories exist
        Path("data/maintenance-operations").mkdir(parents=True, exist_ok=True)
        Path("data/maintenance-impact").mkdir(parents=True, exist_ok=True)
        Path("data/maintenance-recommendations").mkdir(parents=True, exist_ok=True)
        Path("data/maintenance-reports").mkdir(parents=True, exist_ok=True)
        
        # Demonstrate core features
        operation_id, impact_analysis = demonstrate_operation_lifecycle()
        failed_operation_id = demonstrate_failed_operation_with_rollback()
        recommendations = demonstrate_recommendation_generation()
        schedule = demonstrate_schedule_optimization(recommendations)
        report = demonstrate_comprehensive_reporting()
        demonstrate_statistics_and_analysis()
        demonstrate_data_cleanup()
        
        print("\n" + "=" * 60)
        print("Demonstration completed successfully!")
        print("\nKey artifacts generated:")
        print(f"  • Successful operation: {operation_id}")
        print(f"  • Failed operation: {failed_operation_id}")
        print(f"  • Recommendations: {len(recommendations)} generated")
        print(f"  • Comprehensive report: {report.report_id}")
        print(f"  • HTML report: data/maintenance-reports/report_{report.report_id}.html")
        print(f"  • JSON report: data/maintenance-reports/report_{report.report_id}.json")
        
        print("\nNext steps:")
        print("  1. Review the generated HTML report in your browser")
        print("  2. Examine the JSON data for integration possibilities")
        print("  3. Try the CLI tool: python cli.py --help")
        print("  4. Integrate with your existing maintenance workflows")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()