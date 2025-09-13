"""
Comprehensive maintenance reporting system.
"""

import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import statistics

try:
    from tools.maintenance-reporter.models import (
        MaintenanceReport, MaintenanceOperation, MaintenanceImpactAnalysis,
        MaintenanceRecommendation, MaintenanceScheduleOptimization
    )
    from tools.maintenance-reporter.operation_logger import OperationLogger
    from tools.maintenance-reporter.impact_analyzer import ImpactAnalyzer
    from tools.maintenance-reporter.recommendation_engine import MaintenanceRecommendationEngine
except ImportError:
    from models import (
        MaintenanceReport, MaintenanceOperation, MaintenanceImpactAnalysis,
        MaintenanceRecommendation, MaintenanceScheduleOptimization
    )
    from operation_logger import OperationLogger
    from impact_analyzer import ImpactAnalyzer
    from recommendation_engine import MaintenanceRecommendationEngine


class MaintenanceReportGenerator:
    """Generates comprehensive maintenance reports."""
    
    def __init__(self, data_dir: str = "data/maintenance-reports"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.reports_file = self.data_dir / "reports.json"
        self.reports: Dict[str, MaintenanceReport] = {}
        
        self._load_data()
    
    def _load_data(self):
        """Load existing reports."""
        if self.reports_file.exists():
            try:
                with open(self.reports_file) as f:
                    data = json.load(f)
                
                for report_data in data.get('reports', []):
                    # Load operations
                    operations = []
                    for op_data in report_data.get('operations', []):
                        from tools.maintenance-reporter.models import MaintenanceOperationType, MaintenanceStatus, ImpactLevel
                        operations.append(MaintenanceOperation(
                            id=op_data['id'],
                            operation_type=MaintenanceOperationType(op_data['operation_type']),
                            title=op_data['title'],
                            description=op_data['description'],
                            status=MaintenanceStatus(op_data['status']),
                            impact_level=ImpactLevel(op_data['impact_level']),
                            started_at=datetime.fromisoformat(op_data['started_at']),
                            completed_at=datetime.fromisoformat(op_data['completed_at']) if op_data.get('completed_at') else None,
                            duration_seconds=op_data.get('duration_seconds'),
                            success_metrics=op_data.get('success_metrics', {}),
                            error_details=op_data.get('error_details'),
                            rollback_info=op_data.get('rollback_info'),
                            files_affected=op_data.get('files_affected', []),
                            components_affected=op_data.get('components_affected', [])
                        ))
                    
                    # Create simplified report (without full object reconstruction)
                    report = MaintenanceReport(
                        report_id=report_data['report_id'],
                        report_type=report_data['report_type'],
                        period_start=datetime.fromisoformat(report_data['period_start']),
                        period_end=datetime.fromisoformat(report_data['period_end']),
                        operations=operations,
                        impact_analyses=[],  # Simplified loading
                        recommendations=[],  # Simplified loading
                        schedule_optimization=None,  # Simplified loading
                        summary_statistics=report_data.get('summary_statistics', {}),
                        generated_at=datetime.fromisoformat(report_data['generated_at'])
                    )
                    self.reports[report.report_id] = report
            
            except Exception as e:
                print(f"Error loading reports: {e}")
    
    def _save_data(self):
        """Save reports to storage."""
        data = {
            'reports': [report.to_dict() for report in self.reports.values()],
            'last_updated': datetime.now().isoformat()
        }
        
        with open(self.reports_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def generate_daily_report(
        self,
        operation_logger: OperationLogger,
        impact_analyzer: ImpactAnalyzer,
        target_date: Optional[datetime] = None
    ) -> MaintenanceReport:
        """Generate a daily maintenance report."""
        if target_date is None:
            target_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        period_start = target_date
        period_end = target_date + timedelta(days=1)
        
        return self._generate_report(
            report_type="daily",
            period_start=period_start,
            period_end=period_end,
            operation_logger=operation_logger,
            impact_analyzer=impact_analyzer
        )    

    def generate_weekly_report(
        self,
        operation_logger: OperationLogger,
        impact_analyzer: ImpactAnalyzer,
        target_date: Optional[datetime] = None
    ) -> MaintenanceReport:
        """Generate a weekly maintenance report."""
        if target_date is None:
            target_date = datetime.now()
        
        # Get start of week (Monday)
        days_since_monday = target_date.weekday()
        period_start = target_date - timedelta(days=days_since_monday)
        period_start = period_start.replace(hour=0, minute=0, second=0, microsecond=0)
        period_end = period_start + timedelta(days=7)
        
        return self._generate_report(
            report_type="weekly",
            period_start=period_start,
            period_end=period_end,
            operation_logger=operation_logger,
            impact_analyzer=impact_analyzer
        )
    
    def generate_monthly_report(
        self,
        operation_logger: OperationLogger,
        impact_analyzer: ImpactAnalyzer,
        target_date: Optional[datetime] = None
    ) -> MaintenanceReport:
        """Generate a monthly maintenance report."""
        if target_date is None:
            target_date = datetime.now()
        
        # Get start of month
        period_start = target_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        # Get start of next month
        if period_start.month == 12:
            period_end = period_start.replace(year=period_start.year + 1, month=1)
        else:
            period_end = period_start.replace(month=period_start.month + 1)
        
        return self._generate_report(
            report_type="monthly",
            period_start=period_start,
            period_end=period_end,
            operation_logger=operation_logger,
            impact_analyzer=impact_analyzer
        )
    
    def generate_operation_summary_report(
        self,
        operation_id: str,
        operation_logger: OperationLogger,
        impact_analyzer: ImpactAnalyzer
    ) -> MaintenanceReport:
        """Generate a detailed report for a specific operation."""
        operation = operation_logger.get_operation(operation_id)
        if not operation:
            raise ValueError(f"Operation {operation_id} not found")
        
        # Create a report covering just this operation
        period_start = operation.started_at
        period_end = operation.completed_at or datetime.now()
        
        return self._generate_report(
            report_type="operation_summary",
            period_start=period_start,
            period_end=period_end,
            operation_logger=operation_logger,
            impact_analyzer=impact_analyzer,
            specific_operation_id=operation_id
        )
    
    def generate_comprehensive_report(
        self,
        operation_logger: OperationLogger,
        impact_analyzer: ImpactAnalyzer,
        recommendation_engine: MaintenanceRecommendationEngine,
        period_days: int = 30
    ) -> MaintenanceReport:
        """Generate a comprehensive maintenance report with all components."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_days)
        
        return self._generate_report(
            report_type="comprehensive",
            period_start=start_date,
            period_end=end_date,
            operation_logger=operation_logger,
            impact_analyzer=impact_analyzer,
            recommendation_engine=recommendation_engine
        )
    
    def _generate_report(
        self,
        report_type: str,
        period_start: datetime,
        period_end: datetime,
        operation_logger: OperationLogger,
        impact_analyzer: ImpactAnalyzer,
        recommendation_engine: Optional[MaintenanceRecommendationEngine] = None,
        specific_operation_id: Optional[str] = None
    ) -> MaintenanceReport:
        """Generate a comprehensive maintenance report."""
        report_id = str(uuid.uuid4())
        
        # Get operations for the period
        if specific_operation_id:
            operation = operation_logger.get_operation(specific_operation_id)
            operations = [operation] if operation else []
        else:
            operations = operation_logger.get_operations_in_period(period_start, period_end)
        
        # Get impact analyses for the period
        impact_analyses = impact_analyzer.get_impact_analyses_in_period(period_start, period_end)
        
        # Filter impact analyses to match operations if specific operation requested
        if specific_operation_id:
            impact_analyses = [ia for ia in impact_analyses if ia.operation_id == specific_operation_id]
        
        # Get recommendations if recommendation engine provided
        recommendations = []
        schedule_optimization = None
        if recommendation_engine:
            recommendations = recommendation_engine.get_active_recommendations()
            if recommendations:
                schedule_optimization = recommendation_engine.optimize_maintenance_schedule(recommendations)
        
        # Generate summary statistics
        summary_statistics = self._generate_summary_statistics(
            operations, impact_analyses, recommendations, period_start, period_end
        )
        
        # Create the report
        report = MaintenanceReport(
            report_id=report_id,
            report_type=report_type,
            period_start=period_start,
            period_end=period_end,
            operations=operations,
            impact_analyses=impact_analyses,
            recommendations=recommendations,
            schedule_optimization=schedule_optimization,
            summary_statistics=summary_statistics
        )
        
        # Store the report
        self.reports[report_id] = report
        self._save_data()
        
        return report
    
    def _generate_summary_statistics(
        self,
        operations: List[MaintenanceOperation],
        impact_analyses: List[MaintenanceImpactAnalysis],
        recommendations: List[MaintenanceRecommendation],
        period_start: datetime,
        period_end: datetime
    ) -> Dict[str, Any]:
        """Generate comprehensive summary statistics."""
        total_operations = len(operations)
        
        if total_operations == 0:
            return {
                'period': {
                    'start': period_start.isoformat(),
                    'end': period_end.isoformat(),
                    'duration_days': (period_end - period_start).days
                },
                'operations': {
                    'total': 0,
                    'completed': 0,
                    'failed': 0,
                    'in_progress': 0,
                    'success_rate': 0
                },
                'impact': {
                    'total_analyses': 0,
                    'average_impact_score': 0,
                    'positive_impacts': 0,
                    'negative_impacts': 0
                },
                'recommendations': {
                    'total': len(recommendations),
                    'by_priority': {}
                }
            }
        
        # Operation statistics
        completed_ops = [op for op in operations if op.status.value == 'completed']
        failed_ops = [op for op in operations if op.status.value == 'failed']
        in_progress_ops = [op for op in operations if op.status.value == 'in_progress']
        
        success_rate = (len(completed_ops) / total_operations * 100) if total_operations > 0 else 0
        
        # Duration statistics
        duration_stats = {}
        if completed_ops:
            durations = [op.duration_seconds / 60 for op in completed_ops if op.duration_seconds]
            if durations:
                duration_stats = {
                    'average_minutes': statistics.mean(durations),
                    'median_minutes': statistics.median(durations),
                    'min_minutes': min(durations),
                    'max_minutes': max(durations),
                    'total_hours': sum(durations) / 60
                }
        
        # Operation type breakdown
        type_breakdown = {}
        for operation in operations:
            op_type = operation.operation_type.value
            if op_type not in type_breakdown:
                type_breakdown[op_type] = {
                    'total': 0,
                    'completed': 0,
                    'failed': 0,
                    'success_rate': 0
                }
            
            type_breakdown[op_type]['total'] += 1
            
            if operation.status.value == 'completed':
                type_breakdown[op_type]['completed'] += 1
            elif operation.status.value == 'failed':
                type_breakdown[op_type]['failed'] += 1
        
        # Calculate success rates per type
        for op_type, stats in type_breakdown.items():
            if stats['total'] > 0:
                stats['success_rate'] = (stats['completed'] / stats['total'] * 100)
        
        # Impact analysis statistics
        impact_stats = {}
        if impact_analyses:
            impact_scores = [ia.overall_impact_score for ia in impact_analyses]
            positive_impacts = [score for score in impact_scores if score > 0]
            negative_impacts = [score for score in impact_scores if score < 0]
            
            impact_stats = {
                'total_analyses': len(impact_analyses),
                'average_impact_score': statistics.mean(impact_scores),
                'median_impact_score': statistics.median(impact_scores),
                'positive_impacts': len(positive_impacts),
                'negative_impacts': len(negative_impacts),
                'neutral_impacts': len([score for score in impact_scores if score == 0]),
                'best_impact_score': max(impact_scores),
                'worst_impact_score': min(impact_scores)
            }
            
            # Top improvements and regressions
            all_improvements = {}
            all_regressions = {}
            
            for analysis in impact_analyses:
                for metric, value in analysis.improvements.items():
                    if metric not in all_improvements:
                        all_improvements[metric] = []
                    all_improvements[metric].append(value)
                
                for metric, value in analysis.regressions.items():
                    if metric not in all_regressions:
                        all_regressions[metric] = []
                    all_regressions[metric].append(value)
            
            # Calculate average improvements and regressions
            impact_stats['top_improvements'] = {
                metric: statistics.mean(values)
                for metric, values in all_improvements.items()
            }
            
            impact_stats['top_regressions'] = {
                metric: statistics.mean(values)
                for metric, values in all_regressions.items()
            }
        
        # Recommendation statistics
        rec_stats = {
            'total': len(recommendations),
            'by_priority': {},
            'by_operation_type': {},
            'estimated_total_effort_hours': sum(r.estimated_effort_hours for r in recommendations),
            'estimated_total_impact': sum(r.estimated_impact_score for r in recommendations)
        }
        
        # Group recommendations by priority
        for rec in recommendations:
            priority = rec.priority.value
            rec_stats['by_priority'][priority] = rec_stats['by_priority'].get(priority, 0) + 1
            
            op_type = rec.operation_type.value
            rec_stats['by_operation_type'][op_type] = rec_stats['by_operation_type'].get(op_type, 0) + 1
        
        # Files and components affected
        all_files_affected = set()
        all_components_affected = set()
        
        for operation in operations:
            all_files_affected.update(operation.files_affected)
            all_components_affected.update(operation.components_affected)
        
        return {
            'period': {
                'start': period_start.isoformat(),
                'end': period_end.isoformat(),
                'duration_days': (period_end - period_start).days
            },
            'operations': {
                'total': total_operations,
                'completed': len(completed_ops),
                'failed': len(failed_ops),
                'in_progress': len(in_progress_ops),
                'success_rate': success_rate,
                'duration_statistics': duration_stats,
                'by_type': type_breakdown
            },
            'impact': impact_stats,
            'recommendations': rec_stats,
            'affected_resources': {
                'files_count': len(all_files_affected),
                'components_count': len(all_components_affected),
                'files': list(all_files_affected)[:20],  # Limit to first 20 for readability
                'components': list(all_components_affected)
            },
            'generated_at': datetime.now().isoformat()
        }
    
    def generate_comprehensive_report(
        self,
        operation_logger: OperationLogger,
        impact_analyzer: ImpactAnalyzer,
        recommendation_engine: MaintenanceRecommendationEngine,
        period_days: int = 30
    ) -> MaintenanceReport:
        """Generate a comprehensive maintenance report with all components."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_days)
        
        return self._generate_report(
            report_type="comprehensive",
            period_start=start_date,
            period_end=end_date,
            operation_logger=operation_logger,
            impact_analyzer=impact_analyzer,
            recommendation_engine=recommendation_engine
        )
    
    def export_report_to_html(self, report: MaintenanceReport, output_path: Optional[Path] = None) -> Path:
        """Export a maintenance report to HTML format."""
        if output_path is None:
            output_path = self.data_dir / f"report_{report.report_id}.html"
        
        html_content = self._generate_html_report(report)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return output_path
    
    def _generate_html_report(self, report: MaintenanceReport) -> str:
        """Generate HTML content for a maintenance report."""
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Maintenance Report - {report.report_type.title()}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
        .header {{ background-color: #f4f4f4; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
        .section {{ margin-bottom: 30px; }}
        .section h2 {{ color: #333; border-bottom: 2px solid #007acc; padding-bottom: 5px; }}
        .section h3 {{ color: #555; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
        .stat-card {{ background: #f9f9f9; padding: 15px; border-radius: 5px; text-align: center; }}
        .stat-value {{ font-size: 2em; font-weight: bold; color: #007acc; }}
        .stat-label {{ color: #666; font-size: 0.9em; }}
        .operation-list {{ margin: 15px 0; }}
        .operation-item {{ background: #fff; border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .operation-header {{ font-weight: bold; color: #333; }}
        .operation-status {{ padding: 3px 8px; border-radius: 3px; font-size: 0.8em; }}
        .status-completed {{ background-color: #d4edda; color: #155724; }}
        .status-failed {{ background-color: #f8d7da; color: #721c24; }}
        .status-in-progress {{ background-color: #fff3cd; color: #856404; }}
        .recommendations {{ background: #e7f3ff; padding: 15px; border-radius: 5px; }}
        .recommendation-item {{ margin: 10px 0; padding: 10px; background: white; border-radius: 3px; }}
        .priority-high {{ border-left: 4px solid #dc3545; }}
        .priority-medium {{ border-left: 4px solid #ffc107; }}
        .priority-low {{ border-left: 4px solid #28a745; }}
        .impact-positive {{ color: #28a745; }}
        .impact-negative {{ color: #dc3545; }}
        .impact-neutral {{ color: #6c757d; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f4f4f4; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Maintenance Report - {report.report_type.title()}</h1>
        <p><strong>Period:</strong> {report.period_start.strftime('%Y-%m-%d %H:%M')} to {report.period_end.strftime('%Y-%m-%d %H:%M')}</p>
        <p><strong>Generated:</strong> {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Report ID:</strong> {report.report_id}</p>
    </div>
"""
        
        # Summary statistics
        stats = report.summary_statistics
        if stats:
            html += """
    <div class="section">
        <h2>Summary Statistics</h2>
        <div class="stats-grid">
"""
            
            # Operation stats
            op_stats = stats.get('operations', {})
            html += f"""
            <div class="stat-card">
                <div class="stat-value">{op_stats.get('total', 0)}</div>
                <div class="stat-label">Total Operations</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{op_stats.get('success_rate', 0):.1f}%</div>
                <div class="stat-label">Success Rate</div>
            </div>
"""
            
            # Impact stats
            impact_stats = stats.get('impact', {})
            if impact_stats:
                html += f"""
            <div class="stat-card">
                <div class="stat-value">{impact_stats.get('average_impact_score', 0):.1f}</div>
                <div class="stat-label">Average Impact Score</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{impact_stats.get('positive_impacts', 0)}</div>
                <div class="stat-label">Positive Impacts</div>
            </div>
"""
            
            # Recommendation stats
            rec_stats = stats.get('recommendations', {})
            html += f"""
            <div class="stat-card">
                <div class="stat-value">{rec_stats.get('total', 0)}</div>
                <div class="stat-label">Active Recommendations</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{rec_stats.get('estimated_total_effort_hours', 0):.1f}h</div>
                <div class="stat-label">Estimated Effort</div>
            </div>
"""
            
            html += """
        </div>
    </div>
"""
        
        # Operations section
        if report.operations:
            html += """
    <div class="section">
        <h2>Operations</h2>
        <div class="operation-list">
"""
            
            for operation in report.operations:
                status_class = f"status-{operation.status.value.replace('_', '-')}"
                duration_text = ""
                if operation.duration_seconds:
                    duration_text = f" ({operation.duration_seconds // 60} minutes)"
                
                html += f"""
            <div class="operation-item">
                <div class="operation-header">
                    {operation.title}
                    <span class="operation-status {status_class}">{operation.status.value.replace('_', ' ').title()}</span>
                </div>
                <p><strong>Type:</strong> {operation.operation_type.value.replace('_', ' ').title()}</p>
                <p><strong>Impact Level:</strong> {operation.impact_level.value.title()}</p>
                <p><strong>Started:</strong> {operation.started_at.strftime('%Y-%m-%d %H:%M')}{duration_text}</p>
                <p>{operation.description}</p>
"""
                
                if operation.files_affected:
                    html += f"<p><strong>Files Affected:</strong> {len(operation.files_affected)} files</p>"
                
                if operation.components_affected:
                    html += f"<p><strong>Components Affected:</strong> {', '.join(operation.components_affected)}</p>"
                
                if operation.error_details:
                    html += f"<p><strong>Error:</strong> {operation.error_details}</p>"
                
                html += "</div>"
            
            html += """
        </div>
    </div>
"""
        
        # Impact analyses section
        if report.impact_analyses:
            html += """
    <div class="section">
        <h2>Impact Analyses</h2>
        <table>
            <thead>
                <tr>
                    <th>Operation</th>
                    <th>Impact Score</th>
                    <th>Summary</th>
                    <th>Key Changes</th>
                </tr>
            </thead>
            <tbody>
"""
            
            for analysis in report.impact_analyses:
                # Find corresponding operation
                operation = next((op for op in report.operations if op.id == analysis.operation_id), None)
                operation_title = operation.title if operation else analysis.operation_id
                
                impact_class = "impact-positive" if analysis.overall_impact_score > 0 else "impact-negative" if analysis.overall_impact_score < 0 else "impact-neutral"
                
                # Get top improvements and regressions
                top_improvements = sorted(analysis.improvements.items(), key=lambda x: x[1], reverse=True)[:3]
                top_regressions = sorted(analysis.regressions.items(), key=lambda x: x[1], reverse=True)[:3]
                
                changes = []
                for metric, value in top_improvements:
                    changes.append(f"↑ {metric.replace('_', ' ')}: +{value:.1f}%")
                for metric, value in top_regressions:
                    changes.append(f"↓ {metric.replace('_', ' ')}: -{value:.1f}%")
                
                html += f"""
                <tr>
                    <td>{operation_title}</td>
                    <td class="{impact_class}">{analysis.overall_impact_score:.1f}</td>
                    <td>{analysis.impact_summary}</td>
                    <td>{'<br>'.join(changes) if changes else 'No significant changes'}</td>
                </tr>
"""
            
            html += """
            </tbody>
        </table>
    </div>
"""
        
        # Recommendations section
        if report.recommendations:
            html += """
    <div class="section">
        <h2>Recommendations</h2>
        <div class="recommendations">
"""
            
            for rec in report.recommendations:
                priority_class = f"priority-{rec.priority.value}"
                
                html += f"""
            <div class="recommendation-item {priority_class}">
                <h3>{rec.title}</h3>
                <p><strong>Priority:</strong> {rec.priority.value.title()} | 
                   <strong>Type:</strong> {rec.operation_type.value.replace('_', ' ').title()} | 
                   <strong>Effort:</strong> {rec.estimated_effort_hours:.1f}h | 
                   <strong>Impact:</strong> {rec.estimated_impact_score:.1f}</p>
                <p>{rec.description}</p>
"""
                
                if rec.benefits:
                    html += f"<p><strong>Benefits:</strong> {', '.join(rec.benefits)}</p>"
                
                if rec.risks:
                    html += f"<p><strong>Risks:</strong> {', '.join(rec.risks)}</p>"
                
                if rec.suggested_schedule:
                    html += f"<p><strong>Suggested Schedule:</strong> {rec.suggested_schedule}</p>"
                
                html += "</div>"
            
            html += """
        </div>
    </div>
"""
        
        # Schedule optimization section
        if report.schedule_optimization:
            schedule = report.schedule_optimization
            html += f"""
    <div class="section">
        <h2>Schedule Optimization</h2>
        <p><strong>Estimated Total Duration:</strong> {schedule.estimated_total_duration_hours:.1f} hours</p>
        <p><strong>Resource Requirements:</strong></p>
        <ul>
"""
            
            for resource, hours in schedule.resource_requirements.items():
                html += f"<li>{resource.replace('_', ' ').title()}: {hours:.1f} hours</li>"
            
            html += """
        </ul>
        <p><strong>Risk Mitigation Plan:</strong></p>
        <ul>
"""
            
            for risk in schedule.risk_mitigation_plan:
                html += f"<li>{risk}</li>"
            
            html += """
        </ul>
    </div>
"""
        
        html += """
</body>
</html>
"""
        
        return html
    
    def get_report(self, report_id: str) -> Optional[MaintenanceReport]:
        """Get a specific report by ID."""
        return self.reports.get(report_id)
    
    def get_reports_by_type(self, report_type: str) -> List[MaintenanceReport]:
        """Get all reports of a specific type."""
        return [report for report in self.reports.values() if report.report_type == report_type]
    
    def get_recent_reports(self, days: int = 30) -> List[MaintenanceReport]:
        """Get reports generated in the last N days."""
        cutoff_date = datetime.now() - timedelta(days=days)
        return [
            report for report in self.reports.values()
            if report.generated_at > cutoff_date
        ]
    
    def cleanup_old_reports(self, days: int = 90) -> int:
        """Clean up old reports."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        old_reports = [
            report_id for report_id, report in self.reports.items()
            if report.generated_at < cutoff_date
        ]
        
        for report_id in old_reports:
            del self.reports[report_id]
        
        if old_reports:
            self._save_data()
        
        return len(old_reports)
