"""
Maintenance impact analysis and success metrics reporting.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import statistics

try:
    from tools.maintenance-reporter.models import MaintenanceImpactAnalysis, MaintenanceOperation
    from tools.maintenance-reporter.operation_logger import OperationLogger
except ImportError:
    from models import MaintenanceImpactAnalysis, MaintenanceOperation
    from operation_logger import OperationLogger


class ImpactAnalyzer:
    """Analyzes the impact and success of maintenance operations."""
    
    def __init__(self, data_dir: str = "data/maintenance-impact"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.impact_analyses_file = self.data_dir / "impact_analyses.json"
        self.impact_analyses: Dict[str, MaintenanceImpactAnalysis] = {}
        
        self._load_data()
    
    def _load_data(self):
        """Load existing impact analyses."""
        if self.impact_analyses_file.exists():
            try:
                with open(self.impact_analyses_file) as f:
                    data = json.load(f)
                
                for analysis_data in data.get('analyses', []):
                    analysis = MaintenanceImpactAnalysis(
                        operation_id=analysis_data['operation_id'],
                        before_metrics=analysis_data['before_metrics'],
                        after_metrics=analysis_data['after_metrics'],
                        improvements=analysis_data['improvements'],
                        regressions=analysis_data['regressions'],
                        overall_impact_score=analysis_data['overall_impact_score'],
                        impact_summary=analysis_data['impact_summary'],
                        timestamp=datetime.fromisoformat(analysis_data['timestamp'])
                    )
                    self.impact_analyses[analysis.operation_id] = analysis
            
            except Exception as e:
                print(f"Error loading impact analyses: {e}")
    
    def _save_data(self):
        """Save impact analyses to storage."""
        data = {
            'analyses': [analysis.to_dict() for analysis in self.impact_analyses.values()],
            'last_updated': datetime.now().isoformat()
        }
        
        with open(self.impact_analyses_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def analyze_operation_impact(
        self,
        operation_id: str,
        before_metrics: Dict[str, float],
        after_metrics: Dict[str, float]
    ) -> MaintenanceImpactAnalysis:
        """Analyze the impact of a maintenance operation."""
        improvements = {}
        regressions = {}
        
        # Compare metrics to identify improvements and regressions
        for metric_name in set(before_metrics.keys()) | set(after_metrics.keys()):
            before_value = before_metrics.get(metric_name, 0.0)
            after_value = after_metrics.get(metric_name, 0.0)
            
            change = after_value - before_value
            change_percentage = (change / before_value * 100) if before_value != 0 else 0
            
            # Determine if this is an improvement or regression based on metric type
            is_improvement = self._is_metric_improvement(metric_name, change)
            
            if abs(change_percentage) > 1.0:  # Only consider significant changes
                if is_improvement:
                    improvements[metric_name] = change_percentage
                else:
                    regressions[metric_name] = abs(change_percentage)
        
        # Calculate overall impact score
        overall_impact_score = self._calculate_overall_impact_score(improvements, regressions)
        
        # Generate impact summary
        impact_summary = self._generate_impact_summary(improvements, regressions, overall_impact_score)
        
        analysis = MaintenanceImpactAnalysis(
            operation_id=operation_id,
            before_metrics=before_metrics,
            after_metrics=after_metrics,
            improvements=improvements,
            regressions=regressions,
            overall_impact_score=overall_impact_score,
            impact_summary=impact_summary
        )
        
        self.impact_analyses[operation_id] = analysis
        self._save_data()
        
        return analysis
    
    def _is_metric_improvement(self, metric_name: str, change: float) -> bool:
        """Determine if a metric change is an improvement."""
        # Metrics where higher values are better
        higher_is_better = [
            'test_coverage', 'documentation_coverage', 'type_hint_coverage',
            'performance_score', 'reliability_score', 'maintainability_score'
        ]
        
        # Metrics where lower values are better
        lower_is_better = [
            'code_complexity', 'duplicate_code', 'style_violations',
            'error_rate', 'build_time', 'response_time', 'failing_tests',
            'flaky_tests', 'test_execution_time'
        ]
        
        metric_lower = metric_name.lower()
        
        if any(good_metric in metric_lower for good_metric in higher_is_better):
            return change > 0
        elif any(bad_metric in metric_lower for bad_metric in lower_is_better):
            return change < 0
        else:
            # Default: assume higher is better
            return change > 0
    
    def _calculate_overall_impact_score(
        self,
        improvements: Dict[str, float],
        regressions: Dict[str, float]
    ) -> float:
        """Calculate an overall impact score from -100 to +100."""
        improvement_score = sum(improvements.values()) if improvements else 0
        regression_score = sum(regressions.values()) if regressions else 0
        
        # Weight improvements and regressions
        net_score = improvement_score - (regression_score * 1.5)  # Regressions weighted more heavily
        
        # Normalize to -100 to +100 scale
        return max(-100, min(100, net_score))
    
    def _generate_impact_summary(
        self,
        improvements: Dict[str, float],
        regressions: Dict[str, float],
        overall_score: float
    ) -> str:
        """Generate a human-readable impact summary."""
        summary_parts = []
        
        if overall_score > 20:
            summary_parts.append("Significant positive impact")
        elif overall_score > 5:
            summary_parts.append("Moderate positive impact")
        elif overall_score > -5:
            summary_parts.append("Minimal impact")
        elif overall_score > -20:
            summary_parts.append("Moderate negative impact")
        else:
            summary_parts.append("Significant negative impact")
        
        if improvements:
            top_improvements = sorted(improvements.items(), key=lambda x: x[1], reverse=True)[:3]
            improvement_text = ", ".join([f"{metric.replace('_', ' ')}: +{value:.1f}%" 
                                        for metric, value in top_improvements])
            summary_parts.append(f"Key improvements: {improvement_text}")
        
        if regressions:
            top_regressions = sorted(regressions.items(), key=lambda x: x[1], reverse=True)[:3]
            regression_text = ", ".join([f"{metric.replace('_', ' ')}: -{value:.1f}%" 
                                       for metric, value in top_regressions])
            summary_parts.append(f"Areas of concern: {regression_text}")
        
        return ". ".join(summary_parts) + "."
    
    def get_impact_analysis(self, operation_id: str) -> Optional[MaintenanceImpactAnalysis]:
        """Get impact analysis for a specific operation."""
        return self.impact_analyses.get(operation_id)
    
    def get_impact_analyses_in_period(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[MaintenanceImpactAnalysis]:
        """Get all impact analyses within a specific time period."""
        return [
            analysis for analysis in self.impact_analyses.values()
            if start_date <= analysis.timestamp <= end_date
        ]
    
    def generate_success_metrics_report(
        self,
        operation_logger: OperationLogger,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Generate a comprehensive success metrics report."""
        operations = operation_logger.get_operations_in_period(start_date, end_date)
        analyses = self.get_impact_analyses_in_period(start_date, end_date)
        
        # Basic operation statistics
        total_operations = len(operations)
        completed_operations = [op for op in operations if op.status.value == 'completed']
        failed_operations = [op for op in operations if op.status.value == 'failed']
        
        success_rate = (len(completed_operations) / total_operations * 100) if total_operations > 0 else 0
        
        # Duration statistics
        duration_stats = {}
        if completed_operations:
            durations = [op.duration_seconds / 60 for op in completed_operations if op.duration_seconds]
            if durations:
                duration_stats = {
                    'average_minutes': statistics.mean(durations),
                    'median_minutes': statistics.median(durations),
                    'min_minutes': min(durations),
                    'max_minutes': max(durations)
                }
        
        # Impact analysis statistics
        impact_stats = {}
        if analyses:
            impact_scores = [analysis.overall_impact_score for analysis in analyses]
            positive_impacts = [score for score in impact_scores if score > 0]
            negative_impacts = [score for score in impact_scores if score < 0]
            
            impact_stats = {
                'average_impact_score': statistics.mean(impact_scores),
                'positive_impact_operations': len(positive_impacts),
                'negative_impact_operations': len(negative_impacts),
                'neutral_impact_operations': len([score for score in impact_scores if score == 0]),
                'best_impact_score': max(impact_scores) if impact_scores else 0,
                'worst_impact_score': min(impact_scores) if impact_scores else 0
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
                    'average_impact': 0
                }
            
            type_breakdown[op_type]['total'] += 1
            
            if operation.status.value == 'completed':
                type_breakdown[op_type]['completed'] += 1
            elif operation.status.value == 'failed':
                type_breakdown[op_type]['failed'] += 1
            
            # Add impact score if available
            analysis = self.get_impact_analysis(operation.id)
            if analysis:
                type_breakdown[op_type]['average_impact'] += analysis.overall_impact_score
        
        # Calculate average impact per type
        for op_type, stats in type_breakdown.items():
            if stats['total'] > 0:
                stats['average_impact'] /= stats['total']
        
        # Top improvements and regressions
        all_improvements = {}
        all_regressions = {}
        
        for analysis in analyses:
            for metric, value in analysis.improvements.items():
                if metric not in all_improvements:
                    all_improvements[metric] = []
                all_improvements[metric].append(value)
            
            for metric, value in analysis.regressions.items():
                if metric not in all_regressions:
                    all_regressions[metric] = []
                all_regressions[metric].append(value)
        
        # Calculate average improvements and regressions
        avg_improvements = {
            metric: statistics.mean(values)
            for metric, values in all_improvements.items()
        }
        
        avg_regressions = {
            metric: statistics.mean(values)
            for metric, values in all_regressions.items()
        }
        
        return {
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'operation_summary': {
                'total_operations': total_operations,
                'completed_operations': len(completed_operations),
                'failed_operations': len(failed_operations),
                'success_rate_percentage': success_rate
            },
            'duration_statistics': duration_stats,
            'impact_statistics': impact_stats,
            'operation_type_breakdown': type_breakdown,
            'top_improvements': dict(sorted(avg_improvements.items(), key=lambda x: x[1], reverse=True)[:5]),
            'top_regressions': dict(sorted(avg_regressions.items(), key=lambda x: x[1], reverse=True)[:5]),
            'generated_at': datetime.now().isoformat()
        }
    
    def identify_improvement_opportunities(
        self,
        operation_logger: OperationLogger,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """Identify opportunities for improvement based on historical data."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        operations = operation_logger.get_operations_in_period(start_date, end_date)
        analyses = self.get_impact_analyses_in_period(start_date, end_date)
        
        opportunities = []
        
        # Identify frequently failing operation types
        type_failure_rates = {}
        for operation in operations:
            op_type = operation.operation_type.value
            if op_type not in type_failure_rates:
                type_failure_rates[op_type] = {'total': 0, 'failed': 0}
            
            type_failure_rates[op_type]['total'] += 1
            if operation.status.value == 'failed':
                type_failure_rates[op_type]['failed'] += 1
        
        for op_type, stats in type_failure_rates.items():
            failure_rate = stats['failed'] / stats['total'] if stats['total'] > 0 else 0
            if failure_rate > 0.2:  # More than 20% failure rate
                opportunities.append({
                    'type': 'high_failure_rate',
                    'operation_type': op_type,
                    'failure_rate': failure_rate * 100,
                    'recommendation': f"Review and improve {op_type.replace('_', ' ')} procedures",
                    'priority': 'high' if failure_rate > 0.4 else 'medium'
                })
        
        # Identify operations with consistently negative impact
        negative_impact_types = {}
        for analysis in analyses:
            if analysis.overall_impact_score < -10:
                # Find the operation type
                operation = operation_logger.get_operation(analysis.operation_id)
                if operation:
                    op_type = operation.operation_type.value
                    if op_type not in negative_impact_types:
                        negative_impact_types[op_type] = []
                    negative_impact_types[op_type].append(analysis.overall_impact_score)
        
        for op_type, scores in negative_impact_types.items():
            if len(scores) >= 2 and statistics.mean(scores) < -15:
                opportunities.append({
                    'type': 'negative_impact',
                    'operation_type': op_type,
                    'average_impact': statistics.mean(scores),
                    'recommendation': f"Investigate why {op_type.replace('_', ' ')} operations have negative impact",
                    'priority': 'high'
                })
        
        # Identify slow operations
        slow_operations = {}
        for operation in operations:
            if operation.duration_seconds and operation.duration_seconds > 3600:  # More than 1 hour
                op_type = operation.operation_type.value
                if op_type not in slow_operations:
                    slow_operations[op_type] = []
                slow_operations[op_type].append(operation.duration_seconds / 60)  # Convert to minutes
        
        for op_type, durations in slow_operations.items():
            if len(durations) >= 2 and statistics.mean(durations) > 90:  # Average more than 1.5 hours
                opportunities.append({
                    'type': 'slow_operations',
                    'operation_type': op_type,
                    'average_duration_minutes': statistics.mean(durations),
                    'recommendation': f"Optimize {op_type.replace('_', ' ')} operations to reduce duration",
                    'priority': 'medium'
                })
        
        return opportunities
    
    def cleanup_old_analyses(self, days: int = 90) -> int:
        """Clean up old impact analyses."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        old_analyses = [
            op_id for op_id, analysis in self.impact_analyses.items()
            if analysis.timestamp < cutoff_date
        ]
        
        for op_id in old_analyses:
            del self.impact_analyses[op_id]
        
        if old_analyses:
            self._save_data()
        
        return len(old_analyses)