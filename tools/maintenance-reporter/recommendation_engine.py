"""
Maintenance recommendation engine based on project analysis.
"""

import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import statistics

try:
    from tools.maintenance-reporter.models import (
        MaintenanceRecommendation, MaintenanceOperationType, ImpactLevel,
        MaintenanceScheduleOptimization
    )
    from tools.maintenance-reporter.operation_logger import OperationLogger
    from tools.maintenance-reporter.impact_analyzer import ImpactAnalyzer
except ImportError:
    from models import (
        MaintenanceRecommendation, MaintenanceOperationType, ImpactLevel,
        MaintenanceScheduleOptimization
    )
    from operation_logger import OperationLogger
    from impact_analyzer import ImpactAnalyzer


class MaintenanceRecommendationEngine:
    """Generates maintenance recommendations and optimizes scheduling."""
    
    def __init__(self, data_dir: str = "data/maintenance-recommendations"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.recommendations_file = self.data_dir / "recommendations.json"
        self.recommendations: Dict[str, MaintenanceRecommendation] = {}
        
        self._load_data()
        self._load_recommendation_rules()
    
    def _load_data(self):
        """Load existing recommendations."""
        if self.recommendations_file.exists():
            try:
                with open(self.recommendations_file) as f:
                    data = json.load(f)
                
                for rec_data in data.get('recommendations', []):
                    rec = MaintenanceRecommendation(
                        id=rec_data['id'],
                        title=rec_data['title'],
                        description=rec_data['description'],
                        operation_type=MaintenanceOperationType(rec_data['operation_type']),
                        priority=ImpactLevel(rec_data['priority']),
                        estimated_effort_hours=rec_data['estimated_effort_hours'],
                        estimated_impact_score=rec_data['estimated_impact_score'],
                        prerequisites=rec_data.get('prerequisites', []),
                        risks=rec_data.get('risks', []),
                        benefits=rec_data.get('benefits', []),
                        suggested_schedule=rec_data.get('suggested_schedule'),
                        created_at=datetime.fromisoformat(rec_data['created_at'])
                    )
                    self.recommendations[rec.id] = rec
            
            except Exception as e:
                print(f"Error loading recommendations: {e}")
    
    def _save_data(self):
        """Save recommendations to storage."""
        data = {
            'recommendations': [rec.to_dict() for rec in self.recommendations.values()],
            'last_updated': datetime.now().isoformat()
        }
        
        with open(self.recommendations_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_recommendation_rules(self):
        """Load recommendation generation rules."""
        self.recommendation_rules = {
            'test_repair': {
                'triggers': ['low_test_coverage', 'failing_tests', 'flaky_tests'],
                'base_effort_hours': 8,
                'base_impact_score': 25,
                'benefits': [
                    'Improved code reliability',
                    'Better test coverage',
                    'Reduced regression risk',
                    'Enhanced development confidence'
                ],
                'risks': [
                    'Temporary test instability',
                    'Potential breaking changes',
                    'Time investment required'
                ]
            },
            'code_cleanup': {
                'triggers': ['duplicate_code', 'dead_code', 'high_complexity'],
                'base_effort_hours': 12,
                'base_impact_score': 20,
                'benefits': [
                    'Improved code maintainability',
                    'Reduced technical debt',
                    'Better code organization',
                    'Enhanced readability'
                ],
                'risks': [
                    'Potential introduction of bugs',
                    'Temporary code instability',
                    'Merge conflicts with ongoing work'
                ]
            },
            'documentation_update': {
                'triggers': ['low_documentation_coverage', 'outdated_docs'],
                'base_effort_hours': 6,
                'base_impact_score': 15,
                'benefits': [
                    'Better developer onboarding',
                    'Improved code understanding',
                    'Reduced support overhead',
                    'Enhanced team collaboration'
                ],
                'risks': [
                    'Documentation may become outdated',
                    'Time investment with indirect benefits'
                ]
            },
            'configuration_consolidation': {
                'triggers': ['scattered_configs', 'config_conflicts'],
                'base_effort_hours': 16,
                'base_impact_score': 30,
                'benefits': [
                    'Simplified configuration management',
                    'Reduced deployment errors',
                    'Better environment consistency',
                    'Easier troubleshooting'
                ],
                'risks': [
                    'Potential configuration errors',
                    'Service disruption during migration',
                    'Complex rollback procedures'
                ]
            },
            'quality_improvement': {
                'triggers': ['style_violations', 'low_type_hints', 'quality_regression'],
                'base_effort_hours': 10,
                'base_impact_score': 18,
                'benefits': [
                    'Improved code quality',
                    'Better development standards',
                    'Enhanced code consistency',
                    'Reduced review overhead'
                ],
                'risks': [
                    'Temporary development slowdown',
                    'Potential team resistance',
                    'Learning curve for new standards'
                ]
            },
            'dependency_update': {
                'triggers': ['outdated_dependencies', 'security_vulnerabilities'],
                'base_effort_hours': 4,
                'base_impact_score': 22,
                'benefits': [
                    'Improved security',
                    'Access to new features',
                    'Better performance',
                    'Reduced technical debt'
                ],
                'risks': [
                    'Breaking changes in dependencies',
                    'Compatibility issues',
                    'Regression in functionality'
                ]
            },
            'performance_optimization': {
                'triggers': ['slow_performance', 'resource_usage_high'],
                'base_effort_hours': 20,
                'base_impact_score': 35,
                'benefits': [
                    'Improved system performance',
                    'Better user experience',
                    'Reduced resource costs',
                    'Enhanced scalability'
                ],
                'risks': [
                    'Complex optimization challenges',
                    'Potential performance regressions',
                    'Significant time investment'
                ]
            }
        }
    
    def generate_recommendations_from_analysis(
        self,
        operation_logger: OperationLogger,
        impact_analyzer: ImpactAnalyzer,
        project_metrics: Dict[str, float]
    ) -> List[MaintenanceRecommendation]:
        """Generate recommendations based on project analysis."""
        recommendations = []
        
        # Analyze recent operations for patterns
        recent_operations = operation_logger.get_operations_in_period(
            datetime.now() - timedelta(days=30),
            datetime.now()
        )
        
        # Get improvement opportunities
        opportunities = impact_analyzer.identify_improvement_opportunities(operation_logger)
        
        # Generate recommendations based on project metrics
        metric_recommendations = self._generate_metric_based_recommendations(project_metrics)
        recommendations.extend(metric_recommendations)
        
        # Generate recommendations based on operation patterns
        pattern_recommendations = self._generate_pattern_based_recommendations(recent_operations)
        recommendations.extend(pattern_recommendations)
        
        # Generate recommendations based on improvement opportunities
        opportunity_recommendations = self._generate_opportunity_based_recommendations(opportunities)
        recommendations.extend(opportunity_recommendations)
        
        # Store new recommendations
        for rec in recommendations:
            self.recommendations[rec.id] = rec
        
        self._save_data()
        return recommendations
    
    def _generate_metric_based_recommendations(
        self,
        project_metrics: Dict[str, float]
    ) -> List[MaintenanceRecommendation]:
        """Generate recommendations based on project quality metrics."""
        recommendations = []
        
        # Test coverage recommendations
        test_coverage = project_metrics.get('test_coverage', 0)
        if test_coverage < 70:
            priority = ImpactLevel.HIGH if test_coverage < 50 else ImpactLevel.MEDIUM
            effort_multiplier = 2.0 if test_coverage < 50 else 1.5
            
            rec = self._create_recommendation(
                operation_type=MaintenanceOperationType.TEST_REPAIR,
                title="Improve test coverage",
                description=f"Current test coverage is {test_coverage:.1f}%. Recommend increasing to at least 80%.",
                priority=priority,
                effort_multiplier=effort_multiplier,
                impact_multiplier=1.5 if test_coverage < 50 else 1.2,
                prerequisites=["Identify untested code paths", "Set up test infrastructure"],
                suggested_schedule="Next sprint - high priority"
            )
            recommendations.append(rec)
        
        # Code complexity recommendations
        code_complexity = project_metrics.get('code_complexity', 0)
        if code_complexity > 10:
            priority = ImpactLevel.HIGH if code_complexity > 15 else ImpactLevel.MEDIUM
            
            rec = self._create_recommendation(
                operation_type=MaintenanceOperationType.CODE_CLEANUP,
                title="Reduce code complexity",
                description=f"Average code complexity is {code_complexity:.1f}. Recommend refactoring complex functions.",
                priority=priority,
                effort_multiplier=1.8 if code_complexity > 15 else 1.3,
                impact_multiplier=1.4,
                prerequisites=["Identify most complex functions", "Plan refactoring approach"],
                suggested_schedule="Next 2-3 sprints"
            )
            recommendations.append(rec)
        
        # Documentation coverage recommendations
        doc_coverage = project_metrics.get('documentation_coverage', 0)
        if doc_coverage < 60:
            rec = self._create_recommendation(
                operation_type=MaintenanceOperationType.DOCUMENTATION_UPDATE,
                title="Improve documentation coverage",
                description=f"Documentation coverage is {doc_coverage:.1f}%. Add docstrings and update docs.",
                priority=ImpactLevel.MEDIUM,
                effort_multiplier=1.2,
                impact_multiplier=1.0,
                prerequisites=["Audit existing documentation", "Define documentation standards"],
                suggested_schedule="Ongoing - allocate 20% of sprint capacity"
            )
            recommendations.append(rec)
        
        # Duplicate code recommendations
        duplicate_code = project_metrics.get('duplicate_code', 0)
        if duplicate_code > 15:
            rec = self._create_recommendation(
                operation_type=MaintenanceOperationType.CODE_CLEANUP,
                title="Eliminate duplicate code",
                description=f"Duplicate code percentage is {duplicate_code:.1f}%. Extract common functionality.",
                priority=ImpactLevel.MEDIUM,
                effort_multiplier=1.5,
                impact_multiplier=1.3,
                prerequisites=["Identify duplicate code patterns", "Plan extraction strategy"],
                suggested_schedule="Next sprint"
            )
            recommendations.append(rec)
        
        # Style violations recommendations
        style_violations = project_metrics.get('style_violations', 0)
        if style_violations > 50:
            rec = self._create_recommendation(
                operation_type=MaintenanceOperationType.QUALITY_IMPROVEMENT,
                title="Fix style violations",
                description=f"Style violations per KLOC: {style_violations:.1f}. Implement automated formatting.",
                priority=ImpactLevel.LOW,
                effort_multiplier=0.8,
                impact_multiplier=1.1,
                prerequisites=["Set up automated formatting tools", "Configure pre-commit hooks"],
                suggested_schedule="Next week - quick win"
            )
            recommendations.append(rec)
        
        return recommendations
    
    def _generate_pattern_based_recommendations(
        self,
        recent_operations: List
    ) -> List[MaintenanceRecommendation]:
        """Generate recommendations based on operation patterns."""
        recommendations = []
        
        if not recent_operations:
            return recommendations
        
        # Analyze failure patterns
        failed_operations = [op for op in recent_operations if op.status.value == 'failed']
        if len(failed_operations) > len(recent_operations) * 0.3:  # More than 30% failure rate
            rec = self._create_recommendation(
                operation_type=MaintenanceOperationType.QUALITY_IMPROVEMENT,
                title="Improve maintenance operation reliability",
                description="High failure rate in recent maintenance operations. Review and improve procedures.",
                priority=ImpactLevel.HIGH,
                effort_multiplier=2.0,
                impact_multiplier=2.0,
                prerequisites=["Analyze failure patterns", "Review operation procedures"],
                suggested_schedule="Immediate - critical issue"
            )
            recommendations.append(rec)
        
        # Analyze duration patterns
        long_operations = [op for op in recent_operations 
                          if op.duration_seconds and op.duration_seconds > 7200]  # More than 2 hours
        if len(long_operations) > 3:
            rec = self._create_recommendation(
                operation_type=MaintenanceOperationType.PERFORMANCE_OPTIMIZATION,
                title="Optimize maintenance operation performance",
                description="Several recent operations took longer than expected. Investigate and optimize.",
                priority=ImpactLevel.MEDIUM,
                effort_multiplier=1.5,
                impact_multiplier=1.3,
                prerequisites=["Profile slow operations", "Identify bottlenecks"],
                suggested_schedule="Next 2 weeks"
            )
            recommendations.append(rec)
        
        return recommendations
    
    def _generate_opportunity_based_recommendations(
        self,
        opportunities: List[Dict[str, Any]]
    ) -> List[MaintenanceRecommendation]:
        """Generate recommendations based on improvement opportunities."""
        recommendations = []
        
        for opportunity in opportunities:
            if opportunity['type'] == 'high_failure_rate':
                rec = self._create_recommendation(
                    operation_type=MaintenanceOperationType.QUALITY_IMPROVEMENT,
                    title=f"Improve {opportunity['operation_type'].replace('_', ' ')} reliability",
                    description=f"Failure rate is {opportunity['failure_rate']:.1f}%. {opportunity['recommendation']}",
                    priority=ImpactLevel.HIGH if opportunity['priority'] == 'high' else ImpactLevel.MEDIUM,
                    effort_multiplier=1.8,
                    impact_multiplier=1.6,
                    prerequisites=["Analyze failure causes", "Develop improvement plan"],
                    suggested_schedule="Next sprint"
                )
                recommendations.append(rec)
            
            elif opportunity['type'] == 'negative_impact':
                rec = self._create_recommendation(
                    operation_type=MaintenanceOperationType.QUALITY_IMPROVEMENT,
                    title=f"Address negative impact in {opportunity['operation_type'].replace('_', ' ')}",
                    description=f"Average impact score: {opportunity['average_impact']:.1f}. {opportunity['recommendation']}",
                    priority=ImpactLevel.HIGH,
                    effort_multiplier=2.0,
                    impact_multiplier=2.2,
                    prerequisites=["Root cause analysis", "Impact assessment"],
                    suggested_schedule="Immediate investigation required"
                )
                recommendations.append(rec)
            
            elif opportunity['type'] == 'slow_operations':
                rec = self._create_recommendation(
                    operation_type=MaintenanceOperationType.PERFORMANCE_OPTIMIZATION,
                    title=f"Optimize {opportunity['operation_type'].replace('_', ' ')} performance",
                    description=f"Average duration: {opportunity['average_duration_minutes']:.1f} minutes. {opportunity['recommendation']}",
                    priority=ImpactLevel.MEDIUM,
                    effort_multiplier=1.5,
                    impact_multiplier=1.4,
                    prerequisites=["Performance profiling", "Optimization planning"],
                    suggested_schedule="Next month"
                )
                recommendations.append(rec)
        
        return recommendations
    
    def _create_recommendation(
        self,
        operation_type: MaintenanceOperationType,
        title: str,
        description: str,
        priority: ImpactLevel,
        effort_multiplier: float = 1.0,
        impact_multiplier: float = 1.0,
        prerequisites: Optional[List[str]] = None,
        suggested_schedule: Optional[str] = None
    ) -> MaintenanceRecommendation:
        """Create a maintenance recommendation."""
        rules = self.recommendation_rules.get(operation_type.value, {})
        
        base_effort = rules.get('base_effort_hours', 8)
        base_impact = rules.get('base_impact_score', 20)
        
        return MaintenanceRecommendation(
            id=str(uuid.uuid4()),
            title=title,
            description=description,
            operation_type=operation_type,
            priority=priority,
            estimated_effort_hours=base_effort * effort_multiplier,
            estimated_impact_score=base_impact * impact_multiplier,
            prerequisites=prerequisites or [],
            risks=rules.get('risks', []),
            benefits=rules.get('benefits', []),
            suggested_schedule=suggested_schedule
        )
    
    def optimize_maintenance_schedule(
        self,
        recommendations: List[MaintenanceRecommendation],
        available_hours_per_week: float = 40,
        max_concurrent_operations: int = 3
    ) -> MaintenanceScheduleOptimization:
        """Optimize the scheduling of maintenance operations."""
        # Sort recommendations by priority and impact
        priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        sorted_recs = sorted(
            recommendations,
            key=lambda r: (priority_order.get(r.priority.value, 4), -r.estimated_impact_score)
        )
        
        # Build dependency graph
        dependencies = {}
        for rec in sorted_recs:
            dependencies[rec.id] = []
            
            # Add dependencies based on operation types
            for other_rec in sorted_recs:
                if other_rec.id != rec.id:
                    if self._has_dependency(rec, other_rec):
                        dependencies[rec.id].append(other_rec.id)
        
        # Schedule operations considering dependencies and constraints
        scheduled_operations = []
        scheduling_rationale = {}
        current_week = 0
        remaining_hours = available_hours_per_week
        concurrent_operations = 0
        
        # Keep track of unscheduled recommendations
        unscheduled = sorted_recs.copy()
        
        while unscheduled:
            scheduled_this_round = []
            
            for rec in unscheduled:
                # Check if dependencies are satisfied
                deps_satisfied = all(
                    dep_id in [op['id'] for op in scheduled_operations]
                    for dep_id in dependencies[rec.id]
                )
                
                if not deps_satisfied:
                    continue
                
                # Check resource constraints
                if (remaining_hours < rec.estimated_effort_hours or 
                    concurrent_operations >= max_concurrent_operations):
                    current_week += 1
                    remaining_hours = available_hours_per_week
                    concurrent_operations = 0
                
                # Schedule the operation
                scheduled_operations.append({
                    'id': rec.id,
                    'week': current_week,
                    'estimated_hours': rec.estimated_effort_hours
                })
                
                remaining_hours -= rec.estimated_effort_hours
                concurrent_operations += 1
                
                scheduling_rationale[rec.id] = f"Scheduled for week {current_week + 1}"
                scheduled_this_round.append(rec)
            
            # Remove scheduled recommendations from unscheduled list
            for rec in scheduled_this_round:
                unscheduled.remove(rec)
            
            # If no progress was made this round, schedule remaining with dependency note
            if not scheduled_this_round and unscheduled:
                for rec in unscheduled:
                    scheduled_operations.append({
                        'id': rec.id,
                        'week': current_week + 1,
                        'estimated_hours': rec.estimated_effort_hours
                    })
                    scheduling_rationale[rec.id] = "Scheduled despite unresolved dependencies"
                break
        
        # Calculate resource requirements
        resource_requirements = {
            'total_hours': sum(rec.estimated_effort_hours for rec in sorted_recs),
            'estimated_weeks': current_week + 1,
            'average_hours_per_week': sum(rec.estimated_effort_hours for rec in sorted_recs) / max(1, current_week + 1)
        }
        
        # Generate risk mitigation plan
        risk_mitigation_plan = [
            "Ensure adequate testing before each operation",
            "Have rollback procedures ready for all operations",
            "Monitor system performance during operations",
            "Communicate maintenance windows to stakeholders",
            "Keep backup of all modified configurations and code"
        ]
        
        # Identify optimal time windows
        optimal_time_windows = [
            {
                'name': 'Low-impact operations',
                'description': 'During business hours for operations with minimal user impact',
                'suitable_for': ['documentation_update', 'quality_improvement']
            },
            {
                'name': 'Medium-impact operations',
                'description': 'During off-hours or maintenance windows',
                'suitable_for': ['code_cleanup', 'test_repair']
            },
            {
                'name': 'High-impact operations',
                'description': 'During scheduled maintenance windows with stakeholder notification',
                'suitable_for': ['configuration_consolidation', 'performance_optimization']
            }
        ]
        
        return MaintenanceScheduleOptimization(
            recommended_schedule=[op['id'] for op in scheduled_operations],
            scheduling_rationale=scheduling_rationale,
            resource_requirements=resource_requirements,
            risk_mitigation_plan=risk_mitigation_plan,
            estimated_total_duration_hours=sum(rec.estimated_effort_hours for rec in sorted_recs),
            optimal_time_windows=optimal_time_windows,
            dependencies=dependencies
        )
    
    def _has_dependency(self, rec1: MaintenanceRecommendation, rec2: MaintenanceRecommendation) -> bool:
        """Check if rec1 depends on rec2."""
        # Define dependency rules
        dependency_rules = {
            MaintenanceOperationType.TEST_REPAIR: [MaintenanceOperationType.CODE_CLEANUP],
            MaintenanceOperationType.PERFORMANCE_OPTIMIZATION: [
                MaintenanceOperationType.CODE_CLEANUP,
                MaintenanceOperationType.CONFIGURATION_CONSOLIDATION
            ],
            MaintenanceOperationType.QUALITY_IMPROVEMENT: [MaintenanceOperationType.CODE_CLEANUP],
        }
        
        dependencies = dependency_rules.get(rec1.operation_type, [])
        return rec2.operation_type in dependencies
    
    def get_active_recommendations(self, days: int = 30) -> List[MaintenanceRecommendation]:
        """Get active recommendations from the last N days."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        return [
            rec for rec in self.recommendations.values()
            if rec.created_at > cutoff_date
        ]
    
    def get_recommendations_by_priority(self, priority: ImpactLevel) -> List[MaintenanceRecommendation]:
        """Get recommendations by priority level."""
        return [
            rec for rec in self.recommendations.values()
            if rec.priority == priority
        ]
    
    def get_recommendations_summary(self) -> Dict[str, Any]:
        """Get summary of all recommendations."""
        active_recs = self.get_active_recommendations()
        
        return {
            'total_recommendations': len(active_recs),
            'by_priority': {
                'critical': len([r for r in active_recs if r.priority == ImpactLevel.CRITICAL]),
                'high': len([r for r in active_recs if r.priority == ImpactLevel.HIGH]),
                'medium': len([r for r in active_recs if r.priority == ImpactLevel.MEDIUM]),
                'low': len([r for r in active_recs if r.priority == ImpactLevel.LOW])
            },
            'by_operation_type': self._group_by_operation_type(active_recs),
            'estimated_total_effort_hours': sum(r.estimated_effort_hours for r in active_recs),
            'estimated_total_impact': sum(r.estimated_impact_score for r in active_recs),
            'recommendations': [rec.to_dict() for rec in active_recs],
            'generated_at': datetime.now().isoformat()
        }
    
    def _group_by_operation_type(self, recommendations: List[MaintenanceRecommendation]) -> Dict[str, int]:
        """Group recommendations by operation type."""
        groups = {}
        
        for rec in recommendations:
            op_type = rec.operation_type.value
            groups[op_type] = groups.get(op_type, 0) + 1
        
        return groups
    
    def cleanup_old_recommendations(self, days: int = 90) -> int:
        """Clean up old recommendations."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        old_recs = [
            rec_id for rec_id, rec in self.recommendations.items()
            if rec.created_at < cutoff_date
        ]
        
        for rec_id in old_recs:
            del self.recommendations[rec_id]
        
        if old_recs:
            self._save_data()
        
        return len(old_recs)