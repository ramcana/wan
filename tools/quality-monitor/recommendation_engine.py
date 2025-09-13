"""
Automated quality improvement recommendation engine.
"""

import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import statistics

try:
    from tools.quality-monitor.models import (
        QualityRecommendation, QualityMetric, QualityTrend, QualityAlert,
        AlertSeverity, MetricType, TrendDirection
    )
except ImportError:
    from models import (
        QualityRecommendation, QualityMetric, QualityTrend, QualityAlert,
        AlertSeverity, MetricType, TrendDirection
    )


class RecommendationEngine:
    """Generates automated recommendations for quality improvements."""
    
    def __init__(self, data_dir: str = "data/quality-recommendations"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.recommendation_rules = self._load_recommendation_rules()
        self.generated_recommendations: Dict[str, QualityRecommendation] = {}
        self._load_existing_recommendations()
    
    def _load_recommendation_rules(self) -> Dict:
        """Load recommendation generation rules."""
        return {
            MetricType.TEST_COVERAGE: {
                "low_threshold": 60.0,
                "critical_threshold": 40.0,
                "recommendations": {
                    "high_impact": [
                        {
                            "title": "Implement comprehensive unit test suite",
                            "description": "Add unit tests for all core functionality to improve coverage",
                            "actions": [
                                "Identify untested code paths using coverage reports",
                                "Write unit tests for critical business logic",
                                "Add integration tests for component interactions",
                                "Set up automated test execution in CI/CD"
                            ],
                            "estimated_impact": 25.0,
                            "estimated_effort": "high"
                        },
                        {
                            "title": "Add missing test fixtures and utilities",
                            "description": "Create reusable test fixtures to improve test quality and coverage",
                            "actions": [
                                "Create shared test fixtures for common scenarios",
                                "Implement test data factories",
                                "Add test utilities for common operations",
                                "Standardize test setup and teardown"
                            ],
                            "estimated_impact": 15.0,
                            "estimated_effort": "medium"
                        }
                    ],
                    "medium_impact": [
                        {
                            "title": "Improve existing test quality",
                            "description": "Enhance existing tests to cover more scenarios",
                            "actions": [
                                "Review existing tests for completeness",
                                "Add edge case testing",
                                "Improve test assertions and validation",
                                "Add performance and load testing"
                            ],
                            "estimated_impact": 10.0,
                            "estimated_effort": "medium"
                        }
                    ]
                }
            },
            MetricType.CODE_COMPLEXITY: {
                "warning_threshold": 10.0,
                "critical_threshold": 15.0,
                "recommendations": {
                    "high_impact": [
                        {
                            "title": "Refactor complex functions",
                            "description": "Break down complex functions into smaller, manageable units",
                            "actions": [
                                "Identify functions with high cyclomatic complexity",
                                "Extract helper functions for repeated logic",
                                "Simplify conditional statements",
                                "Apply single responsibility principle"
                            ],
                            "estimated_impact": 20.0,
                            "estimated_effort": "high"
                        }
                    ],
                    "medium_impact": [
                        {
                            "title": "Apply design patterns",
                            "description": "Use design patterns to reduce complexity",
                            "actions": [
                                "Identify opportunities for strategy pattern",
                                "Apply factory pattern for object creation",
                                "Use decorator pattern for cross-cutting concerns",
                                "Implement command pattern for complex operations"
                            ],
                            "estimated_impact": 15.0,
                            "estimated_effort": "medium"
                        }
                    ]
                }
            },
            MetricType.DOCUMENTATION_COVERAGE: {
                "low_threshold": 70.0,
                "critical_threshold": 50.0,
                "recommendations": {
                    "high_impact": [
                        {
                            "title": "Add comprehensive docstrings",
                            "description": "Document all public functions, classes, and modules",
                            "actions": [
                                "Add docstrings to all public functions",
                                "Document class methods and attributes",
                                "Create module-level documentation",
                                "Use consistent docstring format (Google/Sphinx style)"
                            ],
                            "estimated_impact": 30.0,
                            "estimated_effort": "medium"
                        }
                    ],
                    "medium_impact": [
                        {
                            "title": "Generate API documentation",
                            "description": "Create automated API documentation from docstrings",
                            "actions": [
                                "Set up Sphinx or similar documentation generator",
                                "Configure automated documentation builds",
                                "Add code examples to documentation",
                                "Create developer guides and tutorials"
                            ],
                            "estimated_impact": 15.0,
                            "estimated_effort": "medium"
                        }
                    ]
                }
            },
            MetricType.DUPLICATE_CODE: {
                "warning_threshold": 10.0,
                "critical_threshold": 20.0,
                "recommendations": {
                    "high_impact": [
                        {
                            "title": "Extract common functionality",
                            "description": "Identify and extract duplicate code into shared modules",
                            "actions": [
                                "Scan for duplicate code blocks",
                                "Extract common functions into utilities",
                                "Create shared base classes for similar functionality",
                                "Implement DRY principle consistently"
                            ],
                            "estimated_impact": 25.0,
                            "estimated_effort": "high"
                        }
                    ]
                }
            },
            MetricType.STYLE_VIOLATIONS: {
                "warning_threshold": 50.0,
                "critical_threshold": 100.0,
                "recommendations": {
                    "high_impact": [
                        {
                            "title": "Implement automated code formatting",
                            "description": "Set up automated code formatting and style checking",
                            "actions": [
                                "Configure black or similar formatter",
                                "Set up pre-commit hooks for style checking",
                                "Configure IDE for automatic formatting",
                                "Run formatter on entire codebase"
                            ],
                            "estimated_impact": 40.0,
                            "estimated_effort": "low"
                        }
                    ]
                }
            },
            MetricType.TYPE_HINT_COVERAGE: {
                "low_threshold": 70.0,
                "critical_threshold": 50.0,
                "recommendations": {
                    "high_impact": [
                        {
                            "title": "Add comprehensive type hints",
                            "description": "Add type hints to improve code clarity and catch errors",
                            "actions": [
                                "Add type hints to function parameters and returns",
                                "Use mypy for static type checking",
                                "Add type hints to class attributes",
                                "Use generic types where appropriate"
                            ],
                            "estimated_impact": 20.0,
                            "estimated_effort": "medium"
                        }
                    ]
                }
            }
        }
    
    def _load_existing_recommendations(self) -> None:
        """Load existing recommendations from storage."""
        recommendations_file = self.data_dir / "recommendations.json"
        if recommendations_file.exists():
            try:
                with open(recommendations_file) as f:
                    data = json.load(f)
                
                for rec_data in data.get('recommendations', []):
                    rec = QualityRecommendation(
                        id=rec_data['id'],
                        title=rec_data['title'],
                        description=rec_data['description'],
                        priority=AlertSeverity(rec_data['priority']),
                        metric_types=[MetricType(mt) for mt in rec_data['metric_types']],
                        estimated_impact=rec_data['estimated_impact'],
                        estimated_effort=rec_data['estimated_effort'],
                        actions=rec_data['actions'],
                        timestamp=datetime.fromisoformat(rec_data['timestamp'])
                    )
                    self.generated_recommendations[rec.id] = rec
            
            except Exception as e:
                print(f"Error loading recommendations: {e}")
    
    def _save_recommendations(self) -> None:
        """Save recommendations to storage."""
        recommendations_file = self.data_dir / "recommendations.json"
        data = {
            'recommendations': [rec.to_dict() for rec in self.generated_recommendations.values()],
            'last_updated': datetime.now().isoformat()
        }
        
        with open(recommendations_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def generate_metric_recommendations(self, metrics: List[QualityMetric]) -> List[QualityRecommendation]:
        """Generate recommendations based on current metrics."""
        recommendations = []
        
        for metric in metrics:
            if metric.metric_type not in self.recommendation_rules:
                continue
            
            rules = self.recommendation_rules[metric.metric_type]
            metric_recommendations = self._generate_for_metric(metric, rules)
            recommendations.extend(metric_recommendations)
        
        # Store new recommendations
        for rec in recommendations:
            self.generated_recommendations[rec.id] = rec
        
        self._save_recommendations()
        return recommendations
    
    def _generate_for_metric(self, metric: QualityMetric, rules: Dict) -> List[QualityRecommendation]:
        """Generate recommendations for a specific metric."""
        recommendations = []
        
        # Determine severity based on thresholds
        severity = AlertSeverity.LOW
        if metric.metric_type in [MetricType.TEST_COVERAGE, MetricType.DOCUMENTATION_COVERAGE, MetricType.TYPE_HINT_COVERAGE]:
            # Higher is better metrics
            if metric.value < rules.get('critical_threshold', 40.0):
                severity = AlertSeverity.CRITICAL
            elif metric.value < rules.get('low_threshold', 60.0):
                severity = AlertSeverity.HIGH
        else:
            # Lower is better metrics
            if metric.value > rules.get('critical_threshold', 20.0):
                severity = AlertSeverity.CRITICAL
            elif metric.value > rules.get('warning_threshold', 10.0):
                severity = AlertSeverity.HIGH
        
        if severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]:
            # Generate high-impact recommendations
            for rec_template in rules['recommendations'].get('high_impact', []):
                rec = self._create_recommendation(rec_template, metric, severity)
                recommendations.append(rec)
        
        if severity == AlertSeverity.HIGH:
            # Also add medium-impact recommendations
            for rec_template in rules['recommendations'].get('medium_impact', []):
                rec = self._create_recommendation(rec_template, metric, AlertSeverity.MEDIUM)
                recommendations.append(rec)
        
        return recommendations
    
    def generate_trend_recommendations(self, trends: List[QualityTrend]) -> List[QualityRecommendation]:
        """Generate recommendations based on quality trends."""
        recommendations = []
        
        for trend in trends:
            if trend.direction == TrendDirection.DEGRADING and trend.confidence > 0.5:
                rec = self._create_trend_recommendation(trend)
                if rec:
                    recommendations.append(rec)
                    self.generated_recommendations[rec.id] = rec
        
        self._save_recommendations()
        return recommendations
    
    def _create_recommendation(self, template: Dict, metric: QualityMetric, priority: AlertSeverity) -> QualityRecommendation:
        """Create a recommendation from a template."""
        return QualityRecommendation(
            id=str(uuid.uuid4()),
            title=template['title'],
            description=f"{template['description']} (Current: {metric.value:.2f})",
            priority=priority,
            metric_types=[metric.metric_type],
            estimated_impact=template['estimated_impact'],
            estimated_effort=template['estimated_effort'],
            actions=template['actions']
        )
    
    def _create_trend_recommendation(self, trend: QualityTrend) -> Optional[QualityRecommendation]:
        """Create a recommendation based on a trend."""
        if abs(trend.change_rate) < 2.0:  # Only for significant trends
            return None
        
        return QualityRecommendation(
            id=str(uuid.uuid4()),
            title=f"Address degrading {trend.metric_type.value.replace('_', ' ')} trend",
            description=f"Quality metric is degrading at {trend.change_rate:.2f}% per day",
            priority=AlertSeverity.HIGH if abs(trend.change_rate) > 5.0 else AlertSeverity.MEDIUM,
            metric_types=[trend.metric_type],
            estimated_impact=min(abs(trend.change_rate) * 2, 30.0),
            estimated_effort="medium",
            actions=[
                f"Investigate recent changes affecting {trend.metric_type.value}",
                "Implement quality gates to prevent further degradation",
                "Set up more frequent monitoring for this metric",
                "Review and improve development practices"
            ]
        )
    
    def generate_proactive_recommendations(self, metrics: List[QualityMetric], trends: List[QualityTrend]) -> List[QualityRecommendation]:
        """Generate proactive recommendations for quality improvements."""
        recommendations = []
        
        # Analyze overall quality health
        quality_scores = self._calculate_quality_scores(metrics)
        
        # Generate recommendations based on overall health
        if quality_scores['overall'] < 70.0:
            recommendations.extend(self._generate_comprehensive_improvement_plan(metrics, trends))
        elif quality_scores['overall'] < 85.0:
            recommendations.extend(self._generate_targeted_improvements(metrics, trends))
        else:
            recommendations.extend(self._generate_maintenance_recommendations(metrics, trends))
        
        # Store new recommendations
        for rec in recommendations:
            self.generated_recommendations[rec.id] = rec
        
        self._save_recommendations()
        return recommendations
    
    def _calculate_quality_scores(self, metrics: List[QualityMetric]) -> Dict[str, float]:
        """Calculate overall quality scores."""
        scores = {}
        
        for metric in metrics:
            if metric.metric_type == MetricType.TEST_COVERAGE:
                scores['test_coverage'] = metric.value
            elif metric.metric_type == MetricType.CODE_COMPLEXITY:
                # Convert complexity to score (lower is better)
                scores['code_complexity'] = max(0, 100 - metric.value * 5)
            elif metric.metric_type == MetricType.DOCUMENTATION_COVERAGE:
                scores['documentation'] = metric.value
            elif metric.metric_type == MetricType.DUPLICATE_CODE:
                # Convert duplicate percentage to score
                scores['duplicate_code'] = max(0, 100 - metric.value * 2)
            elif metric.metric_type == MetricType.STYLE_VIOLATIONS:
                # Convert violations to score
                scores['style'] = max(0, 100 - metric.value)
            elif metric.metric_type == MetricType.TYPE_HINT_COVERAGE:
                scores['type_hints'] = metric.value
        
        # Calculate overall score
        if scores:
            scores['overall'] = statistics.mean(scores.values())
        else:
            scores['overall'] = 0.0
        
        return scores
    
    def _generate_comprehensive_improvement_plan(self, metrics: List[QualityMetric], trends: List[QualityTrend]) -> List[QualityRecommendation]:
        """Generate comprehensive improvement plan for low-quality codebases."""
        return [
            QualityRecommendation(
                id=str(uuid.uuid4()),
                title="Implement comprehensive quality improvement program",
                description="Establish systematic approach to improve overall code quality",
                priority=AlertSeverity.HIGH,
                metric_types=list(MetricType),
                estimated_impact=40.0,
                estimated_effort="high",
                actions=[
                    "Establish quality standards and guidelines",
                    "Implement automated quality checking in CI/CD",
                    "Create quality improvement roadmap",
                    "Set up regular quality reviews",
                    "Train team on quality best practices"
                ]
            )
        ]
    
    def _generate_targeted_improvements(self, metrics: List[QualityMetric], trends: List[QualityTrend]) -> List[QualityRecommendation]:
        """Generate targeted improvements for moderate-quality codebases."""
        recommendations = []
        
        # Find the worst-performing metrics
        metric_scores = []
        for metric in metrics:
            if metric.metric_type in [MetricType.TEST_COVERAGE, MetricType.DOCUMENTATION_COVERAGE, MetricType.TYPE_HINT_COVERAGE]:
                score = metric.value
            else:
                score = 100 - metric.value  # Invert for "lower is better" metrics
            
            metric_scores.append((metric.metric_type, score))
        
        # Sort by score and target the worst ones
        metric_scores.sort(key=lambda x: x[1])
        
        for metric_type, score in metric_scores[:3]:  # Top 3 worst metrics
            if score < 70.0:
                rec = QualityRecommendation(
                    id=str(uuid.uuid4()),
                    title=f"Improve {metric_type.value.replace('_', ' ')}",
                    description=f"Focus on improving {metric_type.value.replace('_', ' ')} (current score: {score:.1f})",
                    priority=AlertSeverity.MEDIUM,
                    metric_types=[metric_type],
                    estimated_impact=20.0,
                    estimated_effort="medium",
                    actions=self._get_improvement_actions(metric_type)
                )
                recommendations.append(rec)
        
        return recommendations
    
    def _generate_maintenance_recommendations(self, metrics: List[QualityMetric], trends: List[QualityTrend]) -> List[QualityRecommendation]:
        """Generate maintenance recommendations for high-quality codebases."""
        return [
            QualityRecommendation(
                id=str(uuid.uuid4()),
                title="Maintain high quality standards",
                description="Continue monitoring and maintaining current quality levels",
                priority=AlertSeverity.LOW,
                metric_types=list(MetricType),
                estimated_impact=5.0,
                estimated_effort="low",
                actions=[
                    "Continue regular quality monitoring",
                    "Review and update quality standards",
                    "Share quality best practices with team",
                    "Consider advanced quality metrics"
                ]
            )
        ]
    
    def _get_improvement_actions(self, metric_type: MetricType) -> List[str]:
        """Get improvement actions for a specific metric type."""
        actions_map = {
            MetricType.TEST_COVERAGE: [
                "Add unit tests for uncovered code",
                "Implement integration tests",
                "Set up test coverage reporting",
                "Review and improve test quality"
            ],
            MetricType.CODE_COMPLEXITY: [
                "Refactor complex functions",
                "Extract helper methods",
                "Simplify conditional logic",
                "Apply design patterns"
            ],
            MetricType.DOCUMENTATION_COVERAGE: [
                "Add docstrings to functions and classes",
                "Create API documentation",
                "Write developer guides",
                "Document configuration options"
            ],
            MetricType.DUPLICATE_CODE: [
                "Extract common functionality",
                "Create shared utilities",
                "Refactor similar code patterns",
                "Implement DRY principle"
            ],
            MetricType.STYLE_VIOLATIONS: [
                "Run automated code formatter",
                "Set up style checking in CI",
                "Configure pre-commit hooks",
                "Review and fix style issues"
            ],
            MetricType.TYPE_HINT_COVERAGE: [
                "Add type hints to functions",
                "Use mypy for type checking",
                "Add type hints to class attributes",
                "Use generic types appropriately"
            ]
        }
        
        return actions_map.get(metric_type, ["Review and improve code quality"])
    
    def get_active_recommendations(self, days: int = 30) -> List[QualityRecommendation]:
        """Get active recommendations from the last N days."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        return [
            rec for rec in self.generated_recommendations.values()
            if rec.timestamp > cutoff_date
        ]
    
    def get_recommendations_by_priority(self, priority: AlertSeverity) -> List[QualityRecommendation]:
        """Get recommendations by priority level."""
        return [
            rec for rec in self.generated_recommendations.values()
            if rec.priority == priority
        ]
    
    def get_recommendations_summary(self) -> Dict[str, any]:
        """Get summary of all recommendations."""
        active_recs = self.get_active_recommendations()
        
        return {
            'total_recommendations': len(active_recs),
            'by_priority': {
                'critical': len([r for r in active_recs if r.priority == AlertSeverity.CRITICAL]),
                'high': len([r for r in active_recs if r.priority == AlertSeverity.HIGH]),
                'medium': len([r for r in active_recs if r.priority == AlertSeverity.MEDIUM]),
                'low': len([r for r in active_recs if r.priority == AlertSeverity.LOW])
            },
            'by_metric_type': self._group_by_metric_type(active_recs),
            'estimated_total_impact': sum(r.estimated_impact for r in active_recs),
            'recommendations': [rec.to_dict() for rec in active_recs],
            'generated_at': datetime.now().isoformat()
        }
    
    def _group_by_metric_type(self, recommendations: List[QualityRecommendation]) -> Dict[str, int]:
        """Group recommendations by metric type."""
        groups = {}
        
        for rec in recommendations:
            for metric_type in rec.metric_types:
                metric_name = metric_type.value
                groups[metric_name] = groups.get(metric_name, 0) + 1
        
        return groups
    
    def cleanup_old_recommendations(self, days: int = 90) -> int:
        """Clean up old recommendations."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        old_recs = [
            rec_id for rec_id, rec in self.generated_recommendations.items()
            if rec.timestamp < cutoff_date
        ]
        
        for rec_id in old_recs:
            del self.generated_recommendations[rec_id]
        
        if old_recs:
            self._save_recommendations()
        
        return len(old_recs)
