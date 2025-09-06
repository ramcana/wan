"""
Actionable recommendations engine for project health improvements
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging

from health_models import (
    HealthReport, HealthIssue, ComponentHealth, Recommendation,
    HealthCategory, Severity, HealthConfig
)


class RecommendationRule:
    """Base class for recommendation rules"""
    
    def __init__(self, name: str, category: HealthCategory, priority: int):
        self.name = name
        self.category = category
        self.priority = priority
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    def applies_to(self, report: HealthReport) -> bool:
        """Check if this rule applies to the given health report"""
        raise NotImplementedError
    
    def generate_recommendation(self, report: HealthReport) -> Optional[Recommendation]:
        """Generate recommendation based on health report"""
        raise NotImplementedError


class TestSuiteRecommendationRule(RecommendationRule):
    """Recommendations for test suite improvements"""
    
    def __init__(self):
        super().__init__("test_suite_improvements", HealthCategory.TESTS, 1)
    
    def applies_to(self, report: HealthReport) -> bool:
        test_component = report.component_scores.get("tests")
        return test_component is not None and test_component.score < 75
    
    def generate_recommendation(self, report: HealthReport) -> Optional[Recommendation]:
        test_component = report.component_scores.get("tests")
        if not test_component:
            return None
        
        action_items = []
        estimated_effort = "medium"
        impact = "high"
        
        # Analyze specific test issues
        if test_component.metrics.get("broken_tests", 0) > 0:
            action_items.append("Fix broken test files with syntax errors")
            estimated_effort = "low"
        
        if test_component.metrics.get("pass_rate", 100) < 80:
            action_items.append("Fix failing tests to achieve 80%+ pass rate")
            action_items.append("Update tests for recent code changes")
        
        if test_component.metrics.get("coverage_percent", 0) < 70:
            action_items.append("Add tests to achieve 70%+ code coverage")
            action_items.append("Focus on testing critical code paths")
            estimated_effort = "high"
        
        if test_component.metrics.get("execution_time", 0) > 900:
            action_items.append("Optimize slow tests for faster execution")
            action_items.append("Implement test parallelization")
        
        if not action_items:
            action_items = ["Review and improve overall test suite quality"]
        
        return Recommendation(
            priority=self.priority,
            category=self.category,
            title="Improve Test Suite Health",
            description=f"Test suite score is {test_component.score:.1f}/100. Addressing test issues will improve code reliability and development confidence.",
            action_items=action_items,
            estimated_effort=estimated_effort,
            impact=impact,
            related_issues=[issue.title for issue in test_component.issues],
            metadata={
                "current_score": test_component.score,
                "target_score": 85,
                "metrics": test_component.metrics
            }
        )


class DocumentationRecommendationRule(RecommendationRule):
    """Recommendations for documentation improvements"""
    
    def __init__(self):
        super().__init__("documentation_improvements", HealthCategory.DOCUMENTATION, 3)
    
    def applies_to(self, report: HealthReport) -> bool:
        doc_component = report.component_scores.get("documentation")
        return doc_component is not None and doc_component.score < 80
    
    def generate_recommendation(self, report: HealthReport) -> Optional[Recommendation]:
        doc_component = report.component_scores.get("documentation")
        if not doc_component:
            return None
        
        action_items = []
        estimated_effort = "medium"
        impact = "medium"
        
        # Analyze documentation issues
        if doc_component.metrics.get("missing_essential", []):
            missing_docs = doc_component.metrics["missing_essential"]
            action_items.append(f"Create missing essential documentation: {', '.join(missing_docs[:3])}")
            estimated_effort = "high"
        
        if doc_component.metrics.get("broken_links", 0) > 0:
            action_items.append("Fix broken links in documentation")
            action_items.append("Implement automated link checking")
        
        if doc_component.metrics.get("scattered_docs", 0) > 5:
            action_items.append("Consolidate scattered documentation into docs/ directory")
            action_items.append("Create unified documentation structure")
        
        if doc_component.metrics.get("outdated_docs", 0) > 0:
            action_items.append("Review and update outdated documentation")
            action_items.append("Establish documentation review schedule")
        
        if not action_items:
            action_items = ["Improve overall documentation quality and organization"]
        
        return Recommendation(
            priority=self.priority,
            category=self.category,
            title="Enhance Documentation System",
            description=f"Documentation score is {doc_component.score:.1f}/100. Better documentation improves onboarding and reduces support burden.",
            action_items=action_items,
            estimated_effort=estimated_effort,
            impact=impact,
            related_issues=[issue.title for issue in doc_component.issues],
            metadata={
                "current_score": doc_component.score,
                "target_score": 85,
                "metrics": doc_component.metrics
            }
        )


class ConfigurationRecommendationRule(RecommendationRule):
    """Recommendations for configuration improvements"""
    
    def __init__(self):
        super().__init__("configuration_improvements", HealthCategory.CONFIGURATION, 2)
    
    def applies_to(self, report: HealthReport) -> bool:
        config_component = report.component_scores.get("configuration")
        return config_component is not None and config_component.score < 75
    
    def generate_recommendation(self, report: HealthReport) -> Optional[Recommendation]:
        config_component = report.component_scores.get("configuration")
        if not config_component:
            return None
        
        action_items = []
        estimated_effort = "medium"
        impact = "high"
        
        # Analyze configuration issues
        if config_component.metrics.get("scattered_configs", 0) > 5:
            action_items.append("Consolidate scattered configuration files")
            action_items.append("Implement unified configuration system")
            estimated_effort = "high"
        
        if config_component.metrics.get("duplicate_configs", 0) > 0:
            action_items.append("Remove duplicate configuration settings")
            action_items.append("Create configuration hierarchy")
        
        if config_component.metrics.get("validation_errors", 0) > 0:
            action_items.append("Fix configuration validation errors")
            action_items.append("Add configuration schema validation")
        
        if config_component.metrics.get("security_issues", 0) > 0:
            action_items.append("Remove hardcoded secrets from configuration")
            action_items.append("Use environment variables for sensitive data")
            impact = "high"
        
        if not config_component.metrics.get("has_unified_config", False):
            action_items.append("Implement unified configuration management system")
            action_items.append("Create configuration API for programmatic access")
        
        if not action_items:
            action_items = ["Review and improve configuration management"]
        
        return Recommendation(
            priority=self.priority,
            category=self.category,
            title="Unify Configuration Management",
            description=f"Configuration score is {config_component.score:.1f}/100. Unified configuration improves maintainability and reduces deployment errors.",
            action_items=action_items,
            estimated_effort=estimated_effort,
            impact=impact,
            related_issues=[issue.title for issue in config_component.issues],
            metadata={
                "current_score": config_component.score,
                "target_score": 85,
                "metrics": config_component.metrics
            }
        )


class CodeQualityRecommendationRule(RecommendationRule):
    """Recommendations for code quality improvements"""
    
    def __init__(self):
        super().__init__("code_quality_improvements", HealthCategory.CODE_QUALITY, 4)
    
    def applies_to(self, report: HealthReport) -> bool:
        code_component = report.component_scores.get("code_quality")
        return code_component is not None and code_component.score < 80
    
    def generate_recommendation(self, report: HealthReport) -> Optional[Recommendation]:
        code_component = report.component_scores.get("code_quality")
        if not code_component:
            return None
        
        action_items = []
        estimated_effort = "medium"
        impact = "medium"
        
        # Analyze code quality issues
        if code_component.metrics.get("syntax_errors", 0) > 0:
            action_items.append("Fix syntax errors in Python files")
            action_items.append("Set up IDE with syntax checking")
            impact = "high"
        
        if code_component.metrics.get("high_complexity_functions", 0) > 0:
            action_items.append("Refactor high-complexity functions")
            action_items.append("Break down large functions into smaller ones")
        
        if code_component.metrics.get("code_smells", 0) > 10:
            action_items.append("Address code quality issues (long lines, bare except, etc.)")
            action_items.append("Use code quality tools like pylint or flake8")
        
        if code_component.metrics.get("import_issues", 0) > 0:
            action_items.append("Organize imports according to PEP 8")
            action_items.append("Use tools like isort for import organization")
        
        if code_component.metrics.get("todo_comments", 0) > 20:
            action_items.append("Review and address TODO/FIXME comments")
            action_items.append("Create issues for important TODOs")
        
        # Add external tool recommendations
        if "flake8_issues" not in code_component.metrics:
            action_items.append("Set up automated code quality checking with flake8")
        
        if "pylint_score" not in code_component.metrics:
            action_items.append("Implement pylint for comprehensive code analysis")
        
        if not action_items:
            action_items = ["Improve overall code quality and maintainability"]
        
        return Recommendation(
            priority=self.priority,
            category=self.category,
            title="Enhance Code Quality",
            description=f"Code quality score is {code_component.score:.1f}/100. Better code quality reduces bugs and improves maintainability.",
            action_items=action_items,
            estimated_effort=estimated_effort,
            impact=impact,
            related_issues=[issue.title for issue in code_component.issues],
            metadata={
                "current_score": code_component.score,
                "target_score": 85,
                "metrics": code_component.metrics
            }
        )


class CriticalIssueRecommendationRule(RecommendationRule):
    """Recommendations for addressing critical issues"""
    
    def __init__(self):
        super().__init__("critical_issue_resolution", HealthCategory.TESTS, 1)  # Highest priority
    
    def applies_to(self, report: HealthReport) -> bool:
        return len(report.get_critical_issues()) > 0
    
    def generate_recommendation(self, report: HealthReport) -> Optional[Recommendation]:
        critical_issues = report.get_critical_issues()
        if not critical_issues:
            return None
        
        action_items = []
        
        # Group issues by category
        issues_by_category = {}
        for issue in critical_issues:
            category = issue.category.value
            if category not in issues_by_category:
                issues_by_category[category] = []
            issues_by_category[category].append(issue)
        
        # Create action items for each category
        for category, issues in issues_by_category.items():
            action_items.append(f"Address {len(issues)} critical {category} issues")
            
            # Add specific remediation steps from the first few issues
            for issue in issues[:2]:  # First 2 issues per category
                action_items.extend(issue.remediation_steps[:2])  # First 2 steps per issue
        
        return Recommendation(
            priority=1,  # Highest priority
            category=HealthCategory.TESTS,  # Use tests as default category
            title="Resolve Critical Issues Immediately",
            description=f"Found {len(critical_issues)} critical issues that require immediate attention to prevent system instability.",
            action_items=action_items,
            estimated_effort="high",
            impact="high",
            related_issues=[issue.title for issue in critical_issues],
            metadata={
                "critical_issue_count": len(critical_issues),
                "issues_by_category": {cat: len(issues) for cat, issues in issues_by_category.items()}
            }
        )


class TrendBasedRecommendationRule(RecommendationRule):
    """Recommendations based on health trends"""
    
    def __init__(self):
        super().__init__("trend_based_improvements", HealthCategory.TESTS, 5)
    
    def applies_to(self, report: HealthReport) -> bool:
        return (report.trends.improvement_rate < -2 or 
                len(report.trends.degradation_alerts) > 0)
    
    def generate_recommendation(self, report: HealthReport) -> Optional[Recommendation]:
        action_items = []
        estimated_effort = "medium"
        impact = "medium"
        
        if report.trends.improvement_rate < -2:
            action_items.append("Investigate causes of declining health trend")
            action_items.append("Review recent changes that may have impacted health")
            action_items.append("Implement monitoring to catch issues earlier")
            impact = "high"
        
        if len(report.trends.degradation_alerts) > 0:
            action_items.append("Address degradation alerts:")
            action_items.extend(report.trends.degradation_alerts[:3])
        
        if len(report.trends.score_history) < 10:
            action_items.append("Establish regular health monitoring schedule")
            action_items.append("Build historical health data for better trend analysis")
        
        if not action_items:
            return None
        
        return Recommendation(
            priority=self.priority,
            category=self.category,
            title="Address Negative Health Trends",
            description=f"Health trends show concerning patterns (improvement rate: {report.trends.improvement_rate:.2f}). Proactive action needed.",
            action_items=action_items,
            estimated_effort=estimated_effort,
            impact=impact,
            related_issues=[],
            metadata={
                "improvement_rate": report.trends.improvement_rate,
                "degradation_alerts": report.trends.degradation_alerts,
                "data_points": len(report.trends.score_history)
            }
        )


class RecommendationEngine:
    """
    Main recommendation engine that generates actionable improvement suggestions
    """
    
    def __init__(self, config: Optional[HealthConfig] = None):
        self.config = config or HealthConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize recommendation rules
        self.rules = [
            CriticalIssueRecommendationRule(),
            TestSuiteRecommendationRule(),
            ConfigurationRecommendationRule(),
            DocumentationRecommendationRule(),
            CodeQualityRecommendationRule(),
            TrendBasedRecommendationRule()
        ]
        
        # Recommendation tracking
        self.recommendation_history = []
        self.progress_tracking = {}
    
    def generate_recommendations(self, report: HealthReport) -> List[Recommendation]:
        """
        Generate prioritized recommendations based on health report
        
        Args:
            report: Current health report
            
        Returns:
            List of recommendations sorted by priority
        """
        recommendations = []
        
        for rule in self.rules:
            try:
                if rule.applies_to(report):
                    recommendation = rule.generate_recommendation(report)
                    if recommendation:
                        recommendations.append(recommendation)
            except Exception as e:
                self.logger.error(f"Failed to generate recommendation for rule {rule.name}: {e}")
        
        # Sort by priority (lower number = higher priority)
        recommendations.sort(key=lambda r: r.priority)
        
        # Add recommendation IDs and timestamps
        for i, rec in enumerate(recommendations):
            rec.metadata["recommendation_id"] = f"rec_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}"
            rec.metadata["generated_at"] = datetime.now().isoformat()
        
        # Store in history
        self.recommendation_history.append({
            "timestamp": datetime.now().isoformat(),
            "report_timestamp": report.timestamp.isoformat(),
            "recommendations": [self._recommendation_to_dict(rec) for rec in recommendations]
        })
        
        # Keep only last 50 recommendation sets
        self.recommendation_history = self.recommendation_history[-50:]
        
        return recommendations
    
    def prioritize_recommendations(self, recommendations: List[Recommendation]) -> List[Recommendation]:
        """
        Re-prioritize recommendations based on additional criteria
        
        Args:
            recommendations: List of recommendations to prioritize
            
        Returns:
            Re-prioritized list of recommendations
        """
        # Calculate dynamic priority scores
        for rec in recommendations:
            score = self._calculate_priority_score(rec)
            rec.metadata["priority_score"] = score
        
        # Sort by priority score (higher = more important)
        recommendations.sort(key=lambda r: r.metadata.get("priority_score", 0), reverse=True)
        
        return recommendations
    
    def _calculate_priority_score(self, recommendation: Recommendation) -> float:
        """Calculate dynamic priority score for recommendation"""
        base_score = 10 - recommendation.priority  # Invert priority (1 becomes 9, 5 becomes 5)
        
        # Impact multiplier
        impact_multipliers = {"high": 2.0, "medium": 1.5, "low": 1.0}
        impact_multiplier = impact_multipliers.get(recommendation.impact, 1.0)
        
        # Effort penalty (easier tasks get slight boost)
        effort_penalties = {"low": 1.2, "medium": 1.0, "high": 0.8}
        effort_penalty = effort_penalties.get(recommendation.estimated_effort, 1.0)
        
        # Category weights
        category_weights = {
            HealthCategory.TESTS: 1.5,
            HealthCategory.CONFIGURATION: 1.3,
            HealthCategory.DOCUMENTATION: 1.0,
            HealthCategory.CODE_QUALITY: 1.1,
            HealthCategory.SECURITY: 2.0,
            HealthCategory.PERFORMANCE: 1.2
        }
        category_weight = category_weights.get(recommendation.category, 1.0)
        
        return base_score * impact_multiplier * effort_penalty * category_weight
    
    def track_recommendation_progress(self, recommendation_id: str, status: str, notes: str = "") -> None:
        """
        Track progress on a recommendation
        
        Args:
            recommendation_id: ID of the recommendation
            status: Current status (not_started, in_progress, completed, cancelled)
            notes: Optional progress notes
        """
        self.progress_tracking[recommendation_id] = {
            "status": status,
            "notes": notes,
            "last_updated": datetime.now().isoformat()
        }
    
    def get_recommendation_progress(self, recommendation_id: str) -> Optional[Dict[str, Any]]:
        """Get progress information for a recommendation"""
        return self.progress_tracking.get(recommendation_id)
    
    def generate_implementation_plan(self, recommendations: List[Recommendation]) -> Dict[str, Any]:
        """
        Generate an implementation plan for recommendations
        
        Args:
            recommendations: List of recommendations to plan
            
        Returns:
            Implementation plan with phases and timelines
        """
        # Group recommendations by effort and priority
        high_priority_low_effort = []
        high_priority_high_effort = []
        medium_priority = []
        low_priority = []
        
        for rec in recommendations:
            if rec.priority <= 2:
                if rec.estimated_effort == "low":
                    high_priority_low_effort.append(rec)
                else:
                    high_priority_high_effort.append(rec)
            elif rec.priority <= 4:
                medium_priority.append(rec)
            else:
                low_priority.append(rec)
        
        # Create implementation phases
        phases = []
        
        if high_priority_low_effort:
            phases.append({
                "phase": 1,
                "name": "Quick Wins",
                "description": "High-impact, low-effort improvements",
                "recommendations": [self._recommendation_to_dict(rec) for rec in high_priority_low_effort],
                "estimated_duration": "1-2 weeks",
                "parallel_execution": True
            })
        
        if high_priority_high_effort:
            phases.append({
                "phase": 2,
                "name": "Critical Improvements",
                "description": "High-priority improvements requiring significant effort",
                "recommendations": [self._recommendation_to_dict(rec) for rec in high_priority_high_effort],
                "estimated_duration": "3-6 weeks",
                "parallel_execution": False
            })
        
        if medium_priority:
            phases.append({
                "phase": 3,
                "name": "Quality Enhancements",
                "description": "Medium-priority quality improvements",
                "recommendations": [self._recommendation_to_dict(rec) for rec in medium_priority],
                "estimated_duration": "2-4 weeks",
                "parallel_execution": True
            })
        
        if low_priority:
            phases.append({
                "phase": 4,
                "name": "Future Improvements",
                "description": "Lower-priority improvements for future consideration",
                "recommendations": [self._recommendation_to_dict(rec) for rec in low_priority],
                "estimated_duration": "Ongoing",
                "parallel_execution": True
            })
        
        return {
            "generated_at": datetime.now().isoformat(),
            "total_recommendations": len(recommendations),
            "phases": phases,
            "estimated_total_duration": self._estimate_total_duration(phases),
            "success_metrics": self._define_success_metrics(recommendations)
        }
    
    def _estimate_total_duration(self, phases: List[Dict[str, Any]]) -> str:
        """Estimate total implementation duration"""
        # Simple estimation based on phase count and complexity
        if len(phases) <= 2:
            return "4-8 weeks"
        elif len(phases) <= 3:
            return "8-12 weeks"
        else:
            return "12-16 weeks"
    
    def _define_success_metrics(self, recommendations: List[Recommendation]) -> List[str]:
        """Define success metrics for implementation plan"""
        metrics = []
        
        # Category-based metrics
        categories = set(rec.category for rec in recommendations)
        
        if HealthCategory.TESTS in categories:
            metrics.append("Test suite health score > 85")
            metrics.append("Test pass rate > 90%")
            metrics.append("Code coverage > 75%")
        
        if HealthCategory.CONFIGURATION in categories:
            metrics.append("Configuration health score > 85")
            metrics.append("Zero configuration validation errors")
        
        if HealthCategory.DOCUMENTATION in categories:
            metrics.append("Documentation health score > 80")
            metrics.append("All essential documentation present")
        
        if HealthCategory.CODE_QUALITY in categories:
            metrics.append("Code quality score > 80")
            metrics.append("Zero syntax errors")
        
        # Overall metrics
        metrics.append("Overall health score > 80")
        metrics.append("Zero critical issues")
        metrics.append("Positive health trend for 4+ weeks")
        
        return metrics
    
    def _recommendation_to_dict(self, recommendation: Recommendation) -> Dict[str, Any]:
        """Convert recommendation to dictionary format"""
        return {
            "priority": recommendation.priority,
            "category": recommendation.category.value,
            "title": recommendation.title,
            "description": recommendation.description,
            "action_items": recommendation.action_items,
            "estimated_effort": recommendation.estimated_effort,
            "impact": recommendation.impact,
            "related_issues": recommendation.related_issues,
            "metadata": recommendation.metadata
        }
    
    def add_custom_rule(self, rule: RecommendationRule) -> None:
        """Add a custom recommendation rule"""
        self.rules.append(rule)
        self.logger.info(f"Added custom recommendation rule: {rule.name}")
    
    def get_recommendation_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent recommendation history"""
        return self.recommendation_history[-limit:]
    
    def export_recommendations(self, recommendations: List[Recommendation], format_type: str = "json") -> str:
        """
        Export recommendations in specified format
        
        Args:
            recommendations: List of recommendations to export
            format_type: Export format ("json", "markdown", "csv")
            
        Returns:
            Formatted string representation
        """
        if format_type == "json":
            return json.dumps([self._recommendation_to_dict(rec) for rec in recommendations], indent=2)
        
        elif format_type == "markdown":
            return self._export_markdown(recommendations)
        
        elif format_type == "csv":
            return self._export_csv(recommendations)
        
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    def _export_markdown(self, recommendations: List[Recommendation]) -> str:
        """Export recommendations as Markdown"""
        lines = ["# Project Health Recommendations", ""]
        
        for i, rec in enumerate(recommendations, 1):
            lines.extend([
                f"## {i}. {rec.title}",
                f"**Priority:** {rec.priority} | **Category:** {rec.category.value} | **Impact:** {rec.impact} | **Effort:** {rec.estimated_effort}",
                "",
                rec.description,
                "",
                "### Action Items:",
                ""
            ])
            
            for item in rec.action_items:
                lines.append(f"- {item}")
            
            if rec.related_issues:
                lines.extend([
                    "",
                    "### Related Issues:",
                    ""
                ])
                for issue in rec.related_issues:
                    lines.append(f"- {issue}")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def _export_csv(self, recommendations: List[Recommendation]) -> str:
        """Export recommendations as CSV"""
        import csv
        from io import StringIO
        
        output = StringIO()
        writer = csv.writer(output)
        
        # Header
        writer.writerow([
            "Priority", "Category", "Title", "Description", "Impact", 
            "Effort", "Action Items", "Related Issues"
        ])
        
        # Data rows
        for rec in recommendations:
            writer.writerow([
                rec.priority,
                rec.category.value,
                rec.title,
                rec.description,
                rec.impact,
                rec.estimated_effort,
                "; ".join(rec.action_items),
                "; ".join(rec.related_issues)
            ])
        
        return output.getvalue()