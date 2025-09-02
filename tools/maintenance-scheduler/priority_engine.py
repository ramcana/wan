"""
Priority engine for maintenance tasks based on impact and effort analysis.
"""

import logging
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from models import MaintenanceTask, TaskPriority, TaskCategory, MaintenanceHistory


@dataclass
class ImpactAnalysis:
    """Analysis of task impact on project health."""
    
    code_quality_impact: float = 0.0  # 0-100
    security_impact: float = 0.0      # 0-100
    performance_impact: float = 0.0   # 0-100
    maintainability_impact: float = 0.0  # 0-100
    user_experience_impact: float = 0.0  # 0-100
    
    # Calculated fields
    total_impact: float = 0.0
    impact_category: str = "low"  # low, medium, high, critical


@dataclass
class EffortAnalysis:
    """Analysis of effort required to complete a task."""
    
    estimated_duration_minutes: int = 30
    complexity_score: float = 1.0  # 1-10 scale
    risk_score: float = 1.0        # 1-10 scale
    resource_requirements: List[str] = None
    
    # Dependencies
    dependency_count: int = 0
    blocking_tasks_count: int = 0
    
    # Historical data
    average_duration_minutes: Optional[int] = None
    success_rate: float = 1.0
    
    def __post_init__(self):
        if self.resource_requirements is None:
            self.resource_requirements = []


class TaskPriorityEngine:
    """
    Engine for calculating task priorities based on impact and effort analysis.
    
    Uses a sophisticated scoring system that considers:
    - Business impact (security, performance, quality)
    - Technical debt reduction
    - Effort required
    - Historical success rates
    - Dependencies and blocking relationships
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Scoring weights (can be configured)
        self.weights = {
            'impact': self.config.get('impact_weight', 0.4),
            'urgency': self.config.get('urgency_weight', 0.3),
            'effort': self.config.get('effort_weight', 0.2),
            'success_rate': self.config.get('success_rate_weight', 0.1)
        }
        
        # Category impact multipliers
        self.category_multipliers = {
            TaskCategory.SECURITY_SCAN: 2.0,
            TaskCategory.CODE_QUALITY: 1.5,
            TaskCategory.TEST_MAINTENANCE: 1.4,
            TaskCategory.PERFORMANCE_OPTIMIZATION: 1.3,
            TaskCategory.CONFIG_CLEANUP: 1.2,
            TaskCategory.DOCUMENTATION: 1.0,
            TaskCategory.DEPENDENCY_UPDATE: 1.1
        }
        
        self.logger.info("TaskPriorityEngine initialized")
    
    def get_priority_score(self, task: MaintenanceTask, 
                          history: Optional[List[MaintenanceHistory]] = None) -> float:
        """
        Calculate comprehensive priority score for a task.
        
        Returns a score from 0-100 where higher scores indicate higher priority.
        """
        impact_analysis = self.analyze_impact(task)
        effort_analysis = self.analyze_effort(task, history)
        urgency_score = self.calculate_urgency(task)
        
        # Calculate component scores
        impact_score = impact_analysis.total_impact
        effort_score = self._calculate_effort_score(effort_analysis)
        success_score = effort_analysis.success_rate * 100
        
        # Apply category multiplier
        category_multiplier = self.category_multipliers.get(task.category, 1.0)
        
        # Calculate weighted score
        priority_score = (
            impact_score * self.weights['impact'] +
            urgency_score * self.weights['urgency'] +
            effort_score * self.weights['effort'] +
            success_score * self.weights['success_rate']
        ) * category_multiplier
        
        # Apply priority level adjustment
        priority_adjustment = self._get_priority_adjustment(task.priority)
        priority_score *= priority_adjustment
        
        # Ensure score is within bounds
        priority_score = max(0, min(100, priority_score))
        
        self.logger.debug(
            f"Priority score for {task.name}: {priority_score:.2f} "
            f"(impact: {impact_score:.1f}, urgency: {urgency_score:.1f}, "
            f"effort: {effort_score:.1f}, success: {success_score:.1f})"
        )
        
        return priority_score
    
    def analyze_impact(self, task: MaintenanceTask) -> ImpactAnalysis:
        """Analyze the potential impact of completing a task."""
        analysis = ImpactAnalysis()
        
        # Base impact by category
        category_impacts = {
            TaskCategory.SECURITY_SCAN: {
                'security_impact': 90,
                'maintainability_impact': 60,
                'user_experience_impact': 70
            },
            TaskCategory.CODE_QUALITY: {
                'code_quality_impact': 85,
                'maintainability_impact': 80,
                'performance_impact': 40
            },
            TaskCategory.TEST_MAINTENANCE: {
                'code_quality_impact': 70,
                'maintainability_impact': 85,
                'user_experience_impact': 50
            },
            TaskCategory.PERFORMANCE_OPTIMIZATION: {
                'performance_impact': 90,
                'user_experience_impact': 80,
                'maintainability_impact': 60
            },
            TaskCategory.CONFIG_CLEANUP: {
                'maintainability_impact': 75,
                'code_quality_impact': 60,
                'performance_impact': 30
            },
            TaskCategory.DOCUMENTATION: {
                'maintainability_impact': 70,
                'user_experience_impact': 40,
                'code_quality_impact': 30
            },
            TaskCategory.DEPENDENCY_UPDATE: {
                'security_impact': 60,
                'maintainability_impact': 50,
                'performance_impact': 40
            }
        }
        
        # Apply base impacts
        base_impacts = category_impacts.get(task.category, {})
        for field, value in base_impacts.items():
            setattr(analysis, field, value)
        
        # Adjust based on task configuration
        config_adjustments = self._analyze_config_impact(task.config)
        for field, adjustment in config_adjustments.items():
            current_value = getattr(analysis, field, 0)
            setattr(analysis, field, min(100, current_value + adjustment))
        
        # Calculate total impact
        impacts = [
            analysis.code_quality_impact,
            analysis.security_impact,
            analysis.performance_impact,
            analysis.maintainability_impact,
            analysis.user_experience_impact
        ]
        
        # Weighted average with emphasis on security and quality
        weights = [0.25, 0.3, 0.2, 0.15, 0.1]
        analysis.total_impact = sum(impact * weight for impact, weight in zip(impacts, weights))
        
        # Determine impact category
        if analysis.total_impact >= 80:
            analysis.impact_category = "critical"
        elif analysis.total_impact >= 60:
            analysis.impact_category = "high"
        elif analysis.total_impact >= 40:
            analysis.impact_category = "medium"
        else:
            analysis.impact_category = "low"
        
        return analysis
    
    def analyze_effort(self, task: MaintenanceTask, 
                      history: Optional[List[MaintenanceHistory]] = None) -> EffortAnalysis:
        """Analyze the effort required to complete a task."""
        analysis = EffortAnalysis()
        
        # Base effort by category
        category_efforts = {
            TaskCategory.SECURITY_SCAN: {'duration': 20, 'complexity': 3, 'risk': 2},
            TaskCategory.CODE_QUALITY: {'duration': 45, 'complexity': 5, 'risk': 3},
            TaskCategory.TEST_MAINTENANCE: {'duration': 60, 'complexity': 6, 'risk': 4},
            TaskCategory.PERFORMANCE_OPTIMIZATION: {'duration': 90, 'complexity': 8, 'risk': 6},
            TaskCategory.CONFIG_CLEANUP: {'duration': 30, 'complexity': 4, 'risk': 3},
            TaskCategory.DOCUMENTATION: {'duration': 30, 'complexity': 3, 'risk': 2},
            TaskCategory.DEPENDENCY_UPDATE: {'duration': 25, 'complexity': 4, 'risk': 5}
        }
        
        base_effort = category_efforts.get(task.category, {'duration': 30, 'complexity': 3, 'risk': 3})
        analysis.estimated_duration_minutes = base_effort['duration']
        analysis.complexity_score = base_effort['complexity']
        analysis.risk_score = base_effort['risk']
        
        # Adjust based on task timeout
        if task.timeout_minutes != 30:  # Default timeout
            duration_ratio = task.timeout_minutes / 30
            analysis.estimated_duration_minutes = int(analysis.estimated_duration_minutes * duration_ratio)
            analysis.complexity_score = min(10, analysis.complexity_score * math.sqrt(duration_ratio))
        
        # Analyze dependencies
        analysis.dependency_count = len(task.depends_on)
        analysis.blocking_tasks_count = len(task.blocks)
        
        # Adjust complexity based on dependencies
        if analysis.dependency_count > 0:
            analysis.complexity_score += min(3, analysis.dependency_count * 0.5)
        
        # Analyze historical data
        if history:
            durations = [h.result.duration_seconds / 60 for h in history if h.result.duration_seconds > 0]
            successes = [h.result.success for h in history]
            
            if durations:
                analysis.average_duration_minutes = int(sum(durations) / len(durations))
                # Adjust estimated duration based on history
                analysis.estimated_duration_minutes = int(
                    (analysis.estimated_duration_minutes + analysis.average_duration_minutes) / 2
                )
            
            if successes:
                analysis.success_rate = sum(successes) / len(successes)
                # Adjust risk based on success rate
                if analysis.success_rate < 0.8:
                    analysis.risk_score += (1 - analysis.success_rate) * 3
        
        # Ensure scores are within bounds
        analysis.complexity_score = max(1, min(10, analysis.complexity_score))
        analysis.risk_score = max(1, min(10, analysis.risk_score))
        analysis.success_rate = max(0, min(1, analysis.success_rate))
        
        return analysis
    
    def calculate_urgency(self, task: MaintenanceTask) -> float:
        """Calculate urgency score based on timing and dependencies."""
        urgency_score = 50.0  # Base urgency
        
        now = datetime.now()
        
        # Time-based urgency
        if task.last_run:
            days_since_last_run = (now - task.last_run).days
            
            # Increase urgency based on time since last run
            if days_since_last_run > 30:
                urgency_score += 30
            elif days_since_last_run > 14:
                urgency_score += 20
            elif days_since_last_run > 7:
                urgency_score += 10
        else:
            # Never run before - moderate urgency
            urgency_score += 15
        
        # Schedule-based urgency
        if task.next_run:
            time_until_scheduled = (task.next_run - now).total_seconds()
            
            if time_until_scheduled < 0:
                # Overdue
                hours_overdue = abs(time_until_scheduled) / 3600
                urgency_score += min(40, hours_overdue * 2)
            elif time_until_scheduled < 3600:  # Less than 1 hour
                urgency_score += 20
            elif time_until_scheduled < 86400:  # Less than 1 day
                urgency_score += 10
        
        # Dependency-based urgency
        blocking_count = len(task.blocks)
        if blocking_count > 0:
            urgency_score += min(20, blocking_count * 5)
        
        # Priority-based urgency
        priority_urgency = {
            TaskPriority.CRITICAL: 40,
            TaskPriority.HIGH: 20,
            TaskPriority.MEDIUM: 0,
            TaskPriority.LOW: -10
        }
        urgency_score += priority_urgency.get(task.priority, 0)
        
        return max(0, min(100, urgency_score))
    
    def get_recommended_execution_order(self, tasks: List[MaintenanceTask],
                                      history_map: Optional[Dict[str, List[MaintenanceHistory]]] = None) -> List[MaintenanceTask]:
        """Get tasks in recommended execution order based on priority scores."""
        if not tasks:
            return []
        
        # Calculate priority scores for all tasks
        task_scores = []
        for task in tasks:
            history = history_map.get(task.id, []) if history_map else None
            score = self.get_priority_score(task, history)
            task_scores.append((task, score))
        
        # Sort by priority score (descending)
        task_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [task for task, score in task_scores]
    
    def _calculate_effort_score(self, effort_analysis: EffortAnalysis) -> float:
        """Convert effort analysis to a score (higher score = less effort)."""
        # Normalize duration (assume max reasonable duration is 180 minutes)
        duration_score = max(0, 100 - (effort_analysis.estimated_duration_minutes / 180) * 100)
        
        # Normalize complexity (1-10 scale)
        complexity_score = max(0, 100 - ((effort_analysis.complexity_score - 1) / 9) * 100)
        
        # Normalize risk (1-10 scale)
        risk_score = max(0, 100 - ((effort_analysis.risk_score - 1) / 9) * 100)
        
        # Dependency penalty
        dependency_penalty = min(30, effort_analysis.dependency_count * 5)
        
        # Calculate weighted effort score
        effort_score = (
            duration_score * 0.4 +
            complexity_score * 0.3 +
            risk_score * 0.3
        ) - dependency_penalty
        
        return max(0, min(100, effort_score))
    
    def _get_priority_adjustment(self, priority: TaskPriority) -> float:
        """Get priority level adjustment multiplier."""
        adjustments = {
            TaskPriority.CRITICAL: 1.5,
            TaskPriority.HIGH: 1.2,
            TaskPriority.MEDIUM: 1.0,
            TaskPriority.LOW: 0.8
        }
        return adjustments.get(priority, 1.0)
    
    def _analyze_config_impact(self, config: Dict) -> Dict[str, float]:
        """Analyze task configuration to determine impact adjustments."""
        adjustments = {}
        
        # Security-related configurations
        security_configs = ['scan_vulnerabilities', 'check_licenses', 'security_audit']
        if any(config.get(key, False) for key in security_configs):
            adjustments['security_impact'] = 20
        
        # Performance-related configurations
        performance_configs = ['optimize_imports', 'profile_code', 'analyze_bottlenecks']
        if any(config.get(key, False) for key in performance_configs):
            adjustments['performance_impact'] = 15
        
        # Quality-related configurations
        quality_configs = ['fix_formatting', 'check_type_hints', 'analyze_complexity']
        if any(config.get(key, False) for key in quality_configs):
            adjustments['code_quality_impact'] = 15
        
        # Maintenance-related configurations
        maintenance_configs = ['update_fixtures', 'validate_configs', 'check_links']
        if any(config.get(key, False) for key in maintenance_configs):
            adjustments['maintainability_impact'] = 10
        
        return adjustments