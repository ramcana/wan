"""
Maintenance reporting data models and types.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
import json


class MaintenanceOperationType(Enum):
    """Types of maintenance operations."""
    TEST_REPAIR = "test_repair"
    CODE_CLEANUP = "code_cleanup"
    DOCUMENTATION_UPDATE = "documentation_update"
    CONFIGURATION_CONSOLIDATION = "configuration_consolidation"
    QUALITY_IMPROVEMENT = "quality_improvement"
    DEPENDENCY_UPDATE = "dependency_update"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    SECURITY_UPDATE = "security_update"


class MaintenanceStatus(Enum):
    """Status of maintenance operations."""
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ROLLBACK = "rollback"


class ImpactLevel(Enum):
    """Impact level of maintenance operations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class MaintenanceOperation:
    """Individual maintenance operation record."""
    id: str
    operation_type: MaintenanceOperationType
    title: str
    description: str
    status: MaintenanceStatus
    impact_level: ImpactLevel
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[int] = None
    success_metrics: Dict[str, Any] = field(default_factory=dict)
    error_details: Optional[str] = None
    rollback_info: Optional[Dict[str, Any]] = None
    files_affected: List[str] = field(default_factory=list)
    components_affected: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'operation_type': self.operation_type.value,
            'title': self.title,
            'description': self.description,
            'status': self.status.value,
            'impact_level': self.impact_level.value,
            'started_at': self.started_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'duration_seconds': self.duration_seconds,
            'success_metrics': self.success_metrics,
            'error_details': self.error_details,
            'rollback_info': self.rollback_info,
            'files_affected': self.files_affected,
            'components_affected': self.components_affected
        }


@dataclass
class MaintenanceImpactAnalysis:
    """Analysis of maintenance operation impact."""
    operation_id: str
    before_metrics: Dict[str, float]
    after_metrics: Dict[str, float]
    improvements: Dict[str, float]
    regressions: Dict[str, float]
    overall_impact_score: float
    impact_summary: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'operation_id': self.operation_id,
            'before_metrics': self.before_metrics,
            'after_metrics': self.after_metrics,
            'improvements': self.improvements,
            'regressions': self.regressions,
            'overall_impact_score': self.overall_impact_score,
            'impact_summary': self.impact_summary,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class MaintenanceRecommendation:
    """Maintenance recommendation based on project analysis."""
    id: str
    title: str
    description: str
    operation_type: MaintenanceOperationType
    priority: ImpactLevel
    estimated_effort_hours: float
    estimated_impact_score: float
    prerequisites: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    benefits: List[str] = field(default_factory=list)
    suggested_schedule: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'operation_type': self.operation_type.value,
            'priority': self.priority.value,
            'estimated_effort_hours': self.estimated_effort_hours,
            'estimated_impact_score': self.estimated_impact_score,
            'prerequisites': self.prerequisites,
            'risks': self.risks,
            'benefits': self.benefits,
            'suggested_schedule': self.suggested_schedule,
            'created_at': self.created_at.isoformat()
        }


@dataclass
class MaintenanceScheduleOptimization:
    """Maintenance scheduling optimization analysis."""
    recommended_schedule: List[str]  # Operation IDs in recommended order
    scheduling_rationale: Dict[str, str]  # Operation ID -> reason for scheduling
    resource_requirements: Dict[str, float]  # Resource type -> hours needed
    risk_mitigation_plan: List[str]
    estimated_total_duration_hours: float
    optimal_time_windows: List[Dict[str, Any]]
    dependencies: Dict[str, List[str]]  # Operation ID -> list of prerequisite operation IDs
    generated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'recommended_schedule': self.recommended_schedule,
            'scheduling_rationale': self.scheduling_rationale,
            'resource_requirements': self.resource_requirements,
            'risk_mitigation_plan': self.risk_mitigation_plan,
            'estimated_total_duration_hours': self.estimated_total_duration_hours,
            'optimal_time_windows': self.optimal_time_windows,
            'dependencies': self.dependencies,
            'generated_at': self.generated_at.isoformat()
        }


@dataclass
class MaintenanceReport:
    """Comprehensive maintenance report."""
    report_id: str
    report_type: str  # "daily", "weekly", "monthly", "operation_summary"
    period_start: datetime
    period_end: datetime
    operations: List[MaintenanceOperation]
    impact_analyses: List[MaintenanceImpactAnalysis]
    recommendations: List[MaintenanceRecommendation]
    schedule_optimization: Optional[MaintenanceScheduleOptimization]
    summary_statistics: Dict[str, Any]
    generated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'report_id': self.report_id,
            'report_type': self.report_type,
            'period_start': self.period_start.isoformat(),
            'period_end': self.period_end.isoformat(),
            'operations': [op.to_dict() for op in self.operations],
            'impact_analyses': [ia.to_dict() for ia in self.impact_analyses],
            'recommendations': [rec.to_dict() for rec in self.recommendations],
            'schedule_optimization': self.schedule_optimization.to_dict() if self.schedule_optimization else None,
            'summary_statistics': self.summary_statistics,
            'generated_at': self.generated_at.isoformat()
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class MaintenanceAuditTrail:
    """Audit trail for maintenance operations."""
    operation_id: str
    timestamp: datetime
    action: str
    user: str
    details: Dict[str, Any]
    before_state: Optional[Dict[str, Any]] = None
    after_state: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'operation_id': self.operation_id,
            'timestamp': self.timestamp.isoformat(),
            'action': self.action,
            'user': self.user,
            'details': self.details,
            'before_state': self.before_state,
            'after_state': self.after_state
        }