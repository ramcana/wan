"""
Quality monitoring data models and types.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
import json


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of quality metrics."""
    TEST_COVERAGE = "test_coverage"
    CODE_COMPLEXITY = "code_complexity"
    DOCUMENTATION_COVERAGE = "documentation_coverage"
    DUPLICATE_CODE = "duplicate_code"
    DEAD_CODE = "dead_code"
    STYLE_VIOLATIONS = "style_violations"
    TYPE_HINT_COVERAGE = "type_hint_coverage"
    PERFORMANCE = "performance"


class TrendDirection(Enum):
    """Trend direction for metrics."""
    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"
    UNKNOWN = "unknown"


@dataclass
class QualityMetric:
    """Individual quality metric data point."""
    metric_type: MetricType
    value: float
    timestamp: datetime
    component: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'metric_type': self.metric_type.value,
            'value': self.value,
            'timestamp': self.timestamp.isoformat(),
            'component': self.component,
            'details': self.details
        }


@dataclass
class QualityTrend:
    """Quality trend analysis for a metric."""
    metric_type: MetricType
    direction: TrendDirection
    change_rate: float  # Percentage change per time period
    confidence: float  # 0.0 to 1.0
    time_period_days: int
    current_value: float
    previous_value: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'metric_type': self.metric_type.value,
            'direction': self.direction.value,
            'change_rate': self.change_rate,
            'confidence': self.confidence,
            'time_period_days': self.time_period_days,
            'current_value': self.current_value,
            'previous_value': self.previous_value
        }


@dataclass
class QualityAlert:
    """Quality alert for regressions or maintenance needs."""
    id: str
    severity: AlertSeverity
    metric_type: MetricType
    message: str
    description: str
    current_value: float
    threshold_value: float
    component: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'severity': self.severity.value,
            'metric_type': self.metric_type.value,
            'message': self.message,
            'description': self.description,
            'current_value': self.current_value,
            'threshold_value': self.threshold_value,
            'component': self.component,
            'timestamp': self.timestamp.isoformat(),
            'resolved': self.resolved,
            'recommendations': self.recommendations
        }


@dataclass
class QualityThreshold:
    """Quality threshold configuration."""
    metric_type: MetricType
    warning_threshold: float
    critical_threshold: float
    component: Optional[str] = None
    enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'metric_type': self.metric_type.value,
            'warning_threshold': self.warning_threshold,
            'critical_threshold': self.critical_threshold,
            'component': self.component,
            'enabled': self.enabled
        }


@dataclass
class QualityRecommendation:
    """Automated quality improvement recommendation."""
    id: str
    title: str
    description: str
    priority: AlertSeverity
    metric_types: List[MetricType]
    estimated_impact: float  # Expected improvement percentage
    estimated_effort: str  # "low", "medium", "high"
    actions: List[str]
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'priority': self.priority.value,
            'metric_types': [mt.value for mt in self.metric_types],
            'estimated_impact': self.estimated_impact,
            'estimated_effort': self.estimated_effort,
            'actions': self.actions,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class QualityDashboard:
    """Quality monitoring dashboard data."""
    metrics: List[QualityMetric]
    trends: List[QualityTrend]
    alerts: List[QualityAlert]
    recommendations: List[QualityRecommendation]
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'metrics': [m.to_dict() for m in self.metrics],
            'trends': [t.to_dict() for t in self.trends],
            'alerts': [a.to_dict() for a in self.alerts],
            'recommendations': [r.to_dict() for r in self.recommendations],
            'last_updated': self.last_updated.isoformat()
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
