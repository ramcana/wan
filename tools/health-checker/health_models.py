"""
Health monitoring data models and enums
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional
from pathlib import Path


class Severity(Enum):
    """Issue severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class HealthCategory(Enum):
    """Health check categories"""
    TESTS = "tests"
    DOCUMENTATION = "documentation"
    CONFIGURATION = "configuration"
    CODE_QUALITY = "code_quality"
    PERFORMANCE = "performance"
    SECURITY = "security"
    DEPENDENCIES = "dependencies"


@dataclass
class HealthIssue:
    """Represents a project health issue"""
    severity: Severity
    category: HealthCategory
    title: str
    description: str
    affected_components: List[str]
    remediation_steps: List[str]
    file_path: Optional[Path] = None
    line_number: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Recommendation:
    """Actionable recommendation for improvement"""
    priority: int  # 1-10, 1 being highest priority
    category: HealthCategory
    title: str
    description: str
    action_items: List[str]
    estimated_effort: str  # "low", "medium", "high"
    impact: str  # "low", "medium", "high"
    related_issues: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComponentHealth:
    """Health status for a specific component"""
    component_name: str
    category: HealthCategory
    score: float  # 0-100
    status: str  # "healthy", "warning", "critical"
    issues: List[HealthIssue] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    last_checked: datetime = field(default_factory=datetime.now)


@dataclass
class HealthTrends:
    """Historical health trend data"""
    score_history: List[tuple[datetime, float]] = field(default_factory=list)
    issue_trends: Dict[str, List[tuple[datetime, int]]] = field(default_factory=dict)
    improvement_rate: float = 0.0
    degradation_alerts: List[str] = field(default_factory=list)


@dataclass
class HealthReport:
    """Comprehensive project health report"""
    timestamp: datetime
    overall_score: float  # 0-100
    component_scores: Dict[str, ComponentHealth]
    issues: List[HealthIssue]
    recommendations: List[Recommendation]
    trends: HealthTrends
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_issues_by_severity(self, severity: Severity) -> List[HealthIssue]:
        """Get all issues of a specific severity"""
        return [issue for issue in self.issues if issue.severity == severity]
    
    def get_issues_by_category(self, category: HealthCategory) -> List[HealthIssue]:
        """Get all issues in a specific category"""
        return [issue for issue in self.issues if issue.category == category]
    
    def get_critical_issues(self) -> List[HealthIssue]:
        """Get all critical issues"""
        return self.get_issues_by_severity(Severity.CRITICAL)
    
    def get_component_score(self, component: str) -> float:
        """Get score for a specific component"""
        return self.component_scores.get(component, ComponentHealth("", HealthCategory.TESTS, 0.0, "")).score


@dataclass
class HealthConfig:
    """Configuration for health monitoring"""
    # Scoring weights
    test_weight: float = 0.3
    documentation_weight: float = 0.2
    configuration_weight: float = 0.2
    code_quality_weight: float = 0.15
    performance_weight: float = 0.1
    security_weight: float = 0.05
    
    # Thresholds
    critical_threshold: float = 50.0  # Below this is critical
    warning_threshold: float = 75.0   # Below this is warning
    
    # Check intervals (in minutes)
    full_check_interval: int = 60
    quick_check_interval: int = 15
    
    # Paths
    project_root: Path = Path(".")
    test_directory: Path = Path("tests")
    docs_directory: Path = Path("docs")
    config_directory: Path = Path("config")
    
    # Notification settings
    enable_notifications: bool = True
    notification_channels: List[str] = field(default_factory=lambda: ["console"])
    
    # Performance settings
    max_check_duration: int = 300  # 5 minutes max
    parallel_checks: bool = True
    max_workers: int = 4