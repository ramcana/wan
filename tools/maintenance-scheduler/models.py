"""
Data models for the automated maintenance scheduling system.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
import uuid


class TaskPriority(Enum):
    """Priority levels for maintenance tasks."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TaskStatus(Enum):
    """Status of maintenance tasks."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ROLLED_BACK = "rolled_back"


class TaskCategory(Enum):
    """Categories of maintenance tasks."""
    CODE_QUALITY = "code_quality"
    TEST_MAINTENANCE = "test_maintenance"
    CONFIG_CLEANUP = "config_cleanup"
    DOCUMENTATION = "documentation"
    DEPENDENCY_UPDATE = "dependency_update"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    SECURITY_SCAN = "security_scan"


@dataclass
class MaintenanceTask:
    """Represents a maintenance task to be scheduled and executed."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    category: TaskCategory = TaskCategory.CODE_QUALITY
    priority: TaskPriority = TaskPriority.MEDIUM
    
    # Scheduling
    schedule_cron: Optional[str] = None  # Cron expression for recurring tasks
    next_run: Optional[datetime] = None
    last_run: Optional[datetime] = None
    
    # Execution
    executor: Optional[Callable] = None
    timeout_minutes: int = 30
    retry_count: int = 3
    retry_delay_minutes: int = 5
    
    # Dependencies
    depends_on: List[str] = field(default_factory=list)  # Task IDs this depends on
    blocks: List[str] = field(default_factory=list)  # Task IDs this blocks
    
    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)
    environment_requirements: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    created_by: str = "system"
    tags: List[str] = field(default_factory=list)
    
    # Rollback
    rollback_enabled: bool = True
    rollback_data: Optional[Dict[str, Any]] = None


@dataclass
class MaintenanceResult:
    """Result of a maintenance task execution."""
    
    task_id: str
    status: TaskStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    
    # Results
    success: bool = False
    output: str = ""
    error_message: Optional[str] = None
    
    # Metrics
    duration_seconds: float = 0.0
    files_modified: int = 0
    lines_changed: int = 0
    issues_fixed: int = 0
    
    # Impact analysis
    impact_score: float = 0.0  # 0-100 scale
    quality_improvement: float = 0.0  # Percentage improvement
    
    # Rollback info
    rollback_available: bool = False
    rollback_data: Optional[Dict[str, Any]] = None
    
    # Logs and artifacts
    log_file: Optional[str] = None
    artifacts: List[str] = field(default_factory=list)


@dataclass
class TaskSchedule:
    """Schedule configuration for maintenance tasks."""
    
    task_id: str
    cron_expression: str
    timezone: str = "UTC"
    enabled: bool = True
    
    # Execution windows
    allowed_hours: List[int] = field(default_factory=lambda: list(range(24)))  # 0-23
    allowed_days: List[int] = field(default_factory=lambda: list(range(7)))   # 0-6 (Mon-Sun)
    
    # Constraints
    max_concurrent: int = 1
    cooldown_minutes: int = 60
    
    # Notifications
    notify_on_success: bool = False
    notify_on_failure: bool = True
    notification_channels: List[str] = field(default_factory=list)


@dataclass
class MaintenanceHistory:
    """Historical record of maintenance operations."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str = ""
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Execution details
    result: MaintenanceResult = field(default_factory=lambda: MaintenanceResult(""))
    
    # Context
    triggered_by: str = "scheduler"  # scheduler, manual, webhook, etc.
    environment: str = "development"
    git_commit: Optional[str] = None
    
    # Rollback tracking
    rolled_back: bool = False
    rollback_reason: Optional[str] = None
    rollback_timestamp: Optional[datetime] = None


@dataclass
class MaintenanceMetrics:
    """Metrics for maintenance system performance."""
    
    # Task execution metrics
    total_tasks_run: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    average_duration_seconds: float = 0.0
    
    # Quality metrics
    total_issues_fixed: int = 0
    total_files_modified: int = 0
    total_lines_changed: int = 0
    average_quality_improvement: float = 0.0
    
    # System health
    system_uptime_hours: float = 0.0
    last_successful_run: Optional[datetime] = None
    consecutive_failures: int = 0
    
    # Time period
    period_start: datetime = field(default_factory=datetime.now)
    period_end: datetime = field(default_factory=datetime.now)
