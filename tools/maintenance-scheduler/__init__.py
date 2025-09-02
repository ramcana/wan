"""
Automated Maintenance Scheduling System

A comprehensive system for scheduling and executing automated maintenance tasks
with intelligent prioritization, rollback capabilities, and detailed tracking.

Key Components:
- MaintenanceScheduler: Core scheduling and execution engine
- TaskManager: Task lifecycle management
- PriorityEngine: Impact and effort analysis for task prioritization
- HistoryTracker: Execution history and metrics tracking
- RollbackManager: Safe operation rollbacks with backup management

Usage:
    from tools.maintenance_scheduler import MaintenanceScheduler, MaintenanceTask
    from tools.maintenance_scheduler.models import TaskCategory, TaskPriority
    
    # Create scheduler
    scheduler = MaintenanceScheduler()
    
    # Create task
    task = MaintenanceTask(
        name="Code Quality Check",
        category=TaskCategory.CODE_QUALITY,
        priority=TaskPriority.HIGH
    )
    
    # Add and run task
    scheduler.add_task(task)
    scheduler.start()
"""

from .scheduler import MaintenanceScheduler
from .task_manager import MaintenanceTaskManager
from .priority_engine import TaskPriorityEngine, ImpactAnalysis, EffortAnalysis
from .history_tracker import MaintenanceHistoryTracker
from .rollback_manager import RollbackManager
from .models import (
    MaintenanceTask,
    MaintenanceResult,
    TaskSchedule,
    MaintenanceHistory,
    MaintenanceMetrics,
    TaskPriority,
    TaskStatus,
    TaskCategory
)

__version__ = "1.0.0"
__author__ = "WAN22 Development Team"

__all__ = [
    # Core components
    'MaintenanceScheduler',
    'MaintenanceTaskManager',
    'TaskPriorityEngine',
    'MaintenanceHistoryTracker',
    'RollbackManager',
    
    # Data models
    'MaintenanceTask',
    'MaintenanceResult',
    'TaskSchedule',
    'MaintenanceHistory',
    'MaintenanceMetrics',
    
    # Enums
    'TaskPriority',
    'TaskStatus',
    'TaskCategory',
    
    # Analysis models
    'ImpactAnalysis',
    'EffortAnalysis'
]