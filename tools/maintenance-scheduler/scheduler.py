"""
Main maintenance scheduler that orchestrates automated maintenance tasks.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from croniter import croniter
import threading
import time

from models import (
    MaintenanceTask, MaintenanceResult, TaskStatus, TaskPriority,
    TaskSchedule, MaintenanceHistory
)
from task_manager import MaintenanceTaskManager
from priority_engine import TaskPriorityEngine
from history_tracker import MaintenanceHistoryTracker
from rollback_manager import RollbackManager


class MaintenanceScheduler:
    """
    Main scheduler for automated maintenance tasks.
    
    Handles task scheduling, execution, prioritization, and monitoring.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.task_manager = MaintenanceTaskManager()
        self.priority_engine = TaskPriorityEngine()
        self.history_tracker = MaintenanceHistoryTracker()
        self.rollback_manager = RollbackManager()
        
        # Scheduler state
        self.running = False
        self.scheduler_thread: Optional[threading.Thread] = None
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.task_schedules: Dict[str, TaskSchedule] = {}
        
        # Configuration
        self.max_concurrent_tasks = self.config.get('max_concurrent_tasks', 3)
        self.check_interval_seconds = self.config.get('check_interval_seconds', 60)
        self.task_timeout_minutes = self.config.get('task_timeout_minutes', 30)
        
        self.logger.info("MaintenanceScheduler initialized")
    
    def start(self) -> None:
        """Start the maintenance scheduler."""
        if self.running:
            self.logger.warning("Scheduler is already running")
            return
        
        self.running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        self.logger.info("Maintenance scheduler started")
    
    def stop(self) -> None:
        """Stop the maintenance scheduler."""
        if not self.running:
            return
        
        self.running = False
        
        # Cancel running tasks
        for task_id, task in self.running_tasks.items():
            if not task.done():
                task.cancel()
                self.logger.info(f"Cancelled running task: {task_id}")
        
        # Wait for scheduler thread to finish
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5)
        
        self.logger.info("Maintenance scheduler stopped")
    
    def add_task(self, task: MaintenanceTask, schedule: Optional[TaskSchedule] = None) -> None:
        """Add a maintenance task to the scheduler."""
        self.task_manager.add_task(task)
        
        if schedule:
            self.task_schedules[task.id] = schedule
            self.logger.info(f"Added scheduled task: {task.name} ({task.id})")
        else:
            self.logger.info(f"Added one-time task: {task.name} ({task.id})")
    
    def remove_task(self, task_id: str) -> bool:
        """Remove a task from the scheduler."""
        # Cancel if currently running
        if task_id in self.running_tasks:
            self.running_tasks[task_id].cancel()
            del self.running_tasks[task_id]
        
        # Remove from schedules
        if task_id in self.task_schedules:
            del self.task_schedules[task_id]
        
        # Remove from task manager
        return self.task_manager.remove_task(task_id)
    
    def run_task_now(self, task_id: str) -> Optional[MaintenanceResult]:
        """Run a specific task immediately."""
        task = self.task_manager.get_task(task_id)
        if not task:
            self.logger.error(f"Task not found: {task_id}")
            return None
        
        return asyncio.run(self._execute_task(task))
    
    def get_next_scheduled_tasks(self, limit: int = 10) -> List[MaintenanceTask]:
        """Get the next tasks scheduled to run."""
        now = datetime.now()
        upcoming_tasks = []
        
        for task_id, schedule in self.task_schedules.items():
            if not schedule.enabled:
                continue
            
            task = self.task_manager.get_task(task_id)
            if not task:
                continue
            
            # Calculate next run time
            if task.next_run and task.next_run <= now + timedelta(hours=24):
                upcoming_tasks.append(task)
        
        # Sort by next run time and priority
        upcoming_tasks.sort(key=lambda t: (
            t.next_run or datetime.max,
            self.priority_engine.get_priority_score(t)
        ), reverse=True)
        
        return upcoming_tasks[:limit]
    
    def get_running_tasks(self) -> List[str]:
        """Get list of currently running task IDs."""
        return list(self.running_tasks.keys())
    
    def get_task_history(self, task_id: str, limit: int = 50) -> List[MaintenanceHistory]:
        """Get execution history for a specific task."""
        return self.history_tracker.get_task_history(task_id, limit)
    
    def _scheduler_loop(self) -> None:
        """Main scheduler loop that runs in a separate thread."""
        self.logger.info("Scheduler loop started")
        
        while self.running:
            try:
                self._check_and_run_tasks()
                time.sleep(self.check_interval_seconds)
            except Exception as e:
                self.logger.error(f"Error in scheduler loop: {e}", exc_info=True)
                time.sleep(self.check_interval_seconds)
        
        self.logger.info("Scheduler loop stopped")
    
    def _check_and_run_tasks(self) -> None:
        """Check for tasks that need to run and execute them."""
        now = datetime.now()
        
        # Clean up completed tasks
        completed_tasks = [
            task_id for task_id, task in self.running_tasks.items()
            if task.done()
        ]
        for task_id in completed_tasks:
            del self.running_tasks[task_id]
        
        # Check if we can run more tasks
        if len(self.running_tasks) >= self.max_concurrent_tasks:
            return
        
        # Find tasks that need to run
        tasks_to_run = self._get_tasks_ready_to_run(now)
        
        # Prioritize and run tasks
        for task in tasks_to_run:
            if len(self.running_tasks) >= self.max_concurrent_tasks:
                break
            
            if self._can_run_task(task, now):
                self._start_task_execution(task)
    
    def _get_tasks_ready_to_run(self, now: datetime) -> List[MaintenanceTask]:
        """Get tasks that are ready to run based on schedule."""
        ready_tasks = []
        
        for task_id, schedule in self.task_schedules.items():
            if not schedule.enabled:
                continue
            
            task = self.task_manager.get_task(task_id)
            if not task or task_id in self.running_tasks:
                continue
            
            # Check if task is due
            if self._is_task_due(task, schedule, now):
                ready_tasks.append(task)
        
        # Sort by priority
        ready_tasks.sort(
            key=lambda t: self.priority_engine.get_priority_score(t),
            reverse=True
        )
        
        return ready_tasks
    
    def _is_task_due(self, task: MaintenanceTask, schedule: TaskSchedule, now: datetime) -> bool:
        """Check if a task is due to run."""
        # Check if we're in allowed time window
        if now.hour not in schedule.allowed_hours:
            return False
        
        if now.weekday() not in schedule.allowed_days:
            return False
        
        # Check cooldown period
        if task.last_run:
            cooldown_end = task.last_run + timedelta(minutes=schedule.cooldown_minutes)
            if now < cooldown_end:
                return False
        
        # Check cron schedule
        if schedule.cron_expression:
            cron = croniter(schedule.cron_expression, task.last_run or now)
            next_run = cron.get_next(datetime)
            return now >= next_run
        
        # Check next_run time
        if task.next_run:
            return now >= task.next_run
        
        return False
    
    def _can_run_task(self, task: MaintenanceTask, now: datetime) -> bool:
        """Check if a task can be run (dependencies, constraints, etc.)."""
        # Check dependencies
        for dep_task_id in task.depends_on:
            if dep_task_id in self.running_tasks:
                return False  # Dependency is still running
            
            # Check if dependency completed successfully recently
            dep_history = self.history_tracker.get_recent_execution(dep_task_id)
            if not dep_history or dep_history.result.status != TaskStatus.COMPLETED:
                return False
        
        # Check environment requirements
        # This would be implemented based on specific environment checks
        
        return True
    
    def _start_task_execution(self, task: MaintenanceTask) -> None:
        """Start executing a task asynchronously."""
        self.logger.info(f"Starting task execution: {task.name} ({task.id})")
        
        # Create async task
        async_task = asyncio.create_task(self._execute_task(task))
        self.running_tasks[task.id] = async_task
        
        # Update task timing
        task.last_run = datetime.now()
        self._update_next_run_time(task)
    
    async def _execute_task(self, task: MaintenanceTask) -> MaintenanceResult:
        """Execute a maintenance task."""
        result = MaintenanceResult(
            task_id=task.id,
            status=TaskStatus.RUNNING,
            started_at=datetime.now()
        )
        
        try:
            self.logger.info(f"Executing task: {task.name}")
            
            # Create rollback point if enabled
            if task.rollback_enabled:
                rollback_data = await self.rollback_manager.create_rollback_point(task)
                result.rollback_data = rollback_data
                result.rollback_available = True
            
            # Execute the task
            if task.executor:
                # Run with timeout
                execution_result = await asyncio.wait_for(
                    task.executor(task.config),
                    timeout=task.timeout_minutes * 60
                )
                
                # Process execution result
                if isinstance(execution_result, dict):
                    result.success = execution_result.get('success', False)
                    result.output = execution_result.get('output', '')
                    result.files_modified = execution_result.get('files_modified', 0)
                    result.lines_changed = execution_result.get('lines_changed', 0)
                    result.issues_fixed = execution_result.get('issues_fixed', 0)
                else:
                    result.success = bool(execution_result)
                    result.output = str(execution_result)
            else:
                self.logger.warning(f"No executor defined for task: {task.name}")
                result.success = False
                result.error_message = "No executor defined"
            
            result.status = TaskStatus.COMPLETED if result.success else TaskStatus.FAILED
            
        except asyncio.TimeoutError:
            result.success = False
            result.status = TaskStatus.FAILED
            result.error_message = f"Task timed out after {task.timeout_minutes} minutes"
            self.logger.error(f"Task timed out: {task.name}")
            
        except Exception as e:
            result.success = False
            result.status = TaskStatus.FAILED
            result.error_message = str(e)
            self.logger.error(f"Task execution failed: {task.name} - {e}", exc_info=True)
        
        finally:
            result.completed_at = datetime.now()
            if result.started_at:
                result.duration_seconds = (result.completed_at - result.started_at).total_seconds()
            
            # Record execution history
            self.history_tracker.record_execution(task, result)
            
            self.logger.info(
                f"Task completed: {task.name} - "
                f"Status: {result.status.value} - "
                f"Duration: {result.duration_seconds:.2f}s"
            )
        
        return result
    
    def _update_next_run_time(self, task: MaintenanceTask) -> None:
        """Update the next run time for a task based on its schedule."""
        schedule = self.task_schedules.get(task.id)
        if not schedule or not schedule.cron_expression:
            return
        
        try:
            cron = croniter(schedule.cron_expression, task.last_run)
            task.next_run = cron.get_next(datetime)
        except Exception as e:
            self.logger.error(f"Error updating next run time for task {task.id}: {e}")