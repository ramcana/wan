"""
Task management system for maintenance tasks.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set
from threading import Lock

from models import MaintenanceTask, TaskCategory, TaskPriority, TaskStatus


class MaintenanceTaskManager:
    """
    Manages maintenance tasks including storage, retrieval, and lifecycle management.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path or "data/maintenance/tasks.json")
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.tasks: Dict[str, MaintenanceTask] = {}
        self.task_lock = Lock()
        
        # Load existing tasks
        self._load_tasks()
        
        self.logger.info(f"TaskManager initialized with {len(self.tasks)} tasks")
    
    def add_task(self, task: MaintenanceTask) -> None:
        """Add a new maintenance task."""
        with self.task_lock:
            task.updated_at = datetime.now()
            self.tasks[task.id] = task
            self._save_tasks()
            
        self.logger.info(f"Added task: {task.name} ({task.id})")
    
    def remove_task(self, task_id: str) -> bool:
        """Remove a maintenance task."""
        with self.task_lock:
            if task_id in self.tasks:
                del self.tasks[task_id]
                self._save_tasks()
                self.logger.info(f"Removed task: {task_id}")
                return True
            
        self.logger.warning(f"Task not found for removal: {task_id}")
        return False
    
    def get_task(self, task_id: str) -> Optional[MaintenanceTask]:
        """Get a specific task by ID."""
        return self.tasks.get(task_id)
    
    def get_all_tasks(self) -> List[MaintenanceTask]:
        """Get all maintenance tasks."""
        return list(self.tasks.values())
    
    def get_tasks_by_category(self, category: TaskCategory) -> List[MaintenanceTask]:
        """Get tasks filtered by category."""
        return [task for task in self.tasks.values() if task.category == category]
    
    def get_tasks_by_priority(self, priority: TaskPriority) -> List[MaintenanceTask]:
        """Get tasks filtered by priority."""
        return [task for task in self.tasks.values() if task.priority == priority]
    
    def get_tasks_by_tags(self, tags: List[str]) -> List[MaintenanceTask]:
        """Get tasks that have any of the specified tags."""
        return [
            task for task in self.tasks.values()
            if any(tag in task.tags for tag in tags)
        ]
    
    def update_task(self, task_id: str, updates: Dict) -> bool:
        """Update a task with new values."""
        with self.task_lock:
            if task_id not in self.tasks:
                self.logger.warning(f"Task not found for update: {task_id}")
                return False
            
            task = self.tasks[task_id]
            
            # Update allowed fields
            allowed_fields = {
                'name', 'description', 'priority', 'schedule_cron', 'timeout_minutes',
                'retry_count', 'retry_delay_minutes', 'config', 'tags', 'rollback_enabled'
            }
            
            for field, value in updates.items():
                if field in allowed_fields and hasattr(task, field):
                    setattr(task, field, value)
            
            task.updated_at = datetime.now()
            self._save_tasks()
            
        self.logger.info(f"Updated task: {task_id}")
        return True
    
    def get_task_dependencies(self, task_id: str) -> Dict[str, List[str]]:
        """Get dependency information for a task."""
        task = self.get_task(task_id)
        if not task:
            return {}
        
        dependencies = {
            'depends_on': task.depends_on.copy(),
            'blocks': task.blocks.copy(),
            'dependents': []  # Tasks that depend on this one
        }
        
        # Find tasks that depend on this one
        for other_task in self.tasks.values():
            if task_id in other_task.depends_on:
                dependencies['dependents'].append(other_task.id)
        
        return dependencies
    
    def validate_task_dependencies(self, task: MaintenanceTask) -> List[str]:
        """Validate task dependencies and return any issues."""
        issues = []
        
        # Check for circular dependencies
        visited = set()
        path = []
        
        def check_circular(current_id: str) -> bool:
            if current_id in path:
                cycle = path[path.index(current_id):] + [current_id]
                issues.append(f"Circular dependency detected: {' -> '.join(cycle)}")
                return True
            
            if current_id in visited:
                return False
            
            visited.add(current_id)
            path.append(current_id)
            
            current_task = self.get_task(current_id)
            if current_task:
                for dep_id in current_task.depends_on:
                    if check_circular(dep_id):
                        return True
            
            path.pop()
            return False
        
        check_circular(task.id)
        
        # Check for missing dependencies
        for dep_id in task.depends_on:
            if dep_id not in self.tasks:
                issues.append(f"Dependency task not found: {dep_id}")
        
        # Check for self-dependency
        if task.id in task.depends_on:
            issues.append("Task cannot depend on itself")
        
        return issues
    
    def get_execution_order(self) -> List[List[str]]:
        """
        Get tasks in execution order, grouped by dependency level.
        Returns list of lists, where each inner list contains tasks that can run in parallel.
        """
        # Build dependency graph
        in_degree = {task_id: 0 for task_id in self.tasks}
        graph = {task_id: [] for task_id in self.tasks}
        
        for task_id, task in self.tasks.items():
            for dep_id in task.depends_on:
                if dep_id in self.tasks:
                    graph[dep_id].append(task_id)
                    in_degree[task_id] += 1
        
        # Topological sort with levels
        levels = []
        remaining = set(self.tasks.keys())
        
        while remaining:
            # Find tasks with no dependencies
            current_level = [
                task_id for task_id in remaining
                if in_degree[task_id] == 0
            ]
            
            if not current_level:
                # Circular dependency - add remaining tasks to final level
                levels.append(list(remaining))
                break
            
            levels.append(current_level)
            
            # Remove current level tasks and update in-degrees
            for task_id in current_level:
                remaining.remove(task_id)
                for dependent in graph[task_id]:
                    in_degree[dependent] -= 1
        
        return levels
    
    def create_default_tasks(self) -> None:
        """Create a set of default maintenance tasks."""
        default_tasks = [
            MaintenanceTask(
                name="Code Quality Check",
                description="Run comprehensive code quality analysis and fixes",
                category=TaskCategory.CODE_QUALITY,
                priority=TaskPriority.HIGH,
                schedule_cron="0 2 * * *",  # Daily at 2 AM
                timeout_minutes=45,
                tags=["quality", "automated", "daily"],
                config={
                    "fix_formatting": True,
                    "check_type_hints": True,
                    "analyze_complexity": True
                }
            ),
            MaintenanceTask(
                name="Test Suite Maintenance",
                description="Analyze and fix test suite issues",
                category=TaskCategory.TEST_MAINTENANCE,
                priority=TaskPriority.HIGH,
                schedule_cron="0 3 * * 0",  # Weekly on Sunday at 3 AM
                timeout_minutes=60,
                tags=["testing", "weekly", "automated"],
                config={
                    "fix_broken_tests": True,
                    "update_fixtures": True,
                    "analyze_coverage": True
                }
            ),
            MaintenanceTask(
                name="Configuration Cleanup",
                description="Clean up and validate configuration files",
                category=TaskCategory.CONFIG_CLEANUP,
                priority=TaskPriority.MEDIUM,
                schedule_cron="0 4 * * 1",  # Weekly on Monday at 4 AM
                timeout_minutes=30,
                tags=["config", "cleanup", "weekly"],
                config={
                    "validate_configs": True,
                    "remove_unused": True,
                    "standardize_format": True
                }
            ),
            MaintenanceTask(
                name="Documentation Update",
                description="Update and validate project documentation",
                category=TaskCategory.DOCUMENTATION,
                priority=TaskPriority.MEDIUM,
                schedule_cron="0 5 * * 2",  # Weekly on Tuesday at 5 AM
                timeout_minutes=30,
                tags=["docs", "weekly", "automated"],
                config={
                    "check_links": True,
                    "update_structure": True,
                    "validate_examples": True
                }
            ),
            MaintenanceTask(
                name="Dependency Security Scan",
                description="Scan dependencies for security vulnerabilities",
                category=TaskCategory.SECURITY_SCAN,
                priority=TaskPriority.CRITICAL,
                schedule_cron="0 1 * * *",  # Daily at 1 AM
                timeout_minutes=20,
                tags=["security", "dependencies", "daily"],
                config={
                    "scan_vulnerabilities": True,
                    "check_licenses": True,
                    "update_advisories": True
                }
            ),
            MaintenanceTask(
                name="Performance Optimization",
                description="Analyze and optimize system performance",
                category=TaskCategory.PERFORMANCE_OPTIMIZATION,
                priority=TaskPriority.LOW,
                schedule_cron="0 6 * * 3",  # Weekly on Wednesday at 6 AM
                timeout_minutes=90,
                tags=["performance", "optimization", "weekly"],
                config={
                    "profile_code": True,
                    "optimize_imports": True,
                    "analyze_bottlenecks": True
                }
            )
        ]
        
        for task in default_tasks:
            if task.id not in self.tasks:
                self.add_task(task)
        
        self.logger.info(f"Created {len(default_tasks)} default maintenance tasks")
    
    def _load_tasks(self) -> None:
        """Load tasks from storage."""
        if not self.storage_path.exists():
            self.logger.info("No existing tasks file found")
            return
        
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            
            for task_data in data.get('tasks', []):
                task = self._deserialize_task(task_data)
                if task:
                    self.tasks[task.id] = task
            
            self.logger.info(f"Loaded {len(self.tasks)} tasks from storage")
            
        except Exception as e:
            self.logger.error(f"Error loading tasks: {e}", exc_info=True)
    
    def _save_tasks(self) -> None:
        """Save tasks to storage."""
        try:
            data = {
                'version': '1.0',
                'saved_at': datetime.now().isoformat(),
                'tasks': [self._serialize_task(task) for task in self.tasks.values()]
            }
            
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
        except Exception as e:
            self.logger.error(f"Error saving tasks: {e}", exc_info=True)
    
    def _serialize_task(self, task: MaintenanceTask) -> Dict:
        """Serialize a task to dictionary."""
        return {
            'id': task.id,
            'name': task.name,
            'description': task.description,
            'category': task.category.value,
            'priority': task.priority.value,
            'schedule_cron': task.schedule_cron,
            'next_run': task.next_run.isoformat() if task.next_run else None,
            'last_run': task.last_run.isoformat() if task.last_run else None,
            'timeout_minutes': task.timeout_minutes,
            'retry_count': task.retry_count,
            'retry_delay_minutes': task.retry_delay_minutes,
            'depends_on': task.depends_on,
            'blocks': task.blocks,
            'config': task.config,
            'environment_requirements': task.environment_requirements,
            'created_at': task.created_at.isoformat(),
            'updated_at': task.updated_at.isoformat(),
            'created_by': task.created_by,
            'tags': task.tags,
            'rollback_enabled': task.rollback_enabled
        }
    
    def _deserialize_task(self, data: Dict) -> Optional[MaintenanceTask]:
        """Deserialize a task from dictionary."""
        try:
            return MaintenanceTask(
                id=data['id'],
                name=data['name'],
                description=data['description'],
                category=TaskCategory(data['category']),
                priority=TaskPriority(data['priority']),
                schedule_cron=data.get('schedule_cron'),
                next_run=datetime.fromisoformat(data['next_run']) if data.get('next_run') else None,
                last_run=datetime.fromisoformat(data['last_run']) if data.get('last_run') else None,
                timeout_minutes=data.get('timeout_minutes', 30),
                retry_count=data.get('retry_count', 3),
                retry_delay_minutes=data.get('retry_delay_minutes', 5),
                depends_on=data.get('depends_on', []),
                blocks=data.get('blocks', []),
                config=data.get('config', {}),
                environment_requirements=data.get('environment_requirements', []),
                created_at=datetime.fromisoformat(data['created_at']),
                updated_at=datetime.fromisoformat(data['updated_at']),
                created_by=data.get('created_by', 'system'),
                tags=data.get('tags', []),
                rollback_enabled=data.get('rollback_enabled', True)
            )
        except Exception as e:
            self.logger.error(f"Error deserializing task: {e}", exc_info=True)
            return None