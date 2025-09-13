#!/usr/bin/env python3
"""
Example usage of the Automated Maintenance Scheduling System.

This script demonstrates how to:
1. Create and configure maintenance tasks
2. Set up scheduling
3. Execute tasks with rollback capabilities
4. Monitor execution and metrics
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

from scheduler import MaintenanceScheduler
from task_manager import MaintenanceTaskManager
from priority_engine import TaskPriorityEngine
from history_tracker import MaintenanceHistoryTracker
from rollback_manager import RollbackManager
from models import MaintenanceTask, TaskCategory, TaskPriority, TaskSchedule


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def example_code_quality_task(config: dict) -> dict:
    """Example code quality maintenance task."""
    logger.info("Running code quality maintenance task...")
    
    # Simulate code quality operations
    await asyncio.sleep(2)  # Simulate processing time
    
    # Simulate results
    files_processed = 25
    issues_found = 12
    issues_fixed = 10
    
    logger.info(f"Processed {files_processed} files, fixed {issues_fixed}/{issues_found} issues")
    
    return {
        'success': True,
        'output': f'Code quality check completed. Fixed {issues_fixed} issues in {files_processed} files.',
        'files_modified': files_processed,
        'issues_fixed': issues_fixed,
        'quality_improvement': 15.5  # Percentage improvement
    }


async def example_test_maintenance_task(config: dict) -> dict:
    """Example test maintenance task."""
    logger.info("Running test maintenance task...")
    
    # Simulate test maintenance operations
    await asyncio.sleep(3)  # Simulate processing time
    
    # Simulate results
    tests_analyzed = 150
    broken_tests_fixed = 5
    coverage_improved = 2.3
    
    logger.info(f"Analyzed {tests_analyzed} tests, fixed {broken_tests_fixed} broken tests")
    
    return {
        'success': True,
        'output': f'Test maintenance completed. Fixed {broken_tests_fixed} tests, improved coverage by {coverage_improved}%.',
        'files_modified': 8,
        'issues_fixed': broken_tests_fixed,
        'quality_improvement': coverage_improved
    }


async def example_config_cleanup_task(config: dict) -> dict:
    """Example configuration cleanup task."""
    logger.info("Running configuration cleanup task...")
    
    # Simulate config cleanup operations
    await asyncio.sleep(1.5)  # Simulate processing time
    
    # Simulate results
    configs_processed = 12
    duplicates_removed = 3
    inconsistencies_fixed = 2
    
    logger.info(f"Processed {configs_processed} config files, removed {duplicates_removed} duplicates")
    
    return {
        'success': True,
        'output': f'Config cleanup completed. Processed {configs_processed} files, removed {duplicates_removed} duplicates.',
        'files_modified': configs_processed,
        'issues_fixed': duplicates_removed + inconsistencies_fixed,
        'quality_improvement': 8.2
    }


def create_example_tasks() -> list:
    """Create example maintenance tasks."""
    tasks = []
    
    # Code Quality Task
    code_quality_task = MaintenanceTask(
        name="Daily Code Quality Check",
        description="Comprehensive code quality analysis and automated fixes",
        category=TaskCategory.CODE_QUALITY,
        priority=TaskPriority.HIGH,
        timeout_minutes=30,
        tags=["quality", "automated", "daily"],
        config={
            'fix_formatting': True,
            'check_type_hints': True,
            'analyze_complexity': True,
            'max_complexity': 10
        }
    )
    code_quality_task.executor = example_code_quality_task
    tasks.append(code_quality_task)
    
    # Test Maintenance Task
    test_maintenance_task = MaintenanceTask(
        name="Weekly Test Suite Maintenance",
        description="Analyze and fix test suite issues, update fixtures",
        category=TaskCategory.TEST_MAINTENANCE,
        priority=TaskPriority.HIGH,
        timeout_minutes=45,
        tags=["testing", "weekly", "maintenance"],
        config={
            'fix_broken_tests': True,
            'update_fixtures': True,
            'analyze_coverage': True,
            'min_coverage_threshold': 80
        }
    )
    test_maintenance_task.executor = example_test_maintenance_task
    tasks.append(test_maintenance_task)
    
    # Configuration Cleanup Task
    config_cleanup_task = MaintenanceTask(
        name="Configuration Cleanup",
        description="Clean up and validate configuration files",
        category=TaskCategory.CONFIG_CLEANUP,
        priority=TaskPriority.MEDIUM,
        timeout_minutes=20,
        tags=["config", "cleanup", "validation"],
        config={
            'validate_configs': True,
            'remove_duplicates': True,
            'standardize_format': True
        }
    )
    config_cleanup_task.executor = example_config_cleanup_task
    tasks.append(config_cleanup_task)
    
    return tasks


async def demonstrate_basic_usage():
    """Demonstrate basic maintenance scheduler usage."""
    print("=== Basic Maintenance Scheduler Usage ===")
    
    # Create scheduler with custom config
    config = {
        'max_concurrent_tasks': 2,
        'check_interval_seconds': 30,
        'task_timeout_minutes': 60
    }
    scheduler = MaintenanceScheduler(config)
    
    # Create example tasks
    tasks = create_example_tasks()
    
    # Add tasks to scheduler
    for task in tasks:
        scheduler.add_task(task)
        print(f"Added task: {task.name}")
    
    # Run tasks immediately for demonstration
    print("\nRunning tasks immediately...")
    for task in tasks:
        print(f"\nExecuting: {task.name}")
        result = scheduler.run_task_now(task.id)
        
        if result and result.success:
            print(f"✓ Success: {result.output}")
            print(f"  Duration: {result.duration_seconds:.2f}s")
            print(f"  Files modified: {result.files_modified}")
            print(f"  Issues fixed: {result.issues_fixed}")
            print(f"  Quality improvement: {result.quality_improvement:.1f}%")
        else:
            print(f"✗ Failed: {result.error_message if result else 'Unknown error'}")


async def demonstrate_scheduling():
    """Demonstrate task scheduling functionality."""
    print("\n=== Task Scheduling Demonstration ===")
    
    scheduler = MaintenanceScheduler()
    tasks = create_example_tasks()
    
    # Create schedules for tasks
    schedules = [
        TaskSchedule(
            task_id=tasks[0].id,
            cron_expression="0 2 * * *",  # Daily at 2 AM
            enabled=True,
            allowed_hours=[1, 2, 3, 4, 5],  # Only during low-usage hours
            cooldown_minutes=60
        ),
        TaskSchedule(
            task_id=tasks[1].id,
            cron_expression="0 3 * * 0",  # Weekly on Sunday at 3 AM
            enabled=True,
            allowed_hours=[1, 2, 3, 4, 5],
            cooldown_minutes=120
        ),
        TaskSchedule(
            task_id=tasks[2].id,
            cron_expression="0 4 * * 1",  # Weekly on Monday at 4 AM
            enabled=True,
            allowed_hours=[1, 2, 3, 4, 5],
            cooldown_minutes=60
        )
    ]
    
    # Add scheduled tasks
    for task, schedule in zip(tasks, schedules):
        scheduler.add_task(task, schedule)
        print(f"Scheduled: {task.name} - {schedule.cron_expression}")
    
    # Show next scheduled tasks
    next_tasks = scheduler.get_next_scheduled_tasks(5)
    print(f"\nNext {len(next_tasks)} scheduled tasks:")
    for task in next_tasks:
        next_run = task.next_run.strftime("%Y-%m-%d %H:%M") if task.next_run else "Not scheduled"
        print(f"  - {task.name}: {next_run}")


async def demonstrate_priority_analysis():
    """Demonstrate priority analysis functionality."""
    print("\n=== Priority Analysis Demonstration ===")
    
    priority_engine = TaskPriorityEngine()
    tasks = create_example_tasks()
    
    print("Task Priority Analysis:")
    print(f"{'Task':<35} {'Priority':<10} {'Score':<8} {'Impact':<15}")
    print("-" * 70)
    
    for task in tasks:
        priority_score = priority_engine.get_priority_score(task)
        impact_analysis = priority_engine.analyze_impact(task)
        
        print(f"{task.name[:34]:<35} {task.priority.value:<10} "
              f"{priority_score:>7.1f} {impact_analysis.impact_category:<15}")
    
    # Get recommended execution order
    recommended_order = priority_engine.get_recommended_execution_order(tasks)
    
    print(f"\nRecommended Execution Order:")
    for i, task in enumerate(recommended_order, 1):
        score = priority_engine.get_priority_score(task)
        print(f"  {i}. {task.name} (Score: {score:.1f})")


async def demonstrate_rollback_system():
    """Demonstrate rollback system functionality."""
    print("\n=== Rollback System Demonstration ===")
    
    # Create rollback manager
    rollback_manager = RollbackManager({
        'backup_root': 'data/maintenance/demo_rollbacks',
        'max_rollback_points': 10
    })
    
    # Create a task for rollback demonstration
    task = MaintenanceTask(
        name="Rollback Demo Task",
        description="Task to demonstrate rollback functionality",
        category=TaskCategory.CODE_QUALITY,
        priority=TaskPriority.MEDIUM,
        rollback_enabled=True
    )
    
    # Create rollback point
    print("Creating rollback point...")
    rollback_data = await rollback_manager.create_rollback_point(task)
    
    if rollback_data:
        rollback_id = rollback_data['rollback_id']
        print(f"✓ Rollback point created: {rollback_id}")
        print(f"  Size: {rollback_data['size_bytes'] / 1024:.1f} KB")
        print(f"  Files backed up: {rollback_data['file_count']}")
        
        # List rollback points
        rollback_points = rollback_manager.get_rollback_points()
        print(f"\nAvailable rollback points: {len(rollback_points)}")
        
        for point in rollback_points[:3]:  # Show first 3
            size_mb = point.size_bytes / 1024 / 1024
            print(f"  - {point.id[:8]}... ({size_mb:.1f} MB) - {point.description}")
        
        # Get rollback statistics
        stats = rollback_manager.get_rollback_statistics()
        print(f"\nRollback Statistics:")
        print(f"  Total rollback points: {stats['total_rollback_points']}")
        print(f"  Total storage: {stats['total_size_mb']:.1f} MB")
        print(f"  Valid rollback points: {stats['valid_rollback_points']}")
    else:
        print("✗ Failed to create rollback point")


async def demonstrate_history_and_metrics():
    """Demonstrate history tracking and metrics."""
    print("\n=== History and Metrics Demonstration ===")
    
    # Create history tracker
    history_tracker = MaintenanceHistoryTracker('data/maintenance/demo_history.json')
    
    # Simulate some historical data by running tasks
    scheduler = MaintenanceScheduler()
    tasks = create_example_tasks()
    
    print("Executing tasks to generate history...")
    for task in tasks:
        result = scheduler.run_task_now(task.id)
        if result:
            history_tracker.record_execution(task, result)
            print(f"✓ Recorded execution: {task.name}")
    
    # Get metrics
    metrics = history_tracker.get_maintenance_metrics(30)  # Last 30 days
    
    print(f"\nMaintenance Metrics (30 days):")
    print(f"  Total tasks run: {metrics.total_tasks_run}")
    print(f"  Successful tasks: {metrics.successful_tasks}")
    print(f"  Failed tasks: {metrics.failed_tasks}")
    print(f"  Success rate: {(metrics.successful_tasks / max(1, metrics.total_tasks_run)) * 100:.1f}%")
    print(f"  Average duration: {metrics.average_duration_seconds:.1f}s")
    print(f"  Total issues fixed: {metrics.total_issues_fixed}")
    print(f"  Total files modified: {metrics.total_files_modified}")
    print(f"  Average quality improvement: {metrics.average_quality_improvement:.1f}%")
    
    # Show recent history
    print(f"\nRecent Execution History:")
    for task in tasks:
        task_history = history_tracker.get_task_history(task.id, 1)
        if task_history:
            record = task_history[0]
            status = "✓" if record.result.success else "✗"
            print(f"  {status} {task.name}: {record.result.duration_seconds:.1f}s")


async def demonstrate_full_workflow():
    """Demonstrate a complete maintenance workflow."""
    print("\n=== Complete Maintenance Workflow ===")
    
    # Setup
    config = {
        'max_concurrent_tasks': 2,
        'check_interval_seconds': 5,
        'backup_root': 'data/maintenance/demo_rollbacks'
    }
    
    scheduler = MaintenanceScheduler(config)
    task_manager = MaintenanceTaskManager('data/maintenance/demo_tasks.json')
    history_tracker = MaintenanceHistoryTracker('data/maintenance/demo_history.json')
    
    # Create and add tasks
    tasks = create_example_tasks()
    for task in tasks:
        task_manager.add_task(task)
        scheduler.add_task(task)
    
    print(f"Created {len(tasks)} maintenance tasks")
    
    # Start scheduler for a short demonstration
    print("Starting scheduler for 10 seconds...")
    scheduler.start()
    
    # Let it run briefly
    await asyncio.sleep(10)
    
    # Stop scheduler
    scheduler.stop()
    print("Scheduler stopped")
    
    # Show final status
    running_tasks = scheduler.get_running_tasks()
    print(f"Running tasks: {len(running_tasks)}")
    
    # Show metrics
    metrics = history_tracker.get_maintenance_metrics(1)
    if metrics.total_tasks_run > 0:
        print(f"Tasks executed: {metrics.total_tasks_run}")
        print(f"Success rate: {(metrics.successful_tasks / metrics.total_tasks_run) * 100:.1f}%")


async def main():
    """Run all demonstrations."""
    print("Automated Maintenance Scheduling System - Example Usage")
    print("=" * 60)
    
    # Ensure data directory exists
    Path('data/maintenance').mkdir(parents=True, exist_ok=True)
    
    try:
        await demonstrate_basic_usage()
        await demonstrate_scheduling()
        await demonstrate_priority_analysis()
        await demonstrate_rollback_system()
        await demonstrate_history_and_metrics()
        await demonstrate_full_workflow()
        
        print("\n" + "=" * 60)
        print("All demonstrations completed successfully!")
        print("\nTo use the maintenance scheduler in production:")
        print("1. Configure tasks for your specific needs")
        print("2. Set up appropriate cron schedules")
        print("3. Start the scheduler daemon")
        print("4. Monitor execution through the CLI or metrics")
        
    except Exception as e:
        logger.error(f"Error during demonstration: {e}", exc_info=True)
        print(f"\n✗ Error during demonstration: {e}")


if __name__ == '__main__':
    asyncio.run(main())
