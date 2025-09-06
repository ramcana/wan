#!/usr/bin/env python3
"""
Command-line interface for the automated maintenance scheduling system.
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from scheduler import MaintenanceScheduler
from task_manager import MaintenanceTaskManager
from priority_engine import TaskPriorityEngine
from history_tracker import MaintenanceHistoryTracker
from rollback_manager import RollbackManager
from models import MaintenanceTask, TaskCategory, TaskPriority, TaskSchedule


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('data/maintenance/scheduler.log')
        ]
    )


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Automated Maintenance Scheduling System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s start                          # Start the scheduler daemon
  %(prog)s status                         # Show scheduler status
  %(prog)s list-tasks                     # List all maintenance tasks
  %(prog)s run-task TASK_ID               # Run a specific task immediately
  %(prog)s create-task --name "Code Quality" --category code_quality
  %(prog)s schedule-task TASK_ID --cron "0 2 * * *"
  %(prog)s history TASK_ID                # Show task execution history
  %(prog)s rollback ROLLBACK_ID           # Execute a rollback
  %(prog)s cleanup                        # Clean up old data
        """
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Start command
    start_parser = subparsers.add_parser('start', help='Start the maintenance scheduler')
    start_parser.add_argument(
        '--daemon',
        action='store_true',
        help='Run as daemon process'
    )
    
    # Status command
    subparsers.add_parser('status', help='Show scheduler status')
    
    # List tasks command
    list_parser = subparsers.add_parser('list-tasks', help='List maintenance tasks')
    list_parser.add_argument(
        '--category',
        choices=[cat.value for cat in TaskCategory],
        help='Filter by task category'
    )
    list_parser.add_argument(
        '--priority',
        choices=[pri.value for pri in TaskPriority],
        help='Filter by task priority'
    )
    
    # Run task command
    run_parser = subparsers.add_parser('run-task', help='Run a specific task immediately')
    run_parser.add_argument('task_id', help='Task ID to run')
    run_parser.add_argument(
        '--wait',
        action='store_true',
        help='Wait for task completion'
    )
    
    # Create task command
    create_parser = subparsers.add_parser('create-task', help='Create a new maintenance task')
    create_parser.add_argument('--name', required=True, help='Task name')
    create_parser.add_argument('--description', help='Task description')
    create_parser.add_argument(
        '--category',
        choices=[cat.value for cat in TaskCategory],
        default='code_quality',
        help='Task category'
    )
    create_parser.add_argument(
        '--priority',
        choices=[pri.value for pri in TaskPriority],
        default='medium',
        help='Task priority'
    )
    create_parser.add_argument('--timeout', type=int, default=30, help='Timeout in minutes')
    create_parser.add_argument('--tags', nargs='*', help='Task tags')
    create_parser.add_argument('--config-file', help='Path to task configuration file')
    
    # Schedule task command
    schedule_parser = subparsers.add_parser('schedule-task', help='Schedule a task')
    schedule_parser.add_argument('task_id', help='Task ID to schedule')
    schedule_parser.add_argument('--cron', required=True, help='Cron expression')
    schedule_parser.add_argument('--enabled', action='store_true', default=True, help='Enable schedule')
    
    # History command
    history_parser = subparsers.add_parser('history', help='Show task execution history')
    history_parser.add_argument('task_id', nargs='?', help='Task ID (optional)')
    history_parser.add_argument('--limit', type=int, default=20, help='Number of records to show')
    history_parser.add_argument('--days', type=int, default=30, help='Days of history to show')
    
    # Metrics command
    metrics_parser = subparsers.add_parser('metrics', help='Show maintenance metrics')
    metrics_parser.add_argument('--days', type=int, default=30, help='Days to analyze')
    
    # Rollback commands
    rollback_parser = subparsers.add_parser('rollback', help='Execute a rollback')
    rollback_parser.add_argument('rollback_id', help='Rollback point ID')
    rollback_parser.add_argument('--reason', help='Reason for rollback')
    
    list_rollbacks_parser = subparsers.add_parser('list-rollbacks', help='List available rollback points')
    list_rollbacks_parser.add_argument('--task-id', help='Filter by task ID')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up old data')
    cleanup_parser.add_argument('--history-days', type=int, default=365, help='Days of history to keep')
    cleanup_parser.add_argument('--rollback-days', type=int, default=30, help='Days of rollback points to keep')
    
    # Priority analysis command
    priority_parser = subparsers.add_parser('analyze-priority', help='Analyze task priorities')
    priority_parser.add_argument('task_id', nargs='?', help='Specific task ID to analyze')
    
    return parser


async def cmd_start(args, config: Dict) -> int:
    """Start the maintenance scheduler."""
    scheduler = MaintenanceScheduler(config)
    
    # Create default tasks if none exist
    task_manager = MaintenanceTaskManager()
    if not task_manager.get_all_tasks():
        print("Creating default maintenance tasks...")
        task_manager.create_default_tasks()
    
    print("Starting maintenance scheduler...")
    scheduler.start()
    
    if args.daemon:
        print("Running as daemon. Press Ctrl+C to stop.")
        try:
            while True:
                await asyncio.sleep(60)
                # Print status periodically
                running_tasks = scheduler.get_running_tasks()
                if running_tasks:
                    print(f"Currently running {len(running_tasks)} tasks: {', '.join(running_tasks)}")
        except KeyboardInterrupt:
            print("\nShutting down scheduler...")
    else:
        print("Scheduler started. Use 'status' command to check progress.")
    
    scheduler.stop()
    return 0


async def cmd_status(args, config: Dict) -> int:
    """Show scheduler status."""
    scheduler = MaintenanceScheduler(config)
    task_manager = MaintenanceTaskManager()
    history_tracker = MaintenanceHistoryTracker()
    
    # Get basic stats
    all_tasks = task_manager.get_all_tasks()
    running_tasks = scheduler.get_running_tasks()
    next_tasks = scheduler.get_next_scheduled_tasks(5)
    
    # Get recent metrics
    metrics = history_tracker.get_maintenance_metrics(7)  # Last 7 days
    
    print("=== Maintenance Scheduler Status ===")
    print(f"Total tasks: {len(all_tasks)}")
    print(f"Running tasks: {len(running_tasks)}")
    print(f"Next scheduled tasks: {len(next_tasks)}")
    print()
    
    if running_tasks:
        print("Currently running:")
        for task_id in running_tasks:
            task = task_manager.get_task(task_id)
            if task:
                print(f"  - {task.name} ({task_id})")
        print()
    
    if next_tasks:
        print("Next scheduled tasks:")
        for task in next_tasks:
            next_run = task.next_run.strftime("%Y-%m-%d %H:%M") if task.next_run else "Not scheduled"
            print(f"  - {task.name}: {next_run}")
        print()
    
    print("=== Recent Metrics (7 days) ===")
    print(f"Tasks run: {metrics.total_tasks_run}")
    print(f"Success rate: {(metrics.successful_tasks / max(1, metrics.total_tasks_run)) * 100:.1f}%")
    print(f"Average duration: {metrics.average_duration_seconds:.1f}s")
    print(f"Issues fixed: {metrics.total_issues_fixed}")
    print(f"Files modified: {metrics.total_files_modified}")
    
    if metrics.consecutive_failures > 0:
        print(f"⚠️  Consecutive failures: {metrics.consecutive_failures}")
    
    return 0


async def cmd_list_tasks(args, config: Dict) -> int:
    """List maintenance tasks."""
    task_manager = MaintenanceTaskManager()
    priority_engine = TaskPriorityEngine()
    
    tasks = task_manager.get_all_tasks()
    
    # Apply filters
    if args.category:
        tasks = [t for t in tasks if t.category.value == args.category]
    
    if args.priority:
        tasks = [t for t in tasks if t.priority.value == args.priority]
    
    if not tasks:
        print("No tasks found matching criteria.")
        return 0
    
    print(f"=== Maintenance Tasks ({len(tasks)}) ===")
    print(f"{'Name':<30} {'Category':<20} {'Priority':<10} {'Score':<8} {'Last Run':<20}")
    print("-" * 90)
    
    for task in tasks:
        priority_score = priority_engine.get_priority_score(task)
        last_run = task.last_run.strftime("%Y-%m-%d %H:%M") if task.last_run else "Never"
        
        print(f"{task.name[:29]:<30} {task.category.value:<20} {task.priority.value:<10} "
              f"{priority_score:>7.1f} {last_run:<20}")
    
    return 0


async def cmd_run_task(args, config: Dict) -> int:
    """Run a specific task immediately."""
    scheduler = MaintenanceScheduler(config)
    task_manager = MaintenanceTaskManager()
    
    task = task_manager.get_task(args.task_id)
    if not task:
        print(f"Task not found: {args.task_id}")
        return 1
    
    print(f"Running task: {task.name}")
    
    if args.wait:
        result = scheduler.run_task_now(args.task_id)
        if result:
            print(f"Task completed: {result.status.value}")
            if result.success:
                print(f"Duration: {result.duration_seconds:.2f}s")
                print(f"Files modified: {result.files_modified}")
                print(f"Issues fixed: {result.issues_fixed}")
                if result.output:
                    print(f"Output: {result.output}")
            else:
                print(f"Error: {result.error_message}")
        else:
            print("Failed to run task")
            return 1
    else:
        # Start task asynchronously
        scheduler.start()
        scheduler.run_task_now(args.task_id)
        print("Task started. Use 'status' command to check progress.")
    
    return 0


async def cmd_create_task(args, config: Dict) -> int:
    """Create a new maintenance task."""
    task_manager = MaintenanceTaskManager()
    
    # Load config from file if provided
    task_config = {}
    if args.config_file:
        config_path = Path(args.config_file)
        if config_path.exists():
            with open(config_path, 'r') as f:
                task_config = json.load(f)
        else:
            print(f"Config file not found: {args.config_file}")
            return 1
    
    task = MaintenanceTask(
        name=args.name,
        description=args.description or "",
        category=TaskCategory(args.category),
        priority=TaskPriority(args.priority),
        timeout_minutes=args.timeout,
        tags=args.tags or [],
        config=task_config
    )
    
    task_manager.add_task(task)
    print(f"Created task: {task.name} ({task.id})")
    
    return 0


async def cmd_schedule_task(args, config: Dict) -> int:
    """Schedule a task."""
    scheduler = MaintenanceScheduler(config)
    task_manager = MaintenanceTaskManager()
    
    task = task_manager.get_task(args.task_id)
    if not task:
        print(f"Task not found: {args.task_id}")
        return 1
    
    schedule = TaskSchedule(
        task_id=args.task_id,
        cron_expression=args.cron,
        enabled=args.enabled
    )
    
    scheduler.add_task(task, schedule)
    print(f"Scheduled task: {task.name}")
    print(f"Cron expression: {args.cron}")
    
    return 0


async def cmd_history(args, config: Dict) -> int:
    """Show task execution history."""
    history_tracker = MaintenanceHistoryTracker()
    task_manager = MaintenanceTaskManager()
    
    if args.task_id:
        # Show history for specific task
        history = history_tracker.get_task_history(args.task_id, args.limit)
        task = task_manager.get_task(args.task_id)
        task_name = task.name if task else args.task_id
        
        print(f"=== Execution History: {task_name} ===")
    else:
        # Show recent history for all tasks
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.days)
        history = history_tracker.get_history_by_date_range(start_date, end_date)
        history = history[:args.limit]
        
        print(f"=== Recent Execution History ({args.days} days) ===")
    
    if not history:
        print("No execution history found.")
        return 0
    
    print(f"{'Task':<30} {'Status':<12} {'Duration':<10} {'Started':<20} {'Issues Fixed':<12}")
    print("-" * 90)
    
    for record in history:
        task = task_manager.get_task(record.task_id)
        task_name = task.name[:29] if task else record.task_id[:29]
        
        status = "✓ Success" if record.result.success else "✗ Failed"
        duration = f"{record.result.duration_seconds:.1f}s"
        started = record.result.started_at.strftime("%Y-%m-%d %H:%M")
        issues_fixed = str(record.result.issues_fixed)
        
        print(f"{task_name:<30} {status:<12} {duration:<10} {started:<20} {issues_fixed:<12}")
    
    return 0


async def cmd_metrics(args, config: Dict) -> int:
    """Show maintenance metrics."""
    history_tracker = MaintenanceHistoryTracker()
    rollback_manager = RollbackManager()
    
    metrics = history_tracker.get_maintenance_metrics(args.days)
    rollback_stats = rollback_manager.get_rollback_statistics()
    
    print(f"=== Maintenance Metrics ({args.days} days) ===")
    print(f"Period: {metrics.period_start.strftime('%Y-%m-%d')} to {metrics.period_end.strftime('%Y-%m-%d')}")
    print()
    
    print("Task Execution:")
    print(f"  Total tasks run: {metrics.total_tasks_run}")
    print(f"  Successful: {metrics.successful_tasks}")
    print(f"  Failed: {metrics.failed_tasks}")
    print(f"  Success rate: {(metrics.successful_tasks / max(1, metrics.total_tasks_run)) * 100:.1f}%")
    print(f"  Average duration: {metrics.average_duration_seconds:.1f}s")
    print()
    
    print("Quality Improvements:")
    print(f"  Issues fixed: {metrics.total_issues_fixed}")
    print(f"  Files modified: {metrics.total_files_modified}")
    print(f"  Lines changed: {metrics.total_lines_changed}")
    print(f"  Average quality improvement: {metrics.average_quality_improvement:.1f}%")
    print()
    
    print("System Health:")
    if metrics.last_successful_run:
        print(f"  Last successful run: {metrics.last_successful_run.strftime('%Y-%m-%d %H:%M')}")
    else:
        print("  Last successful run: Never")
    
    if metrics.consecutive_failures > 0:
        print(f"  ⚠️  Consecutive failures: {metrics.consecutive_failures}")
    else:
        print("  ✓ No consecutive failures")
    print()
    
    print("Rollback Points:")
    print(f"  Total rollback points: {rollback_stats['total_rollback_points']}")
    print(f"  Valid rollback points: {rollback_stats['valid_rollback_points']}")
    print(f"  Used rollback points: {rollback_stats['used_rollback_points']}")
    print(f"  Total storage: {rollback_stats['total_size_mb']:.1f} MB")
    
    return 0


async def cmd_rollback(args, config: Dict) -> int:
    """Execute a rollback."""
    rollback_manager = RollbackManager()
    
    success = await rollback_manager.execute_rollback(args.rollback_id, args.reason or "Manual rollback")
    
    if success:
        print(f"✓ Rollback {args.rollback_id} executed successfully")
        return 0
    else:
        print(f"✗ Failed to execute rollback {args.rollback_id}")
        return 1


async def cmd_list_rollbacks(args, config: Dict) -> int:
    """List available rollback points."""
    rollback_manager = RollbackManager()
    
    rollback_points = rollback_manager.get_rollback_points(args.task_id)
    
    if not rollback_points:
        print("No rollback points found.")
        return 0
    
    print(f"=== Rollback Points ({len(rollback_points)}) ===")
    print(f"{'ID':<36} {'Task':<30} {'Created':<20} {'Size':<10} {'Status':<10}")
    print("-" * 110)
    
    for point in rollback_points:
        size_mb = point.size_bytes / 1024 / 1024
        status = "Used" if point.used else ("Invalid" if not point.valid else "Available")
        
        print(f"{point.id:<36} {point.task_id[:29]:<30} "
              f"{point.created_at.strftime('%Y-%m-%d %H:%M'):<20} "
              f"{size_mb:>9.1f}M {status:<10}")
    
    return 0


async def cmd_cleanup(args, config: Dict) -> int:
    """Clean up old data."""
    history_tracker = MaintenanceHistoryTracker()
    rollback_manager = RollbackManager()
    
    print("Cleaning up old data...")
    
    # Clean up history
    history_cleaned = history_tracker.cleanup_old_history(args.history_days)
    print(f"Cleaned up {history_cleaned} old history records")
    
    # Clean up rollback points
    rollback_cleaned = rollback_manager.cleanup_old_rollback_points()
    print(f"Cleaned up {rollback_cleaned} old rollback points")
    
    print("Cleanup completed.")
    return 0


async def cmd_analyze_priority(args, config: Dict) -> int:
    """Analyze task priorities."""
    task_manager = MaintenanceTaskManager()
    priority_engine = TaskPriorityEngine()
    history_tracker = MaintenanceHistoryTracker()
    
    if args.task_id:
        # Analyze specific task
        task = task_manager.get_task(args.task_id)
        if not task:
            print(f"Task not found: {args.task_id}")
            return 1
        
        history = history_tracker.get_task_history(args.task_id, 10)
        
        impact_analysis = priority_engine.analyze_impact(task)
        effort_analysis = priority_engine.analyze_effort(task, history)
        urgency_score = priority_engine.calculate_urgency(task)
        priority_score = priority_engine.get_priority_score(task, history)
        
        print(f"=== Priority Analysis: {task.name} ===")
        print(f"Overall Priority Score: {priority_score:.2f}/100")
        print()
        
        print("Impact Analysis:")
        print(f"  Code Quality Impact: {impact_analysis.code_quality_impact:.1f}/100")
        print(f"  Security Impact: {impact_analysis.security_impact:.1f}/100")
        print(f"  Performance Impact: {impact_analysis.performance_impact:.1f}/100")
        print(f"  Maintainability Impact: {impact_analysis.maintainability_impact:.1f}/100")
        print(f"  User Experience Impact: {impact_analysis.user_experience_impact:.1f}/100")
        print(f"  Total Impact: {impact_analysis.total_impact:.1f}/100 ({impact_analysis.impact_category})")
        print()
        
        print("Effort Analysis:")
        print(f"  Estimated Duration: {effort_analysis.estimated_duration_minutes} minutes")
        print(f"  Complexity Score: {effort_analysis.complexity_score:.1f}/10")
        print(f"  Risk Score: {effort_analysis.risk_score:.1f}/10")
        print(f"  Dependencies: {effort_analysis.dependency_count}")
        print(f"  Success Rate: {effort_analysis.success_rate * 100:.1f}%")
        print()
        
        print(f"Urgency Score: {urgency_score:.1f}/100")
        
    else:
        # Analyze all tasks
        tasks = task_manager.get_all_tasks()
        recommended_order = priority_engine.get_recommended_execution_order(tasks)
        
        print(f"=== Priority Analysis: All Tasks ({len(tasks)}) ===")
        print(f"{'Rank':<5} {'Task':<30} {'Priority':<10} {'Score':<8} {'Category':<20}")
        print("-" * 75)
        
        for i, task in enumerate(recommended_order, 1):
            score = priority_engine.get_priority_score(task)
            print(f"{i:<5} {task.name[:29]:<30} {task.priority.value:<10} "
                  f"{score:>7.1f} {task.category.value:<20}")
    
    return 0


def load_config(config_path: Optional[str]) -> Dict:
    """Load configuration from file."""
    config = {
        'max_concurrent_tasks': 3,
        'check_interval_seconds': 60,
        'task_timeout_minutes': 30,
        'backup_root': 'data/maintenance/rollbacks',
        'max_rollback_points': 50,
        'cleanup_after_days': 30,
        'max_backup_size_mb': 1000
    }
    
    if config_path:
        config_file = Path(config_path)
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    file_config = json.load(f)
                config.update(file_config)
            except Exception as e:
                print(f"Warning: Failed to load config file {config_path}: {e}")
    
    return config


async def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Load configuration
    config = load_config(args.config)
    
    # Ensure data directory exists
    Path('data/maintenance').mkdir(parents=True, exist_ok=True)
    
    # Execute command
    command_map = {
        'start': cmd_start,
        'status': cmd_status,
        'list-tasks': cmd_list_tasks,
        'run-task': cmd_run_task,
        'create-task': cmd_create_task,
        'schedule-task': cmd_schedule_task,
        'history': cmd_history,
        'metrics': cmd_metrics,
        'rollback': cmd_rollback,
        'list-rollbacks': cmd_list_rollbacks,
        'cleanup': cmd_cleanup,
        'analyze-priority': cmd_analyze_priority
    }
    
    command_func = command_map.get(args.command)
    if command_func:
        return await command_func(args, config)
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))