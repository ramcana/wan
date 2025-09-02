# Automated Maintenance Scheduling System

A comprehensive system for scheduling and executing automated maintenance tasks with intelligent prioritization, rollback capabilities, and detailed tracking.

## Features

### Core Capabilities

- **Automated Task Scheduling**: Cron-based scheduling with flexible time windows
- **Intelligent Prioritization**: Impact and effort analysis for optimal task ordering
- **Safe Rollback System**: Automatic backup creation and rollback capabilities
- **Comprehensive Tracking**: Detailed execution history and performance metrics
- **Dependency Management**: Task dependencies and execution ordering
- **Concurrent Execution**: Configurable parallel task execution with resource management

### Task Categories

- **Code Quality**: Formatting, linting, type checking, complexity analysis
- **Test Maintenance**: Test fixing, coverage analysis, fixture updates
- **Configuration Cleanup**: Config validation, standardization, unused removal
- **Documentation**: Link checking, structure updates, example validation
- **Security Scanning**: Vulnerability detection, license checking, advisory updates
- **Performance Optimization**: Profiling, bottleneck analysis, import optimization
- **Dependency Updates**: Package updates, compatibility checking

## Architecture

### Core Components

```
MaintenanceScheduler (scheduler.py)
├── TaskManager (task_manager.py)          # Task lifecycle management
├── PriorityEngine (priority_engine.py)    # Impact/effort analysis
├── HistoryTracker (history_tracker.py)    # Execution tracking
└── RollbackManager (rollback_manager.py)  # Safe operation rollbacks
```

### Data Models (models.py)

- `MaintenanceTask`: Task definition and configuration
- `MaintenanceResult`: Execution results and metrics
- `TaskSchedule`: Scheduling configuration
- `MaintenanceHistory`: Historical execution records
- `MaintenanceMetrics`: System performance metrics

## Installation

```bash
# Install dependencies
pip install croniter

# Create data directories
mkdir -p data/maintenance/{tasks,history,rollbacks}

# Initialize with default tasks
python -m tools.maintenance-scheduler.cli create-default-tasks
```

## Usage

### Command Line Interface

```bash
# Start the scheduler daemon
python -m tools.maintenance-scheduler.cli start --daemon

# Check scheduler status
python -m tools.maintenance-scheduler.cli status

# List all maintenance tasks
python -m tools.maintenance-scheduler.cli list-tasks

# Run a specific task immediately
python -m tools.maintenance-scheduler.cli run-task TASK_ID --wait

# Create a new maintenance task
python -m tools.maintenance-scheduler.cli create-task \
    --name "Custom Quality Check" \
    --category code_quality \
    --priority high \
    --timeout 45

# Schedule a task with cron expression
python -m tools.maintenance-scheduler.cli schedule-task TASK_ID \
    --cron "0 2 * * *"  # Daily at 2 AM

# View execution history
python -m tools.maintenance-scheduler.cli history TASK_ID --limit 10

# Show system metrics
python -m tools.maintenance-scheduler.cli metrics --days 30

# List available rollback points
python -m tools.maintenance-scheduler.cli list-rollbacks

# Execute a rollback
python -m tools.maintenance-scheduler.cli rollback ROLLBACK_ID \
    --reason "Reverting failed optimization"

# Analyze task priorities
python -m tools.maintenance-scheduler.cli analyze-priority TASK_ID

# Clean up old data
python -m tools.maintenance-scheduler.cli cleanup \
    --history-days 365 \
    --rollback-days 30
```

### Programmatic Usage

```python
from tools.maintenance_scheduler import MaintenanceScheduler, MaintenanceTask
from tools.maintenance_scheduler.models import TaskCategory, TaskPriority

# Create scheduler
scheduler = MaintenanceScheduler({
    'max_concurrent_tasks': 3,
    'check_interval_seconds': 60
})

# Create a custom task
task = MaintenanceTask(
    name="Custom Code Quality Check",
    description="Run comprehensive code quality analysis",
    category=TaskCategory.CODE_QUALITY,
    priority=TaskPriority.HIGH,
    timeout_minutes=45,
    config={
        'fix_formatting': True,
        'check_type_hints': True,
        'analyze_complexity': True
    }
)

# Add task to scheduler
scheduler.add_task(task)

# Start scheduler
scheduler.start()

# Run task immediately
result = scheduler.run_task_now(task.id)
print(f"Task completed: {result.success}")
```

## Configuration

### Scheduler Configuration

```json
{
  "max_concurrent_tasks": 3,
  "check_interval_seconds": 60,
  "task_timeout_minutes": 30,
  "backup_root": "data/maintenance/rollbacks",
  "max_rollback_points": 50,
  "cleanup_after_days": 30,
  "max_backup_size_mb": 1000,
  "impact_weight": 0.4,
  "urgency_weight": 0.3,
  "effort_weight": 0.2,
  "success_rate_weight": 0.1
}
```

### Task Configuration Examples

#### Code Quality Task

```json
{
  "fix_formatting": true,
  "check_type_hints": true,
  "analyze_complexity": true,
  "max_complexity": 10,
  "exclude_patterns": ["tests/*", "migrations/*"]
}
```

#### Test Maintenance Task

```json
{
  "fix_broken_tests": true,
  "update_fixtures": true,
  "analyze_coverage": true,
  "min_coverage_threshold": 80,
  "timeout_per_test": 30
}
```

#### Security Scan Task

```json
{
  "scan_vulnerabilities": true,
  "check_licenses": true,
  "update_advisories": true,
  "severity_threshold": "medium",
  "exclude_dev_dependencies": false
}
```

## Task Scheduling

### Cron Expressions

```bash
# Every day at 2 AM
"0 2 * * *"

# Every Sunday at 3 AM
"0 3 * * 0"

# Every weekday at 1 AM
"0 1 * * 1-5"

# Every 6 hours
"0 */6 * * *"

# First day of every month at midnight
"0 0 1 * *"
```

### Execution Windows

```python
schedule = TaskSchedule(
    task_id="task_id",
    cron_expression="0 2 * * *",
    allowed_hours=[1, 2, 3, 4, 5],  # Only run between 1-5 AM
    allowed_days=[0, 1, 2, 3, 4],   # Only run on weekdays
    cooldown_minutes=60              # Wait 1 hour between runs
)
```

## Priority System

### Impact Analysis

Tasks are analyzed across multiple dimensions:

- **Code Quality Impact** (0-100): Effect on code maintainability
- **Security Impact** (0-100): Security vulnerability reduction
- **Performance Impact** (0-100): System performance improvement
- **Maintainability Impact** (0-100): Long-term maintenance benefits
- **User Experience Impact** (0-100): End-user experience improvement

### Effort Analysis

- **Estimated Duration**: Time required for completion
- **Complexity Score** (1-10): Technical complexity
- **Risk Score** (1-10): Risk of failure or side effects
- **Success Rate**: Historical success percentage
- **Dependencies**: Number of dependent tasks

### Priority Calculation

```
Priority Score = (
    Impact Score × 0.4 +
    Urgency Score × 0.3 +
    Effort Score × 0.2 +
    Success Rate × 0.1
) × Category Multiplier × Priority Adjustment
```

## Rollback System

### Automatic Backup Creation

Before each task execution, the system automatically creates:

- **File Backups**: Copies of files that might be modified
- **Git State**: Current commit hash and branch information
- **Configuration Snapshots**: Backup of configuration files
- **Integrity Checksums**: Verification of backup completeness

### Rollback Execution

```bash
# List available rollback points
python -m tools.maintenance-scheduler.cli list-rollbacks

# Execute rollback
python -m tools.maintenance-scheduler.cli rollback ROLLBACK_ID \
    --reason "Task caused performance regression"
```

### Rollback Point Management

- Automatic cleanup of old rollback points
- Size-based limits to prevent excessive storage usage
- Integrity verification before rollback execution
- Detailed logging of rollback operations

## Monitoring and Metrics

### System Metrics

- **Task Execution**: Success rates, failure counts, duration trends
- **Quality Improvements**: Issues fixed, files modified, lines changed
- **System Health**: Consecutive failures, last successful run
- **Resource Usage**: Storage usage, execution time distribution

### Performance Tracking

```python
# Get comprehensive metrics
metrics = history_tracker.get_maintenance_metrics(days=30)

print(f"Success rate: {metrics.successful_tasks / metrics.total_tasks_run * 100:.1f}%")
print(f"Average duration: {metrics.average_duration_seconds:.1f}s")
print(f"Issues fixed: {metrics.total_issues_fixed}")
```

### Trend Analysis

```python
# Get performance trends for a task
trends = history_tracker.get_task_performance_trends(task_id, days=90)

# Analyze success rate trends
for date, success_rate in zip(trends['dates'], trends['success_rates']):
    print(f"{date}: {success_rate * 100:.1f}% success rate")
```

## Integration

### CI/CD Integration

```yaml
# GitHub Actions example
- name: Run Maintenance Tasks
  run: |
    python -m tools.maintenance-scheduler.cli run-task code-quality-check --wait
    python -m tools.maintenance-scheduler.cli run-task test-maintenance --wait
```

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: maintenance-check
        name: Run maintenance checks
        entry: python -m tools.maintenance-scheduler.cli run-task pre-commit-check
        language: system
        pass_filenames: false
```

### Custom Task Executors

```python
async def custom_task_executor(config: Dict) -> Dict:
    """Custom task implementation."""
    try:
        # Perform maintenance operations
        files_modified = perform_custom_maintenance(config)

        return {
            'success': True,
            'output': f'Processed {files_modified} files',
            'files_modified': files_modified,
            'issues_fixed': count_issues_fixed(),
            'quality_improvement': calculate_improvement()
        }
    except Exception as e:
        return {
            'success': False,
            'output': str(e)
        }

# Register custom executor
task.executor = custom_task_executor
```

## Troubleshooting

### Common Issues

#### Task Not Running

1. Check if scheduler is running: `python -m tools.maintenance-scheduler.cli status`
2. Verify task schedule: `python -m tools.maintenance-scheduler.cli list-tasks`
3. Check execution history: `python -m tools.maintenance-scheduler.cli history TASK_ID`

#### High Failure Rate

1. Analyze task configuration and dependencies
2. Check system resources and permissions
3. Review error logs in execution history
4. Consider adjusting timeout or retry settings

#### Storage Issues

1. Clean up old data: `python -m tools.maintenance-scheduler.cli cleanup`
2. Adjust rollback point limits in configuration
3. Monitor storage usage with metrics command

### Logging

```python
import logging

# Enable debug logging
logging.getLogger('tools.maintenance_scheduler').setLevel(logging.DEBUG)

# Log to file
logging.basicConfig(
    filename='data/maintenance/scheduler.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

## Best Practices

### Task Design

1. **Keep tasks focused**: Each task should have a single, clear purpose
2. **Make tasks idempotent**: Tasks should be safe to run multiple times
3. **Include proper error handling**: Handle expected failures gracefully
4. **Set appropriate timeouts**: Prevent tasks from running indefinitely
5. **Use meaningful names and descriptions**: Aid in monitoring and debugging

### Scheduling

1. **Avoid peak hours**: Schedule intensive tasks during low-usage periods
2. **Consider dependencies**: Ensure dependent tasks run in correct order
3. **Use appropriate frequencies**: Balance maintenance needs with system load
4. **Monitor execution patterns**: Adjust schedules based on performance data

### Rollback Strategy

1. **Enable rollbacks for risky operations**: Especially for automated changes
2. **Test rollback procedures**: Verify rollbacks work before relying on them
3. **Document rollback scenarios**: Know when and how to use rollbacks
4. **Monitor rollback point storage**: Clean up old points regularly

## Contributing

### Adding New Task Categories

1. Add category to `TaskCategory` enum in `models.py`
2. Update priority engine category multipliers
3. Add category-specific impact analysis logic
4. Create default task configuration examples

### Extending Priority Analysis

1. Add new impact dimensions to `ImpactAnalysis`
2. Update priority calculation weights
3. Add category-specific analysis logic
4. Test with representative task samples

### Custom Executors

1. Implement async executor function
2. Return standardized result dictionary
3. Handle errors and timeouts appropriately
4. Include relevant metrics in results

## License

This maintenance scheduling system is part of the WAN22 project and follows the same licensing terms.
