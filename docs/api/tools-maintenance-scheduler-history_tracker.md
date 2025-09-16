---
title: tools.maintenance-scheduler.history_tracker
category: api
tags: [api, tools]
---

# tools.maintenance-scheduler.history_tracker

History tracking system for maintenance operations.

## Classes

### MaintenanceHistoryTracker

Tracks and manages maintenance operation history and metrics.

#### Methods

##### __init__(self: Any, storage_path: <ast.Subscript object at 0x0000019427919F30>)



##### record_execution(self: Any, task: MaintenanceTask, result: MaintenanceResult) -> None

Record a maintenance task execution.

##### get_task_history(self: Any, task_id: str, limit: int) -> <ast.Subscript object at 0x000001942791BAC0>

Get execution history for a specific task.

##### get_recent_execution(self: Any, task_id: str) -> <ast.Subscript object at 0x000001942791B220>

Get the most recent execution for a task.

##### get_history_by_date_range(self: Any, start_date: datetime, end_date: datetime) -> <ast.Subscript object at 0x0000019427919930>

Get history records within a date range.

##### get_failed_executions(self: Any, days: int) -> <ast.Subscript object at 0x0000019428929F00>

Get failed executions from the last N days.

##### get_task_success_rate(self: Any, task_id: str, days: int) -> float

Calculate success rate for a task over the last N days.

##### get_average_duration(self: Any, task_id: str, days: int) -> <ast.Subscript object at 0x0000019428929C60>

Get average execution duration for a task.

##### get_maintenance_metrics(self: Any, days: int) -> MaintenanceMetrics

Calculate comprehensive maintenance metrics.

##### get_task_performance_trends(self: Any, task_id: str, days: int) -> <ast.Subscript object at 0x000001942C6871C0>

Get performance trends for a task over time.

##### cleanup_old_history(self: Any, days_to_keep: int) -> int

Remove history records older than specified days.

##### export_history(self: Any, output_path: str, task_id: <ast.Subscript object at 0x0000019427B63040>, start_date: <ast.Subscript object at 0x0000019427B62410>, end_date: <ast.Subscript object at 0x0000019427B62B90>) -> None

Export history to a file.

##### _load_history(self: Any) -> None

Load history from storage.

##### _save_history(self: Any) -> None

Save history to storage.

##### _serialize_history_record(self: Any, record: MaintenanceHistory) -> Dict

Serialize a history record to dictionary.

##### _deserialize_history_record(self: Any, data: Dict) -> <ast.Subscript object at 0x0000019427CA7B80>

Deserialize a history record from dictionary.

##### _get_current_git_commit(self: Any) -> <ast.Subscript object at 0x0000019427CA7E20>

Get current git commit hash.

