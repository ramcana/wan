---
title: tools.maintenance-scheduler.task_manager
category: api
tags: [api, tools]
---

# tools.maintenance-scheduler.task_manager

Task management system for maintenance tasks.

## Classes

### MaintenanceTaskManager

Manages maintenance tasks including storage, retrieval, and lifecycle management.

#### Methods

##### __init__(self: Any, storage_path: <ast.Subscript object at 0x00000194346091B0>)



##### add_task(self: Any, task: MaintenanceTask) -> None

Add a new maintenance task.

##### remove_task(self: Any, task_id: str) -> bool

Remove a maintenance task.

##### get_task(self: Any, task_id: str) -> <ast.Subscript object at 0x000001942F0294B0>

Get a specific task by ID.

##### get_all_tasks(self: Any) -> <ast.Subscript object at 0x000001942F028D90>

Get all maintenance tasks.

##### get_tasks_by_category(self: Any, category: TaskCategory) -> <ast.Subscript object at 0x000001942F028460>

Get tasks filtered by category.

##### get_tasks_by_priority(self: Any, priority: TaskPriority) -> <ast.Subscript object at 0x000001942F029750>

Get tasks filtered by priority.

##### get_tasks_by_tags(self: Any, tags: <ast.Subscript object at 0x000001942F029BA0>) -> <ast.Subscript object at 0x000001942F028CD0>

Get tasks that have any of the specified tags.

##### update_task(self: Any, task_id: str, updates: Dict) -> bool

Update a task with new values.

##### get_task_dependencies(self: Any, task_id: str) -> <ast.Subscript object at 0x0000019433D2CA00>

Get dependency information for a task.

##### validate_task_dependencies(self: Any, task: MaintenanceTask) -> <ast.Subscript object at 0x0000019433D2CD00>

Validate task dependencies and return any issues.

##### get_execution_order(self: Any) -> <ast.Subscript object at 0x0000019434444BB0>

Get tasks in execution order, grouped by dependency level.
Returns list of lists, where each inner list contains tasks that can run in parallel.

##### create_default_tasks(self: Any) -> None

Create a set of default maintenance tasks.

##### _load_tasks(self: Any) -> None

Load tasks from storage.

##### _save_tasks(self: Any) -> None

Save tasks to storage.

##### _serialize_task(self: Any, task: MaintenanceTask) -> Dict

Serialize a task to dictionary.

##### _deserialize_task(self: Any, data: Dict) -> <ast.Subscript object at 0x00000194344DB820>

Deserialize a task from dictionary.

