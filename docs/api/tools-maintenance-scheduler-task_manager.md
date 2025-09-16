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

##### __init__(self: Any, storage_path: <ast.Subscript object at 0x000001942C81D0F0>)



##### add_task(self: Any, task: MaintenanceTask) -> None

Add a new maintenance task.

##### remove_task(self: Any, task_id: str) -> bool

Remove a maintenance task.

##### get_task(self: Any, task_id: str) -> <ast.Subscript object at 0x000001942C81F370>

Get a specific task by ID.

##### get_all_tasks(self: Any) -> <ast.Subscript object at 0x000001942C81FC10>

Get all maintenance tasks.

##### get_tasks_by_category(self: Any, category: TaskCategory) -> <ast.Subscript object at 0x000001942C81D2A0>

Get tasks filtered by category.

##### get_tasks_by_priority(self: Any, priority: TaskPriority) -> <ast.Subscript object at 0x000001942C81ED70>

Get tasks filtered by priority.

##### get_tasks_by_tags(self: Any, tags: <ast.Subscript object at 0x000001942C81DC00>) -> <ast.Subscript object at 0x0000019428935DE0>

Get tasks that have any of the specified tags.

##### update_task(self: Any, task_id: str, updates: Dict) -> bool

Update a task with new values.

##### get_task_dependencies(self: Any, task_id: str) -> <ast.Subscript object at 0x000001942757B2E0>

Get dependency information for a task.

##### validate_task_dependencies(self: Any, task: MaintenanceTask) -> <ast.Subscript object at 0x00000194275792A0>

Validate task dependencies and return any issues.

##### get_execution_order(self: Any) -> <ast.Subscript object at 0x0000019427579B40>

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

##### _deserialize_task(self: Any, data: Dict) -> <ast.Subscript object at 0x00000194280531F0>

Deserialize a task from dictionary.

