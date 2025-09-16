---
title: tools.maintenance-scheduler.scheduler
category: api
tags: [api, tools]
---

# tools.maintenance-scheduler.scheduler

Main maintenance scheduler that orchestrates automated maintenance tasks.

## Classes

### MaintenanceScheduler

Main scheduler for automated maintenance tasks.

Handles task scheduling, execution, prioritization, and monitoring.

#### Methods

##### __init__(self: Any, config: <ast.Subscript object at 0x000001942A2E4310>)



##### start(self: Any) -> None

Start the maintenance scheduler.

##### stop(self: Any) -> None

Stop the maintenance scheduler.

##### add_task(self: Any, task: MaintenanceTask, schedule: <ast.Subscript object at 0x0000019427F117E0>) -> None

Add a maintenance task to the scheduler.

##### remove_task(self: Any, task_id: str) -> bool

Remove a task from the scheduler.

##### run_task_now(self: Any, task_id: str) -> <ast.Subscript object at 0x000001942CC899C0>

Run a specific task immediately.

##### get_next_scheduled_tasks(self: Any, limit: int) -> <ast.Subscript object at 0x000001942CC886A0>

Get the next tasks scheduled to run.

##### get_running_tasks(self: Any) -> <ast.Subscript object at 0x000001942CC8B8B0>

Get list of currently running task IDs.

##### get_task_history(self: Any, task_id: str, limit: int) -> <ast.Subscript object at 0x000001942CC8BD60>

Get execution history for a specific task.

##### _scheduler_loop(self: Any) -> None

Main scheduler loop that runs in a separate thread.

##### _check_and_run_tasks(self: Any) -> None

Check for tasks that need to run and execute them.

##### _get_tasks_ready_to_run(self: Any, now: datetime) -> <ast.Subscript object at 0x000001942CB37130>

Get tasks that are ready to run based on schedule.

##### _is_task_due(self: Any, task: MaintenanceTask, schedule: TaskSchedule, now: datetime) -> bool

Check if a task is due to run.

##### _can_run_task(self: Any, task: MaintenanceTask, now: datetime) -> bool

Check if a task can be run (dependencies, constraints, etc.).

##### _start_task_execution(self: Any, task: MaintenanceTask) -> None

Start executing a task asynchronously.

##### _update_next_run_time(self: Any, task: MaintenanceTask) -> None

Update the next run time for a task based on its schedule.

