---
title: tools.health-checker.parallel_executor
category: api
tags: [api, tools]
---

# tools.health-checker.parallel_executor

Parallel execution system for health checks.

This module provides parallel and asynchronous execution capabilities
for health checks to improve performance in CI/CD environments.

## Classes

### HealthCheckTask

Represents a health check task for parallel execution.

#### Methods

##### __post_init__(self: Any)



### TaskResult

Represents the result of a health check task.

### ResourceMonitor

Monitors system resources during parallel execution.

#### Methods

##### __init__(self: Any)



##### start_monitoring(self: Any)

Start resource monitoring.

##### stop_monitoring(self: Any)

Stop resource monitoring and calculate averages.

##### should_throttle(self: Any) -> bool

Check if execution should be throttled due to resource constraints.

### DependencyResolver

Resolves task dependencies for parallel execution.

#### Methods

##### __init__(self: Any, tasks: <ast.Subscript object at 0x000001942F75FA00>)



##### _build_dependency_graph(self: Any) -> <ast.Subscript object at 0x000001942F6D4850>

Build dependency graph from tasks.

##### _topological_sort(self: Any) -> <ast.Subscript object at 0x000001942FB49090>

Perform topological sort to determine execution order.

##### get_execution_levels(self: Any) -> <ast.Subscript object at 0x000001942FB4A830>

Get tasks grouped by execution level (parallel groups).

##### validate_dependencies(self: Any) -> <ast.Subscript object at 0x000001942FB48700>

Validate that all dependencies exist and there are no cycles.

### ParallelHealthExecutor

Executes health checks in parallel with resource management.

#### Methods

##### __init__(self: Any, max_workers: <ast.Subscript object at 0x000001942FB48F10>, use_processes: bool)



##### execute_tasks(self: Any, tasks: <ast.Subscript object at 0x000001942FB4AE60>) -> <ast.Subscript object at 0x000001942F6B37F0>

Execute health check tasks in parallel.

##### _execute_level(self: Any, task_names: <ast.Subscript object at 0x000001942F6B31F0>, all_tasks: <ast.Subscript object at 0x000001942F6B26B0>)

Execute a single level of tasks in parallel.

##### _execute_with_threads(self: Any, tasks: <ast.Subscript object at 0x000001942F6B3A30>, max_workers: int)

Execute tasks using thread pool.

##### _execute_with_processes(self: Any, tasks: <ast.Subscript object at 0x000001942EFC2E60>, max_workers: int)

Execute tasks using process pool.

##### _execute_single_task(self: Any, task: HealthCheckTask) -> TaskResult

Execute a single health check task.

##### _execute_with_timeout(self: Any, func: Callable, args: Tuple, kwargs: Dict, timeout: int) -> Any

Execute function with timeout.

##### _execute_task_in_process(self: Any, task: HealthCheckTask) -> TaskResult

Execute task in separate process (for ProcessPoolExecutor).

##### _print_execution_summary(self: Any)

Print execution summary.

### AsyncHealthExecutor

Asynchronous executor for I/O-bound health checks.

#### Methods

##### __init__(self: Any, max_concurrent: int)



