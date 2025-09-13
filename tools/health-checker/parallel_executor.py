#!/usr/bin/env python3
"""
Parallel execution system for health checks.

This module provides parallel and asynchronous execution capabilities
for health checks to improve performance in CI/CD environments.
"""

import asyncio
import concurrent.futures
import multiprocessing
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import psutil


@dataclass
class HealthCheckTask:
    """Represents a health check task for parallel execution."""
    name: str
    function: Callable
    args: Tuple = ()
    kwargs: Dict = None
    priority: int = 1  # Higher number = higher priority
    timeout: Optional[int] = None
    dependencies: List[str] = None
    category: str = "general"
    
    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class TaskResult:
    """Represents the result of a health check task."""
    task_name: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    memory_usage: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


class ResourceMonitor:
    """Monitors system resources during parallel execution."""
    
    def __init__(self):
        self.cpu_threshold = 80  # Maximum CPU usage percentage
        self.memory_threshold = 80  # Maximum memory usage percentage
        self.monitoring = False
        self.stats = {
            "max_cpu": 0,
            "max_memory": 0,
            "avg_cpu": 0,
            "avg_memory": 0,
            "samples": []
        }
    
    def start_monitoring(self):
        """Start resource monitoring."""
        self.monitoring = True
        self.stats["samples"] = []
        
        def monitor_loop():
            while self.monitoring:
                try:
                    cpu_percent = psutil.cpu_percent(interval=1)
                    memory_percent = psutil.virtual_memory().percent
                    
                    sample = {
                        "timestamp": time.time(),
                        "cpu": cpu_percent,
                        "memory": memory_percent
                    }
                    
                    self.stats["samples"].append(sample)
                    self.stats["max_cpu"] = max(self.stats["max_cpu"], cpu_percent)
                    self.stats["max_memory"] = max(self.stats["max_memory"], memory_percent)
                    
                    time.sleep(1)
                except Exception:
                    break
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop resource monitoring and calculate averages."""
        self.monitoring = False
        
        if self.stats["samples"]:
            cpu_values = [s["cpu"] for s in self.stats["samples"]]
            memory_values = [s["memory"] for s in self.stats["samples"]]
            
            self.stats["avg_cpu"] = sum(cpu_values) / len(cpu_values)
            self.stats["avg_memory"] = sum(memory_values) / len(memory_values)
    
    def should_throttle(self) -> bool:
        """Check if execution should be throttled due to resource constraints."""
        if not self.stats["samples"]:
            return False
        
        recent_samples = self.stats["samples"][-5:]  # Last 5 samples
        if len(recent_samples) < 3:
            return False
        
        avg_cpu = sum(s["cpu"] for s in recent_samples) / len(recent_samples)
        avg_memory = sum(s["memory"] for s in recent_samples) / len(recent_samples)
        
        return avg_cpu > self.cpu_threshold or avg_memory > self.memory_threshold


class DependencyResolver:
    """Resolves task dependencies for parallel execution."""
    
    def __init__(self, tasks: List[HealthCheckTask]):
        self.tasks = {task.name: task for task in tasks}
        self.dependency_graph = self._build_dependency_graph()
        self.execution_order = self._topological_sort()
    
    def _build_dependency_graph(self) -> Dict[str, List[str]]:
        """Build dependency graph from tasks."""
        graph = {}
        
        for task_name, task in self.tasks.items():
            graph[task_name] = task.dependencies.copy()
        
        return graph
    
    def _topological_sort(self) -> List[List[str]]:
        """Perform topological sort to determine execution order."""
        # Kahn's algorithm for topological sorting
        in_degree = {task: 0 for task in self.tasks}
        
        # Calculate in-degrees
        for task_name, dependencies in self.dependency_graph.items():
            for dep in dependencies:
                if dep in in_degree:
                    in_degree[task_name] += 1
        
        # Find tasks with no dependencies (can run immediately)
        queue = [task for task, degree in in_degree.items() if degree == 0]
        execution_levels = []
        
        while queue:
            # Current level - tasks that can run in parallel
            current_level = queue.copy()
            execution_levels.append(current_level)
            queue = []
            
            # Remove current level tasks and update in-degrees
            for task in current_level:
                for other_task, dependencies in self.dependency_graph.items():
                    if task in dependencies:
                        in_degree[other_task] -= 1
                        if in_degree[other_task] == 0 and other_task not in [t for level in execution_levels for t in level]:
                            queue.append(other_task)
        
        return execution_levels
    
    def get_execution_levels(self) -> List[List[str]]:
        """Get tasks grouped by execution level (parallel groups)."""
        return self.execution_order
    
    def validate_dependencies(self) -> List[str]:
        """Validate that all dependencies exist and there are no cycles."""
        errors = []
        
        # Check for missing dependencies
        for task_name, task in self.tasks.items():
            for dep in task.dependencies:
                if dep not in self.tasks:
                    errors.append(f"Task '{task_name}' depends on non-existent task '{dep}'")
        
        # Check for cycles using DFS
        visited = set()
        rec_stack = set()
        
        def has_cycle(node):
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in self.dependency_graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        for task in self.tasks:
            if task not in visited:
                if has_cycle(task):
                    errors.append(f"Circular dependency detected involving task '{task}'")
        
        return errors


class ParallelHealthExecutor:
    """Executes health checks in parallel with resource management."""
    
    def __init__(self, max_workers: Optional[int] = None, use_processes: bool = False):
        self.max_workers = max_workers or min(multiprocessing.cpu_count(), 8)
        self.use_processes = use_processes
        self.resource_monitor = ResourceMonitor()
        self.results = {}
        self.failed_tasks = []
        
    def execute_tasks(self, tasks: List[HealthCheckTask]) -> Dict[str, TaskResult]:
        """Execute health check tasks in parallel."""
        
        print(f"🚀 Starting parallel execution of {len(tasks)} tasks")
        print(f"   Max workers: {self.max_workers}")
        print(f"   Execution mode: {'processes' if self.use_processes else 'threads'}")
        
        # Validate and resolve dependencies
        resolver = DependencyResolver(tasks)
        dependency_errors = resolver.validate_dependencies()
        
        if dependency_errors:
            print("❌ Dependency validation failed:")
            for error in dependency_errors:
                print(f"   - {error}")
            raise ValueError("Invalid task dependencies")
        
        execution_levels = resolver.get_execution_levels()
        print(f"   Execution levels: {len(execution_levels)}")
        
        # Start resource monitoring
        self.resource_monitor.start_monitoring()
        
        try:
            # Execute tasks level by level
            for level_idx, level_tasks in enumerate(execution_levels):
                print(f"\n📋 Executing level {level_idx + 1}: {len(level_tasks)} tasks")
                self._execute_level(level_tasks, tasks)
                
                # Check if we should throttle between levels
                if self.resource_monitor.should_throttle():
                    print("⚠️ Resource usage high, throttling execution...")
                    time.sleep(2)
        
        finally:
            self.resource_monitor.stop_monitoring()
        
        # Generate execution summary
        self._print_execution_summary()
        
        return self.results
    
    def _execute_level(self, task_names: List[str], all_tasks: List[HealthCheckTask]):
        """Execute a single level of tasks in parallel."""
        
        # Get task objects for this level
        level_tasks = [task for task in all_tasks if task.name in task_names]
        
        # Sort by priority (higher priority first)
        level_tasks.sort(key=lambda t: t.priority, reverse=True)
        
        # Determine number of workers for this level
        level_workers = min(len(level_tasks), self.max_workers)
        
        # Execute tasks
        if self.use_processes:
            self._execute_with_processes(level_tasks, level_workers)
        else:
            self._execute_with_threads(level_tasks, level_workers)
    
    def _execute_with_threads(self, tasks: List[HealthCheckTask], max_workers: int):
        """Execute tasks using thread pool."""
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_task = {}
            
            for task in tasks:
                future = executor.submit(self._execute_single_task, task)
                future_to_task[future] = task
            
            # Collect results as they complete
            for future in as_completed(future_to_task, timeout=300):  # 5 minute timeout
                task = future_to_task[future]
                
                try:
                    result = future.result()
                    self.results[task.name] = result
                    
                    status = "✅" if result.success else "❌"
                    print(f"   {status} {task.name} ({result.execution_time:.2f}s)")
                    
                except Exception as e:
                    error_result = TaskResult(
                        task_name=task.name,
                        success=False,
                        error=str(e),
                        execution_time=0.0
                    )
                    self.results[task.name] = error_result
                    self.failed_tasks.append(task.name)
                    print(f"   ❌ {task.name} (error: {e})")
    
    def _execute_with_processes(self, tasks: List[HealthCheckTask], max_workers: int):
        """Execute tasks using process pool."""
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_task = {}
            
            for task in tasks:
                # For process execution, we need to ensure the function is picklable
                future = executor.submit(self._execute_task_in_process, task)
                future_to_task[future] = task
            
            # Collect results as they complete
            for future in as_completed(future_to_task, timeout=300):
                task = future_to_task[future]
                
                try:
                    result = future.result()
                    self.results[task.name] = result
                    
                    status = "✅" if result.success else "❌"
                    print(f"   {status} {task.name} ({result.execution_time:.2f}s)")
                    
                except Exception as e:
                    error_result = TaskResult(
                        task_name=task.name,
                        success=False,
                        error=str(e),
                        execution_time=0.0
                    )
                    self.results[task.name] = error_result
                    self.failed_tasks.append(task.name)
                    print(f"   ❌ {task.name} (error: {e})")
    
    def _execute_single_task(self, task: HealthCheckTask) -> TaskResult:
        """Execute a single health check task."""
        
        start_time = datetime.now()
        start_memory = psutil.Process().memory_info().rss
        execution_start = time.time()
        
        try:
            # Apply timeout if specified
            if task.timeout:
                result = self._execute_with_timeout(task.function, task.args, task.kwargs, task.timeout)
            else:
                result = task.function(*task.args, **task.kwargs)
            
            execution_time = time.time() - execution_start
            end_memory = psutil.Process().memory_info().rss
            memory_delta = end_memory - start_memory
            
            return TaskResult(
                task_name=task.name,
                success=True,
                result=result,
                execution_time=execution_time,
                memory_usage=memory_delta,
                start_time=start_time,
                end_time=datetime.now()
            )
            
        except Exception as e:
            execution_time = time.time() - execution_start
            
            return TaskResult(
                task_name=task.name,
                success=False,
                error=str(e),
                execution_time=execution_time,
                start_time=start_time,
                end_time=datetime.now()
            )
    
    def _execute_with_timeout(self, func: Callable, args: Tuple, kwargs: Dict, timeout: int) -> Any:
        """Execute function with timeout."""
        
        result_queue = queue.Queue()
        exception_queue = queue.Queue()
        
        def target():
            try:
                result = func(*args, **kwargs)
                result_queue.put(result)
            except Exception as e:
                exception_queue.put(e)
        
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout)
        
        if thread.is_alive():
            # Timeout occurred
            raise TimeoutError(f"Task timed out after {timeout} seconds")
        
        if not exception_queue.empty():
            raise exception_queue.get()
        
        if not result_queue.empty():
            return result_queue.get()
        
        raise RuntimeError("Task completed but no result available")
    
    def _execute_task_in_process(self, task: HealthCheckTask) -> TaskResult:
        """Execute task in separate process (for ProcessPoolExecutor)."""
        # This is a simplified version for process execution
        # In practice, you'd need to ensure all functions are properly serializable
        return self._execute_single_task(task)
    
    def _print_execution_summary(self):
        """Print execution summary."""
        
        total_tasks = len(self.results)
        successful_tasks = sum(1 for r in self.results.values() if r.success)
        failed_tasks = total_tasks - successful_tasks
        
        total_time = sum(r.execution_time for r in self.results.values())
        avg_time = total_time / total_tasks if total_tasks > 0 else 0
        
        print(f"\n📊 Execution Summary:")
        print(f"   Total tasks: {total_tasks}")
        print(f"   Successful: {successful_tasks}")
        print(f"   Failed: {failed_tasks}")
        print(f"   Total execution time: {total_time:.2f}s")
        print(f"   Average task time: {avg_time:.2f}s")
        
        # Resource usage summary
        stats = self.resource_monitor.stats
        print(f"   Max CPU usage: {stats['max_cpu']:.1f}%")
        print(f"   Max memory usage: {stats['max_memory']:.1f}%")
        print(f"   Avg CPU usage: {stats['avg_cpu']:.1f}%")
        print(f"   Avg memory usage: {stats['avg_memory']:.1f}%")
        
        if self.failed_tasks:
            print(f"\n❌ Failed tasks:")
            for task_name in self.failed_tasks:
                result = self.results[task_name]
                print(f"   - {task_name}: {result.error}")


# Async version for I/O bound tasks
class AsyncHealthExecutor:
    """Asynchronous executor for I/O-bound health checks."""
    
    def __init__(self, max_concurrent: int = 10):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.results = {}
    
    async def execute_async_tasks(self, tasks: List[HealthCheckTask]) -> Dict[str, TaskResult]:
        """Execute async health check tasks."""
        
        print(f"🚀 Starting async execution of {len(tasks)} tasks")
        print(f"   Max concurrent: {self.max_concurrent}")
        
        # Create coroutines for all tasks
        coroutines = [self._execute_async_task(task) for task in tasks]
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(results):
            task_name = tasks[i].name
            
            if isinstance(result, Exception):
                self.results[task_name] = TaskResult(
                    task_name=task_name,
                    success=False,
                    error=str(result)
                )
            else:
                self.results[task_name] = result
        
        return self.results
    
    async def _execute_async_task(self, task: HealthCheckTask) -> TaskResult:
        """Execute a single async task."""
        
        async with self.semaphore:
            start_time = datetime.now()
            execution_start = time.time()
            
            try:
                # If the function is not async, run it in a thread pool
                if asyncio.iscoroutinefunction(task.function):
                    result = await task.function(*task.args, **task.kwargs)
                else:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None, 
                        lambda: task.function(*task.args, **task.kwargs)
                    )
                
                execution_time = time.time() - execution_start
                
                return TaskResult(
                    task_name=task.name,
                    success=True,
                    result=result,
                    execution_time=execution_time,
                    start_time=start_time,
                    end_time=datetime.now()
                )
                
            except Exception as e:
                execution_time = time.time() - execution_start
                
                return TaskResult(
                    task_name=task.name,
                    success=False,
                    error=str(e),
                    execution_time=execution_time,
                    start_time=start_time,
                    end_time=datetime.now()
                )


def main():
    """Main function for testing parallel execution."""
    
    import random
    
    def dummy_check(name: str, duration: float = 1.0) -> Dict:
        """Dummy health check for testing."""
        time.sleep(duration)
        success = random.random() > 0.1  # 90% success rate
        return {
            "status": "passed" if success else "failed",
            "message": f"Check {name} completed",
            "duration": duration
        }
    
    # Create test tasks
    tasks = [
        HealthCheckTask("syntax_check", dummy_check, ("syntax", 0.5), priority=3),
        HealthCheckTask("import_check", dummy_check, ("imports", 1.0), priority=3),
        HealthCheckTask("config_check", dummy_check, ("config", 0.8), priority=2),
        HealthCheckTask("test_health", dummy_check, ("tests", 2.0), priority=2, dependencies=["syntax_check"]),
        HealthCheckTask("doc_check", dummy_check, ("docs", 1.5), priority=1),
        HealthCheckTask("integration_test", dummy_check, ("integration", 3.0), priority=1, 
                       dependencies=["test_health", "config_check"]),
    ]
    
    # Test parallel execution
    executor = ParallelHealthExecutor(max_workers=4)
    results = executor.execute_tasks(tasks)
    
    print(f"\n✅ Parallel execution completed")
    print(f"   Results: {len(results)} tasks")


if __name__ == "__main__":
    main()
