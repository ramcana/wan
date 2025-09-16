---
title: tools.health-checker.performance_optimizer
category: api
tags: [api, tools]
---

# tools.health-checker.performance_optimizer

Performance optimizer for health monitoring system.

This module provides performance profiling, caching, and optimization
capabilities for health checks to ensure fast execution in CI/CD environments.

## Classes

### HealthCheckCache

Caching system for health check results.

#### Methods

##### __init__(self: Any, cache_dir: Path, default_ttl: int)



##### _get_cache_key(self: Any, check_name: str, inputs: Dict) -> str

Generate cache key from check name and inputs.

##### _get_cache_file(self: Any, cache_key: str) -> Path

Get cache file path for given key.

##### _is_cache_valid(self: Any, cache_file: Path, ttl: int) -> bool

Check if cache file is still valid.

##### get(self: Any, check_name: str, inputs: Dict, ttl: int) -> <ast.Subscript object at 0x000001942C6D4E50>

Get cached result for health check.

##### set(self: Any, check_name: str, inputs: Dict, result: Any)

Cache result for health check.

##### _add_to_memory_cache(self: Any, cache_key: str, result: Any)

Add result to memory cache with LRU eviction.

##### invalidate(self: Any, check_name: str)

Invalidate cache entries.

##### cleanup_expired(self: Any, ttl: int)

Clean up expired cache entries.

### PerformanceProfiler

Performance profiler for health checks.

#### Methods

##### __init__(self: Any, output_dir: Path)



##### profile_function(self: Any, func_name: str)

Decorator to profile function execution.

##### get_performance_summary(self: Any) -> Dict

Get performance summary for all profiled functions.

##### generate_optimization_recommendations(self: Any) -> <ast.Subscript object at 0x000001942757FC10>

Generate optimization recommendations based on profiling data.

### IncrementalAnalyzer

Incremental analysis system for large codebases.

#### Methods

##### __init__(self: Any, state_file: Path)



##### _load_state(self: Any) -> Dict

Load previous analysis state.

##### _save_state(self: Any)

Save current analysis state.

##### get_file_hash(self: Any, file_path: Path) -> str

Get hash of file content.

##### get_changed_files(self: Any, file_patterns: <ast.Subscript object at 0x000001942757DC60>) -> <ast.Subscript object at 0x000001942757CB80>

Get list of files that have changed since last analysis.

##### should_run_check(self: Any, check_name: str, dependencies: <ast.Subscript object at 0x000001942757C280>) -> bool

Determine if a check should run based on changes.

##### finalize_analysis(self: Any)

Finalize analysis and save state.

### LightweightHealthChecker

Lightweight health checker for frequent execution.

#### Methods

##### __init__(self: Any, cache: HealthCheckCache, profiler: PerformanceProfiler)



##### lightweight_checks(self: Any) -> <ast.Subscript object at 0x0000019427B8B4C0>

List of checks suitable for lightweight mode.

##### run_lightweight_health_check(self: Any) -> Dict

Run lightweight health check with minimal overhead.

##### _run_single_lightweight_check(self: Any, check_name: str) -> Dict

Run a single lightweight health check.

##### _check_python_syntax(self: Any) -> Dict

Quick syntax check for Python files.

##### _check_imports(self: Any) -> Dict

Quick import validation.

##### _check_config_files(self: Any) -> Dict

Quick configuration file validation.

##### _check_basic_test_health(self: Any) -> Dict

Quick test health check.

##### _check_critical_files(self: Any) -> Dict

Check for presence of critical files.

