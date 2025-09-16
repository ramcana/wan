---
title: tools.health-checker.health_checker
category: api
tags: [api, tools]
---

# tools.health-checker.health_checker

Main project health checker implementation

## Classes

### ProjectHealthChecker

Main health checker that orchestrates all health checks and generates reports

#### Methods

##### __init__(self: Any, config: <ast.Subscript object at 0x00000194280EAD40>)



##### _run_single_check(self: Any, category: HealthCategory) -> ComponentHealth

Run a single health check

##### _create_error_component(self: Any, category: HealthCategory, error_msg: str) -> ComponentHealth

Create a component health result for errors

##### _calculate_overall_score(self: Any, components: <ast.Subscript object at 0x000001942CC0B3D0>) -> float

Calculate weighted overall health score

##### get_health_score(self: Any) -> float

Get current health score (synchronous version)

##### _load_health_trends(self: Any) -> HealthTrends

Load historical health trends

##### _save_health_history(self: Any, report: HealthReport) -> None

Save health report to history

##### schedule_health_checks(self: Any) -> None

Schedule automated health checks

##### _run_cached_check(self: Any, checker: Any, category: HealthCategory)

Run health check with caching support.

##### _get_cache_inputs_for_category(self: Any, category: HealthCategory) -> Dict

Get cache inputs for a specific health check category.

##### _get_files_hash(self: Any, files: <ast.Subscript object at 0x0000019427F13070>) -> str

Get hash of file modification times for cache invalidation.

##### _get_check_priority(self: Any, category: HealthCategory) -> int

Get priority for health check category.

##### _get_check_timeout(self: Any, category: HealthCategory) -> int

Get timeout for health check category.

##### _convert_lightweight_results(self: Any, lightweight_results: Dict) -> HealthReport

Convert lightweight check results to HealthReport format.

##### clear_cache(self: Any)

Clear health check cache.

##### get_performance_summary(self: Any) -> Dict

Get performance summary from profiler.

##### get_optimization_recommendations(self: Any) -> <ast.Subscript object at 0x000001942C800100>

Get performance optimization recommendations.

### Severity



### HealthCategory



### HealthConfig



#### Methods

##### __init__(self: Any)



### HealthIssue



#### Methods

##### __init__(self: Any, severity: Any, category: Any, title: Any, description: Any, affected_components: Any, remediation_steps: Any)



### ComponentHealth



#### Methods

##### __init__(self: Any, component_name: Any, category: Any, score: Any, status: Any, issues: Any, metrics: Any)



### HealthTrends



#### Methods

##### __init__(self: Any, score_history: Any, issue_trends: Any, improvement_rate: Any, degradation_alerts: Any)



### HealthReport



#### Methods

##### __init__(self: Any, timestamp: Any, overall_score: Any, component_scores: Any, issues: Any, recommendations: Any, trends: Any, metadata: Any)



##### get_issues_by_category(self: Any, category: Any)



##### get_critical_issues(self: Any)



##### get_issues_by_severity(self: Any, severity: Any)



### TestHealthChecker



#### Methods

##### __init__(self: Any, config: Any)



##### check_health(self: Any)



### DocumentationHealthChecker



#### Methods

##### __init__(self: Any, config: Any)



##### check_health(self: Any)



### ConfigurationHealthChecker



#### Methods

##### __init__(self: Any, config: Any)



##### check_health(self: Any)



### CodeQualityChecker



#### Methods

##### __init__(self: Any, config: Any)



##### check_health(self: Any)



### HealthCheckCache



#### Methods

##### get(self: Any, key: Any, inputs: Any, ttl: Any)



##### set(self: Any, key: Any, inputs: Any, result: Any)



##### invalidate(self: Any)



### PerformanceProfiler



#### Methods

##### get_performance_summary(self: Any)



##### generate_optimization_recommendations(self: Any)



### LightweightHealthChecker



#### Methods

##### __init__(self: Any, cache: Any, profiler: Any)



##### run_lightweight_health_check(self: Any)



### HealthCheckTask



#### Methods

##### __init__(self: Any, name: Any, function: Any, args: Any, kwargs: Any, priority: Any, timeout: Any, category: Any)



### ParallelHealthExecutor



#### Methods

##### execute_tasks(self: Any, tasks: Any)



## Constants

### LOW

Type: `str`

Value: `low`

### MEDIUM

Type: `str`

Value: `medium`

### HIGH

Type: `str`

Value: `high`

### CRITICAL

Type: `str`

Value: `critical`

### TESTS

Type: `str`

Value: `tests`

### DOCUMENTATION

Type: `str`

Value: `documentation`

### CONFIGURATION

Type: `str`

Value: `configuration`

### CODE_QUALITY

Type: `str`

Value: `code_quality`

### PERFORMANCE

Type: `str`

Value: `performance`

### SECURITY

Type: `str`

Value: `security`

