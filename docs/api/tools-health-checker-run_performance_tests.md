---
title: tools.health-checker.run_performance_tests
category: api
tags: [api, tools]
---

# tools.health-checker.run_performance_tests

Performance testing script for health monitoring system.

This script runs various performance tests to validate and optimize
the health monitoring system performance.

## Classes

### HealthPerformanceTester

Performance tester for health monitoring system.

#### Methods

##### __init__(self: Any)



##### run_all_performance_tests(self: Any) -> Dict

Run comprehensive performance tests.

##### test_baseline_performance(self: Any) -> Dict

Test baseline sequential health check performance.

##### test_parallel_performance(self: Any) -> Dict

Test parallel execution performance.

##### test_cached_performance(self: Any) -> Dict

Test cached execution performance.

##### test_lightweight_performance(self: Any) -> Dict

Test lightweight execution performance.

##### test_incremental_performance(self: Any) -> Dict

Test incremental analysis performance.

##### test_resource_usage(self: Any) -> Dict

Test resource usage during health checks.

##### calculate_performance_improvements(self: Any)

Calculate performance improvements from optimizations.

##### generate_performance_report(self: Any)

Generate comprehensive performance report.

##### generate_performance_recommendations(self: Any) -> <ast.Subscript object at 0x000001942836D360>

Generate performance optimization recommendations.

