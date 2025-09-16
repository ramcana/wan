---
title: tools.test-quality.performance_optimizer
category: api
tags: [api, tools]
---

# tools.test-quality.performance_optimizer



## Classes

### TestPerformanceMetric

Performance metrics for a single test

### TestPerformanceProfile

Complete performance profile for test execution

### PerformanceRegression

Detected performance regression

### OptimizationRecommendation

Performance optimization recommendation

### TestPerformanceProfiler

Profiles test performance and identifies bottlenecks

#### Methods

##### __init__(self: Any, project_root: Path)



##### profile_test_suite(self: Any, test_files: <ast.Subscript object at 0x00000194302DD630>) -> TestPerformanceProfile

Profile performance of test suite

##### _discover_test_files(self: Any) -> <ast.Subscript object at 0x0000019430288340>

Discover test files in the project

##### _run_performance_tests(self: Any, test_files: <ast.Subscript object at 0x00000194302881F0>) -> <ast.Subscript object at 0x0000019432E65210>

Run tests with performance monitoring

##### _profile_test_file(self: Any, test_file: Path) -> <ast.Subscript object at 0x0000019432E64FD0>

Profile performance of a single test file

##### _parse_pytest_output(self: Any, test_file: Path, output: str, total_time: float) -> <ast.Subscript object at 0x0000019432D7F9D0>

Parse pytest output to extract individual test timings

##### _analyze_performance_data(self: Any, metrics: <ast.Subscript object at 0x0000019432D7E170>) -> TestPerformanceProfile

Analyze performance metrics to create profile

##### _percentile(self: Any, data: <ast.Subscript object at 0x0000019432D7DCF0>, percentile: int) -> float

Calculate percentile of data

### TestCacheManager

Manages test caching and memoization for expensive operations

#### Methods

##### __init__(self: Any, project_root: Path)



##### cached_test_data(self: Any, cache_key: str, generator_func: Callable, ttl: int)

Decorator for caching expensive test data generation

##### _hash_args(self: Any, args: tuple, kwargs: dict) -> str

Create hash of function arguments for cache key

##### clear_cache(self: Any, pattern: str)

Clear cache entries matching pattern

##### get_cache_stats(self: Any) -> <ast.Subscript object at 0x0000019434076740>

Get cache performance statistics

### PerformanceRegressionDetector

Detects performance regressions in test execution

#### Methods

##### __init__(self: Any, project_root: Path)



##### _init_database(self: Any)

Initialize performance tracking database

##### record_performance(self: Any, metrics: <ast.Subscript object at 0x0000019434074610>)

Record performance metrics for regression detection

##### detect_regressions(self: Any, current_metrics: <ast.Subscript object at 0x000001942F3326E0>) -> <ast.Subscript object at 0x000001942F331A20>

Detect performance regressions compared to baseline

##### get_performance_history(self: Any, test_id: str, days: int) -> <ast.Subscript object at 0x000001942FC676A0>

Get performance history for a specific test

##### _get_current_commit(self: Any) -> <ast.Subscript object at 0x0000019431A9D150>

Get current git commit hash

##### _get_current_branch(self: Any) -> <ast.Subscript object at 0x0000019431A9E710>

Get current git branch

### TestOptimizationRecommendationEngine

Generates optimization recommendations based on performance analysis

#### Methods

##### __init__(self: Any, project_root: Path)



##### generate_recommendations(self: Any, profile: TestPerformanceProfile, regressions: <ast.Subscript object at 0x0000019431A9FB20>) -> <ast.Subscript object at 0x0000019431A9F010>

Generate optimization recommendations

##### _recommend_slow_test_optimizations(self: Any, slow_tests: <ast.Subscript object at 0x0000019431A9ED10>) -> <ast.Subscript object at 0x000001942F8432B0>

Generate recommendations for slow tests

##### _recommend_regression_fixes(self: Any, regressions: <ast.Subscript object at 0x000001942F841750>) -> <ast.Subscript object at 0x00000194318D4220>

Generate recommendations for performance regressions

##### _recommend_general_optimizations(self: Any, profile: TestPerformanceProfile) -> <ast.Subscript object at 0x000001942F048AF0>

Generate general optimization recommendations

### TestPerformanceOptimizer

Main system that orchestrates all performance optimization components

#### Methods

##### __init__(self: Any, project_root: Path)



##### optimize_test_performance(self: Any, test_files: <ast.Subscript object at 0x000001942F049660>) -> <ast.Subscript object at 0x000001942F048190>

Run comprehensive test performance optimization

##### save_optimization_report(self: Any, optimization_result: <ast.Subscript object at 0x000001942F049930>, output_path: Path)

Save optimization report to file

