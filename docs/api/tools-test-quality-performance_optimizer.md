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



##### profile_test_suite(self: Any, test_files: <ast.Subscript object at 0x000001942834D630>) -> TestPerformanceProfile

Profile performance of test suite

##### _discover_test_files(self: Any) -> <ast.Subscript object at 0x0000019428328340>

Discover test files in the project

##### _run_performance_tests(self: Any, test_files: <ast.Subscript object at 0x00000194283281F0>) -> <ast.Subscript object at 0x0000019428307BB0>

Run tests with performance monitoring

##### _profile_test_file(self: Any, test_file: Path) -> <ast.Subscript object at 0x0000019428305FF0>

Profile performance of a single test file

##### _parse_pytest_output(self: Any, test_file: Path, output: str, total_time: float) -> <ast.Subscript object at 0x000001942833F610>

Parse pytest output to extract individual test timings

##### _analyze_performance_data(self: Any, metrics: <ast.Subscript object at 0x000001942833F4C0>) -> TestPerformanceProfile

Analyze performance metrics to create profile

##### _percentile(self: Any, data: <ast.Subscript object at 0x000001942833D990>, percentile: int) -> float

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

##### get_cache_stats(self: Any) -> <ast.Subscript object at 0x0000019427F97670>

Get cache performance statistics

### PerformanceRegressionDetector

Detects performance regressions in test execution

#### Methods

##### __init__(self: Any, project_root: Path)



##### _init_database(self: Any)

Initialize performance tracking database

##### record_performance(self: Any, metrics: <ast.Subscript object at 0x0000019427F95A50>)

Record performance metrics for regression detection

##### detect_regressions(self: Any, current_metrics: <ast.Subscript object at 0x0000019428D7C160>) -> <ast.Subscript object at 0x0000019427547970>

Detect performance regressions compared to baseline

##### get_performance_history(self: Any, test_id: str, days: int) -> <ast.Subscript object at 0x0000019429C053C0>

Get performance history for a specific test

##### _get_current_commit(self: Any) -> <ast.Subscript object at 0x00000194288B3AC0>

Get current git commit hash

##### _get_current_branch(self: Any) -> <ast.Subscript object at 0x00000194288B14E0>

Get current git branch

### TestOptimizationRecommendationEngine

Generates optimization recommendations based on performance analysis

#### Methods

##### __init__(self: Any, project_root: Path)



##### generate_recommendations(self: Any, profile: TestPerformanceProfile, regressions: <ast.Subscript object at 0x00000194288B3BE0>) -> <ast.Subscript object at 0x00000194288B2530>

Generate optimization recommendations

##### _recommend_slow_test_optimizations(self: Any, slow_tests: <ast.Subscript object at 0x0000019427B88B80>) -> <ast.Subscript object at 0x0000019428149150>

Generate recommendations for slow tests

##### _recommend_regression_fixes(self: Any, regressions: <ast.Subscript object at 0x0000019428149000>) -> <ast.Subscript object at 0x0000019428148460>

Generate recommendations for performance regressions

##### _recommend_general_optimizations(self: Any, profile: TestPerformanceProfile) -> <ast.Subscript object at 0x000001942814B580>

Generate general optimization recommendations

### TestPerformanceOptimizer

Main system that orchestrates all performance optimization components

#### Methods

##### __init__(self: Any, project_root: Path)



##### optimize_test_performance(self: Any, test_files: <ast.Subscript object at 0x000001942814BDC0>) -> <ast.Subscript object at 0x000001942B2F94E0>

Run comprehensive test performance optimization

##### save_optimization_report(self: Any, optimization_result: <ast.Subscript object at 0x000001942B2F9690>, output_path: Path)

Save optimization report to file

