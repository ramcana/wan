---
title: tests.test_performance_benchmarks
category: api
tags: [api, tests]
---

# tests.test_performance_benchmarks

Performance Benchmarking Tests
Comprehensive performance testing for generation speed and resource usage

## Classes

### PerformanceBenchmarkSuite

Performance benchmark test suite

#### Methods

##### __init__(self: Any)



##### record_generation_benchmark(self: Any, model_type: str, resolution: str, duration: float, success: bool, metadata: dict)

Record generation benchmark result

##### record_api_benchmark(self: Any, endpoint: str, duration: float, status_code: int, metadata: dict)

Record API benchmark result

##### record_resource_benchmark(self: Any, test_name: str, peak_cpu: float, peak_memory: float, peak_gpu_memory: float)

Record resource usage benchmark

##### save_results(self: Any, filepath: str)

Save benchmark results to file

##### get_summary(self: Any) -> dict

Get benchmark summary

##### _calculate_average_api_time(self: Any) -> float

Calculate average API response time

##### _get_peak_resource_usage(self: Any) -> dict

Get peak resource usage across all tests

### TestGenerationPerformanceBenchmarks

Generation performance benchmark tests

### TestAPIPerformanceBenchmarks

API performance benchmark tests

### TestResourceUsageBenchmarks

Resource usage benchmark tests

### TaskStatus



## Constants

### COMPLETED

Type: `str`

Value: `completed`

### FAILED

Type: `str`

Value: `failed`

### PENDING

Type: `str`

Value: `pending`

### PROCESSING

Type: `str`

Value: `processing`

