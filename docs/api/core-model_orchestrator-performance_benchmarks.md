---
title: core.model_orchestrator.performance_benchmarks
category: api
tags: [api, core]
---

# core.model_orchestrator.performance_benchmarks

Performance benchmarking and optimization testing for the model orchestrator.

## Classes

### BenchmarkResult

Results from a performance benchmark.

### MockFileSpec

Mock file specification for testing.

### MockHttpServer

Mock HTTP server for testing downloads without external dependencies.

#### Methods

##### __init__(self: Any, port: int)



##### add_file(self: Any, path: str, content: bytes)

Add a file to serve.

##### add_random_file(self: Any, path: str, size: int) -> str

Add a random file of specified size.

##### set_bandwidth_limit(self: Any, bytes_per_second: <ast.Subscript object at 0x0000019427C49120>)

Set bandwidth limit for testing.

##### set_latency(self: Any, milliseconds: int)

Set artificial latency for testing.

##### set_error_rate(self: Any, rate: float)

Set error rate for testing (0.0 to 1.0).

##### get_stats(self: Any) -> <ast.Subscript object at 0x0000019427C485B0>

Get server statistics.

### PerformanceBenchmark

Comprehensive performance benchmarking suite for the model orchestrator.

#### Methods

##### __init__(self: Any, temp_dir: <ast.Subscript object at 0x0000019427C48370>)



##### setup(self: Any)

Set up benchmark environment.

##### teardown(self: Any)

Clean up benchmark environment.

##### _prepare_test_files(self: Any)

Prepare test files for benchmarking.

##### run_all_benchmarks(self: Any) -> <ast.Subscript object at 0x0000019427C47490>

Run all performance benchmarks.

##### _benchmark_concurrent_downloads(self: Any) -> <ast.Subscript object at 0x0000019427C344C0>

Benchmark concurrent download performance.

##### _benchmark_bandwidth_limiting(self: Any) -> <ast.Subscript object at 0x000001942C517CD0>

Benchmark bandwidth limiting effectiveness.

##### _benchmark_adaptive_chunking(self: Any) -> <ast.Subscript object at 0x00000194280CD240>

Benchmark adaptive chunking performance.

##### _benchmark_memory_optimization(self: Any) -> <ast.Subscript object at 0x0000019428063D30>

Benchmark memory optimization features.

##### _benchmark_connection_pooling(self: Any) -> <ast.Subscript object at 0x0000019427FE9A80>

Benchmark connection pooling effectiveness.

##### _benchmark_queue_management(self: Any) -> <ast.Subscript object at 0x0000019427CB2E90>

Benchmark download queue management.

##### _benchmark_error_recovery(self: Any) -> <ast.Subscript object at 0x0000019428151660>

Benchmark error recovery and retry mechanisms.

##### _benchmark_scalability(self: Any) -> <ast.Subscript object at 0x000001942818C250>

Benchmark scalability with increasing load.

##### _simulate_download_time(self: Any, file_specs: <ast.Subscript object at 0x000001942818C3A0>, concurrency: int) -> float

Simulate realistic download time based on file specs and concurrency.

##### _get_memory_usage(self: Any) -> float

Get current memory usage in MB.

##### _get_cpu_usage(self: Any) -> float

Get current CPU usage percentage.

##### _generate_summary_report(self: Any)

Generate a comprehensive summary report.

