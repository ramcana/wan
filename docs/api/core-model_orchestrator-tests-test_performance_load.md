---
title: core.model_orchestrator.tests.test_performance_load
category: api
tags: [api, core]
---

# core.model_orchestrator.tests.test_performance_load

Performance and load testing suites for Model Orchestrator.

Tests system behavior under various load conditions:
- Concurrent downloads and access patterns
- Large model handling and memory usage
- Network bandwidth and timeout scenarios
- Storage backend performance characteristics

## Classes

### PerformanceTestBase

Base class for performance tests with common utilities.

#### Methods

##### performance_manifest(self: Any, temp_models_root: Any)

Create manifest with multiple models for performance testing.

##### temp_models_root(self: Any)

Create temporary models root directory.

##### performance_orchestrator(self: Any, temp_models_root: Any, performance_manifest: Any)

Set up orchestrator for performance testing.

##### measure_execution_time(self: Any, func: Any)

Measure execution time of a function.

##### measure_memory_usage(self: Any, func: Any)

Measure memory usage during function execution.

##### create_mock_download(self: Any, delay_seconds: Any, size_bytes: Any)

Create mock download function with configurable delay and size.

### TestConcurrentPerformance

Test performance under concurrent load.

#### Methods

##### test_concurrent_model_requests(self: Any, performance_orchestrator: Any)

Test performance with multiple concurrent model requests.

##### test_mixed_model_size_performance(self: Any, performance_orchestrator: Any)

Test performance with mixed small and large model requests.

##### test_lock_contention_performance(self: Any, performance_orchestrator: Any)

Test performance under high lock contention.

### TestMemoryPerformance

Test memory usage and performance characteristics.

#### Methods

##### test_large_model_memory_usage(self: Any, performance_orchestrator: Any)

Test memory usage during large model operations.

##### test_memory_cleanup_after_operations(self: Any, performance_orchestrator: Any)

Test that memory is properly cleaned up after operations.

##### test_concurrent_memory_usage(self: Any, performance_orchestrator: Any)

Test memory usage under concurrent operations.

### TestNetworkPerformance

Test network-related performance characteristics.

#### Methods

##### test_download_timeout_handling(self: Any, performance_orchestrator: Any)

Test performance under network timeout conditions.

##### test_retry_performance(self: Any, performance_orchestrator: Any)

Test performance of retry mechanisms.

##### test_bandwidth_limiting_performance(self: Any, performance_orchestrator: Any)

Test performance with bandwidth limiting.

### TestStoragePerformance

Test storage backend performance characteristics.

#### Methods

##### test_disk_io_performance(self: Any, performance_orchestrator: Any, temp_models_root: Any)

Test disk I/O performance during operations.

##### test_atomic_operation_performance(self: Any, performance_orchestrator: Any, temp_models_root: Any)

Test performance of atomic file operations.

##### test_garbage_collection_performance(self: Any, performance_orchestrator: Any, temp_models_root: Any)

Test garbage collection performance.

### TestScalabilityLimits

Test system behavior at scale limits.

#### Methods

##### test_maximum_concurrent_downloads(self: Any, performance_orchestrator: Any)

Test system behavior at maximum concurrency.

##### test_large_manifest_performance(self: Any, temp_models_root: Any)

Test performance with large model manifests.

