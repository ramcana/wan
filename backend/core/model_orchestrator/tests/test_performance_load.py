"""
Performance and load testing suites for Model Orchestrator.

Tests system behavior under various load conditions:
- Concurrent downloads and access patterns
- Large model handling and memory usage
- Network bandwidth and timeout scenarios
- Storage backend performance characteristics
"""

import asyncio
import concurrent.futures
import gc
import psutil
import threading
import time
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import Mock, patch
import tempfile

import pytest

from backend.core.model_orchestrator.model_ensurer import ModelEnsurer
from backend.core.model_orchestrator.model_registry import ModelRegistry
from backend.core.model_orchestrator.model_resolver import ModelResolver
from backend.core.model_orchestrator.lock_manager import LockManager
from backend.core.model_orchestrator.garbage_collector import GarbageCollector
from backend.core.model_orchestrator.storage_backends.hf_store import HFStore
from backend.core.model_orchestrator.metrics import MetricsCollector


class PerformanceTestBase:
    """Base class for performance tests with common utilities."""

    @pytest.fixture
    def performance_manifest(self, temp_models_root):
        """Create manifest with multiple models for performance testing."""
        manifest_content = """
schema_version = 1

[models."small-model@1.0.0"]
description = "Small test model"
version = "1.0.0"
variants = ["fp16"]
default_variant = "fp16"

[[models."small-model@1.0.0".files]]
path = "model.bin"
size = 1048576  # 1MB
sha256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

[models."medium-model@1.0.0"]
description = "Medium test model"
version = "1.0.0"
variants = ["fp16", "bf16"]
default_variant = "fp16"

[[models."medium-model@1.0.0".files]]
path = "model.bin"
size = 1073741824  # 1GB
sha256 = "a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3"

[models."large-model@1.0.0"]
description = "Large test model"
version = "1.0.0"
variants = ["fp16"]
default_variant = "fp16"

[[models."large-model@1.0.0".files]]
path = "model_part_1.bin"
size = 5368709120  # 5GB
sha256 = "b3f0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

[[models."large-model@1.0.0".files]]
path = "model_part_2.bin"
size = 5368709120  # 5GB
sha256 = "c4f0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
"""
        manifest_path = Path(temp_models_root) / "models.toml"
        manifest_path.write_text(manifest_content)
        return str(manifest_path)

    @pytest.fixture
    def temp_models_root(self):
        """Create temporary models root directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def performance_orchestrator(self, temp_models_root, performance_manifest):
        """Set up orchestrator for performance testing."""
        registry = ModelRegistry(performance_manifest)
        resolver = ModelResolver(temp_models_root)
        lock_manager = LockManager(temp_models_root)
        
        # Mock storage backend with configurable delays
        mock_store = Mock(spec=HFStore)
        mock_store.can_handle.return_value = True
        
        ensurer = ModelEnsurer(
            registry=registry,
            resolver=resolver,
            lock_manager=lock_manager,
            storage_backends=[mock_store]
        )
        
        metrics = MetricsCollector()
        
        return {
            'ensurer': ensurer,
            'registry': registry,
            'resolver': resolver,
            'lock_manager': lock_manager,
            'mock_store': mock_store,
            'metrics': metrics
        }

    def measure_execution_time(self, func, *args, **kwargs):
        """Measure execution time of a function."""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        return result, end_time - start_time

    def measure_memory_usage(self, func, *args, **kwargs):
        """Measure memory usage during function execution."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        result = func(*args, **kwargs)
        
        final_memory = process.memory_info().rss
        memory_delta = final_memory - initial_memory
        
        return result, memory_delta

    def create_mock_download(self, delay_seconds=0, size_bytes=1024):
        """Create mock download function with configurable delay and size."""
        def mock_download(source_url, local_dir, file_specs, progress_callback=None):
            if delay_seconds > 0:
                time.sleep(delay_seconds)
            
            # Create mock files
            model_dir = Path(local_dir)
            model_dir.mkdir(parents=True, exist_ok=True)
            
            for file_spec in file_specs:
                file_path = model_dir / file_spec.path
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_bytes(b"x" * min(size_bytes, file_spec.size))
            
            return Mock(success=True, bytes_downloaded=size_bytes)
        
        return mock_download


class TestConcurrentPerformance(PerformanceTestBase):
    """Test performance under concurrent load."""

    def test_concurrent_model_requests(self, performance_orchestrator):
        """Test performance with multiple concurrent model requests."""
        orchestrator = performance_orchestrator
        ensurer = orchestrator['ensurer']
        mock_store = orchestrator['mock_store']
        
        # Configure mock with small delay
        mock_store.download.side_effect = self.create_mock_download(delay_seconds=0.1)
        
        def request_model(model_id):
            start_time = time.perf_counter()
            try:
                path = ensurer.ensure(model_id, variant="fp16")
                end_time = time.perf_counter()
                return {
                    'success': True,
                    'model_id': model_id,
                    'path': path,
                    'duration': end_time - start_time
                }
            except Exception as e:
                end_time = time.perf_counter()
                return {
                    'success': False,
                    'model_id': model_id,
                    'error': str(e),
                    'duration': end_time - start_time
                }
        
        # Test with increasing concurrency levels
        concurrency_levels = [1, 2, 4, 8, 16]
        results = {}
        
        for concurrency in concurrency_levels:
            model_ids = [f"small-model@1.0.0" for _ in range(concurrency)]
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                start_time = time.perf_counter()
                futures = [executor.submit(request_model, model_id) for model_id in model_ids]
                concurrent_results = [future.result() for future in futures]
                total_time = time.perf_counter() - start_time
            
            # Analyze results
            successful_requests = [r for r in concurrent_results if r['success']]
            avg_duration = sum(r['duration'] for r in successful_requests) / len(successful_requests)
            
            results[concurrency] = {
                'total_time': total_time,
                'avg_duration': avg_duration,
                'success_rate': len(successful_requests) / len(concurrent_results),
                'throughput': len(successful_requests) / total_time
            }
        
        # Verify performance characteristics
        assert results[1]['success_rate'] == 1.0  # Single request should always succeed
        
        # Throughput should generally increase with concurrency (up to a point)
        assert results[4]['throughput'] > results[1]['throughput']
        
        # All requests should eventually succeed
        for concurrency, result in results.items():
            assert result['success_rate'] >= 0.9  # Allow for some failures under high load

    def test_mixed_model_size_performance(self, performance_orchestrator):
        """Test performance with mixed small and large model requests."""
        orchestrator = performance_orchestrator
        ensurer = orchestrator['ensurer']
        mock_store = orchestrator['mock_store']
        
        # Configure different delays for different model sizes
        def size_based_download(source_url, local_dir, file_specs, progress_callback=None):
            total_size = sum(spec.size for spec in file_specs)
            # Simulate download time proportional to size
            delay = min(total_size / (100 * 1024 * 1024), 2.0)  # Max 2 seconds
            time.sleep(delay)
            
            model_dir = Path(local_dir)
            model_dir.mkdir(parents=True, exist_ok=True)
            
            for file_spec in file_specs:
                file_path = model_dir / file_spec.path
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_bytes(b"x" * min(1024, file_spec.size))
            
            return Mock(success=True, bytes_downloaded=total_size)
        
        mock_store.download.side_effect = size_based_download
        
        # Mix of small and medium models
        model_requests = [
            "small-model@1.0.0",
            "medium-model@1.0.0",
            "small-model@1.0.0",
            "medium-model@1.0.0"
        ]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            start_time = time.perf_counter()
            futures = [
                executor.submit(ensurer.ensure, model_id, "fp16")
                for model_id in model_requests
            ]
            results = [future.result() for future in futures]
            total_time = time.perf_counter() - start_time
        
        # All requests should succeed
        assert len(results) == len(model_requests)
        
        # Total time should be less than sum of individual times (parallelization benefit)
        expected_sequential_time = 4 * 1.0  # Rough estimate
        assert total_time < expected_sequential_time

    def test_lock_contention_performance(self, performance_orchestrator):
        """Test performance under high lock contention."""
        orchestrator = performance_orchestrator
        lock_manager = orchestrator['lock_manager']
        
        contention_results = []
        
        def acquire_lock_with_timing(lock_id, hold_time=0.1):
            start_time = time.perf_counter()
            try:
                with lock_manager.acquire_model_lock(lock_id, timeout=5.0):
                    acquire_time = time.perf_counter() - start_time
                    time.sleep(hold_time)  # Hold lock briefly
                    return {'success': True, 'acquire_time': acquire_time}
            except Exception as e:
                return {'success': False, 'error': str(e)}
        
        # Test with multiple threads competing for same lock
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(acquire_lock_with_timing, "contested-model", 0.05)
                for _ in range(10)
            ]
            results = [future.result() for future in futures]
        
        successful_acquisitions = [r for r in results if r['success']]
        
        # All should eventually succeed
        assert len(successful_acquisitions) == 10
        
        # Acquisition times should increase due to contention
        acquire_times = [r['acquire_time'] for r in successful_acquisitions]
        assert max(acquire_times) > min(acquire_times)  # Some variation expected


class TestMemoryPerformance(PerformanceTestBase):
    """Test memory usage and performance characteristics."""

    def test_large_model_memory_usage(self, performance_orchestrator):
        """Test memory usage during large model operations."""
        orchestrator = performance_orchestrator
        ensurer = orchestrator['ensurer']
        mock_store = orchestrator['mock_store']
        
        # Mock download that simulates large file handling
        def memory_efficient_download(source_url, local_dir, file_specs, progress_callback=None):
            model_dir = Path(local_dir)
            model_dir.mkdir(parents=True, exist_ok=True)
            
            for file_spec in file_specs:
                file_path = model_dir / file_spec.path
                file_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Simulate streaming write (memory efficient)
                with open(file_path, 'wb') as f:
                    chunk_size = 1024 * 1024  # 1MB chunks
                    remaining = file_spec.size
                    while remaining > 0:
                        chunk = min(chunk_size, remaining)
                        f.write(b"x" * chunk)
                        remaining -= chunk
            
            return Mock(success=True, bytes_downloaded=sum(spec.size for spec in file_specs))
        
        mock_store.download.side_effect = memory_efficient_download
        
        # Measure memory usage during large model download
        def download_large_model():
            return ensurer.ensure("large-model@1.0.0", variant="fp16")
        
        result, memory_delta = self.measure_memory_usage(download_large_model)
        
        # Memory usage should be reasonable (not proportional to model size)
        max_acceptable_memory = 500 * 1024 * 1024  # 500MB
        assert memory_delta < max_acceptable_memory
        
        # Model should be successfully downloaded
        assert Path(result).exists()

    def test_memory_cleanup_after_operations(self, performance_orchestrator):
        """Test that memory is properly cleaned up after operations."""
        orchestrator = performance_orchestrator
        ensurer = orchestrator['ensurer']
        mock_store = orchestrator['mock_store']
        
        mock_store.download.side_effect = self.create_mock_download(size_bytes=1024*1024)
        
        # Measure baseline memory
        gc.collect()
        process = psutil.Process()
        baseline_memory = process.memory_info().rss
        
        # Perform multiple operations
        for i in range(10):
            ensurer.ensure("small-model@1.0.0", variant="fp16")
        
        # Force garbage collection
        gc.collect()
        final_memory = process.memory_info().rss
        
        # Memory should not grow significantly
        memory_growth = final_memory - baseline_memory
        max_acceptable_growth = 100 * 1024 * 1024  # 100MB
        assert memory_growth < max_acceptable_growth

    def test_concurrent_memory_usage(self, performance_orchestrator):
        """Test memory usage under concurrent operations."""
        orchestrator = performance_orchestrator
        ensurer = orchestrator['ensurer']
        mock_store = orchestrator['mock_store']
        
        mock_store.download.side_effect = self.create_mock_download(
            delay_seconds=0.1, 
            size_bytes=10*1024*1024  # 10MB per model
        )
        
        def concurrent_downloads():
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                futures = [
                    executor.submit(ensurer.ensure, "medium-model@1.0.0", "fp16")
                    for _ in range(8)
                ]
                return [future.result() for future in futures]
        
        result, memory_delta = self.measure_memory_usage(concurrent_downloads)
        
        # Memory usage should scale reasonably with concurrency
        max_acceptable_memory = 200 * 1024 * 1024  # 200MB for 8 concurrent operations
        assert memory_delta < max_acceptable_memory


class TestNetworkPerformance(PerformanceTestBase):
    """Test network-related performance characteristics."""

    def test_download_timeout_handling(self, performance_orchestrator):
        """Test performance under network timeout conditions."""
        orchestrator = performance_orchestrator
        ensurer = orchestrator['ensurer']
        mock_store = orchestrator['mock_store']
        
        # Mock download with timeout simulation
        def timeout_download(source_url, local_dir, file_specs, progress_callback=None):
            time.sleep(10)  # Simulate very slow download
            raise TimeoutError("Download timeout")
        
        mock_store.download.side_effect = timeout_download
        
        # Measure timeout handling performance
        start_time = time.perf_counter()
        
        with pytest.raises(Exception):  # Should timeout and raise exception
            ensurer.ensure("small-model@1.0.0", variant="fp16")
        
        elapsed_time = time.perf_counter() - start_time
        
        # Should timeout reasonably quickly (not wait full 10 seconds)
        assert elapsed_time < 8.0  # Allow some overhead

    def test_retry_performance(self, performance_orchestrator):
        """Test performance of retry mechanisms."""
        orchestrator = performance_orchestrator
        ensurer = orchestrator['ensurer']
        mock_store = orchestrator['mock_store']
        
        attempt_count = 0
        
        def failing_then_succeeding_download(source_url, local_dir, file_specs, progress_callback=None):
            nonlocal attempt_count
            attempt_count += 1
            
            if attempt_count < 3:  # Fail first 2 attempts
                raise ConnectionError("Network error")
            
            # Succeed on 3rd attempt
            return self.create_mock_download()(source_url, local_dir, file_specs, progress_callback)
        
        mock_store.download.side_effect = failing_then_succeeding_download
        
        # Measure retry performance
        start_time = time.perf_counter()
        result = ensurer.ensure("small-model@1.0.0", variant="fp16")
        elapsed_time = time.perf_counter() - start_time
        
        # Should succeed after retries
        assert Path(result).exists()
        assert attempt_count == 3
        
        # Retry delays should be reasonable
        assert elapsed_time < 10.0  # Should not take too long with exponential backoff

    def test_bandwidth_limiting_performance(self, performance_orchestrator):
        """Test performance with bandwidth limiting."""
        orchestrator = performance_orchestrator
        ensurer = orchestrator['ensurer']
        mock_store = orchestrator['mock_store']
        
        # Mock download with simulated bandwidth limiting
        def bandwidth_limited_download(source_url, local_dir, file_specs, progress_callback=None):
            # Simulate slower download due to bandwidth limits
            total_size = sum(spec.size for spec in file_specs)
            simulated_bandwidth = 1024 * 1024  # 1MB/s
            download_time = total_size / simulated_bandwidth
            
            time.sleep(min(download_time, 2.0))  # Cap at 2 seconds for testing
            
            model_dir = Path(local_dir)
            model_dir.mkdir(parents=True, exist_ok=True)
            
            for file_spec in file_specs:
                file_path = model_dir / file_spec.path
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_bytes(b"x" * min(1024, file_spec.size))
            
            return Mock(success=True, bytes_downloaded=total_size)
        
        mock_store.can_handle.return_value = True
        mock_store.download.side_effect = bandwidth_limited_download
        
        # Test download under bandwidth constraints
        start_time = time.perf_counter()
        result = ensurer.ensure("medium-model@1.0.0", variant="fp16")
        duration = time.perf_counter() - start_time
        
        # Should complete but take longer due to bandwidth limiting
        assert Path(result).exists()
        assert duration > 1.0  # Should take at least 1 second due to bandwidth limit
        assert duration < 10.0  # But not too long for testing


class TestStoragePerformance(PerformanceTestBase):
    """Test storage backend performance characteristics."""

    def test_disk_io_performance(self, performance_orchestrator, temp_models_root):
        """Test disk I/O performance during operations."""
        orchestrator = performance_orchestrator
        resolver = orchestrator['resolver']
        
        # Test large file creation performance
        large_file_path = Path(temp_models_root) / "large_test_file.bin"
        
        def create_large_file():
            with open(large_file_path, 'wb') as f:
                chunk_size = 1024 * 1024  # 1MB chunks
                for _ in range(100):  # 100MB total
                    f.write(b"x" * chunk_size)
        
        result, duration = self.measure_execution_time(create_large_file)
        
        # Should complete in reasonable time
        assert duration < 30.0  # 30 seconds max for 100MB
        assert large_file_path.exists()
        
        # Cleanup
        large_file_path.unlink()

    def test_atomic_operation_performance(self, performance_orchestrator, temp_models_root):
        """Test performance of atomic file operations."""
        # Test atomic rename performance
        source_file = Path(temp_models_root) / "source.bin"
        target_file = Path(temp_models_root) / "target.bin"
        
        # Create source file
        source_file.write_bytes(b"x" * (10 * 1024 * 1024))  # 10MB
        
        def atomic_rename():
            source_file.rename(target_file)
        
        result, duration = self.measure_execution_time(atomic_rename)
        
        # Atomic rename should be very fast
        assert duration < 1.0  # Should be nearly instantaneous
        assert target_file.exists()
        assert not source_file.exists()

    def test_garbage_collection_performance(self, performance_orchestrator, temp_models_root):
        """Test garbage collection performance."""
        orchestrator = performance_orchestrator
        resolver = orchestrator['resolver']
        
        # Create multiple model directories to clean up
        model_dirs = []
        for i in range(10):
            model_dir = Path(resolver.local_dir(f"test-model-{i}@1.0.0"))
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "model.bin").write_bytes(b"x" * (1024 * 1024))  # 1MB each
            model_dirs.append(model_dir)
        
        gc_instance = GarbageCollector(resolver, max_total_size=5 * 1024 * 1024)  # 5MB limit
        
        def run_garbage_collection():
            return gc_instance.collect(dry_run=False)
        
        result, duration = self.measure_execution_time(run_garbage_collection)
        
        # GC should complete quickly
        assert duration < 5.0  # 5 seconds max
        assert result.bytes_reclaimed > 0


class TestScalabilityLimits(PerformanceTestBase):
    """Test system behavior at scale limits."""

    def test_maximum_concurrent_downloads(self, performance_orchestrator):
        """Test system behavior at maximum concurrency."""
        orchestrator = performance_orchestrator
        ensurer = orchestrator['ensurer']
        mock_store = orchestrator['mock_store']
        
        mock_store.download.side_effect = self.create_mock_download(delay_seconds=0.1)
        
        # Test with very high concurrency
        max_concurrency = 50
        
        def stress_test():
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrency) as executor:
                futures = [
                    executor.submit(ensurer.ensure, "small-model@1.0.0", "fp16")
                    for _ in range(max_concurrency)
                ]
                return [future.result() for future in futures]
        
        start_time = time.perf_counter()
        results = stress_test()
        duration = time.perf_counter() - start_time
        
        # Should handle high concurrency gracefully
        assert len(results) == max_concurrency
        assert duration < 30.0  # Should complete within reasonable time

    def test_large_manifest_performance(self, temp_models_root):
        """Test performance with large model manifests."""
        # Create manifest with many models
        manifest_content = "schema_version = 1\n\n"
        
        for i in range(1000):  # 1000 models
            manifest_content += f"""
[models."model-{i}@1.0.0"]
description = "Test model {i}"
version = "1.0.0"
variants = ["fp16"]
default_variant = "fp16"

[[models."model-{i}@1.0.0".files]]
path = "model.bin"
size = 1048576
sha256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
"""
        
        manifest_path = Path(temp_models_root) / "large_models.toml"
        manifest_path.write_text(manifest_content)
        
        def load_large_manifest():
            return ModelRegistry(str(manifest_path))
        
        result, duration = self.measure_execution_time(load_large_manifest)
        
        # Should load large manifest quickly
        assert duration < 5.0  # 5 seconds max for 1000 models
        assert len(result.list_models()) == 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])