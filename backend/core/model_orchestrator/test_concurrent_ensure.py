"""
Test for concurrent ensure() calls - validates that two concurrent ensure() calls
for the same model result in one downloading while the other waits, and both succeed.
"""

import pytest
import time
import threading
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

from .model_ensurer import ModelEnsurer, ModelStatus
from .model_registry import ModelRegistry, ModelSpec, FileSpec
from .model_resolver import ModelResolver
from .lock_manager import LockManager
from .storage_backends.base_store import StorageBackend, DownloadResult
from .exceptions import ModelOrchestratorError


class MockStorageBackend(StorageBackend):
    """Mock storage backend that simulates slow downloads."""
    
    def __init__(self, download_delay: float = 2.0):
        self.download_delay = download_delay
        self.download_calls = []
        self.files_to_create = {}
        self.call_count = 0
        
    def can_handle(self, source_url: str) -> bool:
        return source_url.startswith("mock://")
    
    def download(self, source_url: str, local_dir: Path, file_specs=None, 
                allow_patterns=None, progress_callback=None) -> DownloadResult:
        """Simulate a slow download with configurable delay."""
        self.call_count += 1
        call_info = {
            'call_number': self.call_count,
            'source_url': source_url,
            'local_dir': str(local_dir),
            'thread_id': threading.get_ident(),
            'start_time': time.time()
        }
        self.download_calls.append(call_info)
        
        # Simulate slow download
        time.sleep(self.download_delay)
        
        # Create the files (handle case where file_specs might be None)
        if file_specs:
            for file_spec in file_specs:
                file_path = local_dir / file_spec.path
                file_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Create file with correct size
                content = b"x" * file_spec.size
                file_path.write_bytes(content)
        
        call_info['end_time'] = time.time()
        call_info['duration'] = call_info['end_time'] - call_info['start_time']
        
        bytes_downloaded = sum(f.size for f in file_specs) if file_specs else 0
        files_downloaded = len(file_specs) if file_specs else 0
        
        return DownloadResult(
            success=True,
            bytes_downloaded=bytes_downloaded,
            files_downloaded=files_downloaded,
            metadata={'backend': 'mock', 'call_info': call_info}
        )
    
    def verify_availability(self, source_url: str) -> bool:
        return True
    
    def estimate_download_size(self, source_url: str, file_specs=None, allow_patterns=None) -> int:
        if file_specs:
            return sum(f.size for f in file_specs)
        return 0


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_registry():
    """Create mock model registry."""
    registry = Mock(spec=ModelRegistry)
    
    # Create a test model spec
    test_spec = ModelSpec(
        model_id="test-model",
        version="1.0.0",
        variants=["fp16", "bf16"],
        default_variant="fp16",
        files=[
            FileSpec(path="config.json", size=1024, sha256="abc123"),
            FileSpec(path="model.bin", size=10485760, sha256="def456")  # 10MB
        ],
        sources=["mock://test.com/test-model"],
        allow_patterns=["*.json", "*.bin"],
        resolution_caps=["720p"],
        optional_components=[],
        lora_required=False
    )
    
    registry.spec.return_value = test_spec
    return registry


@pytest.fixture
def model_resolver(temp_dir):
    """Create model resolver."""
    return ModelResolver(str(temp_dir / "models"))


@pytest.fixture
def lock_manager(temp_dir):
    """Create lock manager."""
    return LockManager(str(temp_dir / "locks"))


@pytest.fixture
def mock_backend():
    """Create mock storage backend."""
    return MockStorageBackend(download_delay=2.0)


@pytest.fixture
def model_ensurer(mock_registry, model_resolver, lock_manager, mock_backend):
    """Create model ensurer with mocked dependencies."""
    return ModelEnsurer(
        registry=mock_registry,
        resolver=model_resolver,
        lock_manager=lock_manager,
        storage_backends=[mock_backend],
        safety_margin_bytes=1024 * 1024  # 1MB safety margin
    )


class TestConcurrentEnsure:
    """Test concurrent ensure() calls for the same model."""
    
    def test_concurrent_ensure_same_model_one_downloads_one_waits(
        self, model_ensurer, mock_backend, temp_dir
    ):
        """
        Test that two concurrent ensure() calls for the same model result in:
        1. One call performing the download
        2. The other call waiting for the first to complete
        3. Both calls succeeding and returning the same path
        """
        model_id = "test-model"
        variant = "fp16"
        
        # Track execution details
        execution_log = []
        results = {}
        exceptions = {}
        
        def ensure_with_logging(thread_name: str):
            """Wrapper function to track ensure() execution."""
            try:
                start_time = time.time()
                execution_log.append(f"{thread_name}: Starting ensure() at {start_time}")
                
                result = model_ensurer.ensure(model_id, variant)
                
                end_time = time.time()
                execution_log.append(f"{thread_name}: Completed ensure() at {end_time}, duration: {end_time - start_time:.2f}s")
                
                results[thread_name] = result
                
            except Exception as e:
                execution_log.append(f"{thread_name}: Failed with exception: {e}")
                exceptions[thread_name] = e
        
        # Start two concurrent ensure() calls
        with ThreadPoolExecutor(max_workers=2) as executor:
            future1 = executor.submit(ensure_with_logging, "Thread-1")
            future2 = executor.submit(ensure_with_logging, "Thread-2")
            
            # Wait for both to complete
            for future in as_completed([future1, future2], timeout=30):
                future.result()  # This will raise any exceptions
        
        # Print execution log for debugging
        print("\nExecution Log:")
        for entry in execution_log:
            print(f"  {entry}")
        
        print(f"\nDownload calls made: {len(mock_backend.download_calls)}")
        for i, call in enumerate(mock_backend.download_calls):
            print(f"  Call {i+1}: Thread {call['thread_id']}, Duration: {call['duration']:.2f}s")
        
        # Verify results
        assert len(exceptions) == 0, f"Exceptions occurred: {exceptions}"
        assert len(results) == 2, f"Expected 2 results, got {len(results)}"
        
        # Both calls should return the same path
        thread1_result = results["Thread-1"]
        thread2_result = results["Thread-2"]
        assert thread1_result == thread2_result, "Both ensure() calls should return the same path"
        
        # Verify the path exists and contains the expected files
        result_path = Path(thread1_result)
        assert result_path.exists(), f"Result path {result_path} should exist"
        assert (result_path / "config.json").exists(), "config.json should exist"
        assert (result_path / "model.bin").exists(), "model.bin should exist"
        
        # Verify file sizes
        assert (result_path / "config.json").stat().st_size == 1024
        assert (result_path / "model.bin").stat().st_size == 10485760
        
        # Critical assertion: Only ONE download should have occurred
        assert len(mock_backend.download_calls) == 1, (
            f"Expected exactly 1 download call, but got {len(mock_backend.download_calls)}. "
            "This means the locking mechanism failed to prevent concurrent downloads."
        )
        
        # Verify that the download was performed by one of the threads
        download_call = mock_backend.download_calls[0]
        download_thread_id = download_call['thread_id']
        print(f"\nDownload performed by thread: {download_thread_id}")
        
        # Both threads should have completed successfully
        assert "Thread-1" in results and "Thread-2" in results
        
        print("\n✅ Test passed: Two concurrent ensure() calls resulted in one download, both succeeded")
    
    def test_concurrent_ensure_different_variants(
        self, model_ensurer, mock_backend, temp_dir
    ):
        """
        Test concurrent ensure() calls for the same model but different variants.
        Each variant should be downloaded separately.
        """
        model_id = "test-model"
        
        results = {}
        exceptions = {}
        
        def ensure_variant(variant: str):
            """Ensure a specific variant."""
            try:
                result = model_ensurer.ensure(model_id, variant)
                results[variant] = result
            except Exception as e:
                exceptions[variant] = e
        
        # Start concurrent ensure() calls for different variants
        with ThreadPoolExecutor(max_workers=2) as executor:
            future1 = executor.submit(ensure_variant, "fp16")
            future2 = executor.submit(ensure_variant, "bf16")
            
            # Wait for both to complete
            for future in as_completed([future1, future2], timeout=30):
                future.result()
        
        # Verify results
        assert len(exceptions) == 0, f"Exceptions occurred: {exceptions}"
        assert len(results) == 2, f"Expected 2 results, got {len(results)}"
        
        # Different variants should have different paths
        fp16_result = results["fp16"]
        bf16_result = results["bf16"]
        assert fp16_result != bf16_result, "Different variants should have different paths"
        
        # Both paths should exist
        assert Path(fp16_result).exists()
        assert Path(bf16_result).exists()
        
        # Should have made 2 download calls (one for each variant)
        assert len(mock_backend.download_calls) == 2, (
            f"Expected 2 download calls for different variants, got {len(mock_backend.download_calls)}"
        )
    
    def test_concurrent_ensure_with_existing_complete_model(
        self, model_ensurer, mock_backend, temp_dir, mock_registry
    ):
        """
        Test concurrent ensure() calls when the model is already complete.
        Neither call should trigger a download.
        """
        model_id = "test-model"
        variant = "fp16"
        
        # Pre-create the model directory and files to simulate existing complete model
        model_path = Path(temp_dir / "models" / model_id)
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Create the expected files
        (model_path / "config.json").write_bytes(b"x" * 1024)
        (model_path / "model.bin").write_bytes(b"x" * 10485760)
        
        # Create verification marker
        verification_data = {
            "model_id": model_id,
            "variant": variant,
            "verified_at": time.time(),
            "files": ["config.json", "model.bin"]
        }
        (model_path / ".verified.json").write_text(json.dumps(verification_data))
        
        results = {}
        exceptions = {}
        
        def ensure_existing_model(thread_name: str):
            """Ensure an existing model."""
            try:
                result = model_ensurer.ensure(model_id, variant)
                results[thread_name] = result
            except Exception as e:
                exceptions[thread_name] = e
        
        # Start concurrent ensure() calls
        with ThreadPoolExecutor(max_workers=2) as executor:
            future1 = executor.submit(ensure_existing_model, "Thread-1")
            future2 = executor.submit(ensure_existing_model, "Thread-2")
            
            # Wait for both to complete
            for future in as_completed([future1, future2], timeout=10):
                future.result()
        
        # Verify results
        assert len(exceptions) == 0, f"Exceptions occurred: {exceptions}"
        assert len(results) == 2, f"Expected 2 results, got {len(results)}"
        
        # Both calls should return the same path
        assert results["Thread-1"] == results["Thread-2"]
        
        # No downloads should have occurred since model was already complete
        assert len(mock_backend.download_calls) == 0, (
            f"Expected 0 download calls for existing model, got {len(mock_backend.download_calls)}"
        )
        
        print("✅ Test passed: Concurrent ensure() calls on existing model avoided downloads")
    
    def test_concurrent_ensure_with_force_redownload(
        self, model_ensurer, mock_backend, temp_dir
    ):
        """
        Test concurrent ensure() calls with force_redownload=True.
        Should still only download once due to locking.
        """
        model_id = "test-model"
        variant = "fp16"
        
        results = {}
        exceptions = {}
        
        def ensure_with_force(thread_name: str):
            """Ensure with force redownload."""
            try:
                result = model_ensurer.ensure(model_id, variant, force_redownload=True)
                results[thread_name] = result
            except Exception as e:
                exceptions[thread_name] = e
        
        # Start concurrent ensure() calls with force redownload
        with ThreadPoolExecutor(max_workers=2) as executor:
            future1 = executor.submit(ensure_with_force, "Thread-1")
            future2 = executor.submit(ensure_with_force, "Thread-2")
            
            # Wait for both to complete
            for future in as_completed([future1, future2], timeout=30):
                future.result()
        
        # Verify results
        assert len(exceptions) == 0, f"Exceptions occurred: {exceptions}"
        assert len(results) == 2, f"Expected 2 results, got {len(results)}"
        
        # Both calls should return the same path
        assert results["Thread-1"] == results["Thread-2"]
        
        # Should have made exactly 1 download call despite force_redownload
        assert len(mock_backend.download_calls) == 1, (
            f"Expected 1 download call even with force_redownload, got {len(mock_backend.download_calls)}"
        )
        
        print("✅ Test passed: Concurrent force redownload still resulted in single download")
    
    def test_sequential_ensure_calls_use_cached_result(
        self, model_ensurer, mock_backend, temp_dir
    ):
        """
        Test that sequential ensure() calls use the cached result from the first call.
        """
        model_id = "test-model"
        variant = "fp16"
        
        # First call should download
        result1 = model_ensurer.ensure(model_id, variant)
        assert len(mock_backend.download_calls) == 1
        
        # Second call should use cached result
        result2 = model_ensurer.ensure(model_id, variant)
        assert len(mock_backend.download_calls) == 1  # Still only 1 download
        
        # Both should return the same path
        assert result1 == result2
        
        # Verify the model is marked as complete
        status = model_ensurer.status(model_id, variant)
        assert status.status == ModelStatus.COMPLETE
        assert status.bytes_needed == 0
        assert len(status.missing_files) == 0
        
        print("✅ Test passed: Sequential calls used cached result")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])