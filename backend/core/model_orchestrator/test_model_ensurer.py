"""
Integration tests for ModelEnsurer - atomic download orchestration with preflight checks.
"""

import os
import json
import shutil
import tempfile
import hashlib
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import List, Optional, Callable

import pytest

from .model_ensurer import ModelEnsurer, ModelStatus, ModelStatusInfo, VerificationResult, FailedSource
from .model_registry import ModelRegistry, ModelSpec, FileSpec
from .model_resolver import ModelResolver
from .lock_manager import LockManager
from .storage_backends.base_store import StorageBackend, DownloadResult
from .exceptions import (
    NoSpaceError, ChecksumError, SizeMismatchError, IncompleteDownloadError,
    ModelOrchestratorError, ErrorCode
)


class MockStorageBackend(StorageBackend):
    """Mock storage backend for testing."""
    
    def __init__(self, url_prefix: str = "mock://", should_fail: bool = False):
        self.url_prefix = url_prefix
        self.should_fail = should_fail
        self.download_calls = []
        self.files_to_create = {}  # path -> (content, size)
    
    def can_handle(self, source_url: str) -> bool:
        return source_url.startswith(self.url_prefix)
    
    def download(
        self,
        source_url: str,
        local_dir: Path,
        file_specs: Optional[List] = None,
        allow_patterns: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> DownloadResult:
        self.download_calls.append({
            'source_url': source_url,
            'local_dir': local_dir,
            'file_specs': file_specs,
            'allow_patterns': allow_patterns
        })
        
        if self.should_fail:
            return DownloadResult(
                success=False,
                bytes_downloaded=0,
                files_downloaded=0,
                error_message="Mock download failure"
            )
        
        # Create mock files
        total_bytes = 0
        files_created = 0
        
        if file_specs:
            for file_spec in file_specs:
                # Only create files that are explicitly specified in files_to_create
                # or if files_to_create is empty (create all files)
                if not self.files_to_create or file_spec.path in self.files_to_create:
                    file_path = local_dir / file_spec.path
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Create file with specified content or generate content
                    if file_spec.path in self.files_to_create:
                        content, expected_size = self.files_to_create[file_spec.path]
                        # Ensure content matches the expected size
                        if len(content) < expected_size:
                            # Pad content to reach expected size
                            content = content + b'\0' * (expected_size - len(content))
                        elif len(content) > expected_size:
                            # Truncate content to expected size
                            content = content[:expected_size]
                    else:
                        # Generate content that matches the expected size and checksum
                        content = self._generate_content_for_spec(file_spec)
                    
                    with open(file_path, 'wb') as f:
                        f.write(content)
                    
                    total_bytes += len(content)
                    files_created += 1
                    
                    # Call progress callback if provided
                    if progress_callback:
                        progress_callback(total_bytes, sum(f.size for f in file_specs if not self.files_to_create or f.path in self.files_to_create))
        
        return DownloadResult(
            success=True,
            bytes_downloaded=total_bytes,
            files_downloaded=files_created,
            duration_seconds=0.1
        )
    
    def verify_availability(self, source_url: str) -> bool:
        return not self.should_fail
    
    def estimate_download_size(
        self,
        source_url: str,
        file_specs: Optional[List] = None,
        allow_patterns: Optional[List[str]] = None
    ) -> int:
        if file_specs:
            return sum(f.size for f in file_specs)
        return 1000  # Default size
    
    def _generate_content_for_spec(self, file_spec: FileSpec) -> bytes:
        """Generate content that matches the file spec."""
        # Always generate content of the expected size
        # For testing, we'll use a simple pattern that fills the required size
        base_content = f"test content for {file_spec.path}".encode()
        
        if len(base_content) < file_spec.size:
            # Pad with repeating pattern to reach expected size
            padding_needed = file_spec.size - len(base_content)
            padding = (b'x' * (padding_needed // 1 + 1))[:padding_needed]
            content = base_content + padding
        else:
            # Truncate to expected size
            content = base_content[:file_spec.size]
        
        return content


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_registry():
    """Create a mock model registry."""
    registry = Mock(spec=ModelRegistry)
    
    # Create test model spec
    # Calculate correct checksums for the test content
    import hashlib
    model_content = b"x" * 1000
    config_content = b"x" * 100
    model_sha256 = hashlib.sha256(model_content).hexdigest()
    config_sha256 = hashlib.sha256(config_content).hexdigest()
    
    test_files = [
        FileSpec(
            path="model.bin",
            size=1000,
            sha256=model_sha256
        ),
        FileSpec(
            path="config.json",
            size=100,
            sha256=config_sha256
        )
    ]
    
    test_spec = ModelSpec(
        model_id="test-model",
        version="1.0.0",
        variants=["fp16", "fp32"],
        default_variant="fp16",
        files=test_files,
        sources=["mock://test-model", "mock2://test-model"],
        allow_patterns=["*.bin", "*.json"],
        resolution_caps=["720p", "1080p"],
        optional_components=[],
        lora_required=False,
        description="Test model for unit tests"
    )
    
    registry.spec.return_value = test_spec
    return registry


@pytest.fixture
def mock_resolver(temp_dir):
    """Create a mock model resolver."""
    resolver = Mock(spec=ModelResolver)
    resolver.models_root = str(temp_dir)
    resolver.local_dir.return_value = str(temp_dir / "models" / "test-model")
    return resolver


@pytest.fixture
def mock_lock_manager():
    """Create a mock lock manager."""
    lock_manager = Mock(spec=LockManager)
    
    # Create a proper context manager mock
    context_manager = Mock()
    context_manager.__enter__ = Mock(return_value=None)
    context_manager.__exit__ = Mock(return_value=None)
    
    lock_manager.acquire_model_lock.return_value = context_manager
    return lock_manager


@pytest.fixture
def mock_backend():
    """Create a mock storage backend."""
    return MockStorageBackend()


@pytest.fixture
def model_ensurer(mock_registry, mock_resolver, mock_lock_manager, mock_backend):
    """Create a ModelEnsurer instance with mocked dependencies."""
    return ModelEnsurer(
        registry=mock_registry,
        resolver=mock_resolver,
        lock_manager=mock_lock_manager,
        storage_backends=[mock_backend],
        safety_margin_bytes=100  # Small margin for testing
    )


class TestModelEnsurer:
    """Test cases for ModelEnsurer."""
    
    def test_ensure_new_model_success(self, model_ensurer, mock_backend, temp_dir):
        """Test successful download of a new model."""
        # Setup: Configure backend to create files with correct content
        mock_backend.files_to_create = {
            "model.bin": (b"x" * 1000, 1000),  # Content with correct size
            "config.json": (b"x" * 100, 100)   # Content with correct size
        }
        
        # Execute
        result_path = model_ensurer.ensure("test-model", "fp16")
        
        # Verify
        assert result_path == str(temp_dir / "models" / "test-model")
        assert len(mock_backend.download_calls) == 1
        
        # Check that files were created
        model_dir = Path(result_path)
        assert (model_dir / "model.bin").exists()
        assert (model_dir / "config.json").exists()
        
        # Check verification marker
        verification_file = model_dir / ".verified.json"
        assert verification_file.exists()
        
        with open(verification_file) as f:
            verification_data = json.load(f)
        assert verification_data["model_id"] == "test-model"
        assert len(verification_data["files"]) == 2
    
    def test_ensure_existing_complete_model(self, model_ensurer, mock_backend, temp_dir):
        """Test that existing complete model is not re-downloaded."""
        # Setup: Create existing model directory with verification marker
        model_dir = temp_dir / "models" / "test-model"
        model_dir.mkdir(parents=True)
        
        # Create model files
        (model_dir / "model.bin").write_bytes(b"x" * 1000)
        (model_dir / "config.json").write_bytes(b"x" * 100)
        
        # Create verification marker
        verification_data = {
            "model_id": "test-model",
            "version": "1.0.0",
            "verified_at": time.time(),
            "files": [
                {"path": "model.bin", "size": 1000, "sha256": "test"},
                {"path": "config.json", "size": 100, "sha256": "test"}
            ]
        }
        with open(model_dir / ".verified.json", 'w') as f:
            json.dump(verification_data, f)
        
        # Execute
        result_path = model_ensurer.ensure("test-model", "fp16")
        
        # Verify
        assert result_path == str(model_dir)
        assert len(mock_backend.download_calls) == 0  # No download should occur
    
    def test_ensure_force_redownload(self, model_ensurer, mock_backend, temp_dir):
        """Test force redownload of existing model."""
        # Setup: Create existing model directory
        model_dir = temp_dir / "models" / "test-model"
        model_dir.mkdir(parents=True)
        (model_dir / "model.bin").write_bytes(b"old content")
        
        # Configure backend
        mock_backend.files_to_create = {
            "model.bin": (b"x" * 1000, 1000),
            "config.json": (b"x" * 100, 100)
        }
        
        # Execute
        result_path = model_ensurer.ensure("test-model", "fp16", force_redownload=True)
        
        # Verify
        assert result_path == str(model_dir)
        assert len(mock_backend.download_calls) == 1  # Download should occur
    
    def test_ensure_disk_space_check_failure(self, model_ensurer, mock_resolver, temp_dir):
        """Test that insufficient disk space raises NoSpaceError."""
        # Setup: Mock disk usage to simulate insufficient space
        with patch('shutil.disk_usage') as mock_disk_usage:
            mock_disk_usage.return_value = Mock(free=500)  # Less than needed (1100 + 100 margin)
            
            # Execute & Verify
            with pytest.raises(NoSpaceError) as exc_info:
                model_ensurer.ensure("test-model", "fp16")
            
            assert exc_info.value.error_code == ErrorCode.NO_SPACE
            assert exc_info.value.details["bytes_needed"] == 1200  # 1100 + 100 margin
            assert exc_info.value.details["bytes_available"] == 500
    
    def test_ensure_disk_space_with_garbage_collection(self, model_ensurer, mock_resolver, temp_dir):
        """Test that garbage collection is triggered when disk space is insufficient."""
        from unittest.mock import Mock
        from collections import namedtuple
        
        # Setup: Mock garbage collector
        mock_gc = Mock()
        mock_gc_result = Mock()
        mock_gc_result.bytes_reclaimed = 1000  # Enough to make space
        mock_gc_result.models_removed = ["old-model"]  # Mock list for len()
        mock_gc.collect.return_value = mock_gc_result
        
        model_ensurer.set_garbage_collector(mock_gc)
        
        # Setup: Mock disk usage - first insufficient, then sufficient after GC
        DiskUsageTuple = namedtuple('usage', ['total', 'used', 'free'])
        
        with patch('shutil.disk_usage') as mock_disk_usage:
            # Always return sufficient space after GC
            mock_disk_usage.side_effect = [
                DiskUsageTuple(total=10000, used=9500, free=500),   # Insufficient initially
                DiskUsageTuple(total=10000, used=8000, free=2000),  # Sufficient after GC
            ] + [DiskUsageTuple(total=10000, used=8000, free=2000)] * 20  # Many additional calls
            
            # Test just the preflight checks directly to avoid download complexity
            spec = model_ensurer.registry.spec("test-model", "fp16")
            
            # This should trigger GC and then succeed
            model_ensurer._preflight_checks(spec, mock_gc)
            
            # Verify: GC was called
            mock_gc.collect.assert_called_once()
    
    def test_ensure_disk_space_gc_insufficient_reclaim(self, model_ensurer, mock_resolver, temp_dir):
        """Test that NoSpaceError is still raised if GC doesn't free enough space."""
        from unittest.mock import Mock
        from collections import namedtuple
        
        # Setup: Mock garbage collector that doesn't reclaim enough
        mock_gc = Mock()
        mock_gc_result = Mock()
        mock_gc_result.bytes_reclaimed = 100  # Not enough
        mock_gc_result.models_removed = []  # Mock list for len()
        mock_gc.collect.return_value = mock_gc_result
        
        model_ensurer.set_garbage_collector(mock_gc)
        
        # Setup: Mock disk usage - insufficient before and after GC
        DiskUsageTuple = namedtuple('usage', ['total', 'used', 'free'])
        
        with patch('shutil.disk_usage') as mock_disk_usage:
            mock_disk_usage.side_effect = [
                DiskUsageTuple(total=10000, used=9500, free=500),   # Insufficient initially
                DiskUsageTuple(total=10000, used=9400, free=600),   # Still insufficient after GC
            ] + [DiskUsageTuple(total=10000, used=9400, free=600)] * 10  # Additional calls
            
            # Test just the preflight checks directly
            spec = model_ensurer.registry.spec("test-model", "fp16")
            
            # Execute & Verify: Should still raise NoSpaceError
            with pytest.raises(NoSpaceError):
                model_ensurer._preflight_checks(spec, mock_gc)
            
            # Verify: GC was attempted
            mock_gc.collect.assert_called_once()
    
    def test_ensure_download_failure_fallback(self, model_ensurer, mock_registry, temp_dir):
        """Test fallback to secondary source when primary fails."""
        # Setup: Create backends where first fails, second succeeds
        failing_backend = MockStorageBackend("mock://", should_fail=True)
        success_backend = MockStorageBackend("mock2://", should_fail=False)
        success_backend.files_to_create = {
            "model.bin": (b"x" * 1000, 1000),
            "config.json": (b"x" * 100, 100)
        }
        
        model_ensurer.storage_backends = [failing_backend, success_backend]
        
        # Execute
        result_path = model_ensurer.ensure("test-model", "fp16")
        
        # Verify
        assert result_path == str(temp_dir / "models" / "test-model")
        assert len(failing_backend.download_calls) == 1
        assert len(success_backend.download_calls) == 1
    
    def test_ensure_all_sources_fail(self, model_ensurer):
        """Test that failure of all sources raises appropriate error."""
        # Setup: Make backend fail
        model_ensurer.storage_backends[0].should_fail = True
        
        # Execute & Verify
        with pytest.raises(ModelOrchestratorError) as exc_info:
            model_ensurer.ensure("test-model", "fp16")
        
        assert exc_info.value.error_code == ErrorCode.SOURCE_UNAVAILABLE
    
    def test_ensure_checksum_verification_failure(self, model_ensurer, mock_backend, temp_dir):
        """Test that checksum verification failure raises ChecksumError."""
        # Setup: Configure backend to create files with wrong content
        mock_backend.files_to_create = {
            "model.bin": (b"wrong content", 1000),  # Wrong content for checksum
            "config.json": (b"x" * 100, 100)
        }
        
        # Execute & Verify
        with pytest.raises(ChecksumError) as exc_info:
            model_ensurer.ensure("test-model", "fp16")
        
        assert exc_info.value.error_code == ErrorCode.CHECKSUM_FAIL
        assert "model.bin" in exc_info.value.details["file_path"]
    
    def test_ensure_size_mismatch_failure(self, model_ensurer, mock_backend, temp_dir):
        """Test that size mismatch raises SizeMismatchError."""
        # Setup: Configure backend to create files with wrong size
        mock_backend.files_to_create = {
            "model.bin": (b"hello", 500),  # Wrong size
            "config.json": (b"x" * 100, 100)
        }
        
        # Execute & Verify
        with pytest.raises(SizeMismatchError) as exc_info:
            model_ensurer.ensure("test-model", "fp16")
        
        assert exc_info.value.error_code == ErrorCode.SIZE_MISMATCH
        assert exc_info.value.details["expected"] == 1000
        assert exc_info.value.details["actual"] == 500
    
    def test_ensure_missing_file_failure(self, model_ensurer, mock_backend, temp_dir):
        """Test that missing files after download raise IncompleteDownloadError."""
        # Setup: Configure backend to create only one file
        mock_backend.files_to_create = {
            "model.bin": (b"x" * 1000, 1000)
            # Missing config.json
        }
        
        # Execute & Verify
        with pytest.raises(IncompleteDownloadError) as exc_info:
            model_ensurer.ensure("test-model", "fp16")
        
        assert exc_info.value.error_code == ErrorCode.INCOMPLETE_DOWNLOAD
        assert "config.json" in exc_info.value.details["missing_files"]
    
    def test_status_not_present(self, model_ensurer):
        """Test status for non-existent model."""
        status = model_ensurer.status("test-model", "fp16")
        
        assert status.status == ModelStatus.NOT_PRESENT
        assert len(status.missing_files) == 2
        assert status.bytes_needed == 1100  # 1000 + 100
    
    def test_status_complete_with_verification_marker(self, model_ensurer, temp_dir):
        """Test status for complete model with verification marker."""
        # Setup: Create complete model with verification marker
        model_dir = temp_dir / "models" / "test-model"
        model_dir.mkdir(parents=True)
        
        (model_dir / "model.bin").write_bytes(b"x" * 1000)
        (model_dir / "config.json").write_bytes(b"x" * 100)
        
        verification_data = {
            "model_id": "test-model",
            "version": "1.0.0",
            "verified_at": time.time(),
            "files": []
        }
        with open(model_dir / ".verified.json", 'w') as f:
            json.dump(verification_data, f)
        
        # Execute
        status = model_ensurer.status("test-model", "fp16")
        
        # Verify
        assert status.status == ModelStatus.COMPLETE
        assert len(status.missing_files) == 0
        assert status.bytes_needed == 0
    
    def test_status_partial(self, model_ensurer, temp_dir):
        """Test status for partially downloaded model."""
        # Setup: Create model directory with only one file
        model_dir = temp_dir / "models" / "test-model"
        model_dir.mkdir(parents=True)
        (model_dir / "model.bin").write_bytes(b"x" * 1000)
        # Missing config.json
        
        # Execute
        status = model_ensurer.status("test-model", "fp16")
        
        # Verify
        assert status.status == ModelStatus.PARTIAL
        assert "config.json" in status.missing_files
        assert status.bytes_needed == 100
    
    def test_status_corrupt(self, model_ensurer, temp_dir):
        """Test status for model with corrupt files."""
        # Setup: Create model directory with wrong file sizes
        model_dir = temp_dir / "models" / "test-model"
        model_dir.mkdir(parents=True)
        (model_dir / "model.bin").write_bytes(b"x" * 500)  # Wrong size
        (model_dir / "config.json").write_bytes(b"x" * 100)
        
        # Execute
        status = model_ensurer.status("test-model", "fp16")
        
        # Verify
        assert status.status == ModelStatus.CORRUPT
        assert "model.bin" in status.missing_files
        assert status.bytes_needed == 1000
    
    def test_verify_integrity_success(self, model_ensurer, temp_dir):
        """Test successful integrity verification."""
        # Setup: Create model with correct files
        model_dir = temp_dir / "models" / "test-model"
        model_dir.mkdir(parents=True)
        
        # Create files with content that matches expected checksums
        with open(model_dir / "model.bin", 'wb') as f:
            content = b"x" * 1000  # Content that matches expected checksum
            f.write(content)
        
        with open(model_dir / "config.json", 'wb') as f:
            content = b"x" * 100  # Content that matches expected checksum
            f.write(content)
        
        # Execute
        result = model_ensurer.verify_integrity("test-model", "fp16")
        
        # Verify
        assert result.success
        assert len(result.verified_files) == 2
        assert len(result.failed_files) == 0
        assert len(result.missing_files) == 0
    
    def test_verify_integrity_missing_files(self, model_ensurer, temp_dir):
        """Test integrity verification with missing files."""
        # Setup: Create model directory but no files
        model_dir = temp_dir / "models" / "test-model"
        model_dir.mkdir(parents=True)
        
        # Execute
        result = model_ensurer.verify_integrity("test-model", "fp16")
        
        # Verify
        assert not result.success
        assert len(result.verified_files) == 0
        assert len(result.failed_files) == 0
        assert len(result.missing_files) == 2
    
    def test_verify_integrity_checksum_mismatch(self, model_ensurer, temp_dir):
        """Test integrity verification with checksum mismatch."""
        # Setup: Create model with files that have wrong checksums
        model_dir = temp_dir / "models" / "test-model"
        model_dir.mkdir(parents=True)
        
        (model_dir / "model.bin").write_bytes(b"wrong content" + b"\0" * 987)
        (model_dir / "config.json").write_bytes(b"wrong content" + b"\0" * 87)
        
        # Execute
        result = model_ensurer.verify_integrity("test-model", "fp16")
        
        # Verify
        assert not result.success
        assert len(result.verified_files) == 0
        assert len(result.failed_files) == 2
        assert len(result.missing_files) == 0
    
    def test_estimate_download_size(self, model_ensurer):
        """Test download size estimation."""
        size = model_ensurer.estimate_download_size("test-model", "fp16")
        assert size == 1100  # 1000 + 100
    
    def test_concurrent_downloads(self, model_ensurer, mock_backend, mock_lock_manager, temp_dir):
        """Test that concurrent downloads are properly synchronized."""
        # Setup: Mock lock manager to track lock acquisition
        lock_acquired = []
        
        def mock_acquire_lock(model_id, timeout):
            lock_acquired.append(model_id)
            return MagicMock()
        
        mock_lock_manager.acquire_model_lock.side_effect = mock_acquire_lock
        
        # Configure backend
        mock_backend.files_to_create = {
            "model.bin": (b"x" * 1000, 1000),
            "config.json": (b"x" * 100, 100)
        }
        
        # Execute
        result_path = model_ensurer.ensure("test-model", "fp16")
        
        # Verify
        assert result_path == str(temp_dir / "models" / "test-model")
        assert "test-model" in lock_acquired
        mock_lock_manager.acquire_model_lock.assert_called_once_with("test-model", timeout=300.0)
    
    def test_progress_callback(self, model_ensurer, mock_backend, temp_dir):
        """Test that progress callback is called during download."""
        progress_calls = []
        
        def progress_callback(bytes_downloaded, total_bytes):
            progress_calls.append((bytes_downloaded, total_bytes))
        
        # Configure backend
        mock_backend.files_to_create = {
            "model.bin": (b"x" * 1000, 1000),
            "config.json": (b"x" * 100, 100)
        }
        
        # Execute
        model_ensurer.ensure("test-model", "fp16", progress_callback=progress_callback)
        
        # Verify that progress callback was called
        assert len(progress_calls) > 0
        # Last call should have total bytes
        assert progress_calls[-1][1] == 1100
    
    def test_temp_directory_cleanup(self, model_ensurer, mock_backend, temp_dir):
        """Test that temporary directories are cleaned up after download."""
        # Configure backend
        mock_backend.files_to_create = {
            "model.bin": (b"x" * 1000, 1000),
            "config.json": (b"x" * 100, 100)
        }
        
        # Execute
        model_ensurer.ensure("test-model", "fp16")
        
        # Verify that no .partial directories remain
        tmp_dir = temp_dir / ".tmp"
        if tmp_dir.exists():
            partial_dirs = [d for d in tmp_dir.iterdir() if d.name.endswith('.partial')]
            assert len(partial_dirs) == 0
    
    def test_temp_directory_cleanup_on_failure(self, model_ensurer, mock_backend, temp_dir):
        """Test that temporary directories are cleaned up even on failure."""
        # Setup: Make backend fail after creating temp directory
        mock_backend.should_fail = True
        
        # Execute & Verify
        with pytest.raises(ModelOrchestratorError):
            model_ensurer.ensure("test-model", "fp16")
        
        # Verify that no .partial directories remain
        tmp_dir = temp_dir / ".tmp"
        if tmp_dir.exists():
            partial_dirs = [d for d in tmp_dir.iterdir() if d.name.endswith('.partial')]
            assert len(partial_dirs) == 0
    
    def test_negative_cache_skips_recently_failed_sources(self, model_ensurer, mock_registry, temp_dir):
        """Test that negative cache skips recently failed sources."""
        # Setup: Create backends where first fails, second succeeds
        failing_backend = MockStorageBackend("mock://", should_fail=True)
        success_backend = MockStorageBackend("mock2://", should_fail=False)
        success_backend.files_to_create = {
            "model.bin": (b"x" * 1000, 1000),
            "config.json": (b"x" * 100, 100)
        }
        
        model_ensurer.storage_backends = [failing_backend, success_backend]
        
        # First attempt - should try both backends
        result_path = model_ensurer.ensure("test-model", "fp16")
        assert result_path == str(temp_dir / "models" / "test-model")
        assert len(failing_backend.download_calls) == 1
        assert len(success_backend.download_calls) == 1
        
        # Reset call counts
        failing_backend.download_calls = []
        success_backend.download_calls = []
        
        # Second attempt with force_redownload - should skip failed source
        result_path = model_ensurer.ensure("test-model", "fp16", force_redownload=True)
        assert result_path == str(temp_dir / "models" / "test-model")
        assert len(failing_backend.download_calls) == 0  # Should be skipped
        assert len(success_backend.download_calls) == 1
    
    def test_negative_cache_expires_after_ttl(self, model_ensurer, mock_registry, temp_dir):
        """Test that negative cache entries expire after TTL."""
        # Setup: Short TTL for testing
        model_ensurer.negative_cache_ttl = 0.1  # 100ms
        
        failing_backend = MockStorageBackend("mock://", should_fail=True)
        success_backend = MockStorageBackend("mock2://", should_fail=False)
        success_backend.files_to_create = {
            "model.bin": (b"x" * 1000, 1000),
            "config.json": (b"x" * 100, 100)
        }
        
        model_ensurer.storage_backends = [failing_backend, success_backend]
        
        # First attempt - should try both backends
        result_path = model_ensurer.ensure("test-model", "fp16")
        assert len(failing_backend.download_calls) == 1
        
        # Reset call counts
        failing_backend.download_calls = []
        success_backend.download_calls = []
        
        # Wait for cache to expire
        time.sleep(0.2)
        
        # Second attempt - should try failed source again after TTL expiry
        result_path = model_ensurer.ensure("test-model", "fp16", force_redownload=True)
        assert len(failing_backend.download_calls) == 1  # Should be tried again
        assert len(success_backend.download_calls) == 1
    
    def test_negative_cache_clears_on_success(self, model_ensurer, mock_registry, temp_dir):
        """Test that negative cache is cleared when a source succeeds."""
        # Setup: Backend that fails first time, succeeds second time
        flaky_backend = MockStorageBackend("mock://", should_fail=True)
        model_ensurer.storage_backends = [flaky_backend]
        
        # First attempt - should fail and be cached
        with pytest.raises(ModelOrchestratorError):
            model_ensurer.ensure("test-model", "fp16")
        
        # Verify source is in negative cache
        cache_key = ("test-model", "mock://test-model")
        assert cache_key in model_ensurer._failed_sources
        
        # Make backend succeed
        flaky_backend.should_fail = False
        flaky_backend.files_to_create = {
            "model.bin": (b"x" * 1000, 1000),
            "config.json": (b"x" * 100, 100)
        }
        
        # Wait for cache to expire or manually clear it for this test
        model_ensurer.negative_cache_ttl = 0.01  # Very short TTL
        time.sleep(0.02)  # Wait for expiry
        
        # Second attempt - should succeed and clear cache
        result_path = model_ensurer.ensure("test-model", "fp16", force_redownload=True)
        assert result_path == str(temp_dir / "models" / "test-model")
        
        # Verify source is no longer in negative cache
        assert cache_key not in model_ensurer._failed_sources
    
    def test_negative_cache_cleanup_expired_entries(self, model_ensurer):
        """Test that expired entries are cleaned up from negative cache."""
        # Setup: Add expired entry manually
        model_ensurer.negative_cache_ttl = 0.1  # 100ms
        cache_key = ("test-model", "mock://test-model")
        model_ensurer._failed_sources[cache_key] = FailedSource(
            url="mock://test-model",
            failed_at=time.time() - 0.2,  # Expired
            error_message="Test error",
            retry_count=1
        )
        
        # Add non-expired entry
        cache_key2 = ("test-model2", "mock://test-model2")
        model_ensurer._failed_sources[cache_key2] = FailedSource(
            url="mock://test-model2",
            failed_at=time.time(),  # Not expired
            error_message="Test error",
            retry_count=1
        )
        
        # Execute cleanup
        model_ensurer._cleanup_expired_failures()
        
        # Verify expired entry is removed, non-expired remains
        assert cache_key not in model_ensurer._failed_sources
        assert cache_key2 in model_ensurer._failed_sources
    
    def test_status_verifying_state(self, model_ensurer, temp_dir):
        """Test status returns VERIFYING when verification is in progress."""
        # Note: This is a conceptual test - in practice, VERIFYING state
        # would be set during actual verification process
        # For now, we test the other states as VERIFYING is transient
        pass
    
    def test_verify_integrity_model_not_exist(self, model_ensurer):
        """Test verify_integrity when model directory doesn't exist."""
        result = model_ensurer.verify_integrity("test-model", "fp16")
        
        assert not result.success
        assert len(result.verified_files) == 0
        assert len(result.failed_files) == 0
        assert len(result.missing_files) == 2
        assert "does not exist" in result.error_message
    
    def test_verify_integrity_size_mismatch(self, model_ensurer, temp_dir):
        """Test verify_integrity with size mismatch."""
        # Setup: Create model with wrong file sizes
        model_dir = temp_dir / "models" / "test-model"
        model_dir.mkdir(parents=True)
        
        (model_dir / "model.bin").write_bytes(b"x" * 500)  # Wrong size
        (model_dir / "config.json").write_bytes(b"x" * 100)  # Correct size
        
        # Execute
        result = model_ensurer.verify_integrity("test-model", "fp16")
        
        # Verify
        assert not result.success
        assert "config.json" in result.verified_files  # Correct file
        assert "model.bin" in result.failed_files  # Wrong size
        assert len(result.missing_files) == 0
    
    def test_api_error_codes_structure(self, model_ensurer):
        """Test that API methods return structured error codes."""
        # Test ensure with invalid model
        model_ensurer.registry.spec.side_effect = Exception("Model not found")
        
        with pytest.raises(Exception) as exc_info:
            model_ensurer.ensure("invalid-model")
        
        # Verify exception propagation (specific error handling depends on registry implementation)
        assert "Model not found" in str(exc_info.value)
    
    def test_ensure_with_variant_parameter(self, model_ensurer, mock_backend, temp_dir):
        """Test ensure method with explicit variant parameter."""
        # Configure backend
        mock_backend.files_to_create = {
            "model.bin": (b"x" * 1000, 1000),
            "config.json": (b"x" * 100, 100)
        }
        
        # Execute with explicit variant
        result_path = model_ensurer.ensure("test-model", variant="fp32")
        
        # Verify registry was called with correct variant
        model_ensurer.registry.spec.assert_called_with("test-model", "fp32")
        assert result_path == str(temp_dir / "models" / "test-model")
    
    def test_status_with_variant_parameter(self, model_ensurer):
        """Test status method with explicit variant parameter."""
        # Execute with explicit variant
        status = model_ensurer.status("test-model", variant="fp32")
        
        # Verify registry was called with correct variant
        model_ensurer.registry.spec.assert_called_with("test-model", "fp32")
        assert status.status == ModelStatus.NOT_PRESENT
    
    def test_verify_integrity_with_variant_parameter(self, model_ensurer):
        """Test verify_integrity method with explicit variant parameter."""
        # Execute with explicit variant
        result = model_ensurer.verify_integrity("test-model", variant="fp32")
        
        # Verify registry was called with correct variant
        model_ensurer.registry.spec.assert_called_with("test-model", "fp32")
        assert not result.success  # Model doesn't exist


if __name__ == "__main__":
    pytest.main([__file__])