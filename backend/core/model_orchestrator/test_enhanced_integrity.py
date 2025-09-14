"""
Tests for enhanced integrity verification in ModelEnsurer.
"""

import tempfile
import hashlib
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

from .model_ensurer import ModelEnsurer, ModelStatus
from .model_registry import ModelRegistry, ModelSpec, FileSpec
from .model_resolver import ModelResolver
from .lock_manager import LockManager
from .storage_backends.base_store import StorageBackend, DownloadResult
from .integrity_verifier import IntegrityVerificationResult, FileVerificationResult, VerificationMethod
from .exceptions import ChecksumError, SizeMismatchError, IntegrityVerificationError


class MockStorageBackend(StorageBackend):
    """Mock storage backend for testing."""
    
    def __init__(self, should_succeed=True, metadata=None):
        self.should_succeed = should_succeed
        self.metadata = metadata or {}
    
    def can_handle(self, source_url: str) -> bool:
        return source_url.startswith("mock://")
    
    def download(self, source_url, local_dir, file_specs=None, allow_patterns=None, progress_callback=None):
        if self.should_succeed:
            # Create mock files
            for file_spec in file_specs or []:
                file_path = local_dir / file_spec.path
                file_path.parent.mkdir(parents=True, exist_ok=True)
                # Create file with correct size
                file_path.write_bytes(b"x" * file_spec.size)
            
            return DownloadResult(
                success=True,
                bytes_downloaded=sum(f.size for f in file_specs or []),
                files_downloaded=len(file_specs or []),
                metadata=self.metadata
            )
        else:
            return DownloadResult(
                success=False,
                bytes_downloaded=0,
                files_downloaded=0,
                error_message="Mock download failure"
            )
    
    def verify_availability(self, source_url: str) -> bool:
        return True
    
    def estimate_download_size(self, source_url, file_specs=None, allow_patterns=None) -> int:
        return sum(f.size for f in file_specs or [])


class TestEnhancedIntegrityVerification:
    """Test enhanced integrity verification in ModelEnsurer."""
    
    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def mock_registry(self):
        registry = Mock(spec=ModelRegistry)
        
        # Create test file specs with known checksums
        test_content = b"test file content"
        test_sha256 = hashlib.sha256(test_content).hexdigest()
        
        test_spec = ModelSpec(
            model_id="test-model@1.0.0",
            version="1.0.0",
            variants=["fp16"],
            default_variant="fp16",
            files=[
                FileSpec(
                    path="model.bin",
                    size=len(test_content),
                    sha256=test_sha256
                )
            ],
            sources=["mock://test-source"],
            allow_patterns=["*"],
            resolution_caps=[],
            optional_components=[],
            lora_required=False,
            description="Test model"
        )
        
        registry.spec.return_value = test_spec
        return registry
    
    @pytest.fixture
    def mock_resolver(self, temp_dir):
        resolver = Mock(spec=ModelResolver)
        resolver.models_root = str(temp_dir)
        resolver.local_dir.return_value = str(temp_dir / "models" / "test-model")
        return resolver
    
    @pytest.fixture
    def mock_lock_manager(self):
        lock_manager = Mock(spec=LockManager)
        lock_manager.acquire_model_lock.return_value.__enter__ = Mock()
        lock_manager.acquire_model_lock.return_value.__exit__ = Mock()
        return lock_manager
    
    def test_successful_download_with_sha256_verification(self, temp_dir, mock_registry, mock_resolver, mock_lock_manager):
        """Test successful download with SHA256 verification."""
        # Create correct content for the test file
        test_content = b"test file content"
        
        # Mock storage backend that creates files with correct content
        class CorrectContentBackend(MockStorageBackend):
            def download(self, source_url, local_dir, file_specs=None, **kwargs):
                for file_spec in file_specs or []:
                    file_path = local_dir / file_spec.path
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    file_path.write_bytes(test_content)  # Correct content
                
                return DownloadResult(
                    success=True,
                    bytes_downloaded=len(test_content),
                    files_downloaded=1
                )
        
        storage_backend = CorrectContentBackend()
        
        ensurer = ModelEnsurer(
            registry=mock_registry,
            resolver=mock_resolver,
            lock_manager=mock_lock_manager,
            storage_backends=[storage_backend]
        )
        
        result = ensurer.ensure("test-model@1.0.0")
        
        assert result is not None
        mock_registry.spec.assert_called_once()
    
    def test_download_with_checksum_failure_and_retry(self, temp_dir, mock_registry, mock_resolver, mock_lock_manager):
        """Test download with checksum failure that triggers retry."""
        # Mock storage backend that creates files with wrong content initially
        class RetryableBackend(MockStorageBackend):
            def __init__(self):
                super().__init__()
                self.attempt_count = 0
            
            def download(self, source_url, local_dir, file_specs=None, **kwargs):
                self.attempt_count += 1
                
                for file_spec in file_specs or []:
                    file_path = local_dir / file_spec.path
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    if self.attempt_count == 1:
                        # First attempt: wrong content (will fail checksum)
                        file_path.write_bytes(b"wrong content")
                    else:
                        # Second attempt: correct content
                        file_path.write_bytes(b"test file content")
                
                return DownloadResult(
                    success=True,
                    bytes_downloaded=file_spec.size if file_specs else 0,
                    files_downloaded=len(file_specs or [])
                )
        
        storage_backend = RetryableBackend()
        
        ensurer = ModelEnsurer(
            registry=mock_registry,
            resolver=mock_resolver,
            lock_manager=mock_lock_manager,
            storage_backends=[storage_backend]
        )
        
        # Should eventually succeed after retry
        result = ensurer.ensure("test-model@1.0.0")
        assert result is not None
        assert storage_backend.attempt_count >= 2
    
    def test_download_with_hf_etag_fallback(self, temp_dir, mock_registry, mock_resolver, mock_lock_manager):
        """Test download with HuggingFace ETag fallback verification."""
        # Modify the registry to return spec without SHA256
        test_content = b"test file content"
        test_md5 = hashlib.md5(test_content).hexdigest()
        
        spec_without_sha256 = ModelSpec(
            model_id="test-model@1.0.0",
            version="1.0.0",
            variants=["fp16"],
            default_variant="fp16",
            files=[
                FileSpec(
                    path="model.bin",
                    size=len(test_content),
                    sha256=""  # No SHA256 checksum
                )
            ],
            sources=["mock://test-source"],
            allow_patterns=["*"],
            resolution_caps=[],
            optional_components=[],
            lora_required=False,
            description="Test model"
        )
        
        mock_registry.spec.return_value = spec_without_sha256
        
        # Mock storage backend that provides HF metadata
        hf_metadata = {
            "model.bin": {
                "etag": test_md5,
                "size": len(test_content)
            }
        }
        
        class HFMetadataBackend(MockStorageBackend):
            def download(self, source_url, local_dir, file_specs=None, **kwargs):
                for file_spec in file_specs or []:
                    file_path = local_dir / file_spec.path
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    file_path.write_bytes(test_content)
                
                return DownloadResult(
                    success=True,
                    bytes_downloaded=len(test_content),
                    files_downloaded=1,
                    metadata={"hf_metadata": hf_metadata}
                )
        
        storage_backend = HFMetadataBackend()
        
        ensurer = ModelEnsurer(
            registry=mock_registry,
            resolver=mock_resolver,
            lock_manager=mock_lock_manager,
            storage_backends=[storage_backend]
        )
        
        result = ensurer.ensure("test-model@1.0.0")
        assert result is not None
    
    def test_download_with_size_mismatch_failure(self, temp_dir, mock_registry, mock_resolver, mock_lock_manager):
        """Test download failure due to size mismatch."""
        # Mock storage backend that creates files with wrong size
        class WrongSizeBackend(MockStorageBackend):
            def download(self, source_url, local_dir, file_specs=None, **kwargs):
                for file_spec in file_specs or []:
                    file_path = local_dir / file_spec.path
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    # Create file with wrong size
                    file_path.write_bytes(b"wrong size content that is too long")
                
                return DownloadResult(
                    success=True,
                    bytes_downloaded=100,  # Wrong size
                    files_downloaded=1
                )
        
        storage_backend = WrongSizeBackend()
        
        ensurer = ModelEnsurer(
            registry=mock_registry,
            resolver=mock_resolver,
            lock_manager=mock_lock_manager,
            storage_backends=[storage_backend]
        )
        
        with pytest.raises(SizeMismatchError):
            ensurer.ensure("test-model@1.0.0")
    
    def test_verify_integrity_method_with_comprehensive_verification(self, temp_dir, mock_registry, mock_resolver, mock_lock_manager):
        """Test the verify_integrity method uses comprehensive verification."""
        # Create a model with correct files
        test_content = b"test file content"
        model_dir = temp_dir / "test-model"
        model_dir.mkdir(parents=True)
        (model_dir / "model.bin").write_bytes(test_content)
        
        mock_resolver.local_dir.return_value = str(model_dir)
        
        storage_backend = MockStorageBackend()
        
        ensurer = ModelEnsurer(
            registry=mock_registry,
            resolver=mock_resolver,
            lock_manager=mock_lock_manager,
            storage_backends=[storage_backend]
        )
        
        result = ensurer.verify_integrity("test-model@1.0.0")
        
        assert result.success is True
        assert len(result.verified_files) == 1
        assert "model.bin" in result.verified_files
    
    def test_verify_integrity_with_missing_files(self, temp_dir, mock_registry, mock_resolver, mock_lock_manager):
        """Test verify_integrity with missing files."""
        # Create empty model directory
        model_dir = temp_dir / "test-model"
        model_dir.mkdir(parents=True)
        
        mock_resolver.local_dir.return_value = str(model_dir)
        
        storage_backend = MockStorageBackend()
        
        ensurer = ModelEnsurer(
            registry=mock_registry,
            resolver=mock_resolver,
            lock_manager=mock_lock_manager,
            storage_backends=[storage_backend]
        )
        
        result = ensurer.verify_integrity("test-model@1.0.0")
        
        assert result.success is False
        assert len(result.missing_files) == 1
        assert "model.bin" in result.missing_files
    
    def test_verify_integrity_with_corrupted_files(self, temp_dir, mock_registry, mock_resolver, mock_lock_manager):
        """Test verify_integrity with corrupted files."""
        # Create model with corrupted file
        model_dir = temp_dir / "test-model"
        model_dir.mkdir(parents=True)
        (model_dir / "model.bin").write_bytes(b"corrupted content")
        
        mock_resolver.local_dir.return_value = str(model_dir)
        
        storage_backend = MockStorageBackend()
        
        ensurer = ModelEnsurer(
            registry=mock_registry,
            resolver=mock_resolver,
            lock_manager=mock_lock_manager,
            storage_backends=[storage_backend]
        )
        
        result = ensurer.verify_integrity("test-model@1.0.0")
        
        assert result.success is False
        assert len(result.failed_files) == 1
        assert "model.bin" in result.failed_files
    
    def test_status_method_with_comprehensive_verification(self, temp_dir, mock_registry, mock_resolver, mock_lock_manager):
        """Test that status method works with comprehensive verification."""
        # Create a complete model
        test_content = b"test file content"
        model_dir = temp_dir / "test-model"
        model_dir.mkdir(parents=True)
        (model_dir / "model.bin").write_bytes(test_content)
        
        # Create verification marker
        verification_data = {
            "model_id": "test-model@1.0.0",
            "version": "1.0.0",
            "verified_at": 1234567890,
            "files": [
                {
                    "path": "model.bin",
                    "size": len(test_content),
                    "sha256": hashlib.sha256(test_content).hexdigest()
                }
            ]
        }
        
        with open(model_dir / ".verified.json", "w") as f:
            import json
            json.dump(verification_data, f)
        
        mock_resolver.local_dir.return_value = str(model_dir)
        
        storage_backend = MockStorageBackend()
        
        ensurer = ModelEnsurer(
            registry=mock_registry,
            resolver=mock_resolver,
            lock_manager=mock_lock_manager,
            storage_backends=[storage_backend]
        )
        
        status = ensurer.status("test-model@1.0.0")
        
        assert status.status == ModelStatus.COMPLETE
        assert status.bytes_needed == 0
        assert len(status.missing_files or []) == 0
    
    def test_integrity_verification_with_manifest_signature(self, temp_dir, mock_registry, mock_resolver, mock_lock_manager):
        """Test integrity verification with manifest signature validation."""
        # This test would require implementing manifest signature verification
        # For now, we'll test that the signature parameter is passed through
        
        test_content = b"test file content"
        model_dir = temp_dir / "test-model"
        model_dir.mkdir(parents=True)
        (model_dir / "model.bin").write_bytes(test_content)
        
        mock_resolver.local_dir.return_value = str(model_dir)
        
        storage_backend = MockStorageBackend()
        
        ensurer = ModelEnsurer(
            registry=mock_registry,
            resolver=mock_resolver,
            lock_manager=mock_lock_manager,
            storage_backends=[storage_backend]
        )
        
        # Mock the integrity verifier to check that signature parameters are passed
        with patch.object(ensurer.integrity_verifier, 'verify_model_integrity') as mock_verify:
            mock_verify.return_value = IntegrityVerificationResult(
                success=True,
                verified_files=[],
                failed_files=[],
                missing_files=[],
                total_verification_time=0.1,
                manifest_signature_valid=True
            )
            
            # This would be called during download with signature verification
            # For now, we just test the method exists and can be called
            result = ensurer.verify_integrity("test-model@1.0.0")
            
            assert result.success is True
            mock_verify.assert_called_once()
    
    def test_error_recovery_during_integrity_verification(self, temp_dir, mock_registry, mock_resolver, mock_lock_manager):
        """Test error recovery during integrity verification."""
        test_content = b"test file content"
        
        # Mock storage backend that initially fails verification
        class FailingVerificationBackend(MockStorageBackend):
            def __init__(self):
                super().__init__()
                self.attempt_count = 0
            
            def download(self, source_url, local_dir, file_specs=None, **kwargs):
                self.attempt_count += 1
                
                for file_spec in file_specs or []:
                    file_path = local_dir / file_spec.path
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    if self.attempt_count == 1:
                        # First attempt: create file that will fail verification
                        file_path.write_bytes(b"x" * (file_spec.size + 10))  # Wrong size
                    else:
                        # Subsequent attempts: correct content
                        file_path.write_bytes(test_content)
                
                return DownloadResult(
                    success=True,
                    bytes_downloaded=file_spec.size if file_specs else 0,
                    files_downloaded=len(file_specs or [])
                )
        
        storage_backend = FailingVerificationBackend()
        
        ensurer = ModelEnsurer(
            registry=mock_registry,
            resolver=mock_resolver,
            lock_manager=mock_lock_manager,
            storage_backends=[storage_backend]
        )
        
        # Should eventually succeed after retry
        result = ensurer.ensure("test-model@1.0.0")
        assert result is not None
        assert storage_backend.attempt_count >= 2
    
    def test_metrics_recording_for_integrity_failures(self, temp_dir, mock_registry, mock_resolver, mock_lock_manager):
        """Test that integrity failures are properly recorded in metrics."""
        # Create model with corrupted file
        model_dir = temp_dir / "test-model"
        model_dir.mkdir(parents=True)
        (model_dir / "model.bin").write_bytes(b"corrupted content")
        
        mock_resolver.local_dir.return_value = str(model_dir)
        
        storage_backend = MockStorageBackend()
        
        ensurer = ModelEnsurer(
            registry=mock_registry,
            resolver=mock_resolver,
            lock_manager=mock_lock_manager,
            storage_backends=[storage_backend]
        )
        
        # Mock the metrics collector
        with patch.object(ensurer.metrics, 'record_integrity_failure') as mock_record:
            result = ensurer.verify_integrity("test-model@1.0.0")
            
            assert result.success is False
            # Should have recorded the integrity failure
            mock_record.assert_called_once_with("test-model@1.0.0", "model.bin")


class TestIntegrityVerificationEdgeCases:
    """Test edge cases for integrity verification."""
    
    def test_empty_file_verification(self, tmp_path):
        """Test verification of empty files."""
        # This would test zero-byte files, which should be handled correctly
        pass
    
    def test_very_large_file_verification(self, tmp_path):
        """Test verification of very large files."""
        # This would test memory efficiency for large files
        pass
    
    def test_unicode_filename_verification(self, tmp_path):
        """Test verification with Unicode filenames."""
        # This would test international character support
        pass
    
    def test_concurrent_verification_safety(self, tmp_path):
        """Test that concurrent verification operations are safe."""
        # This would test thread safety
        pass