"""
Tests for comprehensive integrity verification system.
"""

import os
import json
import hashlib
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

from .integrity_verifier import (
    IntegrityVerifier, VerificationMethod, FileVerificationResult,
    IntegrityVerificationResult, HFFileMetadata
)
from .model_registry import ModelSpec, FileSpec
from .exceptions import ChecksumError, SizeMismatchError
from .error_recovery import RetryConfig


class TestIntegrityVerifier:
    """Test suite for IntegrityVerifier."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def sample_files(self, temp_dir):
        """Create sample files with known content and checksums."""
        files = {}
        
        # Create test files with known content
        file1_content = b"This is test file 1 content"
        file1_path = temp_dir / "file1.txt"
        file1_path.write_bytes(file1_content)
        file1_sha256 = hashlib.sha256(file1_content).hexdigest()
        file1_md5 = hashlib.md5(file1_content).hexdigest()
        
        file2_content = b"This is test file 2 with different content"
        file2_path = temp_dir / "subdir" / "file2.bin"
        file2_path.parent.mkdir(parents=True)
        file2_path.write_bytes(file2_content)
        file2_sha256 = hashlib.sha256(file2_content).hexdigest()
        file2_md5 = hashlib.md5(file2_content).hexdigest()
        
        files["file1"] = {
            "path": file1_path,
            "content": file1_content,
            "sha256": file1_sha256,
            "md5": file1_md5,
            "size": len(file1_content)
        }
        
        files["file2"] = {
            "path": file2_path,
            "content": file2_content,
            "sha256": file2_sha256,
            "md5": file2_md5,
            "size": len(file2_content)
        }
        
        return files
    
    @pytest.fixture
    def sample_model_spec(self, sample_files):
        """Create a sample model spec for testing."""
        return ModelSpec(
            model_id="test-model@1.0.0",
            version="1.0.0",
            variants=["fp16"],
            default_variant="fp16",
            files=[
                FileSpec(
                    path="file1.txt",
                    size=sample_files["file1"]["size"],
                    sha256=sample_files["file1"]["sha256"]
                ),
                FileSpec(
                    path="subdir/file2.bin",
                    size=sample_files["file2"]["size"],
                    sha256=sample_files["file2"]["sha256"]
                )
            ],
            sources=["test://source"],
            allow_patterns=["*"],
            resolution_caps=[],
            optional_components=[],
            lora_required=False,
            description="Test model"
        )
    
    @pytest.fixture
    def verifier(self):
        """Create an IntegrityVerifier instance."""
        return IntegrityVerifier(RetryConfig(max_attempts=2, base_delay=0.1))
    
    def test_sha256_verification_success(self, verifier, temp_dir, sample_files, sample_model_spec):
        """Test successful SHA256 checksum verification."""
        result = verifier.verify_model_integrity(sample_model_spec, temp_dir)
        
        assert result.success is True
        assert len(result.verified_files) == 2
        assert len(result.failed_files) == 0
        assert len(result.missing_files) == 0
        assert result.manifest_signature_valid is True
        
        # Check that SHA256 method was used
        for verified_file in result.verified_files:
            assert verified_file.method_used == VerificationMethod.SHA256_CHECKSUM
            assert verified_file.success is True
    
    def test_sha256_verification_checksum_mismatch(self, verifier, temp_dir, sample_files, sample_model_spec):
        """Test SHA256 verification with checksum mismatch."""
        # Modify the expected checksum in the spec
        sample_model_spec.files[0] = FileSpec(
            path="file1.txt",
            size=sample_files["file1"]["size"],
            sha256="invalid_checksum_that_will_not_match_the_actual_file_content"
        )
        
        result = verifier.verify_model_integrity(sample_model_spec, temp_dir)
        
        assert result.success is False
        assert len(result.verified_files) == 1  # Only file2 should pass
        assert len(result.failed_files) == 1    # file1 should fail
        assert len(result.missing_files) == 0
        
        failed_file = result.failed_files[0]
        assert failed_file.file_path == "file1.txt"
        assert failed_file.method_used == VerificationMethod.SHA256_CHECKSUM
        assert "SHA256 mismatch" in failed_file.error_message
    
    def test_size_mismatch_verification(self, verifier, temp_dir, sample_files, sample_model_spec):
        """Test verification with size mismatch."""
        # Modify the expected size in the spec
        sample_model_spec.files[0] = FileSpec(
            path="file1.txt",
            size=999999,  # Wrong size
            sha256=sample_files["file1"]["sha256"]
        )
        
        result = verifier.verify_model_integrity(sample_model_spec, temp_dir)
        
        assert result.success is False
        assert len(result.verified_files) == 1  # Only file2 should pass
        assert len(result.failed_files) == 1    # file1 should fail
        
        failed_file = result.failed_files[0]
        assert failed_file.file_path == "file1.txt"
        assert failed_file.method_used == VerificationMethod.SIZE_ONLY
        assert "Size mismatch" in failed_file.error_message
    
    def test_missing_files_verification(self, verifier, temp_dir, sample_model_spec):
        """Test verification with missing files."""
        # Remove one of the test files
        (temp_dir / "file1.txt").unlink()
        
        result = verifier.verify_model_integrity(sample_model_spec, temp_dir)
        
        assert result.success is False
        assert len(result.verified_files) == 1  # Only file2 should pass
        assert len(result.failed_files) == 0
        assert len(result.missing_files) == 1
        assert "file1.txt" in result.missing_files
    
    def test_hf_etag_fallback_verification(self, verifier, temp_dir, sample_files):
        """Test HuggingFace ETag fallback verification when SHA256 is not available."""
        # Create spec without SHA256 checksums
        spec_without_sha256 = ModelSpec(
            model_id="test-model@1.0.0",
            version="1.0.0",
            variants=["fp16"],
            default_variant="fp16",
            files=[
                FileSpec(
                    path="file1.txt",
                    size=sample_files["file1"]["size"],
                    sha256=""  # No SHA256 checksum
                )
            ],
            sources=["test://source"],
            allow_patterns=["*"],
            resolution_caps=[],
            optional_components=[],
            lora_required=False,
            description="Test model"
        )
        
        # Provide HF metadata with correct ETag
        hf_metadata = {
            "file1.txt": HFFileMetadata(
                etag=sample_files["file1"]["md5"],
                size=sample_files["file1"]["size"]
            )
        }
        
        result = verifier.verify_model_integrity(
            spec_without_sha256, temp_dir, hf_metadata=hf_metadata
        )
        
        assert result.success is True
        assert len(result.verified_files) == 1
        assert result.verified_files[0].method_used == VerificationMethod.HF_ETAG
    
    def test_hf_etag_mismatch(self, verifier, temp_dir, sample_files):
        """Test HuggingFace ETag verification with mismatch."""
        spec_without_sha256 = ModelSpec(
            model_id="test-model@1.0.0",
            version="1.0.0",
            variants=["fp16"],
            default_variant="fp16",
            files=[
                FileSpec(
                    path="file1.txt",
                    size=sample_files["file1"]["size"],
                    sha256=""  # No SHA256 checksum
                )
            ],
            sources=["test://source"],
            allow_patterns=["*"],
            resolution_caps=[],
            optional_components=[],
            lora_required=False,
            description="Test model"
        )
        
        # Provide HF metadata with incorrect ETag
        hf_metadata = {
            "file1.txt": HFFileMetadata(
                etag="incorrect_etag_value",
                size=sample_files["file1"]["size"]
            )
        }
        
        result = verifier.verify_model_integrity(
            spec_without_sha256, temp_dir, hf_metadata=hf_metadata
        )
        
        assert result.success is False
        assert len(result.failed_files) == 1
        assert result.failed_files[0].method_used == VerificationMethod.HF_ETAG
        assert "ETag mismatch" in result.failed_files[0].error_message
    
    def test_size_only_fallback(self, verifier, temp_dir, sample_files):
        """Test size-only verification as last resort."""
        spec_size_only = ModelSpec(
            model_id="test-model@1.0.0",
            version="1.0.0",
            variants=["fp16"],
            default_variant="fp16",
            files=[
                FileSpec(
                    path="file1.txt",
                    size=sample_files["file1"]["size"],
                    sha256=""  # No SHA256 checksum
                )
            ],
            sources=["test://source"],
            allow_patterns=["*"],
            resolution_caps=[],
            optional_components=[],
            lora_required=False,
            description="Test model"
        )
        
        # No HF metadata provided, should fall back to size-only
        result = verifier.verify_model_integrity(spec_size_only, temp_dir)
        
        assert result.success is True
        assert len(result.verified_files) == 1
        assert result.verified_files[0].method_used == VerificationMethod.SIZE_ONLY
    
    def test_manifest_signature_verification_success(self, verifier, temp_dir, sample_model_spec):
        """Test successful manifest signature verification."""
        # Mock the signature verification to return True
        with patch.object(verifier, '_verify_manifest_signature', return_value=True):
            result = verifier.verify_model_integrity(
                sample_model_spec, 
                temp_dir,
                manifest_signature="mock_signature",
                public_key="mock_public_key"
            )
            
            assert result.success is True
            assert result.manifest_signature_valid is True
    
    def test_manifest_signature_verification_failure(self, verifier, temp_dir, sample_model_spec):
        """Test failed manifest signature verification."""
        # Mock the signature verification to return False
        with patch.object(verifier, '_verify_manifest_signature', return_value=False):
            result = verifier.verify_model_integrity(
                sample_model_spec,
                temp_dir,
                manifest_signature="invalid_signature",
                public_key="mock_public_key"
            )
            
            assert result.success is False
            assert result.manifest_signature_valid is False
    
    def test_retry_logic_on_verification_failure(self, temp_dir, sample_files, sample_model_spec):
        """Test retry logic when verification initially fails."""
        retry_config = RetryConfig(max_attempts=3, base_delay=0.01)
        verifier = IntegrityVerifier(retry_config)
        
        # Mock _calculate_sha256 to fail twice then succeed
        call_count = 0
        original_calculate_sha256 = verifier._calculate_sha256
        
        def mock_calculate_sha256(file_path):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise IOError("Temporary file read error")
            return original_calculate_sha256(file_path)
        
        with patch.object(verifier, '_calculate_sha256', side_effect=mock_calculate_sha256):
            result = verifier.verify_model_integrity(sample_model_spec, temp_dir)
            
            # Should eventually succeed after retries
            assert result.success is True
            assert call_count == 3  # Failed twice, succeeded on third attempt
    
    def test_verification_performance_metrics(self, verifier, temp_dir, sample_files, sample_model_spec):
        """Test that verification includes performance metrics."""
        result = verifier.verify_model_integrity(sample_model_spec, temp_dir)
        
        assert result.total_verification_time > 0
        for verified_file in result.verified_files:
            assert verified_file.verification_time > 0
    
    def test_canonical_spec_data_creation(self, verifier, sample_model_spec):
        """Test creation of canonical spec data for signature verification."""
        canonical_data = verifier._create_canonical_spec_data(sample_model_spec)
        
        # Should be valid JSON
        parsed_data = json.loads(canonical_data)
        
        # Should contain all required fields
        assert parsed_data["model_id"] == sample_model_spec.model_id
        assert parsed_data["version"] == sample_model_spec.version
        assert len(parsed_data["files"]) == len(sample_model_spec.files)
        
        # Should be deterministic (same input produces same output)
        canonical_data2 = verifier._create_canonical_spec_data(sample_model_spec)
        assert canonical_data == canonical_data2
    
    def test_extract_hf_metadata_from_download(self, verifier, temp_dir, sample_files):
        """Test extraction of HF metadata from downloaded files."""
        downloaded_files = [sample_files["file1"]["path"], sample_files["file2"]["path"]]
        
        metadata = verifier.extract_hf_metadata_from_download(downloaded_files)
        
        assert len(metadata) == 2
        for file_name, file_metadata in metadata.items():
            assert isinstance(file_metadata, HFFileMetadata)
            assert file_metadata.etag is not None
            assert file_metadata.size > 0
    
    def test_comprehensive_failure_scenario(self, verifier, temp_dir, sample_model_spec):
        """Test comprehensive failure scenario with multiple issues."""
        # Remove one file (missing)
        (temp_dir / "file1.txt").unlink()
        
        # Corrupt the other file (wrong size)
        corrupted_content = b"corrupted content"
        (temp_dir / "subdir" / "file2.bin").write_bytes(corrupted_content)
        
        result = verifier.verify_model_integrity(sample_model_spec, temp_dir)
        
        assert result.success is False
        assert len(result.missing_files) == 1
        assert len(result.failed_files) == 1
        assert "file1.txt" in result.missing_files
        assert result.failed_files[0].file_path == "subdir/file2.bin"
        assert "Size mismatch" in result.failed_files[0].error_message
    
    def test_large_file_chunked_processing(self, verifier, temp_dir):
        """Test that large files are processed in chunks efficiently."""
        # Create a larger test file
        large_content = b"x" * (2 * 1024 * 1024)  # 2MB file
        large_file_path = temp_dir / "large_file.bin"
        large_file_path.write_bytes(large_content)
        
        large_file_sha256 = hashlib.sha256(large_content).hexdigest()
        
        spec = ModelSpec(
            model_id="test-large@1.0.0",
            version="1.0.0",
            variants=["fp16"],
            default_variant="fp16",
            files=[
                FileSpec(
                    path="large_file.bin",
                    size=len(large_content),
                    sha256=large_file_sha256
                )
            ],
            sources=["test://source"],
            allow_patterns=["*"],
            resolution_caps=[],
            optional_components=[],
            lora_required=False,
            description="Test large file model"
        )
        
        start_time = time.time()
        result = verifier.verify_model_integrity(spec, temp_dir)
        duration = time.time() - start_time
        
        assert result.success is True
        assert len(result.verified_files) == 1
        # Should complete in reasonable time (less than 5 seconds for 2MB)
        assert duration < 5.0
    
    def test_concurrent_verification_safety(self, verifier, temp_dir, sample_files, sample_model_spec):
        """Test that verification is safe for concurrent access."""
        import threading
        import concurrent.futures
        
        results = []
        
        def verify_model():
            return verifier.verify_model_integrity(sample_model_spec, temp_dir)
        
        # Run multiple verifications concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(verify_model) for _ in range(4)]
            results = [future.result() for future in futures]
        
        # All should succeed and produce consistent results
        for result in results:
            assert result.success is True
            assert len(result.verified_files) == 2
            assert len(result.failed_files) == 0
    
    def test_error_handling_and_logging(self, verifier, temp_dir, sample_model_spec):
        """Test proper error handling and logging during verification."""
        # Make one of the files unreadable
        file_path = temp_dir / "file1.txt"
        
        with patch('builtins.open', side_effect=PermissionError("Access denied")):
            result = verifier.verify_model_integrity(sample_model_spec, temp_dir)
            
            # Should handle the error gracefully
            assert result.success is False
            # The file should be marked as failed due to the exception
            assert len(result.failed_files) >= 1


class TestIntegrityVerifierEdgeCases:
    """Test edge cases and error conditions for IntegrityVerifier."""
    
    @pytest.fixture
    def verifier(self):
        return IntegrityVerifier()
    
    def test_empty_model_directory(self, verifier, tmp_path):
        """Test verification of empty model directory."""
        spec = ModelSpec(
            model_id="empty-model@1.0.0",
            version="1.0.0",
            variants=["fp16"],
            default_variant="fp16",
            files=[
                FileSpec(path="missing.txt", size=100, sha256="abc123")
            ],
            sources=["test://source"],
            allow_patterns=["*"],
            resolution_caps=[],
            optional_components=[],
            lora_required=False
        )
        
        result = verifier.verify_model_integrity(spec, tmp_path)
        
        assert result.success is False
        assert len(result.missing_files) == 1
        assert "missing.txt" in result.missing_files
    
    def test_nonexistent_model_directory(self, verifier):
        """Test verification of nonexistent model directory."""
        spec = ModelSpec(
            model_id="nonexistent@1.0.0",
            version="1.0.0",
            variants=["fp16"],
            default_variant="fp16",
            files=[FileSpec(path="file.txt", size=100, sha256="abc123")],
            sources=["test://source"],
            allow_patterns=["*"],
            resolution_caps=[],
            optional_components=[],
            lora_required=False
        )
        
        nonexistent_path = Path("/nonexistent/path")
        result = verifier.verify_model_integrity(spec, nonexistent_path)
        
        assert result.success is False
        assert len(result.missing_files) == 1
    
    def test_zero_byte_files(self, verifier, tmp_path):
        """Test verification of zero-byte files."""
        # Create zero-byte file
        zero_file = tmp_path / "zero.txt"
        zero_file.write_bytes(b"")
        
        spec = ModelSpec(
            model_id="zero-file@1.0.0",
            version="1.0.0",
            variants=["fp16"],
            default_variant="fp16",
            files=[
                FileSpec(
                    path="zero.txt",
                    size=0,
                    sha256=hashlib.sha256(b"").hexdigest()
                )
            ],
            sources=["test://source"],
            allow_patterns=["*"],
            resolution_caps=[],
            optional_components=[],
            lora_required=False
        )
        
        result = verifier.verify_model_integrity(spec, tmp_path)
        
        assert result.success is True
        assert len(result.verified_files) == 1
    
    def test_unicode_file_paths(self, verifier, tmp_path):
        """Test verification with Unicode file paths."""
        # Create file with Unicode name
        unicode_file = tmp_path / "测试文件.txt"
        unicode_content = "Unicode content 测试".encode('utf-8')
        unicode_file.write_bytes(unicode_content)
        
        spec = ModelSpec(
            model_id="unicode@1.0.0",
            version="1.0.0",
            variants=["fp16"],
            default_variant="fp16",
            files=[
                FileSpec(
                    path="测试文件.txt",
                    size=len(unicode_content),
                    sha256=hashlib.sha256(unicode_content).hexdigest()
                )
            ],
            sources=["test://source"],
            allow_patterns=["*"],
            resolution_caps=[],
            optional_components=[],
            lora_required=False
        )
        
        result = verifier.verify_model_integrity(spec, tmp_path)
        
        assert result.success is True
        assert len(result.verified_files) == 1