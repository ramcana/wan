"""
End-to-end integration tests for complete Model Orchestrator workflows.

Tests complete user journeys from model request to ready-to-use model paths,
including all storage backends, error recovery, and concurrent access scenarios.
"""

import asyncio
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, List
from unittest.mock import Mock, patch

import pytest

from backend.core.model_orchestrator.model_ensurer import ModelEnsurer
from backend.core.model_orchestrator.model_registry import ModelRegistry
from backend.core.model_orchestrator.model_resolver import ModelResolver
from backend.core.model_orchestrator.lock_manager import LockManager
from backend.core.model_orchestrator.garbage_collector import GarbageCollector
from backend.core.model_orchestrator.storage_backends.hf_store import HFStore
from backend.core.model_orchestrator.storage_backends.s3_store import S3Store
from backend.core.model_orchestrator.exceptions import (
    ModelNotFoundError,
    InsufficientSpaceError,
    ChecksumVerificationError
)


class TestEndToEndWorkflows:
    """Test complete workflows from user request to model availability."""

    @pytest.fixture
    def temp_models_root(self):
        """Create temporary models root directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def sample_manifest(self, temp_models_root):
        """Create sample manifest file for testing."""
        manifest_content = """
schema_version = 1

[models."test-model@1.0.0"]
description = "Test model for end-to-end testing"
version = "1.0.0"
variants = ["fp16", "bf16"]
default_variant = "fp16"
resolution_caps = ["512x512", "1024x1024"]
optional_components = []
lora_required = false

[[models."test-model@1.0.0".files]]
path = "model_index.json"
size = 1024
sha256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

[[models."test-model@1.0.0".files]]
path = "unet/diffusion_pytorch_model.safetensors"
size = 5368709120
sha256 = "a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3"

[models."test-model@1.0.0".sources]
priority = [
    "local://test-models/test-model@1.0.0",
    "hf://test-org/test-model"
]
allow_patterns = ["*.safetensors", "*.json"]
"""
        manifest_path = Path(temp_models_root) / "models.toml"
        manifest_path.write_text(manifest_content)
        return str(manifest_path)

    @pytest.fixture
    def orchestrator_components(self, temp_models_root, sample_manifest):
        """Set up complete orchestrator component stack."""
        registry = ModelRegistry(sample_manifest)
        resolver = ModelResolver(temp_models_root)
        lock_manager = LockManager(temp_models_root)
        
        # Mock storage backends for testing
        hf_store = Mock(spec=HFStore)
        s3_store = Mock(spec=S3Store)
        
        ensurer = ModelEnsurer(
            registry=registry,
            resolver=resolver,
            lock_manager=lock_manager,
            storage_backends=[hf_store, s3_store]
        )
        
        gc = GarbageCollector(resolver, max_total_size=10 * 1024**3)  # 10GB limit
        
        return {
            'registry': registry,
            'resolver': resolver,
            'lock_manager': lock_manager,
            'ensurer': ensurer,
            'gc': gc,
            'hf_store': hf_store,
            's3_store': s3_store
        }

    def test_complete_model_download_workflow(self, orchestrator_components, temp_models_root):
        """Test complete workflow: request → download → verify → ready path."""
        components = orchestrator_components
        ensurer = components['ensurer']
        hf_store = components['hf_store']
        
        # Mock successful download
        def mock_download(source_url, local_dir, file_specs, progress_callback=None):
            # Create mock files
            model_dir = Path(local_dir)
            model_dir.mkdir(parents=True, exist_ok=True)
            
            (model_dir / "model_index.json").write_text('{"test": "data"}')
            (model_dir / "unet").mkdir(exist_ok=True)
            (model_dir / "unet" / "diffusion_pytorch_model.safetensors").write_bytes(b"mock_model_data")
            
            return Mock(success=True, bytes_downloaded=1024)
        
        hf_store.can_handle.return_value = True
        hf_store.download.side_effect = mock_download
        
        # Execute workflow
        model_path = ensurer.ensure("test-model@1.0.0", variant="fp16")
        
        # Verify results
        assert Path(model_path).exists()
        assert Path(model_path, "model_index.json").exists()
        assert Path(model_path, "unet", "diffusion_pytorch_model.safetensors").exists()
        
        # Verify status is complete
        status = ensurer.status("test-model@1.0.0", variant="fp16")
        assert status.state == "COMPLETE"

    def test_concurrent_model_access_workflow(self, orchestrator_components):
        """Test concurrent access: multiple processes requesting same model."""
        components = orchestrator_components
        ensurer = components['ensurer']
        hf_store = components['hf_store']
        
        download_started = threading.Event()
        download_can_complete = threading.Event()
        results = {}
        
        def slow_mock_download(source_url, local_dir, file_specs, progress_callback=None):
            download_started.set()
            download_can_complete.wait(timeout=10)  # Wait for signal
            
            model_dir = Path(local_dir)
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "model_index.json").write_text('{"test": "data"}')
            (model_dir / "unet").mkdir(exist_ok=True)
            (model_dir / "unet" / "diffusion_pytorch_model.safetensors").write_bytes(b"mock_data")
            
            return Mock(success=True, bytes_downloaded=1024)
        
        hf_store.can_handle.return_value = True
        hf_store.download.side_effect = slow_mock_download
        
        def worker(worker_id):
            try:
                path = ensurer.ensure("test-model@1.0.0", variant="fp16")
                results[worker_id] = {'success': True, 'path': path}
            except Exception as e:
                results[worker_id] = {'success': False, 'error': str(e)}
        
        # Start two concurrent workers
        thread1 = threading.Thread(target=worker, args=(1,))
        thread2 = threading.Thread(target=worker, args=(2,))
        
        thread1.start()
        thread2.start()
        
        # Wait for download to start, then allow completion
        download_started.wait(timeout=5)
        time.sleep(0.1)  # Brief pause to ensure both threads are waiting
        download_can_complete.set()
        
        thread1.join(timeout=10)
        thread2.join(timeout=10)
        
        # Both should succeed with same path
        assert results[1]['success']
        assert results[2]['success']
        assert results[1]['path'] == results[2]['path']

    def test_source_failover_workflow(self, orchestrator_components):
        """Test failover: primary source fails, secondary succeeds."""
        components = orchestrator_components
        ensurer = components['ensurer']
        hf_store = components['hf_store']
        s3_store = components['s3_store']
        
        # First source (S3) fails
        s3_store.can_handle.return_value = True
        s3_store.download.side_effect = Exception("S3 connection failed")
        
        # Second source (HF) succeeds
        def mock_hf_download(source_url, local_dir, file_specs, progress_callback=None):
            model_dir = Path(local_dir)
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "model_index.json").write_text('{"test": "data"}')
            (model_dir / "unet").mkdir(exist_ok=True)
            (model_dir / "unet" / "diffusion_pytorch_model.safetensors").write_bytes(b"mock_data")
            return Mock(success=True, bytes_downloaded=1024)
        
        hf_store.can_handle.return_value = True
        hf_store.download.side_effect = mock_hf_download
        
        # Execute workflow - should succeed despite S3 failure
        model_path = ensurer.ensure("test-model@1.0.0", variant="fp16")
        
        assert Path(model_path).exists()
        assert s3_store.download.called  # S3 was attempted
        assert hf_store.download.called  # HF was used as fallback

    def test_disk_space_management_workflow(self, orchestrator_components, temp_models_root):
        """Test disk space management: quota exceeded triggers GC."""
        components = orchestrator_components
        ensurer = components['ensurer']
        gc = components['gc']
        hf_store = components['hf_store']
        
        # Mock disk space check to simulate low space
        with patch('shutil.disk_usage') as mock_disk_usage:
            mock_disk_usage.return_value = Mock(free=1024)  # Very low free space
            
            # Mock download that would exceed space
            def mock_download(source_url, local_dir, file_specs, progress_callback=None):
                raise InsufficientSpaceError("Not enough disk space")
            
            hf_store.can_handle.return_value = True
            hf_store.download.side_effect = mock_download
            
            # Should raise space error
            with pytest.raises(InsufficientSpaceError):
                ensurer.ensure("test-model@1.0.0", variant="fp16")

    def test_integrity_verification_workflow(self, orchestrator_components):
        """Test integrity verification: checksum failure triggers re-download."""
        components = orchestrator_components
        ensurer = components['ensurer']
        hf_store = components['hf_store']
        
        download_count = 0
        
        def mock_download_with_bad_checksum(source_url, local_dir, file_specs, progress_callback=None):
            nonlocal download_count
            download_count += 1
            
            model_dir = Path(local_dir)
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "model_index.json").write_text('{"test": "data"}')
            (model_dir / "unet").mkdir(exist_ok=True)
            
            # First download has wrong checksum, second is correct
            if download_count == 1:
                (model_dir / "unet" / "diffusion_pytorch_model.safetensors").write_bytes(b"wrong_data")
            else:
                (model_dir / "unet" / "diffusion_pytorch_model.safetensors").write_bytes(b"correct_data")
            
            return Mock(success=True, bytes_downloaded=1024)
        
        hf_store.can_handle.return_value = True
        hf_store.download.side_effect = mock_download_with_bad_checksum
        
        # Mock integrity verifier to fail first time, pass second time
        with patch('backend.core.model_orchestrator.integrity_verifier.IntegrityVerifier') as mock_verifier:
            verifier_instance = Mock()
            mock_verifier.return_value = verifier_instance
            
            def mock_verify(model_dir, file_specs):
                if download_count == 1:
                    raise ChecksumVerificationError("Checksum mismatch")
                return Mock(verified=True)
            
            verifier_instance.verify_model.side_effect = mock_verify
            
            # Should succeed after retry
            model_path = ensurer.ensure("test-model@1.0.0", variant="fp16")
            
            assert Path(model_path).exists()
            assert download_count == 2  # Should have retried once

    def test_resume_interrupted_download_workflow(self, orchestrator_components, temp_models_root):
        """Test resume capability: interrupted download resumes correctly."""
        components = orchestrator_components
        ensurer = components['ensurer']
        resolver = components['resolver']
        hf_store = components['hf_store']
        
        model_id = "test-model@1.0.0"
        variant = "fp16"
        
        # Create partial download state
        temp_dir = resolver.temp_dir(model_id)
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        (Path(temp_dir) / "model_index.json").write_text('{"test": "data"}')
        # Missing the large file to simulate partial download
        
        def mock_resume_download(source_url, local_dir, file_specs, progress_callback=None):
            # Complete the missing file
            model_dir = Path(local_dir)
            (model_dir / "unet").mkdir(exist_ok=True)
            (model_dir / "unet" / "diffusion_pytorch_model.safetensors").write_bytes(b"resumed_data")
            return Mock(success=True, bytes_downloaded=512)  # Only partial download needed
        
        hf_store.can_handle.return_value = True
        hf_store.download.side_effect = mock_resume_download
        
        # Should resume and complete
        model_path = ensurer.ensure(model_id, variant=variant)
        
        assert Path(model_path).exists()
        assert Path(model_path, "model_index.json").exists()
        assert Path(model_path, "unet", "diffusion_pytorch_model.safetensors").exists()

    def test_health_monitoring_workflow(self, orchestrator_components):
        """Test health monitoring: status checks without side effects."""
        components = orchestrator_components
        ensurer = components['ensurer']
        
        # Check status of non-existent model
        status = ensurer.status("test-model@1.0.0", variant="fp16")
        assert status.state == "NOT_PRESENT"
        assert status.bytes_needed > 0
        
        # Status check should not trigger download
        components['hf_store'].download.assert_not_called()

    def test_garbage_collection_workflow(self, orchestrator_components, temp_models_root):
        """Test garbage collection: LRU cleanup when quota exceeded."""
        components = orchestrator_components
        gc = components['gc']
        resolver = components['resolver']
        
        # Create some mock model directories with different access times
        old_model_dir = Path(resolver.local_dir("old-model@1.0.0"))
        old_model_dir.mkdir(parents=True, exist_ok=True)
        (old_model_dir / "model.bin").write_bytes(b"old_data" * 1000)
        
        new_model_dir = Path(resolver.local_dir("new-model@1.0.0"))
        new_model_dir.mkdir(parents=True, exist_ok=True)
        (new_model_dir / "model.bin").write_bytes(b"new_data" * 1000)
        
        # Set different access times
        old_time = time.time() - 86400  # 1 day ago
        new_time = time.time()  # Now
        
        os.utime(old_model_dir, (old_time, old_time))
        os.utime(new_model_dir, (new_time, new_time))
        
        # Run garbage collection
        result = gc.collect(dry_run=False)
        
        assert result.bytes_reclaimed > 0
        # Old model should be removed, new model preserved
        assert not old_model_dir.exists()
        assert new_model_dir.exists()


class TestCrossWorkflowIntegration:
    """Test integration between different workflow components."""

    def test_cli_to_api_integration(self, orchestrator_components):
        """Test CLI operations integrate with API operations."""
        components = orchestrator_components
        ensurer = components['ensurer']
        hf_store = components['hf_store']
        
        # Mock successful download
        def mock_download(source_url, local_dir, file_specs, progress_callback=None):
            model_dir = Path(local_dir)
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "model_index.json").write_text('{"test": "data"}')
            return Mock(success=True, bytes_downloaded=1024)
        
        hf_store.can_handle.return_value = True
        hf_store.download.side_effect = mock_download
        
        # Use API to ensure model
        api_path = ensurer.ensure("test-model@1.0.0", variant="fp16")
        
        # Use CLI to check status (simulate CLI call)
        status = ensurer.status("test-model@1.0.0", variant="fp16")
        
        # Both should show consistent state
        assert Path(api_path).exists()
        assert status.state == "COMPLETE"
        
        # CLI and API should return same path
        cli_path = ensurer.ensure("test-model@1.0.0", variant="fp16")  # Should not re-download
        assert api_path == cli_path

    def test_pipeline_integration_workflow(self, orchestrator_components):
        """Test integration with WAN pipeline loader."""
        components = orchestrator_components
        ensurer = components['ensurer']
        hf_store = components['hf_store']
        
        # Mock successful download
        def mock_download(source_url, local_dir, file_specs, progress_callback=None):
            model_dir = Path(local_dir)
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "model_index.json").write_text('{"test": "data"}')
            (model_dir / "unet").mkdir(exist_ok=True)
            (model_dir / "unet" / "diffusion_pytorch_model.safetensors").write_bytes(b"mock_data")
            return Mock(success=True, bytes_downloaded=1024)
        
        hf_store.can_handle.return_value = True
        hf_store.download.side_effect = mock_download
        
        # Simulate get_wan_paths() integration
        model_path = ensurer.ensure("test-model@1.0.0", variant="fp16")
        
        # Verify model structure is suitable for pipeline loading
        assert Path(model_path, "model_index.json").exists()
        assert Path(model_path, "unet", "diffusion_pytorch_model.safetensors").exists()
        
        # Verify path is absolute and accessible
        assert Path(model_path).is_absolute()
        assert Path(model_path).exists()

    def test_metrics_and_monitoring_integration(self, orchestrator_components):
        """Test metrics collection during complete workflows."""
        components = orchestrator_components
        ensurer = components['ensurer']
        hf_store = components['hf_store']
        
        # Mock download with metrics tracking
        download_metrics = []
        
        def mock_download_with_metrics(source_url, local_dir, file_specs, progress_callback=None):
            download_metrics.append({
                'source': source_url,
                'start_time': time.time(),
                'bytes': sum(spec.size for spec in file_specs)
            })
            
            model_dir = Path(local_dir)
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "model_index.json").write_text('{"test": "data"}')
            
            download_metrics[-1]['end_time'] = time.time()
            return Mock(success=True, bytes_downloaded=1024)
        
        hf_store.can_handle.return_value = True
        hf_store.download.side_effect = mock_download_with_metrics
        
        # Perform operation that should generate metrics
        model_path = ensurer.ensure("test-model@1.0.0", variant="fp16")
        
        # Verify metrics were collected
        assert len(download_metrics) > 0
        assert 'start_time' in download_metrics[0]
        assert 'end_time' in download_metrics[0]
        assert download_metrics[0]['bytes'] > 0
        
        # Verify model was successfully downloaded
        assert Path(model_path).exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])