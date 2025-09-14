"""
Integration tests for Model Health Service with Model Orchestrator components.
"""

import json
import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

from backend.services.model_health_service import (
    ModelHealthService, initialize_model_health_service, get_model_health_service
)
from backend.core.model_orchestrator import (
    ModelRegistry, ModelResolver, ModelEnsurer, ModelStatus, ModelStatusInfo
)


class TestHealthServiceIntegration:
    """Integration tests for health service with orchestrator components."""
    
    @pytest.fixture
    def temp_models_dir(self):
        """Create temporary models directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def mock_manifest_data(self):
        """Mock manifest data for testing."""
        return {
            "schema_version": 1,
            "models": {
                "test-model@1.0": {
                    "version": "1.0",
                    "variants": ["fp16", "bf16"],
                    "default_variant": "fp16",
                    "files": [
                        {
                            "path": "config.json",
                            "size": 1024,
                            "sha256": "abc123"
                        },
                        {
                            "path": "model.safetensors",
                            "size": 5000000000,
                            "sha256": "def456"
                        }
                    ],
                    "sources": ["local://test-model@1.0"]
                }
            }
        }
    
    @pytest.mark.asyncio
    async def test_health_service_with_complete_model(self, temp_models_dir, mock_manifest_data):
        """Test health service with a complete model."""
        # Create model directory structure
        model_dir = temp_models_dir / "test-model@1.0"
        model_dir.mkdir(parents=True)
        
        # Create model files
        (model_dir / "config.json").write_text('{"model_type": "test"}')
        (model_dir / "model.safetensors").write_bytes(b"fake model data" * 1000)  # Make it larger
        
        # Create verification file
        verification_data = {
            "model_id": "test-model@1.0",
            "verified_at": time.time(),
            "files": [
                {"path": "config.json", "size": 20},
                {"path": "model.safetensors", "size": 15000}
            ]
        }
        
        verification_file = model_dir / ".verified.json"
        with open(verification_file, 'w') as f:
            json.dump(verification_data, f)
        
        # Create mock components
        registry = Mock(spec=ModelRegistry)
        registry.list_models.return_value = ["test-model@1.0"]
        
        resolver = Mock(spec=ModelResolver)
        resolver.local_dir.return_value = str(model_dir)
        
        ensurer = Mock(spec=ModelEnsurer)
        ensurer.status.return_value = ModelStatusInfo(
            status=ModelStatus.COMPLETE,
            local_path=str(model_dir),
            missing_files=[],
            bytes_needed=0
        )
        
        # Create health service
        health_service = ModelHealthService(registry, resolver, ensurer)
        
        # Test health status
        response = await health_service.get_health_status(dry_run=True)
        
        assert response.status == "healthy"
        assert response.total_models == 1
        assert response.healthy_models == 1
        assert response.missing_models == 0
        assert response.total_bytes_needed == 0
        assert response.response_time_ms < 100  # Should be fast
        
        # Check individual model
        model_health = response.models["test-model@1.0"]
        assert model_health.status == "COMPLETE"
        assert model_health.last_verified is not None
        assert model_health.bytes_needed == 0
    
    @pytest.mark.asyncio
    async def test_health_service_with_missing_model(self, temp_models_dir, mock_manifest_data):
        """Test health service with a missing model."""
        # Create mock components
        registry = Mock(spec=ModelRegistry)
        registry.list_models.return_value = ["missing-model@1.0"]
        
        resolver = Mock(spec=ModelResolver)
        resolver.local_dir.return_value = str(temp_models_dir / "missing-model@1.0")
        
        ensurer = Mock(spec=ModelEnsurer)
        ensurer.status.return_value = ModelStatusInfo(
            status=ModelStatus.NOT_PRESENT,
            local_path=str(temp_models_dir / "missing-model@1.0"),
            missing_files=["config.json", "model.safetensors"],
            bytes_needed=5000001024  # ~5GB
        )
        
        # Create health service
        health_service = ModelHealthService(registry, resolver, ensurer)
        
        # Test health status
        response = await health_service.get_health_status(dry_run=True)
        
        assert response.status == "degraded"
        assert response.total_models == 1
        assert response.healthy_models == 0
        assert response.missing_models == 1
        assert response.total_bytes_needed == 5000001024
        
        # Check individual model
        model_health = response.models["missing-model@1.0"]
        assert model_health.status == "NOT_PRESENT"
        assert len(model_health.missing_files) == 2
        assert model_health.bytes_needed == 5000001024
    
    @pytest.mark.asyncio
    async def test_health_service_performance_requirement(self, temp_models_dir):
        """Test that health service meets performance requirements."""
        # Create multiple models to test performance
        model_ids = [f"model-{i}@1.0" for i in range(5)]
        
        registry = Mock(spec=ModelRegistry)
        registry.list_models.return_value = model_ids
        
        resolver = Mock(spec=ModelResolver)
        resolver.local_dir.side_effect = lambda model_id, variant=None: str(temp_models_dir / model_id)
        
        ensurer = Mock(spec=ModelEnsurer)
        ensurer.status.return_value = ModelStatusInfo(
            status=ModelStatus.COMPLETE,
            local_path=str(temp_models_dir / "test"),
            missing_files=[],
            bytes_needed=0
        )
        
        # Create health service with strict timeout
        health_service = ModelHealthService(registry, resolver, ensurer, timeout_ms=100.0)
        
        # Test health status
        start_time = time.time()
        response = await health_service.get_health_status(dry_run=True)
        end_time = time.time()
        
        # Verify performance requirements
        actual_time_ms = (end_time - start_time) * 1000
        assert actual_time_ms < 150  # Allow some buffer for test environment
        assert response.response_time_ms < 100  # Service's own measurement
    
    @pytest.mark.asyncio
    async def test_health_service_timeout_protection(self, temp_models_dir):
        """Test that health service respects timeout limits."""
        model_ids = [f"model-{i}@1.0" for i in range(10)]  # Many models
        
        registry = Mock(spec=ModelRegistry)
        registry.list_models.return_value = model_ids
        
        resolver = Mock(spec=ModelResolver)
        resolver.local_dir.side_effect = lambda model_id, variant=None: str(temp_models_dir / model_id)
        
        # Mock slow ensurer
        def slow_status(model_id, variant=None):
            time.sleep(0.02)  # 20ms per model
            return ModelStatusInfo(
                status=ModelStatus.COMPLETE,
                local_path=str(temp_models_dir / model_id),
                missing_files=[],
                bytes_needed=0
            )
        
        ensurer = Mock(spec=ModelEnsurer)
        ensurer.status.side_effect = slow_status
        
        # Create health service with short timeout
        health_service = ModelHealthService(registry, resolver, ensurer, timeout_ms=100.0)
        
        # Test health status
        response = await health_service.get_health_status(dry_run=True)
        
        # Should complete within timeout and may have skipped some models
        assert response.response_time_ms < 150  # Allow buffer
        assert response.total_models <= len(model_ids)  # May have skipped some
    
    def test_global_service_initialization_integration(self, temp_models_dir):
        """Test global service initialization with real components."""
        # Create mock components
        registry = Mock(spec=ModelRegistry)
        resolver = Mock(spec=ModelResolver)
        ensurer = Mock(spec=ModelEnsurer)
        
        # Initialize global service
        service = initialize_model_health_service(registry, resolver, ensurer, timeout_ms=50.0)
        
        assert service is not None
        assert service.timeout_ms == 50.0
        assert service.registry is registry
        assert service.resolver is resolver
        assert service.ensurer is ensurer
        
        # Verify global retrieval
        retrieved_service = get_model_health_service()
        assert retrieved_service is service
    
    @pytest.mark.asyncio
    async def test_health_service_dry_run_behavior(self, temp_models_dir):
        """Test that dry_run parameter prevents side effects."""
        registry = Mock(spec=ModelRegistry)
        registry.list_models.return_value = ["test-model@1.0"]
        
        resolver = Mock(spec=ModelResolver)
        resolver.local_dir.return_value = str(temp_models_dir / "test-model@1.0")
        
        ensurer = Mock(spec=ModelEnsurer)
        ensurer.status.return_value = ModelStatusInfo(
            status=ModelStatus.NOT_PRESENT,
            local_path=str(temp_models_dir / "test-model@1.0"),
            missing_files=["model.safetensors"],
            bytes_needed=1000000
        )
        
        health_service = ModelHealthService(registry, resolver, ensurer)
        
        # Test with dry_run=True (default)
        response = await health_service.get_health_status(dry_run=True)
        
        # Verify no download was triggered
        ensurer.ensure.assert_not_called()
        
        # Verify status was checked
        assert ensurer.status.called
        assert response.status == "degraded"
        assert response.total_bytes_needed == 1000000


if __name__ == "__main__":
    pytest.main([__file__])