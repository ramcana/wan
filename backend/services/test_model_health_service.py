"""
Tests for Model Health Service
"""

import json
import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from backend.services.model_health_service import (
    ModelHealthService, ModelHealthInfo, OrchestratorHealthResponse,
    initialize_model_health_service, get_model_health_service
)
from backend.core.model_orchestrator import ModelStatus, ModelStatusInfo
from backend.core.model_orchestrator.exceptions import ModelNotFoundError


class TestModelHealthService:
    """Test cases for ModelHealthService."""
    
    @pytest.fixture
    def mock_registry(self):
        """Mock model registry."""
        registry = Mock()
        registry.list_models.return_value = ["t2v-A14B", "i2v-A14B", "ti2v-5b"]
        return registry
    
    @pytest.fixture
    def mock_resolver(self):
        """Mock model resolver."""
        resolver = Mock()
        resolver.local_dir.side_effect = lambda model_id, variant=None: f"/models/{model_id}"
        return resolver
    
    @pytest.fixture
    def mock_ensurer(self):
        """Mock model ensurer."""
        ensurer = Mock()
        return ensurer
    
    @pytest.fixture
    def health_service(self, mock_registry, mock_resolver, mock_ensurer):
        """Create health service with mocked dependencies."""
        return ModelHealthService(
            registry=mock_registry,
            resolver=mock_resolver,
            ensurer=mock_ensurer,
            timeout_ms=100.0
        )
    
    @pytest.mark.asyncio
    async def test_get_health_status_all_healthy(self, health_service, mock_ensurer):
        """Test health status when all models are healthy."""
        # Mock all models as complete
        mock_ensurer.status.return_value = ModelStatusInfo(
            status=ModelStatus.COMPLETE,
            local_path="/models/test",
            missing_files=[],
            bytes_needed=0
        )
        
        response = await health_service.get_health_status(dry_run=True)
        
        assert response.status == "healthy"
        assert response.total_models == 3
        assert response.healthy_models == 3
        assert response.missing_models == 0
        assert response.partial_models == 0
        assert response.corrupt_models == 0
        assert response.total_bytes_needed == 0
        assert response.response_time_ms < 100  # Should be fast
        assert len(response.models) == 3
    
    @pytest.mark.asyncio
    async def test_get_health_status_mixed_states(self, health_service, mock_ensurer):
        """Test health status with models in different states."""
        def mock_status(model_id, variant=None):
            if model_id == "t2v-A14B":
                return ModelStatusInfo(
                    status=ModelStatus.COMPLETE,
                    local_path="/models/t2v-A14B",
                    missing_files=[],
                    bytes_needed=0
                )
            elif model_id == "i2v-A14B":
                return ModelStatusInfo(
                    status=ModelStatus.NOT_PRESENT,
                    local_path="/models/i2v-A14B",
                    missing_files=["model.safetensors", "config.json"],
                    bytes_needed=5000000000  # 5GB
                )
            else:  # ti2v-5b
                return ModelStatusInfo(
                    status=ModelStatus.PARTIAL,
                    local_path="/models/ti2v-5b",
                    missing_files=["unet/model.safetensors"],
                    bytes_needed=2000000000  # 2GB
                )
        
        mock_ensurer.status.side_effect = mock_status
        
        response = await health_service.get_health_status(dry_run=True)
        
        assert response.status == "degraded"
        assert response.total_models == 3
        assert response.healthy_models == 1
        assert response.missing_models == 1
        assert response.partial_models == 1
        assert response.corrupt_models == 0
        assert response.total_bytes_needed == 7000000000  # 7GB total
        
        # Check individual model statuses
        assert response.models["t2v-A14B"].status == "COMPLETE"
        assert response.models["i2v-A14B"].status == "NOT_PRESENT"
        assert response.models["ti2v-5b"].status == "PARTIAL"
    
    @pytest.mark.asyncio
    async def test_get_health_status_with_verification_cache(self, health_service, mock_ensurer):
        """Test health status reading from .verified.json cache."""
        # Mock file system operations
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "test"
            model_dir.mkdir()
            
            # Create .verified.json file
            verification_data = {
                "model_id": "test",
                "verified_at": time.time() - 3600,  # 1 hour ago
                "files": []
            }
            
            verification_file = model_dir / ".verified.json"
            with open(verification_file, 'w') as f:
                json.dump(verification_data, f)
            
            # Mock resolver to return our temp directory
            def mock_local_dir(model_id, variant=None):
                return str(model_dir)
            health_service.resolver.local_dir.side_effect = mock_local_dir
            
            mock_ensurer.status.return_value = ModelStatusInfo(
                status=ModelStatus.COMPLETE,
                local_path=str(model_dir),
                missing_files=[],
                bytes_needed=0
            )
            
            model_health = await health_service._get_model_health("test", dry_run=True)
            
            assert model_health.last_verified is not None
            assert model_health.last_verified == verification_data["verified_at"]
    
    @pytest.mark.asyncio
    async def test_get_health_status_timeout_protection(self, health_service, mock_ensurer):
        """Test that health check respects timeout limits."""
        # Mock slow status calls
        def slow_status(model_id, variant=None):
            time.sleep(0.05)  # 50ms delay per model
            return ModelStatusInfo(
                status=ModelStatus.COMPLETE,
                local_path=f"/models/{model_id}",
                missing_files=[],
                bytes_needed=0
            )
        
        mock_ensurer.status.side_effect = slow_status
        
        # Set very short timeout
        health_service.timeout_ms = 80.0  # 80ms timeout
        
        response = await health_service.get_health_status(dry_run=True)
        
        # Should complete within timeout and may skip some models
        assert response.response_time_ms < 150  # Allow some buffer
        assert response.total_models <= 3  # May have skipped some models
    
    @pytest.mark.asyncio
    async def test_get_health_status_error_handling(self, health_service, mock_ensurer):
        """Test error handling in health status check."""
        def mock_status_with_error(model_id, variant=None):
            if model_id == "t2v-A14B":
                raise ModelNotFoundError(f"Model {model_id} not found")
            return ModelStatusInfo(
                status=ModelStatus.COMPLETE,
                local_path=f"/models/{model_id}",
                missing_files=[],
                bytes_needed=0
            )
        
        mock_ensurer.status.side_effect = mock_status_with_error
        
        response = await health_service.get_health_status(dry_run=True)
        
        # Should handle errors gracefully
        assert response.status in ["degraded", "error"]
        assert "t2v-A14B" in response.models
        assert response.models["t2v-A14B"].status == "ERROR"
        assert response.models["t2v-A14B"].error_message is not None
    
    @pytest.mark.asyncio
    async def test_get_model_health_individual(self, health_service, mock_ensurer):
        """Test getting health for individual model."""
        mock_ensurer.status.return_value = ModelStatusInfo(
            status=ModelStatus.PARTIAL,
            local_path="/models/test",
            missing_files=["config.json"],
            bytes_needed=1024
        )
        
        health_info = await health_service.get_model_health("test", variant="fp16")
        
        assert health_info.model_id == "test"
        assert health_info.variant == "fp16"
        assert health_info.status == "PARTIAL"
        assert health_info.missing_files == ["config.json"]
        assert health_info.bytes_needed == 1024
    
    @pytest.mark.asyncio
    async def test_get_model_health_with_error(self, health_service, mock_ensurer):
        """Test individual model health with error."""
        mock_ensurer.status.side_effect = Exception("Test error")
        
        health_info = await health_service.get_model_health("test")
        
        assert health_info.model_id == "test"
        assert health_info.status == "ERROR"
        assert health_info.error_message == "Test error"
        assert health_info.bytes_needed == 0
    
    def test_to_dict_conversion(self, health_service):
        """Test conversion of health response to dictionary."""
        model_health = ModelHealthInfo(
            model_id="test",
            variant="fp16",
            status="COMPLETE",
            local_path="/models/test",
            missing_files=[],
            bytes_needed=0
        )
        
        response = OrchestratorHealthResponse(
            status="healthy",
            timestamp=time.time(),
            models={"test": model_health},
            total_models=1,
            healthy_models=1,
            missing_models=0,
            partial_models=0,
            corrupt_models=0,
            total_bytes_needed=0,
            response_time_ms=50.0
        )
        
        result_dict = health_service.to_dict(response)
        
        assert isinstance(result_dict, dict)
        assert result_dict["status"] == "healthy"
        assert result_dict["total_models"] == 1
        assert "test" in result_dict["models"]
        assert result_dict["models"]["test"]["model_id"] == "test"
    
    def test_global_service_initialization(self, mock_registry, mock_resolver, mock_ensurer):
        """Test global service initialization and retrieval."""
        # Initially should be None
        assert get_model_health_service() is None
        
        # Initialize service
        service = initialize_model_health_service(
            mock_registry, mock_resolver, mock_ensurer, timeout_ms=200.0
        )
        
        assert service is not None
        assert service.timeout_ms == 200.0
        
        # Should be retrievable
        retrieved_service = get_model_health_service()
        assert retrieved_service is service


class TestModelHealthServiceIntegration:
    """Integration tests for ModelHealthService."""
    
    @pytest.mark.asyncio
    async def test_health_service_with_real_file_system(self):
        """Test health service with real file system operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            models_dir = Path(temp_dir)
            
            # Create mock model directory structure
            model_dir = models_dir / "test-model"
            model_dir.mkdir()
            
            # Create some model files
            (model_dir / "config.json").write_text('{"model_type": "test"}')
            (model_dir / "model.safetensors").write_bytes(b"fake model data")
            
            # Create verification file
            verification_data = {
                "model_id": "test-model",
                "verified_at": time.time(),
                "files": [
                    {"path": "config.json", "size": 20},
                    {"path": "model.safetensors", "size": 15}
                ]
            }
            
            verification_file = model_dir / ".verified.json"
            with open(verification_file, 'w') as f:
                json.dump(verification_data, f)
            
            # Mock dependencies
            mock_registry = Mock()
            mock_registry.list_models.return_value = ["test-model"]
            
            mock_resolver = Mock()
            mock_resolver.local_dir.return_value = str(model_dir)
            
            mock_ensurer = Mock()
            mock_ensurer.status.return_value = ModelStatusInfo(
                status=ModelStatus.COMPLETE,
                local_path=str(model_dir),
                missing_files=[],
                bytes_needed=0
            )
            
            # Create service and test
            service = ModelHealthService(mock_registry, mock_resolver, mock_ensurer)
            
            health_info = await service._get_model_health("test-model")
            
            assert health_info.model_id == "test-model"
            assert health_info.status == "COMPLETE"
            assert health_info.last_verified is not None
            assert health_info.last_verified == verification_data["verified_at"]


if __name__ == "__main__":
    pytest.main([__file__])