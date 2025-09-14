"""
Tests for Model Health API endpoints
"""

import pytest
import time
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import FastAPI

from backend.api.v1.endpoints.models import router
from backend.services.model_health_service import (
    ModelHealthService, ModelHealthInfo, OrchestratorHealthResponse
)


@pytest.fixture
def app():
    """Create FastAPI app with models router."""
    app = FastAPI()
    app.include_router(router, prefix="/api/v1/models")
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_health_service():
    """Mock health service."""
    service = Mock(spec=ModelHealthService)
    return service


class TestModelsHealthEndpoints:
    """Test cases for model health API endpoints."""
    
    @patch('backend.api.v1.endpoints.models.get_model_health_service')
    def test_get_models_health_success(self, mock_get_service, client, mock_health_service):
        """Test successful health check for all models."""
        # Setup mock service
        mock_get_service.return_value = mock_health_service
        
        # Create mock response
        model_health = ModelHealthInfo(
            model_id="t2v-A14B",
            variant=None,
            status="COMPLETE",
            local_path="/models/t2v-A14B",
            missing_files=[],
            bytes_needed=0,
            last_verified=time.time()
        )
        
        health_response = OrchestratorHealthResponse(
            status="healthy",
            timestamp=time.time(),
            models={"t2v-A14B": model_health},
            total_models=1,
            healthy_models=1,
            missing_models=0,
            partial_models=0,
            corrupt_models=0,
            total_bytes_needed=0,
            response_time_ms=45.0
        )
        
        mock_health_service.get_health_status = AsyncMock(return_value=health_response)
        
        # Make request
        response = client.get("/api/v1/models/health?dry_run=true")
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert data["total_models"] == 1
        assert data["healthy_models"] == 1
        assert data["response_time_ms"] == 45.0
        assert "t2v-A14B" in data["models"]
        assert data["models"]["t2v-A14B"]["status"] == "COMPLETE"
        
        # Verify service was called with correct parameters
        mock_health_service.get_health_status.assert_called_once_with(dry_run=True)
    
    @patch('backend.api.v1.endpoints.models.get_model_health_service')
    def test_get_models_health_degraded_state(self, mock_get_service, client, mock_health_service):
        """Test health check with models in degraded state."""
        mock_get_service.return_value = mock_health_service
        
        # Create mock response with mixed states
        complete_model = ModelHealthInfo(
            model_id="t2v-A14B",
            variant=None,
            status="COMPLETE",
            local_path="/models/t2v-A14B",
            missing_files=[],
            bytes_needed=0
        )
        
        missing_model = ModelHealthInfo(
            model_id="i2v-A14B",
            variant=None,
            status="NOT_PRESENT",
            local_path="/models/i2v-A14B",
            missing_files=["model.safetensors", "config.json"],
            bytes_needed=5000000000
        )
        
        health_response = OrchestratorHealthResponse(
            status="degraded",
            timestamp=time.time(),
            models={"t2v-A14B": complete_model, "i2v-A14B": missing_model},
            total_models=2,
            healthy_models=1,
            missing_models=1,
            partial_models=0,
            corrupt_models=0,
            total_bytes_needed=5000000000,
            response_time_ms=78.0
        )
        
        mock_health_service.get_health_status = AsyncMock(return_value=health_response)
        
        # Make request
        response = client.get("/api/v1/models/health")
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "degraded"
        assert data["total_models"] == 2
        assert data["healthy_models"] == 1
        assert data["missing_models"] == 1
        assert data["total_bytes_needed"] == 5000000000
        
        assert data["models"]["t2v-A14B"]["status"] == "COMPLETE"
        assert data["models"]["i2v-A14B"]["status"] == "NOT_PRESENT"
        assert len(data["models"]["i2v-A14B"]["missing_files"]) == 2
    
    @patch('backend.api.v1.endpoints.models.get_model_health_service')
    def test_get_models_health_service_not_initialized(self, mock_get_service, client):
        """Test health check when service is not initialized."""
        mock_get_service.return_value = None
        
        response = client.get("/api/v1/models/health")
        
        assert response.status_code == 503
        assert "not initialized" in response.json()["detail"]
    
    @patch('backend.api.v1.endpoints.models.get_model_health_service')
    def test_get_models_health_service_error(self, mock_get_service, client, mock_health_service):
        """Test health check when service raises an error."""
        mock_get_service.return_value = mock_health_service
        mock_health_service.get_health_status = AsyncMock(side_effect=Exception("Service error"))
        
        response = client.get("/api/v1/models/health")
        
        assert response.status_code == 500
        assert "Service error" in response.json()["detail"]
    
    @patch('backend.api.v1.endpoints.models.get_model_health_service')
    def test_get_models_health_dry_run_parameter(self, mock_get_service, client, mock_health_service):
        """Test that dry_run parameter is properly passed."""
        mock_get_service.return_value = mock_health_service
        
        health_response = OrchestratorHealthResponse(
            status="healthy",
            timestamp=time.time(),
            models={},
            total_models=0,
            healthy_models=0,
            missing_models=0,
            partial_models=0,
            corrupt_models=0,
            total_bytes_needed=0,
            response_time_ms=10.0
        )
        
        mock_health_service.get_health_status = AsyncMock(return_value=health_response)
        
        # Test with dry_run=false
        response = client.get("/api/v1/models/health?dry_run=false")
        
        assert response.status_code == 200
        mock_health_service.get_health_status.assert_called_once_with(dry_run=False)
    
    @patch('backend.api.v1.endpoints.models.get_model_health_service')
    def test_get_model_health_individual_success(self, mock_get_service, client, mock_health_service):
        """Test successful health check for individual model."""
        mock_get_service.return_value = mock_health_service
        
        model_health = ModelHealthInfo(
            model_id="t2v-A14B",
            variant="fp16",
            status="COMPLETE",
            local_path="/models/t2v-A14B",
            missing_files=[],
            bytes_needed=0,
            last_verified=time.time() - 3600
        )
        
        mock_health_service.get_model_health = AsyncMock(return_value=model_health)
        
        # Make request
        response = client.get("/api/v1/models/health/t2v-A14B?variant=fp16")
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        
        assert data["model_id"] == "t2v-A14B"
        assert data["variant"] == "fp16"
        assert data["status"] == "COMPLETE"
        assert data["bytes_needed"] == 0
        assert data["last_verified"] is not None
        
        # Verify service was called with correct parameters
        mock_health_service.get_model_health.assert_called_once_with("t2v-A14B", "fp16")
    
    @patch('backend.api.v1.endpoints.models.get_model_health_service')
    def test_get_model_health_individual_missing(self, mock_get_service, client, mock_health_service):
        """Test health check for missing individual model."""
        mock_get_service.return_value = mock_health_service
        
        model_health = ModelHealthInfo(
            model_id="missing-model",
            variant=None,
            status="NOT_PRESENT",
            local_path="/models/missing-model",
            missing_files=["config.json", "model.safetensors"],
            bytes_needed=8000000000,
            error_message=None
        )
        
        mock_health_service.get_model_health = AsyncMock(return_value=model_health)
        
        # Make request
        response = client.get("/api/v1/models/health/missing-model")
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        
        assert data["model_id"] == "missing-model"
        assert data["status"] == "NOT_PRESENT"
        assert data["bytes_needed"] == 8000000000
        assert len(data["missing_files"]) == 2
        assert "config.json" in data["missing_files"]
        assert "model.safetensors" in data["missing_files"]
    
    @patch('backend.api.v1.endpoints.models.get_model_health_service')
    def test_get_model_health_individual_error(self, mock_get_service, client, mock_health_service):
        """Test individual model health check with error."""
        mock_get_service.return_value = mock_health_service
        
        model_health = ModelHealthInfo(
            model_id="error-model",
            variant=None,
            status="ERROR",
            local_path=None,
            missing_files=[],
            bytes_needed=0,
            error_message="Model not found in registry"
        )
        
        mock_health_service.get_model_health = AsyncMock(return_value=model_health)
        
        # Make request
        response = client.get("/api/v1/models/health/error-model")
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        
        assert data["model_id"] == "error-model"
        assert data["status"] == "ERROR"
        assert data["error_message"] == "Model not found in registry"
    
    @patch('backend.api.v1.endpoints.models.get_model_health_service')
    def test_get_model_health_individual_service_not_initialized(self, mock_get_service, client):
        """Test individual model health when service is not initialized."""
        mock_get_service.return_value = None
        
        response = client.get("/api/v1/models/health/test-model")
        
        assert response.status_code == 503
        assert "not initialized" in response.json()["detail"]
    
    @patch('backend.api.v1.endpoints.models.get_model_health_service')
    def test_get_model_health_individual_service_error(self, mock_get_service, client, mock_health_service):
        """Test individual model health when service raises an error."""
        mock_get_service.return_value = mock_health_service
        mock_health_service.get_model_health = AsyncMock(side_effect=Exception("Service error"))
        
        response = client.get("/api/v1/models/health/test-model")
        
        assert response.status_code == 500
        assert "Service error" in response.json()["detail"]
    
    def test_health_endpoint_response_time_requirement(self, client):
        """Test that health endpoint meets response time requirements."""
        # This is more of a performance test that would need real implementation
        # For now, we just verify the endpoint structure supports fast responses
        
        with patch('backend.api.v1.endpoints.models.get_model_health_service') as mock_get_service:
            mock_service = Mock()
            mock_get_service.return_value = mock_service
            
            # Mock a fast response
            fast_response = OrchestratorHealthResponse(
                status="healthy",
                timestamp=time.time(),
                models={},
                total_models=0,
                healthy_models=0,
                missing_models=0,
                partial_models=0,
                corrupt_models=0,
                total_bytes_needed=0,
                response_time_ms=45.0  # Under 100ms requirement
            )
            
            mock_service.get_health_status = AsyncMock(return_value=fast_response)
            
            response = client.get("/api/v1/models/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["response_time_ms"] < 100.0  # Meets requirement
    
    def test_health_endpoint_supports_json_output(self, client):
        """Test that health endpoint returns proper JSON structure."""
        with patch('backend.api.v1.endpoints.models.get_model_health_service') as mock_get_service:
            mock_service = Mock()
            mock_get_service.return_value = mock_service
            
            model_health = ModelHealthInfo(
                model_id="test",
                variant=None,
                status="COMPLETE",
                local_path="/models/test",
                missing_files=[],
                bytes_needed=0
            )
            
            health_response = OrchestratorHealthResponse(
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
            
            mock_service.get_health_status = AsyncMock(return_value=health_response)
            
            response = client.get("/api/v1/models/health")
            
            assert response.status_code == 200
            assert response.headers["content-type"] == "application/json"
            
            data = response.json()
            
            # Verify JSON structure matches requirements
            required_fields = [
                "status", "timestamp", "models", "total_models",
                "healthy_models", "missing_models", "partial_models",
                "corrupt_models", "total_bytes_needed", "response_time_ms"
            ]
            
            for field in required_fields:
                assert field in data, f"Missing required field: {field}"
            
            # Verify model structure
            assert "test" in data["models"]
            model_data = data["models"]["test"]
            
            model_required_fields = [
                "model_id", "status", "missing_files", "bytes_needed"
            ]
            
            for field in model_required_fields:
                assert field in model_data, f"Missing required model field: {field}"


if __name__ == "__main__":
    pytest.main([__file__])