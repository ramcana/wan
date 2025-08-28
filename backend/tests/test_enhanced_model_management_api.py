"""
Integration Tests for Enhanced Model Management API Endpoints
Tests all new enhanced model management endpoints with various scenarios.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import HTTPException

# Import the FastAPI app
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from app import app
from api.enhanced_model_management import (
    EnhancedModelManagementAPI, DownloadControlRequest, CleanupRequest, FallbackSuggestionRequest
)
from backend.core.model_availability_manager import ModelAvailabilityStatus
from backend.core.enhanced_model_downloader import DownloadStatus
from backend.core.model_health_monitor import HealthStatus


class TestEnhancedModelManagementAPI:
    """Test suite for Enhanced Model Management API"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    @pytest.fixture
    def mock_api(self):
        """Create mock enhanced API"""
        api = Mock(spec=EnhancedModelManagementAPI)
        api._initialized = True
        return api
    
    @pytest.fixture
    def sample_detailed_status(self):
        """Sample detailed model status response"""
        return {
            "models": {
                "T2V-A14B": {
                    "model_id": "T2V-A14B",
                    "availability_status": "available",
                    "is_available": True,
                    "is_loaded": False,
                    "size_mb": 8192.5,
                    "download_progress": None,
                    "missing_files": [],
                    "integrity_score": 1.0,
                    "last_health_check": "2024-01-15T10:30:00",
                    "performance_score": 0.95,
                    "corruption_detected": False,
                    "usage_frequency": 2.5,
                    "last_used": "2024-01-15T09:00:00",
                    "average_generation_time": 45.2,
                    "can_pause_download": False,
                    "can_resume_download": False,
                    "estimated_download_time": None,
                    "current_version": "1.0.0",
                    "latest_version": "1.0.0",
                    "update_available": False
                },
                "I2V-A14B": {
                    "model_id": "I2V-A14B",
                    "availability_status": "downloading",
                    "is_available": False,
                    "is_loaded": False,
                    "size_mb": 0.0,
                    "download_progress": 45.5,
                    "missing_files": ["model.safetensors"],
                    "integrity_score": 0.0,
                    "last_health_check": None,
                    "performance_score": 0.0,
                    "corruption_detected": False,
                    "usage_frequency": 0.0,
                    "last_used": None,
                    "average_generation_time": None,
                    "can_pause_download": True,
                    "can_resume_download": False,
                    "estimated_download_time": "00:15:30",
                    "current_version": "",
                    "latest_version": "1.0.0",
                    "update_available": False
                }
            },
            "system_statistics": {
                "total_models": 2,
                "available_models": 1,
                "downloading_models": 1,
                "corrupted_models": 0,
                "total_size_gb": 8.0,
                "last_updated": "2024-01-15T10:30:00"
            },
            "timestamp": "2024-01-15T10:30:00"
        }
    
    @pytest.fixture
    def sample_health_data(self):
        """Sample health monitoring response"""
        return {
            "system_health": {
                "overall_health_score": 0.85,
                "models_healthy": 1,
                "models_degraded": 0,
                "models_corrupted": 0,
                "storage_usage_percent": 65.5,
                "last_updated": "2024-01-15T10:30:00"
            },
            "model_health": {
                "T2V-A14B": {
                    "model_id": "T2V-A14B",
                    "health_status": "healthy",
                    "is_healthy": True,
                    "integrity_score": 1.0,
                    "issues": [],
                    "corruption_types": [],
                    "last_check": "2024-01-15T10:30:00",
                    "repair_suggestions": [],
                    "can_auto_repair": False
                }
            },
            "recommendations": [],
            "timestamp": "2024-01-15T10:30:00"
        }
    
    @pytest.fixture
    def sample_analytics_data(self):
        """Sample analytics response"""
        return {
            "time_period": {
                "start_date": "2023-12-16T10:30:00",
                "end_date": "2024-01-15T10:30:00",
                "days": 30
            },
            "system_analytics": {
                "total_uses": 75,
                "average_uses_per_day": 2.5,
                "most_used_model": "T2V-A14B",
                "active_models": 1
            },
            "model_analytics": {
                "T2V-A14B": {
                    "model_id": "T2V-A14B",
                    "total_uses": 75,
                    "uses_per_day": 2.5,
                    "average_generation_time": 45.2,
                    "success_rate": 0.96,
                    "last_30_days_usage": [],
                    "peak_usage_hours": [14, 15, 16]
                }
            },
            "timestamp": "2024-01-15T10:30:00"
        }


class TestDetailedModelStatusEndpoint(TestEnhancedModelManagementAPI):
    """Test /api/v1/models/status/detailed endpoint"""
    
    @patch('api.enhanced_model_management.get_enhanced_model_management_api')
    def test_get_detailed_status_success(self, mock_get_api, client, mock_api, sample_detailed_status):
        """Test successful detailed status retrieval"""
        mock_get_api.return_value = asyncio.create_task(asyncio.coroutine(lambda: mock_api)())
        mock_api.get_detailed_model_status.return_value = asyncio.create_task(
            asyncio.coroutine(lambda: sample_detailed_status)()
        )
        
        response = client.get("/api/v1/models/status/detailed")
        
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert "system_statistics" in data
        assert "timestamp" in data
        assert len(data["models"]) == 2
        assert data["system_statistics"]["total_models"] == 2
        assert data["system_statistics"]["available_models"] == 1
    
    @patch('api.enhanced_model_management.get_enhanced_model_management_api')
    def test_get_detailed_status_api_error(self, mock_get_api, client):
        """Test API error handling"""
        mock_api = Mock()
        mock_api.get_detailed_model_status.side_effect = Exception("API Error")
        mock_get_api.return_value = asyncio.create_task(asyncio.coroutine(lambda: mock_api)())
        
        response = client.get("/api/v1/models/status/detailed")
        
        assert response.status_code == 500
        assert "Failed to get detailed model status" in response.json()["detail"]
    
    def test_detailed_status_model_fields(self, client, sample_detailed_status):
        """Test that all required model fields are present"""
        with patch('api.enhanced_model_management.get_enhanced_model_management_api') as mock_get_api:
            mock_api = Mock()
            mock_api.get_detailed_model_status.return_value = asyncio.create_task(
                asyncio.coroutine(lambda: sample_detailed_status)()
            )
            mock_get_api.return_value = asyncio.create_task(asyncio.coroutine(lambda: mock_api)())
            
            response = client.get("/api/v1/models/status/detailed")
            
            assert response.status_code == 200
            data = response.json()
            
            # Check T2V-A14B model has all required fields
            t2v_model = data["models"]["T2V-A14B"]
            required_fields = [
                "model_id", "availability_status", "is_available", "is_loaded",
                "size_mb", "integrity_score", "performance_score", "usage_frequency"
            ]
            
            for field in required_fields:
                assert field in t2v_model, f"Missing required field: {field}"


class TestDownloadManagementEndpoint(TestEnhancedModelManagementAPI):
    """Test /api/v1/models/download/manage endpoint"""
    
    @patch('api.enhanced_model_management.get_enhanced_model_management_api')
    def test_pause_download_success(self, mock_get_api, client, mock_api):
        """Test successful download pause"""
        mock_get_api.return_value = asyncio.create_task(asyncio.coroutine(lambda: mock_api)())
        mock_api.manage_download.return_value = asyncio.create_task(asyncio.coroutine(lambda: {
            "success": True,
            "message": "Download paused for I2V-A14B",
            "model_id": "I2V-A14B",
            "action": "pause",
            "current_status": "paused",
            "progress_percent": 45.5,
            "timestamp": "2024-01-15T10:30:00"
        })())
        
        request_data = {
            "model_id": "I2V-A14B",
            "action": "pause"
        }
        
        response = client.post("/api/v1/models/download/manage", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["action"] == "pause"
        assert data["model_id"] == "I2V-A14B"
        assert data["current_status"] == "paused"
    
    @patch('api.enhanced_model_management.get_enhanced_model_management_api')
    def test_resume_download_success(self, mock_get_api, client, mock_api):
        """Test successful download resume"""
        mock_get_api.return_value = asyncio.create_task(asyncio.coroutine(lambda: mock_api)())
        mock_api.manage_download.return_value = asyncio.create_task(asyncio.coroutine(lambda: {
            "success": True,
            "message": "Download resumed for I2V-A14B",
            "model_id": "I2V-A14B",
            "action": "resume",
            "current_status": "downloading",
            "progress_percent": 45.5,
            "timestamp": "2024-01-15T10:30:00"
        })())
        
        request_data = {
            "model_id": "I2V-A14B",
            "action": "resume"
        }
        
        response = client.post("/api/v1/models/download/manage", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["action"] == "resume"
        assert data["current_status"] == "downloading"
    
    @patch('api.enhanced_model_management.get_enhanced_model_management_api')
    def test_set_priority_success(self, mock_get_api, client, mock_api):
        """Test successful priority setting"""
        mock_get_api.return_value = asyncio.create_task(asyncio.coroutine(lambda: mock_api)())
        mock_api.manage_download.return_value = asyncio.create_task(asyncio.coroutine(lambda: {
            "success": True,
            "message": "Priority set to HIGH for I2V-A14B",
            "model_id": "I2V-A14B",
            "action": "priority",
            "current_status": "downloading",
            "progress_percent": 45.5,
            "timestamp": "2024-01-15T10:30:00"
        })())
        
        request_data = {
            "model_id": "I2V-A14B",
            "action": "priority",
            "priority": "high"
        }
        
        response = client.post("/api/v1/models/download/manage", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["action"] == "priority"
    
    def test_invalid_action(self, client):
        """Test invalid action handling"""
        request_data = {
            "model_id": "I2V-A14B",
            "action": "invalid_action"
        }
        
        response = client.post("/api/v1/models/download/manage", json=request_data)
        
        assert response.status_code == 500  # Will be caught by exception handler
    
    def test_missing_required_fields(self, client):
        """Test missing required fields validation"""
        request_data = {
            "action": "pause"
            # Missing model_id
        }
        
        response = client.post("/api/v1/models/download/manage", json=request_data)
        
        assert response.status_code == 500  # Pydantic validation error


class TestHealthMonitoringEndpoint(TestEnhancedModelManagementAPI):
    """Test /api/v1/models/health endpoint"""
    
    @patch('api.enhanced_model_management.get_enhanced_model_management_api')
    def test_get_health_data_success(self, mock_get_api, client, mock_api, sample_health_data):
        """Test successful health data retrieval"""
        mock_get_api.return_value = asyncio.create_task(asyncio.coroutine(lambda: mock_api)())
        mock_api.get_health_monitoring_data.return_value = asyncio.create_task(
            asyncio.coroutine(lambda: sample_health_data)()
        )
        
        response = client.get("/api/v1/models/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "system_health" in data
        assert "model_health" in data
        assert "recommendations" in data
        assert data["system_health"]["overall_health_score"] == 0.85
        assert data["system_health"]["models_healthy"] == 1
    
    @patch('api.enhanced_model_management.get_enhanced_model_management_api')
    def test_health_data_model_details(self, mock_get_api, client, mock_api, sample_health_data):
        """Test health data includes model-specific details"""
        mock_get_api.return_value = asyncio.create_task(asyncio.coroutine(lambda: mock_api)())
        mock_api.get_health_monitoring_data.return_value = asyncio.create_task(
            asyncio.coroutine(lambda: sample_health_data)()
        )
        
        response = client.get("/api/v1/models/health")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check T2V-A14B health details
        t2v_health = data["model_health"]["T2V-A14B"]
        assert t2v_health["health_status"] == "healthy"
        assert t2v_health["is_healthy"] is True
        assert t2v_health["integrity_score"] == 1.0
        assert isinstance(t2v_health["issues"], list)
        assert isinstance(t2v_health["corruption_types"], list)


class TestUsageAnalyticsEndpoint(TestEnhancedModelManagementAPI):
    """Test /api/v1/models/analytics endpoint"""
    
    @patch('api.enhanced_model_management.get_enhanced_model_management_api')
    def test_get_analytics_success(self, mock_get_api, client, mock_api, sample_analytics_data):
        """Test successful analytics retrieval"""
        mock_get_api.return_value = asyncio.create_task(asyncio.coroutine(lambda: mock_api)())
        mock_api.get_usage_analytics.return_value = asyncio.create_task(
            asyncio.coroutine(lambda: sample_analytics_data)()
        )
        
        response = client.get("/api/v1/models/analytics")
        
        assert response.status_code == 200
        data = response.json()
        assert "time_period" in data
        assert "system_analytics" in data
        assert "model_analytics" in data
        assert data["system_analytics"]["total_uses"] == 75
        assert data["system_analytics"]["most_used_model"] == "T2V-A14B"
    
    @patch('api.enhanced_model_management.get_enhanced_model_management_api')
    def test_get_analytics_custom_period(self, mock_get_api, client, mock_api, sample_analytics_data):
        """Test analytics with custom time period"""
        mock_get_api.return_value = asyncio.create_task(asyncio.coroutine(lambda: mock_api)())
        mock_api.get_usage_analytics.return_value = asyncio.create_task(
            asyncio.coroutine(lambda: sample_analytics_data)()
        )
        
        response = client.get("/api/v1/models/analytics?time_period_days=7")
        
        assert response.status_code == 200
        # Verify the API was called with the correct parameter
        mock_api.get_usage_analytics.assert_called_with(7)
    
    def test_analytics_model_details(self, client, sample_analytics_data):
        """Test analytics includes detailed model statistics"""
        with patch('api.enhanced_model_management.get_enhanced_model_management_api') as mock_get_api:
            mock_api = Mock()
            mock_api.get_usage_analytics.return_value = asyncio.create_task(
                asyncio.coroutine(lambda: sample_analytics_data)()
            )
            mock_get_api.return_value = asyncio.create_task(asyncio.coroutine(lambda: mock_api)())
            
            response = client.get("/api/v1/models/analytics")
            
            assert response.status_code == 200
            data = response.json()
            
            # Check T2V-A14B analytics
            t2v_analytics = data["model_analytics"]["T2V-A14B"]
            assert t2v_analytics["total_uses"] == 75
            assert t2v_analytics["uses_per_day"] == 2.5
            assert t2v_analytics["success_rate"] == 0.96
            assert isinstance(t2v_analytics["peak_usage_hours"], list)


class TestStorageCleanupEndpoint(TestEnhancedModelManagementAPI):
    """Test /api/v1/models/cleanup endpoint"""
    
    @patch('api.enhanced_model_management.get_enhanced_model_management_api')
    def test_cleanup_dry_run_success(self, mock_get_api, client, mock_api):
        """Test successful cleanup dry run"""
        mock_get_api.return_value = asyncio.create_task(asyncio.coroutine(lambda: mock_api)())
        mock_api.manage_storage_cleanup.return_value = asyncio.create_task(asyncio.coroutine(lambda: {
            "dry_run": True,
            "recommendations": {
                "total_space_available_gb": 50.0,
                "target_space_gb": 20.0,
                "space_to_free_gb": 10.0,
                "cleanup_actions": [
                    {
                        "action_type": "remove_model",
                        "model_id": "TI2V-5B",
                        "space_freed_gb": 6.0,
                        "reason": "Unused for 60 days",
                        "last_used": "2023-11-15T10:30:00"
                    }
                ]
            },
            "executed_actions": [],
            "total_space_freed_gb": 0.0,
            "timestamp": "2024-01-15T10:30:00"
        })())
        
        request_data = {
            "target_space_gb": 20.0,
            "keep_recent_days": 30,
            "dry_run": True
        }
        
        response = client.post("/api/v1/models/cleanup", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["dry_run"] is True
        assert "recommendations" in data
        assert len(data["recommendations"]["cleanup_actions"]) == 1
        assert data["total_space_freed_gb"] == 0.0
    
    @patch('api.enhanced_model_management.get_enhanced_model_management_api')
    def test_cleanup_execute_success(self, mock_get_api, client, mock_api):
        """Test successful cleanup execution"""
        mock_get_api.return_value = asyncio.create_task(asyncio.coroutine(lambda: mock_api)())
        mock_api.manage_storage_cleanup.return_value = asyncio.create_task(asyncio.coroutine(lambda: {
            "dry_run": False,
            "recommendations": {
                "total_space_available_gb": 50.0,
                "target_space_gb": 20.0,
                "space_to_free_gb": 10.0,
                "cleanup_actions": []
            },
            "executed_actions": [
                {
                    "action": "removed_model",
                    "model_id": "TI2V-5B",
                    "space_freed_gb": 6.0,
                    "success": True
                }
            ],
            "total_space_freed_gb": 6.0,
            "timestamp": "2024-01-15T10:30:00"
        })())
        
        request_data = {
            "target_space_gb": 20.0,
            "keep_recent_days": 30,
            "dry_run": False
        }
        
        response = client.post("/api/v1/models/cleanup", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["dry_run"] is False
        assert len(data["executed_actions"]) == 1
        assert data["total_space_freed_gb"] == 6.0
        assert data["executed_actions"][0]["success"] is True


class TestFallbackSuggestionEndpoint(TestEnhancedModelManagementAPI):
    """Test /api/v1/models/fallback/suggest endpoint"""
    
    @patch('api.enhanced_model_management.get_enhanced_model_management_api')
    def test_suggest_alternatives_success(self, mock_get_api, client, mock_api):
        """Test successful fallback suggestion"""
        mock_get_api.return_value = asyncio.create_task(asyncio.coroutine(lambda: mock_api)())
        mock_api.suggest_fallback_alternatives.return_value = asyncio.create_task(asyncio.coroutine(lambda: {
            "requested_model": "T2V-A14B",
            "alternative_suggestion": {
                "suggested_model": "I2V-A14B",
                "compatibility_score": 0.85,
                "performance_difference": -0.1,
                "availability_status": "available",
                "reason": "Similar capabilities with image input support",
                "estimated_quality_difference": "slightly_lower"
            },
            "fallback_strategy": {
                "strategy_type": "alternative_model",
                "recommended_action": "Use I2V-A14B as alternative",
                "alternative_model": "I2V-A14B",
                "estimated_wait_time": None,
                "user_message": "I2V-A14B is available and provides similar functionality",
                "can_queue_request": False
            },
            "wait_time_estimate": None,
            "timestamp": "2024-01-15T10:30:00"
        })())
        
        request_data = {
            "requested_model": "T2V-A14B",
            "quality": "high",
            "speed": "medium",
            "resolution": "1280x720"
        }
        
        response = client.post("/api/v1/models/fallback/suggest", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["requested_model"] == "T2V-A14B"
        assert data["alternative_suggestion"]["suggested_model"] == "I2V-A14B"
        assert data["alternative_suggestion"]["compatibility_score"] == 0.85
        assert data["fallback_strategy"]["strategy_type"] == "alternative_model"
    
    @patch('api.enhanced_model_management.get_enhanced_model_management_api')
    def test_suggest_with_wait_time(self, mock_get_api, client, mock_api):
        """Test fallback suggestion with wait time estimate"""
        mock_get_api.return_value = asyncio.create_task(asyncio.coroutine(lambda: mock_api)())
        mock_api.suggest_fallback_alternatives.return_value = asyncio.create_task(asyncio.coroutine(lambda: {
            "requested_model": "I2V-A14B",
            "alternative_suggestion": None,
            "fallback_strategy": {
                "strategy_type": "queue_and_wait",
                "recommended_action": "Queue request and wait for download",
                "alternative_model": None,
                "estimated_wait_time": "00:15:30",
                "user_message": "I2V-A14B is downloading, estimated completion in 15 minutes",
                "can_queue_request": True
            },
            "wait_time_estimate": {
                "estimated_minutes": 15.5,
                "confidence": "medium"
            },
            "timestamp": "2024-01-15T10:30:00"
        })())
        
        request_data = {
            "requested_model": "I2V-A14B",
            "quality": "medium",
            "speed": "medium",
            "resolution": "1280x720",
            "max_wait_minutes": 30
        }
        
        response = client.post("/api/v1/models/fallback/suggest", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["fallback_strategy"]["strategy_type"] == "queue_and_wait"
        assert data["fallback_strategy"]["can_queue_request"] is True
        assert data["wait_time_estimate"]["estimated_minutes"] == 15.5
    
    def test_suggest_missing_required_field(self, client):
        """Test missing required field validation"""
        request_data = {
            "quality": "high",
            "speed": "medium"
            # Missing requested_model
        }
        
        response = client.post("/api/v1/models/fallback/suggest", json=request_data)
        
        assert response.status_code == 500  # Pydantic validation error


class TestAPIIntegration(TestEnhancedModelManagementAPI):
    """Test API integration and error handling"""
    
    @patch('api.enhanced_model_management.get_enhanced_model_management_api')
    def test_api_initialization_error(self, mock_get_api, client):
        """Test API initialization error handling"""
        mock_get_api.side_effect = Exception("Initialization failed")
        
        response = client.get("/api/v1/models/status/detailed")
        
        assert response.status_code == 500
        assert "Failed to get detailed model status" in response.json()["detail"]
    
    @patch('api.enhanced_model_management.get_enhanced_model_management_api')
    def test_concurrent_requests(self, mock_get_api, client, mock_api, sample_detailed_status):
        """Test handling of concurrent requests"""
        mock_get_api.return_value = asyncio.create_task(asyncio.coroutine(lambda: mock_api)())
        mock_api.get_detailed_model_status.return_value = asyncio.create_task(
            asyncio.coroutine(lambda: sample_detailed_status)()
        )
        
        # Make multiple concurrent requests
        responses = []
        for _ in range(3):
            response = client.get("/api/v1/models/status/detailed")
            responses.append(response)
        
        # All should succeed
        for response in responses:
            assert response.status_code == 200
            data = response.json()
            assert "models" in data
    
    def test_endpoint_response_format_consistency(self, client):
        """Test that all endpoints return consistent response formats"""
        endpoints_to_test = [
            ("/api/v1/models/status/detailed", "GET", None),
            ("/api/v1/models/health", "GET", None),
            ("/api/v1/models/analytics", "GET", None)
        ]
        
        with patch('api.enhanced_model_management.get_enhanced_model_management_api') as mock_get_api:
            mock_api = Mock()
            mock_api.get_detailed_model_status.return_value = asyncio.create_task(
                asyncio.coroutine(lambda: {"timestamp": "2024-01-15T10:30:00"})()
            )
            mock_api.get_health_monitoring_data.return_value = asyncio.create_task(
                asyncio.coroutine(lambda: {"timestamp": "2024-01-15T10:30:00"})()
            )
            mock_api.get_usage_analytics.return_value = asyncio.create_task(
                asyncio.coroutine(lambda: {"timestamp": "2024-01-15T10:30:00"})()
            )
            mock_get_api.return_value = asyncio.create_task(asyncio.coroutine(lambda: mock_api)())
            
            for endpoint, method, data in endpoints_to_test:
                if method == "GET":
                    response = client.get(endpoint)
                else:
                    response = client.post(endpoint, json=data)
                
                assert response.status_code == 200
                response_data = response.json()
                assert "timestamp" in response_data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])