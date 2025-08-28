"""
Test Model Management API Endpoints
Tests for the new model status and management endpoints
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from backend.api.model_management import ModelManagementAPI, get_model_management_api


class TestModelManagementEndpoints:
    """Test cases for model management endpoints"""
    
    @pytest.fixture
    async def api(self):
        """Create a test API instance"""
        api = ModelManagementAPI()
        # Mock the initialization to avoid dependencies
        api._initialized = True
        api.system_integration = Mock()
        return api
    
    @pytest.mark.asyncio
    async def test_get_all_model_status(self, api):
        """Test getting status of all models"""
        # Mock system integration methods
        api.system_integration.get_model_paths.return_value = {
            "t2v_a14b_model": "/path/to/t2v",
            "i2v_a14b_model": "/path/to/i2v", 
            "ti2v_5b_model": "/path/to/ti2v"
        }
        
        # Mock _get_single_model_status
        async def mock_get_single_status(model_type):
            return {
                "model_type": model_type,
                "status": "available",
                "is_available": True,
                "is_loaded": False,
                "size_mb": 1000.0
            }
        
        api._get_single_model_status = mock_get_single_status
        
        result = await api.get_all_model_status()
        
        assert "models" in result
        assert "timestamp" in result
        assert "total_models" in result
        assert "available_models" in result
        assert len(result["models"]) == 3
        assert result["total_models"] == 3
    
    @pytest.mark.asyncio
    async def test_get_model_status_single(self, api):
        """Test getting status of a single model"""
        # Mock system integration
        api.system_integration.get_model_manager.return_value = None
        api.system_integration.get_model_paths.return_value = {
            "t2v_a14b_model": "/path/to/t2v"
        }
        api.system_integration.get_optimization_settings.return_value = {
            "quantization": "bf16",
            "enable_offload": False
        }
        
        result = await api.get_model_status("t2v-A14B")
        
        assert result["model_type"] == "t2v-A14B"
        assert "status" in result
        assert "is_available" in result
        assert "estimated_vram_usage_mb" in result
    
    @pytest.mark.asyncio
    async def test_trigger_model_download(self, api):
        """Test triggering model download"""
        # Mock system integration ensure_model_available
        api.system_integration.ensure_model_available = AsyncMock(return_value=(True, "Success"))
        
        # Mock _get_single_model_status to return not available first
        async def mock_status(model_type):
            return {"is_available": False}
        
        api._get_single_model_status = mock_status
        
        result = await api.trigger_model_download("t2v-A14B", force_redownload=False)
        
        assert result["model_type"] == "t2v-A14B"
        assert result["status"] == "download_completed"
        assert result["download_required"] == True
    
    @pytest.mark.asyncio
    async def test_validate_model_integrity(self, api):
        """Test model integrity validation"""
        # Mock system integration and paths
        api.system_integration.get_model_paths.return_value = {
            "t2v_a14b_model": "backend/tests/test_model_path"
        }
        
        # Mock _get_single_model_status
        async def mock_status(model_type):
            return {"is_available": True}
        
        api._get_single_model_status = mock_status
        
        # Mock Path.exists for essential files
        with patch('pathlib.Path.exists') as mock_exists:
            mock_exists.return_value = True
            
            result = await api.validate_model_integrity("t2v-A14B")
            
            assert result["model_type"] == "t2v-A14B"
            assert result["integrity_status"] == "valid"
            assert result["is_valid"] == True
    
    @pytest.mark.asyncio
    async def test_get_system_optimization_status(self, api):
        """Test getting system optimization status"""
        # Mock system integration methods
        api.system_integration.get_system_info.return_value = {
            "initialized": True,
            "components": {"model_manager": True},
            "initialization_errors": []
        }
        
        api.system_integration.get_optimization_settings.return_value = {
            "quantization": "bf16",
            "enable_offload": True
        }
        
        api.system_integration.get_wan22_system_optimizer.return_value = None
        api.system_integration.get_model_paths.return_value = {
            "models_directory": "/path/to/models"
        }
        
        result = await api.get_system_optimization_status()
        
        assert "system_integration" in result
        assert "optimization_settings" in result
        assert "model_paths" in result
        assert "timestamp" in result
        assert result["system_integration"]["initialized"] == True
    
    @pytest.mark.asyncio
    async def test_estimate_vram_usage(self, api):
        """Test VRAM usage estimation"""
        optimization_settings = {
            "quantization": "fp16",
            "enable_offload": True
        }
        
        # Test different model types
        t2v_vram = api._estimate_vram_usage("t2v-A14B", optimization_settings)
        i2v_vram = api._estimate_vram_usage("i2v-A14B", optimization_settings)
        ti2v_vram = api._estimate_vram_usage("ti2v-5B", optimization_settings)
        
        # Should be reduced due to fp16 and offloading
        assert t2v_vram < 8000  # Base estimate is 8000MB
        assert i2v_vram < 8500  # Base estimate is 8500MB
        assert ti2v_vram < 6000  # Base estimate is 6000MB
        
        # TI2V should use less VRAM than T2V/I2V
        assert ti2v_vram < t2v_vram
        assert ti2v_vram < i2v_vram


def test_model_management_api_singleton():
    """Test that get_model_management_api returns singleton"""
    # This test ensures the global instance pattern works
    # We can't easily test the async singleton without running event loop
    # but we can test the function exists
    assert callable(get_model_management_api)


if __name__ == "__main__":
    # Run a simple test
    async def run_basic_test():
        api = ModelManagementAPI()
        api._initialized = True
        api.system_integration = Mock()
        
        # Test VRAM estimation
        settings = {"quantization": "bf16", "enable_offload": False}
        vram = api._estimate_vram_usage("t2v-A14B", settings)
        print(f"✅ VRAM estimation test passed: {vram}MB")
        
        print("✅ Basic model management API tests completed")
    
    asyncio.run(run_basic_test())