"""
Tests for Model Integration Bridge
"""

import pytest
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.core.model_integration_bridge import (
    ModelIntegrationBridge,
    ModelStatus,
    ModelType,
    ModelIntegrationStatus,
    GenerationParams,
    GenerationResult,
    get_model_integration_bridge
)

class TestModelIntegrationBridge:
    """Test cases for ModelIntegrationBridge"""
    
    def test_bridge_instantiation(self):
        """Test that the bridge can be instantiated"""
        bridge = ModelIntegrationBridge()
        assert bridge is not None
        assert not bridge.is_initialized()
    
    def test_model_type_mappings(self):
        """Test model type mappings"""
        bridge = ModelIntegrationBridge()
        
        assert bridge._get_model_type_enum("t2v-A14B") == ModelType.T2V_A14B
        assert bridge._get_model_type_enum("i2v-A14B") == ModelType.I2V_A14B
        assert bridge._get_model_type_enum("ti2v-5B") == ModelType.TI2V_5B
        assert bridge._get_model_type_enum("T2V") == ModelType.T2V_A14B
    
    def test_vram_estimation(self):
        """Test VRAM usage estimation"""
        bridge = ModelIntegrationBridge()
        
        # Test known model types
        assert bridge._estimate_model_vram_usage("t2v-A14B") == 8000
        assert bridge._estimate_model_vram_usage("i2v-A14B") == 8000
        assert bridge._estimate_model_vram_usage("ti2v-5B") == 6000
        
        # Test unknown model type (should default to 8000)
        assert bridge._estimate_model_vram_usage("unknown") == 8000
    
    @pytest.mark.asyncio
    async def test_bridge_initialization(self):
        """Test bridge initialization"""
        bridge = ModelIntegrationBridge()
        
        # Initialize should not fail even if dependencies are missing
        result = await bridge.initialize()
        
        # Should return True or False, not raise exception
        assert isinstance(result, bool)
        
        # Check integration status
        status = bridge.get_integration_status()
        assert "initialized" in status
        assert "model_manager_available" in status
        assert "system_optimizer_available" in status
        assert "model_downloader_available" in status
    
    @pytest.mark.asyncio
    async def test_model_availability_check(self):
        """Test model availability checking"""
        bridge = ModelIntegrationBridge()
        await bridge.initialize()
        
        # Test checking availability for a model
        status = await bridge.check_model_availability("t2v-A14B")
        
        assert isinstance(status, ModelIntegrationStatus)
        assert status.model_type == ModelType.T2V_A14B
        assert isinstance(status.status, ModelStatus)
        assert isinstance(status.is_cached, bool)
        assert isinstance(status.is_loaded, bool)
        assert isinstance(status.is_valid, bool)
        assert isinstance(status.size_mb, float)
    
    @pytest.mark.asyncio
    async def test_generation_params_creation(self):
        """Test generation parameters creation"""
        params = GenerationParams(
            prompt="test prompt",
            model_type="t2v-A14B",
            resolution="1280x720",
            steps=50
        )
        
        assert params.prompt == "test prompt"
        assert params.model_type == "t2v-A14B"
        assert params.resolution == "1280x720"
        assert params.steps == 50
        assert params.lora_strength == 1.0  # default value
    
    @pytest.mark.asyncio
    async def test_generation_result_creation(self):
        """Test generation result creation"""
        result = GenerationResult(
            success=True,
            task_id="test_task",
            model_used="t2v-A14B"
        )
        
        assert result.success is True
        assert result.task_id == "test_task"
        assert result.model_used == "t2v-A14B"
        assert result.parameters_used == {}  # default empty dict
        assert result.optimizations_applied == []  # default empty list
        assert result.recovery_suggestions == []  # default empty list
    
    @pytest.mark.asyncio
    async def test_global_bridge_instance(self):
        """Test global bridge instance"""
        bridge1 = await get_model_integration_bridge()
        bridge2 = await get_model_integration_bridge()
        
        # Should return the same instance
        assert bridge1 is bridge2
        assert bridge1.is_initialized()

if __name__ == "__main__":
    # Run basic tests
    bridge = ModelIntegrationBridge()
    print("✅ Bridge instantiation test passed")
    
    # Test model type mappings
    assert bridge._get_model_type_enum("t2v-A14B") == ModelType.T2V_A14B
    print("✅ Model type mapping test passed")
    
    # Test VRAM estimation
    assert bridge._estimate_model_vram_usage("t2v-A14B") == 8000
    print("✅ VRAM estimation test passed")
    
    print("✅ All basic tests passed!")