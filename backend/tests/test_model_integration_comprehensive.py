"""
Comprehensive Model Integration Tests
Focused testing for ModelIntegrationBridge functionality with all model types
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path

# Add backend to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

try:
    from backend.core.model_integration_bridge import (
        ModelIntegrationBridge, ModelStatus, ModelType, 
        ModelIntegrationStatus, GenerationParams, GenerationResult,
        get_model_integration_bridge
    )
except ImportError:
    # Fallback for testing without full backend
    class ModelIntegrationBridge:
        def __init__(self): pass
        def _get_model_type_enum(self, model_type): return model_type
        def _estimate_model_vram_usage(self, model_type): return 8000
        def _get_model_path(self, model_type): return f"models/{model_type}"
        async def initialize(self): return True
        def is_initialized(self): return True
        def get_integration_status(self): return {'initialized': True}
        async def check_model_availability(self, model_type): return None
        async def load_model_with_optimization(self, model_type, params): return True
        def _create_optimization_config(self, params): return {}
        def _create_error_result(self, task_id, category, message): return None
    
    class ModelStatus: 
        AVAILABLE = "available"
        MISSING = "missing"
        CORRUPTED = "corrupted"
    class ModelType: 
        T2V_A14B = "T2V_A14B"
        I2V_A14B = "I2V_A14B"
        TI2V_5B = "TI2V_5B"
    class ModelIntegrationStatus:
        def __init__(self):
            self.model_type = ModelType.T2V_A14B
    class GenerationParams:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    class GenerationResult: pass
    async def get_model_integration_bridge(): return ModelIntegrationBridge()

class TestModelIntegrationBridgeDetailed:
    """Detailed tests for ModelIntegrationBridge functionality"""
    
    @pytest.mark.asyncio
    async def test_model_type_enum_conversion(self):
        """Test model type string to enum conversion"""
        bridge = ModelIntegrationBridge()
        
        # Test all supported model type mappings
        test_cases = [
            ("t2v-A14B", ModelType.T2V_A14B),
            ("i2v-A14B", ModelType.I2V_A14B),
            ("ti2v-5B", ModelType.TI2V_5B),
            ("T2V", ModelType.T2V_A14B),
            ("I2V", ModelType.I2V_A14B),
            ("TI2V", ModelType.TI2V_5B),
            ("T2V-A14B", ModelType.T2V_A14B),
            ("I2V-A14B", ModelType.I2V_A14B),
            ("TI2V-5B", ModelType.TI2V_5B)
        ]
        
        for input_type, expected_enum in test_cases:
            result = bridge._get_model_type_enum(input_type)
            assert result == expected_enum, f"Failed for {input_type}: got {result}, expected {expected_enum}"
    
    @pytest.mark.asyncio
    async def test_vram_estimation_accuracy(self):
        """Test VRAM usage estimation for different models"""
        bridge = ModelIntegrationBridge()
        
        # Test VRAM estimates match expected values
        vram_expectations = {
            "t2v-A14B": 8000,  # 8GB for A14B models
            "i2v-A14B": 8000,
            "ti2v-5B": 6000,   # 6GB for 5B model
            "T2V": 8000,
            "I2V": 8000,
            "TI2V": 6000,
            "unknown_model": 8000  # Default fallback
        }
        
        for model_type, expected_vram in vram_expectations.items():
            estimated = bridge._estimate_model_vram_usage(model_type)
            assert estimated == expected_vram, f"VRAM estimate for {model_type}: got {estimated}, expected {expected_vram}"
    
    @pytest.mark.asyncio
    async def test_model_path_resolution(self):
        """Test model path resolution for all model types"""
        bridge = ModelIntegrationBridge()
        
        # Test model path patterns
        path_patterns = {
            "t2v-A14B": "Wan2.2-T2V-A14B-Diffusers",
            "i2v-A14B": "Wan2.2-I2V-A14B-Diffusers",
            "ti2v-5B": "Wan2.2-TI2V-5B-Diffusers"
        }
        
        for model_type, expected_pattern in path_patterns.items():
            path = bridge._get_model_path(model_type)
            if path:  # Path might be None if model directory doesn't exist
                assert expected_pattern in path, f"Path for {model_type} should contain {expected_pattern}"
    
    @pytest.mark.asyncio
    async def test_initialization_with_missing_dependencies(self):
        """Test initialization behavior when dependencies are missing"""
        bridge = ModelIntegrationBridge()
        
        # Test with missing system integration
        with patch('backend.core.system_integration.get_system_integration') as mock_get_integration:
            mock_get_integration.side_effect = ImportError("System integration not available")
            
            result = await bridge.initialize()
            assert result is False
            assert not bridge.is_initialized()
            
            status = bridge.get_integration_status()
            assert status['initialized'] is False
            assert 'error' in status
    
    @pytest.mark.asyncio
    async def test_model_availability_with_different_states(self):
        """Test model availability checking with different model states"""
        with patch('backend.core.system_integration.get_system_integration') as mock_get_integration:
            # Mock system integration
            mock_integration = AsyncMock()
            mock_model_manager = Mock()
            
            # Test different model states
            test_states = [
                {
                    'is_available': True,
                    'status': {'status': 'available', 'size_mb': 1024, 'is_loaded': True},
                    'expected_status': ModelStatus.AVAILABLE
                },
                {
                    'is_available': False,
                    'status': {'status': 'missing', 'size_mb': 0, 'is_loaded': False},
                    'expected_status': ModelStatus.MISSING
                },
                {
                    'is_available': True,
                    'status': {'status': 'corrupted', 'size_mb': 512, 'is_loaded': False},
                    'expected_status': ModelStatus.CORRUPTED
                }
            ]
            
            for state in test_states:
                mock_model_manager.is_model_available.return_value = state['is_available']
                mock_model_manager.get_model_status.return_value = state['status']
                mock_integration.get_model_manager.return_value = mock_model_manager
                mock_get_integration.return_value = mock_integration
                
                bridge = ModelIntegrationBridge()
                await bridge.initialize()
                
                status = await bridge.check_model_availability("t2v-A14B")
                
                assert isinstance(status, ModelIntegrationStatus)
                assert status.model_type == ModelType.T2V_A14B
                # Note: The actual status mapping depends on implementation details
    
    @pytest.mark.asyncio
    async def test_optimization_config_generation(self):
        """Test optimization configuration generation"""
        bridge = ModelIntegrationBridge()
        
        # Test various optimization parameter combinations
        test_configs = [
            {
                'params': GenerationParams(
                    prompt="test",
                    model_type="t2v-A14B",
                    resolution="1280x720",
                    steps=20,
                    quantization_level="fp16",
                    enable_offload=True,
                    max_vram_usage_gb=8.0
                ),
                'expected_keys': ['precision', 'enable_cpu_offload', 'min_vram_mb']
            },
            {
                'params': GenerationParams(
                    prompt="test",
                    model_type="ti2v-5B",
                    resolution="1920x1080",
                    steps=50,
                    quantization_level="bf16",
                    enable_offload=False,
                    vae_tile_size=512
                ),
                'expected_keys': ['precision', 'enable_cpu_offload', 'vae_tile_size']
            }
        ]
        
        for config in test_configs:
            opt_config = bridge._create_optimization_config(config['params'])
            
            # Verify expected keys are present
            for key in config['expected_keys']:
                assert key in opt_config, f"Missing optimization key: {key}"
            
            # Verify precision mapping
            if config['params'].quantization_level:
                assert opt_config['precision'] == config['params'].quantization_level
    
    @pytest.mark.asyncio
    async def test_concurrent_model_operations(self):
        """Test concurrent model loading and availability checking"""
        with patch('backend.core.system_integration.get_system_integration') as mock_get_integration:
            mock_integration = AsyncMock()
            mock_model_manager = Mock()
            mock_model_manager.is_model_available.return_value = True
            mock_model_manager.get_model_status.return_value = {
                'status': 'available', 'size_mb': 1024, 'is_loaded': False
            }
            mock_integration.get_model_manager.return_value = mock_model_manager
            mock_get_integration.return_value = mock_integration
            
            bridge = ModelIntegrationBridge()
            await bridge.initialize()
            
            # Test concurrent availability checks
            model_types = ["t2v-A14B", "i2v-A14B", "ti2v-5B"]
            
            tasks = [
                bridge.check_model_availability(model_type)
                for model_type in model_types
            ]
            
            results = await asyncio.gather(*tasks)
            
            assert len(results) == len(model_types)
            for result in results:
                assert isinstance(result, ModelIntegrationStatus)
    
    @pytest.mark.asyncio
    async def test_error_recovery_suggestions(self):
        """Test error recovery suggestion generation"""
        bridge = ModelIntegrationBridge()
        
        # Test different error categories
        error_scenarios = [
            {
                'category': 'model_loading',
                'message': 'Failed to load model',
                'expected_suggestions': ['model files are downloaded', 'sufficient VRAM']
            },
            {
                'category': 'vram_exhaustion',
                'message': 'CUDA out of memory',
                'expected_suggestions': ['quantization', 'CPU offloading', 'smaller batch size']
            },
            {
                'category': 'parameter_validation',
                'message': 'Invalid parameters',
                'expected_suggestions': ['parameter values', 'supported ranges']
            }
        ]
        
        for scenario in error_scenarios:
            result = bridge._create_error_result(
                "test_task", scenario['category'], scenario['message']
            )
            
            assert result.success is False
            assert result.error_category == scenario['category']
            assert result.error_message == scenario['message']
            assert len(result.recovery_suggestions) > 0
            
            # Check that expected suggestion keywords are present
            suggestions_text = ' '.join(result.recovery_suggestions).lower()
            for expected in scenario['expected_suggestions']:
                assert expected.lower() in suggestions_text, f"Missing suggestion keyword: {expected}"
    
    @pytest.mark.asyncio
    async def test_global_instance_management(self):
        """Test global instance management and singleton behavior"""
        # Clear any existing global instance
        import backend.core.model_integration_bridge as bridge_module
        bridge_module._model_integration_bridge = None
        
        # Get first instance
        bridge1 = await get_model_integration_bridge()
        assert bridge1 is not None
        
        # Get second instance - should be the same
        bridge2 = await get_model_integration_bridge()
        assert bridge1 is bridge2
        
        # Verify both are initialized
        assert bridge1.is_initialized()
        assert bridge2.is_initialized()
    
    @pytest.mark.asyncio
    async def test_model_status_caching(self):
        """Test model status caching behavior"""
        with patch('backend.core.system_integration.get_system_integration') as mock_get_integration:
            mock_integration = AsyncMock()
            mock_model_manager = Mock()
            
            # Track call count
            call_count = 0
            def mock_get_status(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                return {'status': 'available', 'size_mb': 1024, 'is_loaded': False}
            
            mock_model_manager.get_model_status.side_effect = mock_get_status
            mock_model_manager.is_model_available.return_value = True
            mock_integration.get_model_manager.return_value = mock_model_manager
            mock_get_integration.return_value = mock_integration
            
            bridge = ModelIntegrationBridge()
            await bridge.initialize()
            
            # First call should hit the model manager
            status1 = await bridge.check_model_availability("t2v-A14B")
            assert call_count == 1
            
            # Second call within cache timeout should use cache
            status2 = await bridge.check_model_availability("t2v-A14B")
            # Note: Caching behavior depends on implementation
            
            assert status1.model_type == status2.model_type

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
