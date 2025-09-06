"""
Test suite for Configuration Bridge
Tests configuration loading, validation, and runtime updates
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open

import sys
sys.path.append(str(Path(__file__).parent.parent))

try:
    from backend.core.configuration_bridge import ConfigurationBridge
except ImportError:
    # Try alternative import path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from backend.core.configuration_bridge import ConfigurationBridge


class TestConfigurationBridge:
    """Test cases for ConfigurationBridge functionality"""
    
    def test_initialization_with_valid_config(self):
        """Test initialization with a valid configuration file"""
        # Create a temporary config file
        config_data = {
            "system": {"default_quantization": "bf16"},
            "directories": {"models_directory": "models"},
            "generation": {"default_resolution": "1280x720"},
            "models": {"t2v_model": "Wan2.2-T2V-A14B"},
            "optimization": {"enable_offload": True}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name
        
        try:
            bridge = ConfigurationBridge(config_path)
            assert bridge.config_data == config_data
            assert bridge.config_path == Path(config_path)
        finally:
            Path(config_path).unlink()
    
    def test_initialization_with_missing_config(self):
        """Test initialization when config file doesn't exist"""
        bridge = ConfigurationBridge("/nonexistent/config.json")
        
        # Should create default configuration
        assert "system" in bridge.config_data
        assert "directories" in bridge.config_data
        assert "generation" in bridge.config_data
        assert "models" in bridge.config_data
        assert "optimization" in bridge.config_data
    
    def test_get_model_paths(self):
        """Test model path configuration retrieval"""
        bridge = ConfigurationBridge()
        model_paths = bridge.get_model_paths()
        
        assert "models_directory" in model_paths
        assert "loras_directory" in model_paths
        assert "output_directory" in model_paths
        assert "t2v_model" in model_paths
        assert "i2v_model" in model_paths
        assert "ti2v_model" in model_paths
    
    def test_get_optimization_settings(self):
        """Test optimization settings retrieval"""
        bridge = ConfigurationBridge()
        settings = bridge.get_optimization_settings()
        
        assert "quantization" in settings
        assert "enable_offload" in settings
        assert "vae_tile_size" in settings
        assert "max_vram_usage_gb" in settings
        assert "quantization_levels" in settings
    
    def test_get_generation_defaults(self):
        """Test generation defaults retrieval"""
        bridge = ConfigurationBridge()
        defaults = bridge.get_generation_defaults()
        
        assert "resolution" in defaults
        assert "steps" in defaults
        assert "duration" in defaults
        assert "fps" in defaults
        assert "supported_resolutions" in defaults
    
    def test_update_optimization_setting_valid(self):
        """Test updating a valid optimization setting"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = {
                "optimization": {"default_quantization": "bf16", "enable_offload": True}
            }
            json.dump(config_data, f)
            config_path = f.name
        
        try:
            bridge = ConfigurationBridge(config_path)
            
            # Update quantization setting
            success = bridge.update_optimization_setting("default_quantization", "fp16")
            assert success
            assert bridge.config_data["optimization"]["default_quantization"] == "fp16"
            
            # Update boolean setting
            success = bridge.update_optimization_setting("enable_offload", False)
            assert success
            assert bridge.config_data["optimization"]["enable_offload"] is False
            
        finally:
            Path(config_path).unlink()
    
    def test_update_optimization_setting_invalid(self):
        """Test updating optimization setting with invalid values"""
        bridge = ConfigurationBridge()
        
        # Invalid quantization level
        success = bridge.update_optimization_setting("default_quantization", "invalid")
        assert not success
        
        # Invalid vae_tile_size
        success = bridge.update_optimization_setting("vae_tile_size", 1000)
        assert not success
        
        # Invalid VRAM setting
        success = bridge.update_optimization_setting("max_vram_usage_gb", 100)
        assert not success
    
    def test_update_model_path(self):
        """Test updating model path configuration"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = {"models": {"t2v_model": "old-model"}}
            json.dump(config_data, f)
            config_path = f.name
        
        try:
            bridge = ConfigurationBridge(config_path)
            
            success = bridge.update_model_path("t2v_model", "new-model")
            assert success
            assert bridge.config_data["models"]["t2v_model"] == "new-model"
            
            # Test invalid model type
            success = bridge.update_model_path("invalid_model", "some-model")
            assert not success
            
        finally:
            Path(config_path).unlink()
    
    def test_configuration_validation(self):
        """Test configuration validation"""
        # Valid configuration
        bridge = ConfigurationBridge()
        is_valid, errors = bridge.validate_configuration()
        assert is_valid
        assert len(errors) == 0
        
        # Invalid configuration - missing sections
        bridge.config_data = {"system": {}}  # Missing required sections
        is_valid, errors = bridge.validate_configuration()
        assert not is_valid
        assert len(errors) > 0
    
    def test_get_runtime_config_for_generation(self):
        """Test getting runtime configuration for specific model types"""
        bridge = ConfigurationBridge()
        
        # Test T2V model config
        config = bridge.get_runtime_config_for_generation("t2v-A14B")
        assert "model_paths" in config
        assert "optimization" in config
        assert "generation_defaults" in config
        
        # Test TI2V model config (should have more conservative settings)
        config = bridge.get_runtime_config_for_generation("ti2v-5B")
        assert config["optimization"]["enable_offload"] is True
        assert config["optimization"]["vae_tile_size"] <= 256
    
    def test_config_summary(self):
        """Test configuration summary generation"""
        bridge = ConfigurationBridge()
        summary = bridge.get_config_summary()
        
        assert "config_file" in summary
        assert "sections" in summary
        assert "model_paths" in summary
        assert "optimization_settings" in summary
        assert "generation_defaults" in summary
        assert "validation_status" in summary


if __name__ == "__main__":
    # Run basic tests
    test_bridge = TestConfigurationBridge()
    
    print("Testing ConfigurationBridge initialization...")
    test_bridge.test_initialization_with_missing_config()
    print("✓ Initialization test passed")
    
    print("Testing model paths...")
    test_bridge.test_get_model_paths()
    print("✓ Model paths test passed")
    
    print("Testing optimization settings...")
    test_bridge.test_get_optimization_settings()
    print("✓ Optimization settings test passed")
    
    print("Testing configuration validation...")
    test_bridge.test_configuration_validation()
    print("✓ Configuration validation test passed")
    
    print("Testing runtime config...")
    test_bridge.test_get_runtime_config_for_generation()
    print("✓ Runtime config test passed")
    
    print("\nAll ConfigurationBridge tests passed! ✓")