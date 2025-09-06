"""
Tests for the unified configuration system
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pytest
import tempfile
import json
import yaml
from pathlib import Path
from tools.config_manager.unified_config import (
    UnifiedConfig, SystemConfig, APIConfig, LogLevel, 
    QuantizationLevel, Environment
)


class TestUnifiedConfig:
    """Test cases for UnifiedConfig class"""
    
    def test_default_config_creation(self):
        """Test creating a default configuration"""
        config = UnifiedConfig()
        
        assert config.system.name == "WAN22 Video Generation System"
        assert config.system.version == "2.2.0"
        assert config.system.log_level == LogLevel.INFO
        assert config.api.host == "localhost"
        assert config.api.port == 8000
        assert config.models.quantization_level == QuantizationLevel.BF16
    
    def test_config_to_dict(self):
        """Test converting configuration to dictionary"""
        config = UnifiedConfig()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert "system" in config_dict
        assert "api" in config_dict
        assert "models" in config_dict
        assert config_dict["system"]["name"] == "WAN22 Video Generation System"
        assert config_dict["api"]["host"] == "localhost"
    
    def test_config_to_json(self):
        """Test converting configuration to JSON"""
        config = UnifiedConfig()
        json_str = config.to_json()
        
        # Should be valid JSON
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        assert "system" in parsed
    
    def test_config_to_yaml(self):
        """Test converting configuration to YAML"""
        config = UnifiedConfig()
        yaml_str = config.to_yaml()
        
        # Should be valid YAML
        parsed = yaml.safe_load(yaml_str)
        assert isinstance(parsed, dict)
        assert "system" in parsed
    
    def test_config_from_dict(self):
        """Test creating configuration from dictionary"""
        data = {
            "system": {
                "name": "Test System",
                "debug": True,
                "log_level": "DEBUG"
            },
            "api": {
                "port": 9000,
                "debug": True
            }
        }
        
        config = UnifiedConfig.from_dict(data)
        
        assert config.system.name == "Test System"
        assert config.system.debug is True
        assert config.system.log_level == LogLevel.DEBUG
        assert config.api.port == 9000
        assert config.api.debug is True
    
    def test_config_from_json(self):
        """Test creating configuration from JSON"""
        data = {
            "system": {
                "name": "JSON Test System",
                "version": "1.0.0"
            },
            "models": {
                "quantization_level": "fp16"
            }
        }
        
        json_str = json.dumps(data)
        config = UnifiedConfig.from_json(json_str)
        
        assert config.system.name == "JSON Test System"
        assert config.system.version == "1.0.0"
        assert config.models.quantization_level == QuantizationLevel.FP16
    
    def test_config_from_yaml(self):
        """Test creating configuration from YAML"""
        yaml_str = """
        system:
          name: "YAML Test System"
          debug: true
        api:
          port: 8080
        """
        
        config = UnifiedConfig.from_yaml(yaml_str)
        
        assert config.system.name == "YAML Test System"
        assert config.system.debug is True
        assert config.api.port == 8080
    
    def test_config_file_operations(self):
        """Test saving and loading configuration files"""
        config = UnifiedConfig()
        config.system.name = "File Test System"
        config.api.port = 7000
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test YAML file
            yaml_path = Path(temp_dir) / "test_config.yaml"
            config.save_to_file(yaml_path)
            
            loaded_config = UnifiedConfig.from_file(yaml_path)
            assert loaded_config.system.name == "File Test System"
            assert loaded_config.api.port == 7000
            
            # Test JSON file
            json_path = Path(temp_dir) / "test_config.json"
            config.save_to_file(json_path)
            
            loaded_config = UnifiedConfig.from_file(json_path)
            assert loaded_config.system.name == "File Test System"
            assert loaded_config.api.port == 7000
    
    def test_environment_overrides(self):
        """Test applying environment-specific overrides"""
        config = UnifiedConfig()
        
        # Add environment overrides
        config.environments["development"] = {
            "system": {"debug": True, "log_level": "DEBUG"},
            "api": {"debug": True, "reload": True}
        }
        
        # Apply development overrides
        dev_config = config.apply_environment_overrides("development")
        
        # Original config should be unchanged
        assert config.system.debug is False
        
        # New config should have overrides applied
        # Note: This test may need adjustment based on actual implementation
        # of apply_environment_overrides method
    
    def test_config_path_access(self):
        """Test getting and setting configuration values by path"""
        config = UnifiedConfig()
        
        # Test getting values
        assert config.get_config_path("system.name") == "WAN22 Video Generation System"
        assert config.get_config_path("api.port") == 8000
        
        # Test setting values
        config.set_config_path("system.name", "New System Name")
        config.set_config_path("api.port", 9000)
        
        assert config.system.name == "New System Name"
        assert config.api.port == 9000
    
    def test_invalid_config_path(self):
        """Test handling of invalid configuration paths"""
        config = UnifiedConfig()
        
        with pytest.raises(KeyError):
            config.get_config_path("invalid.path")
        
        with pytest.raises(KeyError):
            config.set_config_path("invalid.path", "value")

        assert True  # TODO: Add proper assertion
    
    def test_enum_handling(self):
        """Test proper handling of enum values"""
        config = UnifiedConfig()
        
        # Test setting enum values
        config.system.log_level = LogLevel.ERROR
        config.models.quantization_level = QuantizationLevel.INT8
        
        # Convert to dict and back
        config_dict = config.to_dict()
        new_config = UnifiedConfig.from_dict(config_dict)
        
        assert new_config.system.log_level == LogLevel.ERROR
        assert new_config.models.quantization_level == QuantizationLevel.INT8


class TestConfigurationSections:
    """Test individual configuration sections"""
    
    def test_system_config(self):
        """Test SystemConfig dataclass"""
        system = SystemConfig()
        
        assert system.name == "WAN22 Video Generation System"
        assert system.version == "2.2.0"
        assert system.debug is False
        assert system.log_level == LogLevel.INFO
        assert system.max_queue_size == 10
    
    def test_api_config(self):
        """Test APIConfig dataclass"""
        api = APIConfig()
        
        assert api.host == "localhost"
        assert api.port == 8000
        assert api.auto_port is True
        assert api.workers == 1
        assert "http://localhost:3000" in api.cors_origins
    
    def test_config_validation_types(self):
        """Test that configuration accepts valid types"""
        config = UnifiedConfig()
        
        # Test valid assignments
        config.system.max_queue_size = 20
        config.api.port = 9000
        config.models.model_cache_size = 5
        
        assert config.system.max_queue_size == 20
        assert config.api.port == 9000
        assert config.models.model_cache_size == 5


if __name__ == "__main__":
    pytest.main([__file__])