"""
Tests for the test configuration system
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, mock_open

from tests.config.test_config import (
    TestConfiguration, TestCategory, TestPriority, TestTimeouts,
    get_test_config, reload_test_config
)


class TestConfigurationSystem:
    """Test cases for TestConfig class"""
    
    def setup_method(self):
        """Setup for each test"""
        reset_test_config()
    
    def test_init_with_default_config(self):
        """Test initialization with default configuration"""
        config = TestConfig()
        assert config.config_path.name == "test-config.yaml"
        assert isinstance(config.environment, Environment)
    
    def test_init_with_custom_config_path(self):
        """Test initialization with custom configuration path"""
        # Create a temporary config file
        config_data = {
            'test_categories': {
                'unit': {
                    'description': 'Unit tests',
                    'timeout': 30,
                    'parallel': True,
                    'patterns': ['tests/unit/test_*.py']
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            custom_path = f.name
        
        try:
            config = TestConfig(config_path=custom_path)
            assert str(config.config_path) == custom_path
        finally:
            Path(custom_path).unlink()
    
    def test_init_with_environment(self):
        """Test initialization with specific environment"""
        config = TestConfig(environment="ci")
        assert config.environment == Environment.CI
    
    def test_environment_detection(self):
        """Test automatic environment detection"""
        # Test CI detection
        with patch.dict('os.environ', {'CI': 'true'}):
            config = TestConfig()
            assert config.environment == Environment.CI
        
        # Test local detection
        with patch.dict('os.environ', {'PYTEST_CURRENT_TEST': 'test_something'}):
            config = TestConfig()
            assert config.environment == Environment.LOCAL
    
    def test_load_configuration_success(self):
        """Test successful configuration loading"""
        config_data = {
            'test_categories': {
                'unit': {
                    'description': 'Unit tests',
                    'timeout': 30,
                    'parallel': True,
                    'patterns': ['tests/unit/test_*.py']
                }
            },
            'coverage': {
                'minimum_threshold': 70,
                'exclude_patterns': ['*/tests/*'],
                'report_formats': ['html']
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            config = TestConfig(config_path=config_path)
            
            # Verify categories loaded
            assert TestCategory.UNIT in config.categories
            unit_config = config.categories[TestCategory.UNIT]
            assert unit_config.timeout == 30
            assert unit_config.parallel is True
            
            # Verify coverage loaded
            assert config.coverage is not None
            assert config.coverage.minimum_threshold == 70
            
        finally:
            Path(config_path).unlink()
    
    def test_load_configuration_file_not_found(self):
        """Test configuration loading with missing file"""
        with pytest.raises(ConfigurationError, match="Configuration file not found"):
            TestConfig(config_path="/nonexistent/config.yaml")

        assert True  # TODO: Add proper assertion
    
    def test_environment_overrides(self):
        """Test environment-specific configuration overrides"""
        # Create main config
        main_config = {
            'test_categories': {
                'unit': {
                    'description': 'Unit tests',
                    'timeout': 30,
                    'parallel': True,
                    'patterns': ['tests/unit/test_*.py']
                }
            }
        }
        
        # Create environment override
        env_override = {
            'test_categories': {
                'unit': {
                    'timeout': 60  # Override timeout
                }
            }
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Write main config
            main_config_path = temp_path / "config.yaml"
            with open(main_config_path, 'w') as f:
                yaml.dump(main_config, f)
            
            # Write environment override
            env_dir = temp_path / "environments"
            env_dir.mkdir()
            env_config_path = env_dir / "development.yaml"
            with open(env_config_path, 'w') as f:
                yaml.dump(env_override, f)
            
            # Load config with environment
            config = TestConfig(config_path=main_config_path, environment="development")
            
            # Verify override applied
            unit_config = config.categories[TestCategory.UNIT]
            assert unit_config.timeout == 60  # Should be overridden value
    
    def test_get_category_config(self):
        """Test getting category configuration"""
        config_data = {
            'test_categories': {
                'unit': {
                    'description': 'Unit tests',
                    'timeout': 30,
                    'parallel': True,
                    'patterns': ['tests/unit/test_*.py']
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            config = TestConfig(config_path=config_path)
            
            # Test with enum
            unit_config = config.get_category_config(TestCategory.UNIT)
            assert unit_config is not None
            assert unit_config.timeout == 30
            
            # Test with string
            unit_config_str = config.get_category_config("unit")
            assert unit_config_str is not None
            assert unit_config_str.timeout == 30
            
            # Test with invalid category
            invalid_config = config.get_category_config("invalid")
            assert invalid_config is None
            
        finally:
            Path(config_path).unlink()
    
    def test_get_timeout(self):
        """Test getting timeout for categories"""
        config_data = {
            'test_categories': {
                'unit': {
                    'description': 'Unit tests',
                    'timeout': 45,
                    'parallel': True,
                    'patterns': ['tests/unit/test_*.py']
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            config = TestConfig(config_path=config_path)
            
            # Test existing category
            assert config.get_timeout(TestCategory.UNIT) == 45
            assert config.get_timeout("unit") == 45
            
            # Test non-existing category (should return default)
            assert config.get_timeout("nonexistent") == 300
            
        finally:
            Path(config_path).unlink()
    
    def test_validate_configuration(self):
        """Test configuration validation"""
        # Valid configuration
        valid_config_data = {
            'test_categories': {
                'unit': {
                    'description': 'Unit tests',
                    'timeout': 30,
                    'parallel': True,
                    'patterns': ['tests/unit/test_*.py']
                }
            },
            'coverage': {
                'minimum_threshold': 70,
                'exclude_patterns': ['*/tests/*'],
                'report_formats': ['html']
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(valid_config_data, f)
            config_path = f.name
        
        try:
            config = TestConfig(config_path=config_path)
            issues = config.validate_configuration()
            assert len(issues) == 0
            
        finally:
            Path(config_path).unlink()
        
        # Invalid configuration
        invalid_config_data = {
            'test_categories': {
                'unit': {
                    'description': 'Unit tests',
                    'timeout': -10,  # Invalid timeout
                    'parallel': True,
                    'patterns': []  # Empty patterns
                }
            },
            'coverage': {
                'minimum_threshold': 150,  # Invalid threshold
                'exclude_patterns': ['*/tests/*'],
                'report_formats': ['html']
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(invalid_config_data, f)
            config_path = f.name
        
        try:
            config = TestConfig(config_path=config_path)
            issues = config.validate_configuration()
            assert len(issues) > 0
            assert any("Invalid timeout" in issue for issue in issues)
            assert any("No test patterns" in issue for issue in issues)
            assert any("Invalid coverage threshold" in issue for issue in issues)
            
        finally:
            Path(config_path).unlink()
    
    def test_global_config_instance(self):
        """Test global configuration instance management"""
        # First call should create instance
        config1 = get_test_config()
        assert config1 is not None
        
        # Second call should return same instance
        config2 = get_test_config()
        assert config1 is config2
        
        # Force reload should create new instance
        config3 = get_test_config(force_reload=True)
        assert config3 is not config1
        
        # Reset should clear global instance
        reset_test_config()
        config4 = get_test_config()
        assert config4 is not config3


if __name__ == "__main__":
    pytest.main([__file__])