"""
Unit tests for startup manager configuration loading and validation.
"""

import json
import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open

from scripts.startup_manager.config import (
    StartupConfig, BackendConfig, FrontendConfig, LoggingConfig, 
    RecoveryConfig, EnvironmentConfig, SecurityConfig, ConfigLoader,
    create_default_config_file
)


class TestStartupConfig:
    """Test cases for StartupConfig model."""
    
    def test_default_config_creation(self):
        """Test creating config with default values."""
        config = StartupConfig()
        
        assert config.backend.port == 8000
        assert config.frontend.port == 3000
        assert config.retry_attempts == 3
        assert config.retry_delay == 2.0
        assert config.verbose_logging is False
        assert config.auto_fix_issues is True
    
    def test_config_with_custom_values(self):
        """Test creating config with custom values."""
        config = StartupConfig(
            backend=BackendConfig(port=8080, log_level="debug"),
            frontend=FrontendConfig(port=3001, open_browser=False),
            retry_attempts=5,
            verbose_logging=True
        )
        
        assert config.backend.port == 8080
        assert config.backend.log_level == "debug"
        assert config.frontend.port == 3001
        assert config.frontend.open_browser is False
        assert config.retry_attempts == 5
        assert config.verbose_logging is True
    
    def test_config_validation_constraints(self):
        """Test configuration validation constraints."""
        # Test valid retry attempts
        config = StartupConfig(retry_attempts=5)
        assert config.retry_attempts == 5
        
        # Test invalid retry attempts (should raise validation error)
        with pytest.raises(ValueError):
            StartupConfig(retry_attempts=0)
        
        with pytest.raises(ValueError):
            StartupConfig(retry_attempts=15)
    
    def test_backend_config_log_level_validation(self):
        """Test backend log level validation."""
        # Valid log levels
        for level in ["debug", "info", "warning", "error", "critical"]:
            config = BackendConfig(log_level=level)
            assert config.log_level == level
        
        # Invalid log level
        with pytest.raises(ValueError):
            BackendConfig(log_level="invalid")
    
    def test_server_config_timeout_validation(self):
        """Test server timeout validation."""
        # Valid timeout
        config = BackendConfig(timeout=60)
        assert config.timeout == 60
        
        # Invalid timeouts
        with pytest.raises(ValueError):
            BackendConfig(timeout=0)
        
        with pytest.raises(ValueError):
            BackendConfig(timeout=500)


class TestConfigLoader:
    """Test cases for ConfigLoader class."""
    
    def test_load_config_from_existing_file(self):
        """Test loading configuration from existing file."""
        config_data = {
            "backend": {"port": 8080, "log_level": "debug"},
            "frontend": {"port": 3001},
            "retry_attempts": 5
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_path = Path(f.name)
        
        try:
            loader = ConfigLoader(config_path)
            config = loader.load_config()
            
            assert config.backend.port == 8080
            assert config.backend.log_level == "debug"
            assert config.frontend.port == 3001
            assert config.retry_attempts == 5
        finally:
            config_path.unlink()
    
    def test_load_config_creates_default_when_missing(self):
        """Test that default config is created when file doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "nonexistent_config.json"
            
            loader = ConfigLoader(config_path)
            config = loader.load_config()
            
            # Should create default config
            assert config.backend.port == 8000
            assert config.frontend.port == 3000
            assert config_path.exists()
    
    def test_load_config_invalid_json(self):
        """Test handling of invalid JSON in config file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{ invalid json }")
            config_path = Path(f.name)
        
        try:
            loader = ConfigLoader(config_path)
            with pytest.raises(ValueError, match="Invalid configuration file"):
                loader.load_config()
        finally:
            config_path.unlink()

        assert True  # TODO: Add proper assertion
    
    def test_save_config(self):
        """Test saving configuration to file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.json"
            
            loader = ConfigLoader(config_path)
            config = loader.load_config()
            
            # Modify config
            config.backend.port = 8080
            config.verbose_logging = True
            
            loader.save_config()
            
            # Verify file was saved correctly
            assert config_path.exists()
            with open(config_path, 'r') as f:
                saved_data = json.load(f)
            
            assert saved_data["backend"]["port"] == 8080
            assert saved_data["verbose_logging"] is True
    
    def test_validate_config_port_conflicts(self):
        """Test configuration validation for port conflicts."""
        config_data = {
            "backend": {"port": 8000},
            "frontend": {"port": 8000}  # Same port - should be invalid
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_path = Path(f.name)
        
        try:
            loader = ConfigLoader(config_path)
            loader.load_config()
            validation_result = loader.validate_config()
            
            assert validation_result["valid"] is False
            assert any("same port" in error.lower() for error in validation_result["errors"])
        finally:
            config_path.unlink()
    
    def test_validate_config_port_warnings(self):
        """Test configuration validation warnings for unusual ports."""
        config_data = {
            "backend": {"port": 80},  # Low port - should generate warning
            "frontend": {"port": 70000}  # High port - should generate error
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_path = Path(f.name)
        
        try:
            loader = ConfigLoader(config_path)
            loader.load_config()
            validation_result = loader.validate_config()
            
            assert len(validation_result["warnings"]) >= 1
            # Check for privileged port warning
            assert any("privileged range" in warning for warning in validation_result["warnings"])
            # Check for invalid port error
            assert any("invalid" in error for error in validation_result["errors"])
        finally:
            config_path.unlink()
    
    def test_validate_config_timeout_warnings(self):
        """Test configuration validation warnings for low timeouts."""
        config_data = {
            "backend": {"timeout": 5}  # Very low timeout - should generate warning
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_path = Path(f.name)
        
        try:
            loader = ConfigLoader(config_path)
            loader.load_config()
            validation_result = loader.validate_config()
            
            assert any("timeout is very low" in warning.lower() for warning in validation_result["warnings"])
        finally:
            config_path.unlink()
    
    def test_validate_config_no_config_loaded(self):
        """Test validation when no config is loaded."""
        loader = ConfigLoader()
        
        with pytest.raises(ValueError, match="No configuration loaded to validate"):
            loader.validate_config()

        assert True  # TODO: Add proper assertion
    
    def test_save_config_no_config_loaded(self):
        """Test saving when no config is loaded."""
        loader = ConfigLoader()
        
        with pytest.raises(ValueError, match="No configuration loaded to save"):
            loader.save_config()


        assert True  # TODO: Add proper assertion

class TestNewConfigModels:
    """Test cases for new configuration models."""
    
    def test_logging_config_defaults(self):
        """Test LoggingConfig default values."""
        config = LoggingConfig()
        
        assert config.level == "info"
        assert config.file_enabled is True
        assert config.console_enabled is True
        assert config.max_file_size == 10485760  # 10MB
        assert config.backup_count == 5
    
    def test_recovery_config_defaults(self):
        """Test RecoveryConfig default values."""
        config = RecoveryConfig()
        
        assert config.enabled is True
        assert config.max_retry_attempts == 3
        assert config.retry_delay == 2.0
        assert config.exponential_backoff is True
        assert config.auto_kill_processes is False
        assert config.fallback_ports == [8080, 8081, 8082, 3001, 3002, 3003]
    
    def test_environment_config_defaults(self):
        """Test EnvironmentConfig default values."""
        config = EnvironmentConfig()
        
        assert config.python_min_version == "3.8.0"
        assert config.node_min_version == "16.0.0"
        assert config.npm_min_version == "8.0.0"
        assert config.check_virtual_env is True
        assert config.validate_dependencies is True
        assert config.auto_install_missing is False
    
    def test_security_config_defaults(self):
        """Test SecurityConfig default values."""
        config = SecurityConfig()
        
        assert config.allow_admin_elevation is True
        assert config.firewall_auto_exception is False
        assert config.trusted_port_range == (8000, 9000)
    
    def test_security_config_port_range_validation(self):
        """Test SecurityConfig port range validation."""
        # Valid range
        config = SecurityConfig(trusted_port_range=(8000, 9000))
        assert config.trusted_port_range == (8000, 9000)
        
        # Invalid ranges
        with pytest.raises(ValueError):
            SecurityConfig(trusted_port_range=(9000, 8000))  # start >= end
        
        with pytest.raises(ValueError):
            SecurityConfig(trusted_port_range=(500, 1000))  # start < 1024
        
        with pytest.raises(ValueError):
            SecurityConfig(trusted_port_range=(8000, 70000))  # end > 65535


class TestEnhancedConfigLoader:
    """Test cases for enhanced ConfigLoader functionality."""
    
    def test_load_config_with_env_overrides(self):
        """Test loading configuration with environment variable overrides."""
        config_data = {
            "backend": {"port": 8000},
            "logging": {"level": "info"}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_path = Path(f.name)
        
        try:
            with patch.dict(os.environ, {
                'WAN22_BACKEND__PORT': '8080',
                'WAN22_LOGGING__LEVEL': 'debug',
                'WAN22_RECOVERY__ENABLED': 'false'
            }):
                loader = ConfigLoader(config_path)
                config = loader.load_config(apply_env_overrides=True)
                
                assert config.backend.port == 8080  # Overridden by env var
                assert config.logging.level == "debug"  # Overridden by env var
                assert config.recovery.enabled is False  # Overridden by env var
        finally:
            config_path.unlink()
    
    def test_load_config_without_env_overrides(self):
        """Test loading configuration without environment variable overrides."""
        config_data = {
            "backend": {"port": 8000},
            "logging": {"level": "info"}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_path = Path(f.name)
        
        try:
            with patch.dict(os.environ, {
                'WAN22_BACKEND__PORT': '8080',
                'WAN22_LOGGING__LEVEL': 'debug'
            }):
                loader = ConfigLoader(config_path)
                config = loader.load_config(apply_env_overrides=False)
                
                assert config.backend.port == 8000  # Not overridden
                assert config.logging.level == "info"  # Not overridden
        finally:
            config_path.unlink()
    
    def test_parse_env_value(self):
        """Test environment variable value parsing."""
        loader = ConfigLoader()
        
        # Boolean values
        assert loader._parse_env_value("true") is True
        assert loader._parse_env_value("false") is False
        assert loader._parse_env_value("yes") is True
        assert loader._parse_env_value("no") is False
        assert loader._parse_env_value("1") is True
        assert loader._parse_env_value("0") is False
        
        # Numeric values
        assert loader._parse_env_value("123") == 123
        assert loader._parse_env_value("123.45") == 123.45
        
        # JSON values
        assert loader._parse_env_value('[1, 2, 3]') == [1, 2, 3]
        assert loader._parse_env_value('{"key": "value"}') == {"key": "value"}
        
        # String values
        assert loader._parse_env_value("hello") == "hello"
    
    def test_get_env_overrides_summary(self):
        """Test getting summary of environment overrides."""
        config_data = {
            "backend": {"port": 8000},
            "logging": {"level": "info"}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_path = Path(f.name)
        
        try:
            with patch.dict(os.environ, {
                'WAN22_BACKEND__PORT': '8080',
                'WAN22_LOGGING__LEVEL': 'debug'
            }):
                loader = ConfigLoader(config_path)
                loader.load_config(apply_env_overrides=True)
                
                overrides = loader.get_env_overrides_summary()
                
                assert "backend.port" in overrides
                assert overrides["backend.port"]["value"] == 8080
                assert overrides["backend.port"]["original"] == 8000
        finally:
            config_path.unlink()
    
    def test_export_config_for_ci_env_format(self):
        """Test exporting configuration in environment variable format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.json"
            
            loader = ConfigLoader(config_path)
            config = loader.load_config()
            
            env_export = loader.export_config_for_ci(format="env")
            
            assert "WAN22_BACKEND__PORT=8000" in env_export
            assert "WAN22_FRONTEND__PORT=3000" in env_export
            assert "WAN22_LOGGING__LEVEL=info" in env_export
    
    def test_export_config_for_ci_json_format(self):
        """Test exporting configuration in JSON format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.json"
            
            loader = ConfigLoader(config_path)
            config = loader.load_config()
            
            json_export = loader.export_config_for_ci(format="json")
            parsed = json.loads(json_export)
            
            assert parsed["backend"]["port"] == 8000
            assert parsed["frontend"]["port"] == 3000
            assert parsed["logging"]["level"] == "info"
    
    def test_create_deployment_config_production(self):
        """Test creating production deployment configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.json"
            
            loader = ConfigLoader(config_path)
            loader.load_config()
            
            prod_config = loader.create_deployment_config("production")
            
            assert prod_config["logging"]["level"] == "warning"
            assert prod_config["backend"]["reload"] is False
            assert prod_config["frontend"]["hot_reload"] is False
            assert prod_config["frontend"]["open_browser"] is False
    
    def test_create_deployment_config_development(self):
        """Test creating development deployment configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.json"
            
            loader = ConfigLoader(config_path)
            loader.load_config()
            
            dev_config = loader.create_deployment_config("development")
            
            assert dev_config["logging"]["level"] == "debug"
            assert dev_config["backend"]["reload"] is True
            assert dev_config["frontend"]["hot_reload"] is True
            assert dev_config["frontend"]["open_browser"] is True
    
    def test_create_deployment_config_ci(self):
        """Test creating CI deployment configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.json"
            
            loader = ConfigLoader(config_path)
            loader.load_config()
            
            ci_config = loader.create_deployment_config("ci")
            
            assert ci_config["logging"]["level"] == "info"
            assert ci_config["frontend"]["open_browser"] is False
            assert ci_config["recovery"]["auto_kill_processes"] is True
            assert ci_config["environment"]["auto_install_missing"] is True


class TestEnhancedConfigValidation:
    """Test cases for enhanced configuration validation."""
    
    def test_comprehensive_validation_success(self):
        """Test comprehensive validation with valid configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.json"
            
            loader = ConfigLoader(config_path)
            loader.load_config()
            
            validation_result = loader.validate_config()
            
            assert validation_result["valid"] is True
            assert len(validation_result["errors"]) == 0
    
    def test_validation_port_conflicts(self):
        """Test validation detects port conflicts."""
        config_data = {
            "backend": {"port": 8000},
            "frontend": {"port": 8000}  # Same port
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_path = Path(f.name)
        
        try:
            loader = ConfigLoader(config_path)
            loader.load_config()
            validation_result = loader.validate_config()
            
            assert validation_result["valid"] is False
            assert any("same port" in error.lower() for error in validation_result["errors"])
        finally:
            config_path.unlink()
    
    def test_validation_security_warnings(self):
        """Test validation generates security warnings."""
        config_data = {
            "security": {
                "firewall_auto_exception": True,
                "allow_admin_elevation": False
            },
            "recovery": {
                "auto_kill_processes": True
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_path = Path(f.name)
        
        try:
            loader = ConfigLoader(config_path)
            loader.load_config()
            validation_result = loader.validate_config()
            
            warnings = validation_result["warnings"]
            assert any("firewall exception" in warning.lower() for warning in warnings)
            assert any("process killing" in warning.lower() for warning in warnings)
        finally:
            config_path.unlink()
    
    def test_validation_environment_warnings(self):
        """Test validation generates environment warnings."""
        config_data = {
            "environment": {
                "auto_install_missing": True
            },
            "logging": {
                "max_file_size": 2097152  # 2MB - valid but small
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_path = Path(f.name)
        
        try:
            loader = ConfigLoader(config_path)
            loader.load_config()
            validation_result = loader.validate_config()
            
            warnings = validation_result["warnings"]
            assert any("dependency installation" in warning.lower() for warning in warnings)
        finally:
            config_path.unlink()


class TestConfigUtilities:
    """Test cases for configuration utility functions."""
    
    def test_create_default_config_file(self):
        """Test creating default configuration file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "default_config.json"
            
            created_path = create_default_config_file(config_path)
            
            assert created_path == config_path
            assert config_path.exists()
            
            # Verify content
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            assert "backend" in config_data
            assert "frontend" in config_data
            assert "logging" in config_data
            assert "recovery" in config_data
            assert "environment" in config_data
            assert "security" in config_data


if __name__ == "__main__":
    pytest.main([__file__])
