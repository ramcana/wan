"""
Tests for WAN22 Model Compatibility Configuration Manager

This module tests the comprehensive configuration management system including
user preferences, optimization strategies, pipeline selection, and security settings.
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from wan22_config_manager import (
    ConfigurationManager,
    WAN22Config,
    OptimizationConfig,
    PipelineConfig,
    SecurityConfig,
    CompatibilityConfig,
    UserPreferences,
    OptimizationStrategy,
    PipelineSelectionMode,
    SecurityLevel,
    get_config_manager,
    get_config,
    save_config,
    update_config
)


class TestOptimizationConfig:
    """Test optimization configuration"""
    
    def test_default_optimization_config(self):
        """Test default optimization configuration values"""
        config = OptimizationConfig()
        
        assert config.strategy == OptimizationStrategy.AUTO
        assert config.enable_mixed_precision is True
        assert config.enable_cpu_offload is False
        assert config.cpu_offload_strategy == "sequential"
        assert config.enable_chunked_processing is False
        assert config.max_chunk_size == 8
        assert config.vram_threshold_mb == 8192
        assert config.enable_vae_tiling is False
        assert config.vae_tile_size == 512
        assert isinstance(config.custom_optimizations, dict)
    
    def test_custom_optimization_config(self):
        """Test custom optimization configuration"""
        config = OptimizationConfig(
            strategy=OptimizationStrategy.MEMORY,
            enable_cpu_offload=True,
            cpu_offload_strategy="full",
            enable_chunked_processing=True,
            max_chunk_size=4,
            vram_threshold_mb=4096
        )
        
        assert config.strategy == OptimizationStrategy.MEMORY
        assert config.enable_cpu_offload is True
        assert config.cpu_offload_strategy == "full"
        assert config.enable_chunked_processing is True
        assert config.max_chunk_size == 4
        assert config.vram_threshold_mb == 4096


class TestPipelineConfig:
    """Test pipeline configuration"""
    
    def test_default_pipeline_config(self):
        """Test default pipeline configuration values"""
        config = PipelineConfig()
        
        assert config.selection_mode == PipelineSelectionMode.AUTO
        assert config.preferred_pipeline_class is None
        assert config.enable_fallback is True
        assert "component_isolation" in config.fallback_strategies
        assert "alternative_model" in config.fallback_strategies
        assert config.pipeline_timeout_seconds == 300
        assert config.max_retry_attempts == 3
        assert isinstance(config.custom_pipeline_paths, dict)
    
    def test_custom_pipeline_config(self):
        """Test custom pipeline configuration"""
        config = PipelineConfig(
            selection_mode=PipelineSelectionMode.MANUAL,
            preferred_pipeline_class="CustomWanPipeline",
            enable_fallback=False,
            fallback_strategies=["alternative_model"],
            pipeline_timeout_seconds=600,
            max_retry_attempts=5
        )
        
        assert config.selection_mode == PipelineSelectionMode.MANUAL
        assert config.preferred_pipeline_class == "CustomWanPipeline"
        assert config.enable_fallback is False
        assert config.fallback_strategies == ["alternative_model"]
        assert config.pipeline_timeout_seconds == 600
        assert config.max_retry_attempts == 5


class TestSecurityConfig:
    """Test security configuration"""
    
    def test_default_security_config(self):
        """Test default security configuration values"""
        config = SecurityConfig()
        
        assert config.security_level == SecurityLevel.MODERATE
        assert config.trust_remote_code is True
        assert "huggingface.co" in config.trusted_sources
        assert "hf.co" in config.trusted_sources
        assert config.enable_sandboxing is False
        assert config.sandbox_timeout_seconds == 60
        assert config.allow_local_code_execution is True
        assert config.code_signature_verification is False
    
    def test_strict_security_config(self):
        """Test strict security configuration"""
        config = SecurityConfig(
            security_level=SecurityLevel.STRICT,
            trust_remote_code=False,
            trusted_sources=["internal.company.com"],
            enable_sandboxing=True,
            allow_local_code_execution=False,
            code_signature_verification=True
        )
        
        assert config.security_level == SecurityLevel.STRICT
        assert config.trust_remote_code is False
        assert config.trusted_sources == ["internal.company.com"]
        assert config.enable_sandboxing is True
        assert config.allow_local_code_execution is False
        assert config.code_signature_verification is True


class TestCompatibilityConfig:
    """Test compatibility configuration"""
    
    def test_default_compatibility_config(self):
        """Test default compatibility configuration values"""
        config = CompatibilityConfig()
        
        assert config.enable_architecture_detection is True
        assert config.enable_vae_validation is True
        assert config.enable_component_validation is True
        assert config.strict_validation is False
        assert config.cache_detection_results is True
        assert config.detection_cache_ttl_hours == 24
        assert config.enable_diagnostic_collection is True
        assert config.diagnostic_output_dir == "diagnostics"
    
    def test_strict_compatibility_config(self):
        """Test strict compatibility configuration"""
        config = CompatibilityConfig(
            strict_validation=True,
            cache_detection_results=False,
            detection_cache_ttl_hours=1,
            diagnostic_output_dir="custom_diagnostics"
        )
        
        assert config.strict_validation is True
        assert config.cache_detection_results is False
        assert config.detection_cache_ttl_hours == 1
        assert config.diagnostic_output_dir == "custom_diagnostics"


class TestUserPreferences:
    """Test user preferences"""
    
    def test_default_user_preferences(self):
        """Test default user preferences values"""
        prefs = UserPreferences()
        
        assert prefs.default_output_format == "mp4"
        assert prefs.preferred_video_codec == "h264"
        assert prefs.default_fps == 24.0
        assert prefs.enable_progress_indicators is True
        assert prefs.verbose_logging is False
        assert prefs.auto_cleanup_temp_files is True
        assert prefs.max_concurrent_generations == 1
        assert prefs.notification_preferences["generation_complete"] is True
        assert prefs.notification_preferences["error_notifications"] is True
        assert prefs.notification_preferences["optimization_suggestions"] is True
    
    def test_custom_user_preferences(self):
        """Test custom user preferences"""
        prefs = UserPreferences(
            default_output_format="webm",
            preferred_video_codec="vp9",
            default_fps=30.0,
            verbose_logging=True,
            max_concurrent_generations=2,
            notification_preferences={
                "generation_complete": False,
                "error_notifications": True,
                "optimization_suggestions": False
            }
        )
        
        assert prefs.default_output_format == "webm"
        assert prefs.preferred_video_codec == "vp9"
        assert prefs.default_fps == 30.0
        assert prefs.verbose_logging is True
        assert prefs.max_concurrent_generations == 2
        assert prefs.notification_preferences["generation_complete"] is False
        assert prefs.notification_preferences["error_notifications"] is True
        assert prefs.notification_preferences["optimization_suggestions"] is False


class TestWAN22Config:
    """Test main WAN22 configuration"""
    
    def test_default_wan22_config(self):
        """Test default WAN22 configuration"""
        config = WAN22Config()
        
        assert config.version == "1.0.0"
        assert config.created_at is not None
        assert config.updated_at is not None
        assert isinstance(config.optimization, OptimizationConfig)
        assert isinstance(config.pipeline, PipelineConfig)
        assert isinstance(config.security, SecurityConfig)
        assert isinstance(config.compatibility, CompatibilityConfig)
        assert isinstance(config.user_preferences, UserPreferences)
        assert isinstance(config.experimental_features, dict)
        assert isinstance(config.custom_settings, dict)


class TestConfigurationManager:
    """Test configuration manager"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = ConfigurationManager(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test configuration manager initialization"""
        assert self.config_manager.config_dir == Path(self.temp_dir)
        assert self.config_manager.config_path == Path(self.temp_dir) / "wan22_config.json"
        assert self.config_manager._config is None
    
    def test_load_default_config(self):
        """Test loading default configuration when file doesn't exist"""
        config = self.config_manager.load_config()
        
        assert isinstance(config, WAN22Config)
        assert config.version == "1.0.0"
        assert self.config_manager.config_path.exists()
    
    def test_save_and_load_config(self):
        """Test saving and loading configuration"""
        # Create custom configuration
        config = WAN22Config()
        config.optimization.strategy = OptimizationStrategy.MEMORY
        config.pipeline.selection_mode = PipelineSelectionMode.MANUAL
        config.security.security_level = SecurityLevel.STRICT
        
        # Save configuration
        success = self.config_manager.save_config(config)
        assert success is True
        assert self.config_manager.config_path.exists()
        
        # Load configuration
        loaded_config = self.config_manager.load_config()
        assert loaded_config.optimization.strategy == OptimizationStrategy.MEMORY
        assert loaded_config.pipeline.selection_mode == PipelineSelectionMode.MANUAL
        assert loaded_config.security.security_level == SecurityLevel.STRICT
    
    def test_get_config(self):
        """Test getting configuration"""
        config = self.config_manager.get_config()
        assert isinstance(config, WAN22Config)
        
        # Should return same instance on subsequent calls
        config2 = self.config_manager.get_config()
        assert config is config2
    
    def test_update_config(self):
        """Test updating configuration"""
        updates = {
            "optimization": {
                "strategy": "memory",
                "enable_cpu_offload": True
            },
            "user_preferences": {
                "default_fps": 30.0,
                "verbose_logging": True
            }
        }
        
        success = self.config_manager.update_config(updates)
        assert success is True
        
        config = self.config_manager.get_config()
        assert config.optimization.strategy == OptimizationStrategy.MEMORY
        assert config.optimization.enable_cpu_offload is True
        assert config.user_preferences.default_fps == 30.0
        assert config.user_preferences.verbose_logging is True
    
    def test_reset_to_defaults(self):
        """Test resetting configuration to defaults"""
        # Modify configuration
        config = self.config_manager.get_config()
        config.optimization.strategy = OptimizationStrategy.MEMORY
        self.config_manager.save_config(config)
        
        # Reset to defaults
        success = self.config_manager.reset_to_defaults()
        assert success is True
        
        # Verify reset
        config = self.config_manager.get_config()
        assert config.optimization.strategy == OptimizationStrategy.AUTO
    
    def test_validate_config_valid(self):
        """Test configuration validation with valid config"""
        config = WAN22Config()
        errors = self.config_manager.validate_config(config)
        assert len(errors) == 0
    
    def test_validate_config_invalid(self):
        """Test configuration validation with invalid config"""
        config = WAN22Config()
        config.optimization.max_chunk_size = -1
        config.optimization.vram_threshold_mb = 100
        config.pipeline.pipeline_timeout_seconds = -10
        config.user_preferences.default_fps = -5.0
        
        errors = self.config_manager.validate_config(config)
        assert len(errors) > 0
        assert any("max_chunk_size must be positive" in error for error in errors)
        assert any("vram_threshold_mb should be at least 1024 MB" in error for error in errors)
        assert any("pipeline_timeout_seconds must be positive" in error for error in errors)
        assert any("default_fps must be positive" in error for error in errors)
    
    def test_get_specific_configs(self):
        """Test getting specific configuration sections"""
        opt_config = self.config_manager.get_optimization_config()
        assert isinstance(opt_config, OptimizationConfig)
        
        pipeline_config = self.config_manager.get_pipeline_config()
        assert isinstance(pipeline_config, PipelineConfig)
        
        security_config = self.config_manager.get_security_config()
        assert isinstance(security_config, SecurityConfig)
        
        compat_config = self.config_manager.get_compatibility_config()
        assert isinstance(compat_config, CompatibilityConfig)
        
        user_prefs = self.config_manager.get_user_preferences()
        assert isinstance(user_prefs, UserPreferences)
    
    def test_export_config(self):
        """Test exporting configuration"""
        export_path = Path(self.temp_dir) / "exported_config.json"
        
        success = self.config_manager.export_config(str(export_path))
        assert success is True
        assert export_path.exists()
        
        # Verify exported content
        with open(export_path, 'r') as f:
            exported_data = json.load(f)
        assert "version" in exported_data
        assert "optimization" in exported_data
        assert "pipeline" in exported_data
    
    def test_export_config_without_sensitive(self):
        """Test exporting configuration without sensitive data"""
        export_path = Path(self.temp_dir) / "exported_config_safe.json"
        
        success = self.config_manager.export_config(str(export_path), include_sensitive=False)
        assert success is True
        assert export_path.exists()
        
        # Verify sensitive data is removed
        with open(export_path, 'r') as f:
            exported_data = json.load(f)
        assert "trusted_sources" not in exported_data.get("security", {})
    
    def test_import_config(self):
        """Test importing configuration"""
        # Create import data
        import_data = {
"optimization": {
                "strategy": "performance",
                "enable_mixed_precision": False
            },
            "user_preferences": {
                "default_fps": 60.0
            }
        }
        
        import_path = Path(self.temp_dir) / "import_config.json"
        with open(import_path, 'w') as f:
            json.dump(import_data, f)
        
        # Import configuration
        success = self.config_manager.import_config(str(import_path), merge=True)
        assert success is True
        
        # Verify imported values
        config = self.config_manager.get_config()
        assert config.optimization.strategy == OptimizationStrategy.PERFORMANCE
        assert config.optimization.enable_mixed_precision is False
        assert config.user_preferences.default_fps == 60.0
    
    def test_import_config_replace(self):
        """Test importing configuration with replace mode"""
        # Modify current config
        self.config_manager.update_config({
            "user_preferences": {"verbose_logging": True}
        })
        
        # Create minimal import data
        import_data = {
"version": "1.0.0",
            "optimization": {"strategy": "memory"}
        }
        
        import_path = Path(self.temp_dir) / "import_config_replace.json"
        with open(import_path, 'w') as f:
            json.dump(import_data, f)
        
        # Import with replace mode
        success = self.config_manager.import_config(str(import_path), merge=False)
        assert success is True
        
        # Verify replacement (should have default values except imported ones)
        config = self.config_manager.get_config()
        assert config.optimization.strategy == OptimizationStrategy.MEMORY
        assert config.user_preferences.verbose_logging is False  # Should be default
    
    def test_corrupted_config_file(self):
        """Test handling corrupted configuration file"""
        # Create corrupted config file
        with open(self.config_manager.config_path, 'w') as f:
            f.write("invalid json content {")
        
        # Should create default config when loading fails
        config = self.config_manager.load_config()
        assert isinstance(config, WAN22Config)
        assert config.version == "1.0.0"
    
    def test_config_migration(self):
        """Test configuration migration from older versions"""
        # Create old version config
        old_config = {
            "version": "0.9.0",
            "optimization": {"strategy": "auto"}
        }
        
        with open(self.config_manager.config_path, 'w') as f:
            json.dump(old_config, f)
        
        # Load should trigger migration
        config = self.config_manager.load_config()
        assert config.version == "1.0.0"


class TestGlobalFunctions:
    """Test global configuration functions"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        # Reset global config manager
        import wan22_config_manager
        wan22_config_manager._config_manager = None
    
    def teardown_method(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        # Reset global config manager
        import wan22_config_manager
        wan22_config_manager._config_manager = None
    
    def test_get_config_manager(self):
        """Test getting global configuration manager"""
        manager1 = get_config_manager(self.temp_dir)
        manager2 = get_config_manager(self.temp_dir)
        
        # Should return same instance
        assert manager1 is manager2
        assert isinstance(manager1, ConfigurationManager)
    
    def test_get_config(self):
        """Test getting global configuration"""
        with patch('wan22_config_manager.get_config_manager') as mock_get_manager:
            mock_manager = MagicMock()
            mock_config = WAN22Config()
            mock_manager.get_config.return_value = mock_config
            mock_get_manager.return_value = mock_manager
            
            config = get_config()
            assert config is mock_config
            mock_manager.get_config.assert_called_once()
    
    def test_save_config(self):
        """Test saving global configuration"""
        with patch('wan22_config_manager.get_config_manager') as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.save_config.return_value = True
            mock_get_manager.return_value = mock_manager
            
            config = WAN22Config()
            result = save_config(config)
            
            assert result is True
            mock_manager.save_config.assert_called_once_with(config)
    
    def test_update_config(self):
        """Test updating global configuration"""
        with patch('wan22_config_manager.get_config_manager') as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.update_config.return_value = True
            mock_get_manager.return_value = mock_manager
            
            updates = {"optimization": {"strategy": "memory"}}
            result = update_config(updates)
            
            assert result is True
            mock_manager.update_config.assert_called_once_with(updates)


class TestConfigurationIntegration:
    """Integration tests for configuration system"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = ConfigurationManager(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_complete_configuration_workflow(self):
        """Test complete configuration workflow"""
        # 1. Load default configuration
        config = self.config_manager.load_config()
        assert config.optimization.strategy == OptimizationStrategy.AUTO
        
        # 2. Update optimization settings
        updates = {
            "optimization": {
                "strategy": "memory",
                "enable_cpu_offload": True,
                "max_chunk_size": 4
            }
        }
        success = self.config_manager.update_config(updates)
        assert success is True
        
        # 3. Verify updates
        config = self.config_manager.get_config()
        assert config.optimization.strategy == OptimizationStrategy.MEMORY
        assert config.optimization.enable_cpu_offload is True
        assert config.optimization.max_chunk_size == 4
        
        # 4. Export configuration
        export_path = Path(self.temp_dir) / "exported.json"
        success = self.config_manager.export_config(str(export_path))
        assert success is True
        
        # 5. Reset to defaults
        success = self.config_manager.reset_to_defaults()
        assert success is True
        config = self.config_manager.get_config()
        assert config.optimization.strategy == OptimizationStrategy.AUTO
        
        # 6. Import previous configuration
        success = self.config_manager.import_config(str(export_path))
        assert success is True
        config = self.config_manager.get_config()
        assert config.optimization.strategy == OptimizationStrategy.MEMORY
    
    def test_configuration_persistence(self):
        """Test configuration persistence across manager instances"""
        # Create and save configuration with first manager
        config = WAN22Config()
        config.user_preferences.default_fps = 60.0
        config.optimization.strategy = OptimizationStrategy.PERFORMANCE
        
        success = self.config_manager.save_config(config)
        assert success is True
        
        # Create new manager instance and load configuration
        new_manager = ConfigurationManager(self.temp_dir)
        loaded_config = new_manager.load_config()
        
        assert loaded_config.user_preferences.default_fps == 60.0
        assert loaded_config.optimization.strategy == OptimizationStrategy.PERFORMANCE
    
    def test_configuration_validation_integration(self):
        """Test configuration validation in real scenarios"""
        # Test valid configuration
        config = WAN22Config()
        config.optimization.vram_threshold_mb = 16384
        config.pipeline.max_retry_attempts = 5
        config.user_preferences.max_concurrent_generations = 2
        
        errors = self.config_manager.validate_config(config)
        assert len(errors) == 0
        
        # Test invalid configuration
        config.optimization.max_chunk_size = 0
        config.pipeline.pipeline_timeout_seconds = -1
        config.user_preferences.default_fps = 0
        
        errors = self.config_manager.validate_config(config)
        assert len(errors) >= 3
        
        # Should not save invalid configuration
        success = self.config_manager.update_config({
            "optimization": {"max_chunk_size": -5}
        })
        assert success is False


if __name__ == "__main__":
    pytest.main([__file__])
