"""
Tests for WAN22 Configuration Presets

This module tests the predefined configuration presets and preset management functionality.
"""

import pytest
import tempfile
import shutil
from pathlib import Path

from wan22_config_presets import (
    ConfigurationPresets,
    apply_preset,
    get_preset_comparison,
    recommend_preset
)
from wan22_config_manager import (
    ConfigurationManager,
    WAN22Config,
    OptimizationStrategy,
    PipelineSelectionMode,
    SecurityLevel
)


class TestConfigurationPresets:
    """Test configuration presets"""
    
    def test_get_preset_names(self):
        """Test getting list of preset names"""
        names = ConfigurationPresets.get_preset_names()
        
        assert isinstance(names, list)
        assert len(names) > 0
        assert "default" in names
        assert "high_performance" in names
        assert "memory_optimized" in names
        assert "balanced" in names
        assert "development" in names
        assert "production" in names
    
    def test_get_preset_descriptions(self):
        """Test getting preset descriptions"""
        names = ConfigurationPresets.get_preset_names()
        
        for name in names:
            description = ConfigurationPresets.get_preset_description(name)
            assert isinstance(description, str)
            assert len(description) > 0
            assert description != "No description available"
    
    def test_get_unknown_preset(self):
        """Test getting unknown preset raises error"""
        with pytest.raises(ValueError, match="Unknown preset"):
            ConfigurationPresets.get_preset("unknown_preset")

        assert True  # TODO: Add proper assertion
    
    def test_default_preset(self):
        """Test default preset"""
        config = ConfigurationPresets.get_preset("default")
        
        assert isinstance(config, WAN22Config)
        assert config.optimization.strategy == OptimizationStrategy.AUTO
        assert config.pipeline.selection_mode == PipelineSelectionMode.AUTO
        assert config.security.security_level == SecurityLevel.MODERATE
    
    def test_high_performance_preset(self):
        """Test high performance preset"""
        config = ConfigurationPresets.get_preset("high_performance")
        
        assert config.optimization.strategy == OptimizationStrategy.PERFORMANCE
        assert config.optimization.enable_cpu_offload is False
        assert config.optimization.enable_chunked_processing is False
        assert config.optimization.vram_threshold_mb == 16384
        assert config.pipeline.enable_fallback is False
        assert config.pipeline.max_retry_attempts == 1
        assert config.user_preferences.enable_progress_indicators is False
    
    def test_memory_optimized_preset(self):
        """Test memory optimized preset"""
        config = ConfigurationPresets.get_preset("memory_optimized")
        
        assert config.optimization.strategy == OptimizationStrategy.MEMORY
        assert config.optimization.enable_cpu_offload is True
        assert config.optimization.cpu_offload_strategy == "sequential"
        assert config.optimization.enable_chunked_processing is True
        assert config.optimization.max_chunk_size == 4
        assert config.optimization.vram_threshold_mb == 4096
        assert config.optimization.enable_vae_tiling is True
        assert config.optimization.vae_tile_size == 256
        assert config.pipeline.enable_fallback is True
        assert "reduced_functionality" in config.pipeline.fallback_strategies
    
    def test_balanced_preset(self):
        """Test balanced preset"""
        config = ConfigurationPresets.get_preset("balanced")
        
        assert config.optimization.strategy == OptimizationStrategy.BALANCED
        assert config.optimization.vram_threshold_mb == 8192
        assert config.pipeline.enable_fallback is True
        assert len(config.pipeline.fallback_strategies) == 2
    
    def test_development_preset(self):
        """Test development preset"""
        config = ConfigurationPresets.get_preset("development")
        
        assert config.user_preferences.verbose_logging is True
        assert config.user_preferences.auto_cleanup_temp_files is False
        assert config.compatibility.cache_detection_results is False
        assert config.compatibility.diagnostic_output_dir == "dev_diagnostics"
        assert config.experimental_features.get("debug_mode") is True
        assert config.experimental_features.get("advanced_diagnostics") is True
    
    def test_production_preset(self):
        """Test production preset"""
        config = ConfigurationPresets.get_preset("production")
        
        assert config.optimization.strategy == OptimizationStrategy.BALANCED
        assert config.security.security_level == SecurityLevel.MODERATE
        assert config.compatibility.cache_detection_results is True
        assert config.compatibility.diagnostic_output_dir == "production_diagnostics"
        assert config.user_preferences.verbose_logging is False
        assert config.user_preferences.auto_cleanup_temp_files is True
    
    def test_security_focused_preset(self):
        """Test security focused preset"""
        config = ConfigurationPresets.get_preset("security_focused")
        
        assert config.pipeline.selection_mode == PipelineSelectionMode.MANUAL
        assert config.pipeline.enable_fallback is False
        assert config.pipeline.max_retry_attempts == 1
        assert config.security.security_level == SecurityLevel.STRICT
        assert config.security.trust_remote_code is False
        assert len(config.security.trusted_sources) == 0
        assert config.security.enable_sandboxing is True
        assert config.security.allow_local_code_execution is False
        assert config.security.code_signature_verification is True
        assert config.compatibility.strict_validation is True
        assert config.compatibility.cache_detection_results is False
        assert config.compatibility.enable_diagnostic_collection is False
    
    def test_low_end_hardware_preset(self):
        """Test low-end hardware preset"""
        config = ConfigurationPresets.get_preset("low_end_hardware")
        
        assert config.optimization.strategy == OptimizationStrategy.MEMORY
        assert config.optimization.enable_cpu_offload is True
        assert config.optimization.cpu_offload_strategy == "full"
        assert config.optimization.enable_chunked_processing is True
        assert config.optimization.max_chunk_size == 2
        assert config.optimization.vram_threshold_mb == 4096
        assert config.optimization.enable_vae_tiling is True
        assert config.optimization.vae_tile_size == 128
        assert config.pipeline.pipeline_timeout_seconds == 900
        assert config.pipeline.max_retry_attempts == 5
        assert "reduced_functionality" in config.pipeline.fallback_strategies
    
    def test_high_end_hardware_preset(self):
        """Test high-end hardware preset"""
        config = ConfigurationPresets.get_preset("high_end_hardware")
        
        assert config.optimization.strategy == OptimizationStrategy.PERFORMANCE
        assert config.optimization.enable_cpu_offload is False
        assert config.optimization.enable_chunked_processing is False
        assert config.optimization.vram_threshold_mb == 24576
        assert config.optimization.enable_vae_tiling is False
        assert config.pipeline.enable_fallback is False
        assert config.pipeline.max_retry_attempts == 1
        assert config.user_preferences.max_concurrent_generations == 2
        assert config.user_preferences.default_fps == 30.0
    
    def test_enterprise_preset(self):
        """Test enterprise preset"""
        config = ConfigurationPresets.get_preset("enterprise")
        
        assert config.security.security_level == SecurityLevel.STRICT
        assert config.security.trust_remote_code is False
        assert "internal.company.com" in config.security.trusted_sources
        assert config.security.enable_sandboxing is True
        assert config.security.code_signature_verification is True
        assert config.compatibility.strict_validation is True
        assert config.compatibility.detection_cache_ttl_hours == 12
        assert config.user_preferences.verbose_logging is True
        assert config.user_preferences.notification_preferences["optimization_suggestions"] is False
    
    def test_research_preset(self):
        """Test research preset"""
        config = ConfigurationPresets.get_preset("research")
        
        assert config.pipeline.max_retry_attempts == 5
        assert config.compatibility.cache_detection_results is False
        assert config.compatibility.diagnostic_output_dir == "research_diagnostics"
        assert config.user_preferences.verbose_logging is True
        assert config.user_preferences.auto_cleanup_temp_files is False
        assert config.experimental_features.get("advanced_diagnostics") is True
        assert config.experimental_features.get("performance_profiling") is True
        assert config.experimental_features.get("experimental_optimizations") is True
    
    def test_content_creation_preset(self):
        """Test content creation preset"""
        config = ConfigurationPresets.get_preset("content_creation")
        
        assert config.optimization.vram_threshold_mb == 12288
        assert config.optimization.enable_vae_tiling is False
        assert config.pipeline.pipeline_timeout_seconds == 900
        assert config.user_preferences.default_fps == 30.0
        assert config.user_preferences.preferred_video_codec == "h264"
        assert config.user_preferences.notification_preferences["optimization_suggestions"] is True
    
    def test_all_presets_valid(self):
        """Test that all presets are valid configurations"""
        names = ConfigurationPresets.get_preset_names()
        
        for name in names:
            config = ConfigurationPresets.get_preset(name)
            assert isinstance(config, WAN22Config)
            assert config.version == "1.0.0"
            assert hasattr(config, 'optimization')
            assert hasattr(config, 'pipeline')
            assert hasattr(config, 'security')
            assert hasattr(config, 'compatibility')
            assert hasattr(config, 'user_preferences')


class TestPresetManagement:
    """Test preset management functions"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = ConfigurationManager(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_apply_preset(self):
        """Test applying a preset to configuration manager"""
        success = apply_preset(self.config_manager, "high_performance")
        assert success is True
        
        config = self.config_manager.get_config()
        assert config.optimization.strategy == OptimizationStrategy.PERFORMANCE
        assert config.pipeline.enable_fallback is False
    
    def test_apply_unknown_preset(self):
        """Test applying unknown preset returns False"""
        success = apply_preset(self.config_manager, "unknown_preset")
        assert success is False
    
    def test_get_preset_comparison(self):
        """Test comparing two presets"""
        comparison = get_preset_comparison("high_performance", "memory_optimized")
        
        assert "preset1" in comparison
        assert "preset2" in comparison
        assert "differences" in comparison
        assert comparison["preset1"] == "high_performance"
        assert comparison["preset2"] == "memory_optimized"
        
        differences = comparison["differences"]
        assert len(differences) > 0
        
        # Should show strategy difference
        assert "optimization.strategy" in differences
        assert differences["optimization.strategy"]["preset1"] == "performance"
        assert differences["optimization.strategy"]["preset2"] == "memory"
    
    def test_get_preset_comparison_same(self):
        """Test comparing same preset shows no differences"""
        comparison = get_preset_comparison("default", "default")
        
        assert comparison["differences"] == {}
    
    def test_get_preset_comparison_invalid(self):
        """Test comparing invalid presets returns error"""
        comparison = get_preset_comparison("invalid1", "invalid2")
        
        assert "error" in comparison
    
    def test_recommend_preset_low_vram(self):
        """Test preset recommendation for low VRAM"""
        preset = recommend_preset(4096, "general")
        assert preset == "low_end_hardware"
    
    def test_recommend_preset_high_vram(self):
        """Test preset recommendation for high VRAM"""
        preset = recommend_preset(20480, "general")
        assert preset == "high_end_hardware"
    
    def test_recommend_preset_medium_vram(self):
        """Test preset recommendation for medium VRAM"""
        preset = recommend_preset(8192, "general")
        assert preset == "balanced"
    
    def test_recommend_preset_development(self):
        """Test preset recommendation for development use case"""
        preset = recommend_preset(8192, "development")
        assert preset == "development"
    
    def test_recommend_preset_production(self):
        """Test preset recommendation for production use case"""
        preset = recommend_preset(8192, "production")
        assert preset == "production"
    
    def test_recommend_preset_content_creation(self):
        """Test preset recommendation for content creation use case"""
        preset = recommend_preset(8192, "content_creation")
        assert preset == "content_creation"
    
    def test_recommend_preset_research(self):
        """Test preset recommendation for research use case"""
        preset = recommend_preset(8192, "research")
        assert preset == "research"
    
    def test_recommend_preset_enterprise(self):
        """Test preset recommendation for enterprise use case"""
        preset = recommend_preset(8192, "enterprise")
        assert preset == "enterprise"
    
    def test_recommend_preset_security(self):
        """Test preset recommendation for security use case"""
        preset = recommend_preset(8192, "security")
        assert preset == "security_focused"


class TestPresetIntegration:
    """Integration tests for preset system"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = ConfigurationManager(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_preset_workflow(self):
        """Test complete preset workflow"""
        # 1. Apply development preset
        success = apply_preset(self.config_manager, "development")
        assert success is True
        
        config = self.config_manager.get_config()
        assert config.user_preferences.verbose_logging is True
        assert config.compatibility.cache_detection_results is False
        
        # 2. Switch to production preset
        success = apply_preset(self.config_manager, "production")
        assert success is True
        
        config = self.config_manager.get_config()
        assert config.user_preferences.verbose_logging is False
        assert config.compatibility.cache_detection_results is True
        
        # 3. Compare presets
        comparison = get_preset_comparison("development", "production")
        assert len(comparison["differences"]) > 0
        
        # 4. Get recommendation
        preset = recommend_preset(8192, "development")
        assert preset == "development"
    
    def test_preset_validation(self):
        """Test that all presets pass validation"""
        names = ConfigurationPresets.get_preset_names()
        
        for name in names:
            # Apply preset
            success = apply_preset(self.config_manager, name)
            assert success is True, f"Failed to apply preset: {name}"
            
            # Validate configuration
            config = self.config_manager.get_config()
            errors = self.config_manager.validate_config(config)
            assert len(errors) == 0, f"Preset {name} failed validation: {errors}"
    
    def test_preset_persistence(self):
        """Test preset persistence across manager instances"""
        # Apply preset with first manager
        success = apply_preset(self.config_manager, "memory_optimized")
        assert success is True
        
        # Create new manager and verify preset persisted
        new_manager = ConfigurationManager(self.temp_dir)
        config = new_manager.get_config()
        
        assert config.optimization.strategy == OptimizationStrategy.MEMORY
        assert config.optimization.enable_cpu_offload is True
        assert config.optimization.enable_chunked_processing is True
    
    def test_preset_customization(self):
        """Test customizing preset after application"""
        # Apply base preset
        success = apply_preset(self.config_manager, "balanced")
        assert success is True
        
        # Customize configuration
        updates = {
            "user_preferences": {
                "verbose_logging": True,
                "default_fps": 60.0
            },
            "optimization": {
                "max_chunk_size": 16
            }
        }
        
        success = self.config_manager.update_config(updates)
        assert success is True
        
        # Verify customizations
        config = self.config_manager.get_config()
        assert config.user_preferences.verbose_logging is True
        assert config.user_preferences.default_fps == 60.0
        assert config.optimization.max_chunk_size == 16
        
        # Should still have balanced strategy
        assert config.optimization.strategy == OptimizationStrategy.BALANCED


if __name__ == "__main__":
    pytest.main([__file__])