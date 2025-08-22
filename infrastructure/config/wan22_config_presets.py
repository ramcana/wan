"""
WAN22 Configuration Presets

This module provides predefined configuration presets for common use cases,
hardware configurations, and user scenarios.
"""

from typing import Dict, Any, List
from dataclasses import asdict

from wan22_config_manager import (
    WAN22Config,
    OptimizationConfig,
    PipelineConfig,
    SecurityConfig,
    CompatibilityConfig,
    UserPreferences,
    OptimizationStrategy,
    PipelineSelectionMode,
    SecurityLevel
)


class ConfigurationPresets:
    """Predefined configuration presets for common scenarios"""
    
    @staticmethod
    def get_preset_names() -> List[str]:
        """Get list of available preset names"""
        return [
            "default",
            "high_performance",
            "memory_optimized",
            "balanced",
            "development",
            "production",
            "security_focused",
            "low_end_hardware",
            "high_end_hardware",
            "enterprise",
            "research",
            "content_creation"
        ]
    
    @staticmethod
    def get_preset(preset_name: str) -> WAN22Config:
        """Get configuration preset by name
        
        Args:
            preset_name: Name of the preset to retrieve
            
        Returns:
            WAN22Config object with preset configuration
            
        Raises:
            ValueError: If preset name is not recognized
        """
        presets = {
            "default": ConfigurationPresets._default_preset,
            "high_performance": ConfigurationPresets._high_performance_preset,
            "memory_optimized": ConfigurationPresets._memory_optimized_preset,
            "balanced": ConfigurationPresets._balanced_preset,
            "development": ConfigurationPresets._development_preset,
            "production": ConfigurationPresets._production_preset,
            "security_focused": ConfigurationPresets._security_focused_preset,
            "low_end_hardware": ConfigurationPresets._low_end_hardware_preset,
            "high_end_hardware": ConfigurationPresets._high_end_hardware_preset,
            "enterprise": ConfigurationPresets._enterprise_preset,
            "research": ConfigurationPresets._research_preset,
            "content_creation": ConfigurationPresets._content_creation_preset
        }
        
        if preset_name not in presets:
            raise ValueError(f"Unknown preset: {preset_name}. Available presets: {list(presets.keys())}")
        
        return presets[preset_name]()
    
    @staticmethod
    def get_preset_description(preset_name: str) -> str:
        """Get description of a configuration preset
        
        Args:
            preset_name: Name of the preset
            
        Returns:
            Description of the preset
        """
        descriptions = {
            "default": "Standard configuration suitable for most users with moderate hardware",
            "high_performance": "Optimized for maximum generation speed on high-end hardware",
            "memory_optimized": "Minimizes VRAM usage for systems with limited GPU memory",
            "balanced": "Balanced configuration between performance and memory usage",
            "development": "Development-friendly settings with verbose logging and debugging",
            "production": "Production-ready configuration with reliability and monitoring",
            "security_focused": "Enhanced security settings for enterprise environments",
            "low_end_hardware": "Optimized for systems with 4-8GB VRAM and limited resources",
            "high_end_hardware": "Takes advantage of high-end GPUs with 16GB+ VRAM",
            "enterprise": "Enterprise configuration with security, monitoring, and compliance",
            "research": "Research-oriented settings with detailed logging and diagnostics",
            "content_creation": "Optimized for content creators with quality-focused settings"
        }
        
        return descriptions.get(preset_name, "No description available")
    
    @staticmethod
    def _default_preset() -> WAN22Config:
        """Default configuration preset"""
        return WAN22Config()
    
    @staticmethod
    def _high_performance_preset() -> WAN22Config:
        """High performance configuration preset"""
        config = WAN22Config()
        
        # Optimization for performance
        config.optimization = OptimizationConfig(
            strategy=OptimizationStrategy.PERFORMANCE,
            enable_mixed_precision=True,
            enable_cpu_offload=False,
            enable_chunked_processing=False,
            vram_threshold_mb=16384,
            enable_vae_tiling=False
        )
        
        # Pipeline settings for performance
        config.pipeline = PipelineConfig(
            selection_mode=PipelineSelectionMode.AUTO,
            enable_fallback=False,  # Strict mode for performance
            pipeline_timeout_seconds=600,
            max_retry_attempts=1
        )
        
        # User preferences for performance
        config.user_preferences = UserPreferences(
            max_concurrent_generations=1,
            auto_cleanup_temp_files=True,
            enable_progress_indicators=False  # Reduce overhead
        )
        
        return config
    
    @staticmethod
    def _memory_optimized_preset() -> WAN22Config:
        """Memory optimized configuration preset"""
        config = WAN22Config()
        
        # Optimization for memory
        config.optimization = OptimizationConfig(
            strategy=OptimizationStrategy.MEMORY,
            enable_mixed_precision=True,
            enable_cpu_offload=True,
            cpu_offload_strategy="sequential",
            enable_chunked_processing=True,
            max_chunk_size=4,
            vram_threshold_mb=4096,
            enable_vae_tiling=True,
            vae_tile_size=256
        )
        
        # Pipeline settings for memory efficiency
        config.pipeline = PipelineConfig(
            selection_mode=PipelineSelectionMode.AUTO,
            enable_fallback=True,
            fallback_strategies=["component_isolation", "alternative_model", "reduced_functionality"]
        )
        
        # User preferences for memory efficiency
        config.user_preferences = UserPreferences(
            max_concurrent_generations=1,
            auto_cleanup_temp_files=True
        )
        
        return config
    
    @staticmethod
    def _balanced_preset() -> WAN22Config:
        """Balanced configuration preset"""
        config = WAN22Config()
        
        # Balanced optimization
        config.optimization = OptimizationConfig(
            strategy=OptimizationStrategy.BALANCED,
            enable_mixed_precision=True,
            enable_cpu_offload=False,
            enable_chunked_processing=False,
            vram_threshold_mb=8192,
            enable_vae_tiling=False
        )
        
        # Balanced pipeline settings
        config.pipeline = PipelineConfig(
            selection_mode=PipelineSelectionMode.AUTO,
            enable_fallback=True,
            fallback_strategies=["component_isolation", "alternative_model"]
        )
        
        return config
    
    @staticmethod
    def _development_preset() -> WAN22Config:
        """Development configuration preset"""
        config = WAN22Config()
        
        # Development-friendly optimization
        config.optimization = OptimizationConfig(
            strategy=OptimizationStrategy.BALANCED,
            enable_mixed_precision=True,
            enable_cpu_offload=False,
            vram_threshold_mb=8192
        )
        
        # Development pipeline settings
        config.pipeline = PipelineConfig(
            selection_mode=PipelineSelectionMode.AUTO,
            enable_fallback=True,
            pipeline_timeout_seconds=600,
            max_retry_attempts=3
        )
        
        # Development compatibility settings
        config.compatibility = CompatibilityConfig(
            enable_architecture_detection=True,
            enable_vae_validation=True,
            enable_component_validation=True,
            strict_validation=False,
            cache_detection_results=False,  # Always fresh detection
            enable_diagnostic_collection=True,
            diagnostic_output_dir="dev_diagnostics"
        )
        
        # Development user preferences
        config.user_preferences = UserPreferences(
            verbose_logging=True,
            enable_progress_indicators=True,
            auto_cleanup_temp_files=False,  # Keep files for debugging
            notification_preferences={
                "generation_complete": True,
                "error_notifications": True,
                "optimization_suggestions": True
            }
        )
        
        # Enable experimental features
        config.experimental_features = {
            "advanced_diagnostics": True,
            "performance_profiling": True,
            "debug_mode": True
        }
        
        return config
    
    @staticmethod
    def _production_preset() -> WAN22Config:
        """Production configuration preset"""
        config = WAN22Config()
        
        # Production optimization
        config.optimization = OptimizationConfig(
            strategy=OptimizationStrategy.BALANCED,
            enable_mixed_precision=True,
            enable_cpu_offload=False,
            vram_threshold_mb=8192
        )
        
        # Production pipeline settings
        config.pipeline = PipelineConfig(
            selection_mode=PipelineSelectionMode.AUTO,
            enable_fallback=True,
            pipeline_timeout_seconds=300,
            max_retry_attempts=3
        )
        
        # Production security settings
        config.security = SecurityConfig(
            security_level=SecurityLevel.MODERATE,
            trust_remote_code=True,
            trusted_sources=["huggingface.co", "hf.co"],
            enable_sandboxing=False,
            allow_local_code_execution=True
        )
        
        # Production compatibility settings
        config.compatibility = CompatibilityConfig(
            enable_architecture_detection=True,
            enable_vae_validation=True,
            enable_component_validation=True,
            strict_validation=False,
            cache_detection_results=True,
            detection_cache_ttl_hours=24,
            enable_diagnostic_collection=True,
            diagnostic_output_dir="production_diagnostics"
        )
        
        # Production user preferences
        config.user_preferences = UserPreferences(
            verbose_logging=False,
            enable_progress_indicators=True,
            auto_cleanup_temp_files=True,
            max_concurrent_generations=1
        )
        
        return config
    
    @staticmethod
    def _security_focused_preset() -> WAN22Config:
        """Security focused configuration preset"""
        config = WAN22Config()
        
        # Security-focused optimization (conservative)
        config.optimization = OptimizationConfig(
            strategy=OptimizationStrategy.BALANCED,
            enable_mixed_precision=True,
            enable_cpu_offload=False,
            vram_threshold_mb=8192
        )
        
        # Security-focused pipeline settings
        config.pipeline = PipelineConfig(
            selection_mode=PipelineSelectionMode.MANUAL,  # Manual control
            enable_fallback=False,  # Strict mode
            pipeline_timeout_seconds=180,  # Shorter timeout
            max_retry_attempts=1
        )
        
        # Strict security settings
        config.security = SecurityConfig(
            security_level=SecurityLevel.STRICT,
            trust_remote_code=False,
            trusted_sources=[],  # No trusted sources by default
            enable_sandboxing=True,
            sandbox_timeout_seconds=30,
            allow_local_code_execution=False,
            code_signature_verification=True
        )
        
        # Security-focused compatibility settings
        config.compatibility = CompatibilityConfig(
            enable_architecture_detection=True,
            enable_vae_validation=True,
            enable_component_validation=True,
            strict_validation=True,
            cache_detection_results=False,  # Always fresh validation
            enable_diagnostic_collection=False  # No diagnostic collection
        )
        
        return config
    
    @staticmethod
    def _low_end_hardware_preset() -> WAN22Config:
        """Low-end hardware configuration preset"""
        config = WAN22Config()
        
        # Aggressive memory optimization
        config.optimization = OptimizationConfig(
            strategy=OptimizationStrategy.MEMORY,
            enable_mixed_precision=True,
            enable_cpu_offload=True,
            cpu_offload_strategy="full",
            enable_chunked_processing=True,
            max_chunk_size=2,
            vram_threshold_mb=4096,
            enable_vae_tiling=True,
            vae_tile_size=128
        )
        
        # Conservative pipeline settings
        config.pipeline = PipelineConfig(
            selection_mode=PipelineSelectionMode.AUTO,
            enable_fallback=True,
            fallback_strategies=["component_isolation", "alternative_model", "reduced_functionality"],
            pipeline_timeout_seconds=900,  # Longer timeout for slow hardware
            max_retry_attempts=5
        )
        
        # Low-end hardware user preferences
        config.user_preferences = UserPreferences(
            max_concurrent_generations=1,
            auto_cleanup_temp_files=True,
            enable_progress_indicators=True
        )
        
        return config
    
    @staticmethod
    def _high_end_hardware_preset() -> WAN22Config:
        """High-end hardware configuration preset"""
        config = WAN22Config()
        
        # Maximum performance optimization
        config.optimization = OptimizationConfig(
            strategy=OptimizationStrategy.PERFORMANCE,
            enable_mixed_precision=True,
            enable_cpu_offload=False,
            enable_chunked_processing=False,
            vram_threshold_mb=24576,  # 24GB+ VRAM
            enable_vae_tiling=False
        )
        
        # High-performance pipeline settings
        config.pipeline = PipelineConfig(
            selection_mode=PipelineSelectionMode.AUTO,
            enable_fallback=False,  # Strict mode for reliability
            pipeline_timeout_seconds=300,
            max_retry_attempts=1
        )
        
        # High-end hardware user preferences
        config.user_preferences = UserPreferences(
            max_concurrent_generations=2,  # Can handle multiple generations
            auto_cleanup_temp_files=True,
            enable_progress_indicators=True,
            default_fps=30.0  # Higher quality output
        )
        
        return config
    
    @staticmethod
    def _enterprise_preset() -> WAN22Config:
        """Enterprise configuration preset"""
        config = WAN22Config()
        
        # Enterprise optimization
        config.optimization = OptimizationConfig(
            strategy=OptimizationStrategy.BALANCED,
            enable_mixed_precision=True,
            enable_cpu_offload=False,
            vram_threshold_mb=12288
        )
        
        # Enterprise pipeline settings
        config.pipeline = PipelineConfig(
            selection_mode=PipelineSelectionMode.AUTO,
            enable_fallback=True,
            pipeline_timeout_seconds=300,
            max_retry_attempts=3
        )
        
        # Enterprise security settings
        config.security = SecurityConfig(
            security_level=SecurityLevel.STRICT,
            trust_remote_code=False,
            trusted_sources=["internal.company.com"],
            enable_sandboxing=True,
            allow_local_code_execution=False,
            code_signature_verification=True
        )
        
        # Enterprise compatibility settings
        config.compatibility = CompatibilityConfig(
            enable_architecture_detection=True,
            enable_vae_validation=True,
            enable_component_validation=True,
            strict_validation=True,
            cache_detection_results=True,
            detection_cache_ttl_hours=12,
            enable_diagnostic_collection=True,
            diagnostic_output_dir="enterprise_diagnostics"
        )
        
        # Enterprise user preferences
        config.user_preferences = UserPreferences(
            verbose_logging=True,
            enable_progress_indicators=True,
            auto_cleanup_temp_files=True,
            max_concurrent_generations=1,
            notification_preferences={
                "generation_complete": True,
                "error_notifications": True,
                "optimization_suggestions": False  # Reduce noise
            }
        )
        
        return config
    
    @staticmethod
    def _research_preset() -> WAN22Config:
        """Research configuration preset"""
        config = WAN22Config()
        
        # Research optimization (balanced for experimentation)
        config.optimization = OptimizationConfig(
            strategy=OptimizationStrategy.BALANCED,
            enable_mixed_precision=True,
            enable_cpu_offload=False,
            vram_threshold_mb=8192
        )
        
        # Research pipeline settings
        config.pipeline = PipelineConfig(
            selection_mode=PipelineSelectionMode.AUTO,
            enable_fallback=True,
            pipeline_timeout_seconds=600,
            max_retry_attempts=5  # More retries for experimentation
        )
        
        # Research compatibility settings (comprehensive diagnostics)
        config.compatibility = CompatibilityConfig(
            enable_architecture_detection=True,
            enable_vae_validation=True,
            enable_component_validation=True,
            strict_validation=False,
            cache_detection_results=False,  # Always fresh for research
            enable_diagnostic_collection=True,
            diagnostic_output_dir="research_diagnostics"
        )
        
        # Research user preferences
        config.user_preferences = UserPreferences(
            verbose_logging=True,
            enable_progress_indicators=True,
            auto_cleanup_temp_files=False,  # Keep all files for analysis
            max_concurrent_generations=1,
            notification_preferences={
                "generation_complete": True,
                "error_notifications": True,
                "optimization_suggestions": True
            }
        )
        
        # Enable all experimental features
        config.experimental_features = {
            "advanced_diagnostics": True,
            "performance_profiling": True,
            "debug_mode": True,
            "detailed_logging": True,
            "experimental_optimizations": True
        }
        
        return config
    
    @staticmethod
    def _content_creation_preset() -> WAN22Config:
        """Content creation configuration preset"""
        config = WAN22Config()
        
        # Content creation optimization (quality focused)
        config.optimization = OptimizationConfig(
            strategy=OptimizationStrategy.BALANCED,
            enable_mixed_precision=True,
            enable_cpu_offload=False,
            vram_threshold_mb=12288,
            enable_vae_tiling=False  # Better quality
        )
        
        # Content creation pipeline settings
        config.pipeline = PipelineConfig(
            selection_mode=PipelineSelectionMode.AUTO,
            enable_fallback=True,
            pipeline_timeout_seconds=900,  # Longer timeout for quality
            max_retry_attempts=3
        )
        
        # Content creation user preferences
        config.user_preferences = UserPreferences(
            default_output_format="mp4",
            preferred_video_codec="h264",
            default_fps=30.0,  # Higher quality
            enable_progress_indicators=True,
            auto_cleanup_temp_files=True,
            max_concurrent_generations=1,
            notification_preferences={
                "generation_complete": True,
                "error_notifications": True,
                "optimization_suggestions": True
            }
        )
        
        return config


def apply_preset(config_manager, preset_name: str) -> bool:
    """Apply a configuration preset to the configuration manager
    
    Args:
        config_manager: ConfigurationManager instance
        preset_name: Name of the preset to apply
        
    Returns:
        True if preset applied successfully, False otherwise
    """
    try:
        preset_config = ConfigurationPresets.get_preset(preset_name)
        return config_manager.save_config(preset_config)
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to apply preset '{preset_name}': {e}")
        return False


def get_preset_comparison(preset1: str, preset2: str) -> Dict[str, Any]:
    """Compare two configuration presets
    
    Args:
        preset1: Name of first preset
        preset2: Name of second preset
        
    Returns:
        Dictionary showing differences between presets
    """
    try:
        config1 = ConfigurationPresets.get_preset(preset1)
        config2 = ConfigurationPresets.get_preset(preset2)
        
        # Use the config manager's method to convert to dict with enum handling
        from wan22_config_manager import ConfigurationManager
        temp_manager = ConfigurationManager()
        dict1 = temp_manager._config_to_dict(config1)
        dict2 = temp_manager._config_to_dict(config2)
        
        differences = {}
        
        def compare_dicts(d1, d2, path=""):
            for key in set(d1.keys()) | set(d2.keys()):
                current_path = f"{path}.{key}" if path else key
                
                if key not in d1:
                    differences[current_path] = {"preset1": None, "preset2": d2[key]}
                elif key not in d2:
                    differences[current_path] = {"preset1": d1[key], "preset2": None}
                elif isinstance(d1[key], dict) and isinstance(d2[key], dict):
                    compare_dicts(d1[key], d2[key], current_path)
                elif d1[key] != d2[key]:
                    differences[current_path] = {"preset1": d1[key], "preset2": d2[key]}
        
        compare_dicts(dict1, dict2)
        
        return {
            "preset1": preset1,
            "preset2": preset2,
            "differences": differences
        }
        
    except Exception as e:
        return {"error": str(e)}


def recommend_preset(vram_mb: int, use_case: str = "general") -> str:
    """Recommend a configuration preset based on hardware and use case
    
    Args:
        vram_mb: Available VRAM in megabytes
        use_case: Use case ("general", "development", "production", "content_creation", "research")
        
    Returns:
        Recommended preset name
    """
    # Hardware-based recommendations
    if vram_mb < 6144:  # Less than 6GB
        hardware_preset = "low_end_hardware"
    elif vram_mb >= 16384:  # 16GB or more
        hardware_preset = "high_end_hardware"
    else:  # 6-16GB
        hardware_preset = "balanced"
    
    # Use case specific recommendations
    use_case_presets = {
        "development": "development",
        "production": "production",
        "content_creation": "content_creation",
        "research": "research",
        "enterprise": "enterprise",
        "security": "security_focused"
    }
    
    if use_case in use_case_presets:
        return use_case_presets[use_case]
    
    # Default to hardware-based recommendation
    return hardware_preset