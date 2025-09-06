"""
WAN22 Model Compatibility Configuration Manager

This module provides comprehensive configuration management for the WAN22 model compatibility system,
including user preferences, optimization strategies, pipeline selection options, and security settings.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Available optimization strategies"""
    AUTO = "auto"
    PERFORMANCE = "performance"
    MEMORY = "memory"
    BALANCED = "balanced"
    CUSTOM = "custom"


class PipelineSelectionMode(Enum):
    """Pipeline selection modes"""
    AUTO = "auto"
    MANUAL = "manual"
    FALLBACK_ENABLED = "fallback_enabled"
    STRICT = "strict"


class SecurityLevel(Enum):
    """Security levels for remote code execution"""
    STRICT = "strict"
    MODERATE = "moderate"
    PERMISSIVE = "permissive"


@dataclass
class OptimizationConfig:
    """Configuration for optimization strategies"""
    strategy: OptimizationStrategy = OptimizationStrategy.AUTO
    enable_mixed_precision: bool = True
    enable_cpu_offload: bool = False
    cpu_offload_strategy: str = "sequential"  # "sequential", "model", "full"
    enable_chunked_processing: bool = False
    max_chunk_size: int = 8
    vram_threshold_mb: int = 8192
    enable_vae_tiling: bool = False
    vae_tile_size: int = 512
    custom_optimizations: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    """Configuration for pipeline selection and management"""
    selection_mode: PipelineSelectionMode = PipelineSelectionMode.AUTO
    preferred_pipeline_class: Optional[str] = None
    enable_fallback: bool = True
    fallback_strategies: List[str] = field(default_factory=lambda: ["component_isolation", "alternative_model"])
    pipeline_timeout_seconds: int = 300
    max_retry_attempts: int = 3
    custom_pipeline_paths: Dict[str, str] = field(default_factory=dict)


@dataclass
class SecurityConfig:
    """Configuration for security and remote code handling"""
    security_level: SecurityLevel = SecurityLevel.MODERATE
    trust_remote_code: bool = True
    trusted_sources: List[str] = field(default_factory=lambda: ["huggingface.co", "hf.co"])
    enable_sandboxing: bool = False
    sandbox_timeout_seconds: int = 60
    allow_local_code_execution: bool = True
    code_signature_verification: bool = False


@dataclass
class CompatibilityConfig:
    """Configuration for compatibility detection and handling"""
    enable_architecture_detection: bool = True
    enable_vae_validation: bool = True
    enable_component_validation: bool = True
    strict_validation: bool = False
    cache_detection_results: bool = True
    detection_cache_ttl_hours: int = 24
    enable_diagnostic_collection: bool = True
    diagnostic_output_dir: str = "diagnostics"


@dataclass
class UserPreferences:
    """User preferences and customization options"""
    default_output_format: str = "mp4"
    preferred_video_codec: str = "h264"
    default_fps: float = 24.0
    enable_progress_indicators: bool = True
    verbose_logging: bool = False
    auto_cleanup_temp_files: bool = True
    max_concurrent_generations: int = 1
    notification_preferences: Dict[str, bool] = field(default_factory=lambda: {
        "generation_complete": True,
        "error_notifications": True,
        "optimization_suggestions": True
    })


@dataclass
class WAN22Config:
    """Main configuration class for WAN22 model compatibility system"""
    version: str = "1.0.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    compatibility: CompatibilityConfig = field(default_factory=CompatibilityConfig)
    user_preferences: UserPreferences = field(default_factory=UserPreferences)
    
    # Advanced configuration options
    experimental_features: Dict[str, bool] = field(default_factory=dict)
    custom_settings: Dict[str, Any] = field(default_factory=dict)


class ConfigurationManager:
    """Manages WAN22 model compatibility configuration"""
    
    DEFAULT_CONFIG_NAME = "wan22_config.json"
    CONFIG_VERSION = "1.0.0"
    
    def __init__(self, config_dir: Optional[str] = None):
        """Initialize configuration manager
        
        Args:
            config_dir: Directory to store configuration files. Defaults to current directory.
        """
        self.config_dir = Path(config_dir) if config_dir else Path.cwd()
        self.config_dir.mkdir(exist_ok=True)
        self.config_path = self.config_dir / self.DEFAULT_CONFIG_NAME
        self._config: Optional[WAN22Config] = None
        
    def load_config(self, config_path: Optional[str] = None) -> WAN22Config:
        """Load configuration from file
        
        Args:
            config_path: Path to configuration file. Uses default if None.
            
        Returns:
            Loaded configuration object
        """
        if config_path:
            path = Path(config_path)
        else:
            path = self.config_path
            
        if not path.exists():
            logger.info(f"Configuration file not found at {path}, creating default configuration")
            config = WAN22Config()
            self.save_config(config, str(path))
            return config
            
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Handle version migration if needed
            if data.get('version') != self.CONFIG_VERSION:
                data = self._migrate_config(data)
                
            config = self._dict_to_config(data)
            config.updated_at = datetime.now().isoformat()
            
            # Validate configuration
            validation_errors = self.validate_config(config)
            if validation_errors:
                logger.warning(f"Configuration validation errors: {validation_errors}")
                
            self._config = config
            return config
            
        except Exception as e:
            logger.error(f"Failed to load configuration from {path}: {e}")
            logger.info("Creating default configuration")
            config = WAN22Config()
            self.save_config(config, str(path))
            return config
    
    def save_config(self, config: WAN22Config, config_path: Optional[str] = None) -> bool:
        """Save configuration to file
        
        Args:
            config: Configuration object to save
            config_path: Path to save configuration. Uses default if None.
            
        Returns:
            True if saved successfully, False otherwise
        """
        if config_path:
            path = Path(config_path)
        else:
            path = self.config_path
            
        try:
            # Update timestamp
            config.updated_at = datetime.now().isoformat()
            
            # Convert to dictionary with enum handling
            config_dict = self._config_to_dict(config)
            
            # Ensure directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save with pretty formatting
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Configuration saved to {path}")
            self._config = config
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration to {path}: {e}")
            return False
    
    def get_config(self) -> WAN22Config:
        """Get current configuration, loading from file if not already loaded"""
        if self._config is None:
            self._config = self.load_config()
        return self._config
    
    def update_config(self, updates: Dict[str, Any]) -> bool:
        """Update configuration with new values
        
        Args:
            updates: Dictionary of configuration updates
            
        Returns:
            True if updated successfully, False otherwise
        """
        try:
            config = self.get_config()
            
            # Apply updates recursively
            config_dict = self._config_to_dict(config)
            self._apply_updates(config_dict, updates)
            
            # Convert back to config object
            updated_config = self._dict_to_config(config_dict)
            
            # Validate updated configuration
            validation_errors = self.validate_config(updated_config)
            if validation_errors:
                logger.error(f"Configuration update validation failed: {validation_errors}")
                return False
                
            return self.save_config(updated_config)
            
        except Exception as e:
            logger.error(f"Failed to update configuration: {e}")
            return False
    
    def reset_to_defaults(self) -> bool:
        """Reset configuration to default values"""
        try:
            default_config = WAN22Config()
            return self.save_config(default_config)
        except Exception as e:
            logger.error(f"Failed to reset configuration to defaults: {e}")
            return False
    
    def validate_config(self, config: WAN22Config) -> List[str]:
        """Validate configuration for consistency and correctness
        
        Args:
            config: Configuration to validate
            
        Returns:
            List of validation error messages
        """
        errors = []
        
        try:
            # Validate optimization settings
            if config.optimization.max_chunk_size <= 0:
                errors.append("max_chunk_size must be positive")
                
            if config.optimization.vram_threshold_mb < 1024:
                errors.append("vram_threshold_mb should be at least 1024 MB")
                
            # Validate pipeline settings
            if config.pipeline.pipeline_timeout_seconds <= 0:
                errors.append("pipeline_timeout_seconds must be positive")
                
            if config.pipeline.max_retry_attempts < 0:
                errors.append("max_retry_attempts cannot be negative")
                
            # Validate security settings
            if config.security.sandbox_timeout_seconds <= 0:
                errors.append("sandbox_timeout_seconds must be positive")
                
            # Validate user preferences
            if config.user_preferences.default_fps <= 0:
                errors.append("default_fps must be positive")
                
            if config.user_preferences.max_concurrent_generations <= 0:
                errors.append("max_concurrent_generations must be positive")
                
            # Validate paths exist if specified
            if config.compatibility.diagnostic_output_dir:
                try:
                    Path(config.compatibility.diagnostic_output_dir).mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    errors.append(f"Cannot create diagnostic output directory: {e}")
                    
        except Exception as e:
            errors.append(f"Configuration validation error: {e}")
            
        return errors
    
    def get_optimization_config(self) -> OptimizationConfig:
        """Get optimization configuration"""
        return self.get_config().optimization
    
    def get_pipeline_config(self) -> PipelineConfig:
        """Get pipeline configuration"""
        return self.get_config().pipeline
    
    def get_security_config(self) -> SecurityConfig:
        """Get security configuration"""
        return self.get_config().security
    
    def get_compatibility_config(self) -> CompatibilityConfig:
        """Get compatibility configuration"""
        return self.get_config().compatibility
    
    def get_user_preferences(self) -> UserPreferences:
        """Get user preferences"""
        return self.get_config().user_preferences
    
    def export_config(self, export_path: str, include_sensitive: bool = False) -> bool:
        """Export configuration to a file
        
        Args:
            export_path: Path to export configuration
            include_sensitive: Whether to include sensitive settings
            
        Returns:
            True if exported successfully, False otherwise
        """
        try:
            config = self.get_config()
            config_dict = self._config_to_dict(config)
            
            if not include_sensitive:
                # Remove sensitive information
                if 'security' in config_dict:
                    config_dict['security'].pop('trusted_sources', None)
                if 'pipeline' in config_dict:
                    config_dict['pipeline'].pop('custom_pipeline_paths', None)
                    
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Configuration exported to {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export configuration: {e}")
            return False
    
    def import_config(self, import_path: str, merge: bool = True) -> bool:
        """Import configuration from a file
        
        Args:
            import_path: Path to import configuration from
            merge: Whether to merge with existing config or replace
            
        Returns:
            True if imported successfully, False otherwise
        """
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                imported_data = json.load(f)
                
            if merge:
                current_config = self._config_to_dict(self.get_config())
                self._apply_updates(current_config, imported_data)
                imported_data = current_config
                
            imported_config = self._dict_to_config(imported_data)
            
            # Validate imported configuration
            validation_errors = self.validate_config(imported_config)
            if validation_errors:
                logger.error(f"Imported configuration validation failed: {validation_errors}")
                return False
                
            return self.save_config(imported_config)
            
        except Exception as e:
            logger.error(f"Failed to import configuration: {e}")
            return False
    
    def _config_to_dict(self, config: WAN22Config) -> Dict[str, Any]:
        """Convert WAN22Config object to dictionary with enum handling"""
        config_dict = asdict(config)
        
        # Convert enums to their string values
        if 'optimization' in config_dict and 'strategy' in config_dict['optimization']:
            if hasattr(config_dict['optimization']['strategy'], 'value'):
                config_dict['optimization']['strategy'] = config_dict['optimization']['strategy'].value
            
        if 'pipeline' in config_dict and 'selection_mode' in config_dict['pipeline']:
            if hasattr(config_dict['pipeline']['selection_mode'], 'value'):
                config_dict['pipeline']['selection_mode'] = config_dict['pipeline']['selection_mode'].value
            
        if 'security' in config_dict and 'security_level' in config_dict['security']:
            if hasattr(config_dict['security']['security_level'], 'value'):
                config_dict['security']['security_level'] = config_dict['security']['security_level'].value
        
        return config_dict
    
    def _dict_to_config(self, data: Dict[str, Any]) -> WAN22Config:
        """Convert dictionary to WAN22Config object"""
        # Handle enum conversions
        if 'optimization' in data and 'strategy' in data['optimization']:
            data['optimization']['strategy'] = OptimizationStrategy(data['optimization']['strategy'])
            
        if 'pipeline' in data and 'selection_mode' in data['pipeline']:
            data['pipeline']['selection_mode'] = PipelineSelectionMode(data['pipeline']['selection_mode'])
            
        if 'security' in data and 'security_level' in data['security']:
            data['security']['security_level'] = SecurityLevel(data['security']['security_level'])
        
        # Create nested objects, filtering out unknown fields
        config = WAN22Config()
        
        if 'optimization' in data:
            # Filter out unknown fields for OptimizationConfig
            opt_data = {k: v for k, v in data['optimization'].items() 
                       if k in OptimizationConfig.__dataclass_fields__}
            config.optimization = OptimizationConfig(**opt_data)
            
        if 'pipeline' in data:
            # Filter out unknown fields for PipelineConfig
            pipe_data = {k: v for k, v in data['pipeline'].items() 
                        if k in PipelineConfig.__dataclass_fields__}
            config.pipeline = PipelineConfig(**pipe_data)
            
        if 'security' in data:
            # Filter out unknown fields for SecurityConfig
            sec_data = {k: v for k, v in data['security'].items() 
                       if k in SecurityConfig.__dataclass_fields__}
            config.security = SecurityConfig(**sec_data)
            
        if 'compatibility' in data:
            # Filter out unknown fields for CompatibilityConfig
            comp_data = {k: v for k, v in data['compatibility'].items() 
                        if k in CompatibilityConfig.__dataclass_fields__}
            config.compatibility = CompatibilityConfig(**comp_data)
            
        if 'user_preferences' in data:
            # Filter out unknown fields for UserPreferences
            pref_data = {k: v for k, v in data['user_preferences'].items() 
                        if k in UserPreferences.__dataclass_fields__}
            config.user_preferences = UserPreferences(**pref_data)
            
        # Set top-level attributes
        for key in ['version', 'created_at', 'updated_at', 'experimental_features', 'custom_settings']:
            if key in data:
                setattr(config, key, data[key])
                
        return config
    
    def _apply_updates(self, target: Dict[str, Any], updates: Dict[str, Any]) -> None:
        """Recursively apply updates to target dictionary"""
        for key, value in updates.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._apply_updates(target[key], value)
            else:
                target[key] = value
    
    def _migrate_config(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate configuration from older versions"""
        current_version = data.get('version', '0.0.0')
        
        if current_version < '1.0.0':
            # Add any migration logic for future versions
            data['version'] = self.CONFIG_VERSION
            
        return data


# Global configuration manager instance
_config_manager: Optional[ConfigurationManager] = None


def get_config_manager(config_dir: Optional[str] = None) -> ConfigurationManager:
    """Get global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigurationManager(config_dir)
    return _config_manager


def get_config() -> WAN22Config:
    """Get current WAN22 configuration"""
    return get_config_manager().get_config()


def save_config(config: WAN22Config) -> bool:
    """Save WAN22 configuration"""
    return get_config_manager().save_config(config)


def update_config(updates: Dict[str, Any]) -> bool:
    """Update WAN22 configuration"""
    return get_config_manager().update_config(updates)