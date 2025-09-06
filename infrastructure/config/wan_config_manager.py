"""
Wan Model Compatibility Configuration Manager

This module provides comprehensive configuration management for the Wan model
compatibility system, including user preferences, optimization strategies,
and advanced pipeline selection options.

Requirements addressed:
- 5.4: Memory-constrained processing configuration
- 5.5: Clear guidance configuration options
- 6.4: Security policy and local installation preferences
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
    AGGRESSIVE = "aggressive"
    CONSERVATIVE = "conservative"
    CUSTOM = "custom"


class SecurityMode(Enum):
    """Security modes for remote code handling"""
    STRICT = "strict"
    MODERATE = "moderate"
    PERMISSIVE = "permissive"


class PipelineSelectionMode(Enum):
    """Pipeline selection modes"""
    AUTO = "auto"
    MANUAL = "manual"
    FALLBACK_PREFERRED = "fallback_preferred"


@dataclass
class MemoryOptimizationConfig:
    """Configuration for memory optimization strategies"""
    enable_cpu_offload: bool = True
    use_mixed_precision: bool = True
    enable_chunked_processing: bool = True
    max_chunk_size: int = 8
    sequential_cpu_offload: bool = False
    vram_threshold_mb: int = 8192
    emergency_fallback: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryOptimizationConfig':
        return cls(**data)


@dataclass
class SecurityConfig:
    """Configuration for security and remote code handling"""
    mode: SecurityMode = SecurityMode.MODERATE
    trust_remote_code: bool = False
    allow_local_code_only: bool = False
    trusted_sources: List[str] = field(default_factory=lambda: [
        "huggingface.co",
        "Wan-AI"
    ])
    sandbox_untrusted_code: bool = True
    code_validation_timeout: int = 30
    require_code_signatures: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['mode'] = self.mode.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SecurityConfig':
        config_data = data.copy()
        if 'mode' in config_data:
            config_data['mode'] = SecurityMode(config_data['mode'])
        return cls(**config_data)


@dataclass
class PipelineConfig:
    """Configuration for pipeline selection and management"""
    selection_mode: PipelineSelectionMode = PipelineSelectionMode.AUTO
    preferred_pipeline_class: Optional[str] = None
    fallback_enabled: bool = True
    custom_pipeline_paths: List[str] = field(default_factory=list)
    pipeline_timeout: int = 300
    retry_attempts: int = 3
    enable_pipeline_caching: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['selection_mode'] = self.selection_mode.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PipelineConfig':
        config_data = data.copy()
        if 'selection_mode' in config_data:
            config_data['selection_mode'] = PipelineSelectionMode(config_data['selection_mode'])
        return cls(**config_data)


@dataclass
class DiagnosticConfig:
    """Configuration for diagnostic and error reporting"""
    enable_detailed_logging: bool = True
    save_diagnostic_reports: bool = True
    diagnostic_output_dir: str = "diagnostics"
    include_system_info: bool = True
    enable_performance_metrics: bool = True
    auto_generate_reports: bool = True
    report_retention_days: int = 30
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DiagnosticConfig':
        return cls(**data)


@dataclass
class UserPreferences:
    """User preferences for the compatibility system"""
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.AUTO
    show_advanced_options: bool = False
    enable_experimental_features: bool = False
    preferred_video_format: str = "mp4"
    default_resolution: str = "512x512"
    auto_apply_optimizations: bool = True
    show_progress_details: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['optimization_strategy'] = self.optimization_strategy.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserPreferences':
        config_data = data.copy()
        if 'optimization_strategy' in config_data:
            config_data['optimization_strategy'] = OptimizationStrategy(config_data['optimization_strategy'])
        return cls(**config_data)


@dataclass
class WanCompatibilityConfig:
    """Main configuration class for Wan model compatibility system"""
    version: str = "1.0.0"
    memory_optimization: MemoryOptimizationConfig = field(default_factory=MemoryOptimizationConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    diagnostics: DiagnosticConfig = field(default_factory=DiagnosticConfig)
    user_preferences: UserPreferences = field(default_factory=UserPreferences)
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'version': self.version,
            'memory_optimization': self.memory_optimization.to_dict(),
            'security': self.security.to_dict(),
            'pipeline': self.pipeline.to_dict(),
            'diagnostics': self.diagnostics.to_dict(),
            'user_preferences': self.user_preferences.to_dict(),
            'last_updated': self.last_updated
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WanCompatibilityConfig':
        return cls(
            version=data.get('version', '1.0.0'),
            memory_optimization=MemoryOptimizationConfig.from_dict(
                data.get('memory_optimization', {})
            ),
            security=SecurityConfig.from_dict(
                data.get('security', {})
            ),
            pipeline=PipelineConfig.from_dict(
                data.get('pipeline', {})
            ),
            diagnostics=DiagnosticConfig.from_dict(
                data.get('diagnostics', {})
            ),
            user_preferences=UserPreferences.from_dict(
                data.get('user_preferences', {})
            ),
            last_updated=data.get('last_updated', datetime.now().isoformat())
        )


class ConfigurationManager:
    """Manages configuration loading, saving, validation, and migration"""
    
    DEFAULT_CONFIG_PATH = "wan_compatibility_config.json"
    CONFIG_VERSION = "1.0.0"
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path or self.DEFAULT_CONFIG_PATH)
        self.config: Optional[WanCompatibilityConfig] = None
        self._ensure_config_directory()
    
    def _ensure_config_directory(self):
        """Ensure configuration directory exists"""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
    
    def load_config(self) -> WanCompatibilityConfig:
        """Load configuration from file or create default"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Validate and migrate if necessary
                data = self._migrate_config(data)
                self.config = WanCompatibilityConfig.from_dict(data)
                logger.info(f"Loaded configuration from {self.config_path}")
                
            except Exception as e:
                logger.warning(f"Failed to load config from {self.config_path}: {e}")
                logger.info("Creating default configuration")
                self.config = WanCompatibilityConfig()
                self.save_config()
        else:
            logger.info("No configuration file found, creating default")
            self.config = WanCompatibilityConfig()
            self.save_config()
        
        return self.config
    
    def save_config(self) -> bool:
        """Save current configuration to file"""
        if not self.config:
            logger.error("No configuration to save")
            return False
        
        try:
            # Update timestamp
            self.config.last_updated = datetime.now().isoformat()
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config.to_dict(), f, indent=2, ensure_ascii=False)
            
            logger.info(f"Configuration saved to {self.config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
    
    def _migrate_config(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate configuration to current version"""
        current_version = data.get('version', '0.0.0')
        
        if current_version == self.CONFIG_VERSION:
            return data
        
        logger.info(f"Migrating configuration from {current_version} to {self.CONFIG_VERSION}")
        
        # Add migration logic here as needed
        migrated_data = data.copy()
        migrated_data['version'] = self.CONFIG_VERSION
        
        return migrated_data
    
    def validate_config(self, config: Optional[WanCompatibilityConfig] = None) -> List[str]:
        """Validate configuration and return list of issues"""
        config = config or self.config
        if not config:
            return ["No configuration loaded"]
        
        issues = []
        
        # Validate memory optimization settings
        mem_config = config.memory_optimization
        if mem_config.max_chunk_size <= 0:
            issues.append("max_chunk_size must be positive")
        
        if mem_config.vram_threshold_mb < 1024:
            issues.append("vram_threshold_mb should be at least 1024 MB")
        
        # Validate security settings
        sec_config = config.security
        if sec_config.code_validation_timeout <= 0:
            issues.append("code_validation_timeout must be positive")
        
        # Validate pipeline settings
        pipe_config = config.pipeline
        if pipe_config.pipeline_timeout <= 0:
            issues.append("pipeline_timeout must be positive")
        
        if pipe_config.retry_attempts < 0:
            issues.append("retry_attempts cannot be negative")
        
        # Validate diagnostic settings
        diag_config = config.diagnostics
        if diag_config.report_retention_days <= 0:
            issues.append("report_retention_days must be positive")
        
        return issues
    
    def get_optimization_config_for_vram(self, available_vram_mb: int) -> MemoryOptimizationConfig:
        """Get optimized memory configuration based on available VRAM"""
        if not self.config:
            self.load_config()
        
        base_config = self.config.memory_optimization
        
        # Create optimized config based on available VRAM
        if available_vram_mb < 6144:  # Less than 6GB
            return MemoryOptimizationConfig(
                enable_cpu_offload=True,
                use_mixed_precision=True,
                enable_chunked_processing=True,
                max_chunk_size=4,
                sequential_cpu_offload=True,
                vram_threshold_mb=available_vram_mb,
                emergency_fallback=True
            )
        elif available_vram_mb < 12288:  # Less than 12GB
            return MemoryOptimizationConfig(
                enable_cpu_offload=True,
                use_mixed_precision=True,
                enable_chunked_processing=True,
                max_chunk_size=8,
                sequential_cpu_offload=False,
                vram_threshold_mb=available_vram_mb,
                emergency_fallback=True
            )
        else:  # 12GB or more
            return base_config
    
    def get_security_config_for_source(self, model_source: str) -> SecurityConfig:
        """Get security configuration optimized for specific model source"""
        if not self.config:
            self.load_config()
        
        base_config = self.config.security
        
        # Check if source is trusted
        is_trusted = any(trusted in model_source.lower() 
                        for trusted in base_config.trusted_sources)
        
        if is_trusted:
            # More permissive for trusted sources
            return SecurityConfig(
                mode=SecurityMode.MODERATE,
                trust_remote_code=True,
                allow_local_code_only=False,
                trusted_sources=base_config.trusted_sources,
                sandbox_untrusted_code=False,
                code_validation_timeout=base_config.code_validation_timeout,
                require_code_signatures=False
            )
        else:
            # More restrictive for untrusted sources
            return SecurityConfig(
                mode=SecurityMode.STRICT,
                trust_remote_code=False,
                allow_local_code_only=True,
                trusted_sources=base_config.trusted_sources,
                sandbox_untrusted_code=True,
                code_validation_timeout=base_config.code_validation_timeout,
                require_code_signatures=base_config.require_code_signatures
            )
    
    def update_user_preferences(self, **kwargs) -> bool:
        """Update user preferences"""
        if not self.config:
            self.load_config()
        
        try:
            for key, value in kwargs.items():
                if hasattr(self.config.user_preferences, key):
                    setattr(self.config.user_preferences, key, value)
                else:
                    logger.warning(f"Unknown preference key: {key}")
            
            return self.save_config()
            
        except Exception as e:
            logger.error(f"Failed to update user preferences: {e}")
            return False
    
    def reset_to_defaults(self) -> bool:
        """Reset configuration to defaults"""
        try:
            self.config = WanCompatibilityConfig()
            return self.save_config()
        except Exception as e:
            logger.error(f"Failed to reset configuration: {e}")
            return False
    
    def export_config(self, export_path: str) -> bool:
        """Export configuration to specified path"""
        if not self.config:
            return False
        
        try:
            export_path = Path(export_path)
            export_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(self.config.to_dict(), f, indent=2, ensure_ascii=False)
            
            logger.info(f"Configuration exported to {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export configuration: {e}")
            return False
    
    def import_config(self, import_path: str) -> bool:
        """Import configuration from specified path"""
        try:
            import_path = Path(import_path)
            if not import_path.exists():
                logger.error(f"Import file does not exist: {import_path}")
                return False
            
            with open(import_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate imported data
            imported_config = WanCompatibilityConfig.from_dict(data)
            validation_issues = self.validate_config(imported_config)
            
            if validation_issues:
                logger.error(f"Imported configuration has issues: {validation_issues}")
                return False
            
            self.config = imported_config
            return self.save_config()
            
        except Exception as e:
            logger.error(f"Failed to import configuration: {e}")
            return False


# Global configuration manager instance
_config_manager: Optional[ConfigurationManager] = None


def get_config_manager(config_path: Optional[str] = None) -> ConfigurationManager:
    """Get global configuration manager instance"""
    global _config_manager
    if _config_manager is None or config_path is not None:
        _config_manager = ConfigurationManager(config_path)
    return _config_manager


def get_config() -> WanCompatibilityConfig:
    """Get current configuration"""
    return get_config_manager().load_config()


def save_config(config: WanCompatibilityConfig) -> bool:
    """Save configuration"""
    manager = get_config_manager()
    manager.config = config
    return manager.save_config()


# Configuration presets for common scenarios
PRESETS = {
    "low_vram": WanCompatibilityConfig(
        memory_optimization=MemoryOptimizationConfig(
            enable_cpu_offload=True,
            use_mixed_precision=True,
            enable_chunked_processing=True,
            max_chunk_size=4,
            sequential_cpu_offload=True,
            vram_threshold_mb=4096,
            emergency_fallback=True
        ),
        user_preferences=UserPreferences(
            optimization_strategy=OptimizationStrategy.AGGRESSIVE,
            auto_apply_optimizations=True
        )
    ),
    
    "high_security": WanCompatibilityConfig(
        security=SecurityConfig(
            mode=SecurityMode.STRICT,
            trust_remote_code=False,
            allow_local_code_only=True,
            sandbox_untrusted_code=True,
            require_code_signatures=True
        ),
        pipeline=PipelineConfig(
            selection_mode=PipelineSelectionMode.MANUAL,
            fallback_enabled=False
        )
    ),
    
    "performance": WanCompatibilityConfig(
        memory_optimization=MemoryOptimizationConfig(
            enable_cpu_offload=False,
            use_mixed_precision=True,
            enable_chunked_processing=False,
            sequential_cpu_offload=False,
            emergency_fallback=False
        ),
        user_preferences=UserPreferences(
            optimization_strategy=OptimizationStrategy.CONSERVATIVE,
            enable_experimental_features=True
        )
    )
}


def apply_preset(preset_name: str) -> bool:
    """Apply a configuration preset"""
    if preset_name not in PRESETS:
        logger.error(f"Unknown preset: {preset_name}")
        return False
    
    try:
        config = PRESETS[preset_name]
        return save_config(config)
    except Exception as e:
        logger.error(f"Failed to apply preset {preset_name}: {e}")
        return False