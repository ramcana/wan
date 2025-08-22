"""
WAN22 Configuration Validation System

This module provides comprehensive validation for WAN22 configuration files,
including schema validation, constraint checking, and configuration recommendations.
"""

import re
from typing import Dict, Any, List, Optional, Union, Callable
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

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


class ValidationSeverity(Enum):
    """Validation message severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationMessage:
    """Validation message with context"""
    severity: ValidationSeverity
    code: str
    message: str
    field_path: str
    current_value: Any = None
    suggested_value: Any = None
    help_text: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of configuration validation"""
    is_valid: bool
    messages: List[ValidationMessage]
    score: float  # 0.0 to 1.0, higher is better
    
    def get_messages_by_severity(self, severity: ValidationSeverity) -> List[ValidationMessage]:
        """Get messages by severity level"""
        return [msg for msg in self.messages if msg.severity == severity]
    
    def has_errors(self) -> bool:
        """Check if validation has errors"""
        return any(msg.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] 
                  for msg in self.messages)
    
    def has_warnings(self) -> bool:
        """Check if validation has warnings"""
        return any(msg.severity == ValidationSeverity.WARNING for msg in self.messages)


class ConfigurationValidator:
    """Comprehensive configuration validator"""
    
    def __init__(self):
        """Initialize validator with validation rules"""
        self.validators = {
            "optimization": self._validate_optimization,
            "pipeline": self._validate_pipeline,
            "security": self._validate_security,
            "compatibility": self._validate_compatibility,
            "user_preferences": self._validate_user_preferences,
            "experimental_features": self._validate_experimental_features,
            "custom_settings": self._validate_custom_settings
        }
    
    def validate_config(self, config: WAN22Config) -> ValidationResult:
        """Validate complete configuration
        
        Args:
            config: Configuration to validate
            
        Returns:
            ValidationResult with all validation messages
        """
        messages = []
        
        # Validate top-level structure
        messages.extend(self._validate_structure(config))
        
        # Validate each section
        for section_name, validator in self.validators.items():
            if hasattr(config, section_name):
                section_config = getattr(config, section_name)
                section_messages = validator(section_config, section_name)
                messages.extend(section_messages)
        
        # Cross-section validation
        messages.extend(self._validate_cross_section(config))
        
        # Calculate validation score
        score = self._calculate_score(messages)
        
        # Determine if configuration is valid
        is_valid = not any(msg.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] 
                          for msg in messages)
        
        return ValidationResult(is_valid=is_valid, messages=messages, score=score)
    
    def _validate_structure(self, config: WAN22Config) -> List[ValidationMessage]:
        """Validate top-level configuration structure"""
        messages = []
        
        # Check version
        if not hasattr(config, 'version') or not config.version:
            messages.append(ValidationMessage(
                severity=ValidationSeverity.ERROR,
                code="MISSING_VERSION",
                message="Configuration version is missing",
                field_path="version",
                help_text="Version is required for configuration migration and compatibility"
            ))
        elif config.version != "1.0.0":
            messages.append(ValidationMessage(
                severity=ValidationSeverity.WARNING,
                code="OLD_VERSION",
                message=f"Configuration version {config.version} is outdated",
                field_path="version",
                current_value=config.version,
                suggested_value="1.0.0",
                help_text="Consider migrating to the latest version"
            ))
        
        # Check timestamps
        if not hasattr(config, 'created_at') or not config.created_at:
            messages.append(ValidationMessage(
                severity=ValidationSeverity.INFO,
                code="MISSING_CREATED_AT",
                message="Creation timestamp is missing",
                field_path="created_at",
                help_text="Timestamp helps track configuration history"
            ))
        
        # Check required sections
        required_sections = ["optimization", "pipeline", "security", "compatibility", "user_preferences"]
        for section in required_sections:
            if not hasattr(config, section):
                messages.append(ValidationMessage(
                    severity=ValidationSeverity.ERROR,
                    code="MISSING_SECTION",
                    message=f"Required section '{section}' is missing",
                    field_path=section,
                    help_text=f"The {section} section is required for proper functionality"
                ))
        
        return messages
    
    def _validate_optimization(self, config: OptimizationConfig, path: str) -> List[ValidationMessage]:
        """Validate optimization configuration"""
        messages = []
        
        # Validate strategy
        if not isinstance(config.strategy, OptimizationStrategy):
            messages.append(ValidationMessage(
                severity=ValidationSeverity.ERROR,
                code="INVALID_STRATEGY",
                message="Invalid optimization strategy",
                field_path=f"{path}.strategy",
                current_value=config.strategy,
                help_text="Must be one of: auto, performance, memory, balanced, custom"
            ))
        
        # Validate VRAM threshold
        if config.vram_threshold_mb < 1024:
            messages.append(ValidationMessage(
                severity=ValidationSeverity.WARNING,
                code="LOW_VRAM_THRESHOLD",
                message="VRAM threshold is very low",
                field_path=f"{path}.vram_threshold_mb",
                current_value=config.vram_threshold_mb,
                suggested_value=4096,
                help_text="Consider setting at least 4096 MB for stable operation"
            ))
        elif config.vram_threshold_mb > 32768:
            messages.append(ValidationMessage(
                severity=ValidationSeverity.INFO,
                code="HIGH_VRAM_THRESHOLD",
                message="VRAM threshold is very high",
                field_path=f"{path}.vram_threshold_mb",
                current_value=config.vram_threshold_mb,
                help_text="Ensure your system has sufficient VRAM"
            ))
        
        # Validate chunk size
        if config.max_chunk_size <= 0:
            messages.append(ValidationMessage(
                severity=ValidationSeverity.ERROR,
                code="INVALID_CHUNK_SIZE",
                message="Chunk size must be positive",
                field_path=f"{path}.max_chunk_size",
                current_value=config.max_chunk_size,
                suggested_value=8,
                help_text="Chunk size determines memory usage during processing"
            ))
        elif config.max_chunk_size > 32:
            messages.append(ValidationMessage(
                severity=ValidationSeverity.WARNING,
                code="LARGE_CHUNK_SIZE",
                message="Large chunk size may cause memory issues",
                field_path=f"{path}.max_chunk_size",
                current_value=config.max_chunk_size,
                suggested_value=8,
                help_text="Consider reducing chunk size if experiencing memory issues"
            ))
        
        # Validate VAE tile size
        if config.enable_vae_tiling and config.vae_tile_size < 64:
            messages.append(ValidationMessage(
                severity=ValidationSeverity.WARNING,
                code="SMALL_VAE_TILE",
                message="VAE tile size is very small",
                field_path=f"{path}.vae_tile_size",
                current_value=config.vae_tile_size,
                suggested_value=256,
                help_text="Small tiles may impact generation quality"
            ))
        
        # Validate CPU offload strategy
        valid_strategies = ["sequential", "model", "full"]
        if config.cpu_offload_strategy not in valid_strategies:
            messages.append(ValidationMessage(
                severity=ValidationSeverity.ERROR,
                code="INVALID_OFFLOAD_STRATEGY",
                message="Invalid CPU offload strategy",
                field_path=f"{path}.cpu_offload_strategy",
                current_value=config.cpu_offload_strategy,
                suggested_value="sequential",
                help_text=f"Must be one of: {', '.join(valid_strategies)}"
            ))
        
        # Strategy-specific validation
        if config.strategy == OptimizationStrategy.PERFORMANCE:
            if config.enable_cpu_offload:
                messages.append(ValidationMessage(
                    severity=ValidationSeverity.WARNING,
                    code="PERFORMANCE_WITH_OFFLOAD",
                    message="CPU offload enabled with performance strategy",
                    field_path=f"{path}.enable_cpu_offload",
                    current_value=True,
                    suggested_value=False,
                    help_text="CPU offload may reduce performance"
                ))
            
            if config.enable_chunked_processing:
                messages.append(ValidationMessage(
                    severity=ValidationSeverity.WARNING,
                    code="PERFORMANCE_WITH_CHUNKING",
                    message="Chunked processing enabled with performance strategy",
                    field_path=f"{path}.enable_chunked_processing",
                    current_value=True,
                    suggested_value=False,
                    help_text="Chunked processing may reduce performance"
                ))
        
        elif config.strategy == OptimizationStrategy.MEMORY:
            if not config.enable_cpu_offload and not config.enable_chunked_processing:
                messages.append(ValidationMessage(
                    severity=ValidationSeverity.INFO,
                    code="MEMORY_NO_OPTIMIZATIONS",
                    message="Memory strategy without memory optimizations",
                    field_path=f"{path}.strategy",
                    help_text="Consider enabling CPU offload or chunked processing"
                ))
        
        return messages
    
    def _validate_pipeline(self, config: PipelineConfig, path: str) -> List[ValidationMessage]:
        """Validate pipeline configuration"""
        messages = []
        
        # Validate selection mode
        if not isinstance(config.selection_mode, PipelineSelectionMode):
            messages.append(ValidationMessage(
                severity=ValidationSeverity.ERROR,
                code="INVALID_SELECTION_MODE",
                message="Invalid pipeline selection mode",
                field_path=f"{path}.selection_mode",
                current_value=config.selection_mode,
                help_text="Must be one of: auto, manual, fallback_enabled, strict"
            ))
        
        # Validate timeout
        if config.pipeline_timeout_seconds <= 0:
            messages.append(ValidationMessage(
                severity=ValidationSeverity.ERROR,
                code="INVALID_TIMEOUT",
                message="Pipeline timeout must be positive",
                field_path=f"{path}.pipeline_timeout_seconds",
                current_value=config.pipeline_timeout_seconds,
                suggested_value=300,
                help_text="Timeout prevents hanging operations"
            ))
        elif config.pipeline_timeout_seconds < 60:
            messages.append(ValidationMessage(
                severity=ValidationSeverity.WARNING,
                code="SHORT_TIMEOUT",
                message="Pipeline timeout is very short",
                field_path=f"{path}.pipeline_timeout_seconds",
                current_value=config.pipeline_timeout_seconds,
                suggested_value=300,
                help_text="Short timeout may cause premature failures"
            ))
        
        # Validate retry attempts
        if config.max_retry_attempts < 0:
            messages.append(ValidationMessage(
                severity=ValidationSeverity.ERROR,
                code="INVALID_RETRY_ATTEMPTS",
                message="Retry attempts cannot be negative",
                field_path=f"{path}.max_retry_attempts",
                current_value=config.max_retry_attempts,
                suggested_value=3,
                help_text="Retries help handle transient failures"
            ))
        elif config.max_retry_attempts > 10:
            messages.append(ValidationMessage(
                severity=ValidationSeverity.WARNING,
                code="EXCESSIVE_RETRIES",
                message="Too many retry attempts configured",
                field_path=f"{path}.max_retry_attempts",
                current_value=config.max_retry_attempts,
                suggested_value=3,
                help_text="Excessive retries may cause long delays"
            ))
        
        # Validate fallback strategies
        valid_strategies = ["component_isolation", "alternative_model", "reduced_functionality"]
        for strategy in config.fallback_strategies:
            if strategy not in valid_strategies:
                messages.append(ValidationMessage(
                    severity=ValidationSeverity.WARNING,
                    code="UNKNOWN_FALLBACK_STRATEGY",
                    message=f"Unknown fallback strategy: {strategy}",
                    field_path=f"{path}.fallback_strategies",
                    help_text=f"Valid strategies: {', '.join(valid_strategies)}"
                ))
        
        # Mode-specific validation
        if config.selection_mode == PipelineSelectionMode.MANUAL:
            if not config.preferred_pipeline_class:
                messages.append(ValidationMessage(
                    severity=ValidationSeverity.WARNING,
                    code="MANUAL_NO_PREFERRED",
                    message="Manual mode without preferred pipeline class",
                    field_path=f"{path}.preferred_pipeline_class",
                    help_text="Specify preferred pipeline class for manual mode"
                ))
        
        if config.selection_mode == PipelineSelectionMode.STRICT:
            if config.enable_fallback:
                messages.append(ValidationMessage(
                    severity=ValidationSeverity.WARNING,
                    code="STRICT_WITH_FALLBACK",
                    message="Strict mode with fallback enabled",
                    field_path=f"{path}.enable_fallback",
                    current_value=True,
                    suggested_value=False,
                    help_text="Strict mode typically disables fallback"
                ))
        
        return messages
    
    def _validate_security(self, config: SecurityConfig, path: str) -> List[ValidationMessage]:
        """Validate security configuration"""
        messages = []
        
        # Validate security level
        if not isinstance(config.security_level, SecurityLevel):
            messages.append(ValidationMessage(
                severity=ValidationSeverity.ERROR,
                code="INVALID_SECURITY_LEVEL",
                message="Invalid security level",
                field_path=f"{path}.security_level",
                current_value=config.security_level,
                help_text="Must be one of: strict, moderate, permissive"
            ))
        
        # Validate sandbox timeout
        if config.sandbox_timeout_seconds <= 0:
            messages.append(ValidationMessage(
                severity=ValidationSeverity.ERROR,
                code="INVALID_SANDBOX_TIMEOUT",
                message="Sandbox timeout must be positive",
                field_path=f"{path}.sandbox_timeout_seconds",
                current_value=config.sandbox_timeout_seconds,
                suggested_value=60,
                help_text="Timeout prevents hanging sandbox operations"
            ))
        
        # Validate trusted sources
        for source in config.trusted_sources:
            if not self._is_valid_url_pattern(source):
                messages.append(ValidationMessage(
                    severity=ValidationSeverity.WARNING,
                    code="INVALID_TRUSTED_SOURCE",
                    message=f"Potentially invalid trusted source: {source}",
                    field_path=f"{path}.trusted_sources",
                    help_text="Trusted sources should be valid domain patterns"
                ))
        
        # Security level specific validation
        if config.security_level == SecurityLevel.STRICT:
            if config.trust_remote_code:
                messages.append(ValidationMessage(
                    severity=ValidationSeverity.WARNING,
                    code="STRICT_TRUST_REMOTE",
                    message="Strict security with remote code trust enabled",
                    field_path=f"{path}.trust_remote_code",
                    current_value=True,
                    suggested_value=False,
                    help_text="Strict security typically disables remote code"
                ))
            
            if not config.enable_sandboxing:
                messages.append(ValidationMessage(
                    severity=ValidationSeverity.INFO,
                    code="STRICT_NO_SANDBOX",
                    message="Strict security without sandboxing",
                    field_path=f"{path}.enable_sandboxing",
                    current_value=False,
                    suggested_value=True,
                    help_text="Consider enabling sandboxing for strict security"
                ))
        
        elif config.security_level == SecurityLevel.PERMISSIVE:
            if not config.trust_remote_code:
                messages.append(ValidationMessage(
                    severity=ValidationSeverity.INFO,
                    code="PERMISSIVE_NO_REMOTE",
                    message="Permissive security without remote code trust",
                    field_path=f"{path}.trust_remote_code",
                    help_text="Permissive mode typically allows remote code"
                ))
        
        return messages
    
    def _validate_compatibility(self, config: CompatibilityConfig, path: str) -> List[ValidationMessage]:
        """Validate compatibility configuration"""
        messages = []
        
        # Validate cache TTL
        if config.detection_cache_ttl_hours <= 0:
            messages.append(ValidationMessage(
                severity=ValidationSeverity.ERROR,
                code="INVALID_CACHE_TTL",
                message="Cache TTL must be positive",
                field_path=f"{path}.detection_cache_ttl_hours",
                current_value=config.detection_cache_ttl_hours,
                suggested_value=24,
                help_text="Cache TTL determines how long detection results are cached"
            ))
        elif config.detection_cache_ttl_hours > 168:  # 1 week
            messages.append(ValidationMessage(
                severity=ValidationSeverity.WARNING,
                code="LONG_CACHE_TTL",
                message="Cache TTL is very long",
                field_path=f"{path}.detection_cache_ttl_hours",
                current_value=config.detection_cache_ttl_hours,
                suggested_value=24,
                help_text="Long cache may miss model updates"
            ))
        
        # Validate diagnostic output directory
        if config.enable_diagnostic_collection:
            if not config.diagnostic_output_dir:
                messages.append(ValidationMessage(
                    severity=ValidationSeverity.WARNING,
                    code="MISSING_DIAGNOSTIC_DIR",
                    message="Diagnostic collection enabled without output directory",
                    field_path=f"{path}.diagnostic_output_dir",
                    suggested_value="diagnostics",
                    help_text="Specify directory for diagnostic output"
                ))
            else:
                # Check if directory can be created
                try:
                    Path(config.diagnostic_output_dir).mkdir(parents=True, exist_ok=True)
                except Exception:
                    messages.append(ValidationMessage(
                        severity=ValidationSeverity.ERROR,
                        code="INVALID_DIAGNOSTIC_DIR",
                        message="Cannot create diagnostic output directory",
                        field_path=f"{path}.diagnostic_output_dir",
                        current_value=config.diagnostic_output_dir,
                        help_text="Ensure directory path is valid and writable"
                    ))
        
        # Validation consistency checks
        if config.strict_validation and not config.enable_component_validation:
            messages.append(ValidationMessage(
                severity=ValidationSeverity.WARNING,
                code="STRICT_NO_COMPONENT_VALIDATION",
                message="Strict validation without component validation",
                field_path=f"{path}.enable_component_validation",
                current_value=False,
                suggested_value=True,
                help_text="Strict validation typically includes component validation"
            ))
        
        return messages
    
    def _validate_user_preferences(self, config: UserPreferences, path: str) -> List[ValidationMessage]:
        """Validate user preferences"""
        messages = []
        
        # Validate FPS
        if config.default_fps <= 0:
            messages.append(ValidationMessage(
                severity=ValidationSeverity.ERROR,
                code="INVALID_FPS",
                message="FPS must be positive",
                field_path=f"{path}.default_fps",
                current_value=config.default_fps,
                suggested_value=24.0,
                help_text="FPS determines video playback speed"
            ))
        elif config.default_fps > 120:
            messages.append(ValidationMessage(
                severity=ValidationSeverity.WARNING,
                code="HIGH_FPS",
                message="Very high FPS configured",
                field_path=f"{path}.default_fps",
                current_value=config.default_fps,
                help_text="High FPS increases file size and processing time"
            ))
        
        # Validate concurrent generations
        if config.max_concurrent_generations <= 0:
            messages.append(ValidationMessage(
                severity=ValidationSeverity.ERROR,
                code="INVALID_CONCURRENT_GENERATIONS",
                message="Concurrent generations must be positive",
                field_path=f"{path}.max_concurrent_generations",
                current_value=config.max_concurrent_generations,
                suggested_value=1,
                help_text="Controls how many generations run simultaneously"
            ))
        elif config.max_concurrent_generations > 4:
            messages.append(ValidationMessage(
                severity=ValidationSeverity.WARNING,
                code="HIGH_CONCURRENT_GENERATIONS",
                message="High concurrent generations may cause resource issues",
                field_path=f"{path}.max_concurrent_generations",
                current_value=config.max_concurrent_generations,
                suggested_value=2,
                help_text="Consider system resources when setting concurrent generations"
            ))
        
        # Validate output format
        valid_formats = ["mp4", "webm", "avi", "mov"]
        if config.default_output_format not in valid_formats:
            messages.append(ValidationMessage(
                severity=ValidationSeverity.WARNING,
                code="UNSUPPORTED_OUTPUT_FORMAT",
                message=f"Potentially unsupported output format: {config.default_output_format}",
                field_path=f"{path}.default_output_format",
                help_text=f"Supported formats: {', '.join(valid_formats)}"
            ))
        
        # Validate video codec
        valid_codecs = ["h264", "h265", "vp8", "vp9", "av1"]
        if config.preferred_video_codec not in valid_codecs:
            messages.append(ValidationMessage(
                severity=ValidationSeverity.WARNING,
                code="UNSUPPORTED_VIDEO_CODEC",
                message=f"Potentially unsupported video codec: {config.preferred_video_codec}",
                field_path=f"{path}.preferred_video_codec",
                help_text=f"Supported codecs: {', '.join(valid_codecs)}"
            ))
        
        return messages
    
    def _validate_experimental_features(self, config: Dict[str, Any], path: str) -> List[ValidationMessage]:
        """Validate experimental features"""
        messages = []
        
        # Known experimental features
        known_features = [
            "advanced_diagnostics",
            "performance_profiling", 
            "debug_mode",
            "detailed_logging",
            "experimental_optimizations"
        ]
        
        for feature, enabled in config.items():
            if feature not in known_features:
                messages.append(ValidationMessage(
                    severity=ValidationSeverity.INFO,
                    code="UNKNOWN_EXPERIMENTAL_FEATURE",
                    message=f"Unknown experimental feature: {feature}",
                    field_path=f"{path}.{feature}",
                    help_text="Unknown features may not have any effect"
                ))
            
            if not isinstance(enabled, bool):
                messages.append(ValidationMessage(
                    severity=ValidationSeverity.WARNING,
                    code="INVALID_FEATURE_VALUE",
                    message=f"Experimental feature value should be boolean: {feature}",
                    field_path=f"{path}.{feature}",
                    current_value=enabled,
                    suggested_value=bool(enabled),
                    help_text="Feature flags should be true or false"
                ))
        
        return messages
    
    def _validate_custom_settings(self, config: Dict[str, Any], path: str) -> List[ValidationMessage]:
        """Validate custom settings"""
        messages = []
        
        # Check for potentially problematic custom settings
        for key, value in config.items():
            if key.startswith("_"):
                messages.append(ValidationMessage(
                    severity=ValidationSeverity.WARNING,
                    code="PRIVATE_CUSTOM_SETTING",
                    message=f"Custom setting with private name: {key}",
                    field_path=f"{path}.{key}",
                    help_text="Settings starting with _ may conflict with internal settings"
                ))
            
            # Check for very large values that might cause issues
            if isinstance(value, (int, float)) and abs(value) > 1e6:
                messages.append(ValidationMessage(
                    severity=ValidationSeverity.INFO,
                    code="LARGE_CUSTOM_VALUE",
                    message=f"Very large custom setting value: {key}",
                    field_path=f"{path}.{key}",
                    current_value=value,
                    help_text="Large values may cause unexpected behavior"
                ))
        
        return messages
    
    def _validate_cross_section(self, config: WAN22Config) -> List[ValidationMessage]:
        """Validate cross-section consistency"""
        messages = []
        
        # Security vs Pipeline consistency
        if (config.security.security_level == SecurityLevel.STRICT and 
            config.pipeline.selection_mode != PipelineSelectionMode.MANUAL):
            messages.append(ValidationMessage(
                severity=ValidationSeverity.INFO,
                code="STRICT_SECURITY_AUTO_PIPELINE",
                message="Strict security with automatic pipeline selection",
                field_path="pipeline.selection_mode",
                help_text="Consider manual pipeline selection for strict security"
            ))
        
        # Memory optimization consistency
        if (config.optimization.strategy == OptimizationStrategy.MEMORY and
            config.user_preferences.max_concurrent_generations > 1):
            messages.append(ValidationMessage(
                severity=ValidationSeverity.WARNING,
                code="MEMORY_STRATEGY_CONCURRENT",
                message="Memory optimization with multiple concurrent generations",
                field_path="user_preferences.max_concurrent_generations",
                current_value=config.user_preferences.max_concurrent_generations,
                suggested_value=1,
                help_text="Multiple generations may conflict with memory optimization"
            ))
        
        # Performance optimization consistency
        if (config.optimization.strategy == OptimizationStrategy.PERFORMANCE and
            config.compatibility.strict_validation):
            messages.append(ValidationMessage(
                severity=ValidationSeverity.INFO,
                code="PERFORMANCE_STRICT_VALIDATION",
                message="Performance optimization with strict validation",
                field_path="compatibility.strict_validation",
                help_text="Strict validation may impact performance"
            ))
        
        return messages
    
    def _calculate_score(self, messages: List[ValidationMessage]) -> float:
        """Calculate validation score based on messages"""
        if not messages:
            return 1.0
        
        # Weight by severity
        severity_weights = {
            ValidationSeverity.INFO: 0.0,
            ValidationSeverity.WARNING: 0.1,
            ValidationSeverity.ERROR: 0.5,
            ValidationSeverity.CRITICAL: 1.0
        }
        
        total_penalty = sum(severity_weights.get(msg.severity, 0.0) for msg in messages)
        max_possible_penalty = len(messages) * 1.0
        
        if max_possible_penalty == 0:
            return 1.0
        
        score = max(0.0, 1.0 - (total_penalty / max_possible_penalty))
        return score
    
    def _is_valid_url_pattern(self, pattern: str) -> bool:
        """Check if URL pattern is valid"""
        # Simple validation for domain patterns
        domain_pattern = re.compile(
            r'^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*$'
        )
        return bool(domain_pattern.match(pattern))


def validate_config(config: WAN22Config) -> ValidationResult:
    """Convenience function to validate configuration
    
    Args:
        config: Configuration to validate
        
    Returns:
        ValidationResult with validation messages
    """
    validator = ConfigurationValidator()
    return validator.validate_config(config)


def get_validation_summary(result: ValidationResult) -> str:
    """Get human-readable validation summary
    
    Args:
        result: ValidationResult to summarize
        
    Returns:
        Formatted validation summary
    """
    if result.is_valid and not result.messages:
        return f"✅ Configuration is valid with no issues\n📊 Validation Score: {result.score:.2f}/1.00"
    
    summary_lines = []
    summary_lines.append(f"📊 Validation Score: {result.score:.2f}/1.00")
    
    if result.is_valid:
        summary_lines.append("✅ Configuration is valid")
    else:
        summary_lines.append("❌ Configuration has errors")
    
    # Count messages by severity
    error_count = len(result.get_messages_by_severity(ValidationSeverity.ERROR))
    critical_count = len(result.get_messages_by_severity(ValidationSeverity.CRITICAL))
    warning_count = len(result.get_messages_by_severity(ValidationSeverity.WARNING))
    info_count = len(result.get_messages_by_severity(ValidationSeverity.INFO))
    
    if critical_count > 0:
        summary_lines.append(f"🔴 {critical_count} critical issue(s)")
    if error_count > 0:
        summary_lines.append(f"🟠 {error_count} error(s)")
    if warning_count > 0:
        summary_lines.append(f"🟡 {warning_count} warning(s)")
    if info_count > 0:
        summary_lines.append(f"ℹ️ {info_count} info message(s)")
    
    return "\n".join(summary_lines)