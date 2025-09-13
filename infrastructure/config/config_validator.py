"""
WAN22 Configuration Validator

This module provides comprehensive configuration validation and cleanup for the WAN22 system.
It handles schema validation, automatic cleanup of unexpected attributes, and configuration backup.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Set
from dataclasses import dataclass
from enum import Enum
import logging


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
    cleaned_attributes: List[str]
    backup_path: Optional[str] = None
    
    def has_errors(self) -> bool:
        """Check if validation has errors"""
        return any(msg.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] 
                  for msg in self.messages)
    
    def has_warnings(self) -> bool:
        """Check if validation has warnings"""
        return any(msg.severity == ValidationSeverity.WARNING for msg in self.messages)


@dataclass
class CleanupResult:
    """Result of configuration cleanup"""
    cleaned_attributes: List[str]
    backup_created: bool
    backup_path: Optional[str] = None


class ConfigValidator:
    """
    Configuration validator for WAN22 system.
    
    Provides schema validation, automatic cleanup of unexpected attributes,
    and configuration backup functionality.
    """
    
    def __init__(self, backup_dir: Optional[Path] = None):
        """
        Initialize the configuration validator.
        
        Args:
            backup_dir: Directory for configuration backups (default: ./config_backups)
        """
        self.logger = logging.getLogger(__name__)
        self.backup_dir = backup_dir or Path("config_backups")
        self.backup_dir.mkdir(exist_ok=True)
        
        # Define expected configuration schema
        self.expected_schema = self._define_expected_schema()
        
        # Define attributes that should be automatically removed
        self.cleanup_attributes = self._define_cleanup_attributes()
    
    def _define_expected_schema(self) -> Dict[str, Any]:
        """Define the expected configuration schema"""
        return {
            "system": {
                "type": "object",
                "required": ["default_quantization", "enable_offload"],
                "properties": {
                    "default_quantization": {"type": "string", "enum": ["fp16", "bf16", "int8", "none"]},
                    "enable_offload": {"type": "boolean"},
                    "vae_tile_size": {"type": "integer", "minimum": 128, "maximum": 1024},
                    "max_queue_size": {"type": "integer", "minimum": 1, "maximum": 100},
                    "stats_refresh_interval": {"type": "number", "minimum": 1, "maximum": 60}
                }
            },
            "directories": {
                "type": "object",
                "required": ["output_directory", "models_directory"],
                "properties": {
                    "output_directory": {"type": "string"},
                    "models_directory": {"type": "string"},
                    "loras_directory": {"type": "string"}
                }
            },
            "generation": {
                "type": "object",
                "required": ["default_resolution", "default_steps"],
                "properties": {
                    "default_resolution": {"type": "string", "pattern": r"^\d+x\d+$"},
                    "default_steps": {"type": "integer", "minimum": 1, "maximum": 200},
                    "default_duration": {"type": "number", "minimum": 1, "maximum": 30},
                    "default_fps": {"type": "integer", "minimum": 1, "maximum": 60},
                    "max_prompt_length": {"type": "integer", "minimum": 1, "maximum": 2000},
                    "supported_resolutions": {"type": "array", "items": {"type": "string"}}
                }
            },
            "models": {
                "type": "object",
                "properties": {
                    "t2v_model": {"type": "string"},
                    "i2v_model": {"type": "string"},
                    "ti2v_model": {"type": "string"}
                }
            },
            "optimization": {
                "type": "object",
                "properties": {
                    "default_quantization": {"type": "string", "enum": ["fp16", "bf16", "int8", "none"]},
                    "quantization_levels": {"type": "array", "items": {"type": "string"}},
                    "vae_tile_size_range": {"type": "array", "items": {"type": "integer"}},
                    "max_vram_usage_gb": {"type": "number", "minimum": 1, "maximum": 128},
                    "enable_offload": {"type": "boolean"},
                    "vae_tile_size": {"type": "integer", "minimum": 128, "maximum": 1024}
                }
            },
            "ui": {
                "type": "object",
                "properties": {
                    "max_file_size_mb": {"type": "number", "minimum": 1, "maximum": 1000},
                    "supported_image_formats": {"type": "array", "items": {"type": "string"}},
                    "gallery_thumbnail_size": {"type": "integer", "minimum": 64, "maximum": 512}
                }
            },
            "performance": {
                "type": "object",
                "properties": {
                    "target_720p_time_minutes": {"type": "number", "minimum": 1},
                    "target_1080p_time_minutes": {"type": "number", "minimum": 1},
                    "vram_warning_threshold": {"type": "number", "minimum": 0.1, "maximum": 1.0},
                    "cpu_warning_percent": {"type": "number", "minimum": 50, "maximum": 100},
                    "memory_warning_percent": {"type": "number", "minimum": 50, "maximum": 100},
                    "vram_warning_percent": {"type": "number", "minimum": 50, "maximum": 100},
                    "sample_interval_seconds": {"type": "number", "minimum": 1, "maximum": 300},
                    "max_history_samples": {"type": "integer", "minimum": 10, "maximum": 1000},
                    "cpu_monitoring_enabled": {"type": "boolean"},
                    "disk_io_monitoring_enabled": {"type": "boolean"},
                    "network_monitoring_enabled": {"type": "boolean"}
                }
            },
            "prompt_enhancement": {
                "type": "object",
                "properties": {
                    "max_prompt_length": {"type": "integer", "minimum": 1, "maximum": 2000},
                    "enable_basic_quality": {"type": "boolean"},
                    "enable_vace_detection": {"type": "boolean"},
                    "enable_cinematic_enhancement": {"type": "boolean"},
                    "enable_style_detection": {"type": "boolean"},
                    "max_quality_keywords": {"type": "integer", "minimum": 0, "maximum": 10},
                    "max_cinematic_keywords": {"type": "integer", "minimum": 0, "maximum": 10},
                    "max_style_keywords": {"type": "integer", "minimum": 0, "maximum": 10}
                }
            }
        }
    
    def _define_cleanup_attributes(self) -> Dict[str, Set[str]]:
        """Define attributes that should be automatically removed"""
        return {
            # Model configuration cleanup
            "model_config": {
                "clip_output",  # Unsupported in AutoencoderKLWan
                "force_upcast",  # Legacy attribute
                "use_linear_projection",  # Deprecated
                "cross_attention_dim",  # Model-specific, should not be in config
            },
            # VAE configuration cleanup
            "vae_config": {
                "clip_output",  # Main issue mentioned in requirements
                "force_upcast",
                "use_tiling",  # Should be handled by tile_size
            },
            # Text encoder cleanup
            "text_encoder_config": {
                "use_attention_mask",  # Model handles this internally
                "return_dict",  # Internal parameter
            },
            # Pipeline configuration cleanup
            "pipeline_config": {
                "requires_safety_checker",  # Legacy safety checker
                "safety_checker",  # Should be handled separately
                "feature_extractor",  # Deprecated
            },
            # General cleanup
            "general": {
                "torch_dtype",  # Should be handled by quantization system
                "variant",  # Model loading parameter, not config
                "use_safetensors",  # Loading parameter
                "local_files_only",  # Loading parameter
                "revision",  # Model version parameter
                "cache_dir",  # System parameter
            }
        }
    
    def validate_config_file(self, config_path: Union[str, Path]) -> ValidationResult:
        """
        Validate a configuration file against expected schema.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            ValidationResult with validation messages and cleanup info
        """
        config_path = Path(config_path)
        messages = []
        cleaned_attributes = []
        backup_path = None
        
        try:
            # Check if file exists
            if not config_path.exists():
                messages.append(ValidationMessage(
                    severity=ValidationSeverity.CRITICAL,
                    code="FILE_NOT_FOUND",
                    message=f"Configuration file not found: {config_path}",
                    field_path=str(config_path)
                ))
                return ValidationResult(
                    is_valid=False,
                    messages=messages,
                    cleaned_attributes=cleaned_attributes,
                    backup_path=backup_path
                )
            
            # Load configuration
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # Create backup before validation/cleanup
            backup_path = self.create_backup(config_path)
            
            # Validate schema
            schema_messages = self._validate_schema(config_data, self.expected_schema)
            messages.extend(schema_messages)
            
            # Clean up unexpected attributes
            cleanup_result = self._cleanup_config(config_data)
            cleaned_attributes = cleanup_result.cleaned_attributes
            
            if cleaned_attributes:
                messages.append(ValidationMessage(
                    severity=ValidationSeverity.INFO,
                    code="ATTRIBUTES_CLEANED",
                    message=f"Cleaned {len(cleaned_attributes)} unexpected attributes",
                    field_path="config",
                    current_value=cleaned_attributes,
                    help_text="Removed unsupported or deprecated attributes"
                ))
                
                # Save cleaned configuration
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(config_data, f, indent=2, ensure_ascii=False)
                
                self.logger.info(f"Cleaned configuration saved to {config_path}")
            
            # Determine if configuration is valid
            is_valid = not any(msg.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] 
                             for msg in messages)
            
            return ValidationResult(
                is_valid=is_valid,
                messages=messages,
                cleaned_attributes=cleaned_attributes,
                backup_path=backup_path
            )
            
        except json.JSONDecodeError as e:
            messages.append(ValidationMessage(
                severity=ValidationSeverity.CRITICAL,
                code="INVALID_JSON",
                message=f"Invalid JSON format: {e}",
                field_path=str(config_path),
                help_text="Fix JSON syntax errors"
            ))
            return ValidationResult(
                is_valid=False,
                messages=messages,
                cleaned_attributes=cleaned_attributes,
                backup_path=backup_path
            )
        except Exception as e:
            messages.append(ValidationMessage(
                severity=ValidationSeverity.CRITICAL,
                code="VALIDATION_ERROR",
                message=f"Validation error: {e}",
                field_path=str(config_path)
            ))
            return ValidationResult(
                is_valid=False,
                messages=messages,
                cleaned_attributes=cleaned_attributes,
                backup_path=backup_path
            )
    
    def create_backup(self, config_path: Union[str, Path]) -> str:
        """
        Create a backup of the configuration file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Path to backup file
        """
        config_path = Path(config_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{config_path.stem}_{timestamp}.backup{config_path.suffix}"
        backup_path = self.backup_dir / backup_filename
        
        try:
            shutil.copy2(config_path, backup_path)
            self.logger.info(f"Configuration backup created: {backup_path}")
            return str(backup_path)
        except Exception as e:
            self.logger.error(f"Failed to create backup: {e}")
            raise
    
    def _validate_schema(self, config_data: Dict[str, Any], schema: Dict[str, Any], path: str = "") -> List[ValidationMessage]:
        """
        Validate configuration data against schema.
        
        Args:
            config_data: Configuration data to validate
            schema: Schema to validate against
            path: Current field path for error reporting
            
        Returns:
            List of validation messages
        """
        messages = []
        
        for section_name, section_schema in schema.items():
            section_path = f"{path}.{section_name}" if path else section_name
            
            if section_name not in config_data:
                if section_schema.get("required", False):
                    messages.append(ValidationMessage(
                        severity=ValidationSeverity.ERROR,
                        code="MISSING_REQUIRED_SECTION",
                        message=f"Required section '{section_name}' is missing",
                        field_path=section_path,
                        help_text=f"Add the {section_name} section to your configuration"
                    ))
                continue
            
            section_data = config_data[section_name]
            
            # Validate section type
            if section_schema.get("type") == "object" and not isinstance(section_data, dict):
                messages.append(ValidationMessage(
                    severity=ValidationSeverity.ERROR,
                    code="INVALID_TYPE",
                    message=f"Section '{section_name}' must be an object",
                    field_path=section_path,
                    current_value=type(section_data).__name__
                ))
                continue
            
            # Validate required properties
            required_props = section_schema.get("required", [])
            for prop in required_props:
                if prop not in section_data:
                    messages.append(ValidationMessage(
                        severity=ValidationSeverity.ERROR,
                        code="MISSING_REQUIRED_PROPERTY",
                        message=f"Required property '{prop}' is missing from '{section_name}'",
                        field_path=f"{section_path}.{prop}",
                        help_text=f"Add the {prop} property to the {section_name} section"
                    ))
            
            # Validate properties
            properties = section_schema.get("properties", {})
            for prop_name, prop_schema in properties.items():
                if prop_name in section_data:
                    prop_messages = self._validate_property(
                        section_data[prop_name], 
                        prop_schema, 
                        f"{section_path}.{prop_name}"
                    )
                    messages.extend(prop_messages)
        
        return messages
    
    def _validate_property(self, value: Any, schema: Dict[str, Any], path: str) -> List[ValidationMessage]:
        """
        Validate a single property against its schema.
        
        Args:
            value: Property value to validate
            schema: Property schema
            path: Property path for error reporting
            
        Returns:
            List of validation messages
        """
        messages = []
        
        # Type validation
        expected_type = schema.get("type")
        if expected_type:
            if expected_type == "string" and not isinstance(value, str):
                messages.append(ValidationMessage(
                    severity=ValidationSeverity.ERROR,
                    code="INVALID_TYPE",
                    message=f"Property '{path}' must be a string",
                    field_path=path,
                    current_value=type(value).__name__
                ))
            elif expected_type == "integer" and not isinstance(value, int):
                messages.append(ValidationMessage(
                    severity=ValidationSeverity.ERROR,
                    code="INVALID_TYPE",
                    message=f"Property '{path}' must be an integer",
                    field_path=path,
                    current_value=type(value).__name__
                ))
            elif expected_type == "number" and not isinstance(value, (int, float)):
                messages.append(ValidationMessage(
                    severity=ValidationSeverity.ERROR,
                    code="INVALID_TYPE",
                    message=f"Property '{path}' must be a number",
                    field_path=path,
                    current_value=type(value).__name__
                ))
            elif expected_type == "boolean" and not isinstance(value, bool):
                messages.append(ValidationMessage(
                    severity=ValidationSeverity.ERROR,
                    code="INVALID_TYPE",
                    message=f"Property '{path}' must be a boolean",
                    field_path=path,
                    current_value=type(value).__name__
                ))
            elif expected_type == "array" and not isinstance(value, list):
                messages.append(ValidationMessage(
                    severity=ValidationSeverity.ERROR,
                    code="INVALID_TYPE",
                    message=f"Property '{path}' must be an array",
                    field_path=path,
                    current_value=type(value).__name__
                ))
        
        # Enum validation
        enum_values = schema.get("enum")
        if enum_values and value not in enum_values:
            messages.append(ValidationMessage(
                severity=ValidationSeverity.ERROR,
                code="INVALID_ENUM_VALUE",
                message=f"Property '{path}' has invalid value",
                field_path=path,
                current_value=value,
                suggested_value=enum_values[0],
                help_text=f"Must be one of: {', '.join(map(str, enum_values))}"
            ))
        
        # Range validation for numbers
        if isinstance(value, (int, float)):
            minimum = schema.get("minimum")
            maximum = schema.get("maximum")
            
            if minimum is not None and value < minimum:
                messages.append(ValidationMessage(
                    severity=ValidationSeverity.WARNING,
                    code="VALUE_TOO_LOW",
                    message=f"Property '{path}' value is below minimum",
                    field_path=path,
                    current_value=value,
                    suggested_value=minimum,
                    help_text=f"Minimum value is {minimum}"
                ))
            
            if maximum is not None and value > maximum:
                messages.append(ValidationMessage(
                    severity=ValidationSeverity.WARNING,
                    code="VALUE_TOO_HIGH",
                    message=f"Property '{path}' value is above maximum",
                    field_path=path,
                    current_value=value,
                    suggested_value=maximum,
                    help_text=f"Maximum value is {maximum}"
                ))
        
        # Pattern validation for strings
        pattern = schema.get("pattern")
        if pattern and isinstance(value, str):
            import re
            if not re.match(pattern, value):
                messages.append(ValidationMessage(
                    severity=ValidationSeverity.ERROR,
                    code="INVALID_PATTERN",
                    message=f"Property '{path}' does not match required pattern",
                    field_path=path,
                    current_value=value,
                    help_text=f"Must match pattern: {pattern}"
                ))
        
        return messages
    
    def _cleanup_config(self, config_data: Dict[str, Any]) -> CleanupResult:
        """
        Clean up unexpected attributes from configuration.
        
        Args:
            config_data: Configuration data to clean
            
        Returns:
            CleanupResult with list of cleaned attributes
        """
        cleaned_attributes = []
        
        # Clean up general attributes from root level
        general_attrs = self.cleanup_attributes.get("general", set())
        for attr in list(general_attrs):  # Use list() to avoid modification during iteration
            if attr in config_data:
                del config_data[attr]
                cleaned_attributes.append(f"root.{attr}")
                self.logger.info(f"Removed unexpected attribute: {attr}")
        
        # Also clean up specific problematic attributes from root level (main issues from requirements)
        root_level_cleanup_attrs = ["clip_output", "force_upcast", "use_linear_projection", "cross_attention_dim"]
        for attr in root_level_cleanup_attrs:
            if attr in config_data:
                del config_data[attr]
                cleaned_attributes.append(f"root.{attr}")
                self.logger.info(f"Removed unexpected attribute: {attr}")
        
        # Clean up section-specific attributes
        for config_section_name, section_data in config_data.items():
            if isinstance(section_data, dict):
                # Check all cleanup categories for attributes to remove from this section
                for cleanup_category, cleanup_attrs in self.cleanup_attributes.items():
                    if cleanup_category != "general":
                        for attr in list(cleanup_attrs):  # Use list() to avoid modification during iteration
                            if attr in section_data:
                                del section_data[attr]
                                cleaned_attributes.append(f"{config_section_name}.{attr}")
                                self.logger.info(f"Removed unexpected attribute: {config_section_name}.{attr}")
        
        return CleanupResult(
            cleaned_attributes=cleaned_attributes,
            backup_created=True
        )


def validate_config_file(config_path: Union[str, Path], backup_dir: Optional[Path] = None) -> ValidationResult:
    """
    Convenience function to validate a configuration file.
    
    Args:
        config_path: Path to configuration file
        backup_dir: Directory for backups (optional)
        
    Returns:
        ValidationResult with validation messages and cleanup info
    """
    validator = ConfigValidator(backup_dir)
    return validator.validate_config_file(config_path)


def format_validation_result(result: ValidationResult) -> str:
    """
    Format validation result for display.
    
    Args:
        result: ValidationResult to format
        
    Returns:
        Formatted string representation
    """
    lines = []
    
    if result.is_valid:
        lines.append("✅ Configuration is valid")
    else:
        lines.append("❌ Configuration has errors")
    
    if result.backup_path:
        lines.append(f"📁 Backup created: {result.backup_path}")
    
    if result.cleaned_attributes:
        lines.append(f"🧹 Cleaned {len(result.cleaned_attributes)} attributes:")
        for attr in result.cleaned_attributes:
            lines.append(f"  - {attr}")
    
    # Group messages by severity
    for severity in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR, 
                    ValidationSeverity.WARNING, ValidationSeverity.INFO]:
        severity_messages = [msg for msg in result.messages if msg.severity == severity]
        if severity_messages:
            icon = {"critical": "🚨", "error": "❌", "warning": "⚠️", "info": "ℹ️"}[severity.value]
            lines.append(f"\n{icon} {severity.value.upper()} ({len(severity_messages)}):")
            for msg in severity_messages:
                lines.append(f"  {msg.code}: {msg.message}")
                if msg.field_path:
                    lines.append(f"    Path: {msg.field_path}")
                if msg.current_value is not None:
                    lines.append(f"    Current: {msg.current_value}")
                if msg.suggested_value is not None:
                    lines.append(f"    Suggested: {msg.suggested_value}")
                if msg.help_text:
                    lines.append(f"    Help: {msg.help_text}")
    
    return "\n".join(lines)
