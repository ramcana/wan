"""
Configuration Validation System

This module provides comprehensive validation for the unified configuration,
including schema validation, dependency checking, and consistency validation.
"""

import json
import yaml
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

try:
    import jsonschema
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False

from tools.config_manager.unified_config import UnifiedConfig, LogLevel, QuantizationLevel, Environment


class ValidationSeverity(Enum):
    """Severity levels for validation issues"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Represents a configuration validation issue"""
    severity: ValidationSeverity
    category: str
    field_path: str
    message: str
    current_value: Any
    suggested_value: Optional[Any] = None
    fix_suggestion: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of configuration validation"""
    is_valid: bool
    issues: List[ValidationIssue]
    warnings_count: int
    errors_count: int
    critical_count: int
    
    def has_errors(self) -> bool:
        """Check if there are any errors or critical issues"""
        return self.errors_count > 0 or self.critical_count > 0
    
    def get_issues_by_severity(self, severity: ValidationSeverity) -> List[ValidationIssue]:
        """Get issues filtered by severity"""
        return [issue for issue in self.issues if issue.severity == severity]


class ConfigurationValidator:
    """
    Comprehensive configuration validation system
    """
    
    def __init__(self, schema_path: Optional[Path] = None):
        self.logger = logging.getLogger(__name__)
        self.schema_path = schema_path or Path("config/schemas/unified-config-schema.yaml")
        self.schema = self._load_schema()
        
        # Validation rules
        self.validation_rules = {
            'port_ranges': self._validate_port_ranges,
            'file_paths': self._validate_file_paths,
            'memory_limits': self._validate_memory_limits,
            'timeout_values': self._validate_timeout_values,
            'dependency_consistency': self._validate_dependency_consistency,
            'environment_consistency': self._validate_environment_consistency,
            'security_settings': self._validate_security_settings,
            'performance_settings': self._validate_performance_settings
        }
    
    def _load_schema(self) -> Optional[Dict[str, Any]]:
        """Load the configuration schema"""
        if not self.schema_path.exists():
            self.logger.warning(f"Schema file not found: {self.schema_path}")
            return None
        
        try:
            if self.schema_path.suffix.lower() in ['.yaml', '.yml']:
                return yaml.safe_load(self.schema_path.read_text())
            elif self.schema_path.suffix.lower() == '.json':
                return json.loads(self.schema_path.read_text())
        except Exception as e:
            self.logger.error(f"Failed to load schema: {e}")
        
        return None
    
    def validate_config(self, config: UnifiedConfig) -> ValidationResult:
        """
        Perform comprehensive validation of a configuration
        
        Args:
            config: Configuration to validate
            
        Returns:
            Validation result with any issues found
        """
        issues = []
        
        # Convert config to dict for validation
        config_dict = config.to_dict()
        
        # Schema validation
        if self.schema and JSONSCHEMA_AVAILABLE:
            issues.extend(self._validate_schema(config_dict))
        
        # Custom validation rules
        for rule_name, rule_func in self.validation_rules.items():
            try:
                rule_issues = rule_func(config)
                issues.extend(rule_issues)
            except Exception as e:
                self.logger.error(f"Validation rule {rule_name} failed: {e}")
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="validation_system",
                    field_path=rule_name,
                    message=f"Validation rule failed: {e}",
                    current_value=None
                ))
        
        # Count issues by severity
        warnings_count = len([i for i in issues if i.severity == ValidationSeverity.WARNING])
        errors_count = len([i for i in issues if i.severity == ValidationSeverity.ERROR])
        critical_count = len([i for i in issues if i.severity == ValidationSeverity.CRITICAL])
        
        return ValidationResult(
            is_valid=errors_count == 0 and critical_count == 0,
            issues=issues,
            warnings_count=warnings_count,
            errors_count=errors_count,
            critical_count=critical_count
        )
    
    def _validate_schema(self, config_dict: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate configuration against JSON schema"""
        issues = []
        
        try:
            jsonschema.validate(config_dict, self.schema)
        except jsonschema.ValidationError as e:
            # Convert jsonschema errors to our format
            field_path = '.'.join(str(p) for p in e.absolute_path)
            
            issue = ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="schema",
                field_path=field_path,
                message=e.message,
                current_value=e.instance,
                fix_suggestion=self._generate_schema_fix_suggestion(e)
            )
            issues.append(issue)
        except jsonschema.SchemaError as e:
            self.logger.error(f"Schema validation error: {e}")
        
        return issues
    
    def _generate_schema_fix_suggestion(self, error: 'jsonschema.ValidationError') -> str:
        """Generate fix suggestion for schema validation errors"""
        if 'is not of type' in error.message:
            return f"Change value to correct type: {error.validator_value}"
        elif 'is not one of' in error.message:
            return f"Use one of the allowed values: {error.validator_value}"
        elif 'is too' in error.message:
            return f"Adjust value to meet constraint: {error.message}"
        else:
            return "Check the configuration schema for valid values"
    
    def _validate_port_ranges(self, config: UnifiedConfig) -> List[ValidationIssue]:
        """Validate port number ranges and conflicts"""
        issues = []
        
        # Check port ranges
        ports_to_check = [
            ('api.port', config.api.port),
            ('frontend.port', config.frontend.port)
        ]
        
        for field_path, port in ports_to_check:
            if not (1024 <= port <= 65535):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="ports",
                    field_path=field_path,
                    message=f"Port {port} is outside valid range (1024-65535)",
                    current_value=port,
                    suggested_value=8000 if 'api' in field_path else 3000,
                    fix_suggestion="Use a port number between 1024 and 65535"
                ))
        
        # Check for port conflicts
        if config.api.port == config.frontend.port:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="ports",
                field_path="api.port,frontend.port",
                message="API and frontend ports cannot be the same",
                current_value=f"Both using port {config.api.port}",
                fix_suggestion="Use different port numbers for API and frontend"
            ))
        
        # Check trusted port range
        if len(config.security.trusted_port_range) == 2:
            start, end = config.security.trusted_port_range
            if start >= end:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="ports",
                    field_path="security.trusted_port_range",
                    message="Trusted port range start must be less than end",
                    current_value=config.security.trusted_port_range,
                    fix_suggestion="Ensure start port < end port"
                ))
        
        return issues
    
    def _validate_file_paths(self, config: UnifiedConfig) -> List[ValidationIssue]:
        """Validate file and directory paths"""
        issues = []
        
        # Check directory paths
        directories = [
            ('system.output_directory', config.system.output_directory),
            ('system.models_directory', config.system.models_directory),
            ('system.loras_directory', config.system.loras_directory),
            ('models.base_path', config.models.base_path),
            ('models.cache_dir', config.models.cache_dir)
        ]
        
        for field_path, directory in directories:
            if not directory or directory.strip() == '':
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="paths",
                    field_path=field_path,
                    message="Directory path cannot be empty",
                    current_value=directory,
                    fix_suggestion="Provide a valid directory path"
                ))
            elif directory.startswith('/') and not Path(directory).is_absolute():
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="paths",
                    field_path=field_path,
                    message="Absolute path may not be portable across systems",
                    current_value=directory,
                    fix_suggestion="Consider using relative paths for portability"
                ))
        
        # Check log file path
        if config.logging.file_enabled and config.logging.file:
            log_path = Path(config.logging.file)
            if not log_path.parent.exists() and not log_path.is_absolute():
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="paths",
                    field_path="logging.file",
                    message="Log directory may not exist",
                    current_value=config.logging.file,
                    fix_suggestion="Ensure log directory exists or will be created"
                ))
        
        return issues
    
    def _validate_memory_limits(self, config: UnifiedConfig) -> List[ValidationIssue]:
        """Validate memory and VRAM limits"""
        issues = []
        
        # Check VRAM limits consistency
        if config.hardware.vram_limit_gb != config.hardware.max_vram_usage_gb:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="memory",
                field_path="hardware.vram_limit_gb,hardware.max_vram_usage_gb",
                message="VRAM limit and max usage should typically be the same",
                current_value=f"limit: {config.hardware.vram_limit_gb}, max: {config.hardware.max_vram_usage_gb}",
                fix_suggestion="Set both values to the same amount or ensure the difference is intentional"
            ))
        
        # Check VAE tile size range
        if len(config.hardware.vae_tile_size_range) == 2:
            min_size, max_size = config.hardware.vae_tile_size_range
            if config.hardware.vae_tile_size < min_size or config.hardware.vae_tile_size > max_size:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="memory",
                    field_path="hardware.vae_tile_size",
                    message=f"VAE tile size {config.hardware.vae_tile_size} is outside allowed range {config.hardware.vae_tile_size_range}",
                    current_value=config.hardware.vae_tile_size,
                    suggested_value=min_size,
                    fix_suggestion=f"Set VAE tile size between {min_size} and {max_size}"
                ))
        
        # Check performance thresholds
        thresholds = [
            ('performance.vram_warning_threshold', config.performance.vram_warning_threshold, 0.1, 1.0),
            ('performance.vram_warning_percent', config.performance.vram_warning_percent, 50, 100),
            ('performance.cpu_warning_percent', config.performance.cpu_warning_percent, 50, 100),
            ('performance.memory_warning_percent', config.performance.memory_warning_percent, 50, 100)
        ]
        
        for field_path, value, min_val, max_val in thresholds:
            if not (min_val <= value <= max_val):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="memory",
                    field_path=field_path,
                    message=f"Value {value} is outside valid range ({min_val}-{max_val})",
                    current_value=value,
                    fix_suggestion=f"Set value between {min_val} and {max_val}"
                ))
        
        return issues
    
    def _validate_timeout_values(self, config: UnifiedConfig) -> List[ValidationIssue]:
        """Validate timeout and duration values"""
        issues = []
        
        # Check timeout values are reasonable
        timeouts = [
            ('api.timeout', config.api.timeout, 30, 3600),
            ('frontend.timeout', config.frontend.timeout, 10, 300),
            ('database.connect_timeout', config.database.connect_timeout, 5, 300),
            ('database.query_timeout', config.database.query_timeout, 10, 600),
            ('generation.generation_timeout_minutes', config.generation.generation_timeout_minutes, 5, 180),
            ('generation.default_timeout', config.generation.default_timeout, 60, 3600),
            ('models.download_timeout', config.models.download_timeout, 300, 7200)
        ]
        
        for field_path, value, min_val, max_val in timeouts:
            if not (min_val <= value <= max_val):
                severity = ValidationSeverity.WARNING if value > max_val else ValidationSeverity.ERROR
                issues.append(ValidationIssue(
                    severity=severity,
                    category="timeouts",
                    field_path=field_path,
                    message=f"Timeout value {value} may be too {'high' if value > max_val else 'low'}",
                    current_value=value,
                    fix_suggestion=f"Consider setting timeout between {min_val} and {max_val} seconds/minutes"
                ))
        
        return issues
    
    def _validate_dependency_consistency(self, config: UnifiedConfig) -> List[ValidationIssue]:
        """Validate consistency between dependent settings"""
        issues = []
        
        # Check model and generation consistency
        if config.generation.mode == "real" and not config.generation.enable_real_models:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="dependencies",
                field_path="generation.mode,generation.enable_real_models",
                message="Real generation mode requires enable_real_models to be true",
                current_value=f"mode: {config.generation.mode}, enable_real_models: {config.generation.enable_real_models}",
                fix_suggestion="Set enable_real_models to true or change mode to 'mock'"
            ))
        
        # Check authentication and security
        if config.security.authentication_enabled and config.security.secret_key in ["change-in-production", "dev-secret-key"]:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                category="dependencies",
                field_path="security.secret_key",
                message="Default secret key should not be used with authentication enabled",
                current_value=config.security.secret_key,
                fix_suggestion="Generate a strong, unique secret key for production use"
            ))
        
        # Check logging consistency
        if config.logging.file_enabled and not config.logging.file:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="dependencies",
                field_path="logging.file_enabled,logging.file",
                message="File logging enabled but no log file specified",
                current_value=f"file_enabled: {config.logging.file_enabled}, file: {config.logging.file}",
                fix_suggestion="Specify a log file path or disable file logging"
            ))
        
        # Check performance monitoring
        if config.performance.enabled and config.performance.sample_interval_seconds <= 0:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="dependencies",
                field_path="performance.sample_interval_seconds",
                message="Performance monitoring enabled but invalid sample interval",
                current_value=config.performance.sample_interval_seconds,
                fix_suggestion="Set a positive sample interval (e.g., 30.0 seconds)"
            ))
        
        return issues
    
    def _validate_environment_consistency(self, config: UnifiedConfig) -> List[ValidationIssue]:
        """Validate environment-specific consistency"""
        issues = []
        
        # Check production environment settings
        if config.system.environment == Environment.PRODUCTION:
            if config.system.debug:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="environment",
                    field_path="system.debug",
                    message="Debug mode should be disabled in production",
                    current_value=config.system.debug,
                    suggested_value=False,
                    fix_suggestion="Set debug to false for production environment"
                ))
            
            if config.api.debug:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="environment",
                    field_path="api.debug",
                    message="API debug mode should be disabled in production",
                    current_value=config.api.debug,
                    suggested_value=False,
                    fix_suggestion="Set API debug to false for production environment"
                ))
            
            if not config.security.authentication_enabled:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    category="environment",
                    field_path="security.authentication_enabled",
                    message="Authentication should be enabled in production",
                    current_value=config.security.authentication_enabled,
                    suggested_value=True,
                    fix_suggestion="Enable authentication for production environment"
                ))
        
        # Check development environment settings
        elif config.system.environment == Environment.DEVELOPMENT:
            if not config.frontend.hot_reload:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    category="environment",
                    field_path="frontend.hot_reload",
                    message="Hot reload is typically enabled in development",
                    current_value=config.frontend.hot_reload,
                    suggested_value=True,
                    fix_suggestion="Consider enabling hot reload for better development experience"
                ))
        
        return issues
    
    def _validate_security_settings(self, config: UnifiedConfig) -> List[ValidationIssue]:
        """Validate security-related settings"""
        issues = []
        
        # Check secret key strength
        if len(config.security.secret_key) < 16:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                category="security",
                field_path="security.secret_key",
                message="Secret key is too short (minimum 16 characters)",
                current_value=f"Length: {len(config.security.secret_key)}",
                fix_suggestion="Use a secret key with at least 16 characters"
            ))
        
        # Check for weak secret keys
        weak_keys = ["secret", "password", "key", "change-me", "default"]
        if any(weak in config.security.secret_key.lower() for weak in weak_keys):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                category="security",
                field_path="security.secret_key",
                message="Secret key appears to be weak or default",
                current_value="[REDACTED]",
                fix_suggestion="Generate a strong, random secret key"
            ))
        
        # Check CORS settings
        if config.security.cors_enabled and "*" in config.api.cors_origins:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="security",
                field_path="api.cors_origins",
                message="Wildcard CORS origin allows all domains",
                current_value=config.api.cors_origins,
                fix_suggestion="Specify explicit allowed origins instead of using '*'"
            ))
        
        # Check token expiry
        if config.security.access_token_expire_minutes > 1440:  # 24 hours
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="security",
                field_path="security.access_token_expire_minutes",
                message="Token expiry time is very long",
                current_value=config.security.access_token_expire_minutes,
                fix_suggestion="Consider shorter token expiry for better security"
            ))
        
        return issues
    
    def _validate_performance_settings(self, config: UnifiedConfig) -> List[ValidationIssue]:
        """Validate performance-related settings"""
        issues = []
        
        # Check concurrent limits
        if config.generation.max_concurrent_generations > config.generation.max_concurrent_jobs:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="performance",
                field_path="generation.max_concurrent_generations",
                message="Concurrent generations should not exceed concurrent jobs",
                current_value=config.generation.max_concurrent_generations,
                suggested_value=config.generation.max_concurrent_jobs,
                fix_suggestion="Set max_concurrent_generations <= max_concurrent_jobs"
            ))
        
        # Check worker count vs CPU threads
        if config.api.workers > config.hardware.cpu_threads:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="performance",
                field_path="api.workers",
                message="More API workers than CPU threads may cause contention",
                current_value=config.api.workers,
                suggested_value=min(config.api.workers, config.hardware.cpu_threads),
                fix_suggestion="Consider reducing worker count to match available CPU threads"
            ))
        
        # Check sample intervals
        if config.performance.enabled and config.performance.sample_interval_seconds < 1.0:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="performance",
                field_path="performance.sample_interval_seconds",
                message="Very frequent performance sampling may impact performance",
                current_value=config.performance.sample_interval_seconds,
                fix_suggestion="Consider increasing sample interval to reduce overhead"
            ))
        
        return issues
    
    def validate_config_file(self, config_path: Union[str, Path]) -> ValidationResult:
        """
        Validate a configuration file
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Validation result
        """
        try:
            config = UnifiedConfig.from_file(config_path)
            return self.validate_config(config)
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                issues=[ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    category="file_loading",
                    field_path="file",
                    message=f"Failed to load configuration file: {e}",
                    current_value=str(config_path)
                )],
                warnings_count=0,
                errors_count=0,
                critical_count=1
            )
    
    def generate_validation_report(self, result: ValidationResult) -> str:
        """
        Generate a human-readable validation report
        
        Args:
            result: Validation result to report on
            
        Returns:
            Formatted validation report
        """
        lines = []
        lines.append("Configuration Validation Report")
        lines.append("=" * 50)
        lines.append(f"Overall Status: {'VALID' if result.is_valid else 'INVALID'}")
        lines.append(f"Issues Found: {len(result.issues)}")
        lines.append(f"  - Critical: {result.critical_count}")
        lines.append(f"  - Errors: {result.errors_count}")
        lines.append(f"  - Warnings: {result.warnings_count}")
        lines.append("")
        
        if result.issues:
            # Group issues by category
            by_category = {}
            for issue in result.issues:
                if issue.category not in by_category:
                    by_category[issue.category] = []
                by_category[issue.category].append(issue)
            
            for category, category_issues in sorted(by_category.items()):
                lines.append(f"{category.upper()} Issues:")
                lines.append("-" * 30)
                
                for issue in category_issues:
                    severity_symbol = {
                        ValidationSeverity.INFO: "ℹ",
                        ValidationSeverity.WARNING: "⚠",
                        ValidationSeverity.ERROR: "❌",
                        ValidationSeverity.CRITICAL: "🚨"
                    }.get(issue.severity, "?")
                    
                    lines.append(f"{severity_symbol} [{issue.severity.value.upper()}] {issue.field_path}")
                    lines.append(f"   {issue.message}")
                    if issue.current_value is not None:
                        lines.append(f"   Current: {issue.current_value}")
                    if issue.suggested_value is not None:
                        lines.append(f"   Suggested: {issue.suggested_value}")
                    if issue.fix_suggestion:
                        lines.append(f"   Fix: {issue.fix_suggestion}")
                    lines.append("")
                
                lines.append("")
        
        return "\n".join(lines)