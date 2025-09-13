"""
Configuration Validation System for Enhanced Model Management

Provides comprehensive validation for configuration settings, including
business rule validation, constraint checking, and migration support.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union
from enum import Enum
import re
from pathlib import Path

from .enhanced_model_config import (
    UserPreferences, AdminPolicies, FeatureFlagConfig,
    DownloadConfig, HealthMonitoringConfig, FallbackConfig,
    AnalyticsConfig, UpdateConfig, NotificationConfig, StorageConfig,
    AutomationLevel, FeatureFlag
)


@dataclass
class ValidationError:
    """Represents a configuration validation error"""
    field: str
    message: str
    severity: str = "error"  # error, warning, info
    suggested_value: Optional[Any] = None


@dataclass
class ValidationResult:
    """Result of configuration validation"""
    is_valid: bool
    errors: List[ValidationError]
    warnings: List[ValidationError]
    
    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0
    
    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0


class ConfigurationValidator:
    """Validates enhanced model configuration settings"""
    
    def __init__(self):
        self.min_values = {
            'max_retries': 0,
            'retry_delay_base': 0.1,
            'max_retry_delay': 1.0,
            'bandwidth_limit_mbps': 0.1,
            'max_concurrent_downloads': 1,
            'chunk_size_mb': 0.1,
            'check_interval_hours': 1,
            'corruption_threshold': 0.0,
            'performance_degradation_threshold': 0.0,
            'compatibility_threshold': 0.0,
            'max_wait_time_minutes': 1,
            'retention_days': 1,
            'check_interval_hours': 1,
            'cleanup_threshold_percent': 50.0,
            'min_free_space_gb': 0.1,
            'preserve_recent_models_days': 0,
        }
        
        self.max_values = {
            'max_retries': 10,
            'retry_delay_base': 60.0,
            'max_retry_delay': 3600.0,
            'bandwidth_limit_mbps': 10000.0,
            'max_concurrent_downloads': 10,
            'chunk_size_mb': 1000.0,
            'check_interval_hours': 8760,  # 1 year
            'corruption_threshold': 1.0,
            'performance_degradation_threshold': 1.0,
            'compatibility_threshold': 1.0,
            'max_wait_time_minutes': 1440,  # 24 hours
            'retention_days': 3650,  # 10 years
            'cleanup_threshold_percent': 99.0,
            'min_free_space_gb': 1000.0,
            'preserve_recent_models_days': 365,
        }
    
    def validate_user_preferences(self, preferences: UserPreferences) -> ValidationResult:
        """Validate user preferences configuration"""
        errors = []
        warnings = []
        
        # Validate automation level
        if not isinstance(preferences.automation_level, AutomationLevel):
            errors.append(ValidationError(
                "automation_level",
                "Invalid automation level",
                suggested_value=AutomationLevel.SEMI_AUTOMATIC
            ))
        
        # Validate download config
        download_errors = self._validate_download_config(preferences.download_config)
        errors.extend(download_errors)
        
        # Validate health monitoring config
        health_errors = self._validate_health_monitoring_config(preferences.health_monitoring)
        errors.extend(health_errors)
        
        # Validate fallback config
        fallback_errors = self._validate_fallback_config(preferences.fallback_config)
        errors.extend(fallback_errors)
        
        # Validate analytics config
        analytics_errors = self._validate_analytics_config(preferences.analytics_config)
        errors.extend(analytics_errors)
        
        # Validate update config
        update_errors = self._validate_update_config(preferences.update_config)
        errors.extend(update_errors)
        
        # Validate notification config
        notification_errors = self._validate_notification_config(preferences.notification_config)
        errors.extend(notification_errors)
        
        # Validate storage config
        storage_errors = self._validate_storage_config(preferences.storage_config)
        errors.extend(storage_errors)
        
        # Validate model lists
        model_errors = self._validate_model_lists(preferences.preferred_models, preferences.blocked_models)
        errors.extend(model_errors)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )   
 
    def validate_admin_policies(self, policies: AdminPolicies) -> ValidationResult:
        """Validate admin policies configuration"""
        errors = []
        warnings = []
        
        # Validate storage limits
        if policies.max_user_storage_gb is not None:
            if policies.max_user_storage_gb <= 0:
                errors.append(ValidationError(
                    "max_user_storage_gb",
                    "Maximum user storage must be positive",
                    suggested_value=10.0
                ))
        
        # Validate bandwidth limits
        if policies.bandwidth_limit_per_user_mbps is not None:
            if policies.bandwidth_limit_per_user_mbps <= 0:
                errors.append(ValidationError(
                    "bandwidth_limit_per_user_mbps",
                    "Bandwidth limit must be positive",
                    suggested_value=10.0
                ))
        
        # Validate concurrent users
        if policies.max_concurrent_users is not None:
            if policies.max_concurrent_users <= 0:
                errors.append(ValidationError(
                    "max_concurrent_users",
                    "Maximum concurrent users must be positive",
                    suggested_value=10
                ))
        
        # Validate model sources
        valid_sources = {"huggingface", "local", "custom", "s3", "gcs"}
        for source in policies.allowed_model_sources:
            if source not in valid_sources:
                warnings.append(ValidationError(
                    "allowed_model_sources",
                    f"Unknown model source: {source}",
                    severity="warning"
                ))
        
        # Validate blocked model patterns
        for pattern in policies.blocked_model_patterns:
            try:
                re.compile(pattern)
            except re.error as e:
                errors.append(ValidationError(
                    "blocked_model_patterns",
                    f"Invalid regex pattern '{pattern}': {e}"
                ))
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def validate_feature_flags(self, feature_flags: FeatureFlagConfig) -> ValidationResult:
        """Validate feature flag configuration"""
        errors = []
        warnings = []
        
        # Validate flag names
        valid_flags = {flag.value for flag in FeatureFlag}
        for flag_name in feature_flags.flags:
            if flag_name not in valid_flags:
                warnings.append(ValidationError(
                    "flags",
                    f"Unknown feature flag: {flag_name}",
                    severity="warning"
                ))
        
        # Validate rollout percentages
        for flag_name, percentage in feature_flags.rollout_percentage.items():
            if not 0 <= percentage <= 100:
                errors.append(ValidationError(
                    "rollout_percentage",
                    f"Rollout percentage for {flag_name} must be between 0 and 100",
                    suggested_value=50.0
                ))
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def _validate_download_config(self, config: DownloadConfig) -> List[ValidationError]:
        """Validate download configuration"""
        errors = []
        
        # Validate numeric ranges
        numeric_fields = [
            'max_retries', 'retry_delay_base', 'max_retry_delay',
            'bandwidth_limit_mbps', 'max_concurrent_downloads', 'chunk_size_mb'
        ]
        
        for field in numeric_fields:
            value = getattr(config, field)
            if value is not None:
                min_val = self.min_values.get(field)
                max_val = self.max_values.get(field)
                
                if min_val is not None and value < min_val:
                    errors.append(ValidationError(
                        f"download_config.{field}",
                        f"Value {value} is below minimum {min_val}",
                        suggested_value=min_val
                    ))
                
                if max_val is not None and value > max_val:
                    errors.append(ValidationError(
                        f"download_config.{field}",
                        f"Value {value} is above maximum {max_val}",
                        suggested_value=max_val
                    ))
        
        # Validate logical constraints
        if config.retry_delay_base > config.max_retry_delay:
            errors.append(ValidationError(
                "download_config.retry_delay_base",
                "Base retry delay cannot be greater than maximum retry delay"
            ))
        
        return errors    

    def _validate_health_monitoring_config(self, config: HealthMonitoringConfig) -> List[ValidationError]:
        """Validate health monitoring configuration"""
        errors = []
        
        # Validate numeric ranges
        if config.check_interval_hours < self.min_values['check_interval_hours']:
            errors.append(ValidationError(
                "health_monitoring.check_interval_hours",
                f"Check interval must be at least {self.min_values['check_interval_hours']} hours"
            ))
        
        if not 0 <= config.corruption_threshold <= 1:
            errors.append(ValidationError(
                "health_monitoring.corruption_threshold",
                "Corruption threshold must be between 0 and 1"
            ))
        
        if not 0 <= config.performance_degradation_threshold <= 1:
            errors.append(ValidationError(
                "health_monitoring.performance_degradation_threshold",
                "Performance degradation threshold must be between 0 and 1"
            ))
        
        return errors
    
    def _validate_fallback_config(self, config: FallbackConfig) -> List[ValidationError]:
        """Validate fallback configuration"""
        errors = []
        
        if not 0 <= config.compatibility_threshold <= 1:
            errors.append(ValidationError(
                "fallback_config.compatibility_threshold",
                "Compatibility threshold must be between 0 and 1"
            ))
        
        if config.max_wait_time_minutes < self.min_values['max_wait_time_minutes']:
            errors.append(ValidationError(
                "fallback_config.max_wait_time_minutes",
                f"Max wait time must be at least {self.min_values['max_wait_time_minutes']} minutes"
            ))
        
        return errors
    
    def _validate_analytics_config(self, config: AnalyticsConfig) -> List[ValidationError]:
        """Validate analytics configuration"""
        errors = []
        
        if config.retention_days < self.min_values['retention_days']:
            errors.append(ValidationError(
                "analytics_config.retention_days",
                f"Retention period must be at least {self.min_values['retention_days']} days"
            ))
        
        return errors
    
    def _validate_update_config(self, config: UpdateConfig) -> List[ValidationError]:
        """Validate update configuration"""
        errors = []
        
        if config.check_interval_hours < self.min_values['check_interval_hours']:
            errors.append(ValidationError(
                "update_config.check_interval_hours",
                f"Check interval must be at least {self.min_values['check_interval_hours']} hours"
            ))
        
        return errors
    
    def _validate_notification_config(self, config: NotificationConfig) -> List[ValidationError]:
        """Validate notification configuration"""
        errors = []
        
        # Validate webhook URL if provided
        if config.webhook_url:
            if not config.webhook_url.startswith(('http://', 'https://')):
                errors.append(ValidationError(
                    "notification_config.webhook_url",
                    "Webhook URL must start with http:// or https://"
                ))
        
        return errors
    
    def _validate_storage_config(self, config: StorageConfig) -> List[ValidationError]:
        """Validate storage configuration"""
        errors = []
        
        if not 50 <= config.cleanup_threshold_percent <= 99:
            errors.append(ValidationError(
                "storage_config.cleanup_threshold_percent",
                "Cleanup threshold must be between 50% and 99%"
            ))
        
        if config.min_free_space_gb < self.min_values['min_free_space_gb']:
            errors.append(ValidationError(
                "storage_config.min_free_space_gb",
                f"Minimum free space must be at least {self.min_values['min_free_space_gb']} GB"
            ))
        
        return errors
    
    def _validate_model_lists(self, preferred: List[str], blocked: List[str]) -> List[ValidationError]:
        """Validate model preference lists"""
        errors = []
        
        # Check for overlap between preferred and blocked models
        overlap = set(preferred) & set(blocked)
        if overlap:
            errors.append(ValidationError(
                "model_lists",
                f"Models cannot be both preferred and blocked: {', '.join(overlap)}"
            ))
        
        # Validate model name format (basic validation)
        model_pattern = re.compile(r'^[a-zA-Z0-9_-]+(/[a-zA-Z0-9_.-]+)*$')
        
        for model in preferred + blocked:
            if not model_pattern.match(model):
                errors.append(ValidationError(
                    "model_lists",
                    f"Invalid model name format: {model}"
                ))
        
        return errors
