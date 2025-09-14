"""
Feature Flags System - Gradual rollout and configuration management.

This module provides a centralized feature flag system for controlling
the rollout of model orchestrator features with environment-based configuration.
"""

import os
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorFeatureFlags:
    """Feature flags for model orchestrator functionality."""
    
    # Core orchestrator features
    enable_orchestrator: bool = False
    enable_manifest_validation: bool = True
    enable_legacy_fallback: bool = True
    enable_path_migration: bool = False
    enable_automatic_download: bool = False
    
    # Validation and safety features
    strict_validation: bool = False
    enable_integrity_checks: bool = True
    enable_disk_space_checks: bool = True
    enable_concurrent_downloads: bool = False
    
    # Storage backend features
    enable_s3_backend: bool = False
    enable_hf_backend: bool = True
    enable_local_backend: bool = True
    enable_component_deduplication: bool = False
    
    # Monitoring and observability
    enable_metrics: bool = False
    enable_structured_logging: bool = True
    enable_health_endpoints: bool = True
    enable_performance_tracking: bool = False
    
    # Security features
    enable_credential_encryption: bool = False
    enable_manifest_signing: bool = False
    enable_secure_downloads: bool = True
    
    # Development and debugging
    enable_debug_mode: bool = False
    enable_dry_run_mode: bool = False
    enable_verbose_logging: bool = False
    
    @classmethod
    def from_env(cls, prefix: str = "WAN_") -> 'OrchestratorFeatureFlags':
        """
        Create feature flags from environment variables.
        
        Args:
            prefix: Environment variable prefix (default: "WAN_")
            
        Returns:
            OrchestratorFeatureFlags instance with values from environment
        """
        def get_bool_env(key: str, default: bool) -> bool:
            """Get boolean value from environment variable."""
            value = os.getenv(f"{prefix}{key}", str(default)).lower()
            return value in ('true', '1', 'yes', 'on')
        
        return cls(
            # Core orchestrator features
            enable_orchestrator=get_bool_env('ENABLE_ORCHESTRATOR', False),
            enable_manifest_validation=get_bool_env('ENABLE_MANIFEST_VALIDATION', True),
            enable_legacy_fallback=get_bool_env('ENABLE_LEGACY_FALLBACK', True),
            enable_path_migration=get_bool_env('ENABLE_PATH_MIGRATION', False),
            enable_automatic_download=get_bool_env('ENABLE_AUTO_DOWNLOAD', False),
            
            # Validation and safety features
            strict_validation=get_bool_env('STRICT_VALIDATION', False),
            enable_integrity_checks=get_bool_env('ENABLE_INTEGRITY_CHECKS', True),
            enable_disk_space_checks=get_bool_env('ENABLE_DISK_SPACE_CHECKS', True),
            enable_concurrent_downloads=get_bool_env('ENABLE_CONCURRENT_DOWNLOADS', False),
            
            # Storage backend features
            enable_s3_backend=get_bool_env('ENABLE_S3_BACKEND', False),
            enable_hf_backend=get_bool_env('ENABLE_HF_BACKEND', True),
            enable_local_backend=get_bool_env('ENABLE_LOCAL_BACKEND', True),
            enable_component_deduplication=get_bool_env('ENABLE_COMPONENT_DEDUP', False),
            
            # Monitoring and observability
            enable_metrics=get_bool_env('ENABLE_METRICS', False),
            enable_structured_logging=get_bool_env('ENABLE_STRUCTURED_LOGGING', True),
            enable_health_endpoints=get_bool_env('ENABLE_HEALTH_ENDPOINTS', True),
            enable_performance_tracking=get_bool_env('ENABLE_PERFORMANCE_TRACKING', False),
            
            # Security features
            enable_credential_encryption=get_bool_env('ENABLE_CREDENTIAL_ENCRYPTION', False),
            enable_manifest_signing=get_bool_env('ENABLE_MANIFEST_SIGNING', False),
            enable_secure_downloads=get_bool_env('ENABLE_SECURE_DOWNLOADS', True),
            
            # Development and debugging
            enable_debug_mode=get_bool_env('DEBUG_MODE', False),
            enable_dry_run_mode=get_bool_env('DRY_RUN_MODE', False),
            enable_verbose_logging=get_bool_env('VERBOSE_LOGGING', False),
        )
    
    @classmethod
    def from_file(cls, config_path: str) -> 'OrchestratorFeatureFlags':
        """
        Load feature flags from a JSON configuration file.
        
        Args:
            config_path: Path to JSON configuration file
            
        Returns:
            OrchestratorFeatureFlags instance
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file is invalid
        """
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Feature flags config not found: {config_path}")
        
        try:
            with open(config_file, 'r') as f:
                data = json.load(f)
            
            # Create instance with defaults, then update with file values
            flags = cls()
            
            # Update only fields that exist in the dataclass
            valid_fields = set(field.name for field in flags.__dataclass_fields__.values())
            for key, value in data.items():
                if key in valid_fields and isinstance(value, bool):
                    setattr(flags, key, value)
                elif key in valid_fields:
                    logger.warning(f"Invalid value type for {key}: {type(value)}, expected bool")
                else:
                    logger.warning(f"Unknown feature flag: {key}")
            
            return flags
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in feature flags config: {e}")
        except Exception as e:
            raise ValueError(f"Failed to load feature flags config: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert feature flags to dictionary."""
        return asdict(self)
    
    def to_json(self, indent: int = 2) -> str:
        """Convert feature flags to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def save_to_file(self, config_path: str) -> None:
        """
        Save feature flags to a JSON configuration file.
        
        Args:
            config_path: Path to save configuration file
        """
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_file, 'w') as f:
            f.write(self.to_json())
        
        logger.info(f"Feature flags saved to: {config_path}")
    
    def is_feature_enabled(self, feature_name: str) -> bool:
        """
        Check if a specific feature is enabled.
        
        Args:
            feature_name: Name of the feature flag
            
        Returns:
            True if feature is enabled, False otherwise
        """
        return getattr(self, feature_name, False)
    
    def enable_feature(self, feature_name: str) -> None:
        """
        Enable a specific feature.
        
        Args:
            feature_name: Name of the feature flag to enable
        """
        if hasattr(self, feature_name):
            setattr(self, feature_name, True)
            logger.info(f"Enabled feature: {feature_name}")
        else:
            logger.warning(f"Unknown feature flag: {feature_name}")
    
    def disable_feature(self, feature_name: str) -> None:
        """
        Disable a specific feature.
        
        Args:
            feature_name: Name of the feature flag to disable
        """
        if hasattr(self, feature_name):
            setattr(self, feature_name, False)
            logger.info(f"Disabled feature: {feature_name}")
        else:
            logger.warning(f"Unknown feature flag: {feature_name}")
    
    def get_rollout_stage(self) -> str:
        """
        Determine the current rollout stage based on enabled features.
        
        Returns:
            String indicating rollout stage: "disabled", "development", "staging", "production"
        """
        if not self.enable_orchestrator:
            return "disabled"
        
        # Development stage: basic features enabled
        if (self.enable_orchestrator and 
            self.enable_manifest_validation and 
            self.enable_legacy_fallback):
            
            # Staging stage: additional safety features
            if (self.enable_integrity_checks and 
                self.enable_disk_space_checks and
                self.enable_health_endpoints):
                
                # Production stage: full feature set
                if (self.enable_concurrent_downloads and
                    self.enable_metrics and
                    self.enable_performance_tracking):
                    return "production"
                else:
                    return "staging"
            else:
                return "development"
        
        return "disabled"
    
    def validate_configuration(self) -> Dict[str, str]:
        """
        Validate feature flag configuration for consistency.
        
        Returns:
            Dictionary of validation warnings/errors
        """
        issues = {}
        
        # Check for conflicting configurations
        if self.enable_orchestrator and not self.enable_local_backend and not self.enable_hf_backend and not self.enable_s3_backend:
            issues["no_backends"] = "Orchestrator enabled but no storage backends are enabled"
        
        if self.enable_automatic_download and not self.enable_orchestrator:
            issues["auto_download_without_orchestrator"] = "Automatic download enabled but orchestrator is disabled"
        
        if self.enable_concurrent_downloads and not self.enable_orchestrator:
            issues["concurrent_without_orchestrator"] = "Concurrent downloads enabled but orchestrator is disabled"
        
        if self.enable_metrics and not self.enable_orchestrator:
            issues["metrics_without_orchestrator"] = "Metrics enabled but orchestrator is disabled"
        
        if self.strict_validation and not self.enable_manifest_validation:
            issues["strict_without_validation"] = "Strict validation enabled but manifest validation is disabled"
        
        if self.enable_debug_mode and not self.enable_verbose_logging:
            issues["debug_without_verbose"] = "Debug mode enabled but verbose logging is disabled (recommended)"
        
        # Check for security concerns
        if self.enable_orchestrator and not self.enable_secure_downloads:
            issues["insecure_downloads"] = "Orchestrator enabled but secure downloads are disabled (security risk)"
        
        if self.enable_s3_backend and not self.enable_credential_encryption:
            issues["unencrypted_credentials"] = "S3 backend enabled but credential encryption is disabled (security risk)"
        
        return issues


class FeatureFlagManager:
    """Manager for feature flag operations and rollout control."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize feature flag manager.
        
        Args:
            config_path: Optional path to feature flags configuration file
        """
        self.config_path = config_path
        self._flags: Optional[OrchestratorFeatureFlags] = None
        self.logger = logging.getLogger(__name__ + ".FeatureFlagManager")
    
    @property
    def flags(self) -> OrchestratorFeatureFlags:
        """Get current feature flags, loading if necessary."""
        if self._flags is None:
            self._flags = self.load_flags()
        return self._flags
    
    def load_flags(self) -> OrchestratorFeatureFlags:
        """
        Load feature flags from configuration.
        
        Returns:
            OrchestratorFeatureFlags instance
        """
        # Try to load from file first, then fall back to environment
        if self.config_path and Path(self.config_path).exists():
            try:
                flags = OrchestratorFeatureFlags.from_file(self.config_path)
                self.logger.info(f"Loaded feature flags from file: {self.config_path}")
                return flags
            except Exception as e:
                self.logger.warning(f"Failed to load feature flags from file: {e}")
        
        # Fall back to environment variables
        flags = OrchestratorFeatureFlags.from_env()
        self.logger.info("Loaded feature flags from environment variables")
        return flags
    
    def reload_flags(self) -> None:
        """Reload feature flags from configuration."""
        self._flags = None
        self.logger.info("Feature flags reloaded")
    
    def save_flags(self) -> None:
        """Save current feature flags to file if config path is set."""
        if self.config_path and self._flags:
            self._flags.save_to_file(self.config_path)
    
    def is_enabled(self, feature_name: str) -> bool:
        """Check if a feature is enabled."""
        return self.flags.is_feature_enabled(feature_name)
    
    def enable_rollout_stage(self, stage: str) -> None:
        """
        Enable features for a specific rollout stage.
        
        Args:
            stage: Rollout stage ("development", "staging", "production")
        """
        flags = self.flags
        
        if stage == "development":
            flags.enable_orchestrator = True
            flags.enable_manifest_validation = True
            flags.enable_legacy_fallback = True
            flags.enable_local_backend = True
            flags.enable_hf_backend = True
            
        elif stage == "staging":
            # Enable development features
            self.enable_rollout_stage("development")
            
            # Add staging features
            flags.enable_integrity_checks = True
            flags.enable_disk_space_checks = True
            flags.enable_health_endpoints = True
            flags.enable_structured_logging = True
            
        elif stage == "production":
            # Enable staging features
            self.enable_rollout_stage("staging")
            
            # Add production features
            flags.enable_concurrent_downloads = True
            flags.enable_metrics = True
            flags.enable_performance_tracking = True
            flags.enable_s3_backend = True
            flags.enable_component_deduplication = True
            flags.enable_secure_downloads = True
            
        else:
            raise ValueError(f"Unknown rollout stage: {stage}")
        
        self.logger.info(f"Enabled rollout stage: {stage}")
        
        # Save configuration if path is set
        if self.config_path:
            self.save_flags()
    
    def validate_current_configuration(self) -> bool:
        """
        Validate current feature flag configuration.
        
        Returns:
            True if configuration is valid, False if there are issues
        """
        issues = self.flags.validate_configuration()
        
        if issues:
            self.logger.warning("Feature flag configuration issues found:")
            for issue_key, issue_msg in issues.items():
                self.logger.warning(f"  {issue_key}: {issue_msg}")
            return False
        
        self.logger.info("Feature flag configuration is valid")
        return True
    
    def get_status_report(self) -> Dict[str, Any]:
        """
        Get comprehensive status report of feature flags.
        
        Returns:
            Dictionary with status information
        """
        flags = self.flags
        issues = flags.validate_configuration()
        
        return {
            "rollout_stage": flags.get_rollout_stage(),
            "orchestrator_enabled": flags.enable_orchestrator,
            "total_features_enabled": sum(1 for value in flags.to_dict().values() if value),
            "total_features": len(flags.to_dict()),
            "configuration_issues": len(issues),
            "issues": issues,
            "flags": flags.to_dict()
        }


# Global feature flag manager instance
_global_manager: Optional[FeatureFlagManager] = None


def get_feature_flags() -> OrchestratorFeatureFlags:
    """Get global feature flags instance."""
    global _global_manager
    
    if _global_manager is None:
        # Try to find config file in standard locations
        config_paths = [
            os.getenv('WAN_FEATURE_FLAGS_CONFIG'),
            '.kiro/feature_flags.json',
            'config/feature_flags.json',
            'feature_flags.json'
        ]
        
        config_path = None
        for path in config_paths:
            if path and Path(path).exists():
                config_path = path
                break
        
        _global_manager = FeatureFlagManager(config_path)
    
    return _global_manager.flags


def is_feature_enabled(feature_name: str) -> bool:
    """Check if a feature is enabled globally."""
    return get_feature_flags().is_feature_enabled(feature_name)


def require_feature(feature_name: str) -> None:
    """
    Require a feature to be enabled, raising an exception if not.
    
    Args:
        feature_name: Name of required feature
        
    Raises:
        RuntimeError: If feature is not enabled
    """
    if not is_feature_enabled(feature_name):
        raise RuntimeError(f"Required feature not enabled: {feature_name}")