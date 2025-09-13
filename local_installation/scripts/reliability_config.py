"""
Reliability Configuration System

This module provides comprehensive configuration management for the installation
reliability system, including retry limits, timeout values, recovery strategies,
and feature flags for gradual rollout.

Requirements addressed:
- 1.4: User configurable retry limits and skip options
- 8.1: Health report generation and metrics collection
- 8.5: Cross-instance monitoring and centralized dashboard
"""

import json
import os
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum


class ReliabilityLevel(Enum):
    """Reliability enhancement levels for gradual rollout."""
    DISABLED = "disabled"
    BASIC = "basic"
    STANDARD = "standard"
    AGGRESSIVE = "aggressive"
    MAXIMUM = "maximum"


class RecoveryStrategy(Enum):
    """Available recovery strategies."""
    RETRY_ONLY = "retry_only"
    ALTERNATIVE_SOURCES = "alternative_sources"
    FALLBACK_METHODS = "fallback_methods"
    FULL_RECOVERY = "full_recovery"


@dataclass
class RetryConfiguration:
    """Configuration for retry mechanisms."""
    max_retries: int = 3
    base_delay_seconds: float = 2.0
    max_delay_seconds: float = 60.0
    exponential_base: float = 2.0
    jitter_enabled: bool = True
    user_prompt_after_first_failure: bool = True
    allow_skip_retries: bool = True


@dataclass
class TimeoutConfiguration:
    """Configuration for operation timeouts."""
    model_download_seconds: int = 1800  # 30 minutes
    dependency_install_seconds: int = 600  # 10 minutes
    system_detection_seconds: int = 60  # 1 minute
    validation_seconds: int = 300  # 5 minutes
    network_test_seconds: int = 30  # 30 seconds
    cleanup_seconds: int = 120  # 2 minutes
    context_multipliers: Dict[str, float] = None

    def __post_init__(self):
        if self.context_multipliers is None:
            self.context_multipliers = {
                "large_file": 2.0,
                "slow_network": 1.5,
                "retry_attempt": 1.2,
                "low_memory": 1.3
            }


@dataclass
class RecoveryConfiguration:
    """Configuration for recovery strategies."""
    missing_method_recovery: bool = True
    model_validation_recovery: bool = True
    network_failure_recovery: bool = True
    dependency_recovery: bool = True
    automatic_cleanup: bool = True
    fallback_implementations: bool = True
    alternative_sources: List[str] = None

    def __post_init__(self):
        if self.alternative_sources is None:
            self.alternative_sources = [
                "https://huggingface.co",
                "https://hf-mirror.com",
                "https://mirror.huggingface.co"
            ]


@dataclass
class MonitoringConfiguration:
    """Configuration for monitoring and alerting."""
    enable_health_monitoring: bool = True
    enable_performance_tracking: bool = True
    enable_predictive_analysis: bool = False
    health_check_interval_seconds: int = 60
    performance_sample_rate: float = 0.1
    alert_thresholds: Dict[str, float] = None
    export_metrics: bool = True
    metrics_export_format: str = "json"

    def __post_init__(self):
        if self.alert_thresholds is None:
            self.alert_thresholds = {
                "error_rate": 0.1,
                "response_time_ms": 5000,
                "memory_usage_percent": 90,
                "disk_usage_percent": 95
            }


@dataclass
class FeatureFlags:
    """Feature flags for gradual rollout of reliability enhancements."""
    reliability_level: ReliabilityLevel = ReliabilityLevel.STANDARD
    enable_enhanced_error_context: bool = True
    enable_missing_method_recovery: bool = True
    enable_model_validation_recovery: bool = True
    enable_network_failure_recovery: bool = True
    enable_dependency_recovery: bool = True
    enable_pre_installation_validation: bool = True
    enable_diagnostic_monitoring: bool = True
    enable_health_reporting: bool = True
    enable_timeout_management: bool = True
    enable_user_guidance_enhancements: bool = True


@dataclass
class DeploymentConfiguration:
    """Configuration for deployment and production settings."""
    environment: str = "development"
    log_level: str = "INFO"
    enable_debug_mode: bool = False
    enable_telemetry: bool = True
    telemetry_endpoint: Optional[str] = None
    support_contact: str = "support@wan22.com"
    documentation_url: str = "https://docs.wan22.com"
    enable_automatic_updates: bool = False
    update_check_interval_hours: int = 24


@dataclass
class ReliabilityConfiguration:
    """Complete reliability system configuration."""
    retry: RetryConfiguration = None
    timeouts: TimeoutConfiguration = None
    recovery: RecoveryConfiguration = None
    monitoring: MonitoringConfiguration = None
    features: FeatureFlags = None
    deployment: DeploymentConfiguration = None
    version: str = "1.0.0"
    last_updated: Optional[str] = None

    def __post_init__(self):
        if self.retry is None:
            self.retry = RetryConfiguration()
        if self.timeouts is None:
            self.timeouts = TimeoutConfiguration()
        if self.recovery is None:
            self.recovery = RecoveryConfiguration()
        if self.monitoring is None:
            self.monitoring = MonitoringConfiguration()
        if self.features is None:
            self.features = FeatureFlags()
        if self.deployment is None:
            self.deployment = DeploymentConfiguration()


class ReliabilityConfigManager:
    """Manages reliability configuration loading, validation, and updates."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager."""
        self.config_path = config_path or self._get_default_config_path()
        self.config: Optional[ReliabilityConfiguration] = None
        self.logger = logging.getLogger(__name__)

    def _get_default_config_path(self) -> str:
        """Get default configuration file path."""
        script_dir = Path(__file__).parent
        return str(script_dir / "reliability_config.json")

    def load_config(self) -> ReliabilityConfiguration:
        """Load configuration from file or create default."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config_data = json.load(f)
                self.config = self._dict_to_config(config_data)
                self.logger.info(f"Loaded reliability configuration from {self.config_path}")
            else:
                self.config = ReliabilityConfiguration()
                self.save_config()
                self.logger.info(f"Created default reliability configuration at {self.config_path}")
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            self.config = ReliabilityConfiguration()

        return self.config

    def save_config(self) -> bool:
        """Save current configuration to file."""
        try:
            if self.config is None:
                self.config = ReliabilityConfiguration()

            # Update timestamp
            from datetime import datetime
            self.config.last_updated = datetime.now().isoformat()

            config_data = self._config_to_dict(self.config)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            with open(self.config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            self.logger.info(f"Saved reliability configuration to {self.config_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            return False

    def _dict_to_config(self, data: Dict[str, Any]) -> ReliabilityConfiguration:
        """Convert dictionary to configuration object."""
        # Handle enum conversions
        if 'features' in data and 'reliability_level' in data['features']:
            data['features']['reliability_level'] = ReliabilityLevel(
                data['features']['reliability_level']
            )

        # Create configuration with nested dataclasses
        config = ReliabilityConfiguration()
        
        if 'retry' in data:
            config.retry = RetryConfiguration(**data['retry'])
        if 'timeouts' in data:
            config.timeouts = TimeoutConfiguration(**data['timeouts'])
        if 'recovery' in data:
            config.recovery = RecoveryConfiguration(**data['recovery'])
        if 'monitoring' in data:
            config.monitoring = MonitoringConfiguration(**data['monitoring'])
        if 'features' in data:
            config.features = FeatureFlags(**data['features'])
        if 'deployment' in data:
            config.deployment = DeploymentConfiguration(**data['deployment'])
        
        config.version = data.get('version', '1.0.0')
        config.last_updated = data.get('last_updated')

        return config

    def _config_to_dict(self, config: ReliabilityConfiguration) -> Dict[str, Any]:
        """Convert configuration object to dictionary."""
        data = asdict(config)
        
        # Handle enum serialization
        if 'features' in data and 'reliability_level' in data['features']:
            data['features']['reliability_level'] = data['features']['reliability_level'].value

        return data

    def update_config(self, updates: Dict[str, Any]) -> bool:
        """Update configuration with new values."""
        try:
            if self.config is None:
                self.load_config()

            # Apply updates to nested structures
            for key, value in updates.items():
                if hasattr(self.config, key):
                    if isinstance(value, dict):
                        # Update nested configuration
                        nested_config = getattr(self.config, key)
                        for nested_key, nested_value in value.items():
                            if hasattr(nested_config, nested_key):
                                setattr(nested_config, nested_key, nested_value)
                    else:
                        setattr(self.config, key, value)

            return self.save_config()
        except Exception as e:
            self.logger.error(f"Failed to update configuration: {e}")
            return False

    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        if self.config is None:
            issues.append("Configuration not loaded")
            return issues

        # Validate retry configuration
        if self.config.retry.max_retries < 0:
            issues.append("max_retries must be non-negative")
        if self.config.retry.base_delay_seconds <= 0:
            issues.append("base_delay_seconds must be positive")
        if self.config.retry.max_delay_seconds < self.config.retry.base_delay_seconds:
            issues.append("max_delay_seconds must be >= base_delay_seconds")

        # Validate timeout configuration
        for timeout_name in ['model_download_seconds', 'dependency_install_seconds', 
                           'system_detection_seconds', 'validation_seconds', 
                           'network_test_seconds', 'cleanup_seconds']:
            timeout_value = getattr(self.config.timeouts, timeout_name)
            if timeout_value <= 0:
                issues.append(f"{timeout_name} must be positive")

        # Validate monitoring configuration
        if not 0 <= self.config.monitoring.performance_sample_rate <= 1:
            issues.append("performance_sample_rate must be between 0 and 1")

        # Validate alert thresholds
        for threshold_name, threshold_value in self.config.monitoring.alert_thresholds.items():
            if threshold_value < 0:
                issues.append(f"alert_threshold {threshold_name} must be non-negative")

        return issues

    def get_config_for_environment(self, environment: str) -> ReliabilityConfiguration:
        """Get configuration optimized for specific environment."""
        if self.config is None:
            self.load_config()

        # Create environment-specific configuration
        env_config = ReliabilityConfiguration(
            retry=RetryConfiguration(**asdict(self.config.retry)),
            timeouts=TimeoutConfiguration(**asdict(self.config.timeouts)),
            recovery=RecoveryConfiguration(**asdict(self.config.recovery)),
            monitoring=MonitoringConfiguration(**asdict(self.config.monitoring)),
            features=FeatureFlags(**asdict(self.config.features)),
            deployment=DeploymentConfiguration(**asdict(self.config.deployment))
        )

        # Apply environment-specific optimizations
        if environment == "production":
            env_config.deployment.log_level = "WARNING"
            env_config.deployment.enable_debug_mode = False
            env_config.monitoring.enable_performance_tracking = True
            env_config.features.reliability_level = ReliabilityLevel.MAXIMUM
        elif environment == "development":
            env_config.deployment.log_level = "DEBUG"
            env_config.deployment.enable_debug_mode = True
            env_config.monitoring.enable_performance_tracking = False
            env_config.features.reliability_level = ReliabilityLevel.STANDARD
        elif environment == "testing":
            env_config.deployment.log_level = "INFO"
            env_config.retry.max_retries = 1  # Faster testing
            env_config.timeouts.model_download_seconds = 300  # Shorter timeouts
            env_config.features.reliability_level = ReliabilityLevel.BASIC

        return env_config

    def export_config_template(self, output_path: str) -> bool:
        """Export configuration template with documentation."""
        try:
            template = {
                "_description": "WAN2.2 Installation Reliability System Configuration",
                "_version": "1.0.0",
                "_documentation": "https://docs.wan22.com/reliability-config",
                
                "retry": {
                    "_description": "Retry mechanism configuration",
                    "max_retries": 3,
                    "base_delay_seconds": 2.0,
                    "max_delay_seconds": 60.0,
                    "exponential_base": 2.0,
                    "jitter_enabled": True,
                    "user_prompt_after_first_failure": True,
                    "allow_skip_retries": True
                },
                
                "timeouts": {
                    "_description": "Operation timeout configuration",
                    "model_download_seconds": 1800,
                    "dependency_install_seconds": 600,
                    "system_detection_seconds": 60,
                    "validation_seconds": 300,
                    "network_test_seconds": 30,
                    "cleanup_seconds": 120,
                    "context_multipliers": {
                        "large_file": 2.0,
                        "slow_network": 1.5,
                        "retry_attempt": 1.2,
                        "low_memory": 1.3
                    }
                },
                
                "recovery": {
                    "_description": "Recovery strategy configuration",
                    "missing_method_recovery": True,
                    "model_validation_recovery": True,
                    "network_failure_recovery": True,
                    "dependency_recovery": True,
                    "automatic_cleanup": True,
                    "fallback_implementations": True,
                    "alternative_sources": [
                        "https://huggingface.co",
                        "https://hf-mirror.com",
                        "https://mirror.huggingface.co"
                    ]
                },
                
                "monitoring": {
                    "_description": "Monitoring and alerting configuration",
                    "enable_health_monitoring": True,
                    "enable_performance_tracking": True,
                    "enable_predictive_analysis": False,
                    "health_check_interval_seconds": 60,
                    "performance_sample_rate": 0.1,
                    "alert_thresholds": {
                        "error_rate": 0.1,
                        "response_time_ms": 5000,
                        "memory_usage_percent": 90,
                        "disk_usage_percent": 95
                    },
                    "export_metrics": True,
                    "metrics_export_format": "json"
                },
                
                "features": {
                    "_description": "Feature flags for gradual rollout",
                    "reliability_level": "standard",
                    "enable_enhanced_error_context": True,
                    "enable_missing_method_recovery": True,
                    "enable_model_validation_recovery": True,
                    "enable_network_failure_recovery": True,
                    "enable_dependency_recovery": True,
                    "enable_pre_installation_validation": True,
                    "enable_diagnostic_monitoring": True,
                    "enable_health_reporting": True,
                    "enable_timeout_management": True,
                    "enable_user_guidance_enhancements": True
                },
                
                "deployment": {
                    "_description": "Deployment and production settings",
                    "environment": "development",
                    "log_level": "INFO",
                    "enable_debug_mode": False,
                    "enable_telemetry": True,
                    "telemetry_endpoint": None,
                    "support_contact": "support@wan22.com",
                    "documentation_url": "https://docs.wan22.com",
                    "enable_automatic_updates": False,
                    "update_check_interval_hours": 24
                }
            }

            with open(output_path, 'w') as f:
                json.dump(template, f, indent=2)
            
            self.logger.info(f"Exported configuration template to {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to export configuration template: {e}")
            return False


# Global configuration manager instance
_config_manager = None

def get_config_manager() -> ReliabilityConfigManager:
    """Get global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ReliabilityConfigManager()
    return _config_manager

def get_reliability_config() -> ReliabilityConfiguration:
    """Get current reliability configuration."""
    return get_config_manager().load_config()
