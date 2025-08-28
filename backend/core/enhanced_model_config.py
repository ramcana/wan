"""
Enhanced Model Configuration Management System

This module provides comprehensive configuration management for enhanced model availability features,
including user preferences, admin controls, feature flags, and runtime configuration updates.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List, Union
from enum import Enum
from pathlib import Path
import asyncio
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class AutomationLevel(Enum):
    """Automation levels for model management features"""
    MANUAL = "manual"
    SEMI_AUTOMATIC = "semi_automatic"
    FULLY_AUTOMATIC = "fully_automatic"


class FeatureFlag(Enum):
    """Feature flags for gradual rollout"""
    ENHANCED_DOWNLOADS = "enhanced_downloads"
    HEALTH_MONITORING = "health_monitoring"
    INTELLIGENT_FALLBACK = "intelligent_fallback"
    USAGE_ANALYTICS = "usage_analytics"
    AUTO_UPDATES = "auto_updates"
    REAL_TIME_NOTIFICATIONS = "real_time_notifications"


@dataclass
class DownloadConfig:
    """Configuration for enhanced download features"""
    max_retries: int = 3
    retry_delay_base: float = 2.0  # Base delay for exponential backoff
    max_retry_delay: float = 300.0  # Maximum delay between retries
    bandwidth_limit_mbps: Optional[float] = None
    max_concurrent_downloads: int = 2
    enable_resume: bool = True
    enable_pause: bool = True
    chunk_size_mb: float = 10.0
    verify_integrity: bool = True
    cleanup_failed_downloads: bool = True


@dataclass
class HealthMonitoringConfig:
    """Configuration for model health monitoring"""
    enabled: bool = True
    check_interval_hours: int = 24
    integrity_check_enabled: bool = True
    performance_monitoring_enabled: bool = True
    auto_repair_enabled: bool = False  # Requires user confirmation by default
    corruption_threshold: float = 0.1  # Percentage of files that can be corrupted
    performance_degradation_threshold: float = 0.3  # 30% performance drop triggers alert
    cleanup_corrupted_models: bool = False


@dataclass
class FallbackConfig:
    """Configuration for intelligent fallback system"""
    enabled: bool = True
    suggest_alternatives: bool = True
    auto_queue_requests: bool = False  # Requires user confirmation by default
    compatibility_threshold: float = 0.7  # Minimum compatibility score for suggestions
    max_wait_time_minutes: int = 60
    prefer_local_models: bool = True
    fallback_to_mock_enabled: bool = True


@dataclass
class AnalyticsConfig:
    """Configuration for usage analytics"""
    enabled: bool = True
    collect_usage_stats: bool = True
    collect_performance_metrics: bool = True
    retention_days: int = 90
    anonymize_data: bool = True
    share_analytics: bool = False  # Share with developers for improvements
    enable_recommendations: bool = True
    auto_cleanup_suggestions: bool = False


@dataclass
class UpdateConfig:
    """Configuration for model update management"""
    enabled: bool = True
    auto_check_updates: bool = True
    check_interval_hours: int = 168  # Weekly
    auto_download_updates: bool = False  # Requires user confirmation
    auto_install_updates: bool = False  # Requires user confirmation
    backup_before_update: bool = True
    rollback_on_failure: bool = True
    update_during_low_usage: bool = True


@dataclass
class NotificationConfig:
    """Configuration for real-time notifications"""
    enabled: bool = True
    download_progress: bool = True
    health_alerts: bool = True
    update_notifications: bool = True
    fallback_notifications: bool = True
    analytics_reports: bool = False
    email_notifications: bool = False
    webhook_url: Optional[str] = None


@dataclass
class StorageConfig:
    """Configuration for storage management"""
    max_storage_gb: Optional[float] = None
    cleanup_threshold_percent: float = 85.0  # Trigger cleanup at 85% full
    min_free_space_gb: float = 5.0
    auto_cleanup_enabled: bool = False
    preserve_recent_models_days: int = 7
    preserve_frequently_used: bool = True


@dataclass
class UserPreferences:
    """User-specific preferences for model management"""
    automation_level: AutomationLevel = AutomationLevel.SEMI_AUTOMATIC
    download_config: DownloadConfig = field(default_factory=DownloadConfig)
    health_monitoring: HealthMonitoringConfig = field(default_factory=HealthMonitoringConfig)
    fallback_config: FallbackConfig = field(default_factory=FallbackConfig)
    analytics_config: AnalyticsConfig = field(default_factory=AnalyticsConfig)
    update_config: UpdateConfig = field(default_factory=UpdateConfig)
    notification_config: NotificationConfig = field(default_factory=NotificationConfig)
    storage_config: StorageConfig = field(default_factory=StorageConfig)
    preferred_models: List[str] = field(default_factory=list)
    blocked_models: List[str] = field(default_factory=list)


@dataclass
class AdminPolicies:
    """System-wide administrative policies"""
    enforce_storage_limits: bool = True
    max_user_storage_gb: Optional[float] = None
    allow_external_downloads: bool = True
    require_approval_for_updates: bool = False
    max_concurrent_users: Optional[int] = None
    bandwidth_limit_per_user_mbps: Optional[float] = None
    allowed_model_sources: List[str] = field(default_factory=lambda: ["huggingface", "local"])
    blocked_model_patterns: List[str] = field(default_factory=list)
    security_scan_required: bool = False
    audit_logging_enabled: bool = True


@dataclass
class FeatureFlagConfig:
    """Feature flag configuration for gradual rollout"""
    flags: Dict[str, bool] = field(default_factory=lambda: {
        FeatureFlag.ENHANCED_DOWNLOADS.value: True,
        FeatureFlag.HEALTH_MONITORING.value: True,
        FeatureFlag.INTELLIGENT_FALLBACK.value: True,
        FeatureFlag.USAGE_ANALYTICS.value: True,
        FeatureFlag.AUTO_UPDATES.value: False,
        FeatureFlag.REAL_TIME_NOTIFICATIONS.value: True,
    })
    rollout_percentage: Dict[str, float] = field(default_factory=dict)  # For A/B testing
    user_overrides: Dict[str, Dict[str, bool]] = field(default_factory=dict)  # Per-user overrides


@dataclass
class EnhancedModelConfiguration:
    """Complete configuration for enhanced model availability system"""
    version: str = "1.0.0"
    user_preferences: UserPreferences = field(default_factory=UserPreferences)
    admin_policies: AdminPolicies = field(default_factory=AdminPolicies)
    feature_flags: FeatureFlagConfig = field(default_factory=FeatureFlagConfig)
    last_updated: datetime = field(default_factory=datetime.now)
    config_schema_version: str = "1.0.0"


class ConfigurationManager:
    """Manages enhanced model configuration with validation and runtime updates"""
    
    def __init__(self, config_path: str = "config/enhanced_model_config.json"):
        self.config_path = Path(config_path)
        self.config: EnhancedModelConfiguration = EnhancedModelConfiguration()
        self._observers: List[callable] = []
        self._lock = asyncio.Lock()
        
        # Ensure config directory exists
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing configuration
        self.load_configuration()
    
    def add_observer(self, callback: callable) -> None:
        """Add observer for configuration changes"""
        self._observers.append(callback)
    
    def remove_observer(self, callback: callable) -> None:
        """Remove configuration change observer"""
        if callback in self._observers:
            self._observers.remove(callback)
    
    async def _notify_observers(self, change_type: str, section: str, old_value: Any, new_value: Any) -> None:
        """Notify observers of configuration changes"""
        for observer in self._observers:
            try:
                if asyncio.iscoroutinefunction(observer):
                    await observer(change_type, section, old_value, new_value)
                else:
                    observer(change_type, section, old_value, new_value)
            except Exception as e:
                logger.error(f"Error notifying configuration observer: {e}")
    
    def load_configuration(self) -> bool:
        """Load configuration from file"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config_data = json.load(f)
                
                # Handle version migration if needed
                if self._needs_migration(config_data):
                    config_data = self._migrate_configuration(config_data)
                
                self.config = self._deserialize_config(config_data)
                logger.info(f"Configuration loaded from {self.config_path}")
                return True
            else:
                # Create default configuration
                self.save_configuration()
                logger.info(f"Created default configuration at {self.config_path}")
                return True
                
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            # Use default configuration on error
            self.config = EnhancedModelConfiguration()
            return False
    
    def save_configuration(self) -> bool:
        """Save configuration to file"""
        try:
            self.config.last_updated = datetime.now()
            config_data = self._serialize_config(self.config)
            
            # Create backup before saving
            self._create_backup()
            
            with open(self.config_path, 'w') as f:
                json.dump(config_data, f, indent=2, default=str)
            
            logger.info(f"Configuration saved to {self.config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False
    
    async def update_user_preferences(self, preferences: UserPreferences) -> bool:
        """Update user preferences with validation and notification"""
        async with self._lock:
            old_preferences = self.config.user_preferences
            
            # Validate preferences
            validation_result = self.validate_user_preferences(preferences)
            if not validation_result.is_valid:
                logger.error(f"Invalid user preferences: {validation_result.errors}")
                return False
            
            # Apply admin policy constraints
            constrained_preferences = self._apply_admin_constraints(preferences)
            
            self.config.user_preferences = constrained_preferences
            
            # Save and notify
            if self.save_configuration():
                await self._notify_observers("user_preferences", "preferences", old_preferences, constrained_preferences)
                return True
            
            return False
    
    async def update_admin_policies(self, policies: AdminPolicies) -> bool:
        """Update admin policies with validation and notification"""
        async with self._lock:
            old_policies = self.config.admin_policies
            
            # Validate policies
            validation_result = self.validate_admin_policies(policies)
            if not validation_result.is_valid:
                logger.error(f"Invalid admin policies: {validation_result.errors}")
                return False
            
            self.config.admin_policies = policies
            
            # Save and notify
            if self.save_configuration():
                await self._notify_observers("admin_policies", "policies", old_policies, policies)
                return True
            
            return False
    
    async def update_feature_flag(self, flag: Union[str, FeatureFlag], enabled: bool, user_id: Optional[str] = None) -> bool:
        """Update feature flag with optional user-specific override"""
        async with self._lock:
            flag_name = flag.value if isinstance(flag, FeatureFlag) else flag
            old_flags = self.config.feature_flags.flags.copy()
            
            if user_id:
                # User-specific override
                if user_id not in self.config.feature_flags.user_overrides:
                    self.config.feature_flags.user_overrides[user_id] = {}
                self.config.feature_flags.user_overrides[user_id][flag_name] = enabled
            else:
                # Global flag update
                self.config.feature_flags.flags[flag_name] = enabled
            
            # Save and notify
            if self.save_configuration():
                await self._notify_observers("feature_flags", flag_name, old_flags.get(flag_name), enabled)
                return True
            
            return False
    
    def is_feature_enabled(self, flag: Union[str, FeatureFlag], user_id: Optional[str] = None) -> bool:
        """Check if a feature flag is enabled for a user or globally"""
        flag_name = flag.value if isinstance(flag, FeatureFlag) else flag
        
        # Check user-specific override first
        if user_id and user_id in self.config.feature_flags.user_overrides:
            user_flags = self.config.feature_flags.user_overrides[user_id]
            if flag_name in user_flags:
                return user_flags[flag_name]
        
        # Check rollout percentage for A/B testing
        if flag_name in self.config.feature_flags.rollout_percentage:
            rollout_pct = self.config.feature_flags.rollout_percentage[flag_name]
            if user_id:
                # Use consistent hash for user to ensure stable A/B assignment
                user_hash = hash(user_id) % 100
                if user_hash >= rollout_pct:
                    return False
        
        # Return global flag value
        return self.config.feature_flags.flags.get(flag_name, False)
    
    def get_user_preferences(self) -> UserPreferences:
        """Get current user preferences"""
        return self.config.user_preferences
    
    def get_admin_policies(self) -> AdminPolicies:
        """Get current admin policies"""
        return self.config.admin_policies
    
    def validate_user_preferences(self, preferences: UserPreferences) -> 'ValidationResult':
        """Validate user preferences"""
        from .config_validation import ConfigurationValidator
        validator = ConfigurationValidator()
        return validator.validate_user_preferences(preferences)
    
    def validate_admin_policies(self, policies: AdminPolicies) -> 'ValidationResult':
        """Validate admin policies"""
        from .config_validation import ConfigurationValidator
        validator = ConfigurationValidator()
        return validator.validate_admin_policies(policies)
    
    def _apply_admin_constraints(self, preferences: UserPreferences) -> UserPreferences:
        """Apply admin policy constraints to user preferences"""
        constrained = preferences
        policies = self.config.admin_policies
        
        # Apply storage constraints
        if policies.max_user_storage_gb is not None:
            if constrained.storage_config.max_storage_gb is None or \
               constrained.storage_config.max_storage_gb > policies.max_user_storage_gb:
                constrained.storage_config.max_storage_gb = policies.max_user_storage_gb
        
        # Apply bandwidth constraints
        if policies.bandwidth_limit_per_user_mbps is not None:
            if constrained.download_config.bandwidth_limit_mbps is None or \
               constrained.download_config.bandwidth_limit_mbps > policies.bandwidth_limit_per_user_mbps:
                constrained.download_config.bandwidth_limit_mbps = policies.bandwidth_limit_per_user_mbps
        
        # Apply security constraints
        if policies.security_scan_required:
            constrained.download_config.verify_integrity = True
        
        # Filter blocked models
        for pattern in policies.blocked_model_patterns:
            import re
            regex = re.compile(pattern)
            constrained.preferred_models = [
                model for model in constrained.preferred_models 
                if not regex.match(model)
            ]
        
        return constrained
    
    def _needs_migration(self, config_data: Dict[str, Any]) -> bool:
        """Check if configuration needs migration"""
        current_version = config_data.get('config_schema_version', '0.0.0')
        return current_version != self.config.config_schema_version
    
    def _migrate_configuration(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate configuration to current schema version"""
        current_version = config_data.get('config_schema_version', '0.0.0')
        
        # Migration logic for different versions
        if current_version == '0.0.0':
            # Initial migration from unversioned config
            config_data = self._migrate_from_unversioned(config_data)
            current_version = '1.0.0'
        
        config_data['config_schema_version'] = self.config.config_schema_version
        return config_data
    
    def _migrate_from_unversioned(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate from unversioned configuration"""
        # Add default values for new fields
        if 'feature_flags' not in config_data:
            config_data['feature_flags'] = {
                'flags': {flag.value: True for flag in FeatureFlag},
                'rollout_percentage': {},
                'user_overrides': {}
            }
        
        if 'admin_policies' not in config_data:
            config_data['admin_policies'] = asdict(AdminPolicies())
        
        return config_data
    
    def _serialize_config(self, config: EnhancedModelConfiguration) -> Dict[str, Any]:
        """Serialize configuration to dictionary"""
        return asdict(config)
    
    def _deserialize_config(self, config_data: Dict[str, Any]) -> EnhancedModelConfiguration:
        """Deserialize configuration from dictionary"""
        # Convert datetime strings back to datetime objects
        if 'last_updated' in config_data and isinstance(config_data['last_updated'], str):
            from datetime import datetime
            config_data['last_updated'] = datetime.fromisoformat(config_data['last_updated'])
        
        # Convert enum strings back to enums
        if 'user_preferences' in config_data and 'automation_level' in config_data['user_preferences']:
            automation_level = config_data['user_preferences']['automation_level']
            if isinstance(automation_level, str):
                config_data['user_preferences']['automation_level'] = AutomationLevel(automation_level)
        
        # Reconstruct nested dataclasses
        return self._dict_to_dataclass(config_data, EnhancedModelConfiguration)
    
    def _dict_to_dataclass(self, data: Dict[str, Any], dataclass_type):
        """Convert dictionary to dataclass instance"""
        import inspect
        
        # Get the dataclass fields
        fields = {f.name: f.type for f in dataclass_type.__dataclass_fields__.values()}
        kwargs = {}
        
        for field_name, field_type in fields.items():
            if field_name in data:
                value = data[field_name]
                
                # Handle nested dataclasses
                if hasattr(field_type, '__dataclass_fields__'):
                    if isinstance(value, dict):
                        kwargs[field_name] = self._dict_to_dataclass(value, field_type)
                    else:
                        kwargs[field_name] = value
                else:
                    kwargs[field_name] = value
        
        return dataclass_type(**kwargs)
    
    def _create_backup(self) -> None:
        """Create backup of current configuration"""
        if self.config_path.exists():
            backup_path = self.config_path.with_suffix(f'.backup.{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
            import shutil
            shutil.copy2(self.config_path, backup_path)
            
            # Keep only last 5 backups
            backup_dir = self.config_path.parent
            backups = sorted(backup_dir.glob(f'{self.config_path.stem}.backup.*.json'))
            if len(backups) > 5:
                for old_backup in backups[:-5]:
                    old_backup.unlink()


# Global configuration manager instance
_config_manager: Optional[ConfigurationManager] = None


def get_config_manager() -> ConfigurationManager:
    """Get global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigurationManager()
    return _config_manager


def reset_config_manager() -> None:
    """Reset global configuration manager (for testing)"""
    global _config_manager
    _config_manager = None