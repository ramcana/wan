---
title: core.enhanced_model_config
category: api
tags: [api, core]
---

# core.enhanced_model_config

Enhanced Model Configuration Management System

This module provides comprehensive configuration management for enhanced model availability features,
including user preferences, admin controls, feature flags, and runtime configuration updates.

## Classes

### AutomationLevel

Automation levels for model management features

### FeatureFlag

Feature flags for gradual rollout

### DownloadConfig

Configuration for enhanced download features

### HealthMonitoringConfig

Configuration for model health monitoring

### FallbackConfig

Configuration for intelligent fallback system

### AnalyticsConfig

Configuration for usage analytics

### UpdateConfig

Configuration for model update management

### NotificationConfig

Configuration for real-time notifications

### StorageConfig

Configuration for storage management

### UserPreferences

User-specific preferences for model management

### AdminPolicies

System-wide administrative policies

### FeatureFlagConfig

Feature flag configuration for gradual rollout

### EnhancedModelConfiguration

Complete configuration for enhanced model availability system

### ConfigurationManager

Manages enhanced model configuration with validation and runtime updates

#### Methods

##### __init__(self: Any, config_path: str)



##### add_observer(self: Any, callback: callable) -> None

Add observer for configuration changes

##### remove_observer(self: Any, callback: callable) -> None

Remove configuration change observer

##### load_configuration(self: Any) -> bool

Load configuration from file

##### save_configuration(self: Any) -> bool

Save configuration to file

##### is_feature_enabled(self: Any, flag: <ast.Subscript object at 0x000001942CBC36D0>, user_id: <ast.Subscript object at 0x000001942CBC36A0>) -> bool

Check if a feature flag is enabled for a user or globally

##### get_user_preferences(self: Any) -> UserPreferences

Get current user preferences

##### get_admin_policies(self: Any) -> AdminPolicies

Get current admin policies

##### validate_user_preferences(self: Any, preferences: UserPreferences) -> ValidationResult

Validate user preferences

##### validate_admin_policies(self: Any, policies: AdminPolicies) -> ValidationResult

Validate admin policies

##### _apply_admin_constraints(self: Any, preferences: UserPreferences) -> UserPreferences

Apply admin policy constraints to user preferences

##### _needs_migration(self: Any, config_data: <ast.Subscript object at 0x000001942CBE3580>) -> bool

Check if configuration needs migration

##### _migrate_configuration(self: Any, config_data: <ast.Subscript object at 0x000001942CBE30D0>) -> <ast.Subscript object at 0x000001942CBE29E0>

Migrate configuration to current schema version

##### _migrate_from_unversioned(self: Any, config_data: <ast.Subscript object at 0x000001942CBE2920>) -> <ast.Subscript object at 0x000001942CBE2050>

Migrate from unversioned configuration

##### _serialize_config(self: Any, config: EnhancedModelConfiguration) -> <ast.Subscript object at 0x000001942CBE1510>

Serialize configuration to dictionary

##### _deserialize_config(self: Any, config_data: <ast.Subscript object at 0x000001942CBE1360>) -> EnhancedModelConfiguration

Deserialize configuration from dictionary

##### _dict_to_dataclass(self: Any, data: <ast.Subscript object at 0x000001942CB48520>, dataclass_type: Any)

Convert dictionary to dataclass instance

##### _create_backup(self: Any) -> None

Create backup of current configuration

## Constants

### MANUAL

Type: `str`

Value: `manual`

### SEMI_AUTOMATIC

Type: `str`

Value: `semi_automatic`

### FULLY_AUTOMATIC

Type: `str`

Value: `fully_automatic`

### ENHANCED_DOWNLOADS

Type: `str`

Value: `enhanced_downloads`

### HEALTH_MONITORING

Type: `str`

Value: `health_monitoring`

### INTELLIGENT_FALLBACK

Type: `str`

Value: `intelligent_fallback`

### USAGE_ANALYTICS

Type: `str`

Value: `usage_analytics`

### AUTO_UPDATES

Type: `str`

Value: `auto_updates`

### REAL_TIME_NOTIFICATIONS

Type: `str`

Value: `real_time_notifications`

