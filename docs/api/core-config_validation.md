---
title: core.config_validation
category: api
tags: [api, core]
---

# core.config_validation

Configuration Validation System for Enhanced Model Management

Provides comprehensive validation for configuration settings, including
business rule validation, constraint checking, and migration support.

## Classes

### ValidationError

Represents a configuration validation error

### ValidationResult

Result of configuration validation

#### Methods

##### has_errors(self: Any) -> bool



##### has_warnings(self: Any) -> bool



### ConfigurationValidator

Validates enhanced model configuration settings

#### Methods

##### __init__(self: Any)



##### validate_user_preferences(self: Any, preferences: UserPreferences) -> ValidationResult

Validate user preferences configuration

##### validate_admin_policies(self: Any, policies: AdminPolicies) -> ValidationResult

Validate admin policies configuration

##### validate_feature_flags(self: Any, feature_flags: FeatureFlagConfig) -> ValidationResult

Validate feature flag configuration

##### _validate_download_config(self: Any, config: DownloadConfig) -> <ast.Subscript object at 0x000001942A289ED0>

Validate download configuration

##### _validate_health_monitoring_config(self: Any, config: HealthMonitoringConfig) -> <ast.Subscript object at 0x000001942A2891E0>

Validate health monitoring configuration

##### _validate_fallback_config(self: Any, config: FallbackConfig) -> <ast.Subscript object at 0x000001942A2838B0>

Validate fallback configuration

##### _validate_analytics_config(self: Any, config: AnalyticsConfig) -> <ast.Subscript object at 0x000001942A283190>

Validate analytics configuration

##### _validate_update_config(self: Any, config: UpdateConfig) -> <ast.Subscript object at 0x000001942A282A70>

Validate update configuration

##### _validate_notification_config(self: Any, config: NotificationConfig) -> <ast.Subscript object at 0x000001942A2823E0>

Validate notification configuration

##### _validate_storage_config(self: Any, config: StorageConfig) -> <ast.Subscript object at 0x000001942A2819C0>

Validate storage configuration

##### _validate_model_lists(self: Any, preferred: <ast.Subscript object at 0x000001942A281870>, blocked: <ast.Subscript object at 0x000001942A2817B0>) -> <ast.Subscript object at 0x000001942A280520>

Validate model preference lists

