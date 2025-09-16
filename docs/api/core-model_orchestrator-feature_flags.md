---
title: core.model_orchestrator.feature_flags
category: api
tags: [api, core]
---

# core.model_orchestrator.feature_flags

Feature Flags System - Gradual rollout and configuration management.

This module provides a centralized feature flag system for controlling
the rollout of model orchestrator features with environment-based configuration.

## Classes

### OrchestratorFeatureFlags

Feature flags for model orchestrator functionality.

#### Methods

##### from_env(cls: Any, prefix: str) -> OrchestratorFeatureFlags

Create feature flags from environment variables.

Args:
    prefix: Environment variable prefix (default: "WAN_")
    
Returns:
    OrchestratorFeatureFlags instance with values from environment

##### from_file(cls: Any, config_path: str) -> OrchestratorFeatureFlags

Load feature flags from a JSON configuration file.

Args:
    config_path: Path to JSON configuration file
    
Returns:
    OrchestratorFeatureFlags instance
    
Raises:
    FileNotFoundError: If config file doesn't exist
    ValueError: If config file is invalid

##### to_dict(self: Any) -> <ast.Subscript object at 0x00000194275CBF10>

Convert feature flags to dictionary.

##### to_json(self: Any, indent: int) -> str

Convert feature flags to JSON string.

##### save_to_file(self: Any, config_path: str) -> None

Save feature flags to a JSON configuration file.

Args:
    config_path: Path to save configuration file

##### is_feature_enabled(self: Any, feature_name: str) -> bool

Check if a specific feature is enabled.

Args:
    feature_name: Name of the feature flag
    
Returns:
    True if feature is enabled, False otherwise

##### enable_feature(self: Any, feature_name: str) -> None

Enable a specific feature.

Args:
    feature_name: Name of the feature flag to enable

##### disable_feature(self: Any, feature_name: str) -> None

Disable a specific feature.

Args:
    feature_name: Name of the feature flag to disable

##### get_rollout_stage(self: Any) -> str

Determine the current rollout stage based on enabled features.

Returns:
    String indicating rollout stage: "disabled", "development", "staging", "production"

##### validate_configuration(self: Any) -> <ast.Subscript object at 0x000001942C856890>

Validate feature flag configuration for consistency.

Returns:
    Dictionary of validation warnings/errors

### FeatureFlagManager

Manager for feature flag operations and rollout control.

#### Methods

##### __init__(self: Any, config_path: <ast.Subscript object at 0x000001942C8571F0>)

Initialize feature flag manager.

Args:
    config_path: Optional path to feature flags configuration file

##### flags(self: Any) -> OrchestratorFeatureFlags

Get current feature flags, loading if necessary.

##### load_flags(self: Any) -> OrchestratorFeatureFlags

Load feature flags from configuration.

Returns:
    OrchestratorFeatureFlags instance

##### reload_flags(self: Any) -> None

Reload feature flags from configuration.

##### save_flags(self: Any) -> None

Save current feature flags to file if config path is set.

##### is_enabled(self: Any, feature_name: str) -> bool

Check if a feature is enabled.

##### enable_rollout_stage(self: Any, stage: str) -> None

Enable features for a specific rollout stage.

Args:
    stage: Rollout stage ("development", "staging", "production")

##### validate_current_configuration(self: Any) -> bool

Validate current feature flag configuration.

Returns:
    True if configuration is valid, False if there are issues

##### get_status_report(self: Any) -> <ast.Subscript object at 0x000001942CB70B50>

Get comprehensive status report of feature flags.

Returns:
    Dictionary with status information

