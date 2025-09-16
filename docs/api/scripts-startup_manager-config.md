---
title: scripts.startup_manager.config
category: api
tags: [api, scripts]
---

# scripts.startup_manager.config

Configuration models for the startup manager using Pydantic.
Handles loading and validation of startup_config.json with environment variable overrides.

## Classes

### ServerConfig

Base configuration for server instances.

### BackendConfig

Configuration for FastAPI backend server.

#### Methods

##### validate_log_level(cls: Any, v: Any)



### FrontendConfig

Configuration for React frontend server.

### LoggingConfig

Configuration for logging system.

#### Methods

##### validate_log_level(cls: Any, v: Any)



### RecoveryConfig

Configuration for recovery and error handling.

### EnvironmentConfig

Configuration for environment validation.

### SecurityConfig

Configuration for security settings.

#### Methods

##### validate_port_range(cls: Any, v: Any)



### StartupConfig

Main startup configuration containing all server and system settings.

#### Methods

##### sync_legacy_fields(self: Any)

Sync legacy fields with new structured config for backward compatibility.

### ConfigLoader

Handles loading and validation of startup configuration with environment overrides.

#### Methods

##### __init__(self: Any, config_path: <ast.Subscript object at 0x0000019429CB7A30>, env_prefix: str)



##### load_config(self: Any, apply_env_overrides: bool) -> StartupConfig

Load configuration from file with optional environment variable overrides.

##### _get_env_overrides(self: Any) -> <ast.Subscript object at 0x0000019427BA7130>

Extract configuration overrides from environment variables.

##### _parse_env_value(self: Any, value: str) -> Any

Parse environment variable value to appropriate Python type.

##### _merge_config_data(self: Any, base_config: <ast.Subscript object at 0x0000019427BA5480>, overrides: <ast.Subscript object at 0x0000019427BA60E0>) -> <ast.Subscript object at 0x0000019427BA6AD0>

Recursively merge configuration data with overrides.

##### save_config(self: Any) -> None

Save current configuration to file.

##### validate_config(self: Any) -> <ast.Subscript object at 0x0000019427BA5DE0>

Validate current configuration and return comprehensive validation results.

##### _validate_ports(self: Any, results: <ast.Subscript object at 0x0000019427BA6470>) -> None

Validate port configurations.

##### _validate_timeouts(self: Any, results: <ast.Subscript object at 0x0000019428D06B00>) -> None

Validate timeout configurations.

##### _validate_security(self: Any, results: <ast.Subscript object at 0x0000019428D05960>) -> None

Validate security configurations.

##### _validate_environment_config(self: Any, results: <ast.Subscript object at 0x0000019428D04370>) -> None

Validate environment configuration.

##### _validate_recovery_config(self: Any, results: <ast.Subscript object at 0x0000019428D07730>) -> None

Validate recovery configuration.

##### _validate_logging_config(self: Any, results: <ast.Subscript object at 0x0000019428DA9750>) -> None

Validate logging configuration.

##### get_effective_config(self: Any) -> <ast.Subscript object at 0x0000019428DA83D0>

Get the effective configuration including environment overrides.

##### get_env_overrides_summary(self: Any) -> <ast.Subscript object at 0x0000019427BBF1F0>

Get summary of active environment variable overrides.

##### export_config_for_ci(self: Any, format: str) -> str

Export configuration in format suitable for CI/CD systems.

##### _export_as_env_vars(self: Any) -> str

Export configuration as environment variables.

##### create_deployment_config(self: Any, deployment_type: str) -> <ast.Subscript object at 0x0000019428D660E0>

Create optimized configuration for specific deployment types.

##### config(self: Any) -> <ast.Subscript object at 0x0000019428D66770>

Get current configuration.

