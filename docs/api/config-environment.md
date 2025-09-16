---
title: config.environment
category: api
tags: [api, config]
---

# config.environment

Environment-specific configuration management.

## Classes

### Environment

Application environments.

### EnvironmentConfig

Manages environment-specific configuration.

#### Methods

##### __init__(self: Any, env: <ast.Subscript object at 0x000001942A2AABC0>)



##### _load_config(self: Any) -> None

Load configuration based on environment.

##### _merge_config(self: Any, env_config: <ast.Subscript object at 0x000001942A2A9840>) -> None

Merge environment-specific config with base config.

##### _apply_env_overrides(self: Any) -> None

Apply environment variable overrides.

##### get(self: Any, section: str, key: str, default: Any) -> Any

Get configuration value.

##### get_section(self: Any, section: str) -> <ast.Subscript object at 0x000001942A2178E0>

Get entire configuration section.

##### set(self: Any, section: str, key: str, value: Any) -> None

Set configuration value.

##### is_development(self: Any) -> bool

Check if running in development environment.

##### is_production(self: Any) -> bool

Check if running in production environment.

##### is_testing(self: Any) -> bool

Check if running in testing environment.

##### get_database_url(self: Any) -> str

Get database URL based on environment.

##### get_log_config(self: Any) -> <ast.Subscript object at 0x000001942A214160>

Get logging configuration.

##### get_cors_origins(self: Any) -> list

Get CORS origins based on environment.

##### get_api_config(self: Any) -> <ast.Subscript object at 0x000001942A1B4550>

Get API configuration.

##### get_storage_config(self: Any) -> <ast.Subscript object at 0x00000194278F71C0>

Get storage configuration.

##### get_model_config(self: Any) -> <ast.Subscript object at 0x00000194278F6B30>

Get model configuration.

##### get_optimization_config(self: Any) -> <ast.Subscript object at 0x00000194278F62C0>

Get optimization configuration.

##### save_config(self: Any, path: <ast.Subscript object at 0x00000194278F6110>) -> None

Save current configuration to file.

## Constants

### DEVELOPMENT

Type: `str`

Value: `development`

### TESTING

Type: `str`

Value: `testing`

### STAGING

Type: `str`

Value: `staging`

### PRODUCTION

Type: `str`

Value: `production`

