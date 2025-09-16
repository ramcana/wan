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

##### __init__(self: Any, env: <ast.Subscript object at 0x000001942EFF76D0>)



##### _load_config(self: Any) -> None

Load configuration based on environment.

##### _merge_config(self: Any, env_config: <ast.Subscript object at 0x000001942EFF6350>) -> None

Merge environment-specific config with base config.

##### _apply_env_overrides(self: Any) -> None

Apply environment variable overrides.

##### get(self: Any, section: str, key: str, default: Any) -> Any

Get configuration value.

##### get_section(self: Any, section: str) -> <ast.Subscript object at 0x000001942F028430>

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

##### get_log_config(self: Any) -> <ast.Subscript object at 0x00000194345FBA90>

Get logging configuration.

##### get_cors_origins(self: Any) -> list

Get CORS origins based on environment.

##### get_api_config(self: Any) -> <ast.Subscript object at 0x000001943463AC20>

Get API configuration.

##### get_storage_config(self: Any) -> <ast.Subscript object at 0x0000019434381120>

Get storage configuration.

##### get_model_config(self: Any) -> <ast.Subscript object at 0x00000194343809A0>

Get model configuration.

##### get_optimization_config(self: Any) -> <ast.Subscript object at 0x0000019434380220>

Get optimization configuration.

##### save_config(self: Any, path: <ast.Subscript object at 0x0000019434380070>) -> None

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

