---
title: tools.config_manager.unified_config
category: api
tags: [api, tools]
---

# tools.config_manager.unified_config

Unified Configuration Schema

This module defines the comprehensive configuration schema for the WAN22 project,
including all system, service, and environment settings with validation rules.

## Classes

### LogLevel

Supported logging levels

### QuantizationLevel

Supported model quantization levels

### Environment

Supported deployment environments

### SystemConfig

Core system configuration

### APIConfig

API server configuration

### DatabaseConfig

Database configuration

### ModelConfig

Model management configuration

### HardwareConfig

Hardware optimization configuration

### GenerationConfig

Video generation configuration

### UIConfig

User interface configuration

### FrontendConfig

Frontend application configuration

### WebSocketConfig

WebSocket configuration

### LoggingConfig

Logging configuration

### SecurityConfig

Security configuration

### PerformanceConfig

Performance monitoring configuration

### RecoveryConfig

Error recovery configuration

### EnvironmentValidationConfig

Environment validation configuration

### PromptEnhancementConfig

Prompt enhancement configuration

### FeatureFlags

Feature flags configuration

### EnvironmentOverrides

Environment-specific configuration overrides

### UnifiedConfig

Unified configuration schema for the WAN22 project.

This class provides a comprehensive configuration system that consolidates
all scattered configuration files into a single, validated structure.

#### Methods

##### to_dict(self: Any) -> <ast.Subscript object at 0x0000019433CF6A10>

Convert configuration to dictionary

##### to_json(self: Any, indent: int) -> str

Convert configuration to JSON string

##### to_yaml(self: Any) -> str

Convert configuration to YAML string

##### from_dict(cls: Any, data: <ast.Subscript object at 0x0000019433CF7820>) -> UnifiedConfig

Create configuration from dictionary

##### from_json(cls: Any, json_str: str) -> UnifiedConfig

Create configuration from JSON string

##### from_yaml(cls: Any, yaml_str: str) -> UnifiedConfig

Create configuration from YAML string

##### from_file(cls: Any, file_path: <ast.Subscript object at 0x00000194301074F0>) -> UnifiedConfig

Load configuration from file

##### save_to_file(self: Any, file_path: <ast.Subscript object at 0x00000194300645B0>, format: str) -> None

Save configuration to file

##### apply_environment_overrides(self: Any, environment: <ast.Subscript object at 0x000001942F87CB20>) -> UnifiedConfig

Apply environment-specific overrides

##### get_config_path(self: Any, path: str) -> Any

Get configuration value by dot-separated path

##### set_config_path(self: Any, path: str, value: Any) -> None

Set configuration value by dot-separated path

## Constants

### DEBUG

Type: `str`

Value: `DEBUG`

### INFO

Type: `str`

Value: `INFO`

### WARNING

Type: `str`

Value: `WARNING`

### ERROR

Type: `str`

Value: `ERROR`

### CRITICAL

Type: `str`

Value: `CRITICAL`

### FP16

Type: `str`

Value: `fp16`

### BF16

Type: `str`

Value: `bf16`

### INT8

Type: `str`

Value: `int8`

### FP32

Type: `str`

Value: `fp32`

### DEVELOPMENT

Type: `str`

Value: `development`

### STAGING

Type: `str`

Value: `staging`

### PRODUCTION

Type: `str`

Value: `production`

### TESTING

Type: `str`

Value: `testing`

