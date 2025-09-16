---
title: core.configuration_bridge
category: api
tags: [api, core]
---

# core.configuration_bridge

Configuration Bridge for FastAPI backend integration

## Classes

### ConfigurationBridge

Configuration adapter for existing config.json structure with FastAPI

#### Methods

##### __init__(self: Any, config_path: <ast.Subscript object at 0x000001942A2C3A00>)



##### _load_config(self: Any) -> bool

Load configuration from config.json file

##### _create_default_config(self: Any)

Create default configuration structure

##### get_config(self: Any, section: <ast.Subscript object at 0x000001942A2C0D00>) -> <ast.Subscript object at 0x000001942A2C0940>

Get configuration data

##### get_model_paths(self: Any) -> <ast.Subscript object at 0x00000194278E27A0>

Get model path configuration

##### get_optimization_settings(self: Any) -> <ast.Subscript object at 0x00000194278E2D10>

Get optimization settings

##### update_optimization_setting(self: Any, setting_name: str, value: Any) -> bool

Update optimization setting at runtime

##### _save_config(self: Any) -> bool

Save configuration to file

##### validate_configuration(self: Any) -> <ast.Subscript object at 0x000001942A1FE860>

Validate current configuration

##### get_runtime_config_for_generation(self: Any, model_type: str) -> <ast.Subscript object at 0x000001942A1FD630>

Get runtime configuration for model generation

##### get_config_summary(self: Any) -> <ast.Subscript object at 0x000001942A216AA0>

Get configuration summary

