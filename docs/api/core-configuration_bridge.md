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

##### __init__(self: Any, config_path: <ast.Subscript object at 0x00000194343B35E0>)



##### _load_config(self: Any) -> bool

Load configuration from config.json file

##### _create_default_config(self: Any)

Create default configuration structure

##### get_config(self: Any, section: <ast.Subscript object at 0x000001943442C070>) -> <ast.Subscript object at 0x0000019434371570>

Get configuration data

##### get_model_paths(self: Any) -> <ast.Subscript object at 0x0000019434371B10>

Get model path configuration

##### get_optimization_settings(self: Any) -> <ast.Subscript object at 0x000001942EFF44F0>

Get optimization settings

##### update_optimization_setting(self: Any, setting_name: str, value: Any) -> bool

Update optimization setting at runtime

##### _save_config(self: Any) -> bool

Save configuration to file

##### validate_configuration(self: Any) -> <ast.Subscript object at 0x0000019434639BD0>

Validate current configuration

##### get_runtime_config_for_generation(self: Any, model_type: str) -> <ast.Subscript object at 0x000001943463A4A0>

Get runtime configuration for model generation

##### get_config_summary(self: Any) -> <ast.Subscript object at 0x000001942F04D840>

Get configuration summary

