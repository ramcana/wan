---
title: config.config_validator
category: api
tags: [api, config]
---

# config.config_validator

Configuration validation for ensuring existing config.json works with new system.

## Classes

### ConfigValidationResult

Result of configuration validation.

### ConfigValidator

Validates and migrates existing config.json for new system.

#### Methods

##### __init__(self: Any, config_path: str)



##### load_existing_config(self: Any) -> <ast.Subscript object at 0x000001943442FCA0>

Load existing configuration file.

##### validate_config_structure(self: Any, config: <ast.Subscript object at 0x000001943442FA90>) -> bool

Validate configuration structure against expected schema.

##### _validate_section(self: Any, section: <ast.Subscript object at 0x000001943442EDD0>, schema: <ast.Subscript object at 0x000001943442ECB0>, section_name: str) -> bool

Validate a configuration section.

##### migrate_config(self: Any, old_config: <ast.Subscript object at 0x000001943442DD20>) -> <ast.Subscript object at 0x00000194343B2A70>

Migrate old configuration format to new format.

##### _set_config_defaults(self: Any, config: <ast.Subscript object at 0x00000194343B28C0>) -> None

Set default values for missing configuration options.

##### _handle_special_migrations(self: Any, old_config: <ast.Subscript object at 0x00000194343B1390>, new_config: <ast.Subscript object at 0x00000194343B1270>) -> None

Handle special migration cases that don't fit the simple mapping.

##### validate_paths(self: Any, config: <ast.Subscript object at 0x0000019434463F10>) -> bool

Validate that configured paths exist or can be created.

##### validate_model_compatibility(self: Any, config: <ast.Subscript object at 0x0000019434462DD0>) -> bool

Validate that model settings are compatible with existing models.

##### run_validation(self: Any) -> ConfigValidationResult

Run complete configuration validation.

##### save_migrated_config(self: Any, output_path: str) -> bool

Save migrated configuration to file.

## Constants

### EXPECTED_SCHEMA

Type: `unknown`

### CONFIG_MIGRATION_MAP

Type: `unknown`

