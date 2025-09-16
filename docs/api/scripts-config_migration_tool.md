---
title: scripts.config_migration_tool
category: api
tags: [api, scripts]
---

# scripts.config_migration_tool

Configuration Migration Tool for Enhanced Model Management

This tool helps migrate existing configurations to the new enhanced model
management configuration format, with validation and backup capabilities.

## Classes

### ConfigurationMigrationTool

Tool for migrating and managing enhanced model configurations

#### Methods

##### __init__(self: Any)



##### migrate_configuration(self: Any, source_path: str, target_path: str, backup: bool) -> bool

Migrate configuration from source to target path

Args:
    source_path: Path to source configuration file
    target_path: Path to target configuration file
    backup: Whether to create backup of existing target
    
Returns:
    True if migration successful, False otherwise

##### validate_configuration(self: Any, config_path: str) -> bool

Validate existing configuration file

Args:
    config_path: Path to configuration file
    
Returns:
    True if configuration is valid, False otherwise

##### create_default_configuration(self: Any, config_path: str) -> bool

Create default configuration file

Args:
    config_path: Path where to create configuration file
    
Returns:
    True if creation successful, False otherwise

##### _detect_and_migrate(self: Any, config_data: <ast.Subscript object at 0x00000194344D7B20>) -> EnhancedModelConfiguration

Detect configuration format and migrate to current version

##### _load_enhanced_config(self: Any, config_data: <ast.Subscript object at 0x00000194344D6DA0>) -> EnhancedModelConfiguration

Load existing enhanced configuration

##### _migrate_from_legacy_model_config(self: Any, config_data: <ast.Subscript object at 0x00000194344D69B0>) -> EnhancedModelConfiguration

Migrate from legacy model configuration format

##### _migrate_from_basic_config(self: Any, config_data: <ast.Subscript object at 0x00000194344D57B0>) -> EnhancedModelConfiguration

Migrate from basic application configuration

##### _create_default_with_data(self: Any, config_data: <ast.Subscript object at 0x00000194344D4760>) -> EnhancedModelConfiguration

Create default configuration with any available data

##### _validate_migrated_config(self: Any, config: EnhancedModelConfiguration) -> ValidationResult

Validate migrated configuration

