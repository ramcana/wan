---
title: scripts.config_migration
category: api
tags: [api, scripts]
---

# scripts.config_migration

Configuration migration script to migrate from existing systems to FastAPI integration.
Handles merging configurations from different sources and ensuring compatibility.

## Classes

### ConfigurationMigrator

Handles configuration migration from existing systems.

#### Methods

##### __init__(self: Any)



##### load_existing_configs(self: Any) -> <ast.Subscript object at 0x0000019428987130>

Load all existing configuration files.

##### create_default_config(self: Any) -> <ast.Subscript object at 0x000001942A148070>

Create default configuration structure.

##### merge_wan22_config(self: Any, configs: <ast.Subscript object at 0x0000019427FE9060>) -> None

Merge WAN2.2 configuration settings.

##### merge_local_install_config(self: Any, configs: <ast.Subscript object at 0x00000194280625F0>) -> None

Merge local installation configuration settings.

##### merge_model_config(self: Any, configs: <ast.Subscript object at 0x00000194280590C0>) -> None

Merge model-specific configuration settings.

##### merge_fastapi_config(self: Any, configs: <ast.Subscript object at 0x000001942805B580>) -> None

Merge existing FastAPI configuration settings.

##### validate_merged_config(self: Any) -> <ast.Subscript object at 0x0000019427F7FB20>

Validate the merged configuration and return any issues.

##### backup_existing_config(self: Any) -> bool

Backup existing configuration file.

##### save_merged_config(self: Any) -> bool

Save the merged configuration to file.

##### run_migration(self: Any) -> bool

Run the complete configuration migration.

