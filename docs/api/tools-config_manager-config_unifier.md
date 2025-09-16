---
title: tools.config_manager.config_unifier
category: api
tags: [api, tools]
---

# tools.config_manager.config_unifier



## Classes

### ConfigSource

Represents a discovered configuration source

### MigrationReport

Report of configuration migration process

### ConfigurationUnifier

Handles migration of scattered configuration files to unified system

#### Methods

##### __init__(self: Any, project_root: Path)



##### discover_config_files(self: Any) -> <ast.Subscript object at 0x0000019428D40D30>

Discover all configuration files in the project

Returns:
    List of discovered configuration sources

##### _should_skip_file(self: Any, file_path: Path) -> bool

Check if a file should be skipped during discovery

##### _analyze_config_file(self: Any, file_path: Path) -> <ast.Subscript object at 0x000001942B3020B0>

Analyze a configuration file and extract its content

Args:
    file_path: Path to the configuration file
    
Returns:
    ConfigSource object or None if file cannot be parsed

##### _detect_format(self: Any, file_path: Path) -> str

Detect the format of a configuration file

##### _parse_env_file(self: Any, file_path: Path) -> <ast.Subscript object at 0x000001942B3002E0>

Parse environment file into dictionary

##### _categorize_config(self: Any, file_path: Path, content: <ast.Subscript object at 0x000001942B3000D0>) -> <ast.Subscript object at 0x000001942B2F68C0>

Categorize a configuration file based on path and content

Returns:
    Tuple of (category, confidence_score)

##### migrate_to_unified_config(self: Any, sources: <ast.Subscript object at 0x000001942B2F6710>, output_path: Path, create_backup: bool) -> MigrationReport

Migrate discovered configuration sources to unified configuration

Args:
    sources: List of configuration sources to migrate (auto-discover if None)
    output_path: Path for the unified configuration file
    create_backup: Whether to create backups of original files
    
Returns:
    Migration report with results and any errors

##### _create_backup(self: Any, sources: <ast.Subscript object at 0x000001942B3BA470>) -> Path

Create backup of original configuration files

##### _merge_configurations(self: Any, sources: <ast.Subscript object at 0x000001942B3B8CD0>, report: MigrationReport) -> UnifiedConfig

Merge multiple configuration sources into a unified configuration

Args:
    sources: List of configuration sources to merge
    report: Migration report to update with results
    
Returns:
    Unified configuration object

##### _merge_category(self: Any, unified_config: UnifiedConfig, category: str, sources: <ast.Subscript object at 0x0000019427F3A950>, report: MigrationReport)

Merge sources for a specific category into the unified configuration

##### _merge_backend_config(self: Any, unified_config: UnifiedConfig, source: ConfigSource)

Merge backend configuration source

##### _merge_startup_config(self: Any, unified_config: UnifiedConfig, source: ConfigSource)

Merge startup configuration source

##### _merge_system_config(self: Any, unified_config: UnifiedConfig, source: ConfigSource)

Merge system configuration source

##### _merge_models_config(self: Any, unified_config: UnifiedConfig, source: ConfigSource)

Merge models configuration source

##### _merge_hardware_config(self: Any, unified_config: UnifiedConfig, source: ConfigSource)

Merge hardware configuration source

##### _merge_generation_config(self: Any, unified_config: UnifiedConfig, source: ConfigSource)

Merge generation configuration source

##### _merge_frontend_config(self: Any, unified_config: UnifiedConfig, source: ConfigSource)

Merge frontend configuration source

##### _merge_generic_config(self: Any, unified_config: UnifiedConfig, source: ConfigSource)

Generic merge for unknown configuration categories

##### rollback_migration(self: Any, backup_path: Path) -> bool

Rollback a configuration migration using backup

Args:
    backup_path: Path to the backup directory
    
Returns:
    True if rollback was successful, False otherwise

##### generate_migration_preview(self: Any, sources: <ast.Subscript object at 0x0000019428D4FB80>) -> <ast.Subscript object at 0x0000019428DA5090>

Generate a preview of what would be migrated without actually doing it

Args:
    sources: List of configuration sources (auto-discover if None)
    
Returns:
    Dictionary with migration preview information

##### _detect_conflicts(self: Any, sources: <ast.Subscript object at 0x0000019428DA5240>, preview: <ast.Subscript object at 0x0000019428DA5300>)

Detect potential conflicts in configuration migration

##### _generate_recommendations(self: Any, sources: <ast.Subscript object at 0x0000019428DA60E0>, preview: <ast.Subscript object at 0x0000019428DA61A0>)

Generate recommendations for configuration migration

