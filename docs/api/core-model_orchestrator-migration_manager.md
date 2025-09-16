---
title: core.model_orchestrator.migration_manager
category: api
tags: [api, core]
---

# core.model_orchestrator.migration_manager

Migration Manager - Configuration migration and backward compatibility tools.

This module provides tools for migrating from legacy configuration formats
to the new model orchestrator system, with backward compatibility adapters
and gradual rollout support.

## Classes

### LegacyConfig

Legacy configuration structure from config.json.

### MigrationResult

Result of a configuration migration.

#### Methods

##### __post_init__(self: Any)



### FeatureFlags

Feature flags for gradual rollout of model orchestrator.

#### Methods

##### from_env(cls: Any) -> FeatureFlags

Create feature flags from environment variables.

### LegacyPathAdapter

Adapter for resolving legacy model paths to orchestrator paths.

#### Methods

##### __init__(self: Any, legacy_models_dir: str, orchestrator_models_root: str)

Initialize the legacy path adapter.

Args:
    legacy_models_dir: Legacy models directory path
    orchestrator_models_root: New orchestrator models root path

##### map_legacy_path(self: Any, legacy_model_name: str) -> <ast.Subscript object at 0x0000019430320910>

Map a legacy model name to the new orchestrator path.

Args:
    legacy_model_name: Legacy model name (e.g., "t2v-A14B")
    
Returns:
    New orchestrator path or None if no mapping exists

##### get_legacy_path(self: Any, legacy_model_name: str) -> str

Get the legacy path for a model.

##### path_exists_in_legacy(self: Any, legacy_model_name: str) -> bool

Check if a model exists in the legacy location.

##### migrate_model_files(self: Any, legacy_model_name: str, dry_run: bool) -> bool

Migrate model files from legacy location to orchestrator location.

Args:
    legacy_model_name: Legacy model name
    dry_run: If True, only simulate the migration
    
Returns:
    True if migration was successful or would be successful

### ConfigurationMigrator

Tool for migrating legacy configurations to model orchestrator format.

#### Methods

##### __init__(self: Any)

Initialize the configuration migrator.

##### load_legacy_config(self: Any, config_path: str) -> LegacyConfig

Load legacy configuration from config.json.

Args:
    config_path: Path to the legacy config.json file
    
Returns:
    LegacyConfig object
    
Raises:
    MigrationError: If config cannot be loaded or parsed

##### generate_manifest_from_legacy(self: Any, legacy_config: LegacyConfig) -> <ast.Subscript object at 0x0000019431B13EE0>

Generate a models.toml manifest structure from legacy configuration.

Args:
    legacy_config: Legacy configuration object
    
Returns:
    Dictionary representing the new manifest structure

##### scan_legacy_model_files(self: Any, models_dir: str, model_name: str) -> <ast.Subscript object at 0x0000019431B101F0>

Scan legacy model directory to generate file specifications.

Args:
    models_dir: Legacy models directory
    model_name: Model name to scan
    
Returns:
    List of file specifications

##### _determine_component_type(self: Any, file_path: str) -> <ast.Subscript object at 0x00000194300CCB80>

Determine the component type based on file path.

##### write_manifest(self: Any, manifest_data: <ast.Subscript object at 0x00000194300CCC70>, output_path: str) -> None

Write manifest data to a TOML file.

Args:
    manifest_data: Manifest dictionary
    output_path: Output file path
    
Raises:
    MigrationError: If writing fails

##### migrate_configuration(self: Any, legacy_config_path: str, output_manifest_path: str, legacy_models_dir: <ast.Subscript object at 0x00000194300CC1F0>, backup: bool, scan_files: bool) -> MigrationResult

Perform complete configuration migration.

Args:
    legacy_config_path: Path to legacy config.json
    output_manifest_path: Path for new models.toml
    legacy_models_dir: Legacy models directory (for file scanning)
    backup: Whether to create backup of existing files
    scan_files: Whether to scan existing model files
    
Returns:
    MigrationResult with details of the migration

### ManifestValidator

Validator for model manifest files and configurations.

#### Methods

##### __init__(self: Any)

Initialize the manifest validator.

##### validate_manifest_file(self: Any, manifest_path: str) -> <ast.Subscript object at 0x000001942FD78DF0>

Validate a manifest file for correctness and completeness.

Args:
    manifest_path: Path to the manifest file
    
Returns:
    List of validation errors (empty if valid)

##### _validate_manifest_structure(self: Any, manifest_path: str) -> <ast.Subscript object at 0x000001942FD79990>

Validate manifest structure and content.

##### _validate_model_structure(self: Any, model_id: str, model_data: <ast.Subscript object at 0x000001942FD79660>) -> <ast.Subscript object at 0x0000019434047970>

Validate individual model structure.

##### validate_configuration_compatibility(self: Any, manifest_path: str, legacy_config_path: <ast.Subscript object at 0x0000019434045690>) -> <ast.Subscript object at 0x0000019434046830>

Validate compatibility between new manifest and legacy configuration.

Args:
    manifest_path: Path to new manifest
    legacy_config_path: Path to legacy config (optional)
    
Returns:
    List of compatibility errors

### RollbackManager

Manager for rolling back migrations and configurations.

#### Methods

##### __init__(self: Any)

Initialize the rollback manager.

##### create_rollback_point(self: Any, config_paths: <ast.Subscript object at 0x0000019434045F90>, rollback_dir: str) -> str

Create a rollback point by backing up current configurations.

Args:
    config_paths: List of configuration file paths to backup
    rollback_dir: Directory to store rollback data
    
Returns:
    Rollback point identifier

##### execute_rollback(self: Any, rollback_id: str, rollback_dir: str) -> bool

Execute a rollback to a previous configuration state.

Args:
    rollback_id: Rollback point identifier
    rollback_dir: Directory containing rollback data
    
Returns:
    True if rollback was successful

##### list_rollback_points(self: Any, rollback_dir: str) -> <ast.Subscript object at 0x000001942F6B2DA0>

List available rollback points.

Args:
    rollback_dir: Directory containing rollback data
    
Returns:
    List of rollback point information

