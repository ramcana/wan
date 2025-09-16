---
title: core.model_orchestrator.test_migration_tools
category: api
tags: [api, core]
---

# core.model_orchestrator.test_migration_tools

Tests for migration and compatibility tools.

This module contains comprehensive tests for configuration migration,
validation, rollback functionality, and backward compatibility.

## Classes

### TestLegacyConfig

Test LegacyConfig data class.

#### Methods

##### test_legacy_config_creation(self: Any)

Test creating LegacyConfig from data.

### TestFeatureFlags

Test FeatureFlags functionality.

#### Methods

##### test_default_feature_flags(self: Any)

Test default feature flag values.

##### test_feature_flags_from_env(self: Any)

Test loading feature flags from environment variables.

### TestLegacyPathAdapter

Test LegacyPathAdapter functionality.

#### Methods

##### setup_method(self: Any)

Set up test fixtures.

##### teardown_method(self: Any)

Clean up test fixtures.

##### test_map_legacy_path(self: Any)

Test mapping legacy model names to new paths.

##### test_get_legacy_path(self: Any)

Test getting legacy path for a model.

##### test_path_exists_in_legacy(self: Any)

Test checking if model exists in legacy location.

##### test_migrate_model_files_dry_run(self: Any)

Test dry run migration.

##### test_migrate_model_files_actual(self: Any)

Test actual migration of model files.

##### test_migrate_nonexistent_model(self: Any)

Test migration of nonexistent model.

### TestConfigurationMigrator

Test ConfigurationMigrator functionality.

#### Methods

##### setup_method(self: Any)

Set up test fixtures.

##### teardown_method(self: Any)

Clean up test fixtures.

##### create_legacy_config(self: Any) -> str

Create a test legacy config file.

##### test_load_legacy_config(self: Any)

Test loading legacy configuration.

##### test_load_nonexistent_config(self: Any)

Test loading nonexistent config file.

##### test_generate_manifest_from_legacy(self: Any)

Test generating manifest from legacy config.

##### test_determine_component_type(self: Any)

Test component type determination from file paths.

##### test_scan_legacy_model_files(self: Any)

Test scanning legacy model files.

##### test_write_manifest(self: Any, mock_tomli_w: Any)

Test writing manifest to TOML file.

##### test_write_manifest_without_tomli_w(self: Any)

Test writing manifest when tomli_w is not available.

##### test_migrate_configuration_success(self: Any)

Test successful configuration migration.

##### test_migrate_configuration_with_backup(self: Any)

Test migration with backup creation.

### TestManifestValidator

Test ManifestValidator functionality.

#### Methods

##### setup_method(self: Any)

Set up test fixtures.

##### teardown_method(self: Any)

Clean up test fixtures.

##### create_valid_manifest(self: Any) -> str

Create a valid test manifest file.

##### create_invalid_manifest(self: Any) -> str

Create an invalid test manifest file.

##### test_validate_valid_manifest(self: Any)

Test validation of a valid manifest.

##### test_validate_invalid_manifest(self: Any)

Test validation of an invalid manifest.

##### test_validate_nonexistent_manifest(self: Any)

Test validation of nonexistent manifest.

##### test_validate_configuration_compatibility(self: Any)

Test compatibility validation between manifest and legacy config.

### TestRollbackManager

Test RollbackManager functionality.

#### Methods

##### setup_method(self: Any)

Set up test fixtures.

##### teardown_method(self: Any)

Clean up test fixtures.

##### test_create_rollback_point(self: Any)

Test creating a rollback point.

##### test_execute_rollback(self: Any)

Test executing a rollback.

##### test_execute_nonexistent_rollback(self: Any)

Test executing rollback for nonexistent rollback point.

##### test_list_rollback_points(self: Any)

Test listing rollback points.

### TestMigrationIntegration

Integration tests for migration tools.

#### Methods

##### setup_method(self: Any)

Set up test fixtures.

##### teardown_method(self: Any)

Clean up test fixtures.

##### test_full_migration_workflow(self: Any)

Test complete migration workflow from legacy to orchestrator.

##### test_migration_error_handling(self: Any)

Test error handling in migration workflow.

