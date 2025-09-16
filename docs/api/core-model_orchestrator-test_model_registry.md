---
title: core.model_orchestrator.test_model_registry
category: api
tags: [api, core]
---

# core.model_orchestrator.test_model_registry

Unit tests for the Model Registry system.

Tests manifest parsing, validation, model ID normalization, and error handling.

## Classes

### TestModelIdNormalization

Test model ID normalization functionality.

#### Methods

##### test_normalize_basic_model_id(self: Any)

Test basic model ID normalization.

##### test_normalize_model_id_with_variant(self: Any)

Test model ID normalization with variant.

##### test_normalize_model_id_with_embedded_variant(self: Any)

Test model ID normalization with embedded variant.

##### test_normalize_variant_override(self: Any)

Test that explicit variant overrides embedded variant.

##### test_normalize_invalid_model_id_format(self: Any)

Test error handling for invalid model ID formats.

##### test_normalize_invalid_version_format(self: Any)

Test error handling for invalid version formats.

##### test_normalize_invalid_model_name(self: Any)

Test error handling for invalid model names.

##### test_normalize_invalid_variant(self: Any)

Test error handling for invalid variants.

##### test_normalize_empty_model_id(self: Any)

Test error handling for empty model ID.

### TestManifestParsing

Test manifest parsing and validation.

#### Methods

##### create_temp_manifest(self: Any, content: str) -> str

Create a temporary manifest file with given content.

##### test_load_valid_manifest(self: Any)

Test loading a valid manifest.

##### test_load_manifest_missing_file(self: Any)

Test error handling for missing manifest file.

##### test_load_manifest_invalid_toml(self: Any)

Test error handling for invalid TOML syntax.

##### test_load_manifest_unsupported_schema(self: Any)

Test error handling for unsupported schema version.

##### test_load_manifest_missing_required_fields(self: Any)

Test error handling for missing required fields.

##### test_load_manifest_invalid_model_id(self: Any)

Test error handling for invalid model ID format.

##### test_load_manifest_invalid_file_spec(self: Any)

Test error handling for invalid file specifications.

### TestPathSafetyValidation

Test path safety validation.

#### Methods

##### create_temp_manifest_with_paths(self: Any, file_paths: <ast.Subscript object at 0x0000019427BC2920>) -> str

Create a temporary manifest with specified file paths.

##### test_path_traversal_detection(self: Any)

Test detection of path traversal attempts.

##### test_absolute_path_detection(self: Any)

Test detection of absolute paths.

##### test_windows_reserved_name_detection(self: Any)

Test detection of Windows reserved names.

##### test_case_collision_detection(self: Any)

Test detection of case collisions.

### TestModelSpecRetrieval

Test model specification retrieval.

#### Methods

##### setup_method(self: Any)

Set up test registry with sample models.

##### teardown_method(self: Any)

Clean up test files.

##### test_get_model_spec_basic(self: Any)

Test basic model specification retrieval.

##### test_get_model_spec_with_variant(self: Any)

Test model specification retrieval with variant.

##### test_get_model_spec_nonexistent_model(self: Any)

Test error handling for nonexistent model.

##### test_get_model_spec_invalid_variant(self: Any)

Test error handling for invalid variant.

##### test_list_models(self: Any)

Test listing all available models.

