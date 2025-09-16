---
title: core.model_orchestrator.test_models_toml_validator
category: api
tags: [api, core]
---

# core.model_orchestrator.test_models_toml_validator

Test suite for the models.toml validator.

Tests various validation scenarios including:
- Schema version issues
- Duplicate detection
- Path traversal vulnerabilities
- Windows case sensitivity issues
- Malformed TOML structures

## Classes

### TestModelsTomlValidator

Test cases for the ModelsTomlValidator.

#### Methods

##### setUp(self: Any)

Set up test fixtures.

##### tearDown(self: Any)

Clean up test fixtures.

##### create_test_manifest(self: Any, content: str) -> str

Create a temporary manifest file with the given content.

##### test_valid_manifest(self: Any)

Test validation of a valid manifest.

##### test_missing_schema_version(self: Any)

Test detection of missing schema version.

##### test_unsupported_schema_version(self: Any)

Test detection of unsupported schema version.

##### test_invalid_model_id_format(self: Any)

Test detection of invalid model ID formats.

##### test_duplicate_file_paths(self: Any)

Test detection of duplicate file paths within a model.

##### test_path_traversal_detection(self: Any)

Test detection of path traversal vulnerabilities.

##### test_absolute_path_detection(self: Any)

Test detection of absolute paths.

##### test_windows_reserved_names(self: Any)

Test detection of Windows reserved names.

##### test_case_collision_detection(self: Any)

Test detection of case collisions.

##### test_missing_required_fields(self: Any)

Test detection of missing required fields.

##### test_invalid_sha256_length(self: Any)

Test detection of invalid SHA256 length.

##### test_invalid_file_size(self: Any)

Test detection of invalid file sizes.

##### test_default_variant_not_in_variants(self: Any)

Test detection when default_variant is not in variants list.

##### test_malformed_toml(self: Any)

Test handling of malformed TOML.

##### test_nonexistent_file(self: Any)

Test handling of nonexistent manifest file.

##### test_long_path_warning(self: Any)

Test warning for very long paths.

##### test_trailing_space_warning(self: Any)

Test warning for paths with trailing spaces.

