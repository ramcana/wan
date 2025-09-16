---
title: core.model_orchestrator.validate_models_toml
category: api
tags: [api, core]
---

# core.model_orchestrator.validate_models_toml

Models.toml Validator

Validates the models.toml manifest file for:
- Schema version compatibility
- No duplicate model IDs or file paths
- No path traversal vulnerabilities
- Windows case sensitivity compatibility
- Proper TOML structure and required fields

## Classes

### ModelsTomlValidator

Comprehensive validator for models.toml manifest files.

#### Methods

##### __init__(self: Any, manifest_path: str)

Initialize validator with manifest path.

##### validate(self: Any) -> <ast.Subscript object at 0x0000019434171420>

Perform comprehensive validation of the models.toml file.

Returns:
    Tuple of (is_valid, errors, warnings)

##### _validate_schema_version(self: Any, data: dict) -> None

Validate schema version is present and supported.

##### _validate_models_structure(self: Any, data: dict) -> None

Validate the models section structure.

##### _validate_model_entry(self: Any, model_id: str, model_data: dict) -> None

Validate a single model entry.

##### _validate_files_section(self: Any, model_id: str, files: <ast.Subscript object at 0x0000019434225240>) -> None

Validate the files section of a model.

##### _validate_no_duplicates(self: Any, data: dict) -> None

Validate there are no duplicate model IDs or file paths within models.

##### _validate_path_safety(self: Any, data: dict) -> None

Validate paths for safety (no directory traversal).

##### _validate_windows_compatibility(self: Any, data: dict) -> None

Validate Windows case sensitivity compatibility.

##### _validate_with_model_registry(self: Any) -> None

Validate using the ModelRegistry class for additional checks.

## Constants

### SUPPORTED_SCHEMA_VERSIONS

Type: `unknown`

### WINDOWS_RESERVED_NAMES

Type: `unknown`

