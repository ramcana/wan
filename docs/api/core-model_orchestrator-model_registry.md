---
title: core.model_orchestrator.model_registry
category: api
tags: [api, core]
---

# core.model_orchestrator.model_registry

Model Registry - Manifest parsing and model specification management.

This module handles loading and validating the models.toml manifest file,
providing typed access to model specifications with proper validation.

## Classes

### FileSpec

Specification for a single file within a model.

### VramEstimation

VRAM estimation parameters for a model.

### ModelDefaults

Default parameters for model inference.

### ModelSpec

Complete specification for a model including all variants and metadata.

### NormalizedModelId

Normalized model identifier components.

### ModelRegistry

Registry for managing model specifications from a TOML manifest.

Provides validation, normalization, and typed access to model definitions
with support for schema versioning and migration guidance.

#### Methods

##### __init__(self: Any, manifest_path: str)

Initialize the registry with a manifest file.

Args:
    manifest_path: Path to the models.toml manifest file
    
Raises:
    FileNotFoundError: If manifest file doesn't exist
    ManifestValidationError: If manifest is invalid
    SchemaVersionError: If schema version is unsupported

##### _load_manifest(self: Any) -> None

Load and parse the TOML manifest file.

##### _parse_model_spec(self: Any, model_key: str, model_data: dict) -> ModelSpec

Parse a single model specification from manifest data.

##### spec(self: Any, model_id: str, variant: <ast.Subscript object at 0x000001942792DDB0>) -> ModelSpec

Get the specification for a model.

Args:
    model_id: Model identifier (can include @version)
    variant: Optional variant override
    
Returns:
    ModelSpec for the requested model
    
Raises:
    ModelNotFoundError: If model is not found
    VariantNotFoundError: If variant is not available

##### list_models(self: Any) -> <ast.Subscript object at 0x000001942792FCD0>

Get a list of all available model IDs.

##### validate_manifest(self: Any) -> <ast.Subscript object at 0x000001942792F280>

Validate the loaded manifest for consistency and safety.

Returns:
    List of validation errors (empty if valid)

##### _validate_path_safety(self: Any, paths: <ast.Subscript object at 0x000001942792F130>) -> <ast.Subscript object at 0x000001942C516B00>

Validate paths for safety (no directory traversal).

##### _validate_case_collisions(self: Any, paths: <ast.Subscript object at 0x000001942C516C20>) -> <ast.Subscript object at 0x000001942CE1E260>

Validate for case collisions on case-insensitive file systems.

##### get_schema_version(self: Any) -> str

Get the schema version of the loaded manifest.

##### normalize_model_id(model_id: str, variant: <ast.Subscript object at 0x000001942CE1C9A0>) -> NormalizedModelId

Normalize a model ID into canonical components.

Args:
    model_id: Model identifier (may include @version and #variant)
    variant: Optional variant override
    
Returns:
    NormalizedModelId with separated components
    
Raises:
    InvalidModelIdError: If model ID format is invalid

## Constants

### SUPPORTED_SCHEMA_VERSIONS

Type: `unknown`

### MODEL_ID_PATTERN

Type: `unknown`

### VARIANT_PATTERN

Type: `unknown`

