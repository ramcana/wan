---
title: core.model_orchestrator.validation_tools
category: api
tags: [api, core]
---

# core.model_orchestrator.validation_tools

Validation Tools - Comprehensive validation for manifests and configurations.

This module provides advanced validation tools for model manifests,
configuration files, and system compatibility checks.

## Classes

### ValidationIssue

Represents a validation issue with severity and context.

#### Methods

##### __str__(self: Any) -> str

String representation of validation issue.

### ValidationReport

Comprehensive validation report.

#### Methods

##### __post_init__(self: Any)

Calculate summary statistics.

##### add_issue(self: Any, issue: ValidationIssue) -> None

Add a validation issue to the report.

##### get_issues_by_severity(self: Any, severity: str) -> <ast.Subscript object at 0x000001942F018F40>

Get issues filtered by severity.

##### get_issues_by_category(self: Any, category: str) -> <ast.Subscript object at 0x000001942F018B50>

Get issues filtered by category.

### ManifestSchemaValidator

Validator for manifest schema and structure.

#### Methods

##### __init__(self: Any)

Initialize the schema validator.

##### validate_manifest_schema(self: Any, manifest_path: str) -> ValidationReport

Validate manifest schema and structure.

Args:
    manifest_path: Path to manifest file
    
Returns:
    ValidationReport with schema validation results

##### _validate_top_level_structure(self: Any, data: <ast.Subscript object at 0x000001942EFA23B0>, report: ValidationReport) -> None

Validate top-level manifest structure.

##### _validate_schema_version(self: Any, data: <ast.Subscript object at 0x000001942EFA1BA0>, report: ValidationReport) -> None

Validate schema version.

##### _validate_models_section(self: Any, models: <ast.Subscript object at 0x000001942EFA0DF0>, report: ValidationReport) -> None

Validate models section.

##### _validate_model_entry(self: Any, model_id: str, model_data: <ast.Subscript object at 0x000001942EFA04F0>, report: ValidationReport) -> None

Validate individual model entry.

##### _validate_files_section(self: Any, files: <ast.Subscript object at 0x00000194300A9CC0>, report: ValidationReport, context: str) -> None

Validate files section.

##### _validate_sources_section(self: Any, sources: <ast.Subscript object at 0x000001943459DE10>, report: ValidationReport, context: str) -> None

Validate sources section.

### SecurityValidator

Validator for security-related issues in manifests.

#### Methods

##### __init__(self: Any)

Initialize the security validator.

##### validate_security(self: Any, manifest_path: str) -> ValidationReport

Validate security aspects of manifest.

Args:
    manifest_path: Path to manifest file
    
Returns:
    ValidationReport with security validation results

##### _validate_model_security(self: Any, model_spec: ModelSpec, report: ValidationReport) -> None

Validate security aspects of a model specification.

##### _validate_file_path_security(self: Any, file_spec: FileSpec, report: ValidationReport, context: str) -> None

Validate file path for security issues.

##### _validate_file_patterns(self: Any, model_spec: ModelSpec, report: ValidationReport, context: str) -> None

Validate file patterns for suspicious content.

##### _validate_source_urls(self: Any, model_spec: ModelSpec, report: ValidationReport, context: str) -> None

Validate source URLs for security issues.

### PerformanceValidator

Validator for performance-related issues in manifests.

#### Methods

##### __init__(self: Any)

Initialize the performance validator.

##### validate_performance(self: Any, manifest_path: str) -> ValidationReport

Validate performance aspects of manifest.

Args:
    manifest_path: Path to manifest file
    
Returns:
    ValidationReport with performance validation results

##### _validate_model_performance(self: Any, model_spec: ModelSpec, report: ValidationReport) -> None

Validate performance aspects of a model specification.

### CompatibilityValidator

Validator for compatibility issues across platforms and configurations.

#### Methods

##### __init__(self: Any)

Initialize the compatibility validator.

##### validate_compatibility(self: Any, manifest_path: str) -> ValidationReport

Validate compatibility aspects of manifest.

Args:
    manifest_path: Path to manifest file
    
Returns:
    ValidationReport with compatibility validation results

##### _validate_case_sensitivity(self: Any, registry: ModelRegistry, report: ValidationReport) -> None

Validate for case sensitivity issues.

##### _validate_path_lengths(self: Any, registry: ModelRegistry, report: ValidationReport) -> None

Validate path lengths for Windows compatibility.

##### _validate_platform_compatibility(self: Any, registry: ModelRegistry, report: ValidationReport) -> None

Validate platform-specific compatibility issues.

### ComprehensiveValidator

Comprehensive validator that runs all validation checks.

#### Methods

##### __init__(self: Any)

Initialize the comprehensive validator.

##### validate_manifest(self: Any, manifest_path: str, include_schema: bool, include_security: bool, include_performance: bool, include_compatibility: bool) -> ValidationReport

Run comprehensive validation on a manifest file.

Args:
    manifest_path: Path to manifest file
    include_schema: Whether to include schema validation
    include_security: Whether to include security validation
    include_performance: Whether to include performance validation
    include_compatibility: Whether to include compatibility validation
    
Returns:
    Combined ValidationReport with all validation results

##### validate_file_integrity(self: Any, file_path: str, expected_sha256: str, expected_size: int) -> ValidationReport

Validate file integrity against expected values.

Args:
    file_path: Path to file to validate
    expected_sha256: Expected SHA256 hash
    expected_size: Expected file size
    
Returns:
    ValidationReport with integrity validation results

## Constants

### SUPPORTED_SCHEMA_VERSIONS

Type: `unknown`

### REQUIRED_MODEL_FIELDS

Type: `unknown`

### REQUIRED_FILE_FIELDS

Type: `unknown`

### WINDOWS_RESERVED_NAMES

Type: `unknown`

