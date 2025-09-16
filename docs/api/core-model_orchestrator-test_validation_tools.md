---
title: core.model_orchestrator.test_validation_tools
category: api
tags: [api, core]
---

# core.model_orchestrator.test_validation_tools

Tests for validation tools.

This module contains comprehensive tests for manifest validation,
security checks, performance analysis, and compatibility validation.

## Classes

### TestValidationIssue

Test ValidationIssue functionality.

#### Methods

##### test_validation_issue_creation(self: Any)

Test creating validation issues.

##### test_validation_issue_string_representation(self: Any)

Test string representation of validation issues.

### TestValidationReport

Test ValidationReport functionality.

#### Methods

##### test_empty_validation_report(self: Any)

Test empty validation report.

##### test_validation_report_with_issues(self: Any)

Test validation report with various issues.

##### test_add_issue_to_report(self: Any)

Test adding issues to report.

##### test_filter_issues_by_severity(self: Any)

Test filtering issues by severity.

##### test_filter_issues_by_category(self: Any)

Test filtering issues by category.

### TestManifestSchemaValidator

Test ManifestSchemaValidator functionality.

#### Methods

##### setup_method(self: Any)

Set up test fixtures.

##### teardown_method(self: Any)

Clean up test fixtures.

##### create_valid_manifest(self: Any) -> str

Create a valid test manifest.

##### create_invalid_manifest(self: Any) -> str

Create an invalid test manifest.

##### test_validate_valid_manifest(self: Any)

Test validation of valid manifest.

##### test_validate_invalid_manifest(self: Any)

Test validation of invalid manifest.

##### test_validate_missing_required_fields(self: Any)

Test validation of manifest with missing required fields.

##### test_validate_invalid_model_id_format(self: Any)

Test validation of invalid model ID format.

##### test_validate_file_security_issues(self: Any)

Test validation of file security issues.

### TestSecurityValidator

Test SecurityValidator functionality.

#### Methods

##### setup_method(self: Any)

Set up test fixtures.

##### teardown_method(self: Any)

Clean up test fixtures.

##### create_manifest_with_security_issues(self: Any) -> str

Create manifest with various security issues.

##### test_validate_security_issues(self: Any)

Test detection of various security issues.

### TestPerformanceValidator

Test PerformanceValidator functionality.

#### Methods

##### setup_method(self: Any)

Set up test fixtures.

##### teardown_method(self: Any)

Clean up test fixtures.

##### create_large_model_manifest(self: Any) -> str

Create manifest with performance issues.

##### test_validate_performance_issues(self: Any)

Test detection of performance issues.

### TestCompatibilityValidator

Test CompatibilityValidator functionality.

#### Methods

##### setup_method(self: Any)

Set up test fixtures.

##### teardown_method(self: Any)

Clean up test fixtures.

##### create_compatibility_issues_manifest(self: Any) -> str

Create manifest with compatibility issues.

##### test_validate_compatibility_issues(self: Any)

Test detection of compatibility issues.

### TestComprehensiveValidator

Test ComprehensiveValidator functionality.

#### Methods

##### setup_method(self: Any)

Set up test fixtures.

##### teardown_method(self: Any)

Clean up test fixtures.

##### create_comprehensive_test_manifest(self: Any) -> str

Create manifest with various types of issues.

##### test_comprehensive_validation(self: Any)

Test comprehensive validation with all validators.

##### test_selective_validation(self: Any)

Test running only specific validation types.

##### test_file_integrity_validation(self: Any)

Test file integrity validation.

##### test_nonexistent_file_integrity(self: Any)

Test integrity validation of nonexistent file.

