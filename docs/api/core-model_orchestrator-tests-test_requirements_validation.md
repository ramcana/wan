---
title: core.model_orchestrator.tests.test_requirements_validation
category: api
tags: [api, core]
---

# core.model_orchestrator.tests.test_requirements_validation

Requirements validation tests for Model Orchestrator.

This module validates that all requirements from the specification
are properly implemented and tested.

## Classes

### TestRequirement1_UnifiedModelManifest

Test Requirement 1: Unified Model Manifest System with Versioning.

#### Methods

##### test_1_1_single_manifest_loading(self: Any)

WHEN the system initializes THEN it SHALL load model definitions from a single models.toml manifest file.

##### test_1_2_canonical_model_id_resolution(self: Any)

WHEN a model is referenced by ID THEN the system SHALL resolve it using canonical model_id format.

##### test_1_3_variant_support(self: Any)

WHEN the manifest defines model variants THEN the system SHALL support variant-specific resolution.

##### test_1_6_schema_version_compatibility(self: Any)

WHEN the manifest schema version is incompatible THEN the system SHALL fail gracefully.

##### test_1_8_model_not_found_error(self: Any)

IF a model ID or variant is not found THEN the system SHALL raise a clear error.

### TestRequirement3_DeterministicPathResolution

Test Requirement 3: Deterministic Path Resolution.

#### Methods

##### test_3_1_models_root_based_resolution(self: Any)

WHEN the system resolves a model path THEN it SHALL use only the configured MODELS_ROOT and model ID.

##### test_3_2_identical_paths_for_same_model(self: Any)

WHEN multiple services request the same model THEN they SHALL receive identical absolute paths.

##### test_3_4_path_pattern_compliance(self: Any)

WHEN a model directory is created THEN it SHALL follow the pattern {MODELS_ROOT}/wan22/{model_id}.

##### test_3_5_missing_models_root_error(self: Any)

IF MODELS_ROOT is not configured THEN the system SHALL raise a configuration error.

### TestRequirement4_AtomicDownloads

Test Requirement 4: Atomic Downloads with Concurrency Safety.

#### Methods

##### orchestrator_setup(self: Any)

Set up orchestrator components for testing.

##### test_4_1_temporary_directory_download(self: Any, orchestrator_setup: Any)

WHEN a model is requested for the first time THEN the system SHALL download to a temporary directory.

##### test_4_2_atomic_rename_operation(self: Any, orchestrator_setup: Any)

WHEN a download completes successfully THEN the system SHALL atomically rename from temporary to final location.

##### test_4_3_concurrent_download_safety(self: Any, orchestrator_setup: Any)

WHEN multiple processes request the same model THEN only one SHALL download while others wait.

### TestRequirement5_IntegrityVerification

Test Requirement 5: Comprehensive Integrity and Trust Chain.

#### Methods

##### test_5_1_checksum_verification(self: Any)

WHEN per-file checksums are provided THEN the system SHALL verify both after download.

##### test_5_2_checksum_failure_handling(self: Any)

WHEN checksum verification fails THEN the system SHALL re-download the affected files.

##### test_5_7_basic_completeness_checks(self: Any)

IF verification data is missing THEN the system SHALL perform basic completeness checks.

### TestRequirement10_DiskSpaceManagement

Test Requirement 10: Disk Space Management and Garbage Collection.

#### Methods

##### test_10_1_preflight_space_checks(self: Any)

WHEN a download is requested THEN the system SHALL perform preflight free-space checks.

##### test_10_3_lru_garbage_collection(self: Any)

WHEN storage quota is exceeded THEN the system SHALL trigger LRU/TTL-based garbage collection.

##### test_10_4_model_pinning(self: Any)

WHEN models are marked as pinned THEN garbage collection SHALL preserve them.

### TestRequirement12_Observability

Test Requirement 12: Comprehensive Observability and Error Classification.

#### Methods

##### test_12_1_download_metrics(self: Any)

WHEN download operations occur THEN the system SHALL emit metrics.

##### test_12_2_error_classification(self: Any)

WHEN errors occur THEN the system SHALL classify them with specific error codes.

### TestRequirement13_ProductionAPI

Test Requirement 13: Production API Surface and CLI Tools.

#### Methods

##### test_13_1_ensure_api(self: Any)

WHEN using Python API THEN ensure(model_id, variant=None) SHALL return Path to ready model directory.

##### test_13_2_status_api(self: Any)

WHEN querying model status THEN status(model_id) SHALL return structured data.

