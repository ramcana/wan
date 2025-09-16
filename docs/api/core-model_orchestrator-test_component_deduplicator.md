---
title: core.model_orchestrator.test_component_deduplicator
category: api
tags: [api, core]
---

# core.model_orchestrator.test_component_deduplicator

Tests for Component Deduplication System.

Tests component sharing across t2v/i2v/ti2v models with various scenarios
including hardlink/symlink creation, reference tracking, and cleanup.

## Classes

### TestComponentDeduplicator

Test suite for ComponentDeduplicator.

#### Methods

##### temp_models_root(self: Any)

Create a temporary models root directory.

##### deduplicator(self: Any, temp_models_root: Any)

Create a ComponentDeduplicator instance.

##### sample_models(self: Any, temp_models_root: Any)

Create sample model directories with common components.

##### test_initialization(self: Any, temp_models_root: Any)

Test ComponentDeduplicator initialization.

##### test_hardlink_support_detection(self: Any, deduplicator: Any, temp_models_root: Any)

Test hardlink support detection.

##### test_symlink_support_detection(self: Any, deduplicator: Any, temp_models_root: Any)

Test symlink support detection.

##### test_component_type_identification(self: Any, deduplicator: Any)

Test component type identification from file paths.

##### test_file_hash_calculation(self: Any, deduplicator: Any, temp_models_root: Any)

Test file hash calculation.

##### test_single_model_deduplication(self: Any, deduplicator: Any, sample_models: Any)

Test deduplication within a single model.

##### test_cross_model_deduplication(self: Any, deduplicator: Any, sample_models: Any)

Test deduplication across multiple models.

##### test_component_metadata_persistence(self: Any, deduplicator: Any, sample_models: Any)

Test that component metadata is properly saved and loaded.

##### test_reference_tracking(self: Any, deduplicator: Any, sample_models: Any)

Test component reference tracking.

##### test_orphaned_component_cleanup(self: Any, deduplicator: Any, sample_models: Any)

Test cleanup of orphaned components.

##### test_component_stats(self: Any, deduplicator: Any, sample_models: Any)

Test component statistics reporting.

##### test_link_creation_fallback(self: Any, deduplicator: Any, temp_models_root: Any)

Test link creation with various fallback strategies.

##### test_cross_platform_compatibility(self: Any, deduplicator: Any)

Test cross-platform compatibility features.

##### test_error_handling(self: Any, deduplicator: Any, temp_models_root: Any)

Test error handling in various scenarios.

##### test_concurrent_access_safety(self: Any, deduplicator: Any, sample_models: Any)

Test thread safety of metadata operations.

##### test_model_ensurer_integration(self: Any, temp_models_root: Any, sample_models: Any)

Test integration with ModelEnsurer for automatic deduplication during atomic move.

##### test_wan22_specific_components(self: Any, deduplicator: Any, sample_models: Any)

Test handling of WAN2.2-specific model components.

##### test_large_file_handling(self: Any, deduplicator: Any, temp_models_root: Any)

Test handling of large files during deduplication.

##### test_component_versioning(self: Any, deduplicator: Any, temp_models_root: Any)

Test component versioning based on content hash.

##### test_unix_hardlink_creation(self: Any, deduplicator: Any, temp_models_root: Any)

Test hardlink creation on Unix systems.

##### test_windows_junction_creation(self: Any, deduplicator: Any, temp_models_root: Any)

Test Windows junction creation for directories.

