---
title: core.model_orchestrator.test_garbage_collector
category: api
tags: [api, core]
---

# core.model_orchestrator.test_garbage_collector

Tests for the Garbage Collector.

## Classes

### TestGarbageCollector

Test cases for the GarbageCollector class.

#### Methods

##### test_init(self: Any, mock_registry: Any, mock_resolver: Any, gc_config: Any)

Test garbage collector initialization.

##### test_pin_unpin_model(self: Any, garbage_collector: Any)

Test pinning and unpinning models.

##### test_pin_model_without_variant(self: Any, garbage_collector: Any)

Test pinning models without variants.

##### test_get_disk_usage(self: Any, mock_disk_usage: Any, garbage_collector: Any, temp_models_dir: Any)

Test disk usage calculation.

##### test_discover_models(self: Any, garbage_collector: Any, sample_models: Any)

Test model discovery.

##### test_auto_pin_recent_models(self: Any, garbage_collector: Any, sample_models: Any)

Test automatic pinning of recently accessed models.

##### test_select_removal_candidates_by_age(self: Any, garbage_collector: Any, sample_models: Any)

Test selection of removal candidates based on age.

##### test_select_removal_candidates_by_size(self: Any, garbage_collector: Any, sample_models: Any)

Test selection of removal candidates based on total size.

##### test_select_removal_candidates_respects_pinned(self: Any, garbage_collector: Any, sample_models: Any)

Test that pinned models are not selected for removal.

##### test_select_removal_candidates_by_disk_space(self: Any, mock_disk_usage: Any, garbage_collector: Any, sample_models: Any)

Test selection based on low disk space.

##### test_dry_run_collection(self: Any, garbage_collector: Any, sample_models: Any)

Test dry run garbage collection.

##### test_actual_collection(self: Any, garbage_collector: Any, sample_models: Any)

Test actual garbage collection (not dry run).

##### test_estimate_reclaimable_space(self: Any, garbage_collector: Any, sample_models: Any)

Test estimation of reclaimable space.

##### test_should_trigger_gc_low_disk_space(self: Any, mock_disk_usage: Any, garbage_collector: Any)

Test automatic GC trigger for low disk space.

##### test_should_trigger_gc_quota_exceeded(self: Any, garbage_collector: Any, sample_models: Any)

Test automatic GC trigger for quota exceeded.

##### test_should_not_trigger_gc_when_disabled(self: Any, garbage_collector: Any, sample_models: Any)

Test that GC doesn't trigger when disabled.

##### test_pin_file_persistence(self: Any, garbage_collector: Any, temp_models_dir: Any)

Test that pinned models persist across instances.

##### test_calculate_directory_size(self: Any, garbage_collector: Any, temp_models_dir: Any)

Test directory size calculation.

##### test_get_last_access_time_from_verification_file(self: Any, garbage_collector: Any, temp_models_dir: Any)

Test getting last access time from verification file.

##### test_get_last_access_time_fallback(self: Any, garbage_collector: Any, temp_models_dir: Any)

Test fallback to directory modification time.

##### test_error_handling_in_collection(self: Any, garbage_collector: Any, sample_models: Any)

Test error handling during garbage collection.

##### test_model_key_generation(self: Any, garbage_collector: Any)

Test model key generation for different scenarios.

### TestGCConfig

Test cases for GCConfig.

#### Methods

##### test_default_config(self: Any)

Test default configuration values.

##### test_custom_config(self: Any)

Test custom configuration values.

### TestModelInfo

Test cases for ModelInfo dataclass.

#### Methods

##### test_model_info_creation(self: Any)

Test ModelInfo creation and attributes.

##### test_model_info_defaults(self: Any)

Test ModelInfo default values.

### TestDiskUsage

Test cases for DiskUsage dataclass.

#### Methods

##### test_disk_usage_creation(self: Any)

Test DiskUsage creation and calculations.

### TestGCResult

Test cases for GCResult dataclass.

#### Methods

##### test_gc_result_creation(self: Any)

Test GCResult creation and default values.

##### test_gc_result_with_data(self: Any)

Test GCResult with actual data.

