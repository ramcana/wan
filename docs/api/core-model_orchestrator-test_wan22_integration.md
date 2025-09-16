---
title: core.model_orchestrator.test_wan22_integration
category: api
tags: [api, core]
---

# core.model_orchestrator.test_wan22_integration

Integration tests for WAN2.2 model handling with the orchestrator.

Tests the complete integration between WAN2.2 handler and model orchestrator,
including selective downloading, variant switching, and component validation.

## Classes

### MockStorageBackend

Mock storage backend for testing.

#### Methods

##### __init__(self: Any)



##### can_handle(self: Any, source_url: str) -> bool



##### download(self: Any, source_url: str, local_dir: Path, file_specs: list, allow_patterns: Any, progress_callback: Any) -> DownloadResult



##### verify_availability(self: Any, source_url: str) -> bool



##### estimate_download_size(self: Any, source_url: str, file_specs: list) -> int



### TestWAN22Integration

Test WAN2.2 integration with model orchestrator.

#### Methods

##### temp_models_root(self: Any)

Create temporary models root directory.

##### sample_manifest(self: Any, temp_models_root: Any)

Create sample manifest file.

##### model_ensurer(self: Any, temp_models_root: Any, sample_manifest: Any)

Create model ensurer with WAN2.2 support.

##### test_wan22_handler_creation(self: Any, model_ensurer: Any)

Test WAN2.2 handler creation for different model types.

##### test_selective_file_download_production(self: Any, model_ensurer: Any)

Test selective file download for production variants.

##### test_selective_file_download_development(self: Any, model_ensurer: Any)

Test selective file download for development variants.

##### test_full_download_workflow_production(self: Any, mock_logger: Any, model_ensurer: Any, temp_models_root: Any)

Test complete download workflow for production variant.

##### test_full_download_workflow_development(self: Any, mock_logger: Any, model_ensurer: Any, temp_models_root: Any)

Test complete download workflow for development variant.

##### test_model_status_with_wan22(self: Any, model_ensurer: Any)

Test model status checking with WAN2.2 models.

##### test_input_validation_integration(self: Any, model_ensurer: Any, temp_models_root: Any)

Test input validation integration with model ensurer.

##### test_vae_config_integration(self: Any, model_ensurer: Any)

Test VAE configuration integration.

##### test_memory_estimation_integration(self: Any, model_ensurer: Any)

Test memory estimation integration.

##### test_text_embedding_cache_integration(self: Any, model_ensurer: Any)

Test text embedding cache integration.

##### test_component_validation_integration(self: Any, model_ensurer: Any)

Test component validation integration.

##### test_variant_conversion_integration(self: Any, model_ensurer: Any)

Test variant conversion integration.

##### test_error_handling_integration(self: Any, model_ensurer: Any)

Test error handling integration with WAN2.2 models.

