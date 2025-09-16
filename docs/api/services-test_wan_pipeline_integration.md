---
title: services.test_wan_pipeline_integration
category: api
tags: [api, services]
---

# services.test_wan_pipeline_integration

Integration tests for WAN Pipeline Integration with Model Orchestrator.

Tests the integration between the Model Orchestrator and WAN pipeline loading,
including component validation, VRAM estimation, and model-specific handling.

## Classes

### TestWanPipelineIntegration

Test WAN Pipeline Integration functionality.

#### Methods

##### mock_model_ensurer(self: Any)

Create a mock model ensurer.

##### mock_model_registry(self: Any)

Create a mock model registry.

##### wan_integration(self: Any, mock_model_ensurer: Any, mock_model_registry: Any)

Create WAN pipeline integration instance.

##### temp_model_dir(self: Any)

Create a temporary model directory with test files.

##### test_get_wan_paths_success(self: Any, wan_integration: Any, mock_model_ensurer: Any)

Test successful model path retrieval.

##### test_get_wan_paths_failure(self: Any, wan_integration: Any, mock_model_ensurer: Any)

Test model path retrieval failure.

##### test_get_pipeline_class_t2v(self: Any, wan_integration: Any)

Test pipeline class retrieval for T2V model.

##### test_get_pipeline_class_i2v(self: Any, wan_integration: Any)

Test pipeline class retrieval for I2V model.

##### test_get_pipeline_class_ti2v(self: Any, wan_integration: Any)

Test pipeline class retrieval for TI2V model.

##### test_get_pipeline_class_unknown(self: Any, wan_integration: Any)

Test pipeline class retrieval for unknown model.

##### test_validate_components_success(self: Any, wan_integration: Any, temp_model_dir: Any)

Test successful component validation.

##### test_validate_components_missing_model_index(self: Any, wan_integration: Any)

Test component validation with missing model_index.json.

##### test_validate_components_invalid_for_t2v(self: Any, wan_integration: Any)

Test component validation with invalid components for T2V model.

##### test_validate_components_missing_required(self: Any, wan_integration: Any)

Test component validation with missing required components.

##### test_estimate_vram_usage_t2v(self: Any, wan_integration: Any)

Test VRAM estimation for T2V model.

##### test_estimate_vram_usage_ti2v(self: Any, wan_integration: Any)

Test VRAM estimation for TI2V model.

##### test_estimate_vram_usage_large_generation(self: Any, wan_integration: Any)

Test VRAM estimation for large generation parameters.

##### test_get_model_capabilities_t2v(self: Any, wan_integration: Any)

Test getting model capabilities for T2V model.

##### test_get_model_capabilities_i2v(self: Any, wan_integration: Any)

Test getting model capabilities for I2V model.

##### test_get_model_capabilities_ti2v(self: Any, wan_integration: Any)

Test getting model capabilities for TI2V model.

##### test_get_model_capabilities_unknown(self: Any, wan_integration: Any)

Test getting model capabilities for unknown model.

##### test_extract_model_base(self: Any, wan_integration: Any)

Test model base extraction from full model ID.

### TestGlobalFunctions

Test global functions for WAN pipeline integration.

#### Methods

##### test_initialize_wan_integration(self: Any)

Test initialization of global WAN integration.

##### test_get_wan_paths_global(self: Any)

Test global get_wan_paths function.

##### test_get_wan_integration_not_initialized(self: Any)

Test getting WAN integration when not initialized.

### TestWanModelSpecs

Test WAN model specifications.

#### Methods

##### test_wan_pipeline_mappings_completeness(self: Any)

Test that all expected WAN models are mapped.

##### test_t2v_model_spec(self: Any)

Test T2V model specification.

##### test_i2v_model_spec(self: Any)

Test I2V model specification.

##### test_ti2v_model_spec(self: Any)

Test TI2V model specification.

##### test_vram_estimations(self: Any)

Test VRAM estimations are reasonable.

##### test_resolution_caps(self: Any)

Test resolution capabilities.

### TestWanPipelineLoaderIntegration

Integration tests for WAN pipeline loader with Model Orchestrator.

#### Methods

##### mock_wan_pipeline_loader(self: Any)

Create a mock WAN pipeline loader.

##### setup_integration(self: Any)

Set up integration environment.

##### test_load_wan_pipeline_with_orchestrator(self: Any, mock_wan_pipeline_loader: Any, setup_integration: Any)

Test loading WAN pipeline with Model Orchestrator integration.

##### test_component_validation_before_gpu_init(self: Any, setup_integration: Any)

Test that component validation happens before GPU initialization.

##### test_vram_estimation_integration(self: Any, setup_integration: Any)

Test VRAM estimation integration with pipeline loading.

