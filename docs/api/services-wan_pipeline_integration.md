---
title: services.wan_pipeline_integration
category: api
tags: [api, services]
---

# services.wan_pipeline_integration

WAN Pipeline Integration with Model Orchestrator.

This module provides the integration layer between the Model Orchestrator
and WAN pipeline loading, including model-specific handling and component validation.

## Classes

### WanModelType

WAN model types with their characteristics.

### WanModelSpec

Specification for a WAN model type.

### ComponentValidationResult

Result of component validation.

### WanPipelineIntegration

Integration layer between Model Orchestrator and WAN pipeline loading.

#### Methods

##### __init__(self: Any, model_ensurer: ModelEnsurer, model_registry: ModelRegistry)

Initialize the WAN pipeline integration.

Args:
    model_ensurer: Model ensurer for downloading and managing models
    model_registry: Model registry for model specifications

##### get_wan_paths(self: Any, model_id: str, variant: <ast.Subscript object at 0x00000194285AEFE0>) -> str

Get the local path for a WAN model, ensuring it's downloaded.

This is the main integration point that replaces hardcoded paths
in the pipeline loader.

Args:
    model_id: Model identifier (e.g., "t2v-A14B@2.2.0")
    variant: Optional variant (e.g., "fp16", "bf16")
    
Returns:
    Absolute path to the ready-to-use model directory
    
Raises:
    ModelNotFoundError: If model is not found in registry
    ModelOrchestratorError: If model cannot be ensured

##### get_pipeline_class(self: Any, model_id: str) -> str

Get the appropriate pipeline class for a model.

Args:
    model_id: Model identifier
    
Returns:
    Pipeline class name
    
Raises:
    ModelNotFoundError: If model type is not recognized

##### validate_components(self: Any, model_id: str, model_path: str) -> ComponentValidationResult

Validate that all required components are present before GPU initialization.

Args:
    model_id: Model identifier
    model_path: Path to the model directory
    
Returns:
    ComponentValidationResult with validation details

##### estimate_vram_usage(self: Any, model_id: str) -> float

Estimate VRAM usage for a model with given generation parameters.

Args:
    model_id: Model identifier
    **generation_params: Generation parameters (num_frames, width, height, etc.)
    
Returns:
    Estimated VRAM usage in GB

##### get_model_capabilities(self: Any, model_id: str) -> <ast.Subscript object at 0x000001942C844070>

Get model capabilities and constraints.

Args:
    model_id: Model identifier
    
Returns:
    Dictionary with model capabilities

##### _extract_model_base(self: Any, model_id: str) -> str

Extract the base model name from a full model ID.

## Constants

### WAN_PIPELINE_MAPPINGS

Type: `unknown`

### T2V

Type: `str`

Value: `t2v`

### I2V

Type: `str`

Value: `i2v`

### TI2V

Type: `str`

Value: `ti2v`

