---
title: services.real_generation_pipeline
category: api
tags: [api, services]
---

# services.real_generation_pipeline

Real Generation Pipeline
Integrates with existing WanPipelineLoader infrastructure for actual video generation

## Classes

### GenerationStage

Stages of video generation process

### ProgressUpdate

Progress update information

### RealGenerationPipeline

Real generation pipeline using existing WanPipelineLoader infrastructure
Handles T2V, I2V, and TI2V generation with progress tracking and WebSocket updates

#### Methods

##### __init__(self: Any, wan_pipeline_loader: Any, websocket_manager: Any)

Initialize the real generation pipeline

Args:
    wan_pipeline_loader: WanPipelineLoader instance (will be initialized if None)
    websocket_manager: WebSocket manager for progress updates

##### _create_simple_pipeline_wrapper(self: Any, pipeline: Any, model_type: str)

Create a simple wrapper for pipelines loaded with simplified loader

##### _get_model_path(self: Any, model_type: str) -> <ast.Subscript object at 0x000001942CD27AC0>

Get model path for a given model type

##### _create_optimization_config(self: Any, params: GenerationParams) -> <ast.Subscript object at 0x000001942CD25840>

Create optimization configuration from generation parameters

##### _create_generation_config(self: Any, prompt: str, params: GenerationParams)

Create generation configuration from parameters

##### _parse_resolution(self: Any, resolution: str) -> <ast.Subscript object at 0x000001942C52C490>

Parse resolution string to width, height tuple

##### _get_basic_lora_fallback(self: Any, base_prompt: str, lora_name: str) -> str

Basic LoRA fallback prompt enhancement

##### get_lora_status(self: Any) -> <ast.Subscript object at 0x000001942B36A170>

Get current LoRA status information

##### _map_to_progress_stage(self: Any, stage: GenerationStage)

Map pipeline GenerationStage to progress integration GenerationStage

##### _validate_t2v_params(self: Any, prompt: str, params: GenerationParams) -> <ast.Subscript object at 0x00000194284B3610>

Validate parameters for T2V generation

##### _validate_i2v_params(self: Any, image_path: str, prompt: str, params: GenerationParams) -> <ast.Subscript object at 0x00000194284B3E80>

Validate parameters for I2V generation

##### _validate_ti2v_params(self: Any, image_path: str, prompt: str, params: GenerationParams) -> <ast.Subscript object at 0x000001942A11DCF0>

Validate parameters for TI2V generation

##### _validate_lora_params(self: Any, params: GenerationParams) -> <ast.Subscript object at 0x000001942A11FD60>

Validate LoRA parameters

##### _create_error_result(self: Any, task_id: str, error_category: str, error_message: str) -> GenerationResult

Create error result with recovery suggestions

##### setup_progress_callbacks(self: Any, websocket_manager: Any)

Setup progress callbacks for WebSocket updates

##### get_generation_stats(self: Any) -> <ast.Subscript object at 0x0000019428507520>

Get generation statistics

##### clear_pipeline_cache(self: Any)

Clear the pipeline cache to free memory

##### set_hardware_optimizer(self: Any, optimizer: Any)

Set the hardware optimizer for pipeline optimization

##### _extract_model_from_pipeline(self: Any, pipeline_wrapper: Any)

Extract the actual model from pipeline wrapper for LoRA application

Args:
    pipeline_wrapper: Pipeline wrapper instance
    
Returns:
    Model instance or None if not found

##### _get_pipeline_id(self: Any, pipeline_wrapper: Any) -> str

Get unique identifier for pipeline wrapper

##### get_applied_loras_status(self: Any, pipeline_wrapper: Any) -> <ast.Subscript object at 0x00000194284A6710>

Get status of applied LoRAs

Args:
    pipeline_wrapper: Optional specific pipeline to check
    
Returns:
    Dictionary with LoRA status information

### SimplePipelineWrapper



#### Methods

##### __init__(self: Any, pipeline: Any, model_type: Any)



##### generate(self: Any, config: Any)

Generate using the wrapped pipeline

### FallbackGenerationConfig



#### Methods

##### __init__(self: Any)



### SimpleGenerationResult



#### Methods

##### __post_init__(self: Any)



### SimpleGenerationResult



#### Methods

##### __post_init__(self: Any)



## Constants

### LORA_MANAGER_AVAILABLE

Type: `bool`

Value: `True`

### INITIALIZING

Type: `str`

Value: `initializing`

### LOADING_MODEL

Type: `str`

Value: `loading_model`

### PREPARING_INPUTS

Type: `str`

Value: `preparing_inputs`

### GENERATING

Type: `str`

Value: `generating`

### POST_PROCESSING

Type: `str`

Value: `post_processing`

### SAVING

Type: `str`

Value: `saving`

### COMPLETED

Type: `str`

Value: `completed`

### FAILED

Type: `str`

Value: `failed`

### LORA_MANAGER_AVAILABLE

Type: `bool`

Value: `False`

