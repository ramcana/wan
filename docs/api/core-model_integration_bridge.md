---
title: core.model_integration_bridge
category: api
tags: [api, core]
---

# core.model_integration_bridge



## Classes

### ModelStatus

Model availability status

### ModelType

Supported model types

### HardwareProfile

Hardware profile information

### ModelIntegrationStatus

Status of model integration

### GenerationParams

Parameters for video generation

### GenerationResult

Result of video generation

#### Methods

##### __post_init__(self: Any)



### ModelIntegrationBridge

Bridges existing ModelManager with FastAPI backend
Provides adapter methods to convert between existing model interfaces and FastAPI requirements

#### Methods

##### __init__(self: Any)



##### _get_recommended_hardware_profile(self: Any, model_type: str, available_vram_gb: float) -> <ast.Subscript object at 0x000001942EF964A0>

Get recommended hardware profile for given VRAM

##### replace_placeholder_model_mappings(self: Any) -> <ast.Subscript object at 0x000001942EF9C730>

Replace placeholder model mappings with real WAN model references

Returns:
    Dictionary mapping old placeholder references to new WAN implementations

##### get_model_implementation_info(self: Any, model_type: str) -> <ast.Subscript object at 0x000001942EF9D570>

Get information about model implementation (real vs placeholder)

Args:
    model_type: Model type to check
    
Returns:
    Dictionary with implementation information

##### get_model_status_from_existing_system(self: Any) -> <ast.Subscript object at 0x000001943459CC10>

Get model status for all supported models from existing system

##### _get_model_type_enum(self: Any, model_type: str) -> ModelType

Convert model type string to enum

##### _estimate_model_vram_usage(self: Any, model_type: str) -> float

Estimate VRAM usage for a model type in MB

##### get_hardware_profile(self: Any) -> <ast.Subscript object at 0x000001943459EAD0>

Get the detected hardware profile

##### is_initialized(self: Any) -> bool

Check if the bridge is initialized

##### get_download_progress(self: Any, model_type: str) -> <ast.Subscript object at 0x000001942F33F490>

Get current download progress for a specific model

##### get_all_download_progress(self: Any) -> <ast.Subscript object at 0x000001942F33F520>

Get download progress for all models

##### get_integration_status(self: Any) -> <ast.Subscript object at 0x000001942F33F820>

Get comprehensive integration status

##### set_hardware_optimizer(self: Any, optimizer: Any)

Set the hardware optimizer for model optimization

##### _schedule_hardware_detection(self: Any)

Schedule hardware detection to run asynchronously

##### _sync_hardware_detection(self: Any)

Synchronous hardware detection fallback

##### get_lora_status(self: Any, model_type: <ast.Subscript object at 0x000001942FDC97E0>) -> <ast.Subscript object at 0x000001942FE5B790>

Get LoRA status information for models

Args:
    model_type: Optional specific model type to check
    
Returns:
    Dictionary with LoRA status information

##### get_lora_memory_impact(self: Any, lora_name: str) -> <ast.Subscript object at 0x000001942FE3EDD0>

Get memory impact estimation for a LoRA

Args:
    lora_name: Name of LoRA to analyze
    
Returns:
    Dictionary with memory impact information

### WANPipelineFactory



#### Methods

##### __init__(self: Any)



### WANModelStatus



#### Methods

##### __init__(self: Any)



### WANModelType



## Constants

### ERROR_HANDLER_AVAILABLE

Type: `bool`

Value: `True`

### FALLBACK_RECOVERY_AVAILABLE

Type: `bool`

Value: `True`

### WAN_MODELS_AVAILABLE

Type: `bool`

Value: `True`

### MISSING

Type: `str`

Value: `missing`

### AVAILABLE

Type: `str`

Value: `available`

### LOADED

Type: `str`

Value: `loaded`

### CORRUPTED

Type: `str`

Value: `corrupted`

### DOWNLOADING

Type: `str`

Value: `downloading`

### T2V_A14B

Type: `str`

Value: `t2v-A14B`

### I2V_A14B

Type: `str`

Value: `i2v-A14B`

### TI2V_5B

Type: `str`

Value: `ti2v-5B`

### ERROR_HANDLER_AVAILABLE

Type: `bool`

Value: `False`

### FALLBACK_RECOVERY_AVAILABLE

Type: `bool`

Value: `False`

### WAN_MODELS_AVAILABLE

Type: `bool`

Value: `False`

### T2V_A14B

Type: `str`

Value: `t2v-A14B`

### I2V_A14B

Type: `str`

Value: `i2v-A14B`

### TI2V_5B

Type: `str`

Value: `ti2v-5B`

