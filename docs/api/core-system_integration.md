---
title: core.system_integration
category: api
tags: [api, core]
---

# core.system_integration



## Classes

### SystemIntegration

Manages integration with existing Wan2.2 system components

#### Methods

##### __init__(self: Any)



##### _initialize_configuration_bridge(self: Any)

Initialize configuration bridge for enhanced config management

##### _load_config(self: Any) -> <ast.Subscript object at 0x000001942A1DDF60>

Load system configuration from config.json (fallback method)

##### _initialize_system_stats(self: Any)

Initialize system stats monitoring

##### get_system_info(self: Any) -> <ast.Subscript object at 0x0000019428D7FA60>

Get comprehensive system information combining all integrated components

##### _get_model_status_from_manager(self: Any) -> <ast.Subscript object at 0x0000019428DE7400>

Get model status information from the ModelManager

##### get_model_downloader(self: Any)

Get the model downloader instance

##### get_wan22_system_optimizer(self: Any)

Get the WAN22 system optimizer instance

##### get_wan_pipeline_loader(self: Any)

Get the WAN pipeline loader instance

##### get_model_manager(self: Any)

Get the initialized ModelManager instance

##### get_model_downloader(self: Any)

Get the initialized ModelDownloader instance

##### get_wan22_system_optimizer(self: Any)

Get the initialized WAN22SystemOptimizer instance

##### get_wan_pipeline_loader(self: Any)

Get the initialized WanPipelineLoader instance

##### get_configuration_bridge(self: Any)

Get the initialized ConfigurationBridge instance

##### get_initialization_errors(self: Any) -> <ast.Subscript object at 0x0000019428516EF0>

Get list of initialization errors that occurred during setup

##### get_runtime_config_for_generation(self: Any, model_type: str) -> <ast.Subscript object at 0x00000194285176D0>

Get runtime configuration optimized for specific model generation

##### update_optimization_setting(self: Any, setting_name: str, value: Any) -> bool

Update optimization setting at runtime

##### get_model_paths(self: Any) -> <ast.Subscript object at 0x0000019428551E10>

Get model path configuration

##### get_optimization_settings(self: Any) -> <ast.Subscript object at 0x00000194285522C0>

Get current optimization settings

##### validate_current_configuration(self: Any) -> <ast.Subscript object at 0x00000194285511B0>

Validate current configuration

##### scan_available_loras(self: Any, loras_directory: str) -> <ast.Subscript object at 0x0000019428552A10>

Scan for available LoRA files in the specified directory

##### _create_mock_model_manager(self: Any)

Create a mock ModelManager with basic functionality

##### _create_mock_model_downloader(self: Any)

Create a mock ModelDownloader with basic functionality

##### _create_real_wan_pipeline_loader(self: Any)

Create a real WanPipelineLoader with full functionality

##### _create_simplified_wan_pipeline_loader(self: Any)

Create a simplified WanPipelineLoader that can actually load models

##### _create_basic_system_stats(self: Any)

Create a basic system stats function

### MockModelManager



#### Methods

##### __init__(self: Any)



##### get_model_id(self: Any, model_type: str) -> <ast.Subscript object at 0x0000019428551570>



##### is_model_available(self: Any, model_id: str) -> bool



##### unload_model(self: Any, model_id: str)



### MockModelDownloader



#### Methods

##### __init__(self: Any, models_dir: str)



##### is_model_available(self: Any, model_name: str) -> bool



### SimplifiedWanPipelineLoader



#### Methods

##### __init__(self: Any)



##### load_pipeline(self: Any, model_type: str, model_path: str)

Load a pipeline for the specified model type

##### _get_model_path_for_type(self: Any, model_type: str) -> str

Get the model path for a given model type

##### load_wan_pipeline(self: Any, model_path: str, trust_remote_code: bool, apply_optimizations: bool, optimization_config: dict)

Load WAN pipeline with the same interface as the real loader

##### _create_pipeline_wrapper(self: Any, pipeline: Any, model_type: Any)

Create a wrapper that provides the expected interface for generation

### SimpleConfigurationBridge



#### Methods

##### __init__(self: Any, config_path: Any)



##### _load_config(self: Any)



##### get_config(self: Any, section: Any)



##### get_model_paths(self: Any)



##### get_optimization_settings(self: Any)



##### update_optimization_setting(self: Any, setting_name: Any, value: Any)



##### validate_configuration(self: Any)



##### get_runtime_config_for_generation(self: Any, model_type: Any)



##### get_config_summary(self: Any)



### ModelBridge



#### Methods

##### __init__(self: Any, model_manager: Any, system_integration: Any)



##### get_system_model_status(self: Any)

Get model status for system monitoring

##### is_model_available(self: Any, model_type: Any)

Check if a specific model is available

##### get_available_models(self: Any)

Get list of available models

##### check_model_availability(self: Any, model_type: Any)

Check model availability with detailed status

### PipelineWrapper



#### Methods

##### __init__(self: Any, pipeline: Any, model_type: Any)



##### generate(self: Any, config: Any)

Generate using the wrapped pipeline

##### __call__(self: Any)

Allow the wrapper to be called directly

### GenerationResult



#### Methods

##### __init__(self: Any, success: Any, frames: Any, errors: Any)



