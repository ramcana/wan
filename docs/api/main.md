---
title: main
category: api
tags: [api, main]
---

# main

Wan2.2 UI Variant - Main Application Entry Point
Handles configuration loading, command-line arguments, and application lifecycle

## Classes

### ApplicationConfig

Manages application configuration loading and validation

#### Methods

##### __init__(self: Any, config_path: str)



##### _load_config(self: Any) -> <ast.Subscript object at 0x000001942EF991B0>

Load configuration from JSON file with fallback to defaults

##### _get_default_config(self: Any) -> <ast.Subscript object at 0x000001942EF95480>

Return default configuration

##### _save_config(self: Any, config: <ast.Subscript object at 0x000001942EF952D0>)

Save configuration to JSON file

##### _validate_config(self: Any)

Validate configuration values and fix any issues

##### _ensure_directories(self: Any)

Create required directories if they don't exist

##### get_config(self: Any) -> <ast.Subscript object at 0x000001942EFCC5B0>

Get the loaded configuration

##### update_config(self: Any, updates: <ast.Subscript object at 0x000001942EFCC400>)

Update configuration with new values

### ApplicationManager

Manages the application lifecycle and cleanup

#### Methods

##### __init__(self: Any, config: ApplicationConfig)



##### _signal_handler(self: Any, signum: Any, frame: Any)

Handle shutdown signals

##### initialize(self: Any)

Initialize the application components

##### register_cleanup_handler(self: Any, handler: Any)

Register a cleanup handler to be called on shutdown

##### cleanup(self: Any)

Perform application cleanup

##### _cleanup_models(self: Any)

Cleanup loaded models and free GPU memory

##### _cleanup_temp_files(self: Any)

Cleanup temporary files

##### _cleanup_optimizer(self: Any)

Cleanup system optimizer

##### get_optimization_status(self: Any) -> <ast.Subscript object at 0x000001942F0112D0>

Get current optimization status for UI display

##### get_system_optimizer(self: Any) -> <ast.Subscript object at 0x000001942F011540>

Get the system optimizer instance

##### launch(self: Any)

Launch the Gradio application

### GenerationErrorHandler



#### Methods

##### handle_error(self: Any, error: Any, context: Any)



##### set_system_optimizer(self: Any, optimizer: Any)



### WAN22SystemOptimizer



#### Methods

##### __init__(self: Any)



##### get_optimization_history(self: Any)



## Constants

### ERROR_HANDLER_AVAILABLE

Type: `bool`

Value: `True`

### SYSTEM_OPTIMIZER_AVAILABLE

Type: `bool`

Value: `True`

### ERROR_HANDLER_AVAILABLE

Type: `bool`

Value: `False`

### SYSTEM_OPTIMIZER_AVAILABLE

Type: `bool`

Value: `True`

### SYSTEM_OPTIMIZER_AVAILABLE

Type: `bool`

Value: `False`

