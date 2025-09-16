---
title: tests.test_model_integration_comprehensive
category: api
tags: [api, tests]
---

# tests.test_model_integration_comprehensive

Comprehensive Model Integration Tests
Focused testing for ModelIntegrationBridge functionality with all model types

## Classes

### TestModelIntegrationBridgeDetailed

Detailed tests for ModelIntegrationBridge functionality

### ModelIntegrationBridge



#### Methods

##### __init__(self: Any)



##### _get_model_type_enum(self: Any, model_type: Any)



##### _estimate_model_vram_usage(self: Any, model_type: Any)



##### _get_model_path(self: Any, model_type: Any)



##### is_initialized(self: Any)



##### get_integration_status(self: Any)



##### _create_optimization_config(self: Any, params: Any)



##### _create_error_result(self: Any, task_id: Any, category: Any, message: Any)



### ModelStatus



### ModelType



### ModelIntegrationStatus



#### Methods

##### __init__(self: Any)



### GenerationParams



#### Methods

##### __init__(self: Any)



### GenerationResult



## Constants

### AVAILABLE

Type: `str`

Value: `available`

### MISSING

Type: `str`

Value: `missing`

### CORRUPTED

Type: `str`

Value: `corrupted`

### T2V_A14B

Type: `str`

Value: `T2V_A14B`

### I2V_A14B

Type: `str`

Value: `I2V_A14B`

### TI2V_5B

Type: `str`

Value: `TI2V_5B`

