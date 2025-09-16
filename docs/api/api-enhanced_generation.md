---
title: api.enhanced_generation
category: api
tags: [api, api]
---

# api.enhanced_generation

Enhanced Generation API for Phase 1 MVP
Provides seamless T2V, I2V, and TI2V generation with auto-detection and model switching

## Classes

### GenerationRequest

Enhanced generation request with auto-detection capabilities

#### Methods

##### validate_resolution(cls: Any, v: Any)



##### validate_priority(cls: Any, v: Any)



### GenerationResponse

Enhanced generation response

### ModelDetectionService

Service for auto-detecting optimal model type based on inputs

#### Methods

##### detect_model_type(prompt: str, has_image: bool, has_end_image: bool) -> str

Auto-detect optimal model type based on inputs

Args:
    prompt: Text prompt
    has_image: Whether start image is provided
    has_end_image: Whether end image is provided
    
Returns:
    Detected model type (T2V-A14B, I2V-A14B, or TI2V-5B)

##### get_model_requirements(model_type: str) -> <ast.Subscript object at 0x000001942CD17790>

Get requirements and capabilities for a model type

### PromptEnhancementService

Service for enhancing prompts for better generation results

#### Methods

##### enhance_prompt(prompt: str, model_type: str, options: <ast.Subscript object at 0x000001942CD174C0>) -> str

Enhance prompt based on model type and options

Args:
    prompt: Original prompt
    model_type: Target model type
    options: Enhancement options
    
Returns:
    Enhanced prompt

### GenerationParams



#### Methods

##### __init__(self: Any)



### ModelType



