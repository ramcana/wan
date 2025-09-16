---
title: tests.test_lora_validation
category: api
tags: [api, tests]
---

# tests.test_lora_validation

Test LoRA parameter validation logic

## Classes

### MockGenerationParams

Mock generation parameters for testing

### MockLoRAValidator

Mock LoRA validator for testing validation logic

#### Methods

##### __init__(self: Any)



##### _validate_lora_params(self: Any, params: MockGenerationParams) -> dict

Validate LoRA parameters - extracted logic from RealGenerationPipeline

##### _get_basic_lora_fallback(self: Any, base_prompt: str, lora_name: str) -> str

Basic LoRA fallback prompt enhancement - extracted from RealGenerationPipeline

### TestLoRAValidation

Test LoRA validation logic without external dependencies

#### Methods

##### test_valid_lora_strength(self: Any)

Test valid LoRA strength values

##### test_invalid_lora_strength(self: Any)

Test invalid LoRA strength values

##### test_lora_file_extension_validation(self: Any)

Test LoRA file extension validation

##### test_missing_lora_file(self: Any)

Test validation with missing LoRA file

##### test_lora_fallback_enhancement(self: Any)

Test LoRA fallback prompt enhancement

##### test_empty_prompt_fallback(self: Any)

Test LoRA fallback with empty prompt

##### test_no_lora_path_validation(self: Any)

Test validation when no LoRA path is specified

##### test_lora_manager_warning(self: Any)

Test warning when LoRA is specified but manager is not available

