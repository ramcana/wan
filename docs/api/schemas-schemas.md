---
title: schemas.schemas
category: api
tags: [api, schemas]
---

# schemas.schemas

Pydantic models for API request/response schemas

## Classes

### ModelType

Supported model types

### TaskStatus

Task status enumeration

### QuantizationLevel

Quantization levels for optimization

### GenerationRequest

Request model for video generation

#### Methods

##### validate_prompt(cls: Any, v: Any)



### GenerationResponse

Response model for generation requests

### TaskInfo

Task information model

### QueueStatus

Queue status information

### SystemStats

System resource statistics

### OptimizationSettings

Optimization settings model

### VideoMetadata

Generated video metadata

### OutputsResponse

Response model for outputs listing

### ErrorResponse

Error response model

### HealthResponse

Health check response

### PromptEnhanceRequest

Request model for prompt enhancement

### PromptEnhanceResponse

Response model for prompt enhancement

### PromptPreviewResponse

Response model for prompt enhancement preview

### LoRAInfo

LoRA file information

### LoRAListResponse

Response model for LoRA listing

### LoRAUploadResponse

Response model for LoRA upload

### LoRAStatusResponse

Response model for LoRA status

### LoRAPreviewResponse

Response model for LoRA preview

### LoRAMemoryImpactResponse

Response model for LoRA memory impact estimation

## Constants

### T2V_A14B

Type: `str`

Value: `T2V-A14B`

### I2V_A14B

Type: `str`

Value: `I2V-A14B`

### TI2V_5B

Type: `str`

Value: `TI2V-5B`

### PENDING

Type: `str`

Value: `pending`

### PROCESSING

Type: `str`

Value: `processing`

### COMPLETED

Type: `str`

Value: `completed`

### FAILED

Type: `str`

Value: `failed`

### CANCELLED

Type: `str`

Value: `cancelled`

### FP16

Type: `str`

Value: `fp16`

### BF16

Type: `str`

Value: `bf16`

### INT8

Type: `str`

Value: `int8`

