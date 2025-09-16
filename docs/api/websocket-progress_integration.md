---
title: websocket.progress_integration
category: api
tags: [api, websocket]
---

# websocket.progress_integration

WebSocket Progress Integration for Real AI Model Generation
Provides detailed progress tracking and real-time updates for generation pipeline

## Classes

### GenerationStage

Stages of video generation process

### ProgressIntegration

Integration class for sending detailed progress updates via WebSocket
Connects the real generation pipeline with the WebSocket manager

#### Methods

##### __init__(self: Any, websocket_manager: Any)

Initialize progress integration

Args:
    websocket_manager: WebSocket connection manager instance

##### _calculate_estimated_time_remaining(self: Any, current_progress: int) -> <ast.Subscript object at 0x000001942B368610>

Calculate estimated time remaining based on current progress

## Constants

### INITIALIZING

Type: `str`

Value: `initializing`

### LOADING_MODEL

Type: `str`

Value: `loading_model`

### DOWNLOADING_MODEL

Type: `str`

Value: `downloading_model`

### PREPARING_INPUTS

Type: `str`

Value: `preparing_inputs`

### APPLYING_LORA

Type: `str`

Value: `applying_lora`

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

