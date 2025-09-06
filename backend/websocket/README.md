# WebSocket Progress Integration

This module provides enhanced WebSocket progress tracking for real AI model generation in the WAN2.2 system.

## Overview

The WebSocket Progress Integration system provides detailed, real-time progress updates during video generation, including:

- **Detailed Generation Progress**: Stage-by-stage progress with estimated time remaining
- **Model Loading Progress**: Real-time updates during model loading and downloading
- **VRAM Monitoring**: Live VRAM usage tracking during generation
- **Generation Stage Notifications**: Notifications when generation stages change

## Components

### 1. Enhanced WebSocket Manager (`manager.py`)

The WebSocket manager has been enhanced with new methods for detailed progress tracking:

```python
# Detailed generation progress with stage information
await websocket_manager.send_detailed_generation_progress(
    task_id, stage, progress, message,
    estimated_time_remaining=30.0,
    current_step=5, total_steps=20
)

# Model loading progress updates
await websocket_manager.send_model_loading_progress(
    task_id, model_type, progress, status
)

# Real-time VRAM monitoring
await websocket_manager.send_vram_monitoring_update(vram_data)

# Generation stage change notifications
await websocket_manager.send_generation_stage_notification(
    task_id, stage, stage_progress
)
```

### 2. Progress Integration System (`progress_integration.py`)

The core progress integration system that coordinates between the generation pipeline and WebSocket manager:

```python
from backend.websocket.progress_integration import get_progress_integration, GenerationStage

# Get the progress integration instance
progress_integration = await get_progress_integration()

# Start tracking a generation task
await progress_integration.start_generation_tracking(
    task_id="gen_123",
    model_type="t2v-A14B",
    estimated_duration=60.0
)

# Update progress for a specific stage
await progress_integration.update_stage_progress(
    GenerationStage.LOADING_MODEL,
    progress=25,
    message="Loading T2V model"
)

# Update generation step progress
await progress_integration.update_generation_step_progress(
    current_step=5,
    total_steps=20
)

# Complete generation tracking
await progress_integration.complete_generation_tracking(
    success=True,
    output_path="/path/to/output.mp4"
)
```

## Generation Stages

The system tracks the following generation stages:

- `INITIALIZING`: Initial setup and validation
- `LOADING_MODEL`: Model loading from disk or download
- `DOWNLOADING_MODEL`: Model download from Hugging Face Hub
- `PREPARING_INPUTS`: Input processing and preparation
- `APPLYING_LORA`: LoRA application to the model
- `GENERATING`: Actual video generation
- `POST_PROCESSING`: Post-processing of generated frames
- `SAVING`: Saving the final video file
- `COMPLETED`: Generation completed successfully
- `FAILED`: Generation failed with error

## WebSocket Message Types

### 1. Detailed Generation Progress

```json
{
  "type": "detailed_generation_progress",
  "data": {
    "task_id": "gen_123",
    "stage": "generating",
    "progress": 45,
    "message": "Generating frame 9/20",
    "estimated_time_remaining": 30.5,
    "current_step": 9,
    "total_steps": 20,
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

### 2. Model Loading Progress

```json
{
  "type": "model_loading_progress",
  "data": {
    "task_id": "gen_123",
    "model_type": "t2v-A14B",
    "progress": 75,
    "status": "Model loaded successfully",
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

### 3. VRAM Monitoring

```json
{
  "type": "vram_monitoring",
  "data": {
    "allocated_mb": 8192.0,
    "reserved_mb": 9216.0,
    "free_mb": 8192.0,
    "total_mb": 16384.0,
    "allocated_percent": 50.0,
    "warning_level": "normal",
    "device_name": "NVIDIA RTX 4080",
    "task_id": "gen_123",
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

### 4. Generation Stage Notifications

```json
{
  "type": "generation_stage",
  "data": {
    "task_id": "gen_123",
    "stage": "generating",
    "stage_progress": 45,
    "stage_message": "Generating frames",
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

## Integration with Real Generation Pipeline

The progress integration is automatically used by the `RealGenerationPipeline`:

```python
# The pipeline automatically initializes progress tracking
await progress_integration.start_generation_tracking(task_id, model_type)

# Progress updates are sent automatically during generation
await self._send_progress_update(
    task_id, GenerationStage.GENERATING, progress, message
)

# Completion is handled automatically
await progress_integration.complete_generation_tracking(success, output_path)
```

## VRAM Monitoring

Real-time VRAM monitoring is automatically activated during generation:

- **Monitoring Frequency**: Every 1 second during active generation
- **Warning Levels**:
  - `normal`: < 75% VRAM usage
  - `warning`: 75-90% VRAM usage
  - `critical`: > 90% VRAM usage
- **Automatic Cleanup**: Monitoring stops when generation completes

## Frontend Integration

Frontend clients can subscribe to progress updates:

```javascript
// Subscribe to generation progress updates
websocket.send(
  JSON.stringify({
    action: "subscribe",
    topic: "generation_progress",
  })
);

// Handle progress messages
websocket.onmessage = (event) => {
  const message = JSON.parse(event.data);

  switch (message.type) {
    case "detailed_generation_progress":
      updateProgressBar(message.data.progress);
      updateStageMessage(message.data.message);
      updateTimeRemaining(message.data.estimated_time_remaining);
      break;

    case "vram_monitoring":
      updateVRAMDisplay(message.data);
      break;

    case "generation_stage":
      updateCurrentStage(message.data.stage);
      break;
  }
};
```

## Error Handling

The progress integration system includes comprehensive error handling:

- **Graceful Degradation**: If WebSocket manager is unavailable, progress tracking continues without WebSocket updates
- **Error Recovery**: Failed progress updates don't interrupt generation
- **Cleanup**: Resources are properly cleaned up even if generation fails

## Testing

Run the test suite to verify functionality:

```bash
python test_progress_integration.py
```

The test suite verifies:

- Progress integration initialization
- Stage progress updates
- Model loading progress
- VRAM monitoring
- Generation completion tracking
- WebSocket manager enhancements

## Performance Considerations

- **Minimal Overhead**: Progress updates are sent asynchronously without blocking generation
- **Efficient VRAM Monitoring**: VRAM stats are cached and updated at 1-second intervals
- **Memory Management**: Progress tracking state is automatically cleaned up after completion
- **Error Resilience**: Failed progress updates don't affect generation performance

## Requirements Compliance

This implementation satisfies the following requirements:

- **3.1**: ✅ System provides progress updates via WebSocket when generation starts
- **3.2**: ✅ System updates progress percentage when each generation step completes
- **3.3**: ✅ System provides estimated time remaining during generation
- **3.4**: ✅ System notifies frontend immediately via WebSocket when generation completes

The enhanced WebSocket progress integration provides comprehensive, real-time tracking of video generation progress, enabling users to monitor generation status, resource usage, and estimated completion times.
