# Design Document

## Overview

The Wan2.2 UI Variant is a web-based application built using Gradio that provides an intuitive interface for generating videos using advanced AI models. The system is designed as a modular, single-page application optimized for NVIDIA RTX 4080 hardware, supporting multiple generation modes (T2V, I2V, TI2V) with comprehensive VRAM optimization and queue management capabilities.

The architecture follows a clean separation between the frontend UI layer and backend processing logic, enabling efficient resource management and scalable video generation workflows.

## Architecture

### System Architecture

```mermaid
graph TB
    subgraph "Frontend Layer"
        UI[Gradio Web Interface]
        TABS[Generation | Optimizations | Queue & Stats | Outputs]
    end

    subgraph "Backend Layer"
        UTILS[utils.py - Core Logic]
        QUEUE[Queue Manager]
        STATS[Resource Monitor]
        MODEL[Model Manager]
    end

    subgraph "Storage Layer"
        MODELS[models/ - Downloaded Models]
        LORAS[loras/ - LoRA Weights]
        OUTPUTS[outputs/ - Generated Videos]
        CONFIG[config.json - Settings]
    end

    subgraph "External Services"
        HF[Hugging Face Hub]
        GPU[NVIDIA RTX 4080]
    end

    UI --> UTILS
    UTILS --> QUEUE
    UTILS --> STATS
    UTILS --> MODEL
    MODEL --> MODELS
    MODEL --> LORAS
    UTILS --> OUTPUTS
    UTILS --> CONFIG
    MODEL --> HF
    MODEL --> GPU
```

### Component Architecture

The system is organized into distinct layers:

1. **Presentation Layer**: Gradio-based web interface with tabbed navigation
2. **Business Logic Layer**: Core utilities for model management, generation, and optimization
3. **Data Access Layer**: File system operations for models, outputs, and configuration
4. **Infrastructure Layer**: GPU management, memory optimization, and resource monitoring

## Components and Interfaces

### Frontend Components

#### Main UI Controller (`ui.py`)

- **Purpose**: Orchestrates the Gradio interface and handles user interactions
- **Key Methods**:
  - `create_generation_tab()`: Builds the main generation interface
  - `create_optimization_tab()`: Provides VRAM and performance settings
  - `create_queue_tab()`: Displays queue status and system stats
  - `create_outputs_tab()`: Shows generated video gallery
  - `setup_event_handlers()`: Binds UI events to backend functions

#### Dynamic UI Components

- **Model Type Selector**: Dropdown with T2V-A14B, I2V-A14B, TI2V-5B options
- **Conditional Image Input**: Shows/hides based on selected model type
- **Resolution Selector**: Supports 1280x720, 1280x704, 1920x1080
- **Progress Indicators**: Real-time generation progress and queue status

### Backend Components

#### Model Manager (`utils.py`)

- **Purpose**: Handles model loading, optimization, and inference
- **Key Methods**:
  - `load_wan22_model(model_type, lora_path=None)`: Downloads and loads models from Hugging Face
  - `optimize_model(model, quant_level, offload, tile_size)`: Applies VRAM optimizations
  - `apply_lora_weights(model, lora_path, strength)`: Loads and applies LoRA modifications
- **Interfaces**:
  - Input: Model type string, optimization parameters
  - Output: Optimized PyTorch model ready for inference

#### Generation Engine

- **Purpose**: Orchestrates video generation across different modes
- **Key Methods**:
  - `generate_video(pipe, prompt, model_type, **kwargs)`: Main generation function
  - `validate_inputs(prompt, image, resolution)`: Input validation and preprocessing
  - `handle_generation_error(error)`: Error recovery and user feedback
- **Interfaces**:
  - Input: Generation parameters (prompt, image, settings)
  - Output: Generated video frames or error messages

#### Queue Manager

- **Purpose**: Manages batch processing and task scheduling
- **Key Methods**:
  - `add_to_queue(task_params)`: Adds generation task to FIFO queue
  - `process_queue()`: Background thread for sequential task processing
  - `get_queue_status()`: Returns current queue state and progress
- **Interfaces**:
  - Input: Task parameters dictionary
  - Output: Queue status updates and completion notifications

#### Resource Monitor

- **Purpose**: Tracks system performance and resource usage
- **Key Methods**:
  - `get_system_stats()`: Collects CPU, RAM, GPU, VRAM metrics
  - `monitor_vram_usage()`: Tracks GPU memory consumption
  - `check_resource_limits()`: Warns about resource constraints
- **Interfaces**:
  - Input: None (system polling)
  - Output: Formatted resource statistics

#### Prompt Enhancer

- **Purpose**: Improves prompt quality for better generation results
- **Key Methods**:
  - `enhance_prompt(prompt)`: Adds quality keywords and style improvements
  - `detect_vace_keywords(prompt)`: Identifies VACE aesthetic requirements
  - `apply_cinematic_enhancements(prompt)`: Adds cinematic styling terms
- **Interfaces**:
  - Input: Original text prompt
  - Output: Enhanced prompt with quality improvements

## Data Models

### Generation Task Model

```python
@dataclass
class GenerationTask:
    id: str
    model_type: str  # 't2v-A14B', 'i2v-A14B', 'ti2v-5B'
    prompt: str
    image: Optional[PIL.Image] = None
    resolution: str = '1280x720'
    steps: int = 50
    lora_path: Optional[str] = None
    lora_strength: float = 1.0
    status: str = 'pending'  # 'pending', 'processing', 'completed', 'failed'
    created_at: datetime
    completed_at: Optional[datetime] = None
    output_path: Optional[str] = None
    error_message: Optional[str] = None
```

### System Configuration Model

```python
@dataclass
class SystemConfig:
    default_quantization: str = 'bf16'
    enable_offload: bool = True
    vae_tile_size: int = 256
    max_queue_size: int = 10
    stats_refresh_interval: int = 5
    output_directory: str = 'outputs'
    models_directory: str = 'models'
    loras_directory: str = 'loras'
```

### Resource Statistics Model

```python
@dataclass
class ResourceStats:
    cpu_percent: float
    ram_percent: float
    ram_used_gb: float
    ram_total_gb: float
    gpu_percent: float
    vram_used_mb: float
    vram_total_mb: float
    timestamp: datetime
```

## Error Handling

### Error Categories and Responses

#### VRAM Out of Memory Errors

- **Detection**: Catch `torch.cuda.OutOfMemoryError`
- **Response**:
  - Clear GPU cache with `torch.cuda.empty_cache()`
  - Suggest lower resolution or enable offloading
  - Display user-friendly error message with optimization tips

#### Model Loading Errors

- **Detection**: Network timeouts, missing models, corrupted files
- **Response**:
  - Retry download with exponential backoff
  - Fallback to cached models if available
  - Clear error messages with troubleshooting steps

#### Input Validation Errors

- **Detection**: Invalid file formats, oversized images, empty prompts
- **Response**:
  - Real-time validation with immediate feedback
  - Automatic format conversion where possible
  - Clear guidance on acceptable inputs

#### Queue Processing Errors

- **Detection**: Task failures, system resource exhaustion
- **Response**:
  - Mark failed tasks with error status
  - Continue processing remaining queue items
  - Provide retry mechanisms for failed tasks

### Error Recovery Strategies

1. **Graceful Degradation**: Reduce quality settings automatically when resources are constrained
2. **Automatic Retry**: Retry failed operations with modified parameters
3. **User Notification**: Clear, actionable error messages with suggested solutions
4. **System Recovery**: Automatic cleanup of GPU memory and temporary files

## Testing Strategy

### Unit Testing

#### Model Manager Tests

- Test model loading with different quantization levels
- Verify LoRA weight application and strength adjustment
- Validate optimization parameter effects on VRAM usage

#### Generation Engine Tests

- Test T2V, I2V, and TI2V generation modes
- Verify input validation and error handling
- Test resolution scaling and performance metrics

#### Queue Manager Tests

- Test FIFO queue ordering and task processing
- Verify concurrent access safety
- Test queue persistence across application restarts

### Integration Testing

#### End-to-End Generation Workflows

- Complete T2V generation from prompt to output video
- I2V workflow with image upload and processing
- TI2V hybrid generation with both text and image inputs

#### Resource Management Integration

- VRAM optimization effectiveness across different settings
- Queue processing under resource constraints
- Error recovery during generation failures

#### UI Integration

- Dynamic UI updates based on model selection
- Real-time stats refresh and accuracy
- File upload and output gallery functionality

### Performance Testing

#### Generation Performance Benchmarks

- 720p video generation time (target: <9 minutes)
- 1080p video generation time (target: <17 minutes)
- VRAM usage optimization (target: <12GB for 720p)

#### System Resource Testing

- CPU and RAM usage during generation
- Queue throughput with multiple concurrent tasks
- UI responsiveness during heavy processing

#### Stress Testing

- Maximum queue size handling
- Extended generation sessions
- Resource exhaustion recovery

### User Acceptance Testing

#### Usability Testing

- Intuitive navigation between tabs
- Clear feedback for generation progress
- Effective error message communication

#### Feature Validation

- All generation modes produce expected results
- Optimization settings provide measurable benefits
- Queue management works reliably

#### Compatibility Testing

- Different image formats and sizes
- Various prompt lengths and complexity
- Different hardware configurations (within RTX 4080 constraints)

## Performance Considerations

### VRAM Optimization Strategies

1. **Model Quantization**: Support for fp16, bf16, and int8 precision levels
2. **Sequential CPU Offloading**: Move model components between GPU and CPU as needed
3. **VAE Tiling**: Process large images in smaller tiles to reduce memory peaks
4. **Gradient Checkpointing**: Trade computation for memory during inference

### Generation Performance Optimizations

1. **Model Caching**: Keep frequently used models in memory
2. **Batch Processing**: Optimize queue processing for maximum throughput
3. **Asynchronous Operations**: Non-blocking UI updates during generation
4. **Memory Management**: Proactive cleanup of GPU memory between tasks

### UI Responsiveness

1. **Background Processing**: All generation tasks run in separate threads
2. **Progressive Updates**: Real-time progress indicators and stats refresh
3. **Lazy Loading**: Load UI components and data on demand
4. **Efficient Rendering**: Optimize gallery display for large numbers of outputs

This design provides a robust foundation for implementing the Wan2.2 UI variant with all required features while maintaining optimal performance on the target hardware.
