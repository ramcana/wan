---
category: reference
last_updated: '2025-09-15T22:49:59.965832'
original_path: docs\WAN22_USER_GUIDE.md
tags:
- configuration
- api
- troubleshooting
- installation
- security
- performance
title: Wan2.2 Compatibility System - User Guide
---

# Wan2.2 Compatibility System - User Guide

## Overview

The Wan2.2 Compatibility System provides seamless integration for Wan2.2 video generation models with automatic compatibility detection, optimization, and error handling. This guide will help you understand how to use the system effectively.

## Quick Start

### Basic Usage

```python
from wan22_compatibility_system import get_compatibility_system

# Initialize the compatibility system
compat_system = get_compatibility_system()

# Load a Wan model
result = compat_system.load_model("path/to/wan2.2-model")

if result.success:
    # Generate a video
    video_result = compat_system.generate_video(
        result.pipeline,
        "A beautiful sunset over mountains",
        "output.mp4"
    )

    if video_result.success:
        print(f"Video saved to: {video_result.output_path}")
    else:
        print(f"Generation failed: {video_result.errors}")
else:
    print(f"Model loading failed: {result.errors}")
```

### Configuration

```python
from wan22_compatibility_system import CompatibilitySystemConfig, initialize_compatibility_system

# Custom configuration
config = CompatibilitySystemConfig(
    enable_optimization=True,
    enable_safe_loading=True,
    max_memory_usage_gb=8.0,
    default_precision="bf16"
)

# Initialize with custom config
compat_system = initialize_compatibility_system(config)
```

## Features

### 1. Automatic Model Detection

The system automatically detects:

- Model architecture (Wan T2V, T2I, standard Diffusers)
- Required pipeline classes
- Component compatibility
- Resource requirements

### 2. Smart Optimization

Based on your hardware, the system applies:

- **Mixed Precision**: Reduces memory usage by 30-50%
- **CPU Offloading**: Moves unused components to CPU
- **Chunked Processing**: Processes videos in smaller chunks
- **Memory Management**: Automatic cleanup and optimization

### 3. Fallback Strategies

When primary loading fails:

- **Component Isolation**: Uses compatible components only
- **Alternative Models**: Suggests similar models
- **Reduced Functionality**: Loads with basic features

### 4. Security Features

- **Safe Loading**: Validates model sources
- **Remote Code Handling**: Secure execution of custom pipelines
- **Sandboxing**: Isolates untrusted code execution

## Supported Models

### Wan Models

- ✅ Wan2.2-T2V-A14B (Text-to-Video)
- ✅ Wan2.2-I2V-A14B (Image-to-Video)
- ✅ Wan2.2-TI2V-5B (Text+Image-to-Video)
- ✅ Wan2.2 Mini variants
- ✅ Community fine-tuned models

### Standard Models

- ✅ Stable Diffusion variants
- ✅ Other Diffusers-compatible models

## Hardware Requirements

### Minimum Requirements

- **GPU**: NVIDIA GTX 1060 6GB or equivalent
- **VRAM**: 6GB (with optimizations)
- **RAM**: 16GB system memory
- **Storage**: 50GB free space

### Recommended Requirements

- **GPU**: NVIDIA RTX 3080 or better
- **VRAM**: 12GB or more
- **RAM**: 32GB system memory
- **Storage**: 100GB+ SSD storage

### Optimal Requirements

- **GPU**: NVIDIA RTX 4090 or better
- **VRAM**: 24GB or more
- **RAM**: 64GB system memory
- **Storage**: 500GB+ NVMe SSD

## Configuration Options

### System Configuration

```python
config = CompatibilitySystemConfig(
    # Core features
    enable_diagnostics=True,           # Detailed error reporting
    enable_performance_monitoring=True, # Performance tracking
    enable_safe_loading=True,          # Security features
    enable_optimization=True,          # Memory optimizations
    enable_fallback=True,             # Fallback strategies

    # Resource limits
    max_memory_usage_gb=12.0,         # Maximum VRAM usage
    default_precision="bf16",         # Default precision (fp16/bf16/fp32)

    # Optimization settings
    enable_cpu_offload=True,          # CPU offloading
    enable_chunked_processing=True,   # Chunked video processing

    # Logging
    log_level="INFO",                 # Logging level
    diagnostics_dir="diagnostics",    # Diagnostics output directory
)
```

### Model-Specific Settings

```python
# Load with specific optimizations
result = compat_system.load_model(
    "path/to/model",
    torch_dtype="float16",           # Precision override
    low_cpu_mem_usage=True,         # Memory optimization
    device_map="auto",              # Automatic device mapping
    trust_remote_code=True,         # Allow remote pipeline code
    use_safetensors=True           # Use safe tensor format
)
```

### Generation Parameters

```python
video_result = compat_system.generate_video(
    pipeline,
    prompt="Your prompt here",
    output_path="output.mp4",

    # Video settings
    num_frames=16,                  # Number of frames
    height=720,                     # Video height
    width=1280,                     # Video width
    fps=8,                         # Frames per second

    # Generation settings
    num_inference_steps=50,         # Quality vs speed
    guidance_scale=7.5,            # Prompt adherence
    seed=42,                       # Reproducibility

    # Memory settings
    enable_cpu_offload=True,       # CPU offloading
    chunk_size=4,                  # Chunk size for processing
)
```

## Error Handling

### Common Issues and Solutions

#### 1. "Model not found" Error

```
Error: Could not find model at specified path
```

**Solution**:

- Verify the model path is correct
- Ensure model files are downloaded completely
- Check file permissions

#### 2. "Insufficient VRAM" Error

```
Error: CUDA out of memory
```

**Solutions**:

- Enable CPU offloading: `enable_cpu_offload=True`
- Use lower precision: `torch_dtype="float16"`
- Reduce batch size or video resolution
- Enable chunked processing

#### 3. "Pipeline not found" Error

```
Error: WanPipeline class not found
```

**Solutions**:

- Enable remote code: `trust_remote_code=True`
- Install required dependencies
- Check internet connection for remote code download

#### 4. "VAE loading failed" Error

```
Error: VAE shape mismatch
```

**Solutions**:

- The system handles this automatically
- If issues persist, try safe loading mode
- Check model compatibility

### Getting Help

1. **Check Diagnostics**: Look in the `diagnostics/` folder for detailed reports
2. **Enable Debug Logging**: Set `log_level="DEBUG"` for detailed logs
3. **System Status**: Use `compat_system.get_system_status()` for system info
4. **Performance Stats**: Check performance metrics for bottlenecks

## Performance Optimization

### Memory Optimization

```python
# For 8GB VRAM systems
config = CompatibilitySystemConfig(
    max_memory_usage_gb=6.0,
    default_precision="fp16",
    enable_cpu_offload=True,
    enable_chunked_processing=True
)

# Load with aggressive optimizations
result = compat_system.load_model(
    model_path,
    torch_dtype="float16",
    low_cpu_mem_usage=True,
    device_map="auto"
)
```

### Speed Optimization

```python
# For faster generation (may use more memory)
config = CompatibilitySystemConfig(
    default_precision="bf16",
    enable_cpu_offload=False,  # Keep everything on GPU
    enable_chunked_processing=False
)

# Generate with speed settings
video_result = compat_system.generate_video(
    pipeline,
    prompt,
    output_path,
    num_inference_steps=25,  # Fewer steps = faster
    guidance_scale=7.0       # Lower guidance = faster
)
```

### Quality Optimization

```python
# For best quality (slower, more memory)
video_result = compat_system.generate_video(
    pipeline,
    prompt,
    output_path,
    num_inference_steps=100,  # More steps = better quality
    guidance_scale=12.0,      # Higher guidance = better prompt adherence
    height=1080,              # Higher resolution
    width=1920,
    num_frames=24             # More frames = smoother video
)
```

## Advanced Usage

### Custom Pipeline Loading

```python
from pipeline_manager import PipelineManager

pipeline_manager = PipelineManager()

# Load custom pipeline
pipeline = pipeline_manager.load_custom_pipeline(
    model_path,
    "CustomWanPipeline",
    custom_arg1="value1",
    custom_arg2="value2"
)
```

### Manual Optimization

```python
from optimization_manager import OptimizationManager

opt_manager = OptimizationManager()

# Analyze system
resources = opt_manager.analyze_system_resources()
print(f"Available VRAM: {resources.available_vram_mb}MB")

# Create optimization plan
plan = opt_manager.recommend_optimizations(model_requirements, resources)
print(f"Recommended optimizations: {plan}")
```

### Diagnostics and Debugging

```python
# Enable detailed diagnostics
config = CompatibilitySystemConfig(
    enable_diagnostics=True,
    log_level="DEBUG"
)

# Get system status
status = compat_system.get_system_status()
print(json.dumps(status, indent=2))

# Check specific component
from architecture_detector import ArchitectureDetector

detector = ArchitectureDetector()
architecture = detector.detect_model_architecture(model_path)
print(f"Detected architecture: {architecture.architecture_type}")
```

## Best Practices

### 1. Model Management

- Keep models in a dedicated directory
- Use descriptive folder names
- Regularly clean up unused models
- Backup important fine-tuned models

### 2. Memory Management

- Monitor VRAM usage during generation
- Use appropriate precision for your hardware
- Enable optimizations for resource-constrained systems
- Close unused applications during generation

### 3. Generation Settings

- Start with default settings and adjust as needed
- Use lower inference steps for testing
- Increase steps and resolution for final outputs
- Save successful parameter combinations

### 4. Troubleshooting

- Always check the diagnostics folder first
- Enable debug logging for persistent issues
- Test with simple prompts first
- Verify model integrity if loading fails

## Integration with Existing Code

### Gradio UI Integration

```python
import gradio as gr
from wan22_compatibility_system import get_compatibility_system

compat_system = get_compatibility_system()

def generate_video_ui(prompt, model_path):
    # Load model
    load_result = compat_system.load_model(model_path)
    if not load_result.success:
        return None, f"Error: {load_result.errors[0]}"

    # Generate video
    video_result = compat_system.generate_video(
        load_result.pipeline, prompt, "output.mp4"
    )

    if video_result.success:
        return video_result.output_path, "Success!"
    else:
        return None, f"Error: {video_result.errors[0]}"

# Create Gradio interface
interface = gr.Interface(
    fn=generate_video_ui,
    inputs=[
        gr.Textbox(label="Prompt"),
        gr.Textbox(label="Model Path")
    ],
    outputs=[
        gr.Video(label="Generated Video"),
        gr.Textbox(label="Status")
    ]
)
```

### Flask API Integration

```python
from flask import Flask, request, jsonify
from wan22_compatibility_system import get_compatibility_system

app = Flask(__name__)
compat_system = get_compatibility_system()

@app.route('/generate', methods=['POST'])
def generate_video():
    data = request.json

    # Load model
    load_result = compat_system.load_model(data['model_path'])
    if not load_result.success:
        return jsonify({'error': load_result.errors[0]}), 400

    # Generate video
    video_result = compat_system.generate_video(
        load_result.pipeline,
        data['prompt'],
        f"outputs/{data['output_name']}.mp4"
    )

    if video_result.success:
        return jsonify({
            'success': True,
            'output_path': video_result.output_path,
            'generation_time': video_result.generation_time
        })
    else:
        return jsonify({'error': video_result.errors[0]}), 500

if __name__ == '__main__':
    app.run(debug=True)
```

## Conclusion

The Wan2.2 Compatibility System provides a robust, user-friendly way to work with Wan models and other video generation systems. With automatic optimization, comprehensive error handling, and detailed diagnostics, it simplifies the complex process of video generation while maintaining flexibility for advanced users.

For additional support or questions, check the troubleshooting section or enable debug logging for detailed information about system behavior.
