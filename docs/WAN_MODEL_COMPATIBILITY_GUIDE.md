# WAN Model Compatibility System Guide

This guide provides comprehensive information about the WAN model compatibility system, which addresses the unique loading requirements of WAN2.2 models that differ from standard diffusers models.

## Overview

WAN2.2 models have a completely different structure than standard diffusers models:

**Standard Diffusers Models:**

- `unet/` - Main model component
- `vae/` - VAE encoder/decoder
- `text_encoder/` - Text encoder
- `tokenizer/` - Tokenizer
- `scheduler/` - Noise scheduler

**WAN2.2 Models:**

- `pytorch_model.bin` - Main model weights
- `Wan2.1_VAE.pth` - VAE weights
- `models_t5_umt5-xxl-enc-bf16.pth` - Text encoder
- `transformer/` and `transformer_2/` - Transformer components
- `high_noise_model/` and `low_noise_model/` - Specialized components
- `boundary_ratio` - WAN-specific parameter

## System Components

### 1. Architecture Detector (`architecture_detector.py`)

Detects model architecture and identifies WAN models based on:

- Component structure analysis
- `model_index.json` parsing
- VAE dimension detection (2D vs 3D)
- WAN-specific attributes (`transformer_2`, `boundary_ratio`)

```python
from architecture_detector import ArchitectureDetector

detector = ArchitectureDetector()
architecture = detector.detect_model_architecture("path/to/wan/model")
print(f"Architecture: {architecture.architecture_type.value}")
print(f"Is WAN: {architecture.signature.is_wan_architecture()}")
```

### 2. Pipeline Manager (`pipeline_manager.py`)

Selects appropriate pipeline class and handles loading:

- Maps architecture types to pipeline classes
- Validates pipeline requirements
- Handles `trust_remote_code` requirements
- Provides fallback mechanisms

```python
from pipeline_manager import PipelineManager

manager = PipelineManager()
pipeline_class = manager.select_pipeline_class(architecture.signature)
load_result = manager.load_custom_pipeline(
    model_path="path/to/wan/model",
    pipeline_class=pipeline_class,
    trust_remote_code=True
)
```

### 3. WAN Pipeline Loader (`wan_pipeline_loader.py`)

Comprehensive WAN model loading with optimizations:

- Automatic architecture detection
- Optimization application
- Memory management
- Resource monitoring

```python
from wan_pipeline_loader import WanPipelineLoader

loader = WanPipelineLoader()
wrapper = loader.load_wan_pipeline(
    model_path="path/to/wan/model",
    trust_remote_code=True,
    apply_optimizations=True
)

# Generate video
config = GenerationConfig(
    prompt="A beautiful sunset over mountains",
    num_frames=16,
    width=1280,
    height=720
)
result = wrapper.generate(config)
```

### 4. Compatibility Registry (`compatibility_registry.py`)

Maps known models to their requirements:

- Pipeline class requirements
- VRAM requirements
- Supported optimizations
- Dependency information

```python
from compatibility_registry import get_compatibility_registry

registry = get_compatibility_registry()
requirements = registry.get_pipeline_requirements("Wan-AI/Wan2.2-T2V-A14B-Diffusers")
print(f"Pipeline class: {requirements.pipeline_class}")
print(f"Min VRAM: {requirements.vram_requirements['min_mb']}MB")
```

### 5. Performance Monitor (`performance_monitor.py`)

Monitors and optimizes performance:

- Generation speed tracking
- Memory usage monitoring
- Optimization effectiveness measurement
- Regression detection

```python
from performance_monitor import get_performance_monitor

monitor = get_performance_monitor()
monitor.start_operation("generation_1", "generation")
# ... perform generation ...
metrics = monitor.end_operation("generation_1", success=True)
print(f"Duration: {metrics.duration:.2f}s, Peak memory: {metrics.memory_peak_mb}MB")
```

## Common Issues and Solutions

### Issue 1: "Failed to load pipeline: Pipeline StableDiffusionPipeline expected..."

**Cause:** System is trying to load WAN model with standard diffusers pipeline.

**Solution:**

```python
# Ensure trust_remote_code=True
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "path/to/wan/model",
    trust_remote_code=True,
    torch_dtype=torch.float16
)
```

### Issue 2: "DLL load failed while importing \_C"

**Cause:** PyTorch installation issue on Windows.

**Solution:**

```bash
# Run the PyTorch DLL fix
python local_installation/fix_pytorch_dll.py

# Or manually reinstall PyTorch
pip uninstall torch torchvision torchaudio
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

### Issue 3: Out of Memory Errors

**Cause:** WAN models require significant VRAM (8-12GB+).

**Solution:**

```python
# Use optimizations
wrapper = loader.load_wan_pipeline(
    model_path="path/to/wan/model",
    apply_optimizations=True,
    optimization_config={
        "precision": "fp16",
        "enable_cpu_offload": True,
        "chunk_size": 8
    }
)
```

### Issue 4: Model Not Detected as WAN

**Cause:** Missing or incorrect `model_index.json`.

**Solution:**

```python
# Use the compatibility fix
from wan_model_compatibility_fix import apply_wan_compatibility_fix

results = apply_wan_compatibility_fix("path/to/wan/model")
print("Fixes applied:", results["fixes_applied"])
```

## Best Practices

### 1. Model Loading

```python
# Always use trust_remote_code for WAN models
pipeline = DiffusionPipeline.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.float16,  # Use FP16 for memory efficiency
    device_map="auto"  # Automatic device placement
)
```

### 2. Memory Management

```python
# For systems with limited VRAM
config = GenerationConfig(
    prompt="Your prompt here",
    num_frames=16,  # Start with fewer frames
    width=512,      # Lower resolution initially
    height=512,
    enable_optimizations=True,
    force_chunked_processing=True  # Process in chunks
)
```

### 3. Error Handling

```python
try:
    wrapper = loader.load_wan_pipeline(model_path, trust_remote_code=True)
    result = wrapper.generate(config)
except ValueError as e:
    if "not a Wan architecture" in str(e):
        # Model is not WAN - use standard pipeline
        pipeline = DiffusionPipeline.from_pretrained(model_path)
    else:
        # Other loading error
        print(f"Loading failed: {e}")
```

### 4. Performance Optimization

```python
# Monitor performance
monitor = get_performance_monitor()
monitor.start_operation("generation", "generation")

# Generate with optimizations
result = wrapper.generate(config)

# Record metrics
metrics = monitor.end_operation("generation", success=result.success)

# Check for regressions
alerts = monitor.detect_performance_regressions()
if alerts:
    print(f"Performance regression detected: {alerts[0].metric_name}")
```

## Troubleshooting Tools

### 1. Debug Script

```bash
python debug_wan_model_loading.py path/to/wan/model
```

### 2. Compatibility Fix

```bash
python wan_model_compatibility_fix.py path/to/wan/model
```

### 3. Performance Analysis

```python
from performance_monitor import get_performance_monitor

monitor = get_performance_monitor()
summary = monitor.get_performance_summary(operation_type="generation")
print(f"Average generation time: {summary['duration_stats']['mean']:.2f}s")
```

## Integration with Existing Code

### Updating utils.py

The system integrates with existing code through the model manager:

```python
# In your existing code, replace:
# pipeline = DiffusionPipeline.from_pretrained(model_path)

# With:
from wan_pipeline_loader import WanPipelineLoader

loader = WanPipelineLoader()
try:
    # Try WAN loading first
    wrapper = loader.load_wan_pipeline(model_path, trust_remote_code=True)
    pipeline = wrapper.pipeline
except ValueError:
    # Fallback to standard loading
    pipeline = DiffusionPipeline.from_pretrained(model_path)
```

### UI Integration

The system provides UI-friendly error messages and progress callbacks:

```python
def progress_callback(message: str, progress: float):
    print(f"{message}: {progress:.1f}%")

wrapper = loader.load_wan_pipeline(
    model_path=model_path,
    progress_callback=progress_callback
)
```

## System Requirements

### Minimum Requirements

- Python 3.8+
- PyTorch 2.0+
- Diffusers 0.21.0+
- Transformers 4.25.0+
- 8GB VRAM (for T2V models)
- 16GB RAM

### Recommended Requirements

- Python 3.10+
- PyTorch 2.1+
- 12GB+ VRAM
- 32GB RAM
- CUDA 12.1+

## Testing

Run the comprehensive test suite:

```bash
# Unit tests
python -m pytest test_architecture_detector.py -v
python -m pytest test_pipeline_manager.py -v
python -m pytest test_wan_pipeline_loader.py -v

# Integration tests
python -m pytest test_wan_model_compatibility_integration.py -v

# Performance tests
python -m pytest test_performance_monitor.py -v
```

## Support

For issues with WAN model compatibility:

1. Run the debug script to identify the problem
2. Apply the compatibility fix if needed
3. Check the performance monitor for optimization opportunities
4. Refer to the troubleshooting section for common solutions

The system is designed to be robust and provide clear error messages to help diagnose and resolve issues quickly.
