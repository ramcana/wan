# WAN22 System Optimization User Guide

## Overview

The WAN22 System Optimization framework provides comprehensive system monitoring, optimization, and error recovery capabilities specifically designed for high-end hardware configurations. This guide covers configuration, optimization settings, troubleshooting, and performance optimization for users running WAN2.2 UI on systems with RTX 4080 GPUs and Threadripper PRO processors.

## Table of Contents

1. [Quick Start](#quick-start)
2. [System Requirements](#system-requirements)
3. [Configuration Settings](#configuration-settings)
4. [Optimization Features](#optimization-features)
5. [Performance Monitoring](#performance-monitoring)
6. [Troubleshooting Guide](#troubleshooting-guide)
7. [Advanced Configuration](#advanced-configuration)
8. [Performance Optimization](#performance-optimization)

## Quick Start

### First-Time Setup for High-End Hardware

If you're running WAN2.2 UI on high-end hardware (RTX 4080, Threadripper PRO, 128GB+ RAM), the system will automatically detect your configuration and apply optimal settings:

1. **Launch the application** - The system optimizer will automatically initialize
2. **Hardware Detection** - Your RTX 4080 and system specifications will be detected
3. **Automatic Optimization** - Optimal settings will be applied based on your hardware
4. **Validation** - System will validate all configurations and repair any issues

### Manual Initialization

If automatic detection fails, you can manually initialize the optimizer:

```python
from wan22_system_optimizer import WAN22SystemOptimizer
from hardware_optimizer import HardwareProfile

# Create hardware profile
profile = HardwareProfile(
    cpu_model="AMD Ryzen Threadripper PRO 5995WX",
    cpu_cores=64,
    total_memory_gb=128,
    gpu_model="NVIDIA GeForce RTX 4080",
    vram_gb=16,
    cuda_version="12.1",
    driver_version="latest"
)

# Initialize optimizer
optimizer = WAN22SystemOptimizer("config.json", profile)
result = optimizer.initialize_system()
```

## System Requirements

### Minimum Requirements

- **GPU**: NVIDIA RTX 3060 (12GB VRAM) or better
- **CPU**: Intel i7-10700K or AMD Ryzen 7 3700X
- **RAM**: 32GB DDR4
- **Storage**: 100GB free space (SSD recommended)
- **OS**: Windows 10/11 64-bit

### Recommended Configuration (Optimized)

- **GPU**: NVIDIA RTX 4080 (16GB VRAM)
- **CPU**: AMD Threadripper PRO 5995WX (64 cores)
- **RAM**: 128GB DDR4-3200
- **Storage**: 500GB NVMe SSD
- **OS**: Windows 11 Pro 64-bit

### Software Dependencies

- Python 3.11+
- PyTorch 2.4+ with CUDA 12.1
- Diffusers 0.35+
- NVIDIA drivers 535.xx or newer

## Configuration Settings

### VRAM Management

The system automatically detects and manages VRAM usage:

**Automatic Detection:**

- Primary: NVIDIA ML (nvml) library
- Secondary: PyTorch CUDA memory info
- Fallback: Manual configuration

**Manual VRAM Configuration:**

```json
{
  "vram_config": {
    "total_vram_gb": 16,
    "reserved_vram_gb": 2,
    "optimization_threshold": 0.9,
    "enable_cpu_offload": true
  }
}
```

**VRAM Optimization Settings:**

- **Memory Threshold**: Automatically optimize when usage exceeds 90%
- **CPU Offloading**: Move text encoder and VAE to CPU when needed
- **Gradient Checkpointing**: Reduce memory usage for large models
- **Dynamic Batching**: Adjust batch sizes based on available memory

### Quantization Settings

Configure quantization behavior for optimal performance:

```json
{
  "quantization": {
    "strategy": "auto",
    "timeout_seconds": 300,
    "fallback_enabled": true,
    "quality_validation": true,
    "supported_methods": ["bf16", "int8", "FP8", "none"]
  }
}
```

**Quantization Strategies:**

- **bf16**: Best quality, moderate speed, high VRAM usage
- **int8**: Balanced quality/performance, lower VRAM usage
- **FP8**: Experimental, highest performance (RTX 4080 optimized)
- **none**: No quantization, highest quality, highest VRAM usage

### Hardware Optimization

**RTX 4080 Specific Settings:**

```json
{
  "rtx4080_optimizations": {
    "enable_tensor_cores": true,
    "optimal_tile_size": [256, 256],
    "memory_allocation_strategy": "optimized",
    "cpu_offload_components": ["text_encoder", "vae"]
  }
}
```

**Threadripper PRO Settings:**

```json
{
  "threadripper_optimizations": {
    "cpu_cores_utilized": 32,
    "numa_aware_allocation": true,
    "parallel_preprocessing": true,
    "thread_allocation_strategy": "balanced"
  }
}
```

## Optimization Features

### Automatic System Optimization

The system provides several automatic optimization features:

1. **Syntax Validation and Repair**

   - Automatically detects and repairs Python syntax errors
   - Creates backups before making changes
   - Validates critical files on startup

2. **Configuration Cleanup**

   - Removes unsupported attributes from config files
   - Validates model compatibility
   - Creates backups before modifications

3. **Memory Management**

   - Real-time VRAM monitoring
   - Automatic memory optimization
   - CPU offloading when needed

4. **Performance Tuning**
   - Hardware-specific optimizations
   - Automatic parameter adjustment
   - Performance benchmarking

### Manual Optimization Controls

**Enable/Disable Features:**

```python
# Enable specific optimizations
optimizer.enable_vram_optimization(True)
optimizer.enable_quantization_fallback(True)
optimizer.enable_hardware_optimizations(True)

# Configure monitoring
optimizer.set_monitoring_interval(5)  # seconds
optimizer.set_health_thresholds({
    'gpu_temp_max': 85,
    'vram_usage_max': 0.95,
    'cpu_usage_max': 0.8
})
```

## Performance Monitoring

### Real-Time Monitoring Dashboard

The system provides comprehensive real-time monitoring:

**GPU Metrics:**

- Temperature monitoring
- VRAM usage and allocation
- GPU utilization percentage
- Power consumption

**CPU Metrics:**

- CPU usage per core
- Memory usage
- Thread allocation
- NUMA node utilization

**System Metrics:**

- Generation speed (images/minute)
- Model loading times
- Error rates
- System stability indicators

### Performance Benchmarks

**Model Loading Benchmarks:**

- TI2V-5B model: Target <5 minutes
- Standard models: Target <2 minutes
- VRAM usage: Target <12GB for TI2V-5B

**Generation Speed Benchmarks:**

- 2-second video: Target <2 minutes
- Single image: Target <30 seconds
- Batch processing: Scales with hardware

### Health Monitoring Alerts

The system provides automatic alerts for:

- GPU temperature exceeding 85°C
- VRAM usage exceeding 95%
- CPU usage exceeding 80%
- System instability detection
- Model loading failures

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. VRAM Detection Failures

**Symptoms:**

- "Unable to detect VRAM" error messages
- Incorrect VRAM capacity displayed
- Out-of-memory errors with sufficient VRAM

**Solutions:**

1. **Update NVIDIA Drivers:**

   ```bash
   # Download latest drivers from NVIDIA website
   # Recommended: 535.xx or newer for RTX 4080
   ```

2. **Manual VRAM Configuration:**

   ```json
   {
     "vram_override": {
       "enabled": true,
       "total_vram_gb": 16,
       "device_id": 0
     }
   }
   ```

3. **Restart NVIDIA Services:**
   ```cmd
   net stop NVDisplay.ContainerLocalSystem
   net start NVDisplay.ContainerLocalSystem
   ```

#### 2. Quantization Timeouts

**Symptoms:**

- System hangs during model loading
- "Quantization timeout" error messages
- Slow model initialization

**Solutions:**

1. **Increase Timeout:**

   ```json
   {
     "quantization": {
       "timeout_seconds": 600,
       "extended_timeout": true
     }
   }
   ```

2. **Disable Quantization:**

   ```json
   {
     "quantization": {
       "strategy": "none",
       "fallback_enabled": false
     }
   }
   ```

3. **Use Alternative Quantization:**
   ```json
   {
     "quantization": {
       "strategy": "bf16",
       "fallback_to_int8": true
     }
   }
   ```

#### 3. Configuration Errors

**Symptoms:**

- "Unexpected attribute" warnings
- Configuration validation failures
- Model compatibility issues

**Solutions:**

1. **Automatic Cleanup:**

   ```python
   from config_validator import ConfigValidator
   validator = ConfigValidator()
   result = validator.clean_unexpected_attributes("config.json")
   ```

2. **Reset to Defaults:**

   ```python
   validator.restore_default_config("config.json")
   ```

3. **Manual Configuration:**
   - Remove unsupported attributes like `clip_output`
   - Validate model-specific settings
   - Check library compatibility

#### 4. Performance Issues

**Symptoms:**

- Slow generation times
- High memory usage
- System instability

**Solutions:**

1. **Enable Hardware Optimizations:**

   ```python
   optimizer.apply_hardware_optimizations()
   ```

2. **Adjust Memory Settings:**

   ```json
   {
     "memory_optimization": {
       "enable_cpu_offload": true,
       "gradient_checkpointing": true,
       "dynamic_batching": true
     }
   }
   ```

3. **Monitor System Health:**
   ```python
   health_status = optimizer.monitor_system_health()
   print(health_status.get_recommendations())
   ```

#### 5. Model Loading Failures

**Symptoms:**

- "Failed to load model" errors
- Incomplete model initialization
- Memory allocation errors

**Solutions:**

1. **Check Model Compatibility:**

   ```python
   from model_loading_manager import ModelLoadingManager
   manager = ModelLoadingManager()
   compatibility = manager.check_model_compatibility("TI2V-5B")
   ```

2. **Use Fallback Models:**

   ```python
   fallback_options = manager.get_fallback_models("TI2V-5B")
   ```

3. **Clear Model Cache:**
   ```python
   manager.clear_model_cache()
   ```

### Error Recovery

The system provides automatic error recovery:

1. **Automatic Recovery:**

   - Exponential backoff retry
   - Fallback configurations
   - State preservation

2. **Manual Recovery:**

   ```python
   recovery_system = ErrorRecoverySystem()
   recovery_system.attempt_recovery(error, context)
   ```

3. **System Restoration:**
   ```python
   recovery_system.restore_system_state("backup_state.json")
   ```

## Advanced Configuration

### Custom Hardware Profiles

Create custom hardware profiles for non-standard configurations:

```python
custom_profile = HardwareProfile(
    cpu_model="Custom CPU",
    cpu_cores=32,
    total_memory_gb=64,
    gpu_model="Custom GPU",
    vram_gb=24,
    cuda_version="12.1",
    driver_version="535.xx"
)

optimizer = WAN22SystemOptimizer("config.json", custom_profile)
```

### Performance Tuning Parameters

**Advanced VRAM Management:**

```json
{
  "advanced_vram": {
    "memory_pool_size": 14000,
    "fragmentation_threshold": 0.1,
    "garbage_collection_interval": 30,
    "preallocation_strategy": "conservative"
  }
}
```

**CPU Optimization:**

```json
{
  "cpu_optimization": {
    "thread_pool_size": 16,
    "numa_binding": true,
    "cpu_affinity": [0, 1, 2, 3, 4, 5, 6, 7],
    "priority_class": "high"
  }
}
```

### Logging Configuration

**Detailed Logging:**

```json
{
  "logging": {
    "level": "DEBUG",
    "file_rotation": true,
    "max_file_size_mb": 100,
    "backup_count": 5,
    "performance_logging": true
  }
}
```

## Performance Optimization

### Best Practices for RTX 4080

1. **Memory Management:**

   - Keep VRAM usage below 14GB for optimal performance
   - Enable CPU offloading for non-critical components
   - Use gradient checkpointing for large models

2. **Quantization Strategy:**

   - Use bf16 for best quality/performance balance
   - Consider int8 for memory-constrained scenarios
   - Test FP8 for maximum performance (experimental)

3. **Thermal Management:**
   - Monitor GPU temperature (keep below 85°C)
   - Ensure adequate case ventilation
   - Consider undervolting for sustained workloads

### Threadripper PRO Optimization

1. **CPU Utilization:**

   - Use 50-75% of available cores for AI workloads
   - Reserve cores for system processes
   - Enable NUMA-aware memory allocation

2. **Memory Configuration:**

   - Use high-speed DDR4 (3200MHz+)
   - Enable XMP profiles
   - Monitor memory bandwidth utilization

3. **Storage Optimization:**
   - Use NVMe SSD for model storage
   - Enable write caching
   - Consider RAID 0 for multiple drives

### Performance Monitoring and Tuning

**Continuous Optimization:**

```python
# Enable performance monitoring
optimizer.enable_performance_monitoring(True)

# Set optimization targets
optimizer.set_performance_targets({
    'model_loading_time_max': 300,  # 5 minutes
    'generation_speed_min': 0.5,   # images per second
    'vram_efficiency_min': 0.8     # 80% efficiency
})

# Get optimization recommendations
recommendations = optimizer.get_optimization_recommendations()
```

**Benchmark Validation:**

```python
# Run performance benchmarks
benchmark_results = optimizer.run_performance_benchmarks()

# Validate against targets
validation_results = optimizer.validate_performance_targets()

# Generate optimization report
report = optimizer.generate_performance_report()
```

## Support and Resources

### Getting Help

1. **Check System Status:**

   ```python
   status = optimizer.get_system_status()
   print(status.get_health_summary())
   ```

2. **Generate Diagnostic Report:**

   ```python
   diagnostic_report = optimizer.generate_diagnostic_report()
   ```

3. **View Optimization History:**
   ```python
   history = optimizer.get_optimization_history()
   ```

### Additional Resources

- **Performance Benchmarking Guide**: See `wan22_performance_benchmarks.py`
- **Hardware Compatibility**: Check `hardware_optimizer.py`
- **Error Recovery**: Reference `error_recovery_system.py`
- **Health Monitoring**: Use `health_monitor.py`

### Community Support

- Report issues with detailed system information
- Include diagnostic reports when seeking help
- Share performance benchmarks for your hardware configuration
- Contribute optimization profiles for new hardware

---

_This guide covers the comprehensive WAN22 System Optimization framework. For specific implementation details, refer to the individual component documentation and source code._
