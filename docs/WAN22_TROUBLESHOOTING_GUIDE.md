# WAN22 System Optimization Troubleshooting Guide

## Quick Diagnostic Checklist

Before diving into specific issues, run this quick diagnostic:

```python
from wan22_system_optimizer import WAN22SystemOptimizer

# Quick system check
optimizer = WAN22SystemOptimizer.create_default()
diagnostic = optimizer.run_quick_diagnostic()
print(diagnostic.get_summary())
```

## Common Error Categories

### 1. Hardware Detection Issues

#### VRAM Detection Failures

**Error Messages:**

- `"Unable to detect VRAM capacity"`
- `"NVIDIA ML library not available"`
- `"GPU not found or not accessible"`

**Diagnostic Steps:**

1. Check NVIDIA driver installation:
   ```cmd
   nvidia-smi
   ```
2. Verify CUDA installation:
   ```python
   import torch
   print(torch.cuda.is_available())
   print(torch.cuda.get_device_properties(0))
   ```
3. Test NVIDIA ML library:
   ```python
   try:
       import pynvml
       pynvml.nvmlInit()
       print("NVML available")
   except ImportError:
       print("NVML not available - install nvidia-ml-py")
   ```

**Solutions:**

1. **Update NVIDIA Drivers:**

   - Download latest drivers (535.xx+ recommended)
   - Perform clean installation
   - Restart system after installation

2. **Install Missing Dependencies:**

   ```bash
   pip install nvidia-ml-py3
   pip install pynvml
   ```

3. **Manual VRAM Configuration:**
   ```json
   {
     "vram_config": {
       "manual_override": true,
       "total_vram_gb": 16,
       "available_vram_gb": 14,
       "device_id": 0
     }
   }
   ```

#### CPU Detection Issues

**Error Messages:**

- `"Unable to detect CPU specifications"`
- `"CPU core count detection failed"`

**Solutions:**

1. **Manual CPU Configuration:**

   ```json
   {
     "cpu_config": {
       "model": "AMD Ryzen Threadripper PRO 5995WX",
       "cores": 64,
       "threads": 128,
       "base_frequency": 2.7,
       "boost_frequency": 4.5
     }
   }
   ```

2. **Install System Information Tools:**
   ```bash
   pip install psutil
   pip install py-cpuinfo
   ```

### 2. Model Loading Issues

#### TI2V-5B Loading Failures

**Error Messages:**

- `"Failed to load TI2V-5B model"`
- `"Out of memory during model loading"`
- `"Model file corrupted or incomplete"`

**Diagnostic Steps:**

1. Check available VRAM:

   ```python
   from vram_manager import VRAMManager
   vram = VRAMManager()
   print(f"Available VRAM: {vram.get_available_vram_gb()}GB")
   ```

2. Verify model files:
   ```python
   from model_loading_manager import ModelLoadingManager
   manager = ModelLoadingManager()
   validation = manager.validate_model_files("TI2V-5B")
   print(validation.get_report())
   ```

**Solutions:**

1. **Enable CPU Offloading:**

   ```json
   {
     "model_loading": {
       "cpu_offload_enabled": true,
       "offload_components": ["text_encoder", "vae"],
       "sequential_loading": true
     }
   }
   ```

2. **Use Model Fallbacks:**

   ```python
   fallback_models = manager.get_fallback_models("TI2V-5B")
   for model in fallback_models:
       try:
           result = manager.load_model(model)
           if result.success:
               break
       except Exception as e:
           continue
   ```

3. **Clear Model Cache:**
   ```python
   manager.clear_model_cache()
   manager.clear_temporary_files()
   ```

#### Quantization Issues

**Error Messages:**

- `"Quantization timeout after 300 seconds"`
- `"Quantization failed with unknown error"`
- `"Quantized model quality degradation detected"`

**Solutions:**

1. **Increase Timeout:**

   ```json
   {
     "quantization": {
       "timeout_seconds": 600,
       "extended_timeout_enabled": true,
       "progress_monitoring": true
     }
   }
   ```

2. **Disable Problematic Quantization:**

   ```json
   {
     "quantization": {
       "strategy": "none",
       "fallback_enabled": false,
       "skip_validation": false
     }
   }
   ```

3. **Use Alternative Quantization Methods:**
   ```json
   {
     "quantization": {
       "primary_method": "bf16",
       "fallback_methods": ["int8", "none"],
       "quality_threshold": 0.95
     }
   }
   ```

### 3. Configuration Issues

#### Syntax Errors in Configuration Files

**Error Messages:**

- `"Syntax error in ui_event_handlers_enhanced.py at line 187"`
- `"Invalid configuration attribute: clip_output"`
- `"Configuration schema validation failed"`

**Automatic Repair:**

```python
from syntax_validator import SyntaxValidator

validator = SyntaxValidator()
result = validator.validate_and_repair_file("ui_event_handlers_enhanced.py")
if result.success:
    print("File repaired successfully")
else:
    print(f"Manual intervention required: {result.errors}")
```

**Manual Fixes:**

1. **Common Syntax Issues:**

   ```python
   # Before (incorrect)
   if condition:
       do_something()
   # Missing else clause

   # After (correct)
   if condition:
       do_something()
   else:
       pass
   ```

2. **Configuration Cleanup:**

   ```python
   from config_validator import ConfigValidator

   validator = ConfigValidator()
   result = validator.clean_config_file("config.json")
   print(f"Removed attributes: {result.removed_attributes}")
   ```

#### Model Configuration Mismatches

**Error Messages:**

- `"Model configuration incompatible with current library version"`
- `"Unexpected model attribute detected"`

**Solutions:**

1. **Update Model Configurations:**

   ```python
   validator.update_model_configs_for_library_version()
   ```

2. **Reset to Default Configuration:**
   ```python
   validator.restore_default_model_config("TI2V-5B")
   ```

### 4. Performance Issues

#### Slow Generation Times

**Symptoms:**

- Generation takes longer than expected
- System appears to hang during generation
- High memory usage during generation

**Diagnostic Steps:**

1. **Check System Resources:**

   ```python
   from health_monitor import HealthMonitor

   monitor = HealthMonitor()
   metrics = monitor.get_current_metrics()
   print(f"GPU Usage: {metrics.gpu_usage}%")
   print(f"VRAM Usage: {metrics.vram_usage_gb}GB")
   print(f"CPU Usage: {metrics.cpu_usage}%")
   ```

2. **Performance Profiling:**

   ```python
   from performance_monitor import PerformanceMonitor

   profiler = PerformanceMonitor()
   profile = profiler.profile_generation_task()
   print(profile.get_bottlenecks())
   ```

**Solutions:**

1. **Enable Hardware Optimizations:**

   ```python
   from hardware_optimizer import HardwareOptimizer

   optimizer = HardwareOptimizer()
   optimizer.apply_rtx4080_optimizations()
   optimizer.apply_threadripper_optimizations()
   ```

2. **Adjust Memory Settings:**
   ```json
   {
     "performance": {
       "tile_size": [256, 256],
       "batch_size": 1,
       "enable_attention_slicing": true,
       "enable_cpu_offload": true
     }
   }
   ```

#### Memory Issues

**Error Messages:**

- `"CUDA out of memory"`
- `"System memory exhausted"`
- `"Memory allocation failed"`

**Solutions:**

1. **Enable Memory Optimization:**

   ```python
   from vram_manager import VRAMManager

   vram = VRAMManager()
   vram.enable_automatic_optimization()
   vram.set_optimization_threshold(0.85)  # 85% usage threshold
   ```

2. **Reduce Memory Usage:**
   ```json
   {
     "memory_optimization": {
       "gradient_checkpointing": true,
       "cpu_offload_vae": true,
       "cpu_offload_text_encoder": true,
       "reduce_precision": true
     }
   }
   ```

### 5. System Health Issues

#### Overheating

**Symptoms:**

- GPU temperature above 85°C
- System throttling
- Unexpected shutdowns

**Solutions:**

1. **Enable Thermal Protection:**

   ```python
   from critical_hardware_protection import CriticalHardwareProtection

   protection = CriticalHardwareProtection()
   protection.enable_thermal_protection()
   protection.set_temperature_limits(gpu_max=85, cpu_max=80)
   ```

2. **Adjust Performance Settings:**
   ```json
   {
     "thermal_management": {
       "enable_thermal_throttling": true,
       "temperature_monitoring_interval": 5,
       "emergency_shutdown_temp": 90,
       "performance_reduction_temp": 85
     }
   }
   ```

#### System Instability

**Symptoms:**

- Random crashes
- System freezes
- Application hangs

**Diagnostic Steps:**

1. **System Health Check:**

   ```python
   health_report = monitor.generate_comprehensive_health_report()
   print(health_report.get_stability_analysis())
   ```

2. **Error Pattern Analysis:**

   ```python
   from error_recovery_system import ErrorRecoverySystem

   recovery = ErrorRecoverySystem()
   patterns = recovery.analyze_error_patterns()
   print(patterns.get_recommendations())
   ```

**Solutions:**

1. **Enable System Monitoring:**

   ```python
   monitor.enable_continuous_monitoring()
   monitor.set_alert_thresholds({
       'gpu_temp': 85,
       'vram_usage': 0.95,
       'cpu_usage': 0.8,
       'memory_usage': 0.9
   })
   ```

2. **Implement Error Recovery:**
   ```python
   recovery.enable_automatic_recovery()
   recovery.set_recovery_strategies([
       'restart_pipeline',
       'clear_cache',
       'reduce_batch_size',
       'enable_cpu_offload'
   ])
   ```

## Advanced Troubleshooting

### Log Analysis

**Enable Detailed Logging:**

```json
{
  "logging": {
    "level": "DEBUG",
    "components": [
      "vram_manager",
      "model_loading",
      "quantization",
      "hardware_optimizer",
      "health_monitor"
    ],
    "file_rotation": true,
    "max_size_mb": 100
  }
}
```

**Analyze Log Patterns:**

```python
from log_analyzer import LogAnalyzer

analyzer = LogAnalyzer()
patterns = analyzer.analyze_recent_logs(hours=24)
print(patterns.get_error_summary())
print(patterns.get_performance_trends())
```

### System State Recovery

**Save System State:**

```python
from error_recovery_system import ErrorRecoverySystem

recovery = ErrorRecoverySystem()
state_file = recovery.save_system_state()
print(f"System state saved to: {state_file}")
```

**Restore from Backup:**

```python
recovery.restore_system_state("backup_state_20240815.json")
```

### Performance Benchmarking

**Run Comprehensive Benchmarks:**

```python
from performance_benchmark_system import PerformanceBenchmarkSystem

benchmark = PerformanceBenchmarkSystem()
results = benchmark.run_comprehensive_benchmarks()
print(results.get_performance_report())
```

**Compare Performance:**

```python
baseline = benchmark.load_baseline_results()
comparison = benchmark.compare_with_baseline(results, baseline)
print(comparison.get_improvement_summary())
```

## Emergency Procedures

### System Recovery Mode

If the system becomes completely unresponsive:

1. **Safe Mode Startup:**

   ```python
   # Create emergency_startup.py
   from wan22_system_optimizer import WAN22SystemOptimizer

   optimizer = WAN22SystemOptimizer.create_safe_mode()
   optimizer.disable_all_optimizations()
   optimizer.use_minimal_configuration()
   ```

2. **Reset All Configurations:**
   ```python
   optimizer.reset_to_factory_defaults()
   optimizer.clear_all_caches()
   optimizer.restore_backup_configurations()
   ```

### Data Recovery

**Recover Generated Content:**

```python
from output_recovery import OutputRecovery

recovery = OutputRecovery()
recovered_files = recovery.scan_for_recoverable_outputs()
recovery.restore_outputs(recovered_files, "recovered_outputs/")
```

**Recover System State:**

```python
state_files = recovery.find_system_state_backups()
latest_state = recovery.get_latest_valid_state(state_files)
recovery.restore_system_state(latest_state)
```

## Prevention and Maintenance

### Regular Maintenance Tasks

1. **Weekly System Health Check:**

   ```python
   # Add to scheduled tasks
   health_report = monitor.generate_weekly_health_report()
   if health_report.has_issues():
       health_report.apply_recommended_fixes()
   ```

2. **Monthly Performance Optimization:**

   ```python
   optimizer.run_monthly_optimization()
   optimizer.update_hardware_profiles()
   optimizer.clean_temporary_files()
   ```

3. **Configuration Validation:**
   ```python
   validator.validate_all_configurations()
   validator.update_deprecated_settings()
   validator.backup_current_configurations()
   ```

### Monitoring Setup

**Automated Monitoring:**

```json
{
  "monitoring": {
    "enabled": true,
    "interval_seconds": 30,
    "alerts": {
      "email_notifications": false,
      "system_notifications": true,
      "log_alerts": true
    },
    "thresholds": {
      "gpu_temp_warning": 80,
      "gpu_temp_critical": 85,
      "vram_usage_warning": 0.9,
      "vram_usage_critical": 0.95
    }
  }
}
```

---

_For additional support, generate a diagnostic report and include it when seeking help from the community or support channels._
