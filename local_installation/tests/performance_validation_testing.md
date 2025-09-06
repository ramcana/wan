# Performance Validation Testing Guide

This guide provides comprehensive procedures for validating the performance of the WAN2.2 local installation system across different hardware configurations. It includes benchmarking methodologies, performance baselines, and optimization verification procedures.

## Overview

Performance validation ensures that:

- Installation completes within acceptable timeframes
- Hardware-specific optimizations are effective
- System resources are used efficiently
- Generated configurations provide optimal performance
- Performance meets user expectations across hardware tiers

## Performance Testing Categories

### 1. Installation Performance Testing

### 2. Hardware Optimization Validation

### 3. Runtime Performance Testing

### 4. Resource Utilization Testing

### 5. Scalability and Stress Testing

---

## 1. Installation Performance Testing

### 1.1 Installation Time Benchmarks

**Objective**: Measure and validate installation completion times across hardware configurations.

#### Test Procedure

1. Start with clean system (no existing Python/WAN2.2)
2. Record start time when `install.bat` is executed
3. Record completion time when installation finishes
4. Measure individual phase durations

#### Performance Baselines

| Hardware Tier | Total Time | Detection | Dependencies | Models    | Config | Validation |
| ------------- | ---------- | --------- | ------------ | --------- | ------ | ---------- |
| High-End      | 10-15 min  | < 30s     | 2-3 min      | 5-8 min   | < 30s  | 1-2 min    |
| Mid-Range     | 15-25 min  | < 45s     | 3-5 min      | 8-12 min  | < 45s  | 2-3 min    |
| Budget        | 20-35 min  | < 60s     | 5-8 min      | 10-15 min | < 60s  | 3-5 min    |
| Minimum       | 30-50 min  | < 90s     | 8-12 min     | 15-25 min | < 90s  | 5-8 min    |

#### Measurement Script

```python
import time
import json
from datetime import datetime

class InstallationPerformanceTracker:
    def __init__(self):
        self.start_time = None
        self.phase_times = {}
        self.current_phase = None

    def start_installation(self):
        self.start_time = time.time()
        self.log_event("installation_started")

    def start_phase(self, phase_name):
        current_time = time.time()
        if self.current_phase:
            self.end_phase()
        self.current_phase = phase_name
        self.phase_times[phase_name] = {"start": current_time}
        self.log_event(f"phase_started_{phase_name}")

    def end_phase(self):
        if self.current_phase:
            current_time = time.time()
            phase_data = self.phase_times[self.current_phase]
            phase_data["end"] = current_time
            phase_data["duration"] = current_time - phase_data["start"]
            self.log_event(f"phase_completed_{self.current_phase}")
            self.current_phase = None

    def end_installation(self):
        if self.current_phase:
            self.end_phase()
        end_time = time.time()
        total_duration = end_time - self.start_time
        self.log_event("installation_completed", {"total_duration": total_duration})
        return self.generate_report()

    def generate_report(self):
        return {
            "timestamp": datetime.now().isoformat(),
            "total_duration": time.time() - self.start_time,
            "phase_durations": {
                phase: data["duration"]
                for phase, data in self.phase_times.items()
                if "duration" in data
            }
        }
```

#### Manual Verification Checklist

- [ ] Installation starts immediately when executed
- [ ] No unnecessary delays between phases
- [ ] Progress indicators update smoothly
- [ ] System remains responsive during installation
- [ ] Installation completes within baseline timeframe
- [ ] No performance regressions compared to previous versions

### 1.2 Download Performance Testing

**Objective**: Validate model download performance and efficiency.

#### Test Procedure

1. Clear any existing model cache
2. Monitor download speeds and progress
3. Test parallel download efficiency
4. Validate resume functionality

#### Performance Metrics

- **Download Speed**: Should utilize 80%+ of available bandwidth
- **Parallel Efficiency**: Multiple models download simultaneously
- **Resume Reliability**: Downloads resume correctly after interruption
- **Progress Accuracy**: Progress indicators reflect actual download status

#### Test Script

```python
import requests
import time
from concurrent.futures import ThreadPoolExecutor

class DownloadPerformanceTester:
    def __init__(self):
        self.download_stats = {}

    def test_download_speed(self, url, expected_size):
        start_time = time.time()
        downloaded = 0

        response = requests.get(url, stream=True)
        for chunk in response.iter_content(chunk_size=8192):
            downloaded += len(chunk)

        end_time = time.time()
        duration = end_time - start_time
        speed_mbps = (downloaded / (1024 * 1024)) / duration

        return {
            "url": url,
            "size_mb": downloaded / (1024 * 1024),
            "duration": duration,
            "speed_mbps": speed_mbps
        }

    def test_parallel_downloads(self, urls):
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(self.test_download_speed, url, 0) for url in urls]
            results = [future.result() for future in futures]
        return results
```

---

## 2. Hardware Optimization Validation

### 2.1 CPU Optimization Testing

**Objective**: Validate CPU thread allocation and utilization optimization.

#### Test Procedure

1. Install on systems with different CPU configurations
2. Review generated CPU thread settings
3. Measure CPU utilization during model operations
4. Validate thread scaling efficiency

#### CPU Configuration Validation

| CPU Type                | Cores/Threads | Expected Thread Allocation | Max CPU Usage |
| ----------------------- | ------------- | -------------------------- | ------------- |
| Threadripper PRO 5995WX | 64C/128T      | 64-96 threads              | 70-85%        |
| Ryzen 9 5950X           | 16C/32T       | 16-24 threads              | 75-90%        |
| Ryzen 7 5800X           | 8C/16T        | 8-12 threads               | 80-95%        |
| Intel i7-12700K         | 12C/20T       | 12-16 threads              | 80-95%        |
| Ryzen 5 3600            | 6C/12T        | 6-8 threads                | 85-95%        |
| Intel i5-8400           | 6C/6T         | 4-6 threads                | 90-100%       |

#### CPU Performance Test Script

```python
import psutil
import time
import threading
from concurrent.futures import ThreadPoolExecutor

class CPUOptimizationTester:
    def __init__(self):
        self.cpu_count = psutil.cpu_count()
        self.cpu_count_logical = psutil.cpu_count(logical=True)

    def test_thread_allocation(self, config_threads):
        """Test if thread allocation is appropriate for CPU."""
        optimal_min = self.cpu_count
        optimal_max = self.cpu_count_logical

        if config_threads < optimal_min * 0.5:
            return "under_allocated"
        elif config_threads > optimal_max * 1.2:
            return "over_allocated"
        else:
            return "optimal"

    def measure_cpu_utilization(self, duration=60):
        """Measure CPU utilization during load test."""
        cpu_percentages = []

        def cpu_monitor():
            for _ in range(duration):
                cpu_percentages.append(psutil.cpu_percent(interval=1))

        monitor_thread = threading.Thread(target=cpu_monitor)
        monitor_thread.start()

        # Simulate CPU load
        self.simulate_cpu_load(duration)

        monitor_thread.join()

        return {
            "average_cpu": sum(cpu_percentages) / len(cpu_percentages),
            "max_cpu": max(cpu_percentages),
            "min_cpu": min(cpu_percentages)
        }

    def simulate_cpu_load(self, duration):
        """Simulate CPU-intensive workload."""
        def cpu_task():
            end_time = time.time() + duration
            while time.time() < end_time:
                # CPU-intensive calculation
                sum(i * i for i in range(1000))

        with ThreadPoolExecutor(max_workers=self.cpu_count) as executor:
            futures = [executor.submit(cpu_task) for _ in range(self.cpu_count)]
            for future in futures:
                future.result()
```

### 2.2 Memory Optimization Testing

**Objective**: Validate memory allocation and usage optimization.

#### Test Procedure

1. Review generated memory allocation settings
2. Monitor memory usage during model loading
3. Test memory efficiency under different loads
4. Validate memory leak prevention

#### Memory Configuration Validation

| System RAM | Expected Allocation | Max Usage | Swap Usage |
| ---------- | ------------------- | --------- | ---------- |
| 128GB      | 32-64GB             | < 80%     | None       |
| 64GB       | 16-32GB             | < 85%     | Minimal    |
| 32GB       | 8-16GB              | < 90%     | < 2GB      |
| 16GB       | 4-8GB               | < 95%     | < 4GB      |
| 8GB        | 2-4GB               | < 98%     | < 2GB      |

#### Memory Performance Test Script

```python
import psutil
import time
import gc

class MemoryOptimizationTester:
    def __init__(self):
        self.memory_info = psutil.virtual_memory()
        self.initial_memory = self.memory_info.used

    def test_memory_allocation(self, config_memory_gb):
        """Test if memory allocation is appropriate."""
        total_gb = self.memory_info.total / (1024**3)
        available_gb = self.memory_info.available / (1024**3)

        if config_memory_gb > available_gb * 0.9:
            return "over_allocated"
        elif config_memory_gb < available_gb * 0.1:
            return "under_allocated"
        else:
            return "optimal"

    def monitor_memory_usage(self, duration=300):
        """Monitor memory usage over time."""
        memory_samples = []

        for _ in range(duration):
            memory = psutil.virtual_memory()
            memory_samples.append({
                "timestamp": time.time(),
                "used_gb": memory.used / (1024**3),
                "available_gb": memory.available / (1024**3),
                "percent": memory.percent
            })
            time.sleep(1)

        return {
            "samples": memory_samples,
            "peak_usage": max(sample["used_gb"] for sample in memory_samples),
            "average_usage": sum(sample["used_gb"] for sample in memory_samples) / len(memory_samples),
            "memory_growth": memory_samples[-1]["used_gb"] - memory_samples[0]["used_gb"]
        }

    def test_memory_leak(self, iterations=10):
        """Test for memory leaks during repeated operations."""
        initial_memory = psutil.virtual_memory().used
        memory_readings = [initial_memory]

        for i in range(iterations):
            # Simulate memory-intensive operation
            self.simulate_memory_operation()
            gc.collect()  # Force garbage collection
            time.sleep(2)  # Allow cleanup

            current_memory = psutil.virtual_memory().used
            memory_readings.append(current_memory)

        # Check for consistent memory growth (potential leak)
        memory_growth = [(memory_readings[i+1] - memory_readings[i]) / (1024**2)
                        for i in range(len(memory_readings)-1)]

        return {
            "memory_readings_mb": [m / (1024**2) for m in memory_readings],
            "memory_growth_mb": memory_growth,
            "total_growth_mb": (memory_readings[-1] - memory_readings[0]) / (1024**2),
            "leak_detected": sum(memory_growth) > 100  # More than 100MB growth
        }
```

### 2.3 GPU Optimization Testing

**Objective**: Validate GPU utilization and VRAM optimization.

#### Test Procedure

1. Review GPU-specific configuration settings
2. Monitor GPU utilization during operations
3. Test VRAM allocation and usage
4. Validate GPU acceleration effectiveness

#### GPU Configuration Validation

| GPU Model   | VRAM | Expected Allocation | Target Utilization |
| ----------- | ---- | ------------------- | ------------------ |
| RTX 4080    | 16GB | 14-15GB             | 85-95%             |
| RTX 4070    | 12GB | 10-11GB             | 85-95%             |
| RTX 3070    | 8GB  | 6-7GB               | 80-90%             |
| RTX 3060    | 12GB | 10-11GB             | 80-90%             |
| GTX 1660 Ti | 6GB  | 4-5GB               | 75-85%             |
| GTX 1060    | 6GB  | 4-5GB               | 75-85%             |

#### GPU Performance Test Script

```python
import subprocess
import json
import time
import re

class GPUOptimizationTester:
    def __init__(self):
        self.gpu_info = self.get_gpu_info()

    def get_gpu_info(self):
        """Get GPU information using nvidia-smi."""
        try:
            result = subprocess.run([
                "nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits"
            ], capture_output=True, text=True)

            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                gpus = []
                for line in lines:
                    parts = [part.strip() for part in line.split(',')]
                    gpus.append({
                        "name": parts[0],
                        "memory_total": int(parts[1]),
                        "memory_used": int(parts[2]),
                        "utilization": int(parts[3])
                    })
                return gpus
            else:
                return []
        except Exception:
            return []

    def test_vram_allocation(self, config_vram_gb):
        """Test if VRAM allocation is appropriate."""
        if not self.gpu_info:
            return "no_gpu"

        gpu = self.gpu_info[0]  # Primary GPU
        total_vram_gb = gpu["memory_total"] / 1024

        if config_vram_gb > total_vram_gb * 0.95:
            return "over_allocated"
        elif config_vram_gb < total_vram_gb * 0.5:
            return "under_allocated"
        else:
            return "optimal"

    def monitor_gpu_utilization(self, duration=300):
        """Monitor GPU utilization over time."""
        utilization_samples = []

        for _ in range(duration):
            gpu_info = self.get_gpu_info()
            if gpu_info:
                utilization_samples.append({
                    "timestamp": time.time(),
                    "gpu_utilization": gpu_info[0]["utilization"],
                    "memory_used_gb": gpu_info[0]["memory_used"] / 1024,
                    "memory_total_gb": gpu_info[0]["memory_total"] / 1024
                })
            time.sleep(1)

        if utilization_samples:
            return {
                "samples": utilization_samples,
                "average_utilization": sum(s["gpu_utilization"] for s in utilization_samples) / len(utilization_samples),
                "peak_utilization": max(s["gpu_utilization"] for s in utilization_samples),
                "average_vram_usage": sum(s["memory_used_gb"] for s in utilization_samples) / len(utilization_samples),
                "peak_vram_usage": max(s["memory_used_gb"] for s in utilization_samples)
            }
        else:
            return {"error": "No GPU data collected"}
```

---

## 3. Runtime Performance Testing

### 3.1 Model Loading Performance

**Objective**: Validate model loading times and efficiency.

#### Performance Baselines

| Hardware Tier | Model Loading Time | Memory Usage | VRAM Usage |
| ------------- | ------------------ | ------------ | ---------- |
| High-End      | 15-30 seconds      | < 16GB       | < 12GB     |
| Mid-Range     | 30-60 seconds      | < 12GB       | < 8GB      |
| Budget        | 60-120 seconds     | < 8GB        | < 6GB      |
| Minimum       | 120-180 seconds    | < 6GB        | < 4GB      |
| CPU-Only      | 180-300 seconds    | < 8GB        | N/A        |

#### Model Loading Test Script

```python
import time
import psutil
from contextlib import contextmanager

class ModelLoadingTester:
    def __init__(self):
        self.gpu_tester = GPUOptimizationTester()

    @contextmanager
    def performance_monitor(self):
        """Context manager to monitor performance during model loading."""
        start_time = time.time()
        start_memory = psutil.virtual_memory().used
        start_gpu = self.gpu_tester.get_gpu_info()

        yield

        end_time = time.time()
        end_memory = psutil.virtual_memory().used
        end_gpu = self.gpu_tester.get_gpu_info()

        self.last_performance = {
            "duration": end_time - start_time,
            "memory_increase_gb": (end_memory - start_memory) / (1024**3),
            "gpu_memory_increase_gb": (
                (end_gpu[0]["memory_used"] - start_gpu[0]["memory_used"]) / 1024
                if start_gpu and end_gpu else 0
            )
        }

    def test_model_loading(self, model_path):
        """Test model loading performance."""
        with self.performance_monitor():
            # Simulate model loading
            self.simulate_model_loading(model_path)

        return self.last_performance

    def simulate_model_loading(self, model_path):
        """Simulate model loading process."""
        # This would be replaced with actual model loading code
        time.sleep(2)  # Simulate loading time
```

### 3.2 Generation Performance Testing

**Objective**: Validate generation speed and quality across hardware configurations.

#### Performance Baselines

| Hardware Tier | 720p 5s Video    | 1080p 5s Video    | Frames per Second |
| ------------- | ---------------- | ----------------- | ----------------- |
| High-End      | 30-60 seconds    | 60-120 seconds    | 2-4 fps           |
| Mid-Range     | 60-120 seconds   | 120-240 seconds   | 1-2 fps           |
| Budget        | 120-300 seconds  | 240-600 seconds   | 0.5-1 fps         |
| Minimum       | 300-600 seconds  | 600-1200 seconds  | 0.2-0.5 fps       |
| CPU-Only      | 600-1800 seconds | 1200-3600 seconds | 0.1-0.2 fps       |

---

## 4. Resource Utilization Testing

### 4.1 System Resource Monitoring

**Objective**: Ensure optimal resource utilization without system overload.

#### Resource Utilization Targets

| Resource | High-End | Mid-Range | Budget | Minimum |
| -------- | -------- | --------- | ------ | ------- |
| CPU      | 70-85%   | 75-90%    | 80-95% | 85-100% |
| Memory   | < 80%    | < 85%     | < 90%  | < 95%   |
| GPU      | 85-95%   | 80-90%    | 75-85% | 70-80%  |
| VRAM     | 85-95%   | 80-90%    | 75-85% | 70-80%  |
| Disk I/O | < 80%    | < 85%     | < 90%  | < 95%   |

### 4.2 Thermal and Power Testing

**Objective**: Validate thermal management and power consumption.

#### Thermal Targets

- **CPU Temperature**: < 85°C under sustained load
- **GPU Temperature**: < 83°C under sustained load
- **System Stability**: No thermal throttling during normal operation

---

## 5. Scalability and Stress Testing

### 5.1 Concurrent Operation Testing

**Objective**: Test system behavior under multiple concurrent operations.

#### Test Scenarios

1. **Multiple Generation Queue**: Test handling of multiple generation requests
2. **Background Processing**: Test performance with other applications running
3. **Extended Operation**: Test stability during hours-long operations
4. **Resource Contention**: Test behavior when system resources are contested

### 5.2 Edge Case Performance Testing

**Objective**: Validate performance under edge conditions.

#### Edge Case Scenarios

1. **Low Disk Space**: Performance when disk space is limited
2. **High Memory Pressure**: Performance when system memory is constrained
3. **Thermal Throttling**: Performance when CPU/GPU throttle due to heat
4. **Network Limitations**: Performance with limited network bandwidth
5. **Power Limitations**: Performance on battery power or limited PSU

---

## Performance Testing Automation

### Automated Performance Test Suite

```python
class PerformanceTestSuite:
    def __init__(self, installation_path):
        self.installation_path = installation_path
        self.cpu_tester = CPUOptimizationTester()
        self.memory_tester = MemoryOptimizationTester()
        self.gpu_tester = GPUOptimizationTester()
        self.model_tester = ModelLoadingTester()

    def run_full_performance_suite(self):
        """Run complete performance test suite."""
        results = {
            "timestamp": time.time(),
            "system_info": self.get_system_info(),
            "tests": {}
        }

        # CPU Performance Tests
        results["tests"]["cpu"] = self.run_cpu_tests()

        # Memory Performance Tests
        results["tests"]["memory"] = self.run_memory_tests()

        # GPU Performance Tests
        results["tests"]["gpu"] = self.run_gpu_tests()

        # Model Loading Tests
        results["tests"]["model_loading"] = self.run_model_loading_tests()

        # Generate performance report
        self.generate_performance_report(results)

        return results

    def generate_performance_report(self, results):
        """Generate comprehensive performance report."""
        # Implementation for report generation
        pass
```

## Performance Regression Testing

### Baseline Management

- Maintain performance baselines for each hardware tier
- Track performance changes across versions
- Alert on significant performance regressions
- Validate performance improvements

### Continuous Performance Monitoring

- Automated performance tests in CI/CD pipeline
- Regular performance benchmarking
- Performance trend analysis
- Early detection of performance issues

This comprehensive performance validation framework ensures that the WAN2.2 local installation system delivers optimal performance across all supported hardware configurations and usage scenarios.
