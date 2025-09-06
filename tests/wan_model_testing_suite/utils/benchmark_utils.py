"""
Benchmark utilities for WAN model testing
"""

import time
import statistics
import psutil
import threading
from typing import Dict, Any, List, Callable, Optional, Tuple
from dataclasses import dataclass, field
from contextlib import contextmanager
import json
from pathlib import Path


@dataclass
class BenchmarkResult:
    """Benchmark result data structure"""
    name: str
    duration_seconds: float
    memory_usage_mb: float
    cpu_usage_percent: float
    gpu_usage_percent: float = 0.0
    throughput: float = 0.0
    quality_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class SystemMetrics:
    """System performance metrics"""
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    gpu_memory_used_gb: float = 0.0
    gpu_memory_total_gb: float = 0.0
    gpu_utilization_percent: float = 0.0
    temperature_c: float = 0.0
    power_draw_watts: float = 0.0


class SystemMonitor:
    """System performance monitoring utility"""
    
    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.monitoring = False
        self.metrics_history: List[SystemMetrics] = []
        self._monitor_thread = None
    
    def start_monitoring(self):
        """Start system monitoring"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.metrics_history.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
    
    def stop_monitoring(self) -> List[SystemMetrics]:
        """Stop monitoring and return collected metrics"""
        self.monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
        return self.metrics_history.copy()
    
    def _monitor_loop(self):
        """Monitoring loop"""
        while self.monitoring:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                time.sleep(self.interval)
            except Exception as e:
                print(f"Monitoring error: {e}")
                break
    
    def _collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        # CPU and memory metrics
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_available_gb = memory.available / (1024**3)
        
        # GPU metrics (mock for testing)
        gpu_memory_used_gb = 0.0
        gpu_memory_total_gb = 0.0
        gpu_utilization_percent = 0.0
        temperature_c = 0.0
        power_draw_watts = 0.0
        
        try:
            # Try to get GPU metrics if available
            # This would use nvidia-ml-py or similar in real implementation
            pass
        except Exception:
            pass
        
        return SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_available_gb=memory_available_gb,
            gpu_memory_used_gb=gpu_memory_used_gb,
            gpu_memory_total_gb=gpu_memory_total_gb,
            gpu_utilization_percent=gpu_utilization_percent,
            temperature_c=temperature_c,
            power_draw_watts=power_draw_watts
        )


class BenchmarkRunner:
    """Main benchmark runner class"""
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("benchmark_results")
        self.output_dir.mkdir(exist_ok=True)
        self.monitor = SystemMonitor()
        self.results: List[BenchmarkResult] = []
    
    @contextmanager
    def benchmark(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        """Context manager for benchmarking operations"""
        metadata = metadata or {}
        
        # Start monitoring
        self.monitor.start_monitoring()
        start_time = time.time()
        start_memory = psutil.virtual_memory().used / (1024**2)  # MB
        
        try:
            yield
        finally:
            # Stop monitoring and collect results
            end_time = time.time()
            end_memory = psutil.virtual_memory().used / (1024**2)  # MB
            metrics_history = self.monitor.stop_monitoring()
            
            # Calculate benchmark metrics
            duration = end_time - start_time
            memory_usage = end_memory - start_memory
            
            # Calculate average system usage during benchmark
            if metrics_history:
                avg_cpu = statistics.mean(m.cpu_percent for m in metrics_history)
                avg_gpu = statistics.mean(m.gpu_utilization_percent for m in metrics_history)
            else:
                avg_cpu = 0.0
                avg_gpu = 0.0
            
            # Create benchmark result
            result = BenchmarkResult(
                name=name,
                duration_seconds=duration,
                memory_usage_mb=memory_usage,
                cpu_usage_percent=avg_cpu,
                gpu_usage_percent=avg_gpu,
                metadata=metadata
            )
            
            self.results.append(result)
    
    def run_benchmark(self, 
                     name: str, 
                     func: Callable, 
                     iterations: int = 1,
                     warmup_iterations: int = 0,
                     metadata: Optional[Dict[str, Any]] = None) -> BenchmarkResult:
        """Run a benchmark function multiple times"""
        metadata = metadata or {}
        
        # Warmup runs
        for _ in range(warmup_iterations):
            func()
        
        # Benchmark runs
        durations = []
        memory_usages = []
        cpu_usages = []
        gpu_usages = []
        
        for i in range(iterations):
            with self.benchmark(f"{name}_iter_{i}", metadata) as _:
                func()
            
            # Get the last result
            last_result = self.results[-1]
            durations.append(last_result.duration_seconds)
            memory_usages.append(last_result.memory_usage_mb)
            cpu_usages.append(last_result.cpu_usage_percent)
            gpu_usages.append(last_result.gpu_usage_percent)
        
        # Calculate aggregate statistics
        avg_duration = statistics.mean(durations)
        std_duration = statistics.stdev(durations) if len(durations) > 1 else 0.0
        min_duration = min(durations)
        max_duration = max(durations)
        
        avg_memory = statistics.mean(memory_usages)
        avg_cpu = statistics.mean(cpu_usages)
        avg_gpu = statistics.mean(gpu_usages)
        
        # Create aggregate result
        aggregate_result = BenchmarkResult(
            name=name,
            duration_seconds=avg_duration,
            memory_usage_mb=avg_memory,
            cpu_usage_percent=avg_cpu,
            gpu_usage_percent=avg_gpu,
            metadata={
                **metadata,
                "iterations": iterations,
                "std_duration": std_duration,
                "min_duration": min_duration,
                "max_duration": max_duration,
                "all_durations": durations
            }
        )
        
        self.results.append(aggregate_result)
        return aggregate_result
    
    def save_results(self, filename: Optional[str] = None):
        """Save benchmark results to file"""
        if not filename:
            timestamp = int(time.time())
            filename = f"benchmark_results_{timestamp}.json"
        
        output_path = self.output_dir / filename
        
        # Convert results to serializable format
        results_data = []
        for result in self.results:
            result_dict = {
                "name": result.name,
                "duration_seconds": result.duration_seconds,
                "memory_usage_mb": result.memory_usage_mb,
                "cpu_usage_percent": result.cpu_usage_percent,
                "gpu_usage_percent": result.gpu_usage_percent,
                "throughput": result.throughput,
                "quality_score": result.quality_score,
                "metadata": result.metadata,
                "timestamp": result.timestamp
            }
            results_data.append(result_dict)
        
        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"Benchmark results saved to: {output_path}")
    
    def print_summary(self):
        """Print benchmark results summary"""
        if not self.results:
            print("No benchmark results available")
            return
        
        print("\n" + "="*80)
        print("BENCHMARK RESULTS SUMMARY")
        print("="*80)
        
        for result in self.results:
            print(f"\nBenchmark: {result.name}")
            print(f"  Duration: {result.duration_seconds:.3f}s")
            print(f"  Memory Usage: {result.memory_usage_mb:.1f} MB")
            print(f"  CPU Usage: {result.cpu_usage_percent:.1f}%")
            print(f"  GPU Usage: {result.gpu_usage_percent:.1f}%")
            
            if result.throughput > 0:
                print(f"  Throughput: {result.throughput:.2f}")
            
            if result.quality_score > 0:
                print(f"  Quality Score: {result.quality_score:.3f}")
            
            # Print metadata
            if result.metadata:
                print("  Metadata:")
                for key, value in result.metadata.items():
                    if key not in ["all_durations"]:  # Skip large arrays
                        print(f"    {key}: {value}")


class GenerationBenchmark:
    """Specialized benchmark for video generation"""
    
    def __init__(self, runner: BenchmarkRunner):
        self.runner = runner
    
    def benchmark_generation(self, 
                           model, 
                           params, 
                           iterations: int = 5,
                           name: str = "generation") -> BenchmarkResult:
        """Benchmark video generation"""
        def generation_func():
            result = model.generate(params)
            return result
        
        metadata = {
            "model_type": getattr(model, 'model_type', 'unknown'),
            "num_frames": getattr(params, 'num_frames', 0),
            "resolution": f"{getattr(params, 'width', 0)}x{getattr(params, 'height', 0)}",
            "inference_steps": getattr(params, 'num_inference_steps', 0)
        }
        
        return self.runner.run_benchmark(
            name=name,
            func=generation_func,
            iterations=iterations,
            warmup_iterations=1,
            metadata=metadata
        )
    
    def benchmark_batch_generation(self, 
                                 model, 
                                 batch_params: List[Any],
                                 name: str = "batch_generation") -> BenchmarkResult:
        """Benchmark batch video generation"""
        def batch_generation_func():
            results = model.generate_batch(batch_params)
            return results
        
        metadata = {
            "model_type": getattr(model, 'model_type', 'unknown'),
            "batch_size": len(batch_params),
            "total_frames": sum(getattr(p, 'num_frames', 0) for p in batch_params)
        }
        
        with self.runner.benchmark(name, metadata):
            results = batch_generation_func()
        
        # Calculate throughput
        last_result = self.runner.results[-1]
        if last_result.duration_seconds > 0:
            last_result.throughput = len(batch_params) / last_result.duration_seconds
        
        return last_result


class MemoryBenchmark:
    """Memory usage benchmarking utilities"""
    
    def __init__(self, runner: BenchmarkRunner):
        self.runner = runner
    
    def benchmark_memory_usage(self, 
                             model, 
                             operation_func: Callable,
                             name: str = "memory_usage") -> Dict[str, float]:
        """Benchmark memory usage of an operation"""
        # Get initial memory state
        initial_memory = self._get_memory_usage()
        
        # Run operation
        with self.runner.benchmark(name):
            result = operation_func()
        
        # Get peak memory usage
        peak_memory = self._get_memory_usage()
        
        memory_metrics = {
            "initial_memory_gb": initial_memory["system_memory_gb"],
            "peak_memory_gb": peak_memory["system_memory_gb"],
            "memory_increase_gb": peak_memory["system_memory_gb"] - initial_memory["system_memory_gb"],
            "gpu_memory_gb": peak_memory.get("gpu_memory_gb", 0.0)
        }
        
        return memory_metrics
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        memory = psutil.virtual_memory()
        system_memory_gb = memory.used / (1024**3)
        
        # GPU memory would be collected here in real implementation
        gpu_memory_gb = 0.0
        
        return {
            "system_memory_gb": system_memory_gb,
            "gpu_memory_gb": gpu_memory_gb
        }


class ThroughputBenchmark:
    """Throughput benchmarking utilities"""
    
    def __init__(self, runner: BenchmarkRunner):
        self.runner = runner
    
    def benchmark_throughput(self, 
                           model, 
                           params_list: List[Any],
                           name: str = "throughput") -> BenchmarkResult:
        """Benchmark throughput (items per second)"""
        def throughput_func():
            results = []
            for params in params_list:
                result = model.generate(params)
                results.append(result)
            return results
        
        metadata = {
            "model_type": getattr(model, 'model_type', 'unknown'),
            "num_items": len(params_list)
        }
        
        with self.runner.benchmark(name, metadata):
            results = throughput_func()
        
        # Calculate throughput
        last_result = self.runner.results[-1]
        if last_result.duration_seconds > 0:
            last_result.throughput = len(params_list) / last_result.duration_seconds
        
        return last_result


class QualityBenchmark:
    """Quality assessment benchmarking"""
    
    def __init__(self, runner: BenchmarkRunner):
        self.runner = runner
    
    def benchmark_quality(self, 
                        model, 
                        params, 
                        quality_func: Callable,
                        name: str = "quality") -> BenchmarkResult:
        """Benchmark generation quality"""
        def quality_benchmark_func():
            # Generate content
            result = model.generate(params)
            
            # Assess quality
            quality_score = quality_func(result)
            
            return result, quality_score
        
        metadata = {
            "model_type": getattr(model, 'model_type', 'unknown')
        }
        
        with self.runner.benchmark(name, metadata):
            result, quality_score = quality_benchmark_func()
        
        # Set quality score
        last_result = self.runner.results[-1]
        last_result.quality_score = quality_score
        
        return last_result


def create_benchmark_suite(output_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Create a complete benchmark suite"""
    runner = BenchmarkRunner(output_dir)
    
    return {
        "runner": runner,
        "generation": GenerationBenchmark(runner),
        "memory": MemoryBenchmark(runner),
        "throughput": ThroughputBenchmark(runner),
        "quality": QualityBenchmark(runner)
    }


def compare_benchmark_results(results1: List[BenchmarkResult], 
                            results2: List[BenchmarkResult]) -> Dict[str, Any]:
    """Compare two sets of benchmark results"""
    comparison = {
        "duration_improvement_percent": 0.0,
        "memory_improvement_percent": 0.0,
        "throughput_improvement_percent": 0.0,
        "quality_improvement_percent": 0.0
    }
    
    if not results1 or not results2:
        return comparison
    
    # Calculate average metrics for each set
    avg_duration1 = statistics.mean(r.duration_seconds for r in results1)
    avg_duration2 = statistics.mean(r.duration_seconds for r in results2)
    
    avg_memory1 = statistics.mean(r.memory_usage_mb for r in results1)
    avg_memory2 = statistics.mean(r.memory_usage_mb for r in results2)
    
    avg_throughput1 = statistics.mean(r.throughput for r in results1 if r.throughput > 0)
    avg_throughput2 = statistics.mean(r.throughput for r in results2 if r.throughput > 0)
    
    avg_quality1 = statistics.mean(r.quality_score for r in results1 if r.quality_score > 0)
    avg_quality2 = statistics.mean(r.quality_score for r in results2 if r.quality_score > 0)
    
    # Calculate improvements (negative means worse performance)
    if avg_duration1 > 0:
        comparison["duration_improvement_percent"] = ((avg_duration1 - avg_duration2) / avg_duration1) * 100
    
    if avg_memory1 > 0:
        comparison["memory_improvement_percent"] = ((avg_memory1 - avg_memory2) / avg_memory1) * 100
    
    if avg_throughput1 > 0:
        comparison["throughput_improvement_percent"] = ((avg_throughput2 - avg_throughput1) / avg_throughput1) * 100
    
    if avg_quality1 > 0:
        comparison["quality_improvement_percent"] = ((avg_quality2 - avg_quality1) / avg_quality1) * 100
    
    return comparison