from unittest.mock import Mock, patch
"""
Demo script for Performance Benchmark System
Demonstrates before/after performance metrics collection, hardware validation, and recommendations
"""

import time
import logging
import random
from pathlib import Path

from performance_benchmark_system import (
    PerformanceBenchmarkSystem, PerformanceMetrics, SystemMonitor
)
from hardware_optimizer import HardwareProfile, OptimalSettings

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def simulate_model_loading(load_time: float = 2.0, vram_usage: int = 8192):
    """Simulate model loading with configurable parameters"""
    logger.info(f"Simulating model loading (target time: {load_time}s, VRAM: {vram_usage}MB)")
    
    # Simulate loading time with some variance
    actual_time = load_time + random.uniform(-0.2, 0.2)
    time.sleep(actual_time)
    
    # Return mock model
    return {
        'model_name': 'TI2V-5B',
        'parameters': '5B',
        'vram_usage': vram_usage,
        'loaded': True
    }

def simulate_video_generation(duration: float = 2.0, resolution: str = "512x512", 
                            generation_time: float = 60.0, vram_usage: int = 12288):
    """Simulate video generation with configurable parameters"""
    logger.info(f"Simulating video generation ({duration}s video, {resolution}, target time: {generation_time}s)")
    
    # Simulate generation time with some variance
    actual_time = generation_time + random.uniform(-5.0, 5.0)
    time.sleep(min(actual_time, 10.0))  # Cap at 10s for demo
    
    return {
        'video_duration': duration,
        'resolution': resolution,
        'frames': int(duration * 24),  # 24 FPS
        'generation_time': actual_time,
        'vram_usage': vram_usage
    }

def create_mock_before_benchmark(benchmark_system: PerformanceBenchmarkSystem):
    """Create a 'before optimization' benchmark function"""
    def before_benchmark():
        logger.info("Running BEFORE optimization benchmark...")
        
        # Simulate unoptimized performance
        def mock_model_load():
            return simulate_model_loading(load_time=5.0, vram_usage=14336)  # Slow, high VRAM
        
        # Mock monitoring data for unoptimized scenario
        mock_monitor_data = {
            'peak_vram_usage_mb': 14336,
            'peak_ram_usage_mb': 8192,
            'gpu_utilization_avg': 65.0,  # Low utilization
            'gpu_temperature_max': 82.0,  # High temperature
            'gpu_power_draw_avg': 310.0,  # High power
            'cpu_utilization_avg': 25.0,
            'cpu_temperature_max': 70.0
        }
        
        # Patch the monitor to return our mock data
        original_stop = benchmark_system.monitor.stop_monitoring
        benchmark_system.monitor.stop_monitoring = lambda: mock_monitor_data
        
        try:
            metrics = benchmark_system.benchmark_model_loading(mock_model_load, "TI2V-5B")
            return metrics
        finally:
            benchmark_system.monitor.stop_monitoring = original_stop
    
    return before_benchmark

def create_mock_after_benchmark(benchmark_system: PerformanceBenchmarkSystem):
    """Create an 'after optimization' benchmark function"""
    def after_benchmark():
        logger.info("Running AFTER optimization benchmark...")
        
        # Simulate optimized performance
        def mock_model_load():
            return simulate_model_loading(load_time=3.0, vram_usage=10240)  # Faster, lower VRAM
        
        # Mock monitoring data for optimized scenario
        mock_monitor_data = {
            'peak_vram_usage_mb': 10240,
            'peak_ram_usage_mb': 6144,
            'gpu_utilization_avg': 92.0,  # High utilization
            'gpu_temperature_max': 75.0,  # Lower temperature
            'gpu_power_draw_avg': 280.0,  # Lower power
            'cpu_utilization_avg': 35.0,
            'cpu_temperature_max': 65.0
        }
        
        # Patch the monitor to return our mock data
        original_stop = benchmark_system.monitor.stop_monitoring
        benchmark_system.monitor.stop_monitoring = lambda: mock_monitor_data
        
        try:
            metrics = benchmark_system.benchmark_model_loading(mock_model_load, "TI2V-5B")
            return metrics
        finally:
            benchmark_system.monitor.stop_monitoring = original_stop
    
    return after_benchmark

def demo_rtx_4080_benchmarking():
    """Demonstrate RTX 4080 performance benchmarking"""
    logger.info("=== RTX 4080 Performance Benchmarking Demo ===")
    
    # Create RTX 4080 hardware profile
    rtx_4080_profile = HardwareProfile(
        cpu_model="Intel Core i7-12700K",
        cpu_cores=12,
        total_memory_gb=32,
        gpu_model="NVIDIA GeForce RTX 4080",
        vram_gb=16,
        cuda_version="12.1",
        driver_version="537.13",
        is_rtx_4080=True,
        is_threadripper_pro=False
    )
    
    # Initialize benchmark system
    benchmark_system = PerformanceBenchmarkSystem(results_dir="demo_benchmark_results")
    
    # Create hardware limits
    limits = benchmark_system.create_hardware_limits(rtx_4080_profile)
    logger.info(f"Hardware limits: VRAM={limits.max_vram_usage_mb}MB, GPU temp={limits.max_gpu_temperature}°C")
    
    # Create before/after benchmark functions
    before_func = create_mock_before_benchmark(benchmark_system)
    after_func = create_mock_after_benchmark(benchmark_system)
    
    # Run before/after benchmark
    result = benchmark_system.run_before_after_benchmark(
        before_func, after_func, rtx_4080_profile, "rtx_4080_optimization"
    )
    
    # Display results
    logger.info(f"Benchmark Success: {result.success}")
    logger.info(f"Performance Improvement: {result.performance_improvement:.1f}%")
    logger.info(f"Memory Savings: {result.memory_savings}MB")
    
    if result.before_metrics and result.after_metrics:
        logger.info(f"Load Time: {result.before_metrics.model_load_time:.1f}s → {result.after_metrics.model_load_time:.1f}s")
        logger.info(f"VRAM Usage: {result.before_metrics.peak_vram_usage_mb}MB → {result.after_metrics.peak_vram_usage_mb}MB")
        logger.info(f"GPU Utilization: {result.before_metrics.gpu_utilization_avg:.1f}% → {result.after_metrics.gpu_utilization_avg:.1f}%")
    
    if result.recommendations:
        logger.info("Recommendations:")
        for rec in result.recommendations[:5]:  # Show first 5
            logger.info(f"  - {rec}")
    
    if result.warnings:
        logger.info("Warnings:")
        for warn in result.warnings:
            logger.info(f"  - {warn}")
    
    return result

def demo_threadripper_pro_benchmarking():
    """Demonstrate Threadripper PRO performance benchmarking"""
    logger.info("=== Threadripper PRO Performance Benchmarking Demo ===")
    
    # Create Threadripper PRO hardware profile
    threadripper_profile = HardwareProfile(
        cpu_model="AMD Ryzen Threadripper PRO 5995WX",
        cpu_cores=64,
        total_memory_gb=128,
        gpu_model="NVIDIA GeForce RTX 4080",
        vram_gb=16,
        cuda_version="12.1",
        driver_version="537.13",
        is_rtx_4080=True,
        is_threadripper_pro=True
    )
    
    # Initialize benchmark system
    benchmark_system = PerformanceBenchmarkSystem(results_dir="demo_benchmark_results")
    
    # Create hardware limits
    limits = benchmark_system.create_hardware_limits(threadripper_profile)
    logger.info(f"Hardware limits: RAM={limits.max_ram_usage_mb}MB, CPU temp={limits.max_cpu_temperature}°C")
    
    # Simulate generation benchmark
    def mock_generation_benchmark():
        def mock_generation(**kwargs):
            return simulate_video_generation(
                duration=2.0, 
                resolution="512x512",
                generation_time=45.0,  # Optimized time
                vram_usage=11264
            )
        
        # Mock monitoring data
        mock_monitor_data = {
            'peak_vram_usage_mb': 11264,
            'peak_ram_usage_mb': 16384,
            'gpu_utilization_avg': 95.0,
            'gpu_temperature_max': 78.0,
            'gpu_power_draw_avg': 290.0,
            'cpu_utilization_avg': 75.0,  # High CPU utilization
            'cpu_temperature_max': 68.0
        }
        
        # Patch the monitor
        original_stop = benchmark_system.monitor.stop_monitoring
        benchmark_system.monitor.stop_monitoring = lambda: mock_monitor_data
        
        try:
            metrics = benchmark_system.benchmark_generation(
                mock_generation, 
                {"prompt": "A beautiful landscape", "duration": 2.0},
                expected_output_count=1
            )
            return metrics
        finally:
            benchmark_system.monitor.stop_monitoring = original_stop
    
    # Run generation benchmark
    metrics = mock_generation_benchmark()
    
    # Validate against hardware limits
    is_valid, warnings, errors = benchmark_system.validate_against_hardware_limits(
        metrics, limits, threadripper_profile
    )
    
    logger.info(f"Generation Time: {metrics.generation_time:.1f}s")
    logger.info(f"VRAM Usage: {metrics.peak_vram_usage_mb}MB")
    logger.info(f"CPU Utilization: {metrics.cpu_utilization_avg:.1f}%")
    logger.info(f"Validation: {'PASSED' if is_valid else 'FAILED'}")
    
    # Generate recommendations
    recommendations = benchmark_system.generate_performance_recommendations(
        metrics, threadripper_profile
    )
    
    if recommendations:
        logger.info("Recommendations:")
        for rec in recommendations[:5]:
            logger.info(f"  - {rec}")
    
    if warnings:
        logger.info("Warnings:")
        for warn in warnings:
            logger.info(f"  - {warn}")
    
    if errors:
        logger.info("Errors:")
        for err in errors:
            logger.info(f"  - {err}")
    
    return metrics

def demo_system_monitoring():
    """Demonstrate real-time system monitoring"""
    logger.info("=== System Monitoring Demo ===")
    
    monitor = SystemMonitor()
    
    # Start monitoring
    logger.info("Starting system monitoring for 3 seconds...")
    monitor.start_monitoring(interval=0.5)
    
    # Simulate some work
    time.sleep(3.0)
    
    # Stop monitoring and get results
    aggregated_metrics = monitor.stop_monitoring()
    
    logger.info("Monitoring Results:")
    for metric, value in aggregated_metrics.items():
        if 'temperature' in metric or 'utilization' in metric:
            logger.info(f"  {metric}: {value:.1f}")
        elif 'usage' in metric:
            logger.info(f"  {metric}: {value:.0f}MB")
        else:
            logger.info(f"  {metric}: {value:.1f}")

def demo_benchmark_report():
    """Demonstrate benchmark report generation"""
    logger.info("=== Benchmark Report Demo ===")
    
    benchmark_system = PerformanceBenchmarkSystem(results_dir="demo_benchmark_results")
    
    # Load existing benchmark results if any
    results = []
    results_dir = Path("demo_benchmark_results")
    
    if results_dir.exists():
        for result_file in results_dir.glob("benchmark_*.json"):
            result = benchmark_system.load_benchmark_result(str(result_file))
            if result:
                results.append(result)
    
    if results:
        logger.info(f"Found {len(results)} existing benchmark results")
        report = benchmark_system.generate_benchmark_report(results)
        
        # Save report
        report_file = results_dir / "benchmark_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Generated benchmark report: {report_file}")
        
        # Show summary
        successful = [r for r in results if r.success]
        if successful:
            avg_improvement = sum(r.performance_improvement for r in successful) / len(successful)
            total_savings = sum(r.memory_savings for r in successful)
            logger.info(f"Average improvement: {avg_improvement:.1f}%")
            logger.info(f"Total memory savings: {total_savings}MB")
    else:
        logger.info("No existing benchmark results found")

def demo_hardware_detection_failure():
    """Demonstrate hardware detection failure handling"""
    logger.info("=== Hardware Detection Failure Demo ===")
    
    benchmark_system = PerformanceBenchmarkSystem(results_dir="demo_benchmark_results")
    
    # Simulate hardware detection failure
    logger.info("Simulating hardware detection failure...")
    
    # Test with no partial profile
    profile, settings = benchmark_system.handle_hardware_detection_failure()
    
    logger.info(f"Fallback profile created:")
    logger.info(f"  CPU: {profile.cpu_model} ({profile.cpu_cores} cores)")
    logger.info(f"  RAM: {profile.total_memory_gb}GB")
    logger.info(f"  GPU: {profile.gpu_model} ({profile.vram_gb}GB VRAM)")
    
    # Create manual configuration guide
    guide = benchmark_system.create_manual_configuration_guide(profile)
    
    logger.info("Manual configuration options:")
    for config_name, config in guide['configuration_options'].items():
        logger.info(f"  {config_name.title()}: {config['description']}")
        logger.info(f"    Batch size: {config['batch_size']}")
        logger.info(f"    Tile size: {config['tile_size']}")
        logger.info(f"    CPU offload: {config['enable_cpu_offload']}")
    
    if guide['warnings']:
        logger.info("Warnings:")
        for warning in guide['warnings']:
            logger.info(f"  - {warning}")
    
    return profile, settings, guide

def demo_settings_validation():
    """Demonstrate settings validation against hardware limits"""
    logger.info("=== Settings Validation Demo ===")
    
    benchmark_system = PerformanceBenchmarkSystem(results_dir="demo_benchmark_results")
    
    # Create test hardware profile
    test_profile = HardwareProfile(
        cpu_model="Test CPU",
        cpu_cores=8,
        total_memory_gb=16,
        gpu_model="Test GPU",
        vram_gb=8,
        is_rtx_4080=False,
        is_threadripper_pro=False
    )
    
    # Generate recommended settings
    settings = benchmark_system.generate_recommended_settings_for_hardware(test_profile)
    
    logger.info("Generated settings:")
    logger.info(f"  Batch size: {settings.batch_size}")
    logger.info(f"  Tile size: {settings.tile_size}")
    logger.info(f"  Memory fraction: {settings.memory_fraction}")
    logger.info(f"  Thread count: {settings.num_threads}")
    
    # Validate settings
    is_valid, warnings, errors = benchmark_system.validate_settings_against_hardware_limits(
        settings, test_profile
    )
    
    logger.info(f"Settings validation: {'PASSED' if is_valid else 'FAILED'}")
    
    if warnings:
        logger.info("Warnings:")
        for warning in warnings:
            logger.info(f"  - {warning}")
    
    if errors:
        logger.info("Errors:")
        for error in errors:
            logger.info(f"  - {error}")
    
    return settings, is_valid, warnings, errors

def demo_detailed_performance_comparison():
    """Demonstrate detailed performance comparison"""
    logger.info("=== Detailed Performance Comparison Demo ===")
    
    benchmark_system = PerformanceBenchmarkSystem(results_dir="demo_benchmark_results")
    
    # Create mock before/after metrics
    before_metrics = PerformanceMetrics(
        timestamp="2024-01-01T12:00:00",
        model_load_time=150.0,
        generation_time=90.0,
        total_time=240.0,
        peak_vram_usage_mb=14336,
        peak_ram_usage_mb=10240,
        gpu_utilization_avg=65.0,
        gpu_temperature_max=82.0,
        gpu_power_draw_avg=310.0,
        throughput_items_per_second=0.4,
        vram_efficiency=0.9
    )
    
    after_metrics = PerformanceMetrics(
        timestamp="2024-01-01T12:10:00",
        model_load_time=100.0,
        generation_time=60.0,
        total_time=160.0,
        peak_vram_usage_mb=10240,
        peak_ram_usage_mb=7168,
        gpu_utilization_avg=88.0,
        gpu_temperature_max=75.0,
        gpu_power_draw_avg=280.0,
        throughput_items_per_second=0.6,
        vram_efficiency=0.7
    )
    
    # Generate detailed comparison
    comparison = benchmark_system._generate_detailed_performance_comparison(
        before_metrics, after_metrics
    )
    
    logger.info("Performance Improvements:")
    
    # Timing improvements
    timing = comparison['timing_comparison']
    logger.info(f"  Model load time: {timing['model_load_time']['improvement_percent']:.1f}% faster")
    logger.info(f"  Generation time: {timing['generation_time']['improvement_percent']:.1f}% faster")
    logger.info(f"  Total time: {timing['total_time']['improvement_percent']:.1f}% faster")
    
    # Memory improvements
    memory = comparison['memory_comparison']
    logger.info(f"  VRAM savings: {memory['peak_vram_usage_mb']['savings_mb']}MB ({memory['peak_vram_usage_mb']['savings_percent']:.1f}%)")
    logger.info(f"  RAM savings: {memory['peak_ram_usage_mb']['savings_mb']}MB ({memory['peak_ram_usage_mb']['savings_percent']:.1f}%)")
    
    # Performance improvements
    performance = comparison['performance_comparison']
    logger.info(f"  GPU utilization: +{performance['gpu_utilization_avg']['improvement']:.1f}%")
    logger.info(f"  Throughput: +{performance['throughput_items_per_second']['improvement_percent']:.1f}%")
    
    # Thermal improvements
    thermal = comparison['thermal_comparison']
    logger.info(f"  GPU temperature: -{thermal['gpu_temperature_max']['improvement']:.1f}°C")
    logger.info(f"  Power savings: -{thermal['gpu_power_draw_avg']['savings']:.1f}W")
    
    return comparison

def main():
    """Main demo function"""
    logger.info("Starting Performance Benchmark System Demo")
    
    try:
        # Demo 1: RTX 4080 benchmarking
        rtx_result = demo_rtx_4080_benchmarking()
        print("\n" + "="*60 + "\n")
        
        # Demo 2: Threadripper PRO benchmarking
        threadripper_metrics = demo_threadripper_pro_benchmarking()
        print("\n" + "="*60 + "\n")
        
        # Demo 3: System monitoring
        demo_system_monitoring()
        print("\n" + "="*60 + "\n")
        
        # Demo 4: Hardware detection failure handling (NEW)
        profile, settings, guide = demo_hardware_detection_failure()
        print("\n" + "="*60 + "\n")
        
        # Demo 5: Settings validation (NEW)
        settings, is_valid, warnings, errors = demo_settings_validation()
        print("\n" + "="*60 + "\n")
        
        # Demo 6: Detailed performance comparison (NEW)
        comparison = demo_detailed_performance_comparison()
        print("\n" + "="*60 + "\n")
        
        # Demo 7: Benchmark report
        demo_benchmark_report()
        
        logger.info("Demo completed successfully!")
        logger.info("New features demonstrated:")
        logger.info("  - Hardware detection failure handling (Requirement 5.4)")
        logger.info("  - Settings validation against hardware limits (Requirement 5.5)")
        logger.info("  - Detailed before/after performance metrics (Requirement 5.3)")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise

if __name__ == "__main__":
    main()