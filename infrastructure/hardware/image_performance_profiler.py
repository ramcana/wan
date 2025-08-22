#!/usr/bin/env python3
"""
Image Performance Profiler for WAN22 Start/End Image Fix
Profiles and optimizes image processing performance bottlenecks
"""

import time
import psutil
import tracemalloc
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from PIL import Image
import io
import threading
from contextlib import contextmanager
import json
from datetime import datetime
import os

@dataclass
class PerformanceMetrics:
    """Performance metrics for image operations"""
    operation_name: str
    execution_time: float
    memory_usage_mb: float
    peak_memory_mb: float
    cpu_usage_percent: float
    image_size: Tuple[int, int]
    file_size_bytes: int
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ProfilerResults:
    """Complete profiler results"""
    total_operations: int
    total_time: float
    average_time: float
    peak_memory_mb: float
    total_memory_mb: float
    bottlenecks: List[str]
    recommendations: List[str]
    metrics: List[PerformanceMetrics] = field(default_factory=list)

class ImagePerformanceProfiler:
    """Profiles image processing performance and identifies bottlenecks"""
    
    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
        self.is_profiling = False
        self.start_time = None
        self.memory_tracker = None
        
    def start_profiling(self):
        """Start performance profiling session"""
        self.is_profiling = True
        self.start_time = time.time()
        tracemalloc.start()
        self.metrics.clear()
        
    def stop_profiling(self) -> ProfilerResults:
        """Stop profiling and return results"""
        self.is_profiling = False
        tracemalloc.stop()
        
        return self._generate_results()
        
    @contextmanager
    def profile_operation(self, operation_name: str, image_size: Tuple[int, int] = (0, 0), 
                         file_size: int = 0):
        """Context manager for profiling individual operations"""
        if not self.is_profiling:
            yield
            return
            
        # Start monitoring
        start_time = time.time()
        start_memory = self._get_memory_usage()
        start_cpu = psutil.cpu_percent()
        
        # Track peak memory during operation
        peak_memory = start_memory
        memory_monitor = threading.Thread(target=self._monitor_memory, 
                                         args=(lambda: peak_memory,))
        memory_monitor.daemon = True
        memory_monitor.start()
        
        try:
            yield
        finally:
            # Calculate metrics
            end_time = time.time()
            end_memory = self._get_memory_usage()
            end_cpu = psutil.cpu_percent()
            
            execution_time = end_time - start_time
            memory_usage = end_memory - start_memory
            cpu_usage = (start_cpu + end_cpu) / 2
            
            # Wait for memory monitor to finish
            memory_monitor.join(timeout=0.1)
            
            # Record metrics
            metrics = PerformanceMetrics(
                operation_name=operation_name,
                execution_time=execution_time,
                memory_usage_mb=memory_usage,
                peak_memory_mb=peak_memory,
                cpu_usage_percent=cpu_usage,
                image_size=image_size,
                file_size_bytes=file_size
            )
            
            self.metrics.append(metrics)
            
    def _monitor_memory(self, peak_memory_ref):
        """Monitor peak memory usage during operation"""
        while self.is_profiling:
            current_memory = self._get_memory_usage()
            if hasattr(peak_memory_ref, '__call__'):
                # This is a lambda, we need a different approach
                pass
            time.sleep(0.01)  # Check every 10ms
            
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
        
    def _generate_results(self) -> ProfilerResults:
        """Generate comprehensive profiler results"""
        if not self.metrics:
            return ProfilerResults(
                total_operations=0,
                total_time=0.0,
                average_time=0.0,
                peak_memory_mb=0.0,
                total_memory_mb=0.0,
                bottlenecks=[],
                recommendations=[]
            )
            
        total_time = sum(m.execution_time for m in self.metrics)
        average_time = total_time / len(self.metrics)
        peak_memory = max(m.peak_memory_mb for m in self.metrics)
        total_memory = sum(m.memory_usage_mb for m in self.metrics)
        
        bottlenecks = self._identify_bottlenecks()
        recommendations = self._generate_recommendations()
        
        return ProfilerResults(
            total_operations=len(self.metrics),
            total_time=total_time,
            average_time=average_time,
            peak_memory_mb=peak_memory,
            total_memory_mb=total_memory,
            bottlenecks=bottlenecks,
            recommendations=recommendations,
            metrics=self.metrics.copy()
        )
        
    def _identify_bottlenecks(self) -> List[str]:
        """Identify performance bottlenecks"""
        bottlenecks = []
        
        if not self.metrics:
            return bottlenecks
            
        # Find slow operations (> 500ms)
        slow_ops = [m for m in self.metrics if m.execution_time > 0.5]
        if slow_ops:
            bottlenecks.append(f"Slow operations detected: {len(slow_ops)} operations > 500ms")
            
        # Find memory-intensive operations (> 100MB)
        memory_intensive = [m for m in self.metrics if m.memory_usage_mb > 100]
        if memory_intensive:
            bottlenecks.append(f"Memory-intensive operations: {len(memory_intensive)} operations > 100MB")
            
        # Find high CPU usage operations (> 80%)
        cpu_intensive = [m for m in self.metrics if m.cpu_usage_percent > 80]
        if cpu_intensive:
            bottlenecks.append(f"CPU-intensive operations: {len(cpu_intensive)} operations > 80% CPU")
            
        # Check for operations that scale poorly with image size
        large_image_ops = [m for m in self.metrics if m.image_size[0] * m.image_size[1] > 2073600]  # > 1920x1080
        if large_image_ops:
            avg_time_large = sum(m.execution_time for m in large_image_ops) / len(large_image_ops)
            small_image_ops = [m for m in self.metrics if m.image_size[0] * m.image_size[1] <= 2073600]
            if small_image_ops:
                avg_time_small = sum(m.execution_time for m in small_image_ops) / len(small_image_ops)
                if avg_time_large > avg_time_small * 3:
                    bottlenecks.append("Poor scaling with image size detected")
                    
        return bottlenecks
        
    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        if not self.metrics:
            return recommendations
            
        # Analyze patterns and suggest optimizations
        validation_ops = [m for m in self.metrics if 'validation' in m.operation_name.lower()]
        if validation_ops and len(validation_ops) > 1:
            recommendations.append("Consider caching validation results for repeated operations")
            
        thumbnail_ops = [m for m in self.metrics if 'thumbnail' in m.operation_name.lower()]
        if thumbnail_ops:
            avg_thumbnail_time = sum(m.execution_time for m in thumbnail_ops) / len(thumbnail_ops)
            if avg_thumbnail_time > 0.1:
                recommendations.append("Optimize thumbnail generation - consider using faster resampling")
                
        large_memory_ops = [m for m in self.metrics if m.memory_usage_mb > 50]
        if large_memory_ops:
            recommendations.append("Implement image streaming for large files to reduce memory usage")
            
        # Check for repeated operations on same images
        operation_counts = {}
        for metric in self.metrics:
            key = f"{metric.operation_name}_{metric.image_size}_{metric.file_size_bytes}"
            operation_counts[key] = operation_counts.get(key, 0) + 1
            
        repeated_ops = [k for k, v in operation_counts.items() if v > 1]
        if repeated_ops:
            recommendations.append("Implement result caching for repeated operations")
            
        return recommendations
        
    def save_results(self, results: ProfilerResults, filename: str = None):
        """Save profiler results to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"image_performance_profile_{timestamp}.json"
            
        # Convert to serializable format
        data = {
            'summary': {
                'total_operations': results.total_operations,
                'total_time': results.total_time,
                'average_time': results.average_time,
                'peak_memory_mb': results.peak_memory_mb,
                'total_memory_mb': results.total_memory_mb,
                'bottlenecks': results.bottlenecks,
                'recommendations': results.recommendations
            },
            'metrics': [
                {
                    'operation_name': m.operation_name,
                    'execution_time': m.execution_time,
                    'memory_usage_mb': m.memory_usage_mb,
                    'peak_memory_mb': m.peak_memory_mb,
                    'cpu_usage_percent': m.cpu_usage_percent,
                    'image_size': m.image_size,
                    'file_size_bytes': m.file_size_bytes,
                    'timestamp': m.timestamp.isoformat()
                }
                for m in results.metrics
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"Performance profile saved to {filename}")
        
    def print_summary(self, results: ProfilerResults):
        """Print performance summary"""
        print("\n" + "="*60)
        print("IMAGE PERFORMANCE PROFILER RESULTS")
        print("="*60)
        print(f"Total Operations: {results.total_operations}")
        print(f"Total Time: {results.total_time:.3f}s")
        print(f"Average Time: {results.average_time:.3f}s")
        print(f"Peak Memory: {results.peak_memory_mb:.2f}MB")
        print(f"Total Memory Used: {results.total_memory_mb:.2f}MB")
        
        if results.bottlenecks:
            print(f"\nBOTTLENECKS IDENTIFIED:")
            for bottleneck in results.bottlenecks:
                print(f"  • {bottleneck}")
                
        if results.recommendations:
            print(f"\nRECOMMENDATIONS:")
            for rec in results.recommendations:
                print(f"  • {rec}")
                
        print("\nTOP 5 SLOWEST OPERATIONS:")
        sorted_metrics = sorted(results.metrics, key=lambda x: x.execution_time, reverse=True)
        for i, metric in enumerate(sorted_metrics[:5]):
            print(f"  {i+1}. {metric.operation_name}: {metric.execution_time:.3f}s")
            
        print("="*60)

# Example usage and testing functions
def profile_image_operations():
    """Profile common image operations"""
    profiler = ImagePerformanceProfiler()
    profiler.start_profiling()
    
    # Test with different image sizes
    test_sizes = [(512, 512), (1024, 1024), (1920, 1080), (2048, 2048)]
    
    for width, height in test_sizes:
        # Create test image
        test_image = Image.new('RGB', (width, height), color='red')
        
        # Profile validation
        with profiler.profile_operation(f"validation_{width}x{height}", (width, height), width*height*3):
            # Simulate validation operations
            _ = test_image.format
            _ = test_image.size
            _ = test_image.mode
            time.sleep(0.01)  # Simulate processing time
            
        # Profile thumbnail generation
        with profiler.profile_operation(f"thumbnail_{width}x{height}", (width, height), width*height*3):
            thumbnail = test_image.copy()
            thumbnail.thumbnail((256, 256), Image.Resampling.LANCZOS)
            
        # Profile format conversion
        with profiler.profile_operation(f"format_conversion_{width}x{height}", (width, height), width*height*3):
            buffer = io.BytesIO()
            test_image.save(buffer, format='JPEG', quality=85)
            buffer.seek(0)
            
    results = profiler.stop_profiling()
    profiler.print_summary(results)
    profiler.save_results(results)
    
    return results

if __name__ == "__main__":
    profile_image_operations()