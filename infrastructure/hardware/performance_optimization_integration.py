#!/usr/bin/env python3
"""
Performance Optimization Integration for WAN22 Start/End Image Fix
Integrates all performance optimization components with existing systems
"""

import os
import sys
import time
from typing import Dict, Any, Optional
from PIL import Image
import threading

# Import performance optimization components
from image_performance_profiler import ImagePerformanceProfiler
from optimized_image_cache import OptimizedImageCache, cache_image_operation, get_global_cache
from progress_performance_monitor import ProgressPerformanceMonitor, get_global_monitor, track_progress_update

class WAN22PerformanceOptimizer:
    """Main performance optimization coordinator for WAN22"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        
        # Initialize components
        self.profiler = ImagePerformanceProfiler()
        self.cache = OptimizedImageCache(
            max_memory_mb=self.config.get('cache_memory_mb', 256),
            max_entries=self.config.get('cache_max_entries', 50),
            cleanup_interval=self.config.get('cache_cleanup_interval', 300)
        )
        self.monitor = get_global_monitor()
        
        # Performance tracking
        self.optimization_enabled = True
        self.performance_reports = []
        
        # Setup alert handling
        self.monitor.register_alert_callback(self._handle_performance_alert)
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default performance optimization configuration"""
        return {
            'cache_memory_mb': 256,
            'cache_max_entries': 50,
            'cache_cleanup_interval': 300,
            'profiling_enabled': True,
            'monitoring_enabled': True,
            'auto_optimization': True,
            'alert_thresholds': {
                'memory_warning_mb': 400,
                'latency_warning_ms': 200,
                'cpu_warning_percent': 85
            }
        }
        
    def start_optimization(self):
        """Start performance optimization systems"""
        if self.config.get('profiling_enabled', True):
            self.profiler.start_profiling()
            
        # Don't auto-start monitoring during initialization to prevent loops
        # Monitoring can be started manually later if needed
            
        print("WAN22 Performance Optimization: Started")
        
    def stop_optimization(self):
        """Stop performance optimization systems"""
        if self.profiler.is_profiling:
            results = self.profiler.stop_profiling()
            self.performance_reports.append(('profiler', results))
            
        self.monitor.stop_monitoring()
        
        print("WAN22 Performance Optimization: Stopped")
        
    def optimize_image_validation(self, image: Image.Image, validation_func, *args, **kwargs):
        """Optimize image validation with caching and profiling"""
        if not self.optimization_enabled:
            return validation_func(image, *args, **kwargs)
            
        # Use cached operation if available
        return cache_image_operation(
            "image_validation",
            image,
            validation_func,
            *args,
            **kwargs
        )
        
    def optimize_thumbnail_generation(self, image: Image.Image, size: tuple = (256, 256)):
        """Optimize thumbnail generation with caching"""
        def generate_thumbnail(img, target_size):
            thumbnail = img.copy()
            thumbnail.thumbnail(target_size, Image.Resampling.LANCZOS)
            return thumbnail
            
        if not self.optimization_enabled:
            return generate_thumbnail(image, size)
            
        return cache_image_operation(
            f"thumbnail_{size[0]}x{size[1]}",
            image,
            generate_thumbnail,
            size
        )
        
    @track_progress_update
    def optimize_progress_update(self, update_func, *args, **kwargs):
        """Optimize progress updates with monitoring"""
        return update_func(*args, **kwargs)
        
    def _handle_performance_alert(self, alert):
        """Handle performance alerts"""
        print(f"PERFORMANCE ALERT [{alert.severity.upper()}]: {alert.message}")
        
        if self.config.get('auto_optimization', True):
            self._apply_auto_optimization(alert)
            
    def _apply_auto_optimization(self, alert):
        """Apply automatic optimizations based on alerts"""
        if alert.alert_type == 'high_memory':
            # Reduce cache size
            current_limit = self.cache.max_memory_mb
            new_limit = max(64, current_limit * 0.8)
            self.cache.max_memory_mb = new_limit
            print(f"AUTO-OPTIMIZATION: Reduced cache memory limit to {new_limit}MB")
            
        elif alert.alert_type == 'high_latency':
            # Reduce cache entries to speed up operations
            current_entries = self.cache.max_entries
            new_entries = max(10, int(current_entries * 0.8))
            self.cache.max_entries = new_entries
            print(f"AUTO-OPTIMIZATION: Reduced cache entries to {new_entries}")
            
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        cache_stats = self.cache.get_stats()
        monitor_summary = self.monitor.get_performance_summary()
        
        return {
            'cache_performance': {
                'hit_rate': cache_stats.hit_rate,
                'memory_usage_mb': cache_stats.total_memory_mb,
                'entries_count': cache_stats.entries_count,
                'total_requests': cache_stats.total_requests
            },
            'monitoring_performance': monitor_summary,
            'optimization_status': {
                'enabled': self.optimization_enabled,
                'profiling_active': self.profiler.is_profiling,
                'monitoring_active': self.monitor._monitoring
            }
        }
        
    def print_performance_report(self):
        """Print comprehensive performance report"""
        summary = self.get_performance_summary()
        
        print("\n" + "="*60)
        print("WAN22 PERFORMANCE OPTIMIZATION REPORT")
        print("="*60)
        
        # Cache performance
        cache_perf = summary['cache_performance']
        print(f"CACHE PERFORMANCE:")
        print(f"  Hit Rate: {cache_perf['hit_rate']:.2%}")
        print(f"  Memory Usage: {cache_perf['memory_usage_mb']:.2f}MB")
        print(f"  Cached Entries: {cache_perf['entries_count']}")
        print(f"  Total Requests: {cache_perf['total_requests']}")
        
        # Monitoring performance
        monitor_perf = summary['monitoring_performance']
        if monitor_perf:
            print(f"\nMONITORING PERFORMANCE:")
            print(f"  Average Latency: {monitor_perf.get('average_latency_ms', 0):.1f}ms")
            print(f"  Peak Memory: {monitor_perf.get('peak_memory_mb', 0):.1f}MB")
            print(f"  Average CPU: {monitor_perf.get('average_cpu_percent', 0):.1f}%")
            print(f"  Total Alerts: {monitor_perf.get('total_alerts', 0)}")
            
        # Optimization status
        opt_status = summary['optimization_status']
        print(f"\nOPTIMIZATION STATUS:")
        print(f"  Optimization Enabled: {opt_status['enabled']}")
        print(f"  Profiling Active: {opt_status['profiling_active']}")
        print(f"  Monitoring Active: {opt_status['monitoring_active']}")
        
        print("="*60)
        
    def export_performance_data(self, filename: str = None) -> str:
        """Export all performance data"""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"wan22_performance_data_{timestamp}.json"
            
        # Collect all performance data
        data = {
            'summary': self.get_performance_summary(),
            'profiler_reports': [],
            'monitor_data': self.monitor.export_metrics()
        }
        
        # Add profiler reports
        for report_type, report_data in self.performance_reports:
            if report_type == 'profiler':
                data['profiler_reports'].append({
                    'total_operations': report_data.total_operations,
                    'total_time': report_data.total_time,
                    'bottlenecks': report_data.bottlenecks,
                    'recommendations': report_data.recommendations
                })
                
        import json
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"Performance data exported to {filename}")
        return filename

# Global optimizer instance
_global_optimizer: Optional[WAN22PerformanceOptimizer] = None
_optimizer_lock = threading.Lock()

def get_global_optimizer(config: Dict[str, Any] = None) -> WAN22PerformanceOptimizer:
    """Get global performance optimizer instance"""
    global _global_optimizer
    
    with _optimizer_lock:
        if _global_optimizer is None:
            _global_optimizer = WAN22PerformanceOptimizer(config)
            _global_optimizer.start_optimization()
        return _global_optimizer

def shutdown_global_optimizer():
    """Shutdown global performance optimizer"""
    global _global_optimizer
    
    with _optimizer_lock:
        if _global_optimizer is not None:
            _global_optimizer.stop_optimization()
            _global_optimizer = None

# Convenience functions for integration with existing code
def optimized_image_validation(image: Image.Image, validation_func, *args, **kwargs):
    """Convenience function for optimized image validation"""
    optimizer = get_global_optimizer()
    return optimizer.optimize_image_validation(image, validation_func, *args, **kwargs)

def optimized_thumbnail_generation(image: Image.Image, size: tuple = (256, 256)):
    """Convenience function for optimized thumbnail generation"""
    optimizer = get_global_optimizer()
    return optimizer.optimize_thumbnail_generation(image, size)

def optimized_progress_update(update_func, *args, **kwargs):
    """Convenience function for optimized progress updates"""
    optimizer = get_global_optimizer()
    return optimizer.optimize_progress_update(update_func, *args, **kwargs)

def demo_performance_optimization():
    """Demonstrate performance optimization features"""
    print("WAN22 Performance Optimization Demo")
    print("="*40)
    
    # Initialize optimizer
    optimizer = WAN22PerformanceOptimizer()
    optimizer.start_optimization()
    
    try:
        # Demo image operations
        print("\n1. Testing Image Validation Optimization...")
        test_image = Image.new('RGB', (512, 512), color='red')
        
        def dummy_validation(img):
            time.sleep(0.02)  # Simulate validation work
            return {"valid": True, "dimensions": img.size}
            
        # First validation (cache miss)
        start_time = time.time()
        result1 = optimizer.optimize_image_validation(test_image, dummy_validation)
        time1 = time.time() - start_time
        
        # Second validation (cache hit)
        start_time = time.time()
        result2 = optimizer.optimize_image_validation(test_image, dummy_validation)
        time2 = time.time() - start_time
        
        print(f"  First validation: {time1:.3f}s")
        print(f"  Second validation: {time2:.3f}s")
        print(f"  Speedup: {time1/time2:.1f}x")
        
        # Demo thumbnail optimization
        print("\n2. Testing Thumbnail Generation Optimization...")
        start_time = time.time()
        thumbnail1 = optimizer.optimize_thumbnail_generation(test_image, (256, 256))
        time1 = time.time() - start_time
        
        start_time = time.time()
        thumbnail2 = optimizer.optimize_thumbnail_generation(test_image, (256, 256))
        time2 = time.time() - start_time
        
        print(f"  First thumbnail: {time1:.3f}s")
        print(f"  Second thumbnail: {time2:.3f}s")
        print(f"  Speedup: {time1/time2:.1f}x" if time2 > 0 else "  Instant cache hit!")
        
        # Demo progress tracking
        print("\n3. Testing Progress Tracking Optimization...")
        
        def dummy_progress_update(step, total):
            time.sleep(0.01)
            return f"Step {step}/{total}"
            
        for i in range(5):
            result = optimizer.optimize_progress_update(dummy_progress_update, i+1, 5)
            
        time.sleep(0.5)  # Let monitor collect data
        
        # Show performance report
        print("\n4. Performance Report:")
        optimizer.print_performance_report()
        
    finally:
        optimizer.stop_optimization()

if __name__ == "__main__":
    demo_performance_optimization()