#!/usr/bin/env python3
"""
Enhanced Memory Manager for WAN22 Performance Optimization
Provides advanced memory management with automatic optimization and monitoring
"""

import gc
import psutil
import threading
import time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import weakref

@dataclass
class MemoryStats:
    """Memory usage statistics"""
    total_system_mb: float
    available_system_mb: float
    used_system_percent: float
    process_memory_mb: float
    process_memory_percent: float
    cache_memory_mb: float
    gpu_memory_mb: float = 0.0
    gpu_memory_percent: float = 0.0

@dataclass
class MemoryThresholds:
    """Memory usage thresholds for optimization"""
    warning_percent: float = 75.0
    critical_percent: float = 85.0
    emergency_percent: float = 95.0
    cache_limit_percent: float = 20.0  # Max cache as % of available memory

class EnhancedMemoryManager:
    """Advanced memory management with automatic optimization"""
    
    def __init__(self, thresholds: MemoryThresholds = None):
        self.thresholds = thresholds or MemoryThresholds()
        self._monitoring = False
        self._monitor_thread = None
        self._callbacks: List[Callable[[MemoryStats], None]] = []
        self._optimization_callbacks: List[Callable[[], None]] = []
        
        # Memory tracking
        self._memory_history: List[MemoryStats] = []
        self._max_history = 100
        
        # Weak references to managed objects
        self._managed_objects: List[weakref.ref] = []
        
        # GPU memory tracking (if available)
        self._gpu_available = self._check_gpu_availability()
        
    def _check_gpu_availability(self) -> bool:
        """Check if GPU memory monitoring is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
            
    def start_monitoring(self, interval: float = 5.0):
        """Start memory monitoring"""
        if self._monitoring:
            return
            
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self._monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop memory monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
            
    def _monitor_loop(self, interval: float):
        """Main monitoring loop"""
        while self._monitoring:
            try:
                stats = self.get_memory_stats()
                self._memory_history.append(stats)
                
                # Keep history size manageable
                if len(self._memory_history) > self._max_history:
                    self._memory_history.pop(0)
                    
                # Check thresholds and trigger optimizations
                self._check_memory_thresholds(stats)
                
                # Notify callbacks
                for callback in self._callbacks:
                    try:
                        callback(stats)
                    except Exception as e:
                        print(f"Error in memory callback: {e}")
                        
            except Exception as e:
                print(f"Error in memory monitoring: {e}")
                
            time.sleep(interval)
            
    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics"""
        # System memory
        system_memory = psutil.virtual_memory()
        
        # Process memory
        process = psutil.Process()
        process_memory = process.memory_info()
        
        # GPU memory (if available)
        gpu_memory_mb = 0.0
        gpu_memory_percent = 0.0
        
        if self._gpu_available:
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                    gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                    gpu_memory_percent = (gpu_memory_mb / gpu_total) * 100 if gpu_total > 0 else 0
            except Exception:
                pass
                
        # Estimate cache memory (from global cache if available)
        cache_memory_mb = 0.0
        try:
            from optimized_image_cache import get_global_cache
            cache = get_global_cache()
            cache_stats = cache.get_stats()
            cache_memory_mb = cache_stats.total_memory_mb
        except Exception:
            pass
            
        return MemoryStats(
            total_system_mb=system_memory.total / (1024 * 1024),
            available_system_mb=system_memory.available / (1024 * 1024),
            used_system_percent=system_memory.percent,
            process_memory_mb=process_memory.rss / (1024 * 1024),
            process_memory_percent=process.memory_percent(),
            cache_memory_mb=cache_memory_mb,
            gpu_memory_mb=gpu_memory_mb,
            gpu_memory_percent=gpu_memory_percent
        )
        
    def _check_memory_thresholds(self, stats: MemoryStats):
        """Check memory thresholds and trigger optimizations"""
        if stats.used_system_percent >= self.thresholds.emergency_percent:
            self._trigger_emergency_optimization(stats)
        elif stats.used_system_percent >= self.thresholds.critical_percent:
            self._trigger_critical_optimization(stats)
        elif stats.used_system_percent >= self.thresholds.warning_percent:
            self._trigger_warning_optimization(stats)
            
    def _trigger_emergency_optimization(self, stats: MemoryStats):
        """Trigger emergency memory optimization"""
        print(f"EMERGENCY: System memory usage at {stats.used_system_percent:.1f}%")
        
        # Force garbage collection
        gc.collect()
        
        # Clear image cache aggressively
        try:
            from optimized_image_cache import get_global_cache
            cache = get_global_cache()
            cache.clear()
            print("Emergency: Cleared image cache")
        except Exception:
            pass
            
        # Clear GPU cache if available
        if self._gpu_available:
            try:
                import torch
                torch.cuda.empty_cache()
                print("Emergency: Cleared GPU cache")
            except Exception:
                pass
                
        # Trigger optimization callbacks
        for callback in self._optimization_callbacks:
            try:
                callback()
            except Exception as e:
                print(f"Error in emergency optimization callback: {e}")
                
    def _trigger_critical_optimization(self, stats: MemoryStats):
        """Trigger critical memory optimization"""
        print(f"CRITICAL: System memory usage at {stats.used_system_percent:.1f}%")
        
        # Force garbage collection
        gc.collect()
        
        # Reduce cache size
        try:
            from optimized_image_cache import get_global_cache
            cache = get_global_cache()
            current_limit = cache.max_memory_mb
            new_limit = max(32, current_limit * 0.5)  # Reduce to 50%
            cache.max_memory_mb = new_limit
            print(f"Critical: Reduced cache limit to {new_limit}MB")
        except Exception:
            pass
            
    def _trigger_warning_optimization(self, stats: MemoryStats):
        """Trigger warning-level memory optimization"""
        print(f"WARNING: System memory usage at {stats.used_system_percent:.1f}%")
        
        # Gentle garbage collection
        gc.collect()
        
        # Reduce cache size slightly
        try:
            from optimized_image_cache import get_global_cache
            cache = get_global_cache()
            current_limit = cache.max_memory_mb
            new_limit = max(64, current_limit * 0.8)  # Reduce to 80%
            cache.max_memory_mb = new_limit
        except Exception:
            pass
            
    def register_memory_callback(self, callback: Callable[[MemoryStats], None]):
        """Register callback for memory statistics updates"""
        self._callbacks.append(callback)
        
    def register_optimization_callback(self, callback: Callable[[], None]):
        """Register callback for memory optimization events"""
        self._optimization_callbacks.append(callback)
        
    def get_optimal_cache_size(self) -> float:
        """Get optimal cache size based on available memory"""
        stats = self.get_memory_stats()
        
        # Use percentage of available memory for cache
        optimal_size = stats.available_system_mb * (self.thresholds.cache_limit_percent / 100)
        
        # Ensure minimum and maximum bounds
        optimal_size = max(32, min(512, optimal_size))
        
        return optimal_size
        
    def get_memory_recommendations(self) -> List[str]:
        """Get memory optimization recommendations"""
        stats = self.get_memory_stats()
        recommendations = []
        
        if stats.used_system_percent > 80:
            recommendations.append("System memory usage is high - consider closing other applications")
            
        if stats.cache_memory_mb > stats.available_system_mb * 0.3:
            recommendations.append("Image cache is using significant memory - consider reducing cache size")
            
        if stats.process_memory_mb > 1000:
            recommendations.append("Process memory usage is high - consider restarting the application")
            
        if self._gpu_available and stats.gpu_memory_percent > 80:
            recommendations.append("GPU memory usage is high - consider reducing batch sizes or image resolution")
            
        if not recommendations:
            recommendations.append("Memory usage is within normal limits")
            
        return recommendations
        
    def get_memory_trend(self, minutes: int = 10) -> Dict[str, float]:
        """Get memory usage trend over specified time period"""
        if not self._memory_history:
            return {}
            
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent_stats = [s for s in self._memory_history if hasattr(s, 'timestamp') and s.timestamp > cutoff_time]
        
        if len(recent_stats) < 2:
            return {}
            
        # Calculate trends
        first_stat = recent_stats[0]
        last_stat = recent_stats[-1]
        
        return {
            'system_memory_trend': last_stat.used_system_percent - first_stat.used_system_percent,
            'process_memory_trend': last_stat.process_memory_mb - first_stat.process_memory_mb,
            'cache_memory_trend': last_stat.cache_memory_mb - first_stat.cache_memory_mb,
            'gpu_memory_trend': last_stat.gpu_memory_mb - first_stat.gpu_memory_mb if self._gpu_available else 0.0
        }
        
    def optimize_for_large_images(self, image_count: int = 1, image_size: tuple = (1920, 1080)) -> Dict[str, Any]:
        """Get optimization settings for processing large images"""
        stats = self.get_memory_stats()
        
        # Estimate memory needed for images
        estimated_memory_per_image = (image_size[0] * image_size[1] * 3) / (1024 * 1024)  # RGB in MB
        total_estimated_memory = estimated_memory_per_image * image_count
        
        # Determine if we can handle the load
        available_memory = stats.available_system_mb
        
        if total_estimated_memory > available_memory * 0.8:
            # Not enough memory - suggest optimizations
            return {
                'can_process': False,
                'suggested_batch_size': max(1, int(available_memory * 0.6 / estimated_memory_per_image)),
                'suggested_max_size': (1280, 720),
                'enable_streaming': True,
                'reduce_cache': True,
                'recommendations': [
                    f"Reduce image size from {image_size} to (1280, 720)",
                    f"Process images in batches of {max(1, int(available_memory * 0.6 / estimated_memory_per_image))}",
                    "Enable image streaming to reduce memory usage",
                    "Reduce cache size to free up memory"
                ]
            }
        else:
            # Sufficient memory available
            return {
                'can_process': True,
                'suggested_batch_size': image_count,
                'suggested_max_size': image_size,
                'enable_streaming': False,
                'reduce_cache': False,
                'recommendations': [
                    "Memory is sufficient for processing",
                    "Consider enabling caching for better performance"
                ]
            }
            
    def print_memory_report(self):
        """Print comprehensive memory report"""
        stats = self.get_memory_stats()
        recommendations = self.get_memory_recommendations()
        
        print("\n" + "="*60)
        print("ENHANCED MEMORY MANAGER REPORT")
        print("="*60)
        
        print(f"SYSTEM MEMORY:")
        print(f"  Total: {stats.total_system_mb:.1f}MB")
        print(f"  Available: {stats.available_system_mb:.1f}MB")
        print(f"  Used: {stats.used_system_percent:.1f}%")
        
        print(f"\nPROCESS MEMORY:")
        print(f"  Usage: {stats.process_memory_mb:.1f}MB")
        print(f"  Percentage: {stats.process_memory_percent:.1f}%")
        
        print(f"\nCACHE MEMORY:")
        print(f"  Usage: {stats.cache_memory_mb:.1f}MB")
        
        if self._gpu_available:
            print(f"\nGPU MEMORY:")
            print(f"  Usage: {stats.gpu_memory_mb:.1f}MB")
            print(f"  Percentage: {stats.gpu_memory_percent:.1f}%")
            
        print(f"\nOPTIMAL CACHE SIZE: {self.get_optimal_cache_size():.1f}MB")
        
        print(f"\nRECOMMENDATIONS:")
        for rec in recommendations:
            print(f"  • {rec}")
            
        # Memory trend if available
        trend = self.get_memory_trend()
        if trend:
            print(f"\nMEMORY TRENDS (Last 10 minutes):")
            for key, value in trend.items():
                direction = "↑" if value > 0 else "↓" if value < 0 else "→"
                print(f"  {key}: {direction} {abs(value):.1f}")
                
        print("="*60)

# Global memory manager instance
_global_memory_manager: Optional[EnhancedMemoryManager] = None
_memory_manager_lock = threading.Lock()

def get_global_memory_manager() -> EnhancedMemoryManager:
    """Get global memory manager instance"""
    global _global_memory_manager
    
    with _memory_manager_lock:
        if _global_memory_manager is None:
            _global_memory_manager = EnhancedMemoryManager()
            _global_memory_manager.start_monitoring()
        return _global_memory_manager

def optimize_memory_for_images(image_count: int = 1, image_size: tuple = (1920, 1080)) -> Dict[str, Any]:
    """Convenience function for image memory optimization"""
    manager = get_global_memory_manager()
    return manager.optimize_for_large_images(image_count, image_size)

def get_memory_recommendations() -> List[str]:
    """Convenience function for memory recommendations"""
    manager = get_global_memory_manager()
    return manager.get_memory_recommendations()

if __name__ == "__main__":
    # Test the enhanced memory manager
    manager = EnhancedMemoryManager()
    
    def memory_callback(stats):
        if stats.used_system_percent > 50:  # Low threshold for demo
            print(f"Memory callback: {stats.used_system_percent:.1f}% used")
            
    manager.register_memory_callback(memory_callback)
    manager.start_monitoring(1.0)  # Monitor every second
    
    # Print initial report
    manager.print_memory_report()
    
    # Test optimization recommendations
    print("\nTesting large image optimization:")
    optimization = manager.optimize_for_large_images(5, (2048, 2048))
    print(f"Can process: {optimization['can_process']}")
    print(f"Suggested batch size: {optimization['suggested_batch_size']}")
    
    time.sleep(3)  # Let monitor run
    manager.stop_monitoring()
