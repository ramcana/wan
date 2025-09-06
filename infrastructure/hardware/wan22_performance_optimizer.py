import pytest
#!/usr/bin/env python3
"""
Wan2.2 Performance Optimizer - Final performance optimizations and cleanup
Provides advanced performance optimizations and system cleanup capabilities
"""

import gc
import time
import threading
import psutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, asdict
import json
import logging

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


@dataclass
class PerformanceMetrics:
    """Performance metrics tracking"""
    cpu_usage: float
    memory_usage: float
    gpu_usage: float = 0.0
    gpu_memory_usage: float = 0.0
    disk_io_read: float = 0.0
    disk_io_write: float = 0.0
    network_io: float = 0.0
    timestamp: float = 0.0


@dataclass
class OptimizationResult:
    """Result of performance optimization"""
    success: bool
    optimizations_applied: List[str]
    memory_freed_mb: float
    performance_improvement: float
    warnings: List[str]
    errors: List[str]


class SystemMonitor:
    """Real-time system performance monitoring"""
    
    def __init__(self, update_interval: float = 1.0):
        self.update_interval = update_interval
        self.monitoring = False
        self.metrics_history = []
        self.monitor_thread = None
        self.logger = logging.getLogger(__name__)
        
        # Performance thresholds
        self.thresholds = {
            'cpu_usage': 80.0,
            'memory_usage': 85.0,
            'gpu_memory_usage': 90.0,
            'disk_io_threshold': 100.0  # MB/s
        }
    
    def start_monitoring(self):
        """Start system monitoring"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop system monitoring"""
        self.monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        self.logger.info("System monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Keep only last 1000 measurements
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-1000:]
                
                # Check for performance issues
                self._check_performance_alerts(metrics)
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.update_interval)
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current system metrics"""
        # CPU and memory
        cpu_usage = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        # Disk I/O
        disk_io = psutil.disk_io_counters()
        disk_read = disk_io.read_bytes / 1024 / 1024 if disk_io else 0  # MB
        disk_write = disk_io.write_bytes / 1024 / 1024 if disk_io else 0  # MB
        
        # Network I/O
        net_io = psutil.net_io_counters()
        network_io = (net_io.bytes_sent + net_io.bytes_recv) / 1024 / 1024 if net_io else 0  # MB
        
        # GPU metrics
        gpu_usage = 0.0
        gpu_memory_usage = 0.0
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                gpu_usage = torch.cuda.utilization()
                gpu_memory = torch.cuda.memory_stats()
                allocated = gpu_memory.get('allocated_bytes.all.current', 0)
                reserved = gpu_memory.get('reserved_bytes.all.current', 0)
                total = torch.cuda.get_device_properties(0).total_memory
                gpu_memory_usage = (allocated / total) * 100 if total > 0 else 0
            except Exception:
                pass
        
        return PerformanceMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            gpu_usage=gpu_usage,
            gpu_memory_usage=gpu_memory_usage,
            disk_io_read=disk_read,
            disk_io_write=disk_write,
            network_io=network_io,
            timestamp=time.time()
        )
    
    def _check_performance_alerts(self, metrics: PerformanceMetrics):
        """Check for performance alerts"""
        alerts = []
        
        if metrics.cpu_usage > self.thresholds['cpu_usage']:
            alerts.append(f"High CPU usage: {metrics.cpu_usage:.1f}%")
        
        if metrics.memory_usage > self.thresholds['memory_usage']:
            alerts.append(f"High memory usage: {metrics.memory_usage:.1f}%")
        
        if metrics.gpu_memory_usage > self.thresholds['gpu_memory_usage']:
            alerts.append(f"High GPU memory usage: {metrics.gpu_memory_usage:.1f}%")
        
        if alerts:
            self.logger.warning(f"Performance alerts: {'; '.join(alerts)}")
    
    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Get current performance metrics"""
        if self.metrics_history:
            return self.metrics_history[-1]
        return self._collect_metrics()
    
    def get_average_metrics(self, duration_seconds: int = 60) -> Optional[PerformanceMetrics]:
        """Get average metrics over specified duration"""
        if not self.metrics_history:
            return None
        
        cutoff_time = time.time() - duration_seconds
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return None
        
        # Calculate averages
        avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
        avg_gpu = sum(m.gpu_usage for m in recent_metrics) / len(recent_metrics)
        avg_gpu_memory = sum(m.gpu_memory_usage for m in recent_metrics) / len(recent_metrics)
        
        return PerformanceMetrics(
            cpu_usage=avg_cpu,
            memory_usage=avg_memory,
            gpu_usage=avg_gpu,
            gpu_memory_usage=avg_gpu_memory,
            disk_io_read=0.0,
            disk_io_write=0.0,
            network_io=0.0,
            timestamp=time.time()
        )


class MemoryOptimizer:
    """Advanced memory optimization and cleanup"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.cleanup_callbacks = []
    
    def register_cleanup_callback(self, callback: Callable[[], None]):
        """Register a cleanup callback"""
        self.cleanup_callbacks.append(callback)
    
    def optimize_memory(self, aggressive: bool = False) -> OptimizationResult:
        """Perform memory optimization"""
        optimizations = []
        memory_before = self._get_memory_usage()
        warnings = []
        errors = []
        
        try:
            # Python garbage collection
            collected = gc.collect()
            if collected > 0:
                optimizations.append(f"Python GC: collected {collected} objects")
            
            # PyTorch memory cleanup
            if TORCH_AVAILABLE and torch.cuda.is_available():
                try:
                    # Clear CUDA cache
                    torch.cuda.empty_cache()
                    optimizations.append("CUDA cache cleared")
                    
                    if aggressive:
                        # Synchronize CUDA operations
                        torch.cuda.synchronize()
                        optimizations.append("CUDA synchronized")
                        
                        # Reset peak memory stats
                        torch.cuda.reset_peak_memory_stats()
                        optimizations.append("CUDA memory stats reset")
                        
                except Exception as e:
                    errors.append(f"CUDA cleanup error: {e}")
            
            # NumPy memory cleanup
            if NUMPY_AVAILABLE and aggressive:
                try:
                    # This is a placeholder - NumPy doesn't have explicit cleanup
                    # but we can suggest garbage collection
                    optimizations.append("NumPy arrays eligible for GC")
                except Exception as e:
                    errors.append(f"NumPy cleanup error: {e}")
            
            # Run registered cleanup callbacks
            for callback in self.cleanup_callbacks:
                try:
                    callback()
                    optimizations.append("Custom cleanup callback executed")
                except Exception as e:
                    errors.append(f"Cleanup callback error: {e}")
            
            # System-level memory optimization
            if aggressive:
                try:
                    # Force garbage collection multiple times
                    for _ in range(3):
                        gc.collect()
                    optimizations.append("Aggressive garbage collection")
                except Exception as e:
                    errors.append(f"Aggressive GC error: {e}")
            
            memory_after = self._get_memory_usage()
            memory_freed = memory_before - memory_after
            
            # Calculate performance improvement estimate
            performance_improvement = min(memory_freed / memory_before * 100, 100) if memory_before > 0 else 0
            
            result = OptimizationResult(
                success=len(errors) == 0,
                optimizations_applied=optimizations,
                memory_freed_mb=memory_freed,
                performance_improvement=performance_improvement,
                warnings=warnings,
                errors=errors
            )
            
            self.logger.info(f"Memory optimization completed: freed {memory_freed:.1f}MB")
            return result
            
        except Exception as e:
            self.logger.error(f"Memory optimization failed: {e}")
            return OptimizationResult(
                success=False,
                optimizations_applied=optimizations,
                memory_freed_mb=0.0,
                performance_improvement=0.0,
                warnings=warnings,
                errors=[str(e)]
            )
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get detailed memory information"""
        info = {}
        
        # System memory
        memory = psutil.virtual_memory()
        info['system'] = {
            'total_mb': memory.total / 1024 / 1024,
            'available_mb': memory.available / 1024 / 1024,
            'used_mb': memory.used / 1024 / 1024,
            'percent': memory.percent
        }
        
        # Process memory
        try:
            process = psutil.Process()
            process_memory = process.memory_info()
            info['process'] = {
                'rss_mb': process_memory.rss / 1024 / 1024,
                'vms_mb': process_memory.vms / 1024 / 1024,
                'percent': process.memory_percent()
            }
        except Exception:
            info['process'] = {'error': 'Could not get process memory info'}
        
        # GPU memory
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.memory_stats()
                total_memory = torch.cuda.get_device_properties(0).total_memory
                
                info['gpu'] = {
                    'total_mb': total_memory / 1024 / 1024,
                    'allocated_mb': gpu_memory.get('allocated_bytes.all.current', 0) / 1024 / 1024,
                    'reserved_mb': gpu_memory.get('reserved_bytes.all.current', 0) / 1024 / 1024,
                    'free_mb': (total_memory - gpu_memory.get('allocated_bytes.all.current', 0)) / 1024 / 1024
                }
            except Exception as e:
                info['gpu'] = {'error': f'Could not get GPU memory info: {e}'}
        else:
            info['gpu'] = {'available': False}
        
        return info


class CacheManager:
    """Manages various caches and temporary files"""
    
    def __init__(self, cache_dirs: Optional[List[str]] = None):
        self.cache_dirs = cache_dirs or [
            "outputs/.cache",
            "models/.cache", 
            ".gradio_temp",
            "__pycache__",
            ".pytest_cache"
        ]
        self.logger = logging.getLogger(__name__)
    
    def cleanup_caches(self, max_age_hours: int = 24) -> OptimizationResult:
        """Clean up cache directories"""
        optimizations = []
        warnings = []
        errors = []
        total_freed = 0.0
        
        cutoff_time = time.time() - (max_age_hours * 3600)
        
        for cache_dir in self.cache_dirs:
            cache_path = Path(cache_dir)
            if not cache_path.exists():
                continue
            
            try:
                freed_mb = self._cleanup_directory(cache_path, cutoff_time)
                if freed_mb > 0:
                    optimizations.append(f"Cleaned {cache_dir}: {freed_mb:.1f}MB freed")
                    total_freed += freed_mb
                
            except Exception as e:
                errors.append(f"Error cleaning {cache_dir}: {e}")
        
        # Clean up temporary files
        try:
            temp_freed = self._cleanup_temp_files()
            if temp_freed > 0:
                optimizations.append(f"Cleaned temp files: {temp_freed:.1f}MB freed")
                total_freed += temp_freed
        except Exception as e:
            errors.append(f"Error cleaning temp files: {e}")
        
        return OptimizationResult(
            success=len(errors) == 0,
            optimizations_applied=optimizations,
            memory_freed_mb=total_freed,
            performance_improvement=min(total_freed / 100, 10),  # Estimate
            warnings=warnings,
            errors=errors
        )
    
    def _cleanup_directory(self, directory: Path, cutoff_time: float) -> float:
        """Clean up a specific directory"""
        freed_mb = 0.0
        
        for item in directory.rglob("*"):
            if item.is_file():
                try:
                    stat = item.stat()
                    if stat.st_mtime < cutoff_time:
                        size_mb = stat.st_size / 1024 / 1024
                        item.unlink()
                        freed_mb += size_mb
                except Exception:
                    continue
        
        return freed_mb
    
    def _cleanup_temp_files(self) -> float:
        """Clean up temporary files"""
        freed_mb = 0.0
        temp_patterns = ["*.tmp", "*.temp", ".DS_Store", "Thumbs.db"]
        
        for pattern in temp_patterns:
            for temp_file in Path(".").rglob(pattern):
                try:
                    if temp_file.is_file():
                        size_mb = temp_file.stat().st_size / 1024 / 1024
                        temp_file.unlink()
                        freed_mb += size_mb
                except Exception:
                    continue
        
        return freed_mb


class Wan22PerformanceOptimizer:
    """Main performance optimizer for Wan2.2 system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.system_monitor = SystemMonitor()
        self.memory_optimizer = MemoryOptimizer()
        self.cache_manager = CacheManager()
        
        # Performance tracking
        self.optimization_history = []
        self.performance_baseline = None
    
    def start_monitoring(self):
        """Start performance monitoring"""
        self.system_monitor.start_monitoring()
        self.performance_baseline = self.system_monitor.get_current_metrics()
        self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.system_monitor.stop_monitoring()
        self.logger.info("Performance monitoring stopped")
    
    def optimize_system(self, aggressive: bool = False) -> OptimizationResult:
        """Perform comprehensive system optimization"""
        self.logger.info(f"Starting system optimization (aggressive={aggressive})")
        
        all_optimizations = []
        all_warnings = []
        all_errors = []
        total_memory_freed = 0.0
        
        # Memory optimization
        memory_result = self.memory_optimizer.optimize_memory(aggressive)
        all_optimizations.extend(memory_result.optimizations_applied)
        all_warnings.extend(memory_result.warnings)
        all_errors.extend(memory_result.errors)
        total_memory_freed += memory_result.memory_freed_mb
        
        # Cache cleanup
        cache_result = self.cache_manager.cleanup_caches(max_age_hours=24 if not aggressive else 1)
        all_optimizations.extend(cache_result.optimizations_applied)
        all_warnings.extend(cache_result.warnings)
        all_errors.extend(cache_result.errors)
        total_memory_freed += cache_result.memory_freed_mb
        
        # Calculate overall performance improvement
        performance_improvement = (
            memory_result.performance_improvement + cache_result.performance_improvement
        ) / 2
        
        result = OptimizationResult(
            success=len(all_errors) == 0,
            optimizations_applied=all_optimizations,
            memory_freed_mb=total_memory_freed,
            performance_improvement=performance_improvement,
            warnings=all_warnings,
            errors=all_errors
        )
        
        # Record optimization
        self.optimization_history.append({
            'timestamp': time.time(),
            'result': asdict(result),
            'aggressive': aggressive
        })
        
        self.logger.info(f"System optimization completed: {len(all_optimizations)} optimizations applied")
        return result
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        current_metrics = self.system_monitor.get_current_metrics()
        avg_metrics = self.system_monitor.get_average_metrics(300)  # 5 minutes
        memory_info = self.memory_optimizer.get_memory_info()
        
        report = {
            'timestamp': time.time(),
            'current_metrics': asdict(current_metrics) if current_metrics else None,
            'average_metrics_5min': asdict(avg_metrics) if avg_metrics else None,
            'memory_info': memory_info,
            'optimization_history': self.optimization_history[-10:],  # Last 10 optimizations
            'recommendations': self._generate_recommendations(current_metrics)
        }
        
        return report
    
    def _generate_recommendations(self, metrics: Optional[PerformanceMetrics]) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        if not metrics:
            return ["Enable monitoring to get performance recommendations"]
        
        if metrics.cpu_usage > 80:
            recommendations.append("High CPU usage detected - consider reducing concurrent operations")
        
        if metrics.memory_usage > 85:
            recommendations.append("High memory usage - run memory optimization or reduce model size")
        
        if metrics.gpu_memory_usage > 90:
            recommendations.append("GPU memory nearly full - enable CPU offloading or reduce batch size")
        
        if metrics.gpu_usage < 50 and TORCH_AVAILABLE and torch.cuda.is_available():
            recommendations.append("Low GPU utilization - check if CPU offloading is unnecessarily enabled")
        
        if not recommendations:
            recommendations.append("System performance is optimal")
        
        return recommendations
    
    def cleanup(self):
        """Final cleanup of performance optimizer"""
        self.logger.info("Cleaning up performance optimizer")
        
        # Stop monitoring
        self.stop_monitoring()
        
        # Final optimization
        final_result = self.optimize_system(aggressive=True)
        
        # Save performance report
        try:
            report = self.get_performance_report()
            report_path = Path("diagnostics/final_performance_report.json")
            report_path.parent.mkdir(exist_ok=True)
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Final performance report saved to {report_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save performance report: {e}")
        
        self.logger.info("Performance optimizer cleanup completed")


# Global performance optimizer instance
_performance_optimizer: Optional[Wan22PerformanceOptimizer] = None


def get_performance_optimizer() -> Wan22PerformanceOptimizer:
    """Get or create the global performance optimizer"""
    global _performance_optimizer
    
    if _performance_optimizer is None:
        _performance_optimizer = Wan22PerformanceOptimizer()
    
    return _performance_optimizer


def cleanup_performance_optimizer():
    """Cleanup the global performance optimizer"""
    global _performance_optimizer
    
    if _performance_optimizer is not None:
        _performance_optimizer.cleanup()
        _performance_optimizer = None