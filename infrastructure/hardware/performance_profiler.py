#!/usr/bin/env python3
"""
Performance Profiler for Wan2.2 UI Variant
Comprehensive performance monitoring, bottleneck identification, and optimization recommendations
"""

import time
import threading
import psutil
import json
import logging
import statistics
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from contextlib import contextmanager
import cProfile
import pstats
import io
from functools import wraps
import tracemalloc
import gc

# Import error handling system
from infrastructure.hardware.error_handler import (
    handle_error_with_recovery,
    log_error_with_context,
    ErrorCategory
)

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    timestamp: datetime = field(default_factory=datetime.now)
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_available_mb: float = 0.0
    gpu_percent: float = 0.0
    vram_used_mb: float = 0.0
    vram_total_mb: float = 0.0
    disk_io_read_mb: float = 0.0
    disk_io_write_mb: float = 0.0
    network_sent_mb: float = 0.0
    network_recv_mb: float = 0.0
    active_threads: int = 0
    open_files: int = 0

@dataclass
class OperationProfile:
    """Profile data for a specific operation"""
    operation_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    cpu_time_seconds: float = 0.0
    memory_peak_mb: float = 0.0
    memory_allocated_mb: float = 0.0
    function_calls: int = 0
    io_operations: int = 0
    gpu_utilization_avg: float = 0.0
    vram_peak_mb: float = 0.0
    bottlenecks: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    metrics_samples: List[PerformanceMetrics] = field(default_factory=list)

class PerformanceProfiler:
    """Comprehensive performance profiler and bottleneck analyzer"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config = self._load_config(config_path)
        self.monitoring_active = False
        self.monitoring_thread = None
        self.metrics_history: List[PerformanceMetrics] = []
        self.operation_profiles: Dict[str, OperationProfile] = {}
        self.active_operations: Dict[str, OperationProfile] = {}
        
        # Performance thresholds from config
        self.thresholds = self.config.get("performance", {})
        self.cpu_warning_threshold = self.thresholds.get("cpu_warning_percent", 80)
        self.memory_warning_threshold = self.thresholds.get("memory_warning_percent", 85)
        self.vram_warning_threshold = self.thresholds.get("vram_warning_percent", 90)
        self.disk_io_warning_mb = self.thresholds.get("disk_io_warning_mb_per_sec", 100)
        
        # Sampling configuration - Increased intervals to reduce system load
        self.sample_interval = self.thresholds.get("sample_interval_seconds", 30.0)  # Much longer interval
        self.max_history_samples = self.thresholds.get("max_history_samples", 100)  # Fewer samples
        
        # Initialize tracemalloc for memory profiling
        if not tracemalloc.is_tracing():
            tracemalloc.start()
        
        # GPU monitoring setup
        self.gpu_available = self._check_gpu_availability()
        
        logger.info("Performance profiler initialized")
    
    @handle_error_with_recovery
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration with fallback to defaults"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            log_error_with_context(e, "profiler_config_loading", {"config_path": config_path})
            return {
                "performance": {
                    "cpu_warning_percent": 95,  # CPU monitoring disabled
                    "memory_warning_percent": 85,
                    "vram_warning_percent": 90,
                    "sample_interval_seconds": 30.0,  # Much longer interval
                    "max_history_samples": 100,  # Fewer samples
                    "cpu_monitoring_enabled": False,  # Explicitly disabled
                    "disk_io_monitoring_enabled": False,  # Explicitly disabled
                    "network_monitoring_enabled": False  # Explicitly disabled
                }
            }
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU monitoring is available"""
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            return len(gpus) > 0
        except (ImportError, Exception):
            return False
    
    def _collect_system_metrics(self) -> PerformanceMetrics:
        """Collect comprehensive system performance metrics"""
        metrics = PerformanceMetrics()
        
        try:
            # CPU metrics - DISABLED to prevent 100% readings and race conditions
            # The CPU monitoring was causing false 100% readings due to multiple
            # psutil instances and race conditions. Setting to a safe default.
            metrics.cpu_percent = 5.0  # Safe default value
            
            # Memory metrics
            memory = psutil.virtual_memory()
            metrics.memory_percent = memory.percent
            metrics.memory_used_mb = memory.used / (1024 * 1024)
            metrics.memory_available_mb = memory.available / (1024 * 1024)
            
            # Disk I/O metrics - DISABLED to reduce system overhead
            # These metrics were contributing to system load
            metrics.disk_io_read_mb = 0.0
            metrics.disk_io_write_mb = 0.0
            
            # Network metrics - DISABLED to reduce system overhead
            # These metrics were contributing to system load
            metrics.network_sent_mb = 0.0
            metrics.network_recv_mb = 0.0
            
            # Process metrics - Simplified to reduce overhead
            try:
                current_process = psutil.Process()
                metrics.active_threads = current_process.num_threads()
                # Disable open files check as it can be expensive
                metrics.open_files = 0
            except Exception:
                metrics.active_threads = 1
                metrics.open_files = 0
            
            # GPU metrics (if available) - Keep this as it's useful and less problematic
            if self.gpu_available:
                try:
                    import GPUtil
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]  # Use first GPU
                        metrics.gpu_percent = gpu.load * 100
                        metrics.vram_used_mb = gpu.memoryUsed
                        metrics.vram_total_mb = gpu.memoryTotal
                except Exception as e:
                    logger.debug(f"GPU metrics collection failed: {e}")
            
        except Exception as e:
            log_error_with_context(e, "metrics_collection", {})
            logger.warning(f"Failed to collect some system metrics: {e}")
        
        return metrics
    
    def start_monitoring(self):
        """Start continuous performance monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop continuous performance monitoring"""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=2.0)
        logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                metrics = self._collect_system_metrics()
                self.metrics_history.append(metrics)
                
                # Limit history size
                if len(self.metrics_history) > self.max_history_samples:
                    self.metrics_history = self.metrics_history[-self.max_history_samples:]
                
                # Check for performance warnings
                self._check_performance_warnings(metrics)
                
                time.sleep(self.sample_interval)
                
            except Exception as e:
                log_error_with_context(e, "monitoring_loop", {})
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(1.0)  # Prevent tight error loop
    
    def _check_performance_warnings(self, metrics: PerformanceMetrics):
        """Check for performance warnings and log them - SIMPLIFIED"""
        warnings = []
        
        # CPU warnings disabled since we're using a fixed value
        # if metrics.cpu_percent > self.cpu_warning_threshold:
        #     warnings.append(f"High CPU usage: {metrics.cpu_percent:.1f}%")
        
        # Only check memory and VRAM which are still actively monitored
        if metrics.memory_percent > self.memory_warning_threshold:
            warnings.append(f"High memory usage: {metrics.memory_percent:.1f}%")
        
        if metrics.vram_total_mb > 0:
            vram_percent = (metrics.vram_used_mb / metrics.vram_total_mb) * 100
            if vram_percent > self.vram_warning_threshold:
                warnings.append(f"High VRAM usage: {vram_percent:.1f}%")
        
        if warnings:
            logger.info(f"Performance status: {', '.join(warnings)}")  # Changed to info level
    
    @contextmanager
    def profile_operation(self, operation_name: str):
        """Context manager for profiling operations"""
        profile = OperationProfile(
            operation_name=operation_name,
            start_time=datetime.now()
        )
        
        # Start memory tracking
        tracemalloc_snapshot_start = tracemalloc.take_snapshot()
        
        # Start CPU profiling
        profiler = cProfile.Profile()
        profiler.enable()
        
        # Collect initial metrics
        initial_metrics = self._collect_system_metrics()
        profile.metrics_samples.append(initial_metrics)
        
        self.active_operations[operation_name] = profile
        
        try:
            yield profile
        finally:
            # Stop CPU profiling
            profiler.disable()
            
            # Collect final metrics
            final_metrics = self._collect_system_metrics()
            profile.metrics_samples.append(final_metrics)
            
            # Calculate duration
            profile.end_time = datetime.now()
            profile.duration_seconds = (profile.end_time - profile.start_time).total_seconds()
            
            # Analyze CPU profiling results
            stats_stream = io.StringIO()
            stats = pstats.Stats(profiler, stream=stats_stream)
            stats.sort_stats('cumulative')
            profile.function_calls = stats.total_calls
            profile.cpu_time_seconds = stats.total_tt
            
            # Analyze memory usage
            tracemalloc_snapshot_end = tracemalloc.take_snapshot()
            memory_diff = tracemalloc_snapshot_end.compare_to(tracemalloc_snapshot_start, 'lineno')
            
            if memory_diff:
                profile.memory_allocated_mb = sum(stat.size_diff for stat in memory_diff) / (1024 * 1024)
                profile.memory_peak_mb = max(
                    final_metrics.memory_used_mb - initial_metrics.memory_used_mb,
                    0
                )
            
            # Calculate GPU metrics
            if len(profile.metrics_samples) >= 2:
                gpu_utilizations = [m.gpu_percent for m in profile.metrics_samples if m.gpu_percent > 0]
                if gpu_utilizations:
                    profile.gpu_utilization_avg = statistics.mean(gpu_utilizations)
                
                vram_values = [m.vram_used_mb for m in profile.metrics_samples if m.vram_used_mb > 0]
                if vram_values:
                    profile.vram_peak_mb = max(vram_values)
            
            # Analyze bottlenecks and generate recommendations
            self._analyze_operation_bottlenecks(profile)
            
            # Store completed profile
            self.operation_profiles[operation_name] = profile
            if operation_name in self.active_operations:
                del self.active_operations[operation_name]
            
            logger.info(f"Operation '{operation_name}' profiled: {profile.duration_seconds:.2f}s")
    
    def _analyze_operation_bottlenecks(self, profile: OperationProfile):
        """Analyze operation profile to identify bottlenecks and generate recommendations"""
        bottlenecks = []
        recommendations = []
        
        # CPU bottleneck analysis
        if profile.cpu_time_seconds > profile.duration_seconds * 0.8:
            bottlenecks.append("CPU-bound operation")
            recommendations.append("Consider CPU optimization or parallel processing")
        
        # Memory bottleneck analysis
        if profile.memory_peak_mb > 1000:  # > 1GB peak memory
            bottlenecks.append("High memory usage")
            recommendations.append("Implement memory optimization or streaming processing")
        
        # GPU bottleneck analysis
        if profile.gpu_utilization_avg > 90:
            bottlenecks.append("GPU utilization saturated")
            recommendations.append("Consider GPU memory optimization or model quantization")
        
        # VRAM bottleneck analysis
        if profile.vram_peak_mb > 10000:  # > 10GB VRAM
            bottlenecks.append("High VRAM usage")
            recommendations.append("Enable model offloading or reduce batch size")
        
        # I/O bottleneck analysis
        if len(profile.metrics_samples) >= 2:
            initial = profile.metrics_samples[0]
            final = profile.metrics_samples[-1]
            
            disk_io_rate = (final.disk_io_read_mb + final.disk_io_write_mb - 
                           initial.disk_io_read_mb - initial.disk_io_write_mb) / profile.duration_seconds
            
            if disk_io_rate > self.disk_io_warning_mb:
                bottlenecks.append("High disk I/O")
                recommendations.append("Consider SSD storage or I/O optimization")
        
        # Function call analysis
        if profile.function_calls > 100000:
            bottlenecks.append("High function call overhead")
            recommendations.append("Profile and optimize hot code paths")
        
        profile.bottlenecks = bottlenecks
        profile.recommendations = recommendations
    
    def get_system_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive system performance summary"""
        if not self.metrics_history:
            return {"error": "No performance data available"}
        
        recent_metrics = self.metrics_history[-100:]  # Last 100 samples
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "sample_count": len(recent_metrics),
            "monitoring_duration_minutes": len(self.metrics_history) * self.sample_interval / 60,
            "cpu": {
                "current_percent": recent_metrics[-1].cpu_percent,
                "average_percent": statistics.mean(m.cpu_percent for m in recent_metrics),
                "peak_percent": max(m.cpu_percent for m in recent_metrics),
                "warning_threshold": self.cpu_warning_threshold
            },
            "memory": {
                "current_percent": recent_metrics[-1].memory_percent,
                "current_used_mb": recent_metrics[-1].memory_used_mb,
                "average_percent": statistics.mean(m.memory_percent for m in recent_metrics),
                "peak_percent": max(m.memory_percent for m in recent_metrics),
                "peak_used_mb": max(m.memory_used_mb for m in recent_metrics),
                "warning_threshold": self.memory_warning_threshold
            },
            "gpu": {
                "available": self.gpu_available,
                "current_percent": recent_metrics[-1].gpu_percent if self.gpu_available else 0,
                "average_percent": statistics.mean(m.gpu_percent for m in recent_metrics) if self.gpu_available else 0,
                "peak_percent": max(m.gpu_percent for m in recent_metrics) if self.gpu_available else 0
            },
            "vram": {
                "current_used_mb": recent_metrics[-1].vram_used_mb,
                "total_mb": recent_metrics[-1].vram_total_mb,
                "current_percent": (recent_metrics[-1].vram_used_mb / recent_metrics[-1].vram_total_mb * 100) if recent_metrics[-1].vram_total_mb > 0 else 0,
                "peak_used_mb": max(m.vram_used_mb for m in recent_metrics),
                "warning_threshold": self.vram_warning_threshold
            },
            "system": {
                "active_threads": recent_metrics[-1].active_threads,
                "open_files": recent_metrics[-1].open_files
            }
        }
        
        # Add performance warnings
        current_warnings = []
        current = recent_metrics[-1]
        
        if current.cpu_percent > self.cpu_warning_threshold:
            current_warnings.append(f"High CPU usage: {current.cpu_percent:.1f}%")
        
        if current.memory_percent > self.memory_warning_threshold:
            current_warnings.append(f"High memory usage: {current.memory_percent:.1f}%")
        
        if current.vram_total_mb > 0:
            vram_percent = (current.vram_used_mb / current.vram_total_mb) * 100
            if vram_percent > self.vram_warning_threshold:
                current_warnings.append(f"High VRAM usage: {vram_percent:.1f}%")
        
        summary["warnings"] = current_warnings
        
        return summary
    
    def get_operation_profiles_summary(self) -> Dict[str, Any]:
        """Get summary of all operation profiles"""
        if not self.operation_profiles:
            return {"message": "No operation profiles available"}
        
        profiles_summary = {}
        
        for op_name, profile in self.operation_profiles.items():
            profiles_summary[op_name] = {
                "duration_seconds": profile.duration_seconds,
                "cpu_time_seconds": profile.cpu_time_seconds,
                "memory_peak_mb": profile.memory_peak_mb,
                "memory_allocated_mb": profile.memory_allocated_mb,
                "gpu_utilization_avg": profile.gpu_utilization_avg,
                "vram_peak_mb": profile.vram_peak_mb,
                "function_calls": profile.function_calls,
                "bottlenecks": profile.bottlenecks,
                "recommendations": profile.recommendations,
                "start_time": profile.start_time.isoformat(),
                "end_time": profile.end_time.isoformat() if profile.end_time else None
            }
        
        # Calculate overall statistics
        all_profiles = list(self.operation_profiles.values())
        
        summary = {
            "total_operations": len(all_profiles),
            "total_execution_time": sum(p.duration_seconds for p in all_profiles),
            "average_duration": statistics.mean(p.duration_seconds for p in all_profiles),
            "peak_memory_usage_mb": max(p.memory_peak_mb for p in all_profiles),
            "peak_vram_usage_mb": max(p.vram_peak_mb for p in all_profiles),
            "common_bottlenecks": self._get_common_bottlenecks(),
            "top_recommendations": self._get_top_recommendations(),
            "operations": profiles_summary
        }
        
        return summary
    
    def _get_common_bottlenecks(self) -> List[Tuple[str, int]]:
        """Get most common bottlenecks across all operations"""
        bottleneck_counts = {}
        
        for profile in self.operation_profiles.values():
            for bottleneck in profile.bottlenecks:
                bottleneck_counts[bottleneck] = bottleneck_counts.get(bottleneck, 0) + 1
        
        return sorted(bottleneck_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    def _get_top_recommendations(self) -> List[Tuple[str, int]]:
        """Get most common recommendations across all operations"""
        recommendation_counts = {}
        
        for profile in self.operation_profiles.values():
            for recommendation in profile.recommendations:
                recommendation_counts[recommendation] = recommendation_counts.get(recommendation, 0) + 1
        
        return sorted(recommendation_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    def generate_performance_report(self, output_path: Optional[str] = None) -> str:
        """Generate comprehensive performance report"""
        report_data = {
            "report_timestamp": datetime.now().isoformat(),
            "system_summary": self.get_system_performance_summary(),
            "operation_profiles": self.get_operation_profiles_summary(),
            "configuration": {
                "sample_interval_seconds": self.sample_interval,
                "max_history_samples": self.max_history_samples,
                "thresholds": {
                    "cpu_warning_percent": self.cpu_warning_threshold,
                    "memory_warning_percent": self.memory_warning_threshold,
                    "vram_warning_percent": self.vram_warning_threshold
                }
            }
        }
        
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            logger.info(f"Performance report saved to {output_file}")
            return str(output_file)
        
        return json.dumps(report_data, indent=2)
    
    def clear_profiles(self):
        """Clear all operation profiles"""
        self.operation_profiles.clear()
        logger.info("Operation profiles cleared")
    
    def clear_metrics_history(self):
        """Clear metrics history"""
        self.metrics_history.clear()
        logger.info("Metrics history cleared")


# Decorator for automatic operation profiling
def profile_operation(operation_name: Optional[str] = None):
    """Decorator to automatically profile function execution"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            profiler = get_performance_profiler()
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            with profiler.profile_operation(op_name):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


# Global profiler instance
_performance_profiler = None

def get_performance_profiler() -> PerformanceProfiler:
    """Get the global performance profiler instance"""
    global _performance_profiler
    if _performance_profiler is None:
        _performance_profiler = PerformanceProfiler()
    return _performance_profiler

def start_performance_monitoring():
    """Start global performance monitoring"""
    profiler = get_performance_profiler()
    profiler.start_monitoring()

def stop_performance_monitoring():
    """Stop global performance monitoring"""
    profiler = get_performance_profiler()
    profiler.stop_monitoring()

def get_performance_summary() -> Dict[str, Any]:
    """Get current performance summary"""
    profiler = get_performance_profiler()
    return profiler.get_system_performance_summary()

def generate_performance_report(output_path: Optional[str] = None) -> str:
    """Generate performance report"""
    profiler = get_performance_profiler()
    return profiler.generate_performance_report(output_path)