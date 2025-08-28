"""
Performance monitoring system for real AI model integration.
Tracks generation performance, resource usage, and provides optimization recommendations.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import psutil
import threading
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics for a generation task."""
    task_id: str
    model_type: str
    resolution: str
    steps: int
    
    # Timing metrics
    start_time: float
    end_time: Optional[float] = None
    generation_time_seconds: float = 0.0
    model_load_time_seconds: float = 0.0
    
    # Resource usage
    peak_vram_usage_mb: float = 0.0
    average_vram_usage_mb: float = 0.0
    peak_ram_usage_mb: float = 0.0
    average_cpu_usage_percent: float = 0.0
    
    # Optimization info
    optimizations_applied: List[str] = None
    quantization_used: Optional[str] = None
    offload_used: bool = False
    
    # Quality metrics
    success: bool = False
    error_category: Optional[str] = None
    
    def __post_init__(self):
        if self.optimizations_applied is None:
            self.optimizations_applied = []

@dataclass
class SystemPerformanceSnapshot:
    """System performance snapshot at a point in time."""
    timestamp: float
    cpu_usage_percent: float
    ram_usage_mb: float
    ram_available_mb: float
    disk_usage_percent: float
    gpu_usage_percent: float = 0.0
    vram_usage_mb: float = 0.0
    vram_available_mb: float = 0.0
    temperature_celsius: float = 0.0

@dataclass
class PerformanceAnalysis:
    """Analysis of performance trends and recommendations."""
    average_generation_time: float
    success_rate: float
    resource_efficiency: float
    bottleneck_analysis: Dict[str, Any]
    optimization_recommendations: List[str]
    performance_trends: Dict[str, List[float]]

class PerformanceMonitor:
    """Monitors and analyzes system and generation performance."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.metrics_history: List[PerformanceMetrics] = []
        self.system_snapshots: deque = deque(maxlen=1000)  # Keep last 1000 snapshots
        self.active_tasks: Dict[str, PerformanceMetrics] = {}
        
        # Monitoring settings
        self.monitoring_interval = self.config.get("monitoring_interval_seconds", 5.0)
        self.max_history_size = self.config.get("max_history_size", 1000)
        self.enable_gpu_monitoring = self.config.get("enable_gpu_monitoring", True)
        
        # Performance thresholds
        self.performance_thresholds = {
            "max_generation_time_720p": 300,  # 5 minutes for 720p
            "max_generation_time_1080p": 900,  # 15 minutes for 1080p
            "max_vram_usage_percent": 90,
            "max_ram_usage_percent": 85,
            "min_success_rate": 0.95
        }
        
        # Monitoring thread
        self._monitoring_active = False
        self._monitoring_thread = None
        
        # GPU monitoring
        self._gpu_available = False
        self._init_gpu_monitoring()
    
    def _init_gpu_monitoring(self):
        """Initialize GPU monitoring if available."""
        if not self.enable_gpu_monitoring:
            return
            
        try:
            import torch
            self._gpu_available = torch.cuda.is_available()
            if self._gpu_available:
                logger.info(f"GPU monitoring enabled. Found {torch.cuda.device_count()} GPU(s)")
            else:
                logger.info("No CUDA GPUs available for monitoring (will retry during monitoring)")
        except ImportError:
            logger.warning("PyTorch not available, GPU monitoring disabled")
    
    def start_monitoring(self):
        """Start continuous system monitoring."""
        if self._monitoring_active:
            return
            
        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop continuous system monitoring."""
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5.0)
        logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        gpu_check_counter = 0
        while self._monitoring_active:
            try:
                # Re-check GPU availability every 10 monitoring cycles (if not available)
                if not self._gpu_available and gpu_check_counter % 10 == 0:
                    self._recheck_gpu_availability()
                
                snapshot = self._capture_system_snapshot()
                self.system_snapshots.append(snapshot)
                gpu_check_counter += 1
                time.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def _recheck_gpu_availability(self):
        """Re-check GPU availability during monitoring"""
        try:
            import torch
            was_available = self._gpu_available
            self._gpu_available = torch.cuda.is_available()
            
            if not was_available and self._gpu_available:
                logger.info(f"GPU now available! Found {torch.cuda.device_count()} GPU(s)")
            elif was_available and not self._gpu_available:
                logger.warning("GPU no longer available")
        except ImportError:
            pass  # PyTorch not available
    
    def _capture_system_snapshot(self) -> SystemPerformanceSnapshot:
        """Capture current system performance snapshot."""
        # CPU and RAM
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('.')
        
        snapshot = SystemPerformanceSnapshot(
            timestamp=time.time(),
            cpu_usage_percent=cpu_percent,
            ram_usage_mb=memory.used / (1024**2),
            ram_available_mb=memory.available / (1024**2),
            disk_usage_percent=disk.percent
        )
        
        # GPU metrics if available
        if self._gpu_available:
            try:
                import torch
                if torch.cuda.is_available():
                    device = torch.cuda.current_device()
                    
                    # VRAM usage
                    vram_used = torch.cuda.memory_allocated(device) / (1024**2)
                    vram_total = torch.cuda.get_device_properties(device).total_memory / (1024**2)
                    
                    snapshot.vram_usage_mb = vram_used
                    snapshot.vram_available_mb = vram_total - vram_used
                    snapshot.gpu_usage_percent = (vram_used / vram_total) * 100
                    
                    # Temperature (if available)
                    try:
                        import nvidia_ml_py3 as nvml
                        nvml.nvmlInit()
                        handle = nvml.nvmlDeviceGetHandleByIndex(0)
                        temp = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                        snapshot.temperature_celsius = temp
                    except:
                        pass  # Temperature monitoring is optional
                        
            except Exception as e:
                logger.debug(f"GPU monitoring error: {e}")
        
        return snapshot
    
    def start_task_monitoring(self, task_id: str, model_type: str, 
                            resolution: str, steps: int) -> PerformanceMetrics:
        """Start monitoring a generation task."""
        metrics = PerformanceMetrics(
            task_id=task_id,
            model_type=model_type,
            resolution=resolution,
            steps=steps,
            start_time=time.time()
        )
        
        self.active_tasks[task_id] = metrics
        logger.debug(f"Started monitoring task {task_id}")
        return metrics
    
    def update_task_metrics(self, task_id: str, **kwargs):
        """Update metrics for an active task."""
        if task_id in self.active_tasks:
            metrics = self.active_tasks[task_id]
            for key, value in kwargs.items():
                if hasattr(metrics, key):
                    setattr(metrics, key, value)
    
    def complete_task_monitoring(self, task_id: str, success: bool = True, 
                                error_category: Optional[str] = None) -> Optional[PerformanceMetrics]:
        """Complete monitoring for a task and calculate final metrics."""
        if task_id not in self.active_tasks:
            logger.warning(f"Task {task_id} not found in active monitoring")
            return None
        
        metrics = self.active_tasks.pop(task_id)
        metrics.end_time = time.time()
        metrics.generation_time_seconds = metrics.end_time - metrics.start_time
        metrics.success = success
        metrics.error_category = error_category
        
        # Calculate resource usage from recent snapshots
        self._calculate_resource_usage(metrics)
        
        # Add to history
        self.metrics_history.append(metrics)
        
        # Trim history if needed
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history = self.metrics_history[-self.max_history_size:]
        
        logger.info(f"Completed monitoring task {task_id}: "
                   f"{metrics.generation_time_seconds:.1f}s, success={success}")
        
        return metrics
    
    def _calculate_resource_usage(self, metrics: PerformanceMetrics):
        """Calculate resource usage for a task from system snapshots."""
        if not self.system_snapshots:
            return
        
        # Find snapshots during task execution
        task_snapshots = [
            s for s in self.system_snapshots
            if metrics.start_time <= s.timestamp <= (metrics.end_time or time.time())
        ]
        
        if not task_snapshots:
            return
        
        # Calculate averages and peaks
        vram_values = [s.vram_usage_mb for s in task_snapshots if s.vram_usage_mb > 0]
        ram_values = [s.ram_usage_mb for s in task_snapshots]
        cpu_values = [s.cpu_usage_percent for s in task_snapshots]
        
        if vram_values:
            metrics.peak_vram_usage_mb = max(vram_values)
            metrics.average_vram_usage_mb = sum(vram_values) / len(vram_values)
        
        if ram_values:
            metrics.peak_ram_usage_mb = max(ram_values)
        
        if cpu_values:
            metrics.average_cpu_usage_percent = sum(cpu_values) / len(cpu_values)
    
    def get_performance_analysis(self, time_window_hours: int = 24) -> PerformanceAnalysis:
        """Analyze performance over a time window."""
        cutoff_time = time.time() - (time_window_hours * 3600)
        recent_metrics = [
            m for m in self.metrics_history
            if m.start_time >= cutoff_time
        ]
        
        if not recent_metrics:
            return PerformanceAnalysis(
                average_generation_time=0.0,
                success_rate=0.0,
                resource_efficiency=0.0,
                bottleneck_analysis={},
                optimization_recommendations=[],
                performance_trends={}
            )
        
        # Calculate basic metrics
        successful_tasks = [m for m in recent_metrics if m.success]
        success_rate = len(successful_tasks) / len(recent_metrics)
        
        if successful_tasks:
            avg_generation_time = sum(m.generation_time_seconds for m in successful_tasks) / len(successful_tasks)
        else:
            avg_generation_time = 0.0
        
        # Analyze bottlenecks
        bottleneck_analysis = self._analyze_bottlenecks(recent_metrics)
        
        # Generate recommendations
        recommendations = self._generate_optimization_recommendations(recent_metrics, bottleneck_analysis)
        
        # Calculate resource efficiency
        resource_efficiency = self._calculate_resource_efficiency(recent_metrics)
        
        # Performance trends
        trends = self._calculate_performance_trends(recent_metrics)
        
        return PerformanceAnalysis(
            average_generation_time=avg_generation_time,
            success_rate=success_rate,
            resource_efficiency=resource_efficiency,
            bottleneck_analysis=bottleneck_analysis,
            optimization_recommendations=recommendations,
            performance_trends=trends
        )
    
    def _analyze_bottlenecks(self, metrics: List[PerformanceMetrics]) -> Dict[str, Any]:
        """Analyze system bottlenecks from metrics."""
        analysis = {
            "vram_bottleneck": False,
            "ram_bottleneck": False,
            "cpu_bottleneck": False,
            "model_load_bottleneck": False,
            "details": {}
        }
        
        if not metrics:
            return analysis
        
        # VRAM analysis
        vram_usage = [m.peak_vram_usage_mb for m in metrics if m.peak_vram_usage_mb > 0]
        if vram_usage:
            avg_vram = sum(vram_usage) / len(vram_usage)
            max_vram = max(vram_usage)
            
            # Estimate total VRAM (this is approximate)
            if self._gpu_available:
                try:
                    import torch
                    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**2)
                    vram_usage_percent = (avg_vram / total_vram) * 100
                    
                    if vram_usage_percent > self.performance_thresholds["max_vram_usage_percent"]:
                        analysis["vram_bottleneck"] = True
                        analysis["details"]["vram_usage_percent"] = vram_usage_percent
                except:
                    pass
        
        # RAM analysis
        ram_usage = [m.peak_ram_usage_mb for m in metrics if m.peak_ram_usage_mb > 0]
        if ram_usage:
            avg_ram = sum(ram_usage) / len(ram_usage)
            total_ram = psutil.virtual_memory().total / (1024**2)
            ram_usage_percent = (avg_ram / total_ram) * 100
            
            if ram_usage_percent > self.performance_thresholds["max_ram_usage_percent"]:
                analysis["ram_bottleneck"] = True
                analysis["details"]["ram_usage_percent"] = ram_usage_percent
        
        # CPU analysis
        cpu_usage = [m.average_cpu_usage_percent for m in metrics if m.average_cpu_usage_percent > 0]
        if cpu_usage:
            avg_cpu = sum(cpu_usage) / len(cpu_usage)
            if avg_cpu > 90:  # High CPU usage
                analysis["cpu_bottleneck"] = True
                analysis["details"]["cpu_usage_percent"] = avg_cpu
        
        # Model loading analysis
        load_times = [m.model_load_time_seconds for m in metrics if m.model_load_time_seconds > 0]
        if load_times:
            avg_load_time = sum(load_times) / len(load_times)
            if avg_load_time > 30:  # Slow model loading
                analysis["model_load_bottleneck"] = True
                analysis["details"]["avg_model_load_time"] = avg_load_time
        
        return analysis
    
    def _generate_optimization_recommendations(self, metrics: List[PerformanceMetrics], 
                                            bottlenecks: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations based on analysis."""
        recommendations = []
        
        # VRAM optimization
        if bottlenecks["vram_bottleneck"]:
            recommendations.extend([
                "Enable model quantization to reduce VRAM usage",
                "Use model offloading to CPU when not actively generating",
                "Reduce batch size or resolution for lower VRAM requirements",
                "Consider upgrading GPU memory if consistently hitting limits"
            ])
        
        # RAM optimization
        if bottlenecks["ram_bottleneck"]:
            recommendations.extend([
                "Close unnecessary applications to free system RAM",
                "Enable model offloading to reduce RAM usage",
                "Consider increasing system RAM if consistently high usage"
            ])
        
        # CPU optimization
        if bottlenecks["cpu_bottleneck"]:
            recommendations.extend([
                "Reduce CPU thread count for model processing",
                "Close CPU-intensive background applications",
                "Consider upgrading CPU for better performance"
            ])
        
        # Model loading optimization
        if bottlenecks["model_load_bottleneck"]:
            recommendations.extend([
                "Use SSD storage for faster model loading",
                "Keep frequently used models loaded in memory",
                "Enable model caching to avoid repeated loading"
            ])
        
        # Performance-based recommendations
        successful_metrics = [m for m in metrics if m.success]
        if successful_metrics:
            avg_time = sum(m.generation_time_seconds for m in successful_metrics) / len(successful_metrics)
            
            # Check against thresholds
            resolution_times = defaultdict(list)
            for m in successful_metrics:
                resolution_times[m.resolution].append(m.generation_time_seconds)
            
            for resolution, times in resolution_times.items():
                avg_res_time = sum(times) / len(times)
                threshold_key = f"max_generation_time_{resolution.lower()}"
                
                if threshold_key in self.performance_thresholds:
                    threshold = self.performance_thresholds[threshold_key]
                    if avg_res_time > threshold:
                        recommendations.append(
                            f"Generation time for {resolution} ({avg_res_time:.1f}s) "
                            f"exceeds target ({threshold}s). Consider optimization."
                        )
        
        # Success rate recommendations
        success_rate = len(successful_metrics) / len(metrics) if metrics else 0
        if success_rate < self.performance_thresholds["min_success_rate"]:
            recommendations.extend([
                "Investigate and fix causes of generation failures",
                "Enable fallback mechanisms for better reliability",
                "Review error logs for common failure patterns"
            ])
        
        return recommendations
    
    def _calculate_resource_efficiency(self, metrics: List[PerformanceMetrics]) -> float:
        """Calculate overall resource efficiency score (0-1)."""
        if not metrics:
            return 0.0
        
        successful_metrics = [m for m in metrics if m.success]
        if not successful_metrics:
            return 0.0
        
        # Factors for efficiency calculation
        factors = []
        
        # Success rate factor
        success_rate = len(successful_metrics) / len(metrics)
        factors.append(success_rate)
        
        # Time efficiency factor (faster is better)
        generation_times = [m.generation_time_seconds for m in successful_metrics]
        if generation_times:
            avg_time = sum(generation_times) / len(generation_times)
            # Normalize against expected time (assume 300s baseline)
            time_efficiency = min(1.0, 300 / max(avg_time, 1))
            factors.append(time_efficiency)
        
        # Resource usage efficiency (lower usage is better)
        vram_usage = [m.peak_vram_usage_mb for m in successful_metrics if m.peak_vram_usage_mb > 0]
        if vram_usage and self._gpu_available:
            try:
                import torch
                total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**2)
                avg_vram_usage = sum(vram_usage) / len(vram_usage)
                vram_efficiency = 1.0 - min(1.0, avg_vram_usage / total_vram)
                factors.append(vram_efficiency)
            except:
                pass
        
        # Calculate overall efficiency as geometric mean
        if factors:
            efficiency = 1.0
            for factor in factors:
                efficiency *= factor
            return efficiency ** (1.0 / len(factors))
        
        return 0.0
    
    def _calculate_performance_trends(self, metrics: List[PerformanceMetrics]) -> Dict[str, List[float]]:
        """Calculate performance trends over time."""
        trends = {
            "generation_times": [],
            "success_rates": [],
            "vram_usage": [],
            "cpu_usage": []
        }
        
        if not metrics:
            return trends
        
        # Sort by start time
        sorted_metrics = sorted(metrics, key=lambda m: m.start_time)
        
        # Calculate trends in time windows
        window_size = max(1, len(sorted_metrics) // 10)  # 10 data points
        
        for i in range(0, len(sorted_metrics), window_size):
            window_metrics = sorted_metrics[i:i + window_size]
            
            # Generation time trend
            successful_in_window = [m for m in window_metrics if m.success]
            if successful_in_window:
                avg_time = sum(m.generation_time_seconds for m in successful_in_window) / len(successful_in_window)
                trends["generation_times"].append(avg_time)
            
            # Success rate trend
            success_rate = len(successful_in_window) / len(window_metrics)
            trends["success_rates"].append(success_rate)
            
            # VRAM usage trend
            vram_values = [m.peak_vram_usage_mb for m in window_metrics if m.peak_vram_usage_mb > 0]
            if vram_values:
                avg_vram = sum(vram_values) / len(vram_values)
                trends["vram_usage"].append(avg_vram)
            
            # CPU usage trend
            cpu_values = [m.average_cpu_usage_percent for m in window_metrics if m.average_cpu_usage_percent > 0]
            if cpu_values:
                avg_cpu = sum(cpu_values) / len(cpu_values)
                trends["cpu_usage"].append(avg_cpu)
        
        return trends
    
    def get_current_system_status(self) -> Dict[str, Any]:
        """Get current system performance status."""
        if not self.system_snapshots:
            return {"error": "No system data available"}
        
        latest = self.system_snapshots[-1]
        
        return {
            "timestamp": latest.timestamp,
            "cpu_usage_percent": latest.cpu_usage_percent,
            "ram_usage_mb": latest.ram_usage_mb,
            "ram_available_mb": latest.ram_available_mb,
            "disk_usage_percent": latest.disk_usage_percent,
            "gpu_usage_percent": latest.gpu_usage_percent,
            "vram_usage_mb": latest.vram_usage_mb,
            "vram_available_mb": latest.vram_available_mb,
            "temperature_celsius": latest.temperature_celsius,
            "active_tasks": len(self.active_tasks)
        }
    
    def export_metrics(self, filepath: str, time_window_hours: int = 24):
        """Export performance metrics to JSON file."""
        cutoff_time = time.time() - (time_window_hours * 3600)
        recent_metrics = [
            asdict(m) for m in self.metrics_history
            if m.start_time >= cutoff_time
        ]
        
        analysis = self.get_performance_analysis(time_window_hours)
        
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "time_window_hours": time_window_hours,
            "metrics_count": len(recent_metrics),
            "performance_analysis": asdict(analysis),
            "metrics": recent_metrics
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported {len(recent_metrics)} metrics to {filepath}")

# Global performance monitor instance
_performance_monitor = None

def get_performance_monitor(config: Dict[str, Any] = None) -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor(config)
    return _performance_monitor