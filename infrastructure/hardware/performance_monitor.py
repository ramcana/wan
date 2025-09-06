"""
Performance Monitoring and Optimization System for WAN Model Compatibility

This module provides performance monitoring for generation speed, memory usage,
optimization effectiveness measurement, and regression detection.

Requirements addressed: 5.1, 5.2, 5.3, 8.1, 8.3
"""

import time
import logging
import threading
import json
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque
import statistics

import torch
import psutil

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for a single operation"""
    operation_id: str
    operation_type: str  # "generation", "model_load", "optimization"
    start_time: float
    end_time: float
    duration: float
    memory_peak_mb: int
    memory_allocated_mb: int
    gpu_utilization: float
    cpu_utilization: float
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "operation_id": self.operation_id,
            "operation_type": self.operation_type,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "memory_peak_mb": self.memory_peak_mb,
            "memory_allocated_mb": self.memory_allocated_mb,
            "gpu_utilization": self.gpu_utilization,
            "cpu_utilization": self.cpu_utilization,
            "success": self.success,
            "error_message": self.error_message,
            "metadata": self.metadata
        }


@dataclass
class OptimizationEffectiveness:
    """Metrics for optimization effectiveness"""
    optimization_name: str
    baseline_duration: float
    optimized_duration: float
    performance_improvement: float  # Percentage improvement
    memory_reduction_mb: int
    memory_reduction_percent: float
    stability_score: float  # 0.0 to 1.0
    recommendation_score: float  # 0.0 to 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "optimization_name": self.optimization_name,
            "baseline_duration": self.baseline_duration,
            "optimized_duration": self.optimized_duration,
            "performance_improvement": self.performance_improvement,
            "memory_reduction_mb": self.memory_reduction_mb,
            "memory_reduction_percent": self.memory_reduction_percent,
            "stability_score": self.stability_score,
            "recommendation_score": self.recommendation_score
        }


@dataclass
class RegressionAlert:
    """Alert for performance regression"""
    alert_id: str
    metric_name: str
    current_value: float
    baseline_value: float
    regression_percent: float
    severity: str  # "low", "medium", "high", "critical"
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "alert_id": self.alert_id,
            "metric_name": self.metric_name,
            "current_value": self.current_value,
            "baseline_value": self.baseline_value,
            "regression_percent": self.regression_percent,
            "severity": self.severity,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context
        }


class PerformanceMonitor:
    """
    Performance monitoring system for WAN model operations.
    
    Tracks generation speed, memory usage, optimization effectiveness,
    and detects performance regressions.
    """
    
    def __init__(self, 
                 metrics_file: str = "performance_metrics.json",
                 max_metrics_history: int = 1000,
                 regression_threshold: float = 0.15):  # 15% regression threshold
        """
        Initialize performance monitor.
        
        Args:
            metrics_file: File to store performance metrics
            max_metrics_history: Maximum number of metrics to keep in memory
            regression_threshold: Threshold for regression detection (0.0 to 1.0)
        """
        self.metrics_file = Path(metrics_file)
        self.max_metrics_history = max_metrics_history
        self.regression_threshold = regression_threshold
        
        # In-memory metrics storage
        self.metrics_history: deque = deque(maxlen=max_metrics_history)
        self.optimization_effectiveness: Dict[str, OptimizationEffectiveness] = {}
        self.regression_alerts: List[RegressionAlert] = []
        
        # Performance baselines
        self.baselines: Dict[str, Dict[str, float]] = {}
        
        # Monitoring state
        self.active_operations: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
        
        # Load existing metrics
        self._load_metrics()
        
        logger.info(f"Performance monitor initialized with {len(self.metrics_history)} historical metrics")
    
    def start_operation(self, 
                       operation_id: str,
                       operation_type: str,
                       metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Start monitoring an operation.
        
        Args:
            operation_id: Unique identifier for the operation
            operation_type: Type of operation (generation, model_load, etc.)
            metadata: Additional metadata for the operation
        """
        with self.lock:
            self.active_operations[operation_id] = {
                "operation_type": operation_type,
                "start_time": time.time(),
                "start_memory": self._get_memory_usage(),
                "start_gpu_memory": self._get_gpu_memory_usage(),
                "metadata": metadata or {}
            }
        
        logger.debug(f"Started monitoring operation: {operation_id} ({operation_type})")
    
    def end_operation(self, 
                     operation_id: str,
                     success: bool = True,
                     error_message: Optional[str] = None) -> PerformanceMetrics:
        """
        End monitoring an operation and record metrics.
        
        Args:
            operation_id: Unique identifier for the operation
            success: Whether the operation succeeded
            error_message: Error message if operation failed
            
        Returns:
            PerformanceMetrics object with recorded metrics
        """
        with self.lock:
            if operation_id not in self.active_operations:
                logger.warning(f"Operation {operation_id} not found in active operations")
                return None
            
            operation_info = self.active_operations.pop(operation_id)
            
            end_time = time.time()
            duration = end_time - operation_info["start_time"]
            
            # Calculate memory usage
            current_memory = self._get_memory_usage()
            current_gpu_memory = self._get_gpu_memory_usage()
            
            memory_peak = max(current_memory, operation_info["start_memory"])
            memory_allocated = current_gpu_memory - operation_info["start_gpu_memory"]
            
            # Get system utilization
            gpu_utilization = self._get_gpu_utilization()
            cpu_utilization = psutil.cpu_percent()
            
            # Create metrics object
            metrics = PerformanceMetrics(
                operation_id=operation_id,
                operation_type=operation_info["operation_type"],
                start_time=operation_info["start_time"],
                end_time=end_time,
                duration=duration,
                memory_peak_mb=memory_peak,
                memory_allocated_mb=memory_allocated,
                gpu_utilization=gpu_utilization,
                cpu_utilization=cpu_utilization,
                success=success,
                error_message=error_message,
                metadata=operation_info["metadata"]
            )
            
            # Store metrics
            self.metrics_history.append(metrics)
            
            # Check for regressions
            self._check_for_regressions(metrics)
            
            # Save metrics periodically
            if len(self.metrics_history) % 10 == 0:
                self._save_metrics()
            
            logger.info(f"Recorded metrics for operation {operation_id}: {duration:.2f}s, {memory_peak}MB peak")
            
            return metrics
    
    def measure_optimization_effectiveness(self,
                                         optimization_name: str,
                                         baseline_metrics: PerformanceMetrics,
                                         optimized_metrics: PerformanceMetrics) -> OptimizationEffectiveness:
        """
        Measure the effectiveness of an optimization.
        
        Args:
            optimization_name: Name of the optimization
            baseline_metrics: Metrics without optimization
            optimized_metrics: Metrics with optimization
            
        Returns:
            OptimizationEffectiveness object
        """
        # Calculate performance improvement
        performance_improvement = 0.0
        if baseline_metrics.duration > 0:
            performance_improvement = ((baseline_metrics.duration - optimized_metrics.duration) / 
                                     baseline_metrics.duration) * 100
        
        # Calculate memory reduction
        memory_reduction_mb = baseline_metrics.memory_peak_mb - optimized_metrics.memory_peak_mb
        memory_reduction_percent = 0.0
        if baseline_metrics.memory_peak_mb > 0:
            memory_reduction_percent = (memory_reduction_mb / baseline_metrics.memory_peak_mb) * 100
        
        # Calculate stability score (based on success rate and error frequency)
        stability_score = 1.0 if optimized_metrics.success else 0.5
        
        # Calculate recommendation score
        recommendation_score = self._calculate_recommendation_score(
            performance_improvement, memory_reduction_percent, stability_score
        )
        
        effectiveness = OptimizationEffectiveness(
            optimization_name=optimization_name,
            baseline_duration=baseline_metrics.duration,
            optimized_duration=optimized_metrics.duration,
            performance_improvement=performance_improvement,
            memory_reduction_mb=memory_reduction_mb,
            memory_reduction_percent=memory_reduction_percent,
            stability_score=stability_score,
            recommendation_score=recommendation_score
        )
        
        # Store effectiveness data
        self.optimization_effectiveness[optimization_name] = effectiveness
        
        logger.info(f"Optimization '{optimization_name}' effectiveness: "
                   f"{performance_improvement:.1f}% speed, {memory_reduction_percent:.1f}% memory")
        
        return effectiveness
    
    def get_performance_summary(self, 
                              operation_type: Optional[str] = None,
                              time_window_hours: Optional[int] = None) -> Dict[str, Any]:
        """
        Get performance summary statistics.
        
        Args:
            operation_type: Filter by operation type
            time_window_hours: Only include metrics from last N hours
            
        Returns:
            Dictionary with performance summary
        """
        # Filter metrics
        filtered_metrics = list(self.metrics_history)
        
        if operation_type:
            filtered_metrics = [m for m in filtered_metrics if m.operation_type == operation_type]
        
        if time_window_hours:
            cutoff_time = time.time() - (time_window_hours * 3600)
            filtered_metrics = [m for m in filtered_metrics if m.end_time >= cutoff_time]
        
        if not filtered_metrics:
            return {"error": "No metrics found for specified criteria"}
        
        # Calculate statistics
        durations = [m.duration for m in filtered_metrics if m.success]
        memory_peaks = [m.memory_peak_mb for m in filtered_metrics if m.success]
        success_rate = sum(1 for m in filtered_metrics if m.success) / len(filtered_metrics)
        
        summary = {
            "total_operations": len(filtered_metrics),
            "success_rate": success_rate,
            "duration_stats": {
                "mean": statistics.mean(durations) if durations else 0,
                "median": statistics.median(durations) if durations else 0,
                "min": min(durations) if durations else 0,
                "max": max(durations) if durations else 0,
                "std_dev": statistics.stdev(durations) if len(durations) > 1 else 0
            },
            "memory_stats": {
                "mean_peak_mb": statistics.mean(memory_peaks) if memory_peaks else 0,
                "median_peak_mb": statistics.median(memory_peaks) if memory_peaks else 0,
                "min_peak_mb": min(memory_peaks) if memory_peaks else 0,
                "max_peak_mb": max(memory_peaks) if memory_peaks else 0
            },
            "optimization_effectiveness": {
                name: eff.to_dict() for name, eff in self.optimization_effectiveness.items()
            },
            "recent_regressions": [
                alert.to_dict() for alert in self.regression_alerts[-10:]
            ]
        }
        
        return summary
    
    def detect_performance_regressions(self) -> List[RegressionAlert]:
        """
        Detect performance regressions by comparing recent metrics to baselines.
        
        Returns:
            List of regression alerts
        """
        new_alerts = []
        
        # Get recent metrics (last 10 operations)
        recent_metrics = list(self.metrics_history)[-10:]
        if len(recent_metrics) < 5:
            return new_alerts
        
        # Group by operation type
        by_operation_type = {}
        for metric in recent_metrics:
            if metric.operation_type not in by_operation_type:
                by_operation_type[metric.operation_type] = []
            by_operation_type[metric.operation_type].append(metric)
        
        # Check each operation type for regressions
        for operation_type, metrics in by_operation_type.items():
            if len(metrics) < 3:
                continue
            
            # Calculate current average performance
            successful_metrics = [m for m in metrics if m.success]
            if not successful_metrics:
                continue
            
            current_avg_duration = statistics.mean([m.duration for m in successful_metrics])
            current_avg_memory = statistics.mean([m.memory_peak_mb for m in successful_metrics])
            
            # Compare to baseline
            baseline = self.baselines.get(operation_type, {})
            
            # Check duration regression
            if "duration" in baseline:
                baseline_duration = baseline["duration"]
                regression_percent = ((current_avg_duration - baseline_duration) / baseline_duration) * 100
                
                if regression_percent > self.regression_threshold * 100:
                    severity = self._calculate_regression_severity(regression_percent)
                    alert = RegressionAlert(
                        alert_id=f"duration_regression_{operation_type}_{int(time.time())}",
                        metric_name="duration",
                        current_value=current_avg_duration,
                        baseline_value=baseline_duration,
                        regression_percent=regression_percent,
                        severity=severity,
                        timestamp=datetime.now(),
                        context={"operation_type": operation_type, "sample_size": len(successful_metrics)}
                    )
                    new_alerts.append(alert)
            
            # Check memory regression
            if "memory_peak" in baseline:
                baseline_memory = baseline["memory_peak"]
                regression_percent = ((current_avg_memory - baseline_memory) / baseline_memory) * 100
                
                if regression_percent > self.regression_threshold * 100:
                    severity = self._calculate_regression_severity(regression_percent)
                    alert = RegressionAlert(
                        alert_id=f"memory_regression_{operation_type}_{int(time.time())}",
                        metric_name="memory_peak",
                        current_value=current_avg_memory,
                        baseline_value=baseline_memory,
                        regression_percent=regression_percent,
                        severity=severity,
                        timestamp=datetime.now(),
                        context={"operation_type": operation_type, "sample_size": len(successful_metrics)}
                    )
                    new_alerts.append(alert)
        
        # Store new alerts
        self.regression_alerts.extend(new_alerts)
        
        # Keep only recent alerts (last 100)
        self.regression_alerts = self.regression_alerts[-100:]
        
        return new_alerts
    
    def update_baselines(self) -> None:
        """Update performance baselines based on recent successful operations."""
        # Group metrics by operation type
        by_operation_type = {}
        for metric in self.metrics_history:
            if not metric.success:
                continue
            if metric.operation_type not in by_operation_type:
                by_operation_type[metric.operation_type] = []
            by_operation_type[metric.operation_type].append(metric)
        
        # Calculate baselines for each operation type
        for operation_type, metrics in by_operation_type.items():
            if len(metrics) < 10:  # Need at least 10 samples
                continue
            
            # Use median for more stable baselines
            durations = [m.duration for m in metrics]
            memory_peaks = [m.memory_peak_mb for m in metrics]
            
            self.baselines[operation_type] = {
                "duration": statistics.median(durations),
                "memory_peak": statistics.median(memory_peaks),
                "sample_size": len(metrics),
                "updated_at": time.time()
            }
        
        logger.info(f"Updated baselines for {len(self.baselines)} operation types")
    
    def _get_memory_usage(self) -> int:
        """Get current memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() // (1024 * 1024)
        return 0
    
    def _get_gpu_memory_usage(self) -> int:
        """Get current GPU memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() // (1024 * 1024)
        return 0
    
    def _get_gpu_utilization(self) -> float:
        """Get current GPU utilization percentage."""
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                return gpus[0].load * 100
        except ImportError:
            pass
        return 0.0
    
    def _check_for_regressions(self, metrics: PerformanceMetrics) -> None:
        """Check if the given metrics indicate a performance regression."""
        # This is called for each operation, so we do lightweight checks
        # Full regression detection is done in detect_performance_regressions()
        
        operation_type = metrics.operation_type
        baseline = self.baselines.get(operation_type, {})
        
        if not baseline or not metrics.success:
            return
        
        # Quick check for significant regressions
        if "duration" in baseline:
            baseline_duration = baseline["duration"]
            if metrics.duration > baseline_duration * (1 + self.regression_threshold * 2):
                logger.warning(f"Potential performance regression detected for {operation_type}: "
                             f"{metrics.duration:.2f}s vs baseline {baseline_duration:.2f}s")
    
    def _calculate_recommendation_score(self,
                                      performance_improvement: float,
                                      memory_reduction_percent: float,
                                      stability_score: float) -> float:
        """Calculate recommendation score for an optimization."""
        # Weight factors
        performance_weight = 0.4
        memory_weight = 0.3
        stability_weight = 0.3
        
        # Normalize scores (0-1)
        performance_score = max(0, min(1, performance_improvement / 50))  # 50% improvement = 1.0
        memory_score = max(0, min(1, memory_reduction_percent / 30))  # 30% reduction = 1.0
        
        recommendation_score = (
            performance_score * performance_weight +
            memory_score * memory_weight +
            stability_score * stability_weight
        )
        
        return recommendation_score
    
    def _calculate_regression_severity(self, regression_percent: float) -> str:
        """Calculate severity level for a regression."""
        if regression_percent >= 50:
            return "critical"
        elif regression_percent >= 30:
            return "high"
        elif regression_percent >= 20:
            return "medium"
        else:
            return "low"
    
    def _load_metrics(self) -> None:
        """Load metrics from file."""
        if not self.metrics_file.exists():
            return
        
        try:
            with open(self.metrics_file, 'r') as f:
                data = json.load(f)
            
            # Load metrics history
            for metric_data in data.get("metrics", []):
                metrics = PerformanceMetrics(**metric_data)
                self.metrics_history.append(metrics)
            
            # Load baselines
            self.baselines = data.get("baselines", {})
            
            # Load optimization effectiveness
            for name, eff_data in data.get("optimization_effectiveness", {}).items():
                self.optimization_effectiveness[name] = OptimizationEffectiveness(**eff_data)
            
            logger.info(f"Loaded {len(self.metrics_history)} metrics from {self.metrics_file}")
            
        except Exception as e:
            logger.error(f"Failed to load metrics: {e}")
    
    def _save_metrics(self) -> None:
        """Save metrics to file."""
        try:
            data = {
                "metrics": [m.to_dict() for m in self.metrics_history],
                "baselines": self.baselines,
                "optimization_effectiveness": {
                    name: eff.to_dict() for name, eff in self.optimization_effectiveness.items()
                },
                "saved_at": datetime.now().isoformat()
            }
            
            with open(self.metrics_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"Saved {len(self.metrics_history)} metrics to {self.metrics_file}")
            
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
    
    def integrate_system_optimizer(self, system_optimizer):
        """Integrate with WAN22 system optimizer for enhanced monitoring"""
        self.system_optimizer = system_optimizer
        logger.info("System optimizer integrated with performance monitor")
    
    def get_enhanced_system_metrics(self) -> Dict[str, Any]:
        """Get enhanced system metrics including optimizer health data"""
        base_metrics = self.get_system_metrics()
        
        if hasattr(self, 'system_optimizer') and self.system_optimizer:
            try:
                # Get health metrics from system optimizer
                health_metrics = self.system_optimizer.monitor_system_health()
                
                # Merge with base metrics
                base_metrics.update({
                    "optimizer_gpu_temp": health_metrics.gpu_temperature,
                    "optimizer_vram_usage_mb": health_metrics.vram_usage_mb,
                    "optimizer_vram_total_mb": health_metrics.vram_total_mb,
                    "optimizer_cpu_usage": health_metrics.cpu_usage_percent,
                    "optimizer_memory_usage_gb": health_metrics.memory_usage_gb,
                    "optimizer_generation_speed": health_metrics.generation_speed,
                    "optimizer_timestamp": health_metrics.timestamp
                })
                
                # Add hardware profile information
                hardware_profile = self.system_optimizer.get_hardware_profile()
                if hardware_profile:
                    base_metrics.update({
                        "hardware_cpu_model": hardware_profile.cpu_model,
                        "hardware_gpu_model": hardware_profile.gpu_model,
                        "hardware_vram_gb": hardware_profile.vram_gb,
                        "hardware_total_memory_gb": hardware_profile.total_memory_gb
                    })
                
            except Exception as e:
                logger.warning(f"Failed to get enhanced metrics from system optimizer: {e}")
        
        return base_metrics
    
    def check_system_health_alerts(self) -> List[Dict[str, Any]]:
        """Check for system health alerts using optimizer data"""
        alerts = []
        
        if not hasattr(self, 'system_optimizer') or not self.system_optimizer:
            return alerts
        
        try:
            health_metrics = self.system_optimizer.monitor_system_health()
            
            # VRAM usage alerts
            if health_metrics.vram_total_mb > 0:
                vram_usage_percent = (health_metrics.vram_usage_mb / health_metrics.vram_total_mb) * 100
                
                if vram_usage_percent > 95:
                    alerts.append({
                        "type": "vram_critical",
                        "severity": "critical",
                        "message": f"VRAM usage critically high: {vram_usage_percent:.1f}%",
                        "recommendation": "Apply immediate VRAM optimization or reduce workload"
                    })
                elif vram_usage_percent > 85:
                    alerts.append({
                        "type": "vram_high",
                        "severity": "high",
                        "message": f"VRAM usage high: {vram_usage_percent:.1f}%",
                        "recommendation": "Consider applying quantization or CPU offloading"
                    })
            
            # GPU temperature alerts
            if health_metrics.gpu_temperature > 85:
                alerts.append({
                    "type": "gpu_temp_critical",
                    "severity": "critical",
                    "message": f"GPU temperature critically high: {health_metrics.gpu_temperature:.1f}°C",
                    "recommendation": "Reduce workload immediately to prevent thermal throttling"
                })
            elif health_metrics.gpu_temperature > 80:
                alerts.append({
                    "type": "gpu_temp_high",
                    "severity": "high",
                    "message": f"GPU temperature high: {health_metrics.gpu_temperature:.1f}°C",
                    "recommendation": "Monitor temperature and consider reducing workload"
                })
            
            # CPU usage alerts
            if health_metrics.cpu_usage_percent > 95:
                alerts.append({
                    "type": "cpu_critical",
                    "severity": "critical",
                    "message": f"CPU usage critically high: {health_metrics.cpu_usage_percent:.1f}%",
                    "recommendation": "Reduce concurrent operations or batch sizes"
                })
            
            # Memory usage alerts
            if health_metrics.memory_usage_gb > 64:  # High memory usage threshold
                alerts.append({
                    "type": "memory_high",
                    "severity": "medium",
                    "message": f"System memory usage high: {health_metrics.memory_usage_gb:.1f}GB",
                    "recommendation": "Consider clearing model cache or reducing concurrent operations"
                })
                
        except Exception as e:
            logger.warning(f"Failed to check system health alerts: {e}")
        
        return alerts


# Global performance monitor instance
_performance_monitor: Optional[PerformanceMonitor] = None

def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor