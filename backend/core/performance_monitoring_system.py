"""
Performance Monitoring and Optimization System for Enhanced Model Availability

This module provides comprehensive performance tracking for download operations,
health checks, fallback strategies, and system resource usage monitoring.
"""

import asyncio
import time
import psutil
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
import threading
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class PerformanceMetricType(Enum):
    """Types of performance metrics tracked"""
    DOWNLOAD_OPERATION = "download_operation"
    HEALTH_CHECK = "health_check"
    FALLBACK_STRATEGY = "fallback_strategy"
    ANALYTICS_COLLECTION = "analytics_collection"
    SYSTEM_RESOURCE = "system_resource"
    MODEL_OPERATION = "model_operation"


@dataclass
class PerformanceMetric:
    """Individual performance metric data"""
    metric_type: PerformanceMetricType
    operation_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    resource_usage: Optional[Dict[str, float]] = None


@dataclass
class SystemResourceSnapshot:
    """System resource usage snapshot"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_usage_percent: float
    disk_free_gb: float
    network_bytes_sent: int
    network_bytes_recv: int
    gpu_memory_used_mb: Optional[float] = None
    gpu_utilization_percent: Optional[float] = None


@dataclass
class PerformanceReport:
    """Comprehensive performance report"""
    report_period: Tuple[datetime, datetime]
    total_operations: int
    success_rate: float
    average_duration: float
    median_duration: float
    p95_duration: float
    operations_by_type: Dict[str, int]
    resource_usage_summary: Dict[str, float]
    bottlenecks_identified: List[str]
    optimization_recommendations: List[str]


class PerformanceTracker:
    """Tracks individual performance metrics"""
    
    def __init__(self):
        self.active_operations: Dict[str, PerformanceMetric] = {}
        self.completed_metrics: deque = deque(maxlen=10000)  # Keep last 10k metrics
        self._lock = threading.Lock()
    
    def start_operation(self, 
                       metric_type: PerformanceMetricType,
                       operation_name: str,
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """Start tracking a performance operation"""
        operation_id = f"{metric_type.value}_{operation_name}_{int(time.time() * 1000)}"
        
        metric = PerformanceMetric(
            metric_type=metric_type,
            operation_name=operation_name,
            start_time=datetime.now(),
            metadata=metadata or {}
        )
        
        with self._lock:
            self.active_operations[operation_id] = metric
        
        logger.debug(f"Started tracking operation: {operation_id}")
        return operation_id
    
    def end_operation(self, 
                     operation_id: str,
                     success: bool = True,
                     error_message: Optional[str] = None,
                     additional_metadata: Optional[Dict[str, Any]] = None) -> Optional[PerformanceMetric]:
        """End tracking a performance operation"""
        with self._lock:
            if operation_id not in self.active_operations:
                logger.warning(f"Operation {operation_id} not found in active operations")
                return None
            
            metric = self.active_operations.pop(operation_id)
        
        # Complete the metric
        metric.end_time = datetime.now()
        metric.duration_seconds = (metric.end_time - metric.start_time).total_seconds()
        metric.success = success
        metric.error_message = error_message
        
        if additional_metadata:
            metric.metadata.update(additional_metadata)
        
        # Capture resource usage at completion
        metric.resource_usage = self._capture_resource_usage()
        
        with self._lock:
            self.completed_metrics.append(metric)
        
        logger.debug(f"Completed tracking operation: {operation_id} (duration: {metric.duration_seconds:.3f}s)")
        return metric
    
    def _capture_resource_usage(self) -> Dict[str, float]:
        """Capture current system resource usage"""
        try:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used_mb': memory.used / (1024 * 1024),
                'disk_usage_percent': disk.percent,
                'disk_free_gb': disk.free / (1024 * 1024 * 1024)
            }
        except Exception as e:
            logger.warning(f"Failed to capture resource usage: {e}")
            return {}
    
    def get_metrics_by_type(self, metric_type: PerformanceMetricType, 
                           hours_back: int = 24) -> List[PerformanceMetric]:
        """Get metrics of specific type within time window"""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        with self._lock:
            return [
                metric for metric in self.completed_metrics
                if (metric.metric_type == metric_type and 
                    metric.start_time >= cutoff_time)
            ]
    
    def get_all_metrics(self, hours_back: int = 24) -> List[PerformanceMetric]:
        """Get all metrics within time window"""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        with self._lock:
            return [
                metric for metric in self.completed_metrics
                if metric.start_time >= cutoff_time
            ]


class SystemResourceMonitor:
    """Monitors system resource usage continuously"""
    
    def __init__(self, sample_interval: int = 30):
        self.sample_interval = sample_interval
        self.resource_history: deque = deque(maxlen=2880)  # 24 hours at 30s intervals
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._lock = threading.Lock()
    
    async def start_monitoring(self):
        """Start continuous resource monitoring"""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Started system resource monitoring")
    
    async def stop_monitoring(self):
        """Stop continuous resource monitoring"""
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped system resource monitoring")
    
    async def _monitor_loop(self):
        """Main monitoring loop"""
        while self._monitoring:
            try:
                snapshot = self._capture_snapshot()
                with self._lock:
                    self.resource_history.append(snapshot)
                
                await asyncio.sleep(self.sample_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in resource monitoring loop: {e}")
                await asyncio.sleep(self.sample_interval)
    
    def _capture_snapshot(self) -> SystemResourceSnapshot:
        """Capture current system resource snapshot"""
        try:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            
            snapshot = SystemResourceSnapshot(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / (1024 * 1024),
                disk_usage_percent=disk.percent,
                disk_free_gb=disk.free / (1024 * 1024 * 1024),
                network_bytes_sent=network.bytes_sent,
                network_bytes_recv=network.bytes_recv
            )
            
            # Try to get GPU info if available
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Use first GPU
                    snapshot.gpu_memory_used_mb = gpu.memoryUsed
                    snapshot.gpu_utilization_percent = gpu.load * 100
            except ImportError:
                pass  # GPU monitoring not available
            except Exception as e:
                logger.debug(f"GPU monitoring error: {e}")
            
            return snapshot
        except Exception as e:
            logger.error(f"Failed to capture resource snapshot: {e}")
            return SystemResourceSnapshot(
                timestamp=datetime.now(),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_used_mb=0.0,
                disk_usage_percent=0.0,
                disk_free_gb=0.0,
                network_bytes_sent=0,
                network_bytes_recv=0
            )
    
    def get_resource_history(self, hours_back: int = 24) -> List[SystemResourceSnapshot]:
        """Get resource usage history"""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        with self._lock:
            return [
                snapshot for snapshot in self.resource_history
                if snapshot.timestamp >= cutoff_time
            ]
    
    def get_current_usage(self) -> SystemResourceSnapshot:
        """Get current resource usage"""
        return self._capture_snapshot()


class PerformanceAnalyzer:
    """Analyzes performance data and provides optimization recommendations"""
    
    def __init__(self, tracker: PerformanceTracker, resource_monitor: SystemResourceMonitor):
        self.tracker = tracker
        self.resource_monitor = resource_monitor
    
    def generate_performance_report(self, hours_back: int = 24) -> PerformanceReport:
        """Generate comprehensive performance report"""
        metrics = self.tracker.get_all_metrics(hours_back)
        
        if not metrics:
            return PerformanceReport(
                report_period=(datetime.now() - timedelta(hours=hours_back), datetime.now()),
                total_operations=0,
                success_rate=0.0,
                average_duration=0.0,
                median_duration=0.0,
                p95_duration=0.0,
                operations_by_type={},
                resource_usage_summary={},
                bottlenecks_identified=[],
                optimization_recommendations=[]
            )
        
        # Calculate basic statistics
        total_operations = len(metrics)
        successful_operations = sum(1 for m in metrics if m.success)
        success_rate = successful_operations / total_operations
        
        durations = [m.duration_seconds for m in metrics if m.duration_seconds is not None]
        average_duration = sum(durations) / len(durations) if durations else 0.0
        median_duration = sorted(durations)[len(durations) // 2] if durations else 0.0
        p95_duration = sorted(durations)[int(len(durations) * 0.95)] if durations else 0.0
        
        # Operations by type
        operations_by_type = defaultdict(int)
        for metric in metrics:
            operations_by_type[metric.metric_type.value] += 1
        
        # Resource usage summary
        resource_history = self.resource_monitor.get_resource_history(hours_back)
        resource_usage_summary = self._calculate_resource_summary(resource_history)
        
        # Identify bottlenecks and recommendations
        bottlenecks = self._identify_bottlenecks(metrics, resource_history)
        recommendations = self._generate_recommendations(metrics, resource_history, bottlenecks)
        
        return PerformanceReport(
            report_period=(datetime.now() - timedelta(hours=hours_back), datetime.now()),
            total_operations=total_operations,
            success_rate=success_rate,
            average_duration=average_duration,
            median_duration=median_duration,
            p95_duration=p95_duration,
            operations_by_type=dict(operations_by_type),
            resource_usage_summary=resource_usage_summary,
            bottlenecks_identified=bottlenecks,
            optimization_recommendations=recommendations
        )
    
    def _calculate_resource_summary(self, resource_history: List[SystemResourceSnapshot]) -> Dict[str, float]:
        """Calculate resource usage summary statistics"""
        if not resource_history:
            return {}
        
        cpu_values = [s.cpu_percent for s in resource_history]
        memory_values = [s.memory_percent for s in resource_history]
        disk_values = [s.disk_usage_percent for s in resource_history]
        
        return {
            'avg_cpu_percent': sum(cpu_values) / len(cpu_values),
            'max_cpu_percent': max(cpu_values),
            'avg_memory_percent': sum(memory_values) / len(memory_values),
            'max_memory_percent': max(memory_values),
            'avg_disk_percent': sum(disk_values) / len(disk_values),
            'max_disk_percent': max(disk_values)
        }
    
    def _identify_bottlenecks(self, 
                            metrics: List[PerformanceMetric],
                            resource_history: List[SystemResourceSnapshot]) -> List[str]:
        """Identify performance bottlenecks"""
        bottlenecks = []
        
        # Check for slow operations
        slow_operations = [m for m in metrics if m.duration_seconds and m.duration_seconds > 30]
        if len(slow_operations) > len(metrics) * 0.1:  # More than 10% slow
            bottlenecks.append("High number of slow operations detected")
        
        # Check for high failure rate
        failed_operations = [m for m in metrics if not m.success]
        if len(failed_operations) > len(metrics) * 0.05:  # More than 5% failures
            bottlenecks.append("High operation failure rate detected")
        
        # Check resource constraints
        if resource_history:
            high_cpu = [s for s in resource_history if s.cpu_percent > 80]
            if len(high_cpu) > len(resource_history) * 0.2:  # More than 20% high CPU
                bottlenecks.append("High CPU usage detected")
            
            high_memory = [s for s in resource_history if s.memory_percent > 85]
            if len(high_memory) > len(resource_history) * 0.2:  # More than 20% high memory
                bottlenecks.append("High memory usage detected")
            
            low_disk = [s for s in resource_history if s.disk_free_gb < 5]
            if low_disk:
                bottlenecks.append("Low disk space detected")
        
        return bottlenecks
    
    def _generate_recommendations(self, 
                                metrics: List[PerformanceMetric],
                                resource_history: List[SystemResourceSnapshot],
                                bottlenecks: List[str]) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # Download operation recommendations
        download_metrics = [m for m in metrics if m.metric_type == PerformanceMetricType.DOWNLOAD_OPERATION]
        if download_metrics:
            avg_download_time = sum(m.duration_seconds for m in download_metrics if m.duration_seconds) / len(download_metrics)
            if avg_download_time > 300:  # 5 minutes
                recommendations.append("Consider implementing parallel downloads to reduce download times")
                recommendations.append("Implement bandwidth optimization for faster downloads")
        
        # Health check recommendations
        health_metrics = [m for m in metrics if m.metric_type == PerformanceMetricType.HEALTH_CHECK]
        if health_metrics:
            avg_health_time = sum(m.duration_seconds for m in health_metrics if m.duration_seconds) / len(health_metrics)
            if avg_health_time > 30:  # 30 seconds
                recommendations.append("Optimize health check algorithms for faster execution")
                recommendations.append("Consider caching health check results to reduce frequency")
        
        # Resource-based recommendations
        if "High CPU usage detected" in bottlenecks:
            recommendations.append("Consider reducing concurrent operations to lower CPU usage")
            recommendations.append("Implement operation scheduling during low-usage periods")
        
        if "High memory usage detected" in bottlenecks:
            recommendations.append("Implement memory cleanup for completed operations")
            recommendations.append("Consider reducing cache sizes to free memory")
        
        if "Low disk space detected" in bottlenecks:
            recommendations.append("Implement automatic cleanup of old model files")
            recommendations.append("Consider compressing stored model data")
        
        # Fallback strategy recommendations
        fallback_metrics = [m for m in metrics if m.metric_type == PerformanceMetricType.FALLBACK_STRATEGY]
        if fallback_metrics:
            failed_fallbacks = [m for m in fallback_metrics if not m.success]
            if len(failed_fallbacks) > len(fallback_metrics) * 0.1:
                recommendations.append("Review and optimize fallback strategy algorithms")
                recommendations.append("Implement better alternative model selection")
        
        return recommendations


class PerformanceMonitoringSystem:
    """Main performance monitoring and optimization system"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.tracker = PerformanceTracker()
        self.resource_monitor = SystemResourceMonitor(
            sample_interval=self.config.get('resource_sample_interval', 30)
        )
        self.analyzer = PerformanceAnalyzer(self.tracker, self.resource_monitor)
        self._dashboard_data_cache = {}
        self._cache_lock = threading.Lock()
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load performance monitoring configuration"""
        default_config = {
            'resource_sample_interval': 30,
            'metrics_retention_hours': 168,  # 1 week
            'dashboard_cache_ttl': 300,  # 5 minutes
            'enable_gpu_monitoring': True,
            'performance_thresholds': {
                'slow_operation_seconds': 30,
                'high_cpu_percent': 80,
                'high_memory_percent': 85,
                'low_disk_gb': 5
            }
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
        
        return default_config
    
    async def start(self):
        """Start the performance monitoring system"""
        await self.resource_monitor.start_monitoring()
        logger.info("Performance monitoring system started")
    
    async def stop(self):
        """Stop the performance monitoring system"""
        await self.resource_monitor.stop_monitoring()
        logger.info("Performance monitoring system stopped")
    
    def track_download_operation(self, operation_name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Track a download operation"""
        return self.tracker.start_operation(
            PerformanceMetricType.DOWNLOAD_OPERATION,
            operation_name,
            metadata
        )
    
    def track_health_check(self, operation_name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Track a health check operation"""
        return self.tracker.start_operation(
            PerformanceMetricType.HEALTH_CHECK,
            operation_name,
            metadata
        )
    
    def track_fallback_strategy(self, operation_name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Track a fallback strategy operation"""
        return self.tracker.start_operation(
            PerformanceMetricType.FALLBACK_STRATEGY,
            operation_name,
            metadata
        )
    
    def track_analytics_collection(self, operation_name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Track an analytics collection operation"""
        return self.tracker.start_operation(
            PerformanceMetricType.ANALYTICS_COLLECTION,
            operation_name,
            metadata
        )
    
    def track_model_operation(self, operation_name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Track a model operation"""
        return self.tracker.start_operation(
            PerformanceMetricType.MODEL_OPERATION,
            operation_name,
            metadata
        )
    
    def end_tracking(self, operation_id: str, success: bool = True, 
                    error_message: Optional[str] = None,
                    additional_metadata: Optional[Dict[str, Any]] = None) -> Optional[PerformanceMetric]:
        """End tracking an operation"""
        return self.tracker.end_operation(operation_id, success, error_message, additional_metadata)
    
    def get_performance_report(self, hours_back: int = 24) -> PerformanceReport:
        """Get comprehensive performance report"""
        return self.analyzer.generate_performance_report(hours_back)
    
    def get_dashboard_data(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Get performance data for dashboard display"""
        cache_key = "dashboard_data"
        cache_ttl = self.config.get('dashboard_cache_ttl', 300)
        
        with self._cache_lock:
            if not force_refresh and cache_key in self._dashboard_data_cache:
                cached_data, cache_time = self._dashboard_data_cache[cache_key]
                if (datetime.now() - cache_time).total_seconds() < cache_ttl:
                    return cached_data
        
        # Generate fresh dashboard data
        report = self.get_performance_report(24)
        current_resources = self.resource_monitor.get_current_usage()
        recent_metrics = self.tracker.get_all_metrics(1)  # Last hour
        
        dashboard_data = {
            'performance_summary': {
                'total_operations_24h': report.total_operations,
                'success_rate': report.success_rate,
                'average_duration': report.average_duration,
                'p95_duration': report.p95_duration
            },
            'current_resources': {
                'cpu_percent': current_resources.cpu_percent,
                'memory_percent': current_resources.memory_percent,
                'disk_free_gb': current_resources.disk_free_gb,
                'gpu_memory_used_mb': current_resources.gpu_memory_used_mb,
                'gpu_utilization_percent': current_resources.gpu_utilization_percent
            },
            'operations_by_type': report.operations_by_type,
            'recent_activity': len(recent_metrics),
            'bottlenecks': report.bottlenecks_identified,
            'recommendations': report.optimization_recommendations[:5],  # Top 5
            'resource_trends': self._calculate_resource_trends()
        }
        
        with self._cache_lock:
            self._dashboard_data_cache[cache_key] = (dashboard_data, datetime.now())
        
        return dashboard_data
    
    def _calculate_resource_trends(self) -> Dict[str, str]:
        """Calculate resource usage trends"""
        history = self.resource_monitor.get_resource_history(2)  # Last 2 hours
        if len(history) < 10:
            return {'cpu': 'stable', 'memory': 'stable', 'disk': 'stable'}
        
        # Split into two halves for trend calculation
        mid_point = len(history) // 2
        first_half = history[:mid_point]
        second_half = history[mid_point:]
        
        def calculate_trend(first_values, second_values):
            first_avg = sum(first_values) / len(first_values)
            second_avg = sum(second_values) / len(second_values)
            diff_percent = ((second_avg - first_avg) / first_avg) * 100
            
            if diff_percent > 10:
                return 'increasing'
            elif diff_percent < -10:
                return 'decreasing'
            else:
                return 'stable'
        
        cpu_trend = calculate_trend(
            [s.cpu_percent for s in first_half],
            [s.cpu_percent for s in second_half]
        )
        
        memory_trend = calculate_trend(
            [s.memory_percent for s in first_half],
            [s.memory_percent for s in second_half]
        )
        
        # Disk trend is inverted (decreasing free space = increasing usage)
        disk_trend = calculate_trend(
            [s.disk_free_gb for s in second_half],  # Inverted
            [s.disk_free_gb for s in first_half]
        )
        
        return {
            'cpu': cpu_trend,
            'memory': memory_trend,
            'disk': disk_trend
        }


# Global performance monitoring instance
_performance_monitor: Optional[PerformanceMonitoringSystem] = None


def get_performance_monitor() -> PerformanceMonitoringSystem:
    """Get the global performance monitoring instance"""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitoringSystem()
    return _performance_monitor


async def initialize_performance_monitoring(config_path: Optional[str] = None):
    """Initialize and start the performance monitoring system"""
    global _performance_monitor
    _performance_monitor = PerformanceMonitoringSystem(config_path)
    await _performance_monitor.start()


async def shutdown_performance_monitoring():
    """Shutdown the performance monitoring system"""
    global _performance_monitor
    if _performance_monitor:
        await _performance_monitor.stop()