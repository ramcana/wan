"""
Application metrics collection and monitoring.
"""

import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import json
import logging

try:
    import pynvml
    NVIDIA_AVAILABLE = True
except ImportError:
    NVIDIA_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    """System resource metrics."""
    timestamp: datetime
    cpu_percent: float
    ram_used_gb: float
    ram_total_gb: float
    gpu_percent: float
    vram_used_mb: float
    vram_total_mb: float
    disk_used_gb: float
    disk_total_gb: float

@dataclass
class ApplicationMetrics:
    """Application-specific metrics."""
    timestamp: datetime
    active_generations: int
    queue_size: int
    completed_generations: int
    failed_generations: int
    average_generation_time: float
    error_rate: float
    uptime_seconds: float

@dataclass
class PerformanceMetrics:
    """Performance metrics."""
    timestamp: datetime
    response_times: Dict[str, float]  # endpoint -> avg response time
    request_counts: Dict[str, int]    # endpoint -> request count
    error_counts: Dict[str, int]      # error_type -> count
    throughput: float                 # requests per second

class MetricsCollector:
    """Collects and stores application metrics."""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.system_metrics_history = deque(maxlen=history_size)
        self.app_metrics_history = deque(maxlen=history_size)
        self.performance_metrics_history = deque(maxlen=history_size)
        
        # Counters
        self.request_counts = defaultdict(int)
        self.error_counts = defaultdict(int)
        self.response_times = defaultdict(list)
        self.generation_times = deque(maxlen=100)
        self.generation_counts = {'completed': 0, 'failed': 0}
        
        # State
        self.start_time = datetime.now()
        self.active_generations = 0
        self.queue_size = 0
        
        # GPU initialization
        self.gpu_available = False
        if NVIDIA_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.gpu_available = True
                logger.info("NVIDIA GPU monitoring initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize GPU monitoring: {e}")
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        # CPU and RAM
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        ram_used_gb = memory.used / (1024**3)
        ram_total_gb = memory.total / (1024**3)
        
        # Disk
        disk = psutil.disk_usage('/')
        disk_used_gb = disk.used / (1024**3)
        disk_total_gb = disk.total / (1024**3)
        
        # GPU
        gpu_percent = 0.0
        vram_used_mb = 0.0
        vram_total_mb = 0.0
        
        if self.gpu_available:
            try:
                # GPU utilization
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                gpu_percent = gpu_util.gpu
                
                # VRAM usage
                vram_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                vram_used_mb = vram_info.used / (1024**2)
                vram_total_mb = vram_info.total / (1024**2)
                
            except Exception as e:
                logger.warning(f"Failed to collect GPU metrics: {e}")
        
        metrics = SystemMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            ram_used_gb=ram_used_gb,
            ram_total_gb=ram_total_gb,
            gpu_percent=gpu_percent,
            vram_used_mb=vram_used_mb,
            vram_total_mb=vram_total_mb,
            disk_used_gb=disk_used_gb,
            disk_total_gb=disk_total_gb
        )
        
        self.system_metrics_history.append(metrics)
        return metrics
    
    def collect_application_metrics(self) -> ApplicationMetrics:
        """Collect current application metrics."""
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        # Calculate average generation time
        avg_generation_time = 0.0
        if self.generation_times:
            avg_generation_time = sum(self.generation_times) / len(self.generation_times)
        
        # Calculate error rate
        total_generations = self.generation_counts['completed'] + self.generation_counts['failed']
        error_rate = 0.0
        if total_generations > 0:
            error_rate = self.generation_counts['failed'] / total_generations
        
        metrics = ApplicationMetrics(
            timestamp=datetime.now(),
            active_generations=self.active_generations,
            queue_size=self.queue_size,
            completed_generations=self.generation_counts['completed'],
            failed_generations=self.generation_counts['failed'],
            average_generation_time=avg_generation_time,
            error_rate=error_rate,
            uptime_seconds=uptime
        )
        
        self.app_metrics_history.append(metrics)
        return metrics
    
    def collect_performance_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        # Calculate average response times
        avg_response_times = {}
        for endpoint, times in self.response_times.items():
            if times:
                avg_response_times[endpoint] = sum(times) / len(times)
        
        # Calculate throughput (requests per second over last minute)
        now = datetime.now()
        one_minute_ago = now - timedelta(minutes=1)
        
        recent_requests = 0
        for metrics in reversed(self.performance_metrics_history):
            if metrics.timestamp < one_minute_ago:
                break
            recent_requests += sum(metrics.request_counts.values())
        
        throughput = recent_requests / 60.0  # requests per second
        
        metrics = PerformanceMetrics(
            timestamp=now,
            response_times=avg_response_times.copy(),
            request_counts=dict(self.request_counts),
            error_counts=dict(self.error_counts),
            throughput=throughput
        )
        
        self.performance_metrics_history.append(metrics)
        
        # Reset counters for next collection
        self.response_times.clear()
        
        return metrics
    
    def record_request(self, endpoint: str, response_time: float, status_code: int) -> None:
        """Record HTTP request metrics."""
        self.request_counts[endpoint] += 1
        self.response_times[endpoint].append(response_time)
        
        if status_code >= 400:
            error_type = f"http_{status_code}"
            self.error_counts[error_type] += 1
    
    def record_generation(self, duration: float, success: bool, error_type: Optional[str] = None) -> None:
        """Record generation metrics."""
        self.generation_times.append(duration)
        
        if success:
            self.generation_counts['completed'] += 1
        else:
            self.generation_counts['failed'] += 1
            if error_type:
                self.error_counts[error_type] += 1
    
    def update_queue_size(self, size: int) -> None:
        """Update current queue size."""
        self.queue_size = size
    
    def update_active_generations(self, count: int) -> None:
        """Update active generation count."""
        self.active_generations = count
    
    def get_latest_metrics(self) -> Dict[str, Any]:
        """Get latest metrics from all categories."""
        system_metrics = self.collect_system_metrics()
        app_metrics = self.collect_application_metrics()
        perf_metrics = self.collect_performance_metrics()
        
        return {
            'system': asdict(system_metrics),
            'application': asdict(app_metrics),
            'performance': asdict(perf_metrics)
        }
    
    def get_metrics_history(self, minutes: int = 60) -> Dict[str, List[Dict[str, Any]]]:
        """Get metrics history for specified time period."""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        def filter_by_time(metrics_list):
            return [
                asdict(m) for m in metrics_list 
                if m.timestamp >= cutoff_time
            ]
        
        return {
            'system': filter_by_time(self.system_metrics_history),
            'application': filter_by_time(self.app_metrics_history),
            'performance': filter_by_time(self.performance_metrics_history)
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall system health status."""
        if not self.system_metrics_history:
            return {'status': 'unknown', 'message': 'No metrics available'}
        
        latest_system = self.system_metrics_history[-1]
        latest_app = self.app_metrics_history[-1] if self.app_metrics_history else None
        
        issues = []
        warnings = []
        
        # Check system resources
        if latest_system.cpu_percent > 90:
            issues.append(f"High CPU usage: {latest_system.cpu_percent:.1f}%")
        elif latest_system.cpu_percent > 70:
            warnings.append(f"Elevated CPU usage: {latest_system.cpu_percent:.1f}%")
        
        ram_usage_percent = (latest_system.ram_used_gb / latest_system.ram_total_gb) * 100
        if ram_usage_percent > 90:
            issues.append(f"High RAM usage: {ram_usage_percent:.1f}%")
        elif ram_usage_percent > 70:
            warnings.append(f"Elevated RAM usage: {ram_usage_percent:.1f}%")
        
        if latest_system.vram_total_mb > 0:
            vram_usage_percent = (latest_system.vram_used_mb / latest_system.vram_total_mb) * 100
            if vram_usage_percent > 95:
                issues.append(f"High VRAM usage: {vram_usage_percent:.1f}%")
            elif vram_usage_percent > 80:
                warnings.append(f"Elevated VRAM usage: {vram_usage_percent:.1f}%")
        
        # Check application metrics
        if latest_app:
            if latest_app.error_rate > 0.1:  # 10% error rate
                issues.append(f"High error rate: {latest_app.error_rate:.1%}")
            elif latest_app.error_rate > 0.05:  # 5% error rate
                warnings.append(f"Elevated error rate: {latest_app.error_rate:.1%}")
        
        # Determine overall status
        if issues:
            status = 'critical'
            message = f"Critical issues detected: {'; '.join(issues)}"
        elif warnings:
            status = 'warning'
            message = f"Warnings detected: {'; '.join(warnings)}"
        else:
            status = 'healthy'
            message = 'All systems operating normally'
        
        return {
            'status': status,
            'message': message,
            'issues': issues,
            'warnings': warnings,
            'timestamp': datetime.now().isoformat()
        }
    
    def export_metrics(self, filepath: str) -> None:
        """Export metrics to JSON file."""
        data = {
            'export_timestamp': datetime.now().isoformat(),
            'system_metrics': [asdict(m) for m in self.system_metrics_history],
            'application_metrics': [asdict(m) for m in self.app_metrics_history],
            'performance_metrics': [asdict(m) for m in self.performance_metrics_history]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Metrics exported to {filepath}")

class MetricsMonitor:
    """Background metrics monitoring service."""
    
    def __init__(self, collector: MetricsCollector, collection_interval: int = 30):
        self.collector = collector
        self.collection_interval = collection_interval
        self.running = False
        self.thread = None
    
    def start(self) -> None:
        """Start metrics collection in background thread."""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._collect_loop, daemon=True)
        self.thread.start()
        logger.info(f"Metrics monitor started with {self.collection_interval}s interval")
    
    def stop(self) -> None:
        """Stop metrics collection."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("Metrics monitor stopped")
    
    def _collect_loop(self) -> None:
        """Main collection loop."""
        while self.running:
            try:
                self.collector.collect_system_metrics()
                self.collector.collect_application_metrics()
                self.collector.collect_performance_metrics()
                
                # Log health status periodically
                health = self.collector.get_health_status()
                if health['status'] != 'healthy':
                    logger.warning(f"Health check: {health['message']}")
                
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
            
            time.sleep(self.collection_interval)

# Global metrics collector instance
metrics_collector = MetricsCollector()
metrics_monitor = MetricsMonitor(metrics_collector)

def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance."""
    return metrics_collector

def start_metrics_monitoring() -> None:
    """Start background metrics monitoring."""
    metrics_monitor.start()

def stop_metrics_monitoring() -> None:
    """Stop background metrics monitoring."""
    metrics_monitor.stop()
