"""
WAN22 System Health Monitor

Provides continuous monitoring of system health metrics including GPU temperature,
VRAM usage, CPU usage, and memory consumption with safety threshold checking
and automatic workload reduction capabilities.
"""

import time
import threading
import logging
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Callable, Any
from collections import deque
import psutil

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    pynvml = None

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


@dataclass
class SystemMetrics:
    """System health metrics at a point in time"""
    timestamp: datetime
    gpu_temperature: float
    gpu_utilization: float
    vram_usage_mb: int
    vram_total_mb: int
    vram_usage_percent: float
    cpu_usage_percent: float
    memory_usage_gb: float
    memory_total_gb: float
    memory_usage_percent: float
    disk_usage_percent: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class SafetyThresholds:
    """Safety thresholds for system health monitoring"""
    gpu_temperature_warning: float = 80.0  # Celsius
    gpu_temperature_critical: float = 85.0  # Celsius
    vram_usage_warning: float = 85.0  # Percent
    vram_usage_critical: float = 95.0  # Percent
    cpu_usage_warning: float = 85.0  # Percent
    cpu_usage_critical: float = 95.0  # Percent
    memory_usage_warning: float = 85.0  # Percent
    memory_usage_critical: float = 95.0  # Percent
    disk_usage_warning: float = 90.0  # Percent
    disk_usage_critical: float = 95.0  # Percent


@dataclass
class HealthAlert:
    """Health alert information"""
    timestamp: datetime
    severity: str  # 'warning', 'critical'
    component: str  # 'gpu', 'cpu', 'memory', 'disk'
    metric: str
    current_value: float
    threshold_value: float
    message: str
    resolved: bool = False
    resolved_timestamp: Optional[datetime] = None


class HealthMonitor:
    """
    Continuous system health monitoring with safety threshold checking
    and automatic workload reduction capabilities.
    """
    
    def __init__(self, 
                 monitoring_interval: float = 5.0,
                 history_size: int = 1000,
                 thresholds: Optional[SafetyThresholds] = None):
        """
        Initialize health monitor
        
        Args:
            monitoring_interval: Seconds between health checks
            history_size: Number of historical metrics to keep
            thresholds: Safety thresholds for alerts
        """
        self.monitoring_interval = monitoring_interval
        self.history_size = history_size
        self.thresholds = thresholds or SafetyThresholds()
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Data storage
        self.metrics_history: deque = deque(maxlen=history_size)
        self.active_alerts: List[HealthAlert] = []
        self.alert_history: List[HealthAlert] = []
        
        # Callbacks for alerts and workload reduction
        self.alert_callbacks: List[Callable[[HealthAlert], None]] = []
        self.workload_reduction_callbacks: List[Callable[[str, float], None]] = []
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
        # Initialize hardware monitoring
        self.gpu_available = False
        self.gpu_handle = None
        self._initialize_gpu_monitoring()
        
    def _initialize_gpu_monitoring(self):
        """Initialize GPU monitoring capabilities"""
        if not NVML_AVAILABLE:
            self.logger.warning("NVML not available - GPU monitoring disabled")
            return
            
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count > 0:
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.gpu_available = True
                self.logger.info(f"GPU monitoring initialized - {device_count} GPU(s) detected")
            else:
                self.logger.warning("No GPUs detected")
        except Exception as e:
            self.logger.error(f"Failed to initialize GPU monitoring: {e}")
            
    def start_monitoring(self):
        """Start continuous health monitoring"""
        if self.is_monitoring:
            self.logger.warning("Health monitoring already running")
            return
            
        self.is_monitoring = True
        self.stop_event.clear()
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Health monitoring started")
        
    def stop_monitoring(self):
        """Stop health monitoring"""
        if not self.is_monitoring:
            return
            
        self.is_monitoring = False
        self.stop_event.set()
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=10.0)
            
        self.logger.info("Health monitoring stopped")
        
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while not self.stop_event.wait(self.monitoring_interval):
            try:
                metrics = self._collect_metrics()
                if metrics:
                    self.metrics_history.append(metrics)
                    self._check_safety_thresholds(metrics)
                    
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                
    def _collect_metrics(self) -> Optional[SystemMetrics]:
        """Collect current system metrics"""
        try:
            # GPU metrics
            gpu_temp = 0.0
            gpu_util = 0.0
            vram_used = 0
            vram_total = 0
            
            if self.gpu_available and self.gpu_handle:
                try:
                    gpu_temp = pynvml.nvmlDeviceGetTemperature(self.gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
                    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle).gpu
                    
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                    vram_used = memory_info.used // (1024 * 1024)  # Convert to MB
                    vram_total = memory_info.total // (1024 * 1024)  # Convert to MB
                except Exception as e:
                    self.logger.warning(f"Failed to collect GPU metrics: {e}")
                    
            # CPU and system memory metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Calculate percentages
            vram_percent = (vram_used / vram_total * 100) if vram_total > 0 else 0
            memory_gb = memory.used / (1024**3)
            memory_total_gb = memory.total / (1024**3)
            
            return SystemMetrics(
                timestamp=datetime.now(),
                gpu_temperature=gpu_temp,
                gpu_utilization=gpu_util,
                vram_usage_mb=vram_used,
                vram_total_mb=vram_total,
                vram_usage_percent=vram_percent,
                cpu_usage_percent=cpu_percent,
                memory_usage_gb=memory_gb,
                memory_total_gb=memory_total_gb,
                memory_usage_percent=memory.percent,
                disk_usage_percent=disk.percent
            )
            
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")
            return None
            
    def _check_safety_thresholds(self, metrics: SystemMetrics):
        """Check metrics against safety thresholds and generate alerts"""
        current_time = datetime.now()
        
        # Check GPU temperature
        if metrics.gpu_temperature > 0:  # Only check if we have valid GPU data
            if metrics.gpu_temperature >= self.thresholds.gpu_temperature_critical:
                self._create_alert('critical', 'gpu', 'temperature', 
                                 metrics.gpu_temperature, self.thresholds.gpu_temperature_critical,
                                 f"GPU temperature critically high: {metrics.gpu_temperature:.1f}°C")
                self._trigger_workload_reduction('gpu_temperature', metrics.gpu_temperature)
                
            elif metrics.gpu_temperature >= self.thresholds.gpu_temperature_warning:
                self._create_alert('warning', 'gpu', 'temperature',
                                 metrics.gpu_temperature, self.thresholds.gpu_temperature_warning,
                                 f"GPU temperature high: {metrics.gpu_temperature:.1f}°C")
                
        # Check VRAM usage
        if metrics.vram_usage_percent >= self.thresholds.vram_usage_critical:
            self._create_alert('critical', 'gpu', 'vram_usage',
                             metrics.vram_usage_percent, self.thresholds.vram_usage_critical,
                             f"VRAM usage critically high: {metrics.vram_usage_percent:.1f}%")
            self._trigger_workload_reduction('vram_usage', metrics.vram_usage_percent)
            
        elif metrics.vram_usage_percent >= self.thresholds.vram_usage_warning:
            self._create_alert('warning', 'gpu', 'vram_usage',
                             metrics.vram_usage_percent, self.thresholds.vram_usage_warning,
                             f"VRAM usage high: {metrics.vram_usage_percent:.1f}%")
            
        # Check CPU usage
        if metrics.cpu_usage_percent >= self.thresholds.cpu_usage_critical:
            self._create_alert('critical', 'cpu', 'usage',
                             metrics.cpu_usage_percent, self.thresholds.cpu_usage_critical,
                             f"CPU usage critically high: {metrics.cpu_usage_percent:.1f}%")
            self._trigger_workload_reduction('cpu_usage', metrics.cpu_usage_percent)
            
        elif metrics.cpu_usage_percent >= self.thresholds.cpu_usage_warning:
            self._create_alert('warning', 'cpu', 'usage',
                             metrics.cpu_usage_percent, self.thresholds.cpu_usage_warning,
                             f"CPU usage high: {metrics.cpu_usage_percent:.1f}%")
            
        # Check memory usage
        if metrics.memory_usage_percent >= self.thresholds.memory_usage_critical:
            self._create_alert('critical', 'memory', 'usage',
                             metrics.memory_usage_percent, self.thresholds.memory_usage_critical,
                             f"Memory usage critically high: {metrics.memory_usage_percent:.1f}%")
            self._trigger_workload_reduction('memory_usage', metrics.memory_usage_percent)
            
        elif metrics.memory_usage_percent >= self.thresholds.memory_usage_warning:
            self._create_alert('warning', 'memory', 'usage',
                             metrics.memory_usage_percent, self.thresholds.memory_usage_warning,
                             f"Memory usage high: {metrics.memory_usage_percent:.1f}%")
            
        # Check disk usage
        if metrics.disk_usage_percent >= self.thresholds.disk_usage_critical:
            self._create_alert('critical', 'disk', 'usage',
                             metrics.disk_usage_percent, self.thresholds.disk_usage_critical,
                             f"Disk usage critically high: {metrics.disk_usage_percent:.1f}%")
            
        elif metrics.disk_usage_percent >= self.thresholds.disk_usage_warning:
            self._create_alert('warning', 'disk', 'usage',
                             metrics.disk_usage_percent, self.thresholds.disk_usage_warning,
                             f"Disk usage high: {metrics.disk_usage_percent:.1f}%")
                             
    def _create_alert(self, severity: str, component: str, metric: str, 
                     current_value: float, threshold_value: float, message: str):
        """Create and manage health alerts"""
        # Check if similar alert already exists
        existing_alert = None
        for alert in self.active_alerts:
            if (alert.component == component and alert.metric == metric and 
                alert.severity == severity and not alert.resolved):
                existing_alert = alert
                break
                
        if existing_alert:
            # Update existing alert
            existing_alert.current_value = current_value
            existing_alert.timestamp = datetime.now()
        else:
            # Create new alert
            alert = HealthAlert(
                timestamp=datetime.now(),
                severity=severity,
                component=component,
                metric=metric,
                current_value=current_value,
                threshold_value=threshold_value,
                message=message
            )
            
            self.active_alerts.append(alert)
            self.alert_history.append(alert)
            
            # Notify callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    self.logger.error(f"Error in alert callback: {e}")
                    
            self.logger.warning(f"Health alert: {message}")
            
    def _trigger_workload_reduction(self, reason: str, value: float):
        """Trigger workload reduction callbacks"""
        for callback in self.workload_reduction_callbacks:
            try:
                callback(reason, value)
            except Exception as e:
                self.logger.error(f"Error in workload reduction callback: {e}")
                
    def get_current_metrics(self) -> Optional[SystemMetrics]:
        """Get the most recent system metrics"""
        if self.metrics_history:
            return self.metrics_history[-1]
        return None
        
    def get_metrics_history(self, duration_minutes: int = 60) -> List[SystemMetrics]:
        """Get metrics history for specified duration"""
        if not self.metrics_history:
            return []
            
        cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
        return [m for m in self.metrics_history if m.timestamp >= cutoff_time]
        
    def get_active_alerts(self) -> List[HealthAlert]:
        """Get all active (unresolved) alerts"""
        return [alert for alert in self.active_alerts if not alert.resolved]
        
    def get_alert_history(self, duration_hours: int = 24) -> List[HealthAlert]:
        """Get alert history for specified duration"""
        cutoff_time = datetime.now() - timedelta(hours=duration_hours)
        return [alert for alert in self.alert_history if alert.timestamp >= cutoff_time]
        
    def resolve_alert(self, alert: HealthAlert):
        """Mark an alert as resolved"""
        alert.resolved = True
        alert.resolved_timestamp = datetime.now()
        self.logger.info(f"Alert resolved: {alert.message}")
        
    def clear_resolved_alerts(self):
        """Remove resolved alerts from active alerts list"""
        self.active_alerts = [alert for alert in self.active_alerts if not alert.resolved]
        
    def add_alert_callback(self, callback: Callable[[HealthAlert], None]):
        """Add callback for health alerts"""
        self.alert_callbacks.append(callback)
        
    def add_workload_reduction_callback(self, callback: Callable[[str, float], None]):
        """Add callback for workload reduction triggers"""
        self.workload_reduction_callbacks.append(callback)
        
    def update_thresholds(self, thresholds: SafetyThresholds):
        """Update safety thresholds"""
        self.thresholds = thresholds
        self.logger.info("Safety thresholds updated")
        
    def get_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive health summary"""
        current_metrics = self.get_current_metrics()
        active_alerts = self.get_active_alerts()
        
        if not current_metrics:
            return {"status": "no_data", "message": "No metrics available"}
            
        # Determine overall health status
        critical_alerts = [a for a in active_alerts if a.severity == 'critical']
        warning_alerts = [a for a in active_alerts if a.severity == 'warning']
        
        if critical_alerts:
            status = "critical"
        elif warning_alerts:
            status = "warning"
        else:
            status = "healthy"
            
        return {
            "status": status,
            "timestamp": current_metrics.timestamp.isoformat(),
            "metrics": current_metrics.to_dict(),
            "active_alerts": len(active_alerts),
            "critical_alerts": len(critical_alerts),
            "warning_alerts": len(warning_alerts),
            "gpu_available": self.gpu_available,
            "monitoring_active": self.is_monitoring
        }
        
    def export_metrics_history(self, filepath: str, duration_hours: int = 24):
        """Export metrics history to JSON file"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=duration_hours)
            history = [m.to_dict() for m in self.metrics_history if m.timestamp >= cutoff_time]
            
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "duration_hours": duration_hours,
                "metrics_count": len(history),
                "metrics": history
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
                
            self.logger.info(f"Metrics history exported to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to export metrics history: {e}")
            raise
            
    def __enter__(self):
        """Context manager entry"""
        self.start_monitoring()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_monitoring()


# Example usage and testing functions
def create_demo_health_monitor():
    """Create a demo health monitor for testing"""
    
    def alert_handler(alert: HealthAlert):
        print(f"ALERT [{alert.severity.upper()}]: {alert.message}")
        
    def workload_reduction_handler(reason: str, value: float):
        print(f"WORKLOAD REDUCTION: {reason} = {value}")
        
    # Create monitor with custom thresholds
    thresholds = SafetyThresholds(
        gpu_temperature_warning=75.0,
        gpu_temperature_critical=80.0,
        vram_usage_warning=80.0,
        vram_usage_critical=90.0
    )
    
    monitor = HealthMonitor(
        monitoring_interval=2.0,
        thresholds=thresholds
    )
    
    # Add callbacks
    monitor.add_alert_callback(alert_handler)
    monitor.add_workload_reduction_callback(workload_reduction_handler)
    
    return monitor


if __name__ == "__main__":
    # Demo usage
    print("WAN22 Health Monitor Demo")
    print("=" * 40)
    
    monitor = create_demo_health_monitor()
    
    try:
        with monitor:
            print("Health monitoring started...")
            print("Press Ctrl+C to stop")
            
            # Monitor for a short time
            for i in range(10):
                time.sleep(5)
                
                current = monitor.get_current_metrics()
                if current:
                    print(f"\nCurrent metrics:")
                    print(f"  GPU Temp: {current.gpu_temperature:.1f}°C")
                    print(f"  VRAM: {current.vram_usage_percent:.1f}%")
                    print(f"  CPU: {current.cpu_usage_percent:.1f}%")
                    print(f"  Memory: {current.memory_usage_percent:.1f}%")
                    
                    alerts = monitor.get_active_alerts()
                    if alerts:
                        print(f"  Active alerts: {len(alerts)}")
                        
    except KeyboardInterrupt:
        print("\nStopping health monitor...")
        
    print("Demo completed")