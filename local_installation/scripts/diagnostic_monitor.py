"""
Comprehensive diagnostic monitoring system for installation reliability
Implements real-time resource monitoring, error pattern detection, and predictive failure analysis
"""

import json
import time
import threading
import gc
import os
import subprocess
import shutil
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import psutil
import logging
from collections import deque, defaultdict

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class HealthStatus(Enum):
    """Component health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class Alert:
    """System alert data structure"""
    level: AlertLevel
    message: str
    metric_name: str
    current_value: float
    threshold_value: float
    component: str = "system"
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "level": self.level.value,
            "message": self.message,
            "metric_name": self.metric_name,
            "current_value": self.current_value,
            "threshold_value": self.threshold_value,
            "component": self.component,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class ResourceMetrics:
    """System resource metrics"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_usage_percent: float
    disk_free_gb: float
    gpu_percent: float = 0.0
    vram_used_mb: int = 0
    vram_total_mb: int = 0
    vram_percent: float = 0.0
    network_bytes_sent: int = 0
    network_bytes_recv: int = 0
    process_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "memory_used_gb": self.memory_used_gb,
            "memory_total_gb": self.memory_total_gb,
            "disk_usage_percent": self.disk_usage_percent,
            "disk_free_gb": self.disk_free_gb,
            "gpu_percent": self.gpu_percent,
            "vram_used_mb": self.vram_used_mb,
            "vram_total_mb": self.vram_total_mb,
            "vram_percent": self.vram_percent,
            "network_bytes_sent": self.network_bytes_sent,
            "network_bytes_recv": self.network_bytes_recv,
            "process_count": self.process_count
        }


@dataclass
class ComponentHealth:
    """Component health information"""
    component_name: str
    status: HealthStatus
    response_time_ms: float
    error_count: int
    last_error: Optional[str]
    uptime_seconds: float
    performance_score: float  # 0-100 scale
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "component_name": self.component_name,
            "status": self.status.value,
            "response_time_ms": self.response_time_ms,
            "error_count": self.error_count,
            "last_error": self.last_error,
            "uptime_seconds": self.uptime_seconds,
            "performance_score": self.performance_score,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class ErrorPattern:
    """Error pattern detection result"""
    pattern_type: str
    frequency: int
    first_occurrence: datetime
    last_occurrence: datetime
    components_affected: List[str]
    severity_trend: str  # "increasing", "stable", "decreasing"
    prediction_confidence: float  # 0-1 scale
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "pattern_type": self.pattern_type,
            "frequency": self.frequency,
            "first_occurrence": self.first_occurrence.isoformat(),
            "last_occurrence": self.last_occurrence.isoformat(),
            "components_affected": self.components_affected,
            "severity_trend": self.severity_trend,
            "prediction_confidence": self.prediction_confidence
        }


@dataclass
class PotentialIssue:
    """Potential issue detected by predictive analysis"""
    issue_type: str
    probability: float  # 0-1 scale
    estimated_time_to_failure: Optional[timedelta]
    affected_components: List[str]
    recommended_actions: List[str]
    confidence_level: float  # 0-1 scale
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        result = {
            "issue_type": self.issue_type,
            "probability": self.probability,
            "affected_components": self.affected_components,
            "recommended_actions": self.recommended_actions,
            "confidence_level": self.confidence_level
        }
        if self.estimated_time_to_failure:
            result["estimated_time_to_failure_seconds"] = self.estimated_time_to_failure.total_seconds()
        return result


@dataclass
class HealthReport:
    """Comprehensive health report"""
    timestamp: datetime
    overall_health: HealthStatus
    resource_metrics: ResourceMetrics
    component_health: List[ComponentHealth]
    active_alerts: List[Alert]
    error_patterns: List[ErrorPattern]
    potential_issues: List[PotentialIssue]
    performance_trends: Dict[str, List[float]]
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "overall_health": self.overall_health.value,
            "resource_metrics": self.resource_metrics.to_dict(),
            "component_health": [ch.to_dict() for ch in self.component_health],
            "active_alerts": [alert.to_dict() for alert in self.active_alerts],
            "error_patterns": [ep.to_dict() for ep in self.error_patterns],
            "potential_issues": [pi.to_dict() for pi in self.potential_issues],
            "performance_trends": self.performance_trends,
            "recommendations": self.recommendations
        }


class DiagnosticMonitor:
    """
    Comprehensive diagnostic monitoring system with real-time monitoring,
    error pattern detection, and predictive failure analysis
    """
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize diagnostic monitor"""
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = self._setup_logging()
        
        # Monitoring configuration
        self.monitoring_interval = self.config.get("monitoring", {}).get("interval", 5)
        self.history_retention_hours = self.config.get("monitoring", {}).get("history_retention_hours", 24)
        self.max_history_size = int(self.history_retention_hours * 3600 / self.monitoring_interval)
        
        # Thresholds
        self.thresholds = {
            "cpu_warning": 80.0,
            "cpu_critical": 95.0,
            "memory_warning": 80.0,
            "memory_critical": 95.0,
            "vram_warning": 85.0,
            "vram_critical": 95.0,
            "disk_warning": 85.0,
            "disk_critical": 95.0,
            "response_time_warning": 1000.0,  # ms
            "response_time_critical": 5000.0  # ms
        }
        self.thresholds.update(self.config.get("monitoring", {}).get("thresholds", {}))
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Data storage
        self.metrics_history: deque = deque(maxlen=self.max_history_size)
        self.component_health_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.error_history: deque = deque(maxlen=1000)
        self.alert_history: deque = deque(maxlen=500)
        
        # Component tracking
        self.monitored_components = {
            "model_downloader": {"last_check": None, "response_times": deque(maxlen=50)},
            "dependency_manager": {"last_check": None, "response_times": deque(maxlen=50)},
            "config_validator": {"last_check": None, "response_times": deque(maxlen=50)},
            "error_handler": {"last_check": None, "response_times": deque(maxlen=50)},
            "reliability_manager": {"last_check": None, "response_times": deque(maxlen=50)}
        }
        
        # Error pattern detection
        self.error_patterns: Dict[str, List[datetime]] = defaultdict(list)
        self.pattern_detection_window = timedelta(hours=1)
        
        # Callbacks
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        self.health_callbacks: List[Callable[[HealthReport], None]] = []
        
        self.logger.info("DiagnosticMonitor initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"Failed to load config from {self.config_path}: {e}")
            return {}
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for diagnostic monitor"""
        logger = logging.getLogger("diagnostic_monitor")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def start_monitoring(self):
        """Start continuous monitoring"""
        if self.is_monitoring:
            self.logger.warning("Monitoring already active")
            return
        
        self.is_monitoring = True
        self.stop_event.clear()
        
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            name="diagnostic-monitor"
        )
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        self.logger.info("Started diagnostic monitoring")
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        self.stop_event.set()
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=10.0)
        
        self.logger.info("Stopped diagnostic monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        self.logger.info(f"Starting monitoring loop with {self.monitoring_interval}s interval")
        
        while not self.stop_event.is_set():
            try:
                # Collect resource metrics
                metrics = self._collect_resource_metrics()
                self.metrics_history.append(metrics)
                
                # Check component health
                component_health = self._check_component_health()
                
                # Check thresholds and generate alerts
                alerts = self._check_thresholds(metrics, component_health)
                for alert in alerts:
                    self.alert_history.append(alert)
                    self._trigger_alert_callbacks(alert)
                
                # Detect error patterns
                error_patterns = self._detect_error_patterns()
                
                # Detect potential issues
                potential_issues = self._detect_potential_issues()
                
                # Generate health report
                health_report = self._generate_health_report(
                    metrics, component_health, list(alerts), 
                    error_patterns, potential_issues
                )
                
                # Trigger health callbacks
                self._trigger_health_callbacks(health_report)
                
                # Clean up old data
                self._cleanup_old_data()
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
            
            # Wait for next interval
            self.stop_event.wait(self.monitoring_interval)
    
    def _collect_resource_metrics(self) -> ResourceMetrics:
        """Collect comprehensive system resource metrics"""
        try:
            timestamp = datetime.now()
            
            # CPU and memory metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_gb = memory.used / (1024**3)
            memory_total_gb = memory.total / (1024**3)
            
            # Disk metrics
            disk_usage = shutil.disk_usage(".")
            disk_usage_percent = ((disk_usage.total - disk_usage.free) / disk_usage.total) * 100
            disk_free_gb = disk_usage.free / (1024**3)
            
            # Network metrics
            net_io = psutil.net_io_counters()
            network_bytes_sent = net_io.bytes_sent
            network_bytes_recv = net_io.bytes_recv
            
            # Process count
            process_count = len(psutil.pids())
            
            # GPU metrics
            gpu_percent = 0.0
            vram_used_mb = 0
            vram_total_mb = 0
            vram_percent = 0.0
            
            if TORCH_AVAILABLE and torch.cuda.is_available():
                try:
                    vram_used_mb = torch.cuda.memory_allocated() // (1024**2)
                    vram_total_mb = torch.cuda.get_device_properties(0).total_memory // (1024**2)
                    vram_percent = (vram_used_mb / vram_total_mb) * 100 if vram_total_mb > 0 else 0.0
                except Exception as e:
                    self.logger.debug(f"Failed to collect GPU metrics: {e}")
            
            return ResourceMetrics(
                timestamp=timestamp,
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_used_gb=memory_used_gb,
                memory_total_gb=memory_total_gb,
                disk_usage_percent=disk_usage_percent,
                disk_free_gb=disk_free_gb,
                gpu_percent=gpu_percent,
                vram_used_mb=vram_used_mb,
                vram_total_mb=vram_total_mb,
                vram_percent=vram_percent,
                network_bytes_sent=network_bytes_sent,
                network_bytes_recv=network_bytes_recv,
                process_count=process_count
            )
            
        except Exception as e:
            self.logger.error(f"Failed to collect resource metrics: {e}")
            return ResourceMetrics(
                timestamp=datetime.now(),
                cpu_percent=0.0, memory_percent=0.0, memory_used_gb=0.0,
                memory_total_gb=0.0, disk_usage_percent=0.0, disk_free_gb=0.0,
                gpu_percent=0.0, vram_used_mb=0, vram_total_mb=0, vram_percent=0.0,
                network_bytes_sent=0, network_bytes_recv=0, process_count=0
            )
    
    def _check_component_health(self) -> List[ComponentHealth]:
        """Check health of monitored components"""
        component_health = []
        
        for component_name, component_data in self.monitored_components.items():
            try:
                # Simulate component health check (in real implementation, this would ping actual components)
                start_time = time.time()
                health_status = self._ping_component(component_name)
                response_time = (time.time() - start_time) * 1000  # Convert to ms
                
                # Store response time
                component_data["response_times"].append(response_time)
                
                # Calculate performance score based on response time and error history
                avg_response_time = statistics.mean(component_data["response_times"]) if component_data["response_times"] else 0
                performance_score = max(0, 100 - (avg_response_time / 10))  # Simple scoring
                
                # Determine health status
                if response_time > self.thresholds["response_time_critical"]:
                    status = HealthStatus.UNHEALTHY
                elif response_time > self.thresholds["response_time_warning"]:
                    status = HealthStatus.DEGRADED
                else:
                    status = HealthStatus.HEALTHY
                
                # Count recent errors for this component
                error_count = self._count_component_errors(component_name, timedelta(minutes=30))
                
                health = ComponentHealth(
                    component_name=component_name,
                    status=status,
                    response_time_ms=response_time,
                    error_count=error_count,
                    last_error=self._get_last_component_error(component_name),
                    uptime_seconds=self._get_component_uptime(component_name),
                    performance_score=performance_score
                )
                
                component_health.append(health)
                self.component_health_history[component_name].append(health)
                
            except Exception as e:
                self.logger.error(f"Failed to check health of {component_name}: {e}")
                health = ComponentHealth(
                    component_name=component_name,
                    status=HealthStatus.UNKNOWN,
                    response_time_ms=0.0,
                    error_count=0,
                    last_error=str(e),
                    uptime_seconds=0.0,
                    performance_score=0.0
                )
                component_health.append(health)
        
        return component_health
    
    def _ping_component(self, component_name: str) -> bool:
        """Ping component to check if it's responsive (placeholder implementation)"""
        # In a real implementation, this would actually test component responsiveness
        # For now, we'll simulate based on system load
        try:
            if len(self.metrics_history) > 0:
                latest_metrics = self.metrics_history[-1]
                # Simulate component being less responsive under high load
                if latest_metrics.cpu_percent > 90 or latest_metrics.memory_percent > 90:
                    time.sleep(0.1)  # Simulate slower response
                return True
            return True
        except Exception:
            return False
    
    def _count_component_errors(self, component_name: str, time_window: timedelta) -> int:
        """Count errors for a specific component within time window"""
        cutoff_time = datetime.now() - time_window
        count = 0
        
        for error_time in self.error_patterns.get(component_name, []):
            if error_time > cutoff_time:
                count += 1
        
        return count
    
    def _get_last_component_error(self, component_name: str) -> Optional[str]:
        """Get last error message for component"""
        # This would typically query error logs or error tracking system
        # For now, return None as placeholder
        return None
    
    def _get_component_uptime(self, component_name: str) -> float:
        """Get component uptime in seconds"""
        # This would typically track when components were started
        # For now, return a placeholder value
        return 3600.0  # 1 hour placeholder
    
    def _check_thresholds(self, metrics: ResourceMetrics, component_health: List[ComponentHealth]) -> List[Alert]:
        """Check metrics against thresholds and generate alerts"""
        alerts = []
        
        # CPU threshold checks
        if metrics.cpu_percent > self.thresholds["cpu_critical"]:
            alerts.append(Alert(
                level=AlertLevel.CRITICAL,
                message=f"Critical CPU usage: {metrics.cpu_percent:.1f}%",
                metric_name="cpu_percent",
                current_value=metrics.cpu_percent,
                threshold_value=self.thresholds["cpu_critical"]
            ))
        elif metrics.cpu_percent > self.thresholds["cpu_warning"]:
            alerts.append(Alert(
                level=AlertLevel.WARNING,
                message=f"High CPU usage: {metrics.cpu_percent:.1f}%",
                metric_name="cpu_percent",
                current_value=metrics.cpu_percent,
                threshold_value=self.thresholds["cpu_warning"]
            ))
        
        # Memory threshold checks
        if metrics.memory_percent > self.thresholds["memory_critical"]:
            alerts.append(Alert(
                level=AlertLevel.CRITICAL,
                message=f"Critical memory usage: {metrics.memory_percent:.1f}%",
                metric_name="memory_percent",
                current_value=metrics.memory_percent,
                threshold_value=self.thresholds["memory_critical"]
            ))
        elif metrics.memory_percent > self.thresholds["memory_warning"]:
            alerts.append(Alert(
                level=AlertLevel.WARNING,
                message=f"High memory usage: {metrics.memory_percent:.1f}%",
                metric_name="memory_percent",
                current_value=metrics.memory_percent,
                threshold_value=self.thresholds["memory_warning"]
            ))
        
        # VRAM threshold checks
        if metrics.vram_percent > self.thresholds["vram_critical"]:
            alerts.append(Alert(
                level=AlertLevel.CRITICAL,
                message=f"Critical VRAM usage: {metrics.vram_percent:.1f}%",
                metric_name="vram_percent",
                current_value=metrics.vram_percent,
                threshold_value=self.thresholds["vram_critical"]
            ))
        elif metrics.vram_percent > self.thresholds["vram_warning"]:
            alerts.append(Alert(
                level=AlertLevel.WARNING,
                message=f"High VRAM usage: {metrics.vram_percent:.1f}%",
                metric_name="vram_percent",
                current_value=metrics.vram_percent,
                threshold_value=self.thresholds["vram_warning"]
            ))
        
        # Disk threshold checks
        if metrics.disk_usage_percent > self.thresholds["disk_critical"]:
            alerts.append(Alert(
                level=AlertLevel.CRITICAL,
                message=f"Critical disk usage: {metrics.disk_usage_percent:.1f}%",
                metric_name="disk_usage_percent",
                current_value=metrics.disk_usage_percent,
                threshold_value=self.thresholds["disk_critical"]
            ))
        elif metrics.disk_usage_percent > self.thresholds["disk_warning"]:
            alerts.append(Alert(
                level=AlertLevel.WARNING,
                message=f"High disk usage: {metrics.disk_usage_percent:.1f}%",
                metric_name="disk_usage_percent",
                current_value=metrics.disk_usage_percent,
                threshold_value=self.thresholds["disk_warning"]
            ))
        
        # Component health alerts
        for health in component_health:
            if health.status == HealthStatus.UNHEALTHY:
                alerts.append(Alert(
                    level=AlertLevel.CRITICAL,
                    message=f"Component {health.component_name} is unhealthy (response time: {health.response_time_ms:.1f}ms)",
                    metric_name="component_health",
                    current_value=health.response_time_ms,
                    threshold_value=self.thresholds["response_time_critical"],
                    component=health.component_name
                ))
            elif health.status == HealthStatus.DEGRADED:
                alerts.append(Alert(
                    level=AlertLevel.WARNING,
                    message=f"Component {health.component_name} performance degraded (response time: {health.response_time_ms:.1f}ms)",
                    metric_name="component_health",
                    current_value=health.response_time_ms,
                    threshold_value=self.thresholds["response_time_warning"],
                    component=health.component_name
                ))
        
        return alerts
    
    def _detect_error_patterns(self) -> List[ErrorPattern]:
        """Detect error patterns from historical data"""
        patterns = []
        
        try:
            # Analyze error frequency patterns
            for pattern_type, occurrences in self.error_patterns.items():
                if len(occurrences) < 2:
                    continue
                
                # Filter to recent occurrences within detection window
                cutoff_time = datetime.now() - self.pattern_detection_window
                recent_occurrences = [t for t in occurrences if t > cutoff_time]
                
                if len(recent_occurrences) >= 3:  # Pattern threshold
                    # Calculate trend
                    if len(recent_occurrences) >= 6:
                        first_half = recent_occurrences[:len(recent_occurrences)//2]
                        second_half = recent_occurrences[len(recent_occurrences)//2:]
                        
                        if len(second_half) > len(first_half):
                            trend = "increasing"
                        elif len(second_half) < len(first_half):
                            trend = "decreasing"
                        else:
                            trend = "stable"
                    else:
                        trend = "stable"
                    
                    # Calculate prediction confidence based on pattern consistency
                    time_intervals = []
                    for i in range(1, len(recent_occurrences)):
                        interval = (recent_occurrences[i] - recent_occurrences[i-1]).total_seconds()
                        time_intervals.append(interval)
                    
                    if time_intervals:
                        interval_variance = statistics.variance(time_intervals) if len(time_intervals) > 1 else 0
                        # Lower variance = higher confidence
                        confidence = max(0.1, 1.0 - (interval_variance / 3600))  # Normalize by hour
                    else:
                        confidence = 0.5
                    
                    pattern = ErrorPattern(
                        pattern_type=pattern_type,
                        frequency=len(recent_occurrences),
                        first_occurrence=min(recent_occurrences),
                        last_occurrence=max(recent_occurrences),
                        components_affected=[pattern_type],  # Simplified
                        severity_trend=trend,
                        prediction_confidence=min(1.0, confidence)
                    )
                    patterns.append(pattern)
        
        except Exception as e:
            self.logger.error(f"Error in pattern detection: {e}")
        
        return patterns
    
    def _detect_potential_issues(self) -> List[PotentialIssue]:
        """Detect potential issues using predictive analysis"""
        issues = []
        
        try:
            if len(self.metrics_history) < 10:  # Need sufficient history
                return issues
            
            # Analyze resource usage trends
            recent_metrics = list(self.metrics_history)[-10:]  # Last 10 measurements
            
            # CPU trend analysis
            cpu_values = [m.cpu_percent for m in recent_metrics]
            if len(cpu_values) >= 5:
                cpu_trend = self._calculate_trend(cpu_values)
                if cpu_trend > 2.0 and cpu_values[-1] > 70:  # Increasing trend above 70%
                    time_to_critical = self._estimate_time_to_threshold(
                        cpu_values, self.thresholds["cpu_critical"], cpu_trend
                    )
                    issues.append(PotentialIssue(
                        issue_type="cpu_exhaustion",
                        probability=min(0.9, (cpu_values[-1] / 100) + (cpu_trend / 10)),
                        estimated_time_to_failure=time_to_critical,
                        affected_components=["system"],
                        recommended_actions=[
                            "Monitor CPU-intensive processes",
                            "Consider scaling resources",
                            "Review process optimization"
                        ],
                        confidence_level=0.7
                    ))
            
            # Memory trend analysis
            memory_values = [m.memory_percent for m in recent_metrics]
            if len(memory_values) >= 5:
                memory_trend = self._calculate_trend(memory_values)
                if memory_trend > 1.5 and memory_values[-1] > 70:
                    time_to_critical = self._estimate_time_to_threshold(
                        memory_values, self.thresholds["memory_critical"], memory_trend
                    )
                    issues.append(PotentialIssue(
                        issue_type="memory_exhaustion",
                        probability=min(0.9, (memory_values[-1] / 100) + (memory_trend / 15)),
                        estimated_time_to_failure=time_to_critical,
                        affected_components=["system"],
                        recommended_actions=[
                            "Clear memory caches",
                            "Restart memory-intensive processes",
                            "Monitor for memory leaks"
                        ],
                        confidence_level=0.75
                    ))
            
            # VRAM trend analysis
            vram_values = [m.vram_percent for m in recent_metrics if m.vram_percent > 0]
            if len(vram_values) >= 5:
                vram_trend = self._calculate_trend(vram_values)
                if vram_trend > 1.0 and vram_values[-1] > 60:
                    time_to_critical = self._estimate_time_to_threshold(
                        vram_values, self.thresholds["vram_critical"], vram_trend
                    )
                    issues.append(PotentialIssue(
                        issue_type="vram_exhaustion",
                        probability=min(0.9, (vram_values[-1] / 100) + (vram_trend / 20)),
                        estimated_time_to_failure=time_to_critical,
                        affected_components=["model_downloader", "gpu_operations"],
                        recommended_actions=[
                            "Clear GPU cache",
                            "Reduce batch sizes",
                            "Monitor GPU memory usage"
                        ],
                        confidence_level=0.8
                    ))
            
            # Disk space trend analysis
            disk_values = [m.disk_usage_percent for m in recent_metrics]
            if len(disk_values) >= 5:
                disk_trend = self._calculate_trend(disk_values)
                if disk_trend > 0.5 and disk_values[-1] > 75:
                    time_to_critical = self._estimate_time_to_threshold(
                        disk_values, self.thresholds["disk_critical"], disk_trend
                    )
                    issues.append(PotentialIssue(
                        issue_type="disk_space_exhaustion",
                        probability=min(0.9, (disk_values[-1] / 100) + (disk_trend / 25)),
                        estimated_time_to_failure=time_to_critical,
                        affected_components=["system", "model_downloader"],
                        recommended_actions=[
                            "Clean temporary files",
                            "Archive old logs",
                            "Free up disk space"
                        ],
                        confidence_level=0.85
                    ))
        
        except Exception as e:
            self.logger.error(f"Error in potential issue detection: {e}")
        
        return issues
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend slope for a series of values"""
        if len(values) < 2:
            return 0.0
        
        try:
            # Simple linear regression slope calculation
            n = len(values)
            x_values = list(range(n))
            
            sum_x = sum(x_values)
            sum_y = sum(values)
            sum_xy = sum(x * y for x, y in zip(x_values, values))
            sum_x2 = sum(x * x for x in x_values)
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            return slope
        except (ZeroDivisionError, ValueError):
            return 0.0
    
    def _estimate_time_to_threshold(self, values: List[float], threshold: float, trend: float) -> Optional[timedelta]:
        """Estimate time until threshold is reached based on trend"""
        if trend <= 0 or len(values) == 0:
            return None
        
        try:
            current_value = values[-1]
            if current_value >= threshold:
                return timedelta(seconds=0)
            
            # Calculate time based on trend (trend is per measurement interval)
            remaining = threshold - current_value
            intervals_to_threshold = remaining / trend
            seconds_to_threshold = intervals_to_threshold * self.monitoring_interval
            
            if seconds_to_threshold > 0 and seconds_to_threshold < 86400:  # Within 24 hours
                return timedelta(seconds=seconds_to_threshold)
            
            return None
        except (ZeroDivisionError, ValueError):
            return None
    
    def _generate_health_report(self, metrics: ResourceMetrics, component_health: List[ComponentHealth],
                              alerts: List[Alert], error_patterns: List[ErrorPattern],
                              potential_issues: List[PotentialIssue]) -> HealthReport:
        """Generate comprehensive health report"""
        
        # Determine overall health status
        overall_health = HealthStatus.HEALTHY
        
        # Check for critical alerts
        if any(alert.level == AlertLevel.CRITICAL for alert in alerts):
            overall_health = HealthStatus.UNHEALTHY
        elif any(alert.level == AlertLevel.WARNING for alert in alerts):
            overall_health = HealthStatus.DEGRADED
        
        # Check component health
        unhealthy_components = [h for h in component_health if h.status == HealthStatus.UNHEALTHY]
        degraded_components = [h for h in component_health if h.status == HealthStatus.DEGRADED]
        
        if unhealthy_components:
            overall_health = HealthStatus.UNHEALTHY
        elif degraded_components and overall_health == HealthStatus.HEALTHY:
            overall_health = HealthStatus.DEGRADED
        
        # Generate performance trends
        performance_trends = self._calculate_performance_trends()
        
        # Generate recommendations
        recommendations = self._generate_recommendations(alerts, potential_issues, component_health)
        
        return HealthReport(
            timestamp=datetime.now(),
            overall_health=overall_health,
            resource_metrics=metrics,
            component_health=component_health,
            active_alerts=alerts,
            error_patterns=error_patterns,
            potential_issues=potential_issues,
            performance_trends=performance_trends,
            recommendations=recommendations
        )
    
    def _calculate_performance_trends(self) -> Dict[str, List[float]]:
        """Calculate performance trends from historical data"""
        trends = {}
        
        if len(self.metrics_history) >= 10:
            recent_metrics = list(self.metrics_history)[-10:]
            
            trends["cpu_percent"] = [m.cpu_percent for m in recent_metrics]
            trends["memory_percent"] = [m.memory_percent for m in recent_metrics]
            trends["vram_percent"] = [m.vram_percent for m in recent_metrics if m.vram_percent > 0]
            trends["disk_usage_percent"] = [m.disk_usage_percent for m in recent_metrics]
        
        return trends
    
    def _generate_recommendations(self, alerts: List[Alert], potential_issues: List[PotentialIssue],
                                component_health: List[ComponentHealth]) -> List[str]:
        """Generate actionable recommendations based on current state"""
        recommendations = []
        
        # Recommendations based on alerts
        critical_alerts = [a for a in alerts if a.level == AlertLevel.CRITICAL]
        if critical_alerts:
            recommendations.append("Immediate attention required: Critical system alerts detected")
            
            cpu_alerts = [a for a in critical_alerts if a.metric_name == "cpu_percent"]
            if cpu_alerts:
                recommendations.append("High CPU usage detected - consider stopping non-essential processes")
            
            memory_alerts = [a for a in critical_alerts if a.metric_name == "memory_percent"]
            if memory_alerts:
                recommendations.append("Critical memory usage - restart applications or clear caches")
            
            vram_alerts = [a for a in critical_alerts if a.metric_name == "vram_percent"]
            if vram_alerts:
                recommendations.append("GPU memory critical - clear GPU cache and reduce model sizes")
        
        # Recommendations based on potential issues
        for issue in potential_issues:
            if issue.probability > 0.7:
                recommendations.extend(issue.recommended_actions)
        
        # Recommendations based on component health
        unhealthy_components = [h for h in component_health if h.status == HealthStatus.UNHEALTHY]
        if unhealthy_components:
            component_names = [h.component_name for h in unhealthy_components]
            recommendations.append(f"Restart or investigate unhealthy components: {', '.join(component_names)}")
        
        # General maintenance recommendations
        if len(self.metrics_history) > 100:
            avg_cpu = statistics.mean([m.cpu_percent for m in list(self.metrics_history)[-50:]])
            if avg_cpu > 60:
                recommendations.append("Consider system optimization - sustained high CPU usage detected")
        
        return list(set(recommendations))  # Remove duplicates
    
    def _cleanup_old_data(self):
        """Clean up old monitoring data"""
        try:
            # Clean up error patterns older than detection window
            cutoff_time = datetime.now() - self.pattern_detection_window
            for pattern_type in list(self.error_patterns.keys()):
                self.error_patterns[pattern_type] = [
                    t for t in self.error_patterns[pattern_type] if t > cutoff_time
                ]
                if not self.error_patterns[pattern_type]:
                    del self.error_patterns[pattern_type]
        
        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {e}")
    
    def _trigger_alert_callbacks(self, alert: Alert):
        """Trigger registered alert callbacks"""
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")
    
    def _trigger_health_callbacks(self, health_report: HealthReport):
        """Trigger registered health callbacks"""
        for callback in self.health_callbacks:
            try:
                callback(health_report)
            except Exception as e:
                self.logger.error(f"Error in health callback: {e}")
    
    # Public interface methods
    
    def check_component_health(self, component: str) -> ComponentHealth:
        """Check health of a specific component"""
        if component not in self.monitored_components:
            return ComponentHealth(
                component_name=component,
                status=HealthStatus.UNKNOWN,
                response_time_ms=0.0,
                error_count=0,
                last_error="Component not monitored",
                uptime_seconds=0.0,
                performance_score=0.0
            )
        
        # Get latest health data for component
        if component in self.component_health_history and self.component_health_history[component]:
            return self.component_health_history[component][-1]
        
        # Perform fresh health check
        start_time = time.time()
        responsive = self._ping_component(component)
        response_time = (time.time() - start_time) * 1000
        
        status = HealthStatus.HEALTHY if responsive else HealthStatus.UNHEALTHY
        error_count = self._count_component_errors(component, timedelta(minutes=30))
        
        return ComponentHealth(
            component_name=component,
            status=status,
            response_time_ms=response_time,
            error_count=error_count,
            last_error=self._get_last_component_error(component),
            uptime_seconds=self._get_component_uptime(component),
            performance_score=100.0 if responsive else 0.0
        )
    
    def detect_potential_issues(self) -> List[PotentialIssue]:
        """Detect potential issues using current data"""
        return self._detect_potential_issues()
    
    def generate_health_report(self) -> HealthReport:
        """Generate current health report"""
        metrics = self._collect_resource_metrics()
        component_health = self._check_component_health()
        alerts = self._check_thresholds(metrics, component_health)
        error_patterns = self._detect_error_patterns()
        potential_issues = self._detect_potential_issues()
        
        return self._generate_health_report(
            metrics, component_health, alerts, error_patterns, potential_issues
        )
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Register callback for alert notifications"""
        self.alert_callbacks.append(callback)
    
    def add_health_callback(self, callback: Callable[[HealthReport], None]):
        """Register callback for health report updates"""
        self.health_callbacks.append(callback)
    
    def record_error(self, component: str, error_message: str):
        """Record an error for pattern detection"""
        self.error_patterns[component].append(datetime.now())
        self.error_history.append({
            "component": component,
            "message": error_message,
            "timestamp": datetime.now()
        })
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status"""
        return {
            "is_monitoring": self.is_monitoring,
            "monitoring_interval": self.monitoring_interval,
            "metrics_history_size": len(self.metrics_history),
            "monitored_components": list(self.monitored_components.keys()),
            "alert_history_size": len(self.alert_history),
            "error_patterns_count": len(self.error_patterns)
        }