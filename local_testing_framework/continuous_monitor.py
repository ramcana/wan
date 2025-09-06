"""
Continuous monitoring system for real-time performance tracking and system stability
"""

import json
import time
import threading
import gc
import os
import subprocess
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import psutil
import logging

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .models.test_results import ResourceMetrics, ValidationStatus, TestStatus
from .models.configuration import TestConfiguration


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    """System alert data structure"""
    level: AlertLevel
    message: str
    metric_name: str
    current_value: float
    threshold_value: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "level": self.level.value,
            "message": self.message,
            "metric_name": self.metric_name,
            "current_value": self.current_value,
            "threshold_value": self.threshold_value,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class ProgressInfo:
    """Progress tracking information"""
    current_step: int
    total_steps: int
    percentage: float
    eta_seconds: Optional[float] = None
    start_time: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "percentage": self.percentage,
            "eta_seconds": self.eta_seconds,
            "start_time": self.start_time.isoformat()
        }


@dataclass
class DiagnosticSnapshot:
    """System diagnostic snapshot"""
    timestamp: datetime
    gpu_memory_state: Dict[str, Any]
    system_processes: List[Dict[str, Any]]
    system_logs: List[str]
    disk_usage: Dict[str, Any]
    network_stats: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "gpu_memory_state": self.gpu_memory_state,
            "system_processes": self.system_processes,
            "system_logs": self.system_logs,
            "disk_usage": self.disk_usage,
            "network_stats": self.network_stats
        }


@dataclass
class RecoveryAction:
    """Recovery action result"""
    action_name: str
    success: bool
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "action_name": self.action_name,
            "success": self.success,
            "message": self.message,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class MonitoringSession:
    """Monitoring session data"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    metrics_history: List[ResourceMetrics] = field(default_factory=list)
    alerts_history: List[Alert] = field(default_factory=list)
    diagnostic_snapshots: List[DiagnosticSnapshot] = field(default_factory=list)
    recovery_actions: List[RecoveryAction] = field(default_factory=list)
    progress_info: Optional[ProgressInfo] = None
    is_active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        result = {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "is_active": self.is_active,
            "metrics_count": len(self.metrics_history),
            "alerts_count": len(self.alerts_history),
            "snapshots_count": len(self.diagnostic_snapshots),
            "recovery_actions_count": len(self.recovery_actions)
        }
        
        if self.end_time:
            result["end_time"] = self.end_time.isoformat()
            result["duration_seconds"] = (self.end_time - self.start_time).total_seconds()
            
        if self.progress_info:
            result["progress_info"] = self.progress_info.to_dict()
            
        return result


class ContinuousMonitor:
    """
    Real-time monitoring system with configurable intervals and thresholds
    """
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize continuous monitor with configuration"""
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = self._setup_logging()
        
        # Monitoring configuration
        self.refresh_interval = self.config.get("system", {}).get("stats_refresh_interval", 5)
        self.vram_warning_threshold = self.config.get("performance", {}).get("vram_warning_threshold", 0.9)
        self.cpu_warning_threshold = 80.0  # Default from requirements
        
        # Monitoring state
        self.current_session: Optional[MonitoringSession] = None
        self.monitoring_thread: Optional[threading.Thread] = None
        self.stop_monitoring = threading.Event()
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        
        # Progress tracking
        self.progress_callbacks: List[Callable[[ProgressInfo], None]] = []
        
        # System stability monitoring
        self.stability_check_interval = 60  # Check stability every minute
        self.last_stability_check = datetime.now()
        self.auto_recovery_enabled = True
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"Failed to load config from {self.config_path}: {e}")
            return {}
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for monitoring system"""
        logger = logging.getLogger("continuous_monitor")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def _collect_resource_metrics(self) -> ResourceMetrics:
        """Collect current system resource metrics"""
        try:
            # CPU and memory metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_gb = memory.used / (1024**3)
            memory_total_gb = memory.total / (1024**3)
            
            # GPU metrics (if available)
            gpu_percent = 0.0
            vram_used_mb = 0
            vram_total_mb = 0
            vram_percent = 0.0
            
            if TORCH_AVAILABLE and torch.cuda.is_available():
                try:
                    # Get GPU utilization (approximation)
                    gpu_percent = 0.0  # Would need nvidia-ml-py for accurate GPU utilization
                    
                    # VRAM metrics
                    vram_used_mb = torch.cuda.memory_allocated() // (1024**2)
                    vram_total_mb = torch.cuda.get_device_properties(0).total_memory // (1024**2)
                    vram_percent = (vram_used_mb / vram_total_mb) * 100 if vram_total_mb > 0 else 0.0
                    
                except Exception as e:
                    self.logger.warning(f"Failed to collect GPU metrics: {e}")
            
            return ResourceMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_used_gb=memory_used_gb,
                memory_total_gb=memory_total_gb,
                gpu_percent=gpu_percent,
                vram_used_mb=vram_used_mb,
                vram_total_mb=vram_total_mb,
                vram_percent=vram_percent
            )
            
        except Exception as e:
            self.logger.error(f"Failed to collect resource metrics: {e}")
            return ResourceMetrics(
                cpu_percent=0.0, memory_percent=0.0, memory_used_gb=0.0,
                memory_total_gb=0.0, gpu_percent=0.0, vram_used_mb=0,
                vram_total_mb=0, vram_percent=0.0
            )
    
    def _check_thresholds(self, metrics: ResourceMetrics) -> List[Alert]:
        """Check metrics against configured thresholds and generate alerts"""
        alerts = []
        
        # CPU threshold check
        if metrics.cpu_percent > self.cpu_warning_threshold:
            alerts.append(Alert(
                level=AlertLevel.WARNING,
                message=f"High CPU usage detected: {metrics.cpu_percent:.1f}%",
                metric_name="cpu_percent",
                current_value=metrics.cpu_percent,
                threshold_value=self.cpu_warning_threshold
            ))
        
        # VRAM threshold check
        vram_threshold_percent = self.vram_warning_threshold * 100
        if metrics.vram_percent > vram_threshold_percent:
            alerts.append(Alert(
                level=AlertLevel.WARNING,
                message=f"High VRAM usage detected: {metrics.vram_percent:.1f}%",
                metric_name="vram_percent",
                current_value=metrics.vram_percent,
                threshold_value=vram_threshold_percent
            ))
        
        # Memory threshold check (using same threshold as CPU for consistency)
        if metrics.memory_percent > self.cpu_warning_threshold:
            alerts.append(Alert(
                level=AlertLevel.WARNING,
                message=f"High memory usage detected: {metrics.memory_percent:.1f}%",
                metric_name="memory_percent",
                current_value=metrics.memory_percent,
                threshold_value=self.cpu_warning_threshold
            ))
        
        return alerts
    
    def _monitoring_loop(self):
        """Main monitoring loop running in separate thread"""
        self.logger.info(f"Starting monitoring loop with {self.refresh_interval}s interval")
        
        while not self.stop_monitoring.is_set():
            try:
                if self.current_session and self.current_session.is_active:
                    # Collect metrics
                    metrics = self._collect_resource_metrics()
                    self.current_session.metrics_history.append(metrics)
                    
                    # Check thresholds and generate alerts
                    alerts = self._check_thresholds(metrics)
                    for alert in alerts:
                        self.current_session.alerts_history.append(alert)
                        self._trigger_alert_callbacks(alert)
                    
                    # Periodic stability check
                    now = datetime.now()
                    if (now - self.last_stability_check).total_seconds() >= self.stability_check_interval:
                        stability_alerts = self._check_system_stability()
                        for alert in stability_alerts:
                            self.current_session.alerts_history.append(alert)
                            self._trigger_alert_callbacks(alert)
                        self.last_stability_check = now
                    
                    # Log current status
                    self.logger.debug(
                        f"Metrics - CPU: {metrics.cpu_percent:.1f}%, "
                        f"Memory: {metrics.memory_percent:.1f}%, "
                        f"VRAM: {metrics.vram_percent:.1f}%"
                    )
                
                # Wait for next interval
                self.stop_monitoring.wait(self.refresh_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.refresh_interval)
    
    def _trigger_alert_callbacks(self, alert: Alert):
        """Trigger registered alert callbacks"""
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")
    
    def _trigger_progress_callbacks(self, progress: ProgressInfo):
        """Trigger registered progress callbacks"""
        for callback in self.progress_callbacks:
            try:
                callback(progress)
            except Exception as e:
                self.logger.error(f"Error in progress callback: {e}")
    
    def start_monitoring(self, session_id: str) -> MonitoringSession:
        """Start a new monitoring session"""
        if self.current_session and self.current_session.is_active:
            self.logger.warning("Stopping existing monitoring session")
            self.stop_monitoring_session()
        
        self.current_session = MonitoringSession(
            session_id=session_id,
            start_time=datetime.now()
        )
        
        # Start monitoring thread
        self.stop_monitoring.clear()
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            name=f"monitor-{session_id}"
        )
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        self.logger.info(f"Started monitoring session: {session_id}")
        return self.current_session
    
    def stop_monitoring_session(self) -> Optional[MonitoringSession]:
        """Stop current monitoring session"""
        if not self.current_session:
            return None
        
        # Signal stop and wait for thread
        self.stop_monitoring.set()
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
        
        # Finalize session
        self.current_session.is_active = False
        self.current_session.end_time = datetime.now()
        
        session = self.current_session
        self.current_session = None
        
        self.logger.info(f"Stopped monitoring session: {session.session_id}")
        return session
    
    def update_progress(self, current_step: int, total_steps: int, 
                       start_time: Optional[datetime] = None) -> ProgressInfo:
        """Update progress tracking with ETA calculation"""
        if not self.current_session:
            raise RuntimeError("No active monitoring session")
        
        percentage = (current_step / total_steps) * 100 if total_steps > 0 else 0.0
        
        # Calculate ETA based on elapsed time and progress
        eta_seconds = None
        if start_time and current_step > 0 and percentage > 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            estimated_total = elapsed * (100 / percentage)
            eta_seconds = estimated_total - elapsed
        
        progress = ProgressInfo(
            current_step=current_step,
            total_steps=total_steps,
            percentage=percentage,
            eta_seconds=eta_seconds,
            start_time=start_time or datetime.now()
        )
        
        self.current_session.progress_info = progress
        self._trigger_progress_callbacks(progress)
        
        return progress
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Register callback for alert notifications"""
        self.alert_callbacks.append(callback)
    
    def add_progress_callback(self, callback: Callable[[ProgressInfo], None]):
        """Register callback for progress updates"""
        self.progress_callbacks.append(callback)
    
    def get_current_metrics(self) -> Optional[ResourceMetrics]:
        """Get current resource metrics without starting monitoring"""
        return self._collect_resource_metrics()
    
    def get_session_summary(self) -> Optional[Dict[str, Any]]:
        """Get summary of current monitoring session"""
        if not self.current_session:
            return None
        
        summary = self.current_session.to_dict()
        
        # Add metrics summary
        if self.current_session.metrics_history:
            latest_metrics = self.current_session.metrics_history[-1]
            summary["latest_metrics"] = latest_metrics.to_dict()
            
            # Calculate averages
            cpu_avg = sum(m.cpu_percent for m in self.current_session.metrics_history) / len(self.current_session.metrics_history)
            memory_avg = sum(m.memory_percent for m in self.current_session.metrics_history) / len(self.current_session.metrics_history)
            vram_avg = sum(m.vram_percent for m in self.current_session.metrics_history) / len(self.current_session.metrics_history)
            
            summary["averages"] = {
                "cpu_percent": cpu_avg,
                "memory_percent": memory_avg,
                "vram_percent": vram_avg
            }
        
        # Add alerts summary
        if self.current_session.alerts_history:
            alert_counts = {}
            for alert in self.current_session.alerts_history:
                level = alert.level.value
                alert_counts[level] = alert_counts.get(level, 0) + 1
            summary["alert_counts"] = alert_counts
        
        return summary
    
    def _capture_diagnostic_snapshot(self) -> DiagnosticSnapshot:
        """Capture comprehensive system diagnostic snapshot"""
        try:
            # GPU memory state
            gpu_memory_state = {}
            if TORCH_AVAILABLE and torch.cuda.is_available():
                try:
                    gpu_memory_state = {
                        "allocated_mb": torch.cuda.memory_allocated() // (1024**2),
                        "reserved_mb": torch.cuda.memory_reserved() // (1024**2),
                        "max_allocated_mb": torch.cuda.max_memory_allocated() // (1024**2),
                        "max_reserved_mb": torch.cuda.max_memory_reserved() // (1024**2),
                        "device_count": torch.cuda.device_count(),
                        "current_device": torch.cuda.current_device()
                    }
                except Exception as e:
                    gpu_memory_state = {"error": str(e)}
            
            # System processes (top 10 by memory usage)
            system_processes = []
            try:
                processes = []
                for proc in psutil.process_iter(['pid', 'name', 'memory_percent', 'cpu_percent']):
                    try:
                        processes.append(proc.info)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                
                # Sort by memory usage and take top 10
                processes.sort(key=lambda x: x.get('memory_percent', 0), reverse=True)
                system_processes = processes[:10]
            except Exception as e:
                system_processes = [{"error": str(e)}]
            
            # System logs (recent error logs)
            system_logs = []
            try:
                # Try to read recent error logs
                log_files = ["wan22_errors.log", "wan22_ui.log"]
                for log_file in log_files:
                    if os.path.exists(log_file):
                        try:
                            with open(log_file, 'r') as f:
                                lines = f.readlines()
                                # Get last 10 lines
                                recent_lines = lines[-10:] if len(lines) > 10 else lines
                                system_logs.extend([f"{log_file}: {line.strip()}" for line in recent_lines])
                        except Exception as e:
                            system_logs.append(f"Error reading {log_file}: {e}")
            except Exception as e:
                system_logs = [f"Error collecting logs: {e}"]
            
            # Disk usage
            disk_usage = {}
            try:
                disk_usage = {
                    "total_gb": shutil.disk_usage(".").total / (1024**3),
                    "used_gb": (shutil.disk_usage(".").total - shutil.disk_usage(".").free) / (1024**3),
                    "free_gb": shutil.disk_usage(".").free / (1024**3),
                    "usage_percent": ((shutil.disk_usage(".").total - shutil.disk_usage(".").free) / shutil.disk_usage(".").total) * 100
                }
            except Exception as e:
                disk_usage = {"error": str(e)}
            
            # Network stats
            network_stats = {}
            try:
                net_io = psutil.net_io_counters()
                network_stats = {
                    "bytes_sent": net_io.bytes_sent,
                    "bytes_recv": net_io.bytes_recv,
                    "packets_sent": net_io.packets_sent,
                    "packets_recv": net_io.packets_recv
                }
            except Exception as e:
                network_stats = {"error": str(e)}
            
            return DiagnosticSnapshot(
                timestamp=datetime.now(),
                gpu_memory_state=gpu_memory_state,
                system_processes=system_processes,
                system_logs=system_logs,
                disk_usage=disk_usage,
                network_stats=network_stats
            )
            
        except Exception as e:
            self.logger.error(f"Failed to capture diagnostic snapshot: {e}")
            return DiagnosticSnapshot(
                timestamp=datetime.now(),
                gpu_memory_state={"error": str(e)},
                system_processes=[],
                system_logs=[],
                disk_usage={},
                network_stats={}
            )
    
    def _attempt_gpu_cache_recovery(self) -> RecoveryAction:
        """Attempt to clear GPU cache for recovery"""
        try:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                return RecoveryAction(
                    action_name="clear_gpu_cache",
                    success=True,
                    message="Successfully cleared GPU cache"
                )
            else:
                return RecoveryAction(
                    action_name="clear_gpu_cache",
                    success=False,
                    message="CUDA not available for cache clearing"
                )
        except Exception as e:
            return RecoveryAction(
                action_name="clear_gpu_cache",
                success=False,
                message=f"Failed to clear GPU cache: {e}"
            )
    
    def _attempt_memory_cleanup(self) -> RecoveryAction:
        """Attempt to clean up system memory"""
        try:
            # Force garbage collection
            collected = gc.collect()
            
            # Clear temporary files in outputs directory
            temp_files_cleared = 0
            if os.path.exists("outputs"):
                for root, dirs, files in os.walk("outputs"):
                    for file in files:
                        if file.endswith(('.tmp', '.temp')):
                            try:
                                os.remove(os.path.join(root, file))
                                temp_files_cleared += 1
                            except Exception:
                                pass
            
            return RecoveryAction(
                action_name="memory_cleanup",
                success=True,
                message=f"Collected {collected} objects, cleared {temp_files_cleared} temp files"
            )
        except Exception as e:
            return RecoveryAction(
                action_name="memory_cleanup",
                success=False,
                message=f"Failed memory cleanup: {e}"
            )
    
    def _attempt_service_restart(self) -> RecoveryAction:
        """Attempt to restart services (placeholder for actual service restart)"""
        try:
            # This would typically restart specific services
            # For now, we'll just log the attempt
            self.logger.info("Service restart would be attempted here")
            
            return RecoveryAction(
                action_name="service_restart",
                success=True,
                message="Service restart procedure initiated (placeholder)"
            )
        except Exception as e:
            return RecoveryAction(
                action_name="service_restart",
                success=False,
                message=f"Failed service restart: {e}"
            )
    
    def _check_system_stability(self) -> List[Alert]:
        """Check system stability and trigger recovery if needed"""
        stability_alerts = []
        
        try:
            current_metrics = self._collect_resource_metrics()
            
            # Check for critical resource usage
            critical_vram = current_metrics.vram_percent > 95.0
            critical_memory = current_metrics.memory_percent > 95.0
            critical_cpu = current_metrics.cpu_percent > 95.0
            
            if critical_vram or critical_memory or critical_cpu:
                # Capture diagnostic snapshot
                snapshot = self._capture_diagnostic_snapshot()
                if self.current_session:
                    self.current_session.diagnostic_snapshots.append(snapshot)
                
                # Create critical alert
                if critical_vram:
                    stability_alerts.append(Alert(
                        level=AlertLevel.CRITICAL,
                        message=f"Critical VRAM usage: {current_metrics.vram_percent:.1f}%",
                        metric_name="vram_percent",
                        current_value=current_metrics.vram_percent,
                        threshold_value=95.0
                    ))
                
                if critical_memory:
                    stability_alerts.append(Alert(
                        level=AlertLevel.CRITICAL,
                        message=f"Critical memory usage: {current_metrics.memory_percent:.1f}%",
                        metric_name="memory_percent",
                        current_value=current_metrics.memory_percent,
                        threshold_value=95.0
                    ))
                
                if critical_cpu:
                    stability_alerts.append(Alert(
                        level=AlertLevel.CRITICAL,
                        message=f"Critical CPU usage: {current_metrics.cpu_percent:.1f}%",
                        metric_name="cpu_percent",
                        current_value=current_metrics.cpu_percent,
                        threshold_value=95.0
                    ))
                
                # Attempt automatic recovery if enabled
                if self.auto_recovery_enabled:
                    recovery_actions = []
                    
                    if critical_vram:
                        recovery_actions.append(self._attempt_gpu_cache_recovery())
                    
                    if critical_memory:
                        recovery_actions.append(self._attempt_memory_cleanup())
                    
                    # Add recovery actions to session
                    if self.current_session:
                        self.current_session.recovery_actions.extend(recovery_actions)
                    
                    self.logger.info(f"Executed {len(recovery_actions)} recovery actions")
            
        except Exception as e:
            self.logger.error(f"Error in stability check: {e}")
            stability_alerts.append(Alert(
                level=AlertLevel.CRITICAL,
                message=f"System stability check failed: {e}",
                metric_name="system_stability",
                current_value=0.0,
                threshold_value=1.0
            ))
        
        return stability_alerts
    
    def force_diagnostic_snapshot(self) -> DiagnosticSnapshot:
        """Force capture of diagnostic snapshot"""
        snapshot = self._capture_diagnostic_snapshot()
        if self.current_session:
            self.current_session.diagnostic_snapshots.append(snapshot)
        return snapshot
    
    def trigger_recovery_procedures(self) -> List[RecoveryAction]:
        """Manually trigger recovery procedures"""
        recovery_actions = [
            self._attempt_gpu_cache_recovery(),
            self._attempt_memory_cleanup(),
            self._attempt_service_restart()
        ]
        
        if self.current_session:
            self.current_session.recovery_actions.extend(recovery_actions)
        
        return recovery_actions
    
    def generate_monitoring_report(self) -> Dict[str, Any]:
        """Generate comprehensive monitoring report with timeline and violations"""
        if not self.current_session:
            return {"error": "No active monitoring session"}
        
        session = self.current_session
        report = {
            "session_info": session.to_dict(),
            "timeline": [],
            "threshold_violations": [],
            "stability_events": [],
            "recovery_summary": {}
        }
        
        # Build timeline from metrics
        for metrics in session.metrics_history:
            report["timeline"].append({
                "timestamp": metrics.timestamp.isoformat(),
                "cpu_percent": metrics.cpu_percent,
                "memory_percent": metrics.memory_percent,
                "vram_percent": metrics.vram_percent
            })
        
        # Threshold violations
        for alert in session.alerts_history:
            if alert.level in [AlertLevel.WARNING, AlertLevel.CRITICAL]:
                report["threshold_violations"].append(alert.to_dict())
        
        # Stability events (diagnostic snapshots)
        for snapshot in session.diagnostic_snapshots:
            report["stability_events"].append({
                "timestamp": snapshot.timestamp.isoformat(),
                "gpu_memory_allocated_mb": snapshot.gpu_memory_state.get("allocated_mb", 0),
                "top_process": snapshot.system_processes[0] if snapshot.system_processes else None,
                "recent_logs_count": len(snapshot.system_logs)
            })
        
        # Recovery summary
        recovery_summary = {}
        for action in session.recovery_actions:
            action_name = action.action_name
            if action_name not in recovery_summary:
                recovery_summary[action_name] = {"attempts": 0, "successes": 0}
            recovery_summary[action_name]["attempts"] += 1
            if action.success:
                recovery_summary[action_name]["successes"] += 1
        
        report["recovery_summary"] = recovery_summary
        
        return report
    
    def cleanup_resources(self):
        """Clean up monitoring resources"""
        if self.current_session and self.current_session.is_active:
            self.stop_monitoring_session()
        
        # Clear callbacks
        self.alert_callbacks.clear()
        self.progress_callbacks.clear()
        
        self.logger.info("Monitoring resources cleaned up")