"""
WAN Model Monitoring Service

Provides comprehensive monitoring and health checking for deployed WAN models
in production environments, including performance metrics, resource usage,
and automated alerting.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import psutil
import torch


class HealthStatus(Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class HealthMetric:
    """A single health metric measurement"""
    name: str
    value: float
    unit: str
    timestamp: datetime
    status: HealthStatus
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None


@dataclass
class ModelHealthReport:
    """Health report for a specific model"""
    model_name: str
    deployment_id: str
    overall_status: HealthStatus
    timestamp: datetime
    metrics: List[HealthMetric]
    alerts: List[str]
    uptime_seconds: float
    last_inference_time: Optional[datetime] = None


@dataclass
class Alert:
    """System alert"""
    alert_id: str
    level: AlertLevel
    message: str
    model_name: Optional[str]
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None


class MonitoringService:
    """
    Service for monitoring deployed WAN models
    
    Provides:
    - Real-time health monitoring
    - Performance metrics collection
    - Resource usage tracking
    - Automated alerting
    - Health trend analysis
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Monitoring state
        self.monitored_deployments: Dict[str, Dict[str, Any]] = {}
        self.health_history: List[ModelHealthReport] = []
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_handlers: List[Callable] = []
        
        # Monitoring configuration
        self.monitoring_interval = config.health_check_interval
        self.max_history_days = 7
        
        # Performance thresholds
        self.thresholds = {
            "cpu_usage_warning": 80.0,
            "cpu_usage_critical": 95.0,
            "memory_usage_warning": 85.0,
            "memory_usage_critical": 95.0,
            "disk_usage_warning": 85.0,
            "disk_usage_critical": 95.0,
            "inference_time_warning": 30.0,  # seconds
            "inference_time_critical": 60.0,  # seconds
            "model_load_time_warning": 60.0,  # seconds
            "model_load_time_critical": 120.0  # seconds
        }
        
        # Start background monitoring task
        self._monitoring_task = None
        if config.monitoring_enabled:
            self._start_monitoring()
    
    def _start_monitoring(self):
        """Start the background monitoring task"""
        if self._monitoring_task is None or self._monitoring_task.done():
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            self.logger.info("Started background monitoring service")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while True:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.monitoring_interval)
            except asyncio.CancelledError:
                self.logger.info("Monitoring loop cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def start_monitoring(self, deployment_id: str, models: List[str]):
        """
        Start monitoring a deployment
        
        Args:
            deployment_id: ID of the deployment to monitor
            models: List of model names to monitor
        """
        self.logger.info(f"Starting monitoring for deployment {deployment_id} with models: {models}")
        
        self.monitored_deployments[deployment_id] = {
            "models": models,
            "start_time": datetime.now(),
            "last_check": None,
            "status": HealthStatus.UNKNOWN
        }
        
        # Ensure monitoring loop is running
        self._start_monitoring()
    
    async def stop_monitoring(self, deployment_id: str):
        """Stop monitoring a deployment"""
        if deployment_id in self.monitored_deployments:
            del self.monitored_deployments[deployment_id]
            self.logger.info(f"Stopped monitoring deployment {deployment_id}")
    
    async def _perform_health_checks(self):
        """Perform health checks for all monitored deployments"""
        for deployment_id, deployment_info in self.monitored_deployments.items():
            try:
                for model_name in deployment_info["models"]:
                    health_report = await self._check_model_health(model_name, deployment_id)
                    self.health_history.append(health_report)
                    
                    # Process alerts
                    await self._process_health_alerts(health_report)
                
                deployment_info["last_check"] = datetime.now()
                
            except Exception as e:
                self.logger.error(f"Health check failed for deployment {deployment_id}: {str(e)}")
        
        # Clean up old history
        await self._cleanup_old_history()
    
    async def _check_model_health(self, model_name: str, deployment_id: str) -> ModelHealthReport:
        """Perform comprehensive health check for a single model"""
        start_time = datetime.now()
        metrics = []
        alerts = []
        overall_status = HealthStatus.HEALTHY
        
        try:
            # System resource metrics
            system_metrics = await self._collect_system_metrics()
            metrics.extend(system_metrics)
            
            # Model-specific metrics
            model_metrics = await self._collect_model_metrics(model_name)
            metrics.extend(model_metrics)
            
            # Performance metrics
            performance_metrics = await self._collect_performance_metrics(model_name)
            metrics.extend(performance_metrics)
            
            # Determine overall status based on metrics
            for metric in metrics:
                if metric.status == HealthStatus.CRITICAL:
                    overall_status = HealthStatus.CRITICAL
                    break
                elif metric.status == HealthStatus.WARNING and overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.WARNING
            
            # Calculate uptime
            deployment_info = self.monitored_deployments.get(deployment_id, {})
            start_time_deployment = deployment_info.get("start_time", datetime.now())
            uptime_seconds = (datetime.now() - start_time_deployment).total_seconds()
            
        except Exception as e:
            self.logger.error(f"Health check error for model {model_name}: {str(e)}")
            overall_status = HealthStatus.UNKNOWN
            alerts.append(f"Health check failed: {str(e)}")
            uptime_seconds = 0.0
        
        return ModelHealthReport(
            model_name=model_name,
            deployment_id=deployment_id,
            overall_status=overall_status,
            timestamp=datetime.now(),
            metrics=metrics,
            alerts=alerts,
            uptime_seconds=uptime_seconds
        )
    
    async def _collect_system_metrics(self) -> List[HealthMetric]:
        """Collect system-level metrics"""
        metrics = []
        timestamp = datetime.now()
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_status = self._determine_status(
                cpu_percent,
                self.thresholds["cpu_usage_warning"],
                self.thresholds["cpu_usage_critical"]
            )
            
            metrics.append(HealthMetric(
                name="cpu_usage",
                value=cpu_percent,
                unit="percent",
                timestamp=timestamp,
                status=cpu_status,
                threshold_warning=self.thresholds["cpu_usage_warning"],
                threshold_critical=self.thresholds["cpu_usage_critical"]
            ))
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_status = self._determine_status(
                memory.percent,
                self.thresholds["memory_usage_warning"],
                self.thresholds["memory_usage_critical"]
            )
            
            metrics.append(HealthMetric(
                name="memory_usage",
                value=memory.percent,
                unit="percent",
                timestamp=timestamp,
                status=memory_status,
                threshold_warning=self.thresholds["memory_usage_warning"],
                threshold_critical=self.thresholds["memory_usage_critical"]
            ))
            
            # Disk usage
            disk = psutil.disk_usage('.')
            disk_percent = (disk.used / disk.total) * 100
            disk_status = self._determine_status(
                disk_percent,
                self.thresholds["disk_usage_warning"],
                self.thresholds["disk_usage_critical"]
            )
            
            metrics.append(HealthMetric(
                name="disk_usage",
                value=disk_percent,
                unit="percent",
                timestamp=timestamp,
                status=disk_status,
                threshold_warning=self.thresholds["disk_usage_warning"],
                threshold_critical=self.thresholds["disk_usage_critical"]
            ))
            
            # GPU metrics (if available)
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    gpu_memory = torch.cuda.memory_allocated(i) / torch.cuda.max_memory_allocated(i) * 100
                    
                    metrics.append(HealthMetric(
                        name=f"gpu_{i}_memory_usage",
                        value=gpu_memory,
                        unit="percent",
                        timestamp=timestamp,
                        status=self._determine_status(gpu_memory, 80.0, 95.0)
                    ))
            
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {str(e)}")
        
        return metrics
    
    async def _collect_model_metrics(self, model_name: str) -> List[HealthMetric]:
        """Collect model-specific metrics"""
        metrics = []
        timestamp = datetime.now()
        
        try:
            model_path = Path(self.config.target_models_path) / model_name
            
            # Model file integrity
            if model_path.exists():
                file_count = len(list(model_path.rglob('*')))
                metrics.append(HealthMetric(
                    name=f"{model_name}_file_count",
                    value=file_count,
                    unit="files",
                    timestamp=timestamp,
                    status=HealthStatus.HEALTHY if file_count > 0 else HealthStatus.CRITICAL
                ))
                
                # Model size
                total_size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
                metrics.append(HealthMetric(
                    name=f"{model_name}_size",
                    value=total_size / (1024**3),  # GB
                    unit="GB",
                    timestamp=timestamp,
                    status=HealthStatus.HEALTHY
                ))
            else:
                metrics.append(HealthMetric(
                    name=f"{model_name}_availability",
                    value=0,
                    unit="boolean",
                    timestamp=timestamp,
                    status=HealthStatus.CRITICAL
                ))
            
        except Exception as e:
            self.logger.error(f"Failed to collect model metrics for {model_name}: {str(e)}")
        
        return metrics
    
    async def _collect_performance_metrics(self, model_name: str) -> List[HealthMetric]:
        """Collect performance metrics for a model"""
        metrics = []
        timestamp = datetime.now()
        
        try:
            # Simulate performance metrics (in a real implementation, these would be actual measurements)
            # Model load time
            load_time = await self._measure_model_load_time(model_name)
            load_status = self._determine_status(
                load_time,
                self.thresholds["model_load_time_warning"],
                self.thresholds["model_load_time_critical"]
            )
            
            metrics.append(HealthMetric(
                name=f"{model_name}_load_time",
                value=load_time,
                unit="seconds",
                timestamp=timestamp,
                status=load_status,
                threshold_warning=self.thresholds["model_load_time_warning"],
                threshold_critical=self.thresholds["model_load_time_critical"]
            ))
            
            # Inference time
            inference_time = await self._measure_inference_time(model_name)
            inference_status = self._determine_status(
                inference_time,
                self.thresholds["inference_time_warning"],
                self.thresholds["inference_time_critical"]
            )
            
            metrics.append(HealthMetric(
                name=f"{model_name}_inference_time",
                value=inference_time,
                unit="seconds",
                timestamp=timestamp,
                status=inference_status,
                threshold_warning=self.thresholds["inference_time_warning"],
                threshold_critical=self.thresholds["inference_time_critical"]
            ))
            
        except Exception as e:
            self.logger.error(f"Failed to collect performance metrics for {model_name}: {str(e)}")
        
        return metrics
    
    async def _measure_model_load_time(self, model_name: str) -> float:
        """Measure model loading time (simulated)"""
        # In a real implementation, this would actually load the model and measure time
        # For now, return a simulated value based on model characteristics
        model_sizes = {
            "t2v-A14B": 25.0,  # seconds
            "i2v-A14B": 25.0,  # seconds
            "ti2v-5B": 15.0    # seconds
        }
        return model_sizes.get(model_name, 20.0)
    
    async def _measure_inference_time(self, model_name: str) -> float:
        """Measure inference time (simulated)"""
        # In a real implementation, this would run actual inference and measure time
        # For now, return a simulated value
        return 5.0  # seconds
    
    def _determine_status(self, value: float, warning_threshold: float, critical_threshold: float) -> HealthStatus:
        """Determine health status based on value and thresholds"""
        if value >= critical_threshold:
            return HealthStatus.CRITICAL
        elif value >= warning_threshold:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY
    
    async def _process_health_alerts(self, health_report: ModelHealthReport):
        """Process health report and generate alerts"""
        for metric in health_report.metrics:
            if metric.status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
                alert_id = f"{health_report.model_name}_{metric.name}_{int(time.time())}"
                
                alert = Alert(
                    alert_id=alert_id,
                    level=AlertLevel.WARNING if metric.status == HealthStatus.WARNING else AlertLevel.CRITICAL,
                    message=f"{health_report.model_name}: {metric.name} is {metric.value}{metric.unit} (threshold: {metric.threshold_warning or metric.threshold_critical})",
                    model_name=health_report.model_name,
                    timestamp=datetime.now()
                )
                
                self.active_alerts[alert_id] = alert
                await self._trigger_alert(alert)
    
    async def _trigger_alert(self, alert: Alert):
        """Trigger an alert through registered handlers"""
        self.logger.warning(f"ALERT [{alert.level.value.upper()}]: {alert.message}")
        
        for handler in self.alert_handlers:
            try:
                await handler(alert)
            except Exception as e:
                self.logger.error(f"Alert handler failed: {str(e)}")
    
    async def _cleanup_old_history(self):
        """Clean up old health history"""
        cutoff_date = datetime.now() - timedelta(days=self.max_history_days)
        
        self.health_history = [
            report for report in self.health_history
            if report.timestamp > cutoff_date
        ]
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status of all monitored models"""
        status_summary = {
            "timestamp": datetime.now().isoformat(),
            "monitored_deployments": len(self.monitored_deployments),
            "active_alerts": len(self.active_alerts),
            "overall_status": "healthy",
            "deployments": {}
        }
        
        # Determine overall status
        critical_count = 0
        warning_count = 0
        
        for deployment_id, deployment_info in self.monitored_deployments.items():
            deployment_status = {
                "models": deployment_info["models"],
                "uptime_seconds": (datetime.now() - deployment_info["start_time"]).total_seconds(),
                "last_check": deployment_info.get("last_check"),
                "status": "unknown"
            }
            
            # Get latest health reports for this deployment
            latest_reports = []
            for model_name in deployment_info["models"]:
                model_reports = [
                    r for r in self.health_history
                    if r.model_name == model_name and r.deployment_id == deployment_id
                ]
                if model_reports:
                    latest_reports.append(max(model_reports, key=lambda x: x.timestamp))
            
            if latest_reports:
                # Determine deployment status based on model statuses
                statuses = [r.overall_status for r in latest_reports]
                if HealthStatus.CRITICAL in statuses:
                    deployment_status["status"] = "critical"
                    critical_count += 1
                elif HealthStatus.WARNING in statuses:
                    deployment_status["status"] = "warning"
                    warning_count += 1
                else:
                    deployment_status["status"] = "healthy"
            
            status_summary["deployments"][deployment_id] = deployment_status
        
        # Set overall status
        if critical_count > 0:
            status_summary["overall_status"] = "critical"
        elif warning_count > 0:
            status_summary["overall_status"] = "warning"
        
        return status_summary
    
    async def get_model_health_history(self, model_name: str, hours: int = 24) -> List[ModelHealthReport]:
        """Get health history for a specific model"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            report for report in self.health_history
            if report.model_name == model_name and report.timestamp > cutoff_time
        ]
    
    async def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        return [alert for alert in self.active_alerts.values() if not alert.resolved]
    
    async def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an active alert"""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].resolved = True
            self.active_alerts[alert_id].resolved_at = datetime.now()
            self.logger.info(f"Resolved alert {alert_id}")
            return True
        return False
    
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add a custom alert handler"""
        self.alert_handlers.append(handler)
    
    async def export_health_report(self, output_path: str, hours: int = 24):
        """Export health monitoring report"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_reports = [
            report for report in self.health_history
            if report.timestamp > cutoff_time
        ]
        
        report_data = {
            "generated_at": datetime.now().isoformat(),
            "time_range_hours": hours,
            "total_reports": len(recent_reports),
            "monitored_deployments": len(self.monitored_deployments),
            "active_alerts": len([a for a in self.active_alerts.values() if not a.resolved]),
            "health_reports": [asdict(report) for report in recent_reports],
            "alerts": [asdict(alert) for alert in self.active_alerts.values()]
        }
        
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        self.logger.info(f"Health report exported to {output_path}")
    
    async def shutdown(self):
        """Shutdown the monitoring service"""
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Monitoring service shutdown complete")
