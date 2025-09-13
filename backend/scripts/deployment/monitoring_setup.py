#!/usr/bin/env python3
"""
Monitoring and Alerting Setup for Enhanced Model Availability System

This script sets up monitoring and alerting for production deployment of the
enhanced model availability system, ensuring operational visibility and proactive issue detection.
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class MetricType(Enum):
    """Types of metrics to monitor"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"

@dataclass
class MonitoringMetric:
    """Represents a monitoring metric"""
    name: str
    metric_type: MetricType
    value: float
    timestamp: str
    labels: Dict[str, str]
    description: str

@dataclass
class AlertRule:
    """Represents an alert rule"""
    name: str
    metric_name: str
    condition: str  # e.g., "> 0.8", "< 0.1"
    threshold: float
    level: AlertLevel
    description: str
    cooldown_minutes: int = 5
    enabled: bool = True

@dataclass
class Alert:
    """Represents an active alert"""
    rule_name: str
    level: AlertLevel
    message: str
    timestamp: str
    metric_value: float
    resolved: bool = False
    resolved_timestamp: Optional[str] = None

class MetricsCollector:
    """Collects metrics from the enhanced model availability system"""
    
    def __init__(self):
        self.metrics: Dict[str, MonitoringMetric] = {}
        self.metrics_history: List[MonitoringMetric] = []
        self.max_history_size = 10000
    
    def record_metric(self, name: str, metric_type: MetricType, value: float, 
                     labels: Dict[str, str] = None, description: str = ""):
        """Record a metric value"""
        labels = labels or {}
        timestamp = datetime.now().isoformat()
        
        metric = MonitoringMetric(
            name=name,
            metric_type=metric_type,
            value=value,
            timestamp=timestamp,
            labels=labels,
            description=description
        )
        
        self.metrics[name] = metric
        self.metrics_history.append(metric)
        
        # Trim history if needed
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history = self.metrics_history[-self.max_history_size:]
    
    def get_metric(self, name: str) -> Optional[MonitoringMetric]:
        """Get current value of a metric"""
        return self.metrics.get(name)
    
    def get_metrics_by_prefix(self, prefix: str) -> List[MonitoringMetric]:
        """Get all metrics with a given prefix"""
        return [metric for name, metric in self.metrics.items() if name.startswith(prefix)]
    
    def get_metric_history(self, name: str, hours: int = 24) -> List[MonitoringMetric]:
        """Get metric history for the specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            metric for metric in self.metrics_history
            if metric.name == name and datetime.fromisoformat(metric.timestamp) >= cutoff_time
        ]

class AlertManager:
    """Manages alerts and notifications"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.alert_cooldowns: Dict[str, datetime] = {}
        self.notification_handlers: List[Callable[[Alert], None]] = []
    
    def add_alert_rule(self, rule: AlertRule):
        """Add an alert rule"""
        self.alert_rules[rule.name] = rule
        logger.info(f"Added alert rule: {rule.name}")
    
    def add_notification_handler(self, handler: Callable[[Alert], None]):
        """Add a notification handler"""
        self.notification_handlers.append(handler)
    
    def check_alerts(self):
        """Check all alert rules against current metrics"""
        for rule_name, rule in self.alert_rules.items():
            if not rule.enabled:
                continue
            
            # Check cooldown
            if rule_name in self.alert_cooldowns:
                if datetime.now() < self.alert_cooldowns[rule_name]:
                    continue
            
            # Get metric
            metric = self.metrics_collector.get_metric(rule.metric_name)
            if not metric:
                continue
            
            # Evaluate condition
            if self._evaluate_condition(metric.value, rule.condition, rule.threshold):
                # Create alert
                alert = Alert(
                    rule_name=rule_name,
                    level=rule.level,
                    message=f"{rule.description} (value: {metric.value}, threshold: {rule.threshold})",
                    timestamp=datetime.now().isoformat(),
                    metric_value=metric.value
                )
                
                # Add to active alerts
                self.active_alerts[rule_name] = alert
                self.alert_history.append(alert)
                
                # Set cooldown
                self.alert_cooldowns[rule_name] = datetime.now() + timedelta(minutes=rule.cooldown_minutes)
                
                # Send notifications
                self._send_notifications(alert)
                
                logger.warning(f"Alert triggered: {rule_name} - {alert.message}")
            
            else:
                # Check if alert should be resolved
                if rule_name in self.active_alerts:
                    alert = self.active_alerts[rule_name]
                    if not alert.resolved:
                        alert.resolved = True
                        alert.resolved_timestamp = datetime.now().isoformat()
                        logger.info(f"Alert resolved: {rule_name}")
    
    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Evaluate alert condition"""
        try:
            if condition.startswith(">"):
                return value > threshold
            elif condition.startswith("<"):
                return value < threshold
            elif condition.startswith(">="):
                return value >= threshold
            elif condition.startswith("<="):
                return value <= threshold
            elif condition.startswith("=="):
                return value == threshold
            elif condition.startswith("!="):
                return value != threshold
            else:
                return False
        except Exception as e:
            logger.error(f"Error evaluating condition {condition}: {e}")
            return False
    
    def _send_notifications(self, alert: Alert):
        """Send alert notifications"""
        for handler in self.notification_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error sending notification: {e}")

class EnhancedModelAvailabilityMonitor:
    """Main monitoring system for enhanced model availability"""
    
    def __init__(self, config_path: str = "monitoring_config.json"):
        self.config_path = Path(config_path)
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager(self.metrics_collector)
        self.monitoring_thread = None
        self.running = False
        
        # Load configuration
        self.config = self._load_config()
        
        # Setup default alert rules
        self._setup_default_alert_rules()
        
        # Setup notification handlers
        self._setup_notification_handlers()
    
    def start_monitoring(self):
        """Start the monitoring system"""
        if self.running:
            logger.warning("Monitoring already running")
            return
        
        self.running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Enhanced model availability monitoring started")
    
    def stop_monitoring(self):
        """Stop the monitoring system"""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        logger.info("Enhanced model availability monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # Collect metrics
                self._collect_system_metrics()
                self._collect_model_metrics()
                self._collect_download_metrics()
                self._collect_health_metrics()
                self._collect_performance_metrics()
                
                # Check alerts
                self.alert_manager.check_alerts()
                
                # Sleep for monitoring interval
                time.sleep(self.config.get("monitoring_interval_seconds", 30))
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)  # Brief pause before retrying
    
    def _collect_system_metrics(self):
        """Collect system-level metrics"""
        try:
            import psutil
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics_collector.record_metric(
                "system_cpu_usage_percent",
                MetricType.GAUGE,
                cpu_percent,
                description="System CPU usage percentage"
            )
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.metrics_collector.record_metric(
                "system_memory_usage_percent",
                MetricType.GAUGE,
                memory.percent,
                description="System memory usage percentage"
            )
            
            # Disk usage
            disk = psutil.disk_usage('.')
            disk_percent = (disk.used / disk.total) * 100
            self.metrics_collector.record_metric(
                "system_disk_usage_percent",
                MetricType.GAUGE,
                disk_percent,
                description="System disk usage percentage"
            )
            
        except ImportError:
            logger.warning("psutil not available for system metrics")
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    def _collect_model_metrics(self):
        """Collect model-related metrics"""
        try:
            # Count models by status
            models_dir = Path("models")
            if models_dir.exists():
                total_models = len([d for d in models_dir.iterdir() if d.is_dir()])
                self.metrics_collector.record_metric(
                    "models_total_count",
                    MetricType.GAUGE,
                    total_models,
                    description="Total number of models"
                )
                
                # Calculate total model storage
                total_size = 0
                for model_dir in models_dir.iterdir():
                    if model_dir.is_dir():
                        for file in model_dir.rglob("*"):
                            if file.is_file():
                                total_size += file.stat().st_size
                
                total_size_gb = total_size / (1024**3)
                self.metrics_collector.record_metric(
                    "models_total_size_gb",
                    MetricType.GAUGE,
                    total_size_gb,
                    description="Total size of all models in GB"
                )
            
        except Exception as e:
            logger.error(f"Error collecting model metrics: {e}")
    
    def _collect_download_metrics(self):
        """Collect download-related metrics"""
        try:
            # Check for active downloads (this would integrate with actual download system)
            # For now, we'll simulate some metrics
            
            # Active downloads count
            self.metrics_collector.record_metric(
                "downloads_active_count",
                MetricType.GAUGE,
                0,  # Would be actual count from download manager
                description="Number of active downloads"
            )
            
            # Download success rate (would be calculated from actual data)
            self.metrics_collector.record_metric(
                "downloads_success_rate",
                MetricType.GAUGE,
                0.95,  # Would be actual success rate
                description="Download success rate (0-1)"
            )
            
        except Exception as e:
            logger.error(f"Error collecting download metrics: {e}")
    
    def _collect_health_metrics(self):
        """Collect health monitoring metrics"""
        try:
            # Model health score (would integrate with actual health monitor)
            self.metrics_collector.record_metric(
                "models_health_score",
                MetricType.GAUGE,
                0.98,  # Would be actual health score
                description="Overall model health score (0-1)"
            )
            
            # Corrupted models count
            self.metrics_collector.record_metric(
                "models_corrupted_count",
                MetricType.GAUGE,
                0,  # Would be actual count
                description="Number of corrupted models detected"
            )
            
        except Exception as e:
            logger.error(f"Error collecting health metrics: {e}")
    
    def _collect_performance_metrics(self):
        """Collect performance metrics"""
        try:
            # Average generation time (would integrate with actual generation service)
            self.metrics_collector.record_metric(
                "generation_avg_time_seconds",
                MetricType.GAUGE,
                2.5,  # Would be actual average
                description="Average generation time in seconds"
            )
            
            # Generation success rate
            self.metrics_collector.record_metric(
                "generation_success_rate",
                MetricType.GAUGE,
                0.99,  # Would be actual success rate
                description="Generation success rate (0-1)"
            )
            
        except Exception as e:
            logger.error(f"Error collecting performance metrics: {e}")
    
    def _setup_default_alert_rules(self):
        """Setup default alert rules"""
        default_rules = [
            AlertRule(
                name="high_cpu_usage",
                metric_name="system_cpu_usage_percent",
                condition=">",
                threshold=80.0,
                level=AlertLevel.WARNING,
                description="High CPU usage detected",
                cooldown_minutes=5
            ),
            AlertRule(
                name="high_memory_usage",
                metric_name="system_memory_usage_percent",
                condition=">",
                threshold=85.0,
                level=AlertLevel.WARNING,
                description="High memory usage detected",
                cooldown_minutes=5
            ),
            AlertRule(
                name="high_disk_usage",
                metric_name="system_disk_usage_percent",
                condition=">",
                threshold=90.0,
                level=AlertLevel.CRITICAL,
                description="High disk usage detected",
                cooldown_minutes=10
            ),
            AlertRule(
                name="low_download_success_rate",
                metric_name="downloads_success_rate",
                condition="<",
                threshold=0.8,
                level=AlertLevel.WARNING,
                description="Low download success rate",
                cooldown_minutes=15
            ),
            AlertRule(
                name="low_model_health_score",
                metric_name="models_health_score",
                condition="<",
                threshold=0.9,
                level=AlertLevel.WARNING,
                description="Low model health score",
                cooldown_minutes=10
            ),
            AlertRule(
                name="corrupted_models_detected",
                metric_name="models_corrupted_count",
                condition=">",
                threshold=0,
                level=AlertLevel.CRITICAL,
                description="Corrupted models detected",
                cooldown_minutes=5
            ),
            AlertRule(
                name="low_generation_success_rate",
                metric_name="generation_success_rate",
                condition="<",
                threshold=0.95,
                level=AlertLevel.WARNING,
                description="Low generation success rate",
                cooldown_minutes=10
            )
        ]
        
        for rule in default_rules:
            self.alert_manager.add_alert_rule(rule)
    
    def _setup_notification_handlers(self):
        """Setup notification handlers"""
        # Console notification handler
        def console_handler(alert: Alert):
            level_emoji = {
                AlertLevel.INFO: "ℹ️",
                AlertLevel.WARNING: "⚠️",
                AlertLevel.CRITICAL: "🚨",
                AlertLevel.EMERGENCY: "🆘"
            }
            
            emoji = level_emoji.get(alert.level, "📢")
            print(f"{emoji} [{alert.level.value.upper()}] {alert.message}")
        
        self.alert_manager.add_notification_handler(console_handler)
        
        # Log file notification handler
        def log_handler(alert: Alert):
            log_level = {
                AlertLevel.INFO: logging.INFO,
                AlertLevel.WARNING: logging.WARNING,
                AlertLevel.CRITICAL: logging.CRITICAL,
                AlertLevel.EMERGENCY: logging.CRITICAL
            }
            
            level = log_level.get(alert.level, logging.INFO)
            logger.log(level, f"ALERT: {alert.message}")
        
        self.alert_manager.add_notification_handler(log_handler)
        
        # File-based alert log
        def file_handler(alert: Alert):
            try:
                alerts_log_path = Path("logs/alerts.log")
                alerts_log_path.parent.mkdir(exist_ok=True)
                
                with open(alerts_log_path, 'a') as f:
                    f.write(f"{alert.timestamp} [{alert.level.value.upper()}] {alert.rule_name}: {alert.message}\n")
            except Exception as e:
                logger.error(f"Error writing alert to file: {e}")
        
        self.alert_manager.add_notification_handler(file_handler)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load monitoring configuration"""
        default_config = {
            "monitoring_interval_seconds": 30,
            "metrics_retention_hours": 168,  # 1 week
            "alert_cooldown_minutes": 5,
            "notification_channels": ["console", "log", "file"]
        }
        
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                
                # Merge with defaults
                default_config.update(config)
                
            except Exception as e:
                logger.error(f"Error loading config: {e}")
        
        return default_config
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status"""
        return {
            "running": self.running,
            "metrics_count": len(self.metrics_collector.metrics),
            "active_alerts": len(self.alert_manager.active_alerts),
            "alert_rules": len(self.alert_manager.alert_rules),
            "last_check": datetime.now().isoformat()
        }
    
    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in specified format"""
        if format == "json":
            return json.dumps({
                "timestamp": datetime.now().isoformat(),
                "metrics": {name: asdict(metric) for name, metric in self.metrics_collector.metrics.items()}
            }, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")

def setup_monitoring_config():
    """Setup default monitoring configuration"""
    config = {
        "monitoring_interval_seconds": 30,
        "metrics_retention_hours": 168,
        "alert_cooldown_minutes": 5,
        "notification_channels": ["console", "log", "file"],
        "custom_alert_rules": [
            {
                "name": "custom_high_generation_time",
                "metric_name": "generation_avg_time_seconds",
                "condition": ">",
                "threshold": 10.0,
                "level": "warning",
                "description": "Generation time is too high",
                "cooldown_minutes": 10,
                "enabled": True
            }
        ]
    }
    
    config_path = Path("monitoring_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Created monitoring configuration: {config_path}")

async def main():
    """Main monitoring setup function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Model Availability Monitoring Setup")
    parser.add_argument("action", choices=["setup", "start", "status", "export"],
                       help="Action to perform")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--duration", type=int, default=60, help="Monitoring duration in seconds")
    parser.add_argument("--format", choices=["json"], default="json", help="Export format")
    
    args = parser.parse_args()
    
    if args.action == "setup":
        setup_monitoring_config()
        print("Monitoring configuration created")
        
    elif args.action == "start":
        config_path = args.config or "monitoring_config.json"
        monitor = EnhancedModelAvailabilityMonitor(config_path)
        
        monitor.start_monitoring()
        print(f"Monitoring started for {args.duration} seconds...")
        
        try:
            await asyncio.sleep(args.duration)
        except KeyboardInterrupt:
            print("\nMonitoring interrupted by user")
        finally:
            monitor.stop_monitoring()
        
    elif args.action == "status":
        config_path = args.config or "monitoring_config.json"
        monitor = EnhancedModelAvailabilityMonitor(config_path)
        
        status = monitor.get_monitoring_status()
        print(json.dumps(status, indent=2))
        
    elif args.action == "export":
        config_path = args.config or "monitoring_config.json"
        monitor = EnhancedModelAvailabilityMonitor(config_path)
        
        # Collect some metrics first
        monitor._collect_system_metrics()
        monitor._collect_model_metrics()
        
        metrics_data = monitor.export_metrics(args.format)
        
        export_path = Path(f"metrics_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{args.format}")
        with open(export_path, 'w') as f:
            f.write(metrics_data)
        
        print(f"Metrics exported to: {export_path}")
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
