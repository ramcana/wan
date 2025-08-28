"""
Production Monitoring and Alerting System

This module provides comprehensive monitoring and alerting capabilities
for production deployments of the reliability system. It includes health
checks, performance monitoring, error tracking, and automated alerting.

Requirements addressed:
- 8.1: Health report generation and metrics collection
- 8.5: Cross-instance monitoring and centralized dashboard
"""

import json
import time
import logging
import smtplib
import requests
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path

try:
    from email.mime.text import MimeText
    from email.mime.multipart import MimeMultipart
    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False
    MimeText = None
    MimeMultipart = None
from threading import Thread, Event
from queue import Queue, Empty
import sqlite3


@dataclass
class AlertThreshold:
    """Alert threshold configuration."""
    metric_name: str
    warning_threshold: float
    critical_threshold: float
    comparison_operator: str = "greater_than"  # greater_than, less_than, equals
    time_window_minutes: int = 5
    min_occurrences: int = 1


@dataclass
class AlertChannel:
    """Alert delivery channel configuration."""
    name: str
    type: str  # email, webhook, log, sms
    enabled: bool = True
    configuration: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.configuration is None:
            self.configuration = {}


@dataclass
class MonitoringConfig:
    """Production monitoring configuration."""
    enabled: bool = True
    check_interval_seconds: int = 60
    health_check_timeout_seconds: int = 30
    metrics_retention_days: int = 30
    alert_thresholds: List[AlertThreshold] = None
    alert_channels: List[AlertChannel] = None
    dashboard_enabled: bool = True
    dashboard_port: int = 8080
    cross_instance_monitoring: bool = False
    instance_discovery_url: Optional[str] = None
    
    def __post_init__(self):
        if self.alert_thresholds is None:
            self.alert_thresholds = self._get_default_thresholds()
        if self.alert_channels is None:
            self.alert_channels = self._get_default_channels()
    
    def _get_default_thresholds(self) -> List[AlertThreshold]:
        """Get default alert thresholds."""
        return [
            AlertThreshold("error_rate", 0.05, 0.10, "greater_than", 5, 3),
            AlertThreshold("response_time_ms", 5000, 10000, "greater_than", 5, 2),
            AlertThreshold("memory_usage_percent", 80, 90, "greater_than", 10, 1),
            AlertThreshold("disk_usage_percent", 85, 95, "greater_than", 15, 1),
            AlertThreshold("cpu_usage_percent", 80, 90, "greater_than", 10, 2),
            AlertThreshold("success_rate", 0.95, 0.90, "less_than", 10, 2),
            AlertThreshold("queue_size", 100, 200, "greater_than", 5, 1),
            AlertThreshold("active_connections", 1000, 2000, "greater_than", 5, 1)
        ]
    
    def _get_default_channels(self) -> List[AlertChannel]:
        """Get default alert channels."""
        return [
            AlertChannel("log", "log", True, {"log_level": "ERROR"}),
            AlertChannel("email", "email", False, {
                "smtp_server": "localhost",
                "smtp_port": 587,
                "username": "",
                "password": "",
                "from_email": "alerts@wan22.com",
                "to_emails": ["admin@wan22.com"]
            }),
            AlertChannel("webhook", "webhook", False, {
                "url": "https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
                "method": "POST",
                "headers": {"Content-Type": "application/json"}
            })
        ]


@dataclass
class HealthMetric:
    """Individual health metric."""
    name: str
    value: float
    unit: str
    timestamp: datetime
    instance_id: str
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


@dataclass
class Alert:
    """Alert instance."""
    id: str
    metric_name: str
    severity: str  # warning, critical
    message: str
    value: float
    threshold: float
    timestamp: datetime
    instance_id: str
    resolved: bool = False
    resolved_timestamp: Optional[datetime] = None


class MetricsCollector:
    """Collects system and application metrics."""
    
    def __init__(self, instance_id: str):
        """Initialize metrics collector."""
        self.instance_id = instance_id
        self.logger = logging.getLogger(__name__)
    
    def collect_system_metrics(self) -> List[HealthMetric]:
        """Collect system-level metrics."""
        metrics = []
        timestamp = datetime.now()
        
        try:
            import psutil
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            metrics.append(HealthMetric(
                "cpu_usage_percent", cpu_percent, "percent", 
                timestamp, self.instance_id
            ))
            
            # Memory usage
            memory = psutil.virtual_memory()
            metrics.append(HealthMetric(
                "memory_usage_percent", memory.percent, "percent",
                timestamp, self.instance_id
            ))
            metrics.append(HealthMetric(
                "memory_available_gb", memory.available / (1024**3), "gb",
                timestamp, self.instance_id
            ))
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            metrics.append(HealthMetric(
                "disk_usage_percent", disk_percent, "percent",
                timestamp, self.instance_id
            ))
            metrics.append(HealthMetric(
                "disk_free_gb", disk.free / (1024**3), "gb",
                timestamp, self.instance_id
            ))
            
            # Network I/O
            network = psutil.net_io_counters()
            metrics.append(HealthMetric(
                "network_bytes_sent", network.bytes_sent, "bytes",
                timestamp, self.instance_id
            ))
            metrics.append(HealthMetric(
                "network_bytes_recv", network.bytes_recv, "bytes",
                timestamp, self.instance_id
            ))
            
        except ImportError:
            self.logger.warning("psutil not available, using basic system metrics")
            # Fallback to basic metrics
            import shutil
            disk_usage = shutil.disk_usage('.')
            disk_percent = (disk_usage.used / disk_usage.total) * 100
            metrics.append(HealthMetric(
                "disk_usage_percent", disk_percent, "percent",
                timestamp, self.instance_id
            ))
        
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")
        
        return metrics
    
    def collect_application_metrics(self) -> List[HealthMetric]:
        """Collect application-specific metrics."""
        metrics = []
        timestamp = datetime.now()
        
        try:
            # Import reliability components
            from reliability_integration import get_reliability_integration
            from feature_flags import get_feature_flag_manager
            
            integration = get_reliability_integration()
            flag_manager = get_feature_flag_manager()
            
            # Reliability system health
            if integration.is_available():
                health_status = integration.get_health_status()
                
                metrics.append(HealthMetric(
                    "reliability_system_available", 1.0, "boolean",
                    timestamp, self.instance_id
                ))
                
                # Extract metrics from health status
                for key, value in health_status.items():
                    if isinstance(value, (int, float)):
                        metrics.append(HealthMetric(
                            f"reliability_{key}", float(value), "count",
                            timestamp, self.instance_id
                        ))
            else:
                metrics.append(HealthMetric(
                    "reliability_system_available", 0.0, "boolean",
                    timestamp, self.instance_id
                ))
            
            # Feature flag metrics
            flags_status = flag_manager.get_all_flags_status()
            summary = flags_status.get("summary", {})
            
            for key, value in summary.items():
                metrics.append(HealthMetric(
                    f"feature_flags_{key}", float(value), "count",
                    timestamp, self.instance_id
                ))
            
        except Exception as e:
            self.logger.error(f"Failed to collect application metrics: {e}")
        
        return metrics


class AlertManager:
    """Manages alerts and notifications."""
    
    def __init__(self, config: MonitoringConfig):
        """Initialize alert manager."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        
        # Initialize database for alert storage
        self.db_path = Path(__file__).parent.parent / "logs" / "alerts.db"
        self._init_database()
    
    def _init_database(self):
        """Initialize alerts database."""
        try:
            self.db_path.parent.mkdir(exist_ok=True)
            
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS alerts (
                        id TEXT PRIMARY KEY,
                        metric_name TEXT,
                        severity TEXT,
                        message TEXT,
                        value REAL,
                        threshold REAL,
                        timestamp TEXT,
                        instance_id TEXT,
                        resolved INTEGER,
                        resolved_timestamp TEXT
                    )
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_alerts_timestamp 
                    ON alerts(timestamp)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_alerts_instance 
                    ON alerts(instance_id)
                """)
        except Exception as e:
            self.logger.error(f"Failed to initialize alerts database: {e}")
    
    def check_thresholds(self, metrics: List[HealthMetric]) -> List[Alert]:
        """Check metrics against alert thresholds."""
        new_alerts = []
        
        for metric in metrics:
            for threshold in self.config.alert_thresholds:
                if threshold.metric_name == metric.name:
                    alert = self._evaluate_threshold(metric, threshold)
                    if alert:
                        new_alerts.append(alert)
        
        return new_alerts
    
    def _evaluate_threshold(self, metric: HealthMetric, threshold: AlertThreshold) -> Optional[Alert]:
        """Evaluate a single threshold against a metric."""
        try:
            # Determine if threshold is breached
            breached = False
            
            if threshold.comparison_operator == "greater_than":
                breached = metric.value > threshold.critical_threshold
                severity = "critical"
                if not breached and metric.value > threshold.warning_threshold:
                    breached = True
                    severity = "warning"
            elif threshold.comparison_operator == "less_than":
                breached = metric.value < threshold.critical_threshold
                severity = "critical"
                if not breached and metric.value < threshold.warning_threshold:
                    breached = True
                    severity = "warning"
            elif threshold.comparison_operator == "equals":
                breached = abs(metric.value - threshold.critical_threshold) < 0.001
                severity = "critical"
                if not breached and abs(metric.value - threshold.warning_threshold) < 0.001:
                    breached = True
                    severity = "warning"
            
            if not breached:
                # Check if we need to resolve an existing alert
                alert_key = f"{metric.name}_{metric.instance_id}"
                if alert_key in self.active_alerts:
                    self._resolve_alert(alert_key)
                return None
            
            # Create alert
            alert_id = f"{metric.name}_{metric.instance_id}_{int(metric.timestamp.timestamp())}"
            used_threshold = threshold.critical_threshold if severity == "critical" else threshold.warning_threshold
            
            message = f"{metric.name} is {metric.value:.2f} {metric.unit}, exceeding {severity} threshold of {used_threshold:.2f}"
            
            alert = Alert(
                id=alert_id,
                metric_name=metric.name,
                severity=severity,
                message=message,
                value=metric.value,
                threshold=used_threshold,
                timestamp=metric.timestamp,
                instance_id=metric.instance_id
            )
            
            return alert
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate threshold for {metric.name}: {e}")
            return None
    
    def process_alerts(self, alerts: List[Alert]):
        """Process and send alerts."""
        for alert in alerts:
            # Check if this is a duplicate alert
            alert_key = f"{alert.metric_name}_{alert.instance_id}"
            
            if alert_key in self.active_alerts:
                # Update existing alert if severity increased
                existing = self.active_alerts[alert_key]
                if (alert.severity == "critical" and existing.severity == "warning"):
                    self.active_alerts[alert_key] = alert
                    self._send_alert(alert)
            else:
                # New alert
                self.active_alerts[alert_key] = alert
                self._send_alert(alert)
                self._store_alert(alert)
    
    def _resolve_alert(self, alert_key: str):
        """Resolve an active alert."""
        if alert_key in self.active_alerts:
            alert = self.active_alerts[alert_key]
            alert.resolved = True
            alert.resolved_timestamp = datetime.now()
            
            # Send resolution notification
            resolution_message = f"RESOLVED: {alert.message}"
            self._send_notification("Alert Resolved", resolution_message, "info")
            
            # Move to history
            self.alert_history.append(alert)
            del self.active_alerts[alert_key]
            
            # Update database
            self._update_alert_in_db(alert)
    
    def _send_alert(self, alert: Alert):
        """Send alert through configured channels."""
        subject = f"[{alert.severity.upper()}] WAN2.2 Reliability Alert"
        message = f"""
Alert Details:
- Metric: {alert.metric_name}
- Value: {alert.value:.2f}
- Threshold: {alert.threshold:.2f}
- Severity: {alert.severity}
- Instance: {alert.instance_id}
- Time: {alert.timestamp.isoformat()}

Message: {alert.message}
"""
        
        self._send_notification(subject, message, alert.severity)
    
    def _send_notification(self, subject: str, message: str, severity: str):
        """Send notification through all enabled channels."""
        for channel in self.config.alert_channels:
            if not channel.enabled:
                continue
            
            try:
                if channel.type == "log":
                    self._send_log_notification(subject, message, severity, channel)
                elif channel.type == "email":
                    self._send_email_notification(subject, message, channel)
                elif channel.type == "webhook":
                    self._send_webhook_notification(subject, message, severity, channel)
            except Exception as e:
                self.logger.error(f"Failed to send notification via {channel.name}: {e}")
    
    def _send_log_notification(self, subject: str, message: str, severity: str, channel: AlertChannel):
        """Send notification to log."""
        log_level = channel.configuration.get("log_level", "ERROR")
        log_message = f"{subject}\n{message}"
        
        if log_level == "ERROR":
            self.logger.error(log_message)
        elif log_level == "WARNING":
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
    
    def _send_email_notification(self, subject: str, message: str, channel: AlertChannel):
        """Send email notification."""
        if not EMAIL_AVAILABLE:
            self.logger.error("Email functionality not available")
            return
            
        config = channel.configuration
        
        msg = MimeMultipart()
        msg['From'] = config['from_email']
        msg['To'] = ', '.join(config['to_emails'])
        msg['Subject'] = subject
        
        msg.attach(MimeText(message, 'plain'))
        
        server = smtplib.SMTP(config['smtp_server'], config['smtp_port'])
        if config.get('username') and config.get('password'):
            server.starttls()
            server.login(config['username'], config['password'])
        
        server.send_message(msg)
        server.quit()
    
    def _send_webhook_notification(self, subject: str, message: str, severity: str, channel: AlertChannel):
        """Send webhook notification."""
        config = channel.configuration
        
        payload = {
            "subject": subject,
            "message": message,
            "severity": severity,
            "timestamp": datetime.now().isoformat(),
            "source": "wan22_reliability_system"
        }
        
        response = requests.post(
            config['url'],
            json=payload,
            headers=config.get('headers', {}),
            timeout=10
        )
        response.raise_for_status()
    
    def _store_alert(self, alert: Alert):
        """Store alert in database."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute("""
                    INSERT INTO alerts 
                    (id, metric_name, severity, message, value, threshold, 
                     timestamp, instance_id, resolved, resolved_timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    alert.id, alert.metric_name, alert.severity, alert.message,
                    alert.value, alert.threshold, alert.timestamp.isoformat(),
                    alert.instance_id, int(alert.resolved),
                    alert.resolved_timestamp.isoformat() if alert.resolved_timestamp else None
                ))
        except Exception as e:
            self.logger.error(f"Failed to store alert in database: {e}")
    
    def _update_alert_in_db(self, alert: Alert):
        """Update alert in database."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute("""
                    UPDATE alerts 
                    SET resolved = ?, resolved_timestamp = ?
                    WHERE id = ?
                """, (
                    int(alert.resolved),
                    alert.resolved_timestamp.isoformat() if alert.resolved_timestamp else None,
                    alert.id
                ))
        except Exception as e:
            self.logger.error(f"Failed to update alert in database: {e}")


class ProductionMonitor:
    """Main production monitoring system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize production monitor."""
        self.config_path = config_path or self._get_default_config_path()
        self.config = self._load_config()
        self.instance_id = self._generate_instance_id()
        
        self.logger = self._setup_logging()
        self.metrics_collector = MetricsCollector(self.instance_id)
        self.alert_manager = AlertManager(self.config)
        
        self.running = False
        self.monitor_thread: Optional[Thread] = None
        self.stop_event = Event()
        
        # Metrics storage
        self.metrics_db_path = Path(__file__).parent.parent / "logs" / "metrics.db"
        self._init_metrics_database()
    
    def _get_default_config_path(self) -> str:
        """Get default configuration path."""
        script_dir = Path(__file__).parent
        return str(script_dir / "production_monitoring_config.json")
    
    def _load_config(self) -> MonitoringConfig:
        """Load monitoring configuration."""
        try:
            if Path(self.config_path).exists():
                with open(self.config_path, 'r') as f:
                    data = json.load(f)
                
                # Convert thresholds
                thresholds = []
                for threshold_data in data.get("alert_thresholds", []):
                    thresholds.append(AlertThreshold(**threshold_data))
                
                # Convert channels
                channels = []
                for channel_data in data.get("alert_channels", []):
                    channels.append(AlertChannel(**channel_data))
                
                # Create config
                config_data = {k: v for k, v in data.items() 
                              if k not in ["alert_thresholds", "alert_channels"]}
                config = MonitoringConfig(**config_data)
                config.alert_thresholds = thresholds
                config.alert_channels = channels
                
                return config
            else:
                # Create default configuration
                config = MonitoringConfig()
                self._save_config(config)
                return config
        except Exception as e:
            logging.error(f"Failed to load monitoring configuration: {e}")
            return MonitoringConfig()
    
    def _save_config(self, config: MonitoringConfig):
        """Save monitoring configuration."""
        try:
            data = asdict(config)
            
            # Ensure directory exists
            Path(self.config_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save monitoring configuration: {e}")
    
    def _generate_instance_id(self) -> str:
        """Generate unique instance ID."""
        import socket
        import uuid
        
        hostname = socket.gethostname()
        unique_id = str(uuid.uuid4())[:8]
        return f"{hostname}_{unique_id}"
    
    def _setup_logging(self) -> logging.Logger:
        """Setup monitoring logging."""
        logger = logging.getLogger("production_monitor")
        logger.setLevel(logging.INFO)
        
        # Create logs directory
        log_dir = Path(__file__).parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # File handler
        log_file = log_dir / "production_monitoring.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def _init_metrics_database(self):
        """Initialize metrics database."""
        try:
            self.metrics_db_path.parent.mkdir(exist_ok=True)
            
            with sqlite3.connect(str(self.metrics_db_path)) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT,
                        value REAL,
                        unit TEXT,
                        timestamp TEXT,
                        instance_id TEXT,
                        tags TEXT
                    )
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_metrics_timestamp 
                    ON metrics(timestamp)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_metrics_name 
                    ON metrics(name)
                """)
        except Exception as e:
            self.logger.error(f"Failed to initialize metrics database: {e}")
    
    def start(self):
        """Start monitoring."""
        if self.running:
            self.logger.warning("Monitor is already running")
            return
        
        if not self.config.enabled:
            self.logger.info("Monitoring is disabled in configuration")
            return
        
        self.logger.info(f"Starting production monitoring for instance {self.instance_id}")
        self.running = True
        self.stop_event.clear()
        
        self.monitor_thread = Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop(self):
        """Stop monitoring."""
        if not self.running:
            return
        
        self.logger.info("Stopping production monitoring")
        self.running = False
        self.stop_event.set()
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running and not self.stop_event.is_set():
            try:
                # Collect metrics
                system_metrics = self.metrics_collector.collect_system_metrics()
                app_metrics = self.metrics_collector.collect_application_metrics()
                all_metrics = system_metrics + app_metrics
                
                # Store metrics
                self._store_metrics(all_metrics)
                
                # Check for alerts
                alerts = self.alert_manager.check_thresholds(all_metrics)
                if alerts:
                    self.alert_manager.process_alerts(alerts)
                
                # Log monitoring status
                self.logger.info(f"Collected {len(all_metrics)} metrics, generated {len(alerts)} alerts")
                
                # Wait for next check
                self.stop_event.wait(self.config.check_interval_seconds)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                self.stop_event.wait(60)  # Wait 1 minute before retrying
    
    def _store_metrics(self, metrics: List[HealthMetric]):
        """Store metrics in database."""
        try:
            with sqlite3.connect(str(self.metrics_db_path)) as conn:
                for metric in metrics:
                    conn.execute("""
                        INSERT INTO metrics 
                        (name, value, unit, timestamp, instance_id, tags)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        metric.name, metric.value, metric.unit,
                        metric.timestamp.isoformat(), metric.instance_id,
                        json.dumps(metric.tags)
                    ))
        except Exception as e:
            self.logger.error(f"Failed to store metrics: {e}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        try:
            # Get recent metrics
            recent_metrics = self._get_recent_metrics(minutes=5)
            
            # Calculate health indicators
            status = {
                "instance_id": self.instance_id,
                "timestamp": datetime.now().isoformat(),
                "monitoring_enabled": self.config.enabled,
                "active_alerts": len(self.alert_manager.active_alerts),
                "metrics_collected": len(recent_metrics),
                "uptime_seconds": self._get_uptime_seconds(),
                "health_score": self._calculate_health_score(recent_metrics)
            }
            
            # Add recent metric values
            for metric in recent_metrics[-10:]:  # Last 10 metrics
                status[f"latest_{metric.name}"] = metric.value
            
            return status
            
        except Exception as e:
            self.logger.error(f"Failed to get health status: {e}")
            return {"error": str(e)}
    
    def _get_recent_metrics(self, minutes: int = 5) -> List[HealthMetric]:
        """Get metrics from the last N minutes."""
        try:
            cutoff_time = datetime.now() - timedelta(minutes=minutes)
            
            with sqlite3.connect(str(self.metrics_db_path)) as conn:
                cursor = conn.execute("""
                    SELECT name, value, unit, timestamp, instance_id, tags
                    FROM metrics
                    WHERE timestamp > ? AND instance_id = ?
                    ORDER BY timestamp DESC
                """, (cutoff_time.isoformat(), self.instance_id))
                
                metrics = []
                for row in cursor.fetchall():
                    tags = json.loads(row[5]) if row[5] else {}
                    metric = HealthMetric(
                        name=row[0],
                        value=row[1],
                        unit=row[2],
                        timestamp=datetime.fromisoformat(row[3]),
                        instance_id=row[4],
                        tags=tags
                    )
                    metrics.append(metric)
                
                return metrics
        except Exception as e:
            self.logger.error(f"Failed to get recent metrics: {e}")
            return []
    
    def _get_uptime_seconds(self) -> float:
        """Get monitoring uptime in seconds."""
        # This is a simplified implementation
        # In a real system, you'd track the actual start time
        return time.time() % (24 * 3600)  # Uptime within current day
    
    def _calculate_health_score(self, metrics: List[HealthMetric]) -> float:
        """Calculate overall health score (0-100)."""
        if not metrics:
            return 0.0
        
        try:
            # Simple health score based on key metrics
            score = 100.0
            
            # Get latest values for key metrics
            latest_metrics = {}
            for metric in metrics:
                if metric.name not in latest_metrics or metric.timestamp > latest_metrics[metric.name].timestamp:
                    latest_metrics[metric.name] = metric
            
            # Deduct points for high resource usage
            if "cpu_usage_percent" in latest_metrics:
                cpu_usage = latest_metrics["cpu_usage_percent"].value
                if cpu_usage > 80:
                    score -= (cpu_usage - 80) * 0.5
            
            if "memory_usage_percent" in latest_metrics:
                memory_usage = latest_metrics["memory_usage_percent"].value
                if memory_usage > 80:
                    score -= (memory_usage - 80) * 0.5
            
            if "disk_usage_percent" in latest_metrics:
                disk_usage = latest_metrics["disk_usage_percent"].value
                if disk_usage > 85:
                    score -= (disk_usage - 85) * 1.0
            
            # Deduct points for active alerts
            score -= len(self.alert_manager.active_alerts) * 10
            
            return max(0.0, min(100.0, score))
            
        except Exception as e:
            self.logger.error(f"Failed to calculate health score: {e}")
            return 50.0  # Default neutral score


def main():
    """Main function for running production monitoring."""
    import argparse
    
    parser = argparse.ArgumentParser(description="WAN2.2 Production Monitoring")
    parser.add_argument("--config", "-c", help="Configuration file path")
    parser.add_argument("--daemon", "-d", action="store_true", help="Run as daemon")
    parser.add_argument("--status", "-s", action="store_true", help="Show status and exit")
    
    args = parser.parse_args()
    
    monitor = ProductionMonitor(args.config)
    
    if args.status:
        status = monitor.get_health_status()
        print(json.dumps(status, indent=2))
        return
    
    try:
        monitor.start()
        
        if args.daemon:
            # Run as daemon
            while True:
                time.sleep(60)
        else:
            # Interactive mode
            print("Production monitoring started. Press Ctrl+C to stop.")
            while True:
                time.sleep(1)
                
    except KeyboardInterrupt:
        print("\nStopping monitoring...")
        monitor.stop()
        print("Monitoring stopped.")


if __name__ == "__main__":
    main()