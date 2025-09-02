"""
Quality monitoring alert system.
"""

import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

try:
    from tools.quality-monitor.models import (
        QualityAlert, QualityMetric, QualityThreshold, QualityTrend,
        AlertSeverity, MetricType, TrendDirection
    )
except ImportError:
    from models import (
        QualityAlert, QualityMetric, QualityThreshold, QualityTrend,
        AlertSeverity, MetricType, TrendDirection
    )


class AlertSystem:
    """Quality monitoring alert system."""
    
    def __init__(self, config_file: str = "config/quality-alerts.json"):
        self.config_file = Path(config_file)
        self.alerts_dir = Path("data/quality-alerts")
        self.alerts_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = self._load_config()
        self.active_alerts: Dict[str, QualityAlert] = {}
        self._load_active_alerts()
    
    def _load_config(self) -> Dict:
        """Load alert configuration."""
        default_config = {
            "thresholds": {
                "test_coverage": {"warning": 70.0, "critical": 50.0},
                "code_complexity": {"warning": 10.0, "critical": 15.0},
                "documentation_coverage": {"warning": 60.0, "critical": 40.0},
                "duplicate_code": {"warning": 10.0, "critical": 20.0},
                "style_violations": {"warning": 50.0, "critical": 100.0},
                "type_hint_coverage": {"warning": 50.0, "critical": 30.0}
            },
            "notifications": {
                "email": {
                    "enabled": False,
                    "smtp_server": "localhost",
                    "smtp_port": 587,
                    "username": "",
                    "password": "",
                    "recipients": []
                },
                "webhook": {
                    "enabled": False,
                    "url": "",
                    "headers": {}
                }
            },
            "alert_cooldown_hours": 24,
            "trend_alert_threshold": 5.0  # Percentage change to trigger trend alert
        }
        
        if self.config_file.exists():
            try:
                with open(self.config_file) as f:
                    config = json.load(f)
                # Merge with defaults
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
            except Exception as e:
                print(f"Error loading alert config: {e}")
        
        # Create default config file
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        return default_config
    
    def _load_active_alerts(self) -> None:
        """Load active alerts from storage."""
        alerts_file = self.alerts_dir / "active_alerts.json"
        if alerts_file.exists():
            try:
                with open(alerts_file) as f:
                    data = json.load(f)
                
                for alert_data in data.get('alerts', []):
                    alert = QualityAlert(
                        id=alert_data['id'],
                        severity=AlertSeverity(alert_data['severity']),
                        metric_type=MetricType(alert_data['metric_type']),
                        message=alert_data['message'],
                        description=alert_data['description'],
                        current_value=alert_data['current_value'],
                        threshold_value=alert_data['threshold_value'],
                        component=alert_data.get('component'),
                        timestamp=datetime.fromisoformat(alert_data['timestamp']),
                        resolved=alert_data.get('resolved', False),
                        recommendations=alert_data.get('recommendations', [])
                    )
                    self.active_alerts[alert.id] = alert
            
            except Exception as e:
                print(f"Error loading active alerts: {e}")
    
    def _save_active_alerts(self) -> None:
        """Save active alerts to storage."""
        alerts_file = self.alerts_dir / "active_alerts.json"
        data = {
            'alerts': [alert.to_dict() for alert in self.active_alerts.values()],
            'last_updated': datetime.now().isoformat()
        }
        
        with open(alerts_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_thresholds(self) -> List[QualityThreshold]:
        """Get configured quality thresholds."""
        thresholds = []
        
        for metric_name, values in self.config['thresholds'].items():
            try:
                metric_type = MetricType(metric_name)
                threshold = QualityThreshold(
                    metric_type=metric_type,
                    warning_threshold=values['warning'],
                    critical_threshold=values['critical']
                )
                thresholds.append(threshold)
            except ValueError:
                continue
        
        return thresholds
    
    def check_metric_alerts(self, metrics: List[QualityMetric]) -> List[QualityAlert]:
        """Check metrics against thresholds and generate alerts."""
        new_alerts = []
        thresholds = {t.metric_type: t for t in self.get_thresholds()}
        
        for metric in metrics:
            if metric.metric_type not in thresholds:
                continue
            
            threshold = thresholds[metric.metric_type]
            alert = self._check_single_metric(metric, threshold)
            
            if alert:
                # Check if we already have an active alert for this metric
                existing_alert = self._find_existing_alert(metric.metric_type, metric.component)
                
                if existing_alert:
                    # Update existing alert
                    existing_alert.current_value = metric.value
                    existing_alert.timestamp = datetime.now()
                    existing_alert.resolved = False
                else:
                    # Create new alert
                    self.active_alerts[alert.id] = alert
                    new_alerts.append(alert)
        
        self._save_active_alerts()
        return new_alerts
    
    def _check_single_metric(self, metric: QualityMetric, threshold: QualityThreshold) -> Optional[QualityAlert]:
        """Check a single metric against its threshold."""
        severity = None
        threshold_value = None
        
        # Determine if metric violates thresholds
        if metric.metric_type in [MetricType.TEST_COVERAGE, MetricType.DOCUMENTATION_COVERAGE, MetricType.TYPE_HINT_COVERAGE]:
            # Higher is better metrics
            if metric.value < threshold.critical_threshold:
                severity = AlertSeverity.CRITICAL
                threshold_value = threshold.critical_threshold
            elif metric.value < threshold.warning_threshold:
                severity = AlertSeverity.MEDIUM
                threshold_value = threshold.warning_threshold
        else:
            # Lower is better metrics
            if metric.value > threshold.critical_threshold:
                severity = AlertSeverity.CRITICAL
                threshold_value = threshold.critical_threshold
            elif metric.value > threshold.warning_threshold:
                severity = AlertSeverity.MEDIUM
                threshold_value = threshold.warning_threshold
        
        if severity:
            recommendations = self._generate_recommendations(metric.metric_type, metric.value, threshold_value)
            
            return QualityAlert(
                id=str(uuid.uuid4()),
                severity=severity,
                metric_type=metric.metric_type,
                message=f"{metric.metric_type.value.replace('_', ' ').title()} threshold exceeded",
                description=f"Current value: {metric.value:.2f}, Threshold: {threshold_value:.2f}",
                current_value=metric.value,
                threshold_value=threshold_value,
                component=metric.component,
                recommendations=recommendations
            )
        
        return None
    
    def check_trend_alerts(self, trends: List[QualityTrend]) -> List[QualityAlert]:
        """Check trends for concerning patterns and generate alerts."""
        new_alerts = []
        trend_threshold = self.config.get('trend_alert_threshold', 5.0)
        
        for trend in trends:
            if (trend.direction == TrendDirection.DEGRADING and 
                abs(trend.change_rate) > trend_threshold and 
                trend.confidence > 0.5):
                
                alert = QualityAlert(
                    id=str(uuid.uuid4()),
                    severity=AlertSeverity.MEDIUM if abs(trend.change_rate) < trend_threshold * 2 else AlertSeverity.HIGH,
                    metric_type=trend.metric_type,
                    message=f"{trend.metric_type.value.replace('_', ' ').title()} is degrading",
                    description=f"Trend shows {trend.change_rate:.2f}% change per day over {trend.time_period_days} days",
                    current_value=trend.current_value,
                    threshold_value=trend.previous_value,
                    recommendations=self._generate_trend_recommendations(trend)
                )
                
                # Check if we already have a similar trend alert
                existing_alert = self._find_existing_trend_alert(trend.metric_type)
                if not existing_alert:
                    self.active_alerts[alert.id] = alert
                    new_alerts.append(alert)
        
        self._save_active_alerts()
        return new_alerts
    
    def _find_existing_alert(self, metric_type: MetricType, component: Optional[str]) -> Optional[QualityAlert]:
        """Find existing alert for metric type and component."""
        for alert in self.active_alerts.values():
            if (alert.metric_type == metric_type and 
                alert.component == component and 
                not alert.resolved):
                return alert
        return None
    
    def _find_existing_trend_alert(self, metric_type: MetricType) -> Optional[QualityAlert]:
        """Find existing trend alert for metric type."""
        cooldown_hours = self.config.get('alert_cooldown_hours', 24)
        cutoff_time = datetime.now() - timedelta(hours=cooldown_hours)
        
        for alert in self.active_alerts.values():
            if (alert.metric_type == metric_type and 
                "degrading" in alert.message.lower() and
                alert.timestamp > cutoff_time):
                return alert
        return None
    
    def _generate_recommendations(self, metric_type: MetricType, current_value: float, threshold_value: float) -> List[str]:
        """Generate recommendations for metric alerts."""
        recommendations = []
        
        if metric_type == MetricType.TEST_COVERAGE:
            recommendations = [
                "Add unit tests for uncovered code paths",
                "Review and improve existing test quality",
                "Consider test-driven development for new features",
                "Use coverage tools to identify specific gaps"
            ]
        elif metric_type == MetricType.CODE_COMPLEXITY:
            recommendations = [
                "Refactor complex functions into smaller units",
                "Extract common logic into helper functions",
                "Consider using design patterns to reduce complexity",
                "Review and simplify conditional logic"
            ]
        elif metric_type == MetricType.DOCUMENTATION_COVERAGE:
            recommendations = [
                "Add docstrings to undocumented functions and classes",
                "Improve existing documentation quality",
                "Use type hints to enhance code documentation",
                "Consider automated documentation generation"
            ]
        elif metric_type == MetricType.DUPLICATE_CODE:
            recommendations = [
                "Extract duplicate code into shared functions",
                "Use inheritance or composition to reduce duplication",
                "Consider refactoring similar code patterns",
                "Review and consolidate similar implementations"
            ]
        elif metric_type == MetricType.STYLE_VIOLATIONS:
            recommendations = [
                "Run automated code formatting tools",
                "Set up pre-commit hooks for style checking",
                "Review and fix style guide violations",
                "Configure IDE for automatic style enforcement"
            ]
        elif metric_type == MetricType.TYPE_HINT_COVERAGE:
            recommendations = [
                "Add type hints to function parameters and return values",
                "Use mypy or similar tools for type checking",
                "Gradually migrate legacy code to use type hints",
                "Consider using dataclasses for structured data"
            ]
        
        return recommendations
    
    def _generate_trend_recommendations(self, trend: QualityTrend) -> List[str]:
        """Generate recommendations for trend alerts."""
        base_recommendations = self._generate_recommendations(trend.metric_type, trend.current_value, trend.previous_value)
        
        trend_specific = [
            f"Monitor {trend.metric_type.value.replace('_', ' ')} closely over the next few days",
            "Consider implementing automated quality gates",
            "Review recent code changes that may have impacted quality",
            "Set up more frequent quality monitoring"
        ]
        
        return base_recommendations + trend_specific
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Mark an alert as resolved."""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].resolved = True
            self._save_active_alerts()
            return True
        return False
    
    def get_active_alerts(self) -> List[QualityAlert]:
        """Get all active (unresolved) alerts."""
        return [alert for alert in self.active_alerts.values() if not alert.resolved]
    
    def get_alert_summary(self) -> Dict[str, any]:
        """Get summary of alert status."""
        active_alerts = self.get_active_alerts()
        
        return {
            'total_active': len(active_alerts),
            'critical': len([a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]),
            'high': len([a for a in active_alerts if a.severity == AlertSeverity.HIGH]),
            'medium': len([a for a in active_alerts if a.severity == AlertSeverity.MEDIUM]),
            'low': len([a for a in active_alerts if a.severity == AlertSeverity.LOW]),
            'alerts': [alert.to_dict() for alert in active_alerts],
            'last_updated': datetime.now().isoformat()
        }
    
    def send_notifications(self, alerts: List[QualityAlert]) -> None:
        """Send notifications for new alerts."""
        if not alerts:
            return
        
        # Email notifications
        if self.config['notifications']['email']['enabled']:
            self._send_email_notifications(alerts)
        
        # Webhook notifications
        if self.config['notifications']['webhook']['enabled']:
            self._send_webhook_notifications(alerts)
    
    def _send_email_notifications(self, alerts: List[QualityAlert]) -> None:
        """Send email notifications for alerts."""
        try:
            email_config = self.config['notifications']['email']
            
            msg = MIMEMultipart()
            msg['From'] = email_config['username']
            msg['To'] = ', '.join(email_config['recipients'])
            msg['Subject'] = f"Quality Alert: {len(alerts)} new alert(s)"
            
            body = "Quality monitoring has detected the following issues:\n\n"
            
            for alert in alerts:
                body += f"• {alert.severity.value.upper()}: {alert.message}\n"
                body += f"  {alert.description}\n"
                if alert.recommendations:
                    body += f"  Recommendations: {', '.join(alert.recommendations[:2])}\n"
                body += "\n"
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
            server.starttls()
            server.login(email_config['username'], email_config['password'])
            server.send_message(msg)
            server.quit()
            
        except Exception as e:
            print(f"Error sending email notifications: {e}")
    
    def _send_webhook_notifications(self, alerts: List[QualityAlert]) -> None:
        """Send webhook notifications for alerts."""
        try:
            import requests
            
            webhook_config = self.config['notifications']['webhook']
            
            payload = {
                'alerts': [alert.to_dict() for alert in alerts],
                'timestamp': datetime.now().isoformat(),
                'summary': f"{len(alerts)} new quality alert(s)"
            }
            
            response = requests.post(
                webhook_config['url'],
                json=payload,
                headers=webhook_config.get('headers', {}),
                timeout=30
            )
            
            response.raise_for_status()
            
        except Exception as e:
            print(f"Error sending webhook notifications: {e}")
    
    def cleanup_resolved_alerts(self, days: int = 30) -> int:
        """Clean up old resolved alerts."""
        cutoff_date = datetime.now() - timedelta(days=days)
        removed_count = 0
        
        alerts_to_remove = []
        for alert_id, alert in self.active_alerts.items():
            if alert.resolved and alert.timestamp < cutoff_date:
                alerts_to_remove.append(alert_id)
        
        for alert_id in alerts_to_remove:
            del self.active_alerts[alert_id]
            removed_count += 1
        
        if removed_count > 0:
            self._save_active_alerts()
        
        return removed_count