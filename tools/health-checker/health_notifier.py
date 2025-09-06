"""
Health notification and alerting system
"""

import json
import smtplib
import subprocess
from datetime import datetime, timedelta
from email.mime.text import MIMEText as MimeText
from email.mime.multipart import MIMEMultipart as MimeMultipart
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
import logging

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from health_models import (
    HealthReport, HealthIssue, ComponentHealth, 
    HealthCategory, Severity, HealthConfig
)


class NotificationChannel:
    """Base class for notification channels"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    async def send_notification(self, message: str, severity: Severity, metadata: Dict[str, Any] = None) -> bool:
        """Send notification through this channel"""
        raise NotImplementedError
    
    def is_enabled(self) -> bool:
        """Check if this channel is enabled"""
        return self.config.get("enabled", True)


class ConsoleNotificationChannel(NotificationChannel):
    """Console/stdout notification channel"""
    
    async def send_notification(self, message: str, severity: Severity, metadata: Dict[str, Any] = None) -> bool:
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            severity_colors = {
                Severity.CRITICAL: "\\033[91m",  # Red
                Severity.HIGH: "\\033[93m",      # Yellow
                Severity.MEDIUM: "\\033[94m",    # Blue
                Severity.LOW: "\\033[92m",       # Green
                Severity.INFO: "\\033[0m"        # Default
            }
            
            color = severity_colors.get(severity, "\\033[0m")
            reset_color = "\\033[0m"
            
            print(f"{color}[{timestamp}] {severity.value.upper()}: {message}{reset_color}")
            
            if metadata:
                for key, value in metadata.items():
                    print(f"  {key}: {value}")
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to send console notification: {e}")
            return False


class EmailNotificationChannel(NotificationChannel):
    """Email notification channel"""
    
    async def send_notification(self, message: str, severity: Severity, metadata: Dict[str, Any] = None) -> bool:
        try:
            smtp_config = self.config.get("smtp", {})
            if not smtp_config:
                self.logger.warning("SMTP configuration not provided")
                return False
            
            # Create email message
            msg = MimeMultipart()
            msg['From'] = smtp_config.get("from_email", "health-monitor@localhost")
            msg['To'] = ", ".join(self.config.get("recipients", []))
            msg['Subject'] = f"[{severity.value.upper()}] Project Health Alert"
            
            # Create email body
            body = self._create_email_body(message, severity, metadata)
            msg.attach(MimeText(body, 'html'))
            
            # Send email
            server = smtplib.SMTP(smtp_config.get("host", "localhost"), smtp_config.get("port", 587))
            
            if smtp_config.get("use_tls", True):
                server.starttls()
            
            if smtp_config.get("username") and smtp_config.get("password"):
                server.login(smtp_config["username"], smtp_config["password"])
            
            server.send_message(msg)
            server.quit()
            
            self.logger.info(f"Email notification sent to {len(self.config.get('recipients', []))} recipients")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send email notification: {e}")
            return False
    
    def _create_email_body(self, message: str, severity: Severity, metadata: Dict[str, Any] = None) -> str:
        """Create HTML email body"""
        severity_colors = {
            Severity.CRITICAL: "#e74c3c",
            Severity.HIGH: "#f39c12",
            Severity.MEDIUM: "#3498db",
            Severity.LOW: "#27ae60",
            Severity.INFO: "#95a5a6"
        }
        
        color = severity_colors.get(severity, "#95a5a6")
        
        html = f"""
        <html>
        <body style="font-family: Arial, sans-serif; margin: 20px;">
            <div style="border-left: 4px solid {color}; padding: 20px; background: #f8f9fa;">
                <h2 style="color: {color}; margin-top: 0;">Project Health Alert</h2>
                <p><strong>Severity:</strong> {severity.value.upper()}</p>
                <p><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Message:</strong> {message}</p>
        """
        
        if metadata:
            html += "<h3>Additional Information:</h3><ul>"
            for key, value in metadata.items():
                html += f"<li><strong>{key}:</strong> {value}</li>"
            html += "</ul>"
        
        html += """
            </div>
            <p style="color: #666; font-size: 12px; margin-top: 20px;">
                This is an automated message from the Project Health Monitoring System.
            </p>
        </body>
        </html>
        """
        
        return html


class SlackNotificationChannel(NotificationChannel):
    """Slack notification channel"""
    
    async def send_notification(self, message: str, severity: Severity, metadata: Dict[str, Any] = None) -> bool:
        if not REQUESTS_AVAILABLE:
            self.logger.error("requests library not available for Slack notifications")
            return False
        
        try:
            webhook_url = self.config.get("webhook_url")
            if not webhook_url:
                self.logger.warning("Slack webhook URL not configured")
                return False
            
            # Create Slack message
            slack_message = self._create_slack_message(message, severity, metadata)
            
            response = requests.post(webhook_url, json=slack_message, timeout=10)
            response.raise_for_status()
            
            self.logger.info("Slack notification sent successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send Slack notification: {e}")
            return False
    
    def _create_slack_message(self, message: str, severity: Severity, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create Slack message payload"""
        severity_colors = {
            Severity.CRITICAL: "danger",
            Severity.HIGH: "warning",
            Severity.MEDIUM: "good",
            Severity.LOW: "good",
            Severity.INFO: "#36a64f"
        }
        
        color = severity_colors.get(severity, "good")
        
        attachment = {
            "color": color,
            "title": f"Project Health Alert - {severity.value.upper()}",
            "text": message,
            "timestamp": int(datetime.now().timestamp()),
            "fields": []
        }
        
        if metadata:
            for key, value in metadata.items():
                attachment["fields"].append({
                    "title": key,
                    "value": str(value),
                    "short": True
                })
        
        return {
            "username": "Health Monitor",
            "icon_emoji": ":warning:",
            "attachments": [attachment]
        }


class WebhookNotificationChannel(NotificationChannel):
    """Generic webhook notification channel"""
    
    async def send_notification(self, message: str, severity: Severity, metadata: Dict[str, Any] = None) -> bool:
        if not REQUESTS_AVAILABLE:
            self.logger.error("requests library not available for webhook notifications")
            return False
        
        try:
            webhook_url = self.config.get("url")
            if not webhook_url:
                self.logger.warning("Webhook URL not configured")
                return False
            
            payload = {
                "message": message,
                "severity": severity.value,
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata or {}
            }
            
            headers = self.config.get("headers", {})
            timeout = self.config.get("timeout", 10)
            
            response = requests.post(webhook_url, json=payload, headers=headers, timeout=timeout)
            response.raise_for_status()
            
            self.logger.info("Webhook notification sent successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send webhook notification: {e}")
            return False


class FileNotificationChannel(NotificationChannel):
    """File-based notification channel"""
    
    async def send_notification(self, message: str, severity: Severity, metadata: Dict[str, Any] = None) -> bool:
        try:
            log_file = Path(self.config.get("file_path", "health_notifications.log"))
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = {
                "timestamp": timestamp,
                "severity": severity.value,
                "message": message,
                "metadata": metadata or {}
            }
            
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + "\n")
            
            self.logger.info(f"Notification logged to {log_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to write file notification: {e}")
            return False


class HealthNotifier:
    """
    Main health notification system that manages multiple channels and alert rules
    """
    
    def __init__(self, config: Optional[HealthConfig] = None):
        self.config = config or HealthConfig()
        self.logger = logging.getLogger(__name__)
        
        # Notification channels
        self.channels = {}
        self._initialize_channels()
        
        # Alert rules and escalation policies
        self.alert_rules = self._load_alert_rules()
        self.notification_history = []
        
        # Rate limiting
        self.last_notifications = {}
        self.rate_limit_window = timedelta(minutes=15)  # Don't spam notifications
    
    def _initialize_channels(self):
        """Initialize notification channels"""
        # Default console channel
        self.channels["console"] = ConsoleNotificationChannel("console", {"enabled": True})
        
        # Load additional channels from config
        notification_config = getattr(self.config, 'notification_config', {})
        
        if "email" in notification_config:
            self.channels["email"] = EmailNotificationChannel("email", notification_config["email"])
        
        if "slack" in notification_config:
            self.channels["slack"] = SlackNotificationChannel("slack", notification_config["slack"])
        
        if "webhook" in notification_config:
            self.channels["webhook"] = WebhookNotificationChannel("webhook", notification_config["webhook"])
        
        if "file" in notification_config:
            self.channels["file"] = FileNotificationChannel("file", notification_config["file"])
    
    def _load_alert_rules(self) -> List[Dict[str, Any]]:
        """Load alert rules configuration"""
        # Default alert rules
        default_rules = [
            {
                "name": "critical_health_score",
                "condition": lambda report: report.overall_score < 30,
                "severity": Severity.CRITICAL,
                "message": "Project health score is critically low",
                "channels": ["console", "email", "slack"],
                "rate_limit": timedelta(hours=1)
            },
            {
                "name": "high_issue_count",
                "condition": lambda report: len(report.get_critical_issues()) > 0,
                "severity": Severity.HIGH,
                "message": "Critical issues detected in project health check",
                "channels": ["console", "email"],
                "rate_limit": timedelta(minutes=30)
            },
            {
                "name": "declining_trend",
                "condition": lambda report: report.trends.improvement_rate < -5,
                "severity": Severity.MEDIUM,
                "message": "Project health showing strong declining trend",
                "channels": ["console", "slack"],
                "rate_limit": timedelta(hours=2)
            },
            {
                "name": "component_failure",
                "condition": lambda report: any(comp.score < 25 for comp in report.component_scores.values()),
                "severity": Severity.HIGH,
                "message": "One or more components in critical failure state",
                "channels": ["console", "email"],
                "rate_limit": timedelta(minutes=45)
            }
        ]
        
        # TODO: Load custom rules from configuration file
        return default_rules
    
    async def process_health_report(self, report: HealthReport) -> List[str]:
        """
        Process health report and send notifications based on alert rules
        
        Returns:
            List of notifications sent
        """
        notifications_sent = []
        
        for rule in self.alert_rules:
            try:
                if self._should_trigger_alert(rule, report):
                    notification_id = await self._send_alert(rule, report)
                    if notification_id:
                        notifications_sent.append(notification_id)
            except Exception as e:
                self.logger.error(f"Failed to process alert rule {rule['name']}: {e}")
        
        return notifications_sent
    
    def _should_trigger_alert(self, rule: Dict[str, Any], report: HealthReport) -> bool:
        """Check if alert rule should trigger"""
        try:
            # Check condition
            if not rule["condition"](report):
                return False
            
            # Check rate limiting
            rule_name = rule["name"]
            rate_limit = rule.get("rate_limit", timedelta(minutes=15))
            
            if rule_name in self.last_notifications:
                time_since_last = datetime.now() - self.last_notifications[rule_name]
                if time_since_last < rate_limit:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking alert rule {rule['name']}: {e}")
            return False
    
    async def _send_alert(self, rule: Dict[str, Any], report: HealthReport) -> Optional[str]:
        """Send alert notification"""
        try:
            message = self._create_alert_message(rule, report)
            severity = rule["severity"]
            channels = rule.get("channels", ["console"])
            
            metadata = {
                "rule_name": rule["name"],
                "overall_score": report.overall_score,
                "critical_issues": len(report.get_critical_issues()),
                "timestamp": report.timestamp.isoformat()
            }
            
            # Send to specified channels
            success_count = 0
            for channel_name in channels:
                if channel_name in self.channels:
                    channel = self.channels[channel_name]
                    if channel.is_enabled():
                        success = await channel.send_notification(message, severity, metadata)
                        if success:
                            success_count += 1
            
            if success_count > 0:
                # Update rate limiting
                self.last_notifications[rule["name"]] = datetime.now()
                
                # Log notification
                notification_id = f"{rule['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                self.notification_history.append({
                    "id": notification_id,
                    "rule": rule["name"],
                    "message": message,
                    "severity": severity.value,
                    "channels": channels,
                    "timestamp": datetime.now().isoformat(),
                    "success_count": success_count
                })
                
                self.logger.info(f"Alert sent: {rule['name']} to {success_count} channels")
                return notification_id
            
        except Exception as e:
            self.logger.error(f"Failed to send alert for rule {rule['name']}: {e}")
        
        return None
    
    def _create_alert_message(self, rule: Dict[str, Any], report: HealthReport) -> str:
        """Create alert message based on rule and report"""
        base_message = rule["message"]
        
        # Add context based on rule type
        if rule["name"] == "critical_health_score":
            return f"{base_message}: {report.overall_score:.1f}/100"
        
        elif rule["name"] == "high_issue_count":
            critical_count = len(report.get_critical_issues())
            return f"{base_message}: {critical_count} critical issues found"
        
        elif rule["name"] == "declining_trend":
            return f"{base_message}: {report.trends.improvement_rate:.2f} points per check"
        
        elif rule["name"] == "component_failure":
            failed_components = [
                name for name, comp in report.component_scores.items() 
                if comp.score < 25
            ]
            return f"{base_message}: {', '.join(failed_components)}"
        
        return base_message
    
    async def send_custom_notification(
        self, 
        message: str, 
        severity: Severity = Severity.INFO,
        channels: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Send a custom notification"""
        channels = channels or ["console"]
        success_count = 0
        
        for channel_name in channels:
            if channel_name in self.channels:
                channel = self.channels[channel_name]
                if channel.is_enabled():
                    success = await channel.send_notification(message, severity, metadata)
                    if success:
                        success_count += 1
        
        return success_count > 0
    
    def get_notification_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent notification history"""
        return self.notification_history[-limit:]
    
    def add_custom_channel(self, name: str, channel: NotificationChannel):
        """Add a custom notification channel"""
        self.channels[name] = channel
        self.logger.info(f"Added custom notification channel: {name}")
    
    def add_custom_rule(self, rule: Dict[str, Any]):
        """Add a custom alert rule"""
        required_fields = ["name", "condition", "severity", "message"]
        
        if not all(field in rule for field in required_fields):
            raise ValueError(f"Alert rule must contain: {required_fields}")
        
        self.alert_rules.append(rule)
        self.logger.info(f"Added custom alert rule: {rule['name']}")
    
    def test_notifications(self) -> Dict[str, bool]:
        """Test all notification channels"""
        results = {}
        
        for name, channel in self.channels.items():
            if channel.is_enabled():
                try:
                    # Use asyncio to run the async method
                    import asyncio
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    success = loop.run_until_complete(
                        channel.send_notification(
                            "Test notification from health monitoring system",
                            Severity.INFO,
                            {"test": True, "timestamp": datetime.now().isoformat()}
                        )
                    )
                    
                    loop.close()
                    results[name] = success
                    
                except Exception as e:
                    self.logger.error(f"Failed to test channel {name}: {e}")
                    results[name] = False
            else:
                results[name] = False
        
        return results


# CI/CD Integration helpers
class CIPipelineIntegration:
    """Integration with CI/CD pipelines"""
    
    def __init__(self, notifier: HealthNotifier):
        self.notifier = notifier
        self.logger = logging.getLogger(__name__)
    
    async def check_health_gate(self, report: HealthReport, threshold: float = 75.0) -> bool:
        """
        Check if health meets threshold for CI/CD gate
        
        Returns:
            True if health passes gate, False otherwise
        """
        passes_gate = report.overall_score >= threshold
        
        if not passes_gate:
            await self.notifier.send_custom_notification(
                f"Health gate failed: Score {report.overall_score:.1f} below threshold {threshold}",
                Severity.HIGH,
                ["console"],
                {
                    "gate_threshold": threshold,
                    "actual_score": report.overall_score,
                    "critical_issues": len(report.get_critical_issues())
                }
            )
        
        return passes_gate
    
    def create_github_status(self, report: HealthReport) -> Dict[str, Any]:
        """Create GitHub status check payload"""
        if report.overall_score >= 75:
            state = "success"
            description = f"Health score: {report.overall_score:.1f}/100"
        elif report.overall_score >= 50:
            state = "failure"
            description = f"Health issues detected: {report.overall_score:.1f}/100"
        else:
            state = "failure"
            description = f"Critical health issues: {report.overall_score:.1f}/100"
        
        return {
            "state": state,
            "description": description,
            "context": "project-health/check",
            "target_url": "https://your-dashboard-url.com"
        }
    
    def set_exit_code(self, report: HealthReport, threshold: float = 50.0) -> int:
        """Set appropriate exit code for CI/CD"""
        if report.overall_score >= threshold:
            return 0  # Success
        else:
            return 1  # Failure