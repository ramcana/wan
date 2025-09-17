from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass
from collections import defaultdict, deque
import threading


@dataclass
class SecurityEvent:
    """Represents a security-related event."""

    event_type: str
    timestamp: datetime
    user_id: Optional[str]
    ip_address: Optional[str]
    details: Dict[str, Any]
    severity: str  # 'low', 'medium', 'high', 'critical'


@dataclass
class SecurityAlert:
    """Represents a security alert."""

    alert_type: str
    timestamp: datetime
    user_id: Optional[str]
    ip_address: Optional[str]
    message: str
    details: Dict[str, Any]
    resolved: bool = False


class SecurityMonitor:
    """Security monitoring service for tracking and alerting on
    security events."""

    def __init__(self, max_events: int = 10000):
        self.events: deque[SecurityEvent] = deque(maxlen=max_events)
        self.alerts: List[SecurityAlert] = []
        self.ip_login_attempts: Dict[str, List[datetime]] = defaultdict(list)
        self.user_login_attempts: Dict[str, List[datetime]] = defaultdict(list)
        self.suspicious_ips: set = set()
        self.suspicious_users: set = set()
        self.logger = logging.getLogger(__name__)
        self._lock = threading.Lock()

    def log_event(
        self,
        event_type: str,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        severity: str = "low",
    ) -> None:
        """Log a security event."""
        if details is None:
            details = {}

        event = SecurityEvent(
            event_type=event_type,
            timestamp=datetime.utcnow(),
            user_id=user_id,
            ip_address=ip_address,
            details=details,
            severity=severity,
        )

        with self._lock:
            self.events.append(event)

        # Log to standard logger as well
        log_level = getattr(logging, severity.upper(), logging.INFO)
        log_message = (
            f"Security Event: {event_type} - User: {user_id}, "
            f"IP: {ip_address}, Details: {details}"
        )
        self.logger.log(log_level, log_message)

    def check_suspicious_activity(
        self, user_id: Optional[str] = None, ip_address: Optional[str] = None
    ) -> List[str]:
        """Check for suspicious activity patterns."""
        warnings = []

        # Check for excessive login attempts from IP
        if ip_address:
            recent_attempts = self._get_recent_attempts(
                self.ip_login_attempts[ip_address]
            )
            # More than 10 attempts in 15 minutes
            if len(recent_attempts) > 10:
                msg = f"Excessive login attempts from IP {ip_address}"
                warnings.append(msg)
                self.suspicious_ips.add(ip_address)
                self.log_event(
                    "suspicious_ip_activity",
                    ip_address=ip_address,
                    details={"attempts_count": len(recent_attempts)},
                    severity="high",
                )

        # Check for excessive login attempts for user
        if user_id:
            recent_attempts = self._get_recent_attempts(
                self.user_login_attempts[user_id]
            )
            # More than 5 attempts in 15 minutes
            if len(recent_attempts) > 5:
                msg = f"Excessive login attempts for user {user_id}"
                warnings.append(msg)
                self.suspicious_users.add(user_id)
                self.log_event(
                    "suspicious_user_activity",
                    user_id=user_id,
                    details={"attempts_count": len(recent_attempts)},
                    severity="high",
                )

        return warnings

    def _get_recent_attempts(self, attempts: List[datetime]) -> List[datetime]:
        """Get attempts from the last 15 minutes."""
        cutoff = datetime.utcnow() - timedelta(minutes=15)
        return [attempt for attempt in attempts if attempt > cutoff]

    def record_login_attempt(
        self,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        success: bool = False,
    ) -> None:
        """Record a login attempt."""
        timestamp = datetime.utcnow()

        with self._lock:
            if ip_address:
                self.ip_login_attempts[ip_address].append(timestamp)
                # Keep only recent attempts
                self.ip_login_attempts[ip_address] = self._get_recent_attempts(
                    self.ip_login_attempts[ip_address]
                )

            if user_id:
                self.user_login_attempts[user_id].append(timestamp)
                # Keep only recent attempts
                self.user_login_attempts[user_id] = self._get_recent_attempts(
                    self.user_login_attempts[user_id]
                )

        # Log the attempt
        self.log_event(
            "successful_login" if success else "failed_login",
            user_id=user_id,
            ip_address=ip_address,
            details={"success": success},
            severity="low" if success else "medium",
        )

        # Check for suspicious activity
        if not success:
            warnings = self.check_suspicious_activity(user_id, ip_address)
            if warnings:
                for warning in warnings:
                    self.create_alert(
                        "suspicious_activity",
                        user_id=user_id,
                        ip_address=ip_address,
                        message=warning,
                        details={"login_success": success},
                    )

    def create_alert(
        self,
        alert_type: str,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        message: str = "",
        details: Optional[Dict[str, Any]] = None,
    ) -> SecurityAlert:
        """Create a security alert."""
        if details is None:
            details = {}

        alert = SecurityAlert(
            alert_type=alert_type,
            timestamp=datetime.utcnow(),
            user_id=user_id,
            ip_address=ip_address,
            message=message,
            details=details,
        )

        with self._lock:
            self.alerts.append(alert)

        # Log the alert
        self.logger.warning(f"Security Alert: {alert_type} - {message}")

        return alert

    def resolve_alert(self, alert_index: int) -> bool:
        """Mark an alert as resolved."""
        with self._lock:
            if 0 <= alert_index < len(self.alerts):
                self.alerts[alert_index].resolved = True
                return True
        return False

    def get_recent_events(self, hours: int = 24) -> List[SecurityEvent]:
        """Get security events from the last N hours."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        with self._lock:
            return [event for event in self.events if event.timestamp > cutoff]

    def get_unresolved_alerts(self) -> List[SecurityAlert]:
        """Get all unresolved alerts."""
        with self._lock:
            return [alert for alert in self.alerts if not alert.resolved]

    def get_suspicious_ips(self) -> set:
        """Get currently flagged suspicious IPs."""
        with self._lock:
            return self.suspicious_ips.copy()

    def get_suspicious_users(self) -> set:
        """Get currently flagged suspicious users."""
        with self._lock:
            return self.suspicious_users.copy()

    def clear_suspicious_flags(
        self, ip_address: Optional[str] = None, user_id: Optional[str] = None
    ) -> None:
        """Clear suspicious flags for an IP or user."""
        with self._lock:
            if ip_address:
                self.suspicious_ips.discard(ip_address)
            if user_id:
                self.suspicious_users.discard(user_id)

    def log_api_key_usage(
        self, api_key_id: str, user_id: str, ip_address: Optional[str] = None
    ) -> None:
        """Log API key usage."""
        self.log_event(
            "api_key_usage",
            user_id=user_id,
            ip_address=ip_address,
            details={"api_key_id": api_key_id},
            severity="low",
        )

    def log_rate_limit_exceeded(
        self,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        endpoint: str = "",
    ) -> None:
        """Log rate limit exceeded events."""
        self.log_event(
            "rate_limit_exceeded",
            user_id=user_id,
            ip_address=ip_address,
            details={"endpoint": endpoint},
            severity="medium",
        )

        # Create an alert for repeated violations
        self.create_alert(
            "rate_limit_violation",
            user_id=user_id,
            ip_address=ip_address,
            message=f"Rate limit exceeded for endpoint: {endpoint}",
            details={"endpoint": endpoint},
        )

    def log_suspicious_content(
        self, user_id: str, content: str, content_type: str = "prompt"
    ) -> None:
        """Log suspicious content detection."""
        # This would integrate with content filtering systems
        self.log_event(
            "suspicious_content_detected",
            user_id=user_id,
            details={
                "content_type": content_type,
                "content_length": len(content) if content else 0,
            },
            severity="high",
        )

        self.create_alert(
            "suspicious_content",
            user_id=user_id,
            message=f"Suspicious {content_type} detected",
            details={
                "content_type": content_type,
                "content_length": len(content) if content else 0,
            },
        )

    def generate_security_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate a security report for the last N hours."""
        recent_events = self.get_recent_events(hours)
        unresolved_alerts = self.get_unresolved_alerts()

        # Categorize events
        event_counts: Dict[str, int] = defaultdict(int)
        severity_counts: Dict[str, int] = defaultdict(int)

        for event in recent_events:
            event_counts[event.event_type] += 1
            severity_counts[event.severity] += 1

        # Alert statistics
        alert_types: Dict[str, int] = defaultdict(int)
        for alert in unresolved_alerts:
            alert_types[alert.alert_type] += 1

        return {
            "report_generated": datetime.utcnow().isoformat(),
            "period_hours": hours,
            "total_events": len(recent_events),
            "event_types": dict(event_counts),
            "severity_distribution": dict(severity_counts),
            "unresolved_alerts": len(unresolved_alerts),
            "alert_types": dict(alert_types),
            "suspicious_ips_count": len(self.suspicious_ips),
            "suspicious_users_count": len(self.suspicious_users),
        }


# Global security monitor instance
security_monitor = SecurityMonitor()


# Convenience functions
def log_security_event(
    event_type: str,
    user_id: Optional[str] = None,
    ip_address: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    severity: str = "low",
) -> None:
    """Convenience function to log security events."""
    if details is None:
        details = {}
    security_monitor.log_event(event_type, user_id, ip_address, details, severity)


def record_login_attempt(
    user_id: Optional[str] = None,
    ip_address: Optional[str] = None,
    success: bool = False,
) -> None:
    """Convenience function to record login attempts."""
    security_monitor.record_login_attempt(user_id, ip_address, success)


def log_api_key_usage(
    api_key_id: str, user_id: str, ip_address: Optional[str] = None
) -> None:
    """Convenience function to log API key usage."""
    security_monitor.log_api_key_usage(api_key_id, user_id, ip_address)


def log_rate_limit_exceeded(
    user_id: Optional[str] = None, ip_address: Optional[str] = None, endpoint: str = ""
) -> None:
    """Convenience function to log rate limit exceeded events."""
    security_monitor.log_rate_limit_exceeded(user_id, ip_address, endpoint)
