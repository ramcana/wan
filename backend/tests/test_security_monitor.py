import unittest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from backend.services.security_monitor import (
    SecurityMonitor,
    SecurityEvent,
    SecurityAlert,
    log_api_key_usage,
    log_rate_limit_exceeded,
)


class TestSecurityMonitor(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.monitor = SecurityMonitor(max_events=100)

    def test_log_event(self):
        """Test that security events are logged correctly."""
        # Log an event
        self.monitor.log_event(
            "test_event",
            user_id="user123",
            ip_address="192.168.1.1",
            details={"test": "data"},
            severity="medium",
        )

        # Check that the event was logged
        events = self.monitor.get_recent_events()
        self.assertEqual(len(events), 1)

        event = events[0]
        self.assertEqual(event.event_type, "test_event")
        self.assertEqual(event.user_id, "user123")
        self.assertEqual(event.ip_address, "192.168.1.1")
        self.assertEqual(event.details, {"test": "data"})
        self.assertEqual(event.severity, "medium")

    def test_check_suspicious_activity_ip(self):
        """Test detection of suspicious activity from IP addresses."""
        # Record many login attempts from the same IP
        ip_address = "192.168.1.100"
        for _ in range(15):  # More than the threshold of 10
            self.monitor.ip_login_attempts[ip_address].append(datetime.utcnow())

        # Check for suspicious activity
        warnings = self.monitor.check_suspicious_activity(ip_address=ip_address)

        # Should have a warning
        self.assertEqual(len(warnings), 1)
        self.assertIn("Excessive login attempts", warnings[0])

        # IP should be flagged as suspicious
        self.assertIn(ip_address, self.monitor.get_suspicious_ips())

    def test_check_suspicious_activity_user(self):
        """Test detection of suspicious activity for users."""
        # Record many login attempts for the same user
        user_id = "user456"
        for _ in range(8):  # More than the threshold of 5
            self.monitor.user_login_attempts[user_id].append(datetime.utcnow())

        # Check for suspicious activity
        warnings = self.monitor.check_suspicious_activity(user_id=user_id)

        # Should have a warning
        self.assertEqual(len(warnings), 1)
        self.assertIn("Excessive login attempts", warnings[0])

        # User should be flagged as suspicious
        self.assertIn(user_id, self.monitor.get_suspicious_users())

    def test_create_alert(self):
        """Test that security alerts are created correctly."""
        # Create an alert
        alert = self.monitor.create_alert(
            "test_alert",
            user_id="user789",
            ip_address="10.0.0.1",
            message="Test alert message",
            details={"test": "alert_data"},
        )

        # Check that the alert was created
        self.assertIsInstance(alert, SecurityAlert)
        self.assertEqual(alert.alert_type, "test_alert")
        self.assertEqual(alert.user_id, "user789")
        self.assertEqual(alert.ip_address, "10.0.0.1")
        self.assertEqual(alert.message, "Test alert message")
        self.assertEqual(alert.details, {"test": "alert_data"})
        self.assertFalse(alert.resolved)

        # Check that the alert is in the alerts list
        alerts = self.monitor.get_unresolved_alerts()
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0], alert)

    def test_resolve_alert(self):
        """Test that alerts can be resolved."""
        # Create an alert
        self.monitor.create_alert("test_alert", message="Test message")

        # Resolve the alert
        success = self.monitor.resolve_alert(0)
        self.assertTrue(success)

        # Check that the alert is now resolved
        alerts = self.monitor.get_unresolved_alerts()
        self.assertEqual(len(alerts), 0)

    def test_clear_suspicious_flags(self):
        """Test that suspicious flags can be cleared."""
        # Flag an IP and user as suspicious
        ip_address = "192.168.1.50"
        user_id = "user999"
        self.monitor.suspicious_ips.add(ip_address)
        self.monitor.suspicious_users.add(user_id)

        # Verify they are flagged
        self.assertIn(ip_address, self.monitor.get_suspicious_ips())
        self.assertIn(user_id, self.monitor.get_suspicious_users())

        # Clear the flags
        self.monitor.clear_suspicious_flags(ip_address=ip_address, user_id=user_id)

        # Verify they are no longer flagged
        self.assertNotIn(ip_address, self.monitor.get_suspicious_ips())
        self.assertNotIn(user_id, self.monitor.get_suspicious_users())

    def test_log_api_key_usage(self):
        """Test logging of API key usage."""
        with patch(
            "backend.services.security_monitor.security_monitor"
        ) as mock_monitor:
            mock_monitor.log_event = MagicMock()

            # Log API key usage
            log_api_key_usage("key123", "user123", "192.168.1.1")

            # Verify the event was logged
            mock_monitor.log_event.assert_called_once_with(
                "api_key_usage",
                user_id="user123",
                ip_address="192.168.1.1",
                details={"api_key_id": "key123"},
                severity="low",
            )

    def test_log_rate_limit_exceeded(self):
        """Test logging of rate limit exceeded events."""
        with patch(
            "backend.services.security_monitor.security_monitor"
        ) as mock_monitor:
            mock_monitor.log_event = MagicMock()
            mock_monitor.create_alert = MagicMock()

            # Log rate limit exceeded
            log_rate_limit_exceeded("user123", "192.168.1.1", "/api/test")

            # Verify the event was logged
            mock_monitor.log_event.assert_called_once_with(
                "rate_limit_exceeded",
                user_id="user123",
                ip_address="192.168.1.1",
                details={"endpoint": "/api/test"},
                severity="medium",
            )

    def test_generate_security_report(self):
        """Test generation of security reports."""
        # Log some events
        self.monitor.log_event("login_success", user_id="user1", severity="low")
        self.monitor.log_event("login_failure", user_id="user2", severity="medium")
        self.monitor.log_event("suspicious_activity", user_id="user3", severity="high")

        # Create some alerts
        self.monitor.create_alert("test_alert_1", message="Test message 1")
        self.monitor.create_alert("test_alert_2", message="Test message 2")

        # Generate report
        report = self.monitor.generate_security_report(hours=24)

        # Check report contents
        self.assertIn("report_generated", report)
        self.assertIn("period_hours", report)
        self.assertIn("total_events", report)
        self.assertIn("event_types", report)
        self.assertIn("severity_distribution", report)
        self.assertIn("unresolved_alerts", report)
        self.assertIn("alert_types", report)
        self.assertIn("suspicious_ips_count", report)
        self.assertIn("suspicious_users_count", report)

        # Check specific values
        self.assertEqual(report["total_events"], 3)
        self.assertEqual(report["unresolved_alerts"], 2)
        self.assertEqual(report["suspicious_ips_count"], 0)
        self.assertEqual(report["suspicious_users_count"], 0)

    def test_event_expiration(self):
        """Test that old events are not included in reports."""
        # Log an old event (2 days ago)
        old_event = SecurityEvent(
            event_type="old_event",
            timestamp=datetime.utcnow() - timedelta(days=2),
            user_id="user1",
            ip_address="192.168.1.1",
            details={},
            severity="low",
        )
        self.monitor.events.append(old_event)

        # Log a recent event
        self.monitor.log_event("recent_event", user_id="user2", severity="medium")

        # Get recent events (last 24 hours)
        recent_events = self.monitor.get_recent_events(hours=24)

        # Should only have the recent event
        self.assertEqual(len(recent_events), 1)
        self.assertEqual(recent_events[0].event_type, "recent_event")


if __name__ == "__main__":
    unittest.main()
