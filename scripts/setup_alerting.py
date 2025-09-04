from unittest.mock import Mock, patch
#!/usr/bin/env python3
"""
Setup Automated Alerting System

This script configures and initializes the automated alerting system
for production health monitoring.
"""

import os
import sys
import asyncio
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "tools" / "health-checker"))

from automated_alerting import AutomatedAlertingSystem
from health_models import HealthReport, HealthTrends, HealthIssue, Severity, HealthCategory


class AlertingSetupManager:
    """Manages setup and configuration of the alerting system"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.setup_log = []
        
    def log_step(self, message: str, success: bool = True) -> None:
        """Log setup step with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status = "✓" if success else "✗"
        log_entry = f"[{timestamp}] {status} {message}"
        print(log_entry)
        self.setup_log.append(log_entry)
    
    def validate_environment(self) -> bool:
        """Validate environment for alerting system"""
        self.log_step("Validating alerting environment...")
        
        try:
            # Check required directories
            required_dirs = [
                "config",
                "logs",
                "tools/health-checker"
            ]
            
            for dir_path in required_dirs:
                full_path = self.project_root / dir_path
                if not full_path.exists():
                    full_path.mkdir(parents=True, exist_ok=True)
                    self.log_step(f"Created directory: {dir_path}")
            
            # Check configuration files
            config_file = self.project_root / "config" / "alerting-config.yaml"
            if not config_file.exists():
                self.log_step("Alerting configuration not found", False)
                return False
            
            # Check Python dependencies
            try:
                import yaml
                self.log_step("YAML library available")
            except ImportError:
                self.log_step("YAML library not available", False)
                return False
            
            self.log_step("Environment validation passed")
            return True
            
        except Exception as e:
            self.log_step(f"Environment validation failed: {e}", False)
            return False
    
    def setup_notification_channels(self) -> bool:
        """Set up and test notification channels"""
        self.log_step("Setting up notification channels...")
        
        try:
            # Check environment variables for notification services
            notifications_configured = []
            
            # Email configuration
            if os.getenv("SMTP_USERNAME") and os.getenv("SMTP_PASSWORD"):
                notifications_configured.append("email")
                self.log_step("Email notifications configured")
            else:
                self.log_step("Email notifications not configured (missing SMTP credentials)")
            
            # Slack configuration
            if os.getenv("SLACK_WEBHOOK_URL"):
                notifications_configured.append("slack")
                self.log_step("Slack notifications configured")
            else:
                self.log_step("Slack notifications not configured (missing webhook URL)")
            
            # SMS configuration
            if os.getenv("SMS_API_KEY"):
                notifications_configured.append("sms")
                self.log_step("SMS notifications configured")
            else:
                self.log_step("SMS notifications not configured (missing API key)")
            
            if not notifications_configured:
                self.log_step("Warning: No external notification channels configured")
                self.log_step("Console notifications will be used as fallback")
                notifications_configured.append("console")
            
            self.log_step(f"Notification channels configured: {', '.join(notifications_configured)}")
            return True
            
        except Exception as e:
            self.log_step(f"Notification setup failed: {e}", False)
            return False
    
    def create_alert_rules(self) -> bool:
        """Create and validate alert rules"""
        self.log_step("Creating alert rules...")
        
        try:
            # Initialize alerting system
            alerting_system = AutomatedAlertingSystem()
            
            # Validate alert rules
            for rule in alerting_system.alert_rules:
                try:
                    # Test rule condition syntax
                    test_context = {
                        'health_report': None,
                        'HealthCategory': HealthCategory,
                        'Severity': Severity,
                        'any': any,
                        'all': all
                    }
                    
                    # This will raise SyntaxError if condition is invalid
                    compile(rule.condition, '<string>', 'eval')
                    
                    self.log_step(f"Alert rule '{rule.name}' validated")
                    
                except SyntaxError as e:
                    self.log_step(f"Invalid alert rule '{rule.name}': {e}", False)
                    return False
            
            # Save configuration
            alerting_system.save_configuration()
            self.log_step("Alert rules configuration saved")
            
            return True
            
        except Exception as e:
            self.log_step(f"Alert rules creation failed: {e}", False)
            return False
    
    async def test_alerting_system(self) -> bool:
        """Test the alerting system with mock data"""
        self.log_step("Testing alerting system...")
        
        try:
            # Initialize alerting system
            alerting_system = AutomatedAlertingSystem()
            
            # Create test health report with issues
            test_issues = [
                HealthIssue(
                    severity=Severity.CRITICAL,
                    category=HealthCategory.TESTS,
                    title="Critical Test Failure",
                    description="Multiple critical tests are failing",
                    affected_components=["test_suite"],
                    remediation_steps=["Fix failing tests", "Review test configuration"]
                ),
                HealthIssue(
                    severity=Severity.HIGH,
                    category=HealthCategory.SECURITY,
                    title="Security Vulnerability",
                    description="High severity security issue detected",
                    affected_components=["security_scanner"],
                    remediation_steps=["Update vulnerable dependencies", "Review security policies"]
                )
            ]
            
            test_report = HealthReport(
                timestamp=datetime.now(),
                overall_score=65.0,  # Below critical threshold
                component_scores={},
                issues=test_issues,
                recommendations=[],
                trends=HealthTrends()
            )
            
            # Test alert evaluation (but don't actually send notifications)
            original_send_method = alerting_system._send_alert_notifications
            
            async def mock_send_notifications(rule, alert_message, health_report):
                self.log_step(f"Mock alert sent: {rule.name} ({rule.alert_level.value})")
            
            alerting_system._send_alert_notifications = mock_send_notifications
            
            # Evaluate test report
            await alerting_system.evaluate_health_report(test_report)
            
            # Restore original method
            alerting_system._send_alert_notifications = original_send_method
            
            self.log_step("Alerting system test completed")
            return True
            
        except Exception as e:
            self.log_step(f"Alerting system test failed: {e}", False)
            return False
    
    def setup_alert_management_cli(self) -> bool:
        """Set up CLI tools for alert management"""
        self.log_step("Setting up alert management CLI...")
        
        try:
            # Create CLI script
            cli_script = self.project_root / "scripts" / "manage_alerts.py"
            
            cli_content = '''#!/usr/bin/env python3
"""
Alert Management CLI

Command-line interface for managing health monitoring alerts.
"""

import sys
import asyncio
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "tools" / "health-checker"))

from automated_alerting import AutomatedAlertingSystem, create_alerting_cli

if __name__ == "__main__":
    # Use the CLI from automated_alerting module
    asyncio.run(main())
'''
            
            with open(cli_script, 'w') as f:
                f.write(cli_content)
            
            # Make executable on Unix-like systems
            if os.name != 'nt':
                os.chmod(cli_script, 0o755)
            
            self.log_step(f"Alert management CLI created: {cli_script}")
            return True
            
        except Exception as e:
            self.log_step(f"CLI setup failed: {e}", False)
            return False
    
    def create_monitoring_dashboard(self) -> bool:
        """Create monitoring dashboard for alerts"""
        self.log_step("Creating alert monitoring dashboard...")
        
        try:
            # Create dashboard HTML file
            dashboard_dir = self.project_root / "reports" / "alerting"
            dashboard_dir.mkdir(parents=True, exist_ok=True)
            
            dashboard_file = dashboard_dir / "alert_dashboard.html"
            
            dashboard_html = '''<!DOCTYPE html>
<html>
<head>
    <title>Health Monitoring Alerts Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .alert { padding: 10px; margin: 10px 0; border-radius: 5px; }
        .alert.critical { background-color: #ffebee; border-left: 5px solid #f44336; }
        .alert.warning { background-color: #fff3e0; border-left: 5px solid #ff9800; }
        .alert.info { background-color: #e3f2fd; border-left: 5px solid #2196f3; }
        .status { display: inline-block; padding: 2px 8px; border-radius: 3px; color: white; }
        .status.active { background-color: #f44336; }
        .status.acknowledged { background-color: #ff9800; }
        .status.resolved { background-color: #4caf50; }
    </style>
</head>
<body>
    <h1>Health Monitoring Alerts Dashboard</h1>
    
    <div id="alerts-container">
        <p>Loading alerts...</p>
    </div>
    
    <script>
        // This would be populated with real-time alert data
        // For now, it's a static template
        
        function loadAlerts() {
            // In a real implementation, this would fetch data from an API
            const alertsContainer = document.getElementById('alerts-container');
            alertsContainer.innerHTML = '<p>No active alerts</p>';
        }
        
        // Load alerts on page load
        loadAlerts();
        
        // Refresh every 30 seconds
        setInterval(loadAlerts, 30000);
    </script>
</body>
</html>'''
            
            with open(dashboard_file, 'w') as f:
                f.write(dashboard_html)
            
            self.log_step(f"Alert dashboard created: {dashboard_file}")
            return True
            
        except Exception as e:
            self.log_step(f"Dashboard creation failed: {e}", False)
            return False
    
    def create_setup_summary(self) -> None:
        """Create setup summary report"""
        summary_file = self.project_root / "ALERTING_SETUP_SUMMARY.md"
        
        summary_content = f"""# Automated Alerting System Setup Summary

**Setup Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")}

## Setup Steps

"""
        
        for log_entry in self.setup_log:
            summary_content += f"- {log_entry}\n"
        
        summary_content += f"""

## Configuration Files

- **Alert Rules:** `config/alerting-config.yaml`
- **Logs Directory:** `logs/alerting/`
- **Dashboard:** `reports/alerting/alert_dashboard.html`
- **CLI Tool:** `scripts/manage_alerts.py`

## Usage

### Start Alerting System
```bash
python -m tools.health_checker.automated_alerting
```

### Manage Alerts
```bash
# List active alerts
python scripts/manage_alerts.py list

# Acknowledge an alert
python scripts/manage_alerts.py acknowledge critical_health_score --by "John Doe"

# Resolve an alert
python scripts/manage_alerts.py resolve critical_health_score --by "Jane Smith" --notes "Fixed configuration issue"

# Test alert system
python scripts/manage_alerts.py test critical_health_score
```

### Environment Variables

Set these environment variables for full functionality:

```bash
# Email notifications
export SMTP_USERNAME="your-email@company.com"
export SMTP_PASSWORD="your-app-password"
export ALERT_FROM_EMAIL="alerts@company.com"

# Slack notifications
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."

# SMS notifications (optional)
export SMS_API_KEY="your-sms-api-key"

# GitHub integration (optional)
export GITHUB_TOKEN="your-github-token"
```

## Alert Rules

The system includes the following default alert rules:

1. **Critical Health Score** - Triggers when overall health score < 70
2. **Emergency Health Score** - Triggers when overall health score < 50
3. **Test Failures** - Triggers on critical test failures
4. **Security Issues** - Triggers on high/critical security issues
5. **Performance Degradation** - Triggers on performance issues
6. **Configuration Errors** - Triggers on critical configuration issues

## Escalation Policies

- **Standard Escalation:** Slack → Email+Slack → Email+Slack+SMS
- **Emergency Escalation:** All channels immediately → Phone calls

## Notification Channels

- **Email:** SMTP-based email notifications
- **Slack:** Webhook-based Slack messages
- **SMS:** API-based SMS alerts (requires service setup)
- **Phone:** Voice call alerts (requires service setup)

## Monitoring

- Alert logs: `logs/alerting/`
- Acknowledgments: `logs/alerting/acknowledgments.jsonl`
- Resolutions: `logs/alerting/resolutions.jsonl`
- Dashboard: `reports/alerting/alert_dashboard.html`

## Next Steps

1. Configure notification channels by setting environment variables
2. Test the alerting system with mock alerts
3. Customize alert rules based on your specific needs
4. Set up integrations with external services (Jira, PagerDuty, etc.)
5. Train team members on alert management procedures
"""
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary_content)
        
        self.log_step(f"Setup summary created: {summary_file}")
    
    async def setup(self) -> bool:
        """Execute complete alerting system setup"""
        self.log_step("Starting automated alerting system setup")
        
        # Validation steps
        if not self.validate_environment():
            return False
        
        if not self.setup_notification_channels():
            return False
        
        if not self.create_alert_rules():
            return False
        
        # Testing
        if not await self.test_alerting_system():
            return False
        
        # Additional setup
        if not self.setup_alert_management_cli():
            return False
        
        if not self.create_monitoring_dashboard():
            return False
        
        # Create summary
        self.create_setup_summary()
        
        self.log_step("Automated alerting system setup completed successfully!")
        return True


async def main():
    """Main setup function"""
    setup_manager = AlertingSetupManager()
    success = await setup_manager.setup()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))