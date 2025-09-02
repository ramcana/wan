# Project Health Monitoring System

A comprehensive health monitoring system for software projects that provides automated health checks, trend analysis, notifications, and actionable recommendations.

## Features

- **Comprehensive Health Checks**: Monitors test suite, documentation, configuration, and code quality
- **Trend Analysis**: Tracks health metrics over time with statistical analysis
- **Smart Notifications**: Multi-channel alerting with rate limiting and escalation policies
- **Actionable Recommendations**: AI-powered suggestions for improving project health
- **Real-time Dashboard**: Web-based dashboard for monitoring health status
- **CI/CD Integration**: Health gates and status checks for deployment pipelines

## Quick Start

### Installation

```bash
# Install required dependencies
pip install -r requirements.txt

# Optional: Install dashboard dependencies
pip install fastapi uvicorn

# Optional: Install notification dependencies
pip install requests
```

### Basic Usage

```bash
# Run a complete health check
python -m tools.health-checker.cli check

# Generate HTML report
python -m tools.health-checker.cli check --format html

# Check specific categories
python -m tools.health-checker.cli check --categories tests,documentation

# Analyze health trends
python -m tools.health-checker.cli trends --days 30

# Test notification channels
python -m tools.health-checker.cli test-notifications

# Run web dashboard
python -m tools.health-checker.cli dashboard --port 8080
```

## Architecture

### Core Components

1. **Health Checker** (`health_checker.py`)

   - Orchestrates health checks across different categories
   - Manages parallel execution and timeout handling
   - Generates comprehensive health reports

2. **Health Reporters** (`health_reporter.py`)

   - Generates reports in multiple formats (HTML, JSON, Markdown)
   - Creates visualizations and charts
   - Provides dashboard data

3. **Health Analytics** (`health_analytics.py`)

   - Performs trend analysis and forecasting
   - Calculates health metrics and insights
   - Detects patterns and anomalies

4. **Notification System** (`health_notifier.py`)

   - Multi-channel notifications (email, Slack, webhooks)
   - Alert rules and escalation policies
   - Rate limiting and notification history

5. **Recommendation Engine** (`recommendation_engine.py`)
   - Generates actionable improvement suggestions
   - Prioritizes recommendations by impact and effort
   - Tracks implementation progress

### Health Categories

- **Tests**: Test suite health, coverage, pass rates, execution time
- **Documentation**: Documentation completeness, broken links, organization
- **Configuration**: Configuration consistency, validation, security
- **Code Quality**: Syntax errors, complexity, code smells, style issues

## Configuration

### Basic Configuration

Create a configuration file or use environment variables:

```python
from tools.health_checker.health_models import HealthConfig

config = HealthConfig(
    # Scoring weights
    test_weight=0.3,
    documentation_weight=0.2,
    configuration_weight=0.2,
    code_quality_weight=0.15,
    performance_weight=0.1,
    security_weight=0.05,

    # Thresholds
    critical_threshold=50.0,
    warning_threshold=75.0,

    # Paths
    project_root=Path("."),
    test_directory=Path("tests"),
    docs_directory=Path("docs"),
    config_directory=Path("config")
)
```

### Notification Configuration

```python
notification_config = {
    "email": {
        "enabled": True,
        "recipients": ["team@example.com"],
        "smtp": {
            "host": "smtp.example.com",
            "port": 587,
            "username": "notifications@example.com",
            "password": "password",
            "use_tls": True
        }
    },
    "slack": {
        "enabled": True,
        "webhook_url": "https://hooks.slack.com/services/..."
    },
    "webhook": {
        "enabled": True,
        "url": "https://api.example.com/health-webhook",
        "headers": {"Authorization": "Bearer token"}
    }
}
```

## Usage Examples

### Programmatic Usage

```python
import asyncio
from tools.health_checker import ProjectHealthChecker, HealthReporter, RecommendationEngine

async def check_project_health():
    # Initialize components
    health_checker = ProjectHealthChecker()
    health_reporter = HealthReporter()
    recommendation_engine = RecommendationEngine()

    # Run health check
    report = await health_checker.run_health_check()

    # Generate recommendations
    recommendations = recommendation_engine.generate_recommendations(report)
    report.recommendations = recommendations

    # Generate HTML report
    html_file = health_reporter.generate_report(report, "html")
    print(f"Report generated: {html_file}")

    # Print console summary
    health_reporter._print_console_report(report)

# Run the check
asyncio.run(check_project_health())
```

### CI/CD Integration

#### GitHub Actions

```yaml
name: Health Check
on: [push, pull_request]

jobs:
  health-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: "3.9"
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run health check
        run: |
          python -m tools.health_checker.cli check \
            --format json \
            --exit-code-threshold 75 \
            --notify
```

#### Pre-commit Hook

```bash
#!/bin/bash
# .git/hooks/pre-commit

echo "Running health check..."
python -m tools.health_checker.cli check --exit-code-threshold 60

if [ $? -ne 0 ]; then
    echo "Health check failed. Commit blocked."
    exit 1
fi
```

### Custom Health Checkers

```python
from tools.health_checker.checkers import HealthChecker
from tools.health_checker.health_models import ComponentHealth, HealthIssue

class CustomSecurityChecker(HealthChecker):
    def check_health(self) -> ComponentHealth:
        issues = []

        # Custom security checks
        if self._has_hardcoded_secrets():
            issues.append(HealthIssue(
                severity=Severity.CRITICAL,
                category=HealthCategory.SECURITY,
                title="Hardcoded Secrets Detected",
                description="Found hardcoded secrets in source code",
                affected_components=["security"],
                remediation_steps=[
                    "Remove hardcoded secrets",
                    "Use environment variables",
                    "Implement secret management"
                ]
            ))

        score = 100 - len(issues) * 25
        status = "healthy" if score >= 75 else "critical"

        return ComponentHealth(
            component_name="security",
            category=HealthCategory.SECURITY,
            score=score,
            status=status,
            issues=issues
        )

# Add to health checker
health_checker.checkers[HealthCategory.SECURITY] = CustomSecurityChecker(config)
```

### Custom Notification Channels

```python
from tools.health_checker.health_notifier import NotificationChannel

class TeamsNotificationChannel(NotificationChannel):
    async def send_notification(self, message: str, severity: Severity, metadata: dict = None) -> bool:
        # Implement Microsoft Teams notification
        webhook_url = self.config.get("webhook_url")

        payload = {
            "@type": "MessageCard",
            "summary": f"Health Alert: {severity.value}",
            "text": message,
            "themeColor": self._get_color_for_severity(severity)
        }

        # Send to Teams webhook
        response = requests.post(webhook_url, json=payload)
        return response.status_code == 200

# Add to notifier
notifier.add_custom_channel("teams", TeamsNotificationChannel("teams", teams_config))
```

## Dashboard

The health monitoring system includes a web-based dashboard for real-time monitoring:

### Features

- Real-time health score display
- Component health breakdown
- Interactive charts and graphs
- Issue tracking and alerts
- Historical trend analysis
- WebSocket updates

### Running the Dashboard

```bash
# Start dashboard server
python -m tools.health_checker.cli dashboard --host 0.0.0.0 --port 8080

# Access dashboard
open http://localhost:8080
```

## API Reference

### Health Checker

```python
class ProjectHealthChecker:
    async def run_health_check(categories: Optional[List[HealthCategory]] = None) -> HealthReport
    def get_health_score() -> float
    def schedule_health_checks() -> None
```

### Health Reporter

```python
class HealthReporter:
    def generate_report(report: HealthReport, format_type: str = "html") -> Path
    def generate_dashboard_data(report: HealthReport) -> Dict[str, Any]
    def analyze_trends(history_file: Path) -> Dict[str, Any]
```

### Recommendation Engine

```python
class RecommendationEngine:
    def generate_recommendations(report: HealthReport) -> List[Recommendation]
    def prioritize_recommendations(recommendations: List[Recommendation]) -> List[Recommendation]
    def generate_implementation_plan(recommendations: List[Recommendation]) -> Dict[str, Any]
    def track_recommendation_progress(recommendation_id: str, status: str, notes: str = "") -> None
```

### Health Notifier

```python
class HealthNotifier:
    async def process_health_report(report: HealthReport) -> List[str]
    async def send_custom_notification(message: str, severity: Severity, channels: List[str]) -> bool
    def test_notifications() -> Dict[str, bool]
    def add_custom_channel(name: str, channel: NotificationChannel) -> None
```

## Best Practices

### Health Check Frequency

- **Development**: Run on every commit (pre-commit hook)
- **CI/CD**: Run on every build and deployment
- **Production**: Schedule regular checks (daily/weekly)
- **Monitoring**: Continuous monitoring with dashboard

### Threshold Configuration

- **Critical Threshold (50)**: Blocks deployments, immediate alerts
- **Warning Threshold (75)**: Requires attention, scheduled notifications
- **Target Score (85+)**: Healthy project state

### Notification Strategy

- **Critical Issues**: Immediate notifications to all channels
- **High Priority**: Email and Slack notifications
- **Medium Priority**: Slack notifications only
- **Low Priority**: Dashboard and reports only

### Recommendation Implementation

1. **Quick Wins**: High-impact, low-effort improvements first
2. **Critical Issues**: Address immediately regardless of effort
3. **Systematic Approach**: Follow generated implementation plan
4. **Progress Tracking**: Monitor recommendation completion

## Troubleshooting

### Common Issues

1. **No Health Data**

   - Ensure health checks are running regularly
   - Check file permissions for history storage
   - Verify project structure matches configuration

2. **Notification Failures**

   - Test notification channels individually
   - Check network connectivity and credentials
   - Verify webhook URLs and API keys

3. **Dashboard Not Loading**

   - Ensure FastAPI and uvicorn are installed
   - Check port availability and firewall settings
   - Verify WebSocket connections

4. **Slow Health Checks**
   - Enable parallel execution
   - Increase timeout settings
   - Optimize individual health checkers

### Debug Mode

```bash
# Enable debug logging
export HEALTH_MONITOR_DEBUG=1
python -m tools.health_checker.cli check

# Verbose output
python -m tools.health_checker.cli check --verbose
```

## Contributing

### Adding New Health Checkers

1. Create a new checker class inheriting from base checker
2. Implement `check_health()` method
3. Register with health checker
4. Add tests and documentation

### Adding Notification Channels

1. Create a new channel class inheriting from `NotificationChannel`
2. Implement `send_notification()` method
3. Add configuration options
4. Test with various message types

### Adding Recommendation Rules

1. Create a new rule class inheriting from `RecommendationRule`
2. Implement `applies_to()` and `generate_recommendation()` methods
3. Register with recommendation engine
4. Test with various health scenarios

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:

- Check the troubleshooting section
- Review the API documentation
- Create an issue in the project repository
- Contact the development team
