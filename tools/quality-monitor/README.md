# Quality Monitoring and Alerting System

A comprehensive quality monitoring system that provides real-time quality metrics monitoring, trend analysis, automated alerting, and proactive recommendations for code quality improvements.

## Features

### 🔍 Real-time Quality Metrics

- **Test Coverage**: Measures code coverage from test suites
- **Code Complexity**: Analyzes cyclomatic complexity of functions
- **Documentation Coverage**: Tracks docstring coverage for functions and classes
- **Duplicate Code**: Identifies duplicate code patterns
- **Style Violations**: Monitors code style compliance
- **Type Hint Coverage**: Tracks type annotation usage

### 📈 Trend Analysis

- Historical trend analysis for all quality metrics
- Statistical analysis to identify improving, stable, or degrading trends
- Confidence scoring based on data quality and variance
- Configurable analysis periods (7, 30, 90 days)

### 🚨 Automated Alerting

- Configurable thresholds for all quality metrics
- Multiple severity levels (Critical, High, Medium, Low)
- Trend-based alerts for degrading quality patterns
- Email and webhook notification support
- Alert cooldown periods to prevent spam

### 💡 Proactive Recommendations

- Automated recommendations based on current metrics
- Trend-based recommendations for preventing quality degradation
- Impact and effort estimation for each recommendation
- Actionable steps for implementing improvements
- Priority-based recommendation ranking

### 📊 Web Dashboard

- Real-time quality monitoring dashboard
- Interactive charts and visualizations
- Alert management and resolution
- Recommendation tracking
- Auto-refresh capabilities

## Installation

```bash
# Install required dependencies
pip install -r requirements.txt

# For test coverage analysis
pip install pytest-cov

# For style checking
pip install flake8

# For type checking
pip install mypy
```

## Quick Start

### Command Line Interface

```bash
# Collect current quality metrics
python -m tools.quality_monitor.cli metrics

# Analyze quality trends over the last 30 days
python -m tools.quality_monitor.cli trends --days 30

# Check for quality alerts
python -m tools.quality_monitor.cli alerts --check-trends

# Get improvement recommendations
python -m tools.quality_monitor.cli recommendations --priority high

# Start the web dashboard
python -m tools.quality_monitor.cli dashboard --host localhost --port 8080
```

### Python API

```python
from tools.quality_monitor import QualityMonitor

# Initialize the monitoring system
monitor = QualityMonitor(project_root=".")

# Collect current metrics
metrics = monitor.collect_metrics()
print(f"Test coverage: {metrics[0].value}%")

# Analyze trends
trends = monitor.analyze_trends(days=30)
for trend in trends:
    print(f"{trend.metric_type}: {trend.direction} ({trend.change_rate:+.1f}%/day)")

# Check for alerts
alerts = monitor.check_alerts()
if alerts:
    print(f"Found {len(alerts)} active alerts")

# Get recommendations
recommendations = monitor.get_recommendations()
for rec in recommendations[:3]:
    print(f"• {rec.title} (Impact: {rec.estimated_impact}%)")

# Start web dashboard
monitor.start_dashboard(host="localhost", port=8080)
```

## Configuration

### Alert Thresholds

Create `config/quality-alerts.json` to configure alert thresholds:

```json
{
  "thresholds": {
    "test_coverage": { "warning": 70.0, "critical": 50.0 },
    "code_complexity": { "warning": 10.0, "critical": 15.0 },
    "documentation_coverage": { "warning": 60.0, "critical": 40.0 },
    "duplicate_code": { "warning": 10.0, "critical": 20.0 },
    "style_violations": { "warning": 50.0, "critical": 100.0 },
    "type_hint_coverage": { "warning": 50.0, "critical": 30.0 }
  },
  "notifications": {
    "email": {
      "enabled": true,
      "smtp_server": "smtp.gmail.com",
      "smtp_port": 587,
      "username": "your-email@gmail.com",
      "password": "your-app-password",
      "recipients": ["team@company.com"]
    },
    "webhook": {
      "enabled": true,
      "url": "https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
      "headers": { "Content-Type": "application/json" }
    }
  },
  "alert_cooldown_hours": 24,
  "trend_alert_threshold": 5.0
}
```

### Dashboard Configuration

The dashboard runs on `http://localhost:8080` by default and provides:

- **Real-time Metrics**: Current quality metric values with color-coded status
- **Trend Indicators**: Visual trend direction indicators with change rates
- **Active Alerts**: List of current quality alerts with resolution actions
- **Recommendations**: Prioritized list of improvement recommendations
- **Auto-refresh**: Automatic data refresh every 5 minutes

## Architecture

### Components

1. **MetricsCollector**: Collects quality metrics from the codebase
2. **TrendAnalyzer**: Analyzes historical trends and patterns
3. **AlertSystem**: Monitors thresholds and generates alerts
4. **RecommendationEngine**: Generates improvement recommendations
5. **DashboardManager**: Orchestrates data collection and serves the web interface

### Data Flow

```
Codebase → MetricsCollector → TrendAnalyzer → AlertSystem
                                    ↓
DashboardManager ← RecommendationEngine ← Quality Analysis
       ↓
Web Dashboard / CLI / API
```

### Data Storage

- **Metrics History**: `data/quality-metrics/metrics_YYYYMMDD_HHMMSS.json`
- **Active Alerts**: `data/quality-alerts/active_alerts.json`
- **Recommendations**: `data/quality-recommendations/recommendations.json`

## Integration

### CI/CD Integration

Add quality monitoring to your CI/CD pipeline:

```yaml
# .github/workflows/quality-check.yml
name: Quality Check
on: [push, pull_request]

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest-cov flake8 mypy

      - name: Run quality checks
        run: |
          python -m tools.quality_monitor.cli metrics --output metrics.json
          python -m tools.quality_monitor.cli alerts --output alerts.json

      - name: Check for critical alerts
        run: |
          python -c "
          import json
          with open('alerts.json') as f:
              alerts = json.load(f)
          critical = [a for a in alerts['alerts'] if a['severity'] == 'critical']
          if critical:
              print(f'Found {len(critical)} critical quality issues')
              exit(1)
          "
```

### Pre-commit Hooks

Add quality checks to pre-commit hooks:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: quality-check
        name: Quality Check
        entry: python -m tools.quality_monitor.cli alerts
        language: system
        pass_filenames: false
```

## Monitoring Best Practices

### 1. Set Appropriate Thresholds

- Start with lenient thresholds and gradually tighten them
- Consider your project's maturity and team size
- Review and adjust thresholds based on historical data

### 2. Regular Monitoring

- Set up automated daily/weekly quality reports
- Review trends monthly to identify patterns
- Address degrading trends before they become critical

### 3. Team Integration

- Share quality metrics with the entire team
- Include quality discussions in code reviews
- Celebrate quality improvements and milestones

### 4. Continuous Improvement

- Act on recommendations promptly
- Track the impact of quality improvements
- Adjust monitoring based on team feedback

## Troubleshooting

### Common Issues

**No metrics collected**

- Ensure Python files are present in the project
- Check that pytest and coverage tools are installed
- Verify project structure is recognized

**Trend analysis shows no data**

- Metrics need to be collected over time to show trends
- Run metrics collection regularly (daily recommended)
- Check that data files are being stored correctly

**Alerts not triggering**

- Verify threshold configuration is correct
- Check that metrics are being collected successfully
- Review alert cooldown settings

**Dashboard not loading**

- Ensure the dashboard server is running
- Check for port conflicts
- Verify network connectivity and firewall settings

### Debug Mode

Enable debug logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from tools.quality_monitor import QualityMonitor
monitor = QualityMonitor()
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
