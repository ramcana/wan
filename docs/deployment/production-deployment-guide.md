# Production Deployment Guide

This guide provides comprehensive instructions for deploying the project health system in production environments.

## Overview

The project health system consists of several components that work together to monitor and maintain project health:

- **Test Suite Orchestrator**: Manages test execution and reporting
- **Documentation Generator**: Consolidates and validates documentation
- **Configuration Manager**: Unifies and validates configuration
- **Health Checker**: Monitors overall project health
- **Developer Tools**: Provides development environment support

## Prerequisites

### System Requirements

- **Operating System**: Linux (Ubuntu 20.04+ recommended), macOS 10.15+, or Windows 10+
- **Python**: 3.8 or higher
- **Memory**: Minimum 4GB RAM, 8GB+ recommended for large projects
- **Storage**: Minimum 10GB free space, SSD recommended
- **CPU**: 2+ cores recommended for parallel operations

### Dependencies

```bash
# Core dependencies
pip install pytest>=7.0.0
pip install pyyaml>=6.0
pip install psutil>=5.8.0
pip install asyncio
pip install pathlib

# Documentation dependencies
pip install mkdocs>=1.4.0
pip install mkdocs-material>=8.0.0

# Optional performance dependencies
pip install uvloop  # For improved async performance on Linux/macOS
```

## Installation

### 1. Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd <project-directory>

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Initialize Project Health System

```bash
# Run initial setup
python -m tools.health_checker.cli --setup

# Verify installation
python -m tools.health_checker.cli --version
```

### 3. Configure for Production

Create production configuration:

```yaml
# config/production.yaml
system:
  name: "production_project"
  version: "1.0.0"
  environment: "production"
  debug: false
  log_level: "INFO"

health_monitoring:
  enabled: true
  check_interval: 300 # 5 minutes
  notification_channels:
    - email
    - slack

test_execution:
  parallel: true
  timeout: 1800 # 30 minutes
  coverage_threshold: 80

documentation:
  auto_generate: true
  validate_links: true
  update_interval: 3600 # 1 hour

notifications:
  email:
    smtp_server: "smtp.company.com"
    smtp_port: 587
    username: "health-monitor@company.com"
    # Password should be set via environment variable

  slack:
    webhook_url: "https://hooks.slack.com/services/..."
    channel: "#project-health"
```

## Deployment Steps

### 1. Environment Setup

```bash
# Set environment variables
export PROJECT_ENV=production
export HEALTH_CONFIG_PATH=/path/to/config/production.yaml
export NOTIFICATION_EMAIL_PASSWORD=<secure-password>

# Create necessary directories
mkdir -p /var/log/project-health
mkdir -p /var/lib/project-health/data
mkdir -p /etc/project-health
```

### 2. Service Configuration

Create systemd service file:

```ini
# /etc/systemd/system/project-health.service
[Unit]
Description=Project Health Monitoring Service
After=network.target

[Service]
Type=simple
User=project-health
Group=project-health
WorkingDirectory=/opt/project-health
Environment=PROJECT_ENV=production
Environment=HEALTH_CONFIG_PATH=/etc/project-health/production.yaml
ExecStart=/opt/project-health/venv/bin/python -m tools.health_checker.cli --daemon
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### 3. Start Services

```bash
# Enable and start the service
sudo systemctl enable project-health
sudo systemctl start project-health

# Verify service status
sudo systemctl status project-health
```

### 4. Configure Web Dashboard (Optional)

```bash
# Start health dashboard
python -m tools.health_checker.dashboard_server --port 8080 --host 0.0.0.0

# Or use nginx reverse proxy
# See nginx configuration section below
```

## Configuration Management

### Environment-Specific Configurations

```bash
# Development
config/environments/development.yaml

# Staging
config/environments/staging.yaml

# Production
config/environments/production.yaml
```

### Configuration Validation

```bash
# Validate configuration before deployment
python -m tools.config_manager.config_cli validate --config production.yaml

# Test configuration migration
python -m tools.config_manager.config_cli migrate --dry-run
```

### Hot Configuration Reload

```bash
# Reload configuration without service restart
sudo systemctl reload project-health

# Or send SIGHUP signal
sudo kill -HUP $(pgrep -f "project-health")
```

## Monitoring and Alerting

### Health Check Endpoints

The system exposes health check endpoints for monitoring:

```bash
# Basic health check
curl http://localhost:8080/health

# Detailed health report
curl http://localhost:8080/health/detailed

# Component-specific health
curl http://localhost:8080/health/tests
curl http://localhost:8080/health/docs
curl http://localhost:8080/health/config
```

### Prometheus Integration

Add Prometheus metrics endpoint:

```yaml
# config/monitoring.yaml
prometheus:
  enabled: true
  port: 9090
  metrics:
    - health_score
    - test_pass_rate
    - documentation_coverage
    - configuration_errors
```

### Grafana Dashboard

Import the provided Grafana dashboard:

```json
{
  "dashboard": {
    "title": "Project Health Dashboard",
    "panels": [
      {
        "title": "Overall Health Score",
        "type": "stat",
        "targets": [
          {
            "expr": "project_health_score"
          }
        ]
      },
      {
        "title": "Test Pass Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "project_test_pass_rate"
          }
        ]
      }
    ]
  }
}
```

## Security Considerations

### Access Control

```bash
# Create dedicated user
sudo useradd -r -s /bin/false project-health

# Set proper permissions
sudo chown -R project-health:project-health /opt/project-health
sudo chmod 750 /opt/project-health
sudo chmod 640 /etc/project-health/production.yaml
```

### Secrets Management

```bash
# Use environment variables for secrets
export NOTIFICATION_EMAIL_PASSWORD=<password>
export SLACK_WEBHOOK_TOKEN=<token>

# Or use external secret management
# - HashiCorp Vault
# - AWS Secrets Manager
# - Azure Key Vault
```

### Network Security

```bash
# Firewall configuration
sudo ufw allow 8080/tcp  # Health dashboard
sudo ufw allow 9090/tcp  # Prometheus metrics
sudo ufw deny 22/tcp from any to any  # Restrict SSH
```

## Performance Tuning

### Resource Allocation

```yaml
# config/performance.yaml
performance:
  test_execution:
    max_workers: 4
    memory_limit: "2GB"
    timeout: 1800

  documentation:
    batch_size: 50
    parallel_processing: true

  health_checks:
    cache_duration: 300
    parallel_checks: true
```

### Database Optimization

```sql
-- Create indexes for health data
CREATE INDEX idx_health_timestamp ON health_reports(timestamp);
CREATE INDEX idx_health_score ON health_reports(overall_score);
CREATE INDEX idx_component_health ON component_health(component_name, timestamp);
```

### Caching Configuration

```yaml
# config/cache.yaml
cache:
  redis:
    host: "localhost"
    port: 6379
    db: 0

  ttl:
    health_reports: 300
    test_results: 600
    documentation: 3600
```

## Backup and Recovery

### Data Backup

```bash
#!/bin/bash
# backup-health-data.sh

BACKUP_DIR="/backup/project-health/$(date +%Y%m%d)"
mkdir -p "$BACKUP_DIR"

# Backup configuration
cp -r /etc/project-health "$BACKUP_DIR/config"

# Backup health data
sqlite3 /var/lib/project-health/health.db ".backup $BACKUP_DIR/health.db"

# Backup logs
tar -czf "$BACKUP_DIR/logs.tar.gz" /var/log/project-health/

# Upload to remote storage
aws s3 sync "$BACKUP_DIR" s3://company-backups/project-health/
```

### Disaster Recovery

```bash
#!/bin/bash
# restore-health-system.sh

BACKUP_DATE="$1"
BACKUP_DIR="/backup/project-health/$BACKUP_DATE"

# Stop services
sudo systemctl stop project-health

# Restore configuration
sudo cp -r "$BACKUP_DIR/config" /etc/project-health/

# Restore database
cp "$BACKUP_DIR/health.db" /var/lib/project-health/

# Restore logs
tar -xzf "$BACKUP_DIR/logs.tar.gz" -C /

# Start services
sudo systemctl start project-health
```

## Troubleshooting

### Common Issues

#### Service Won't Start

```bash
# Check service status
sudo systemctl status project-health

# Check logs
sudo journalctl -u project-health -f

# Common fixes:
# 1. Check configuration syntax
python -m tools.config_manager.config_cli validate

# 2. Check permissions
sudo chown -R project-health:project-health /opt/project-health

# 3. Check dependencies
pip check
```

#### High Memory Usage

```bash
# Monitor memory usage
python -m tools.health_checker.cli --monitor-memory

# Optimize configuration
# Reduce parallel workers
# Increase cache TTL
# Enable memory profiling
```

#### Slow Performance

```bash
# Run performance analysis
python -m tools.health_checker.cli --performance-analysis

# Check system resources
htop
iotop
```

### Log Analysis

```bash
# View health check logs
tail -f /var/log/project-health/health.log

# Search for errors
grep ERROR /var/log/project-health/*.log

# Analyze performance logs
python -m tools.health_checker.cli --analyze-logs /var/log/project-health/
```

## Maintenance Procedures

### Regular Maintenance Tasks

```bash
#!/bin/bash
# daily-maintenance.sh

# Clean old logs (keep 30 days)
find /var/log/project-health -name "*.log" -mtime +30 -delete

# Clean old health reports (keep 90 days)
python -m tools.health_checker.cli --cleanup-reports --days 90

# Update documentation
python -m tools.doc_generator.cli --update

# Run health check
python -m tools.health_checker.cli --full-check
```

### Weekly Maintenance

```bash
#!/bin/bash
# weekly-maintenance.sh

# Update dependencies
pip list --outdated
# Review and update as needed

# Run comprehensive tests
python -m tools.test_runner.cli --comprehensive

# Generate performance report
python -m tools.health_checker.cli --performance-report

# Backup system
./backup-health-data.sh
```

### Monthly Maintenance

```bash
#!/bin/bash
# monthly-maintenance.sh

# Review and rotate logs
logrotate /etc/logrotate.d/project-health

# Update system packages
sudo apt update && sudo apt upgrade

# Review security updates
sudo unattended-upgrades --dry-run

# Performance optimization review
python -m tools.health_checker.cli --optimization-suggestions
```

## Scaling Considerations

### Horizontal Scaling

```yaml
# config/scaling.yaml
scaling:
  load_balancer:
    enabled: true
    algorithm: "round_robin"
    health_check_path: "/health"

  instances:
    min: 2
    max: 10
    target_cpu: 70
    target_memory: 80
```

### Database Scaling

```yaml
# config/database.yaml
database:
  read_replicas: 2
  connection_pool:
    min_connections: 5
    max_connections: 20

  sharding:
    enabled: true
    shard_key: "project_id"
```

## Integration with CI/CD

### GitHub Actions

```yaml
# .github/workflows/health-check.yml
name: Project Health Check
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  schedule:
    - cron: "0 */6 * * *" # Every 6 hours

jobs:
  health-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.9"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run health check
        run: python -m tools.health_checker.cli --ci-mode

      - name: Upload health report
        uses: actions/upload-artifact@v3
        with:
          name: health-report
          path: health-report.json
```

### Jenkins Pipeline

```groovy
pipeline {
    agent any

    triggers {
        cron('H */6 * * *')  // Every 6 hours
    }

    stages {
        stage('Health Check') {
            steps {
                sh 'python -m tools.health_checker.cli --jenkins-mode'
            }
            post {
                always {
                    archiveArtifacts artifacts: 'health-report.json'
                    publishHTML([
                        allowMissing: false,
                        alwaysLinkToLastBuild: true,
                        keepAll: true,
                        reportDir: 'reports',
                        reportFiles: 'health-report.html',
                        reportName: 'Health Report'
                    ])
                }
            }
        }
    }
}
```

## Support and Maintenance Contacts

### Internal Team

- **DevOps Team**: devops@company.com
- **Development Team**: dev-team@company.com
- **On-call Engineer**: +1-555-0123

### External Support

- **System Vendor**: support@vendor.com
- **Cloud Provider**: AWS/Azure/GCP Support

### Documentation Updates

- **Wiki**: https://wiki.company.com/project-health
- **Runbooks**: https://runbooks.company.com/project-health
- **Change Log**: CHANGELOG.md

---

## Appendix

### A. Configuration Reference

See [Configuration Schema](../reference/configuration-schema.md) for complete configuration options.

### B. API Reference

See [API Documentation](../api/health-api.md) for complete API reference.

### C. Troubleshooting Flowcharts

See [Troubleshooting Guide](troubleshooting-guide.md) for detailed troubleshooting procedures.

### D. Performance Benchmarks

See [Performance Guide](performance-guide.md) for performance tuning and benchmarks.
