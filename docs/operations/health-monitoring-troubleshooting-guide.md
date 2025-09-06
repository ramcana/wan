# Health Monitoring Troubleshooting Guide

## Overview

This guide provides detailed troubleshooting procedures for the WAN22 project health monitoring system. It covers common issues, diagnostic procedures, and step-by-step resolution instructions.

## Quick Reference

### Emergency Contacts

- **Primary On-Call**: dev-team@company.com
- **Secondary On-Call**: ops-team@company.com
- **Emergency**: emergency@company.com

### Critical Commands

```bash
# Check service status
systemctl status wan22-health-monitor

# View current health
python -m tools.health_checker.health_checker --quick

# List active alerts
python scripts/manage_alerts.py list

# Check logs
tail -f logs/health-monitoring/health_monitor_$(date +%Y%m%d).log
```

## Common Issues

### 1. Health Monitoring Service Issues

#### Issue: Service Won't Start

**Symptoms:**

- `systemctl start wan22-health-monitor` fails
- No health reports being generated
- Service status shows "failed" or "inactive"

**Diagnostic Steps:**

```bash
# Check service status
systemctl status wan22-health-monitor

# Check service logs
journalctl -u wan22-health-monitor -f

# Check configuration
python tools/config_manager/config_validator.py config/production-health.yaml

# Check file permissions
ls -la config/production-health.yaml
ls -la tools/health-checker/
```

**Common Causes & Solutions:**

1. **Configuration Error**

   ```bash
   # Validate configuration
   python tools/config_manager/config_validator.py config/production-health.yaml

   # Fix syntax errors in YAML files
   nano config/production-health.yaml
   ```

2. **Permission Issues**

   ```bash
   # Fix file permissions
   sudo chown -R healthmonitor:healthmonitor /path/to/project
   chmod 644 config/*.yaml
   chmod 755 tools/health-checker/*.py
   ```

3. **Missing Dependencies**

   ```bash
   # Check Python dependencies
   python -c "import yaml, psutil; print('Dependencies OK')"

   # Install missing packages
   pip install pyyaml psutil
   ```

4. **Port Conflicts**

   ```bash
   # Check if port is in use
   netstat -tlnp | grep :8080

   # Change port in configuration
   nano config/production-health.yaml
   ```

#### Issue: Service Crashes Frequently

**Symptoms:**

- Service starts but stops after a few minutes
- Frequent restart messages in logs
- Health checks incomplete

**Diagnostic Steps:**

```bash
# Check crash logs
journalctl -u wan22-health-monitor --since "1 hour ago"

# Check system resources
free -m
df -h
top

# Check for memory leaks
python tools/health_checker/performance_optimizer.py --memory-check

# Run service manually for debugging
python -m tools.health_checker.production_deployment_simple
```

**Common Causes & Solutions:**

1. **Memory Issues**

   ```bash
   # Check memory usage
   ps aux | grep health_monitor

   # Reduce memory usage
   # Edit config/production-health.yaml
   max_concurrent_checks: 2  # Reduce from default
   ```

2. **Timeout Issues**

   ```bash
   # Increase timeouts
   # Edit config/production-health.yaml
   max_check_duration: 600  # Increase from 300
   ```

3. **Database Connection Issues**
   ```bash
   # Test database connectivity
   python tools/health_checker/production_health_checks.py
   ```

### 2. Health Check Issues

#### Issue: Health Checks Timing Out

**Symptoms:**

- Health checks never complete
- Timeout errors in logs
- Health score shows as 0 or unavailable

**Diagnostic Steps:**

```bash
# Run individual health checks
python tools/health_checker/checkers/test_health_checker.py
python tools/health_checker/checkers/code_quality_checker.py
python tools/health_checker/checkers/documentation_health_checker.py

# Check system performance
python tools/health_checker/performance_optimizer.py --analyze

# Monitor resource usage during checks
top -p $(pgrep -f health_monitor)
```

**Solutions:**

1. **Increase Timeout Values**

   ```yaml
   # config/production-health.yaml
   max_check_duration: 600 # Increase from 300 seconds

   # Per-component timeouts
   component_timeouts:
     tests: 300
     documentation: 120
     configuration: 60
   ```

2. **Enable Parallel Execution**

   ```yaml
   # config/production-health.yaml
   parallel_checks: true
   max_workers: 4
   ```

3. **Optimize Individual Checks**

   ```bash
   # Profile slow checks
   python tools/health_checker/performance_optimizer.py --profile

   # Disable slow checks temporarily
   # Edit config/production-health.yaml
   disabled_checks:
     - slow_performance_check
   ```

#### Issue: Inaccurate Health Scores

**Symptoms:**

- Health scores don't match actual system state
- Scores fluctuate wildly
- Components show incorrect status

**Diagnostic Steps:**

```bash
# Run detailed health analysis
python tools/health_checker/health_analytics.py --detailed

# Check individual component scores
python tools/health_checker/health_checker.py --component tests
python tools/health_checker/health_checker.py --component documentation

# Verify baseline metrics
cat tools/health_checker/baseline_metrics.json

# Check scoring weights
grep -A 10 "scoring_weights" config/production-health.yaml
```

**Solutions:**

1. **Update Baseline Metrics**

   ```bash
   # Recalculate baselines
   python tools/health_checker/establish_baseline.py --recalculate
   ```

2. **Adjust Scoring Weights**

   ```yaml
   # config/production-health.yaml
   scoring_weights:
     tests: 0.4 # Increase test importance
     documentation: 0.2
     configuration: 0.2
     code_quality: 0.1
     performance: 0.1
   ```

3. **Fix Component Issues**

   ```bash
   # Fix test issues
   python -m pytest tests/ --tb=short

   # Fix documentation issues
   python tools/doc-generator/validator.py

   # Fix configuration issues
   python tools/config_manager/config_validator.py --fix
   ```

### 3. Alert System Issues

#### Issue: Alerts Not Firing

**Symptoms:**

- No alerts received despite health issues
- Alert logs show no activity
- Critical issues go unnoticed

**Diagnostic Steps:**

```bash
# Check alerting service status
python scripts/manage_alerts.py list

# Test alert rules
python scripts/manage_alerts.py test critical_health_score

# Check alert configuration
cat config/alerting-config.yaml

# Verify notification channels
python tools/health_checker/health_notifier.py --test
```

**Solutions:**

1. **Check Alert Rule Conditions**

   ```bash
   # Test alert rule syntax
   python -c "
   from tools.health_checker.automated_alerting import AutomatedAlertingSystem
   system = AutomatedAlertingSystem()
   for rule in system.alert_rules:
       print(f'Rule: {rule.name}, Enabled: {rule.enabled}')
   "
   ```

2. **Verify Notification Configuration**

   ```bash
   # Check environment variables
   echo $SLACK_WEBHOOK_URL
   echo $SMTP_USERNAME

   # Test email configuration
   python -c "
   import smtplib
   from email.mime.text import MIMEText
   # Test SMTP connection
   "
   ```

3. **Enable Debug Logging**
   ```yaml
   # config/alerting-config.yaml
   logging:
     level: DEBUG
     file: logs/alerting/debug.log
   ```

#### Issue: Too Many False Positive Alerts

**Symptoms:**

- Constant alert notifications
- Alerts for non-critical issues
- Alert fatigue among team members

**Diagnostic Steps:**

```bash
# Analyze alert frequency
grep "Alert triggered" logs/alerting/alerting_*.log | wc -l

# Check alert thresholds
grep -A 5 "alert_rules" config/alerting-config.yaml

# Review alert history
python scripts/manage_alerts.py stats --last-week
```

**Solutions:**

1. **Adjust Alert Thresholds**

   ```yaml
   # config/alerting-config.yaml
   alert_rules:
     - name: "critical_health_score"
       condition: "health_report.overall_score < 60" # Lower threshold
       cooldown_minutes: 60 # Increase cooldown
   ```

2. **Add Maintenance Windows**

   ```yaml
   # config/alerting-config.yaml
   maintenance_windows:
     - name: "Daily Maintenance"
       start_time: "02:00"
       end_time: "04:00"
       suppress_levels: ["warning", "info"]
   ```

3. **Implement Alert Grouping**
   ```yaml
   # config/alerting-config.yaml
   grouping:
     enabled: true
     group_by: ["alert_level", "category"]
     group_window_minutes: 10
   ```

### 4. Report Generation Issues

#### Issue: Reports Not Generated

**Symptoms:**

- No daily/weekly reports in reports directory
- Report generation errors in logs
- Empty or corrupted report files

**Diagnostic Steps:**

```bash
# Check report directory
ls -la reports/daily/
ls -la reports/weekly/

# Check report generation logs
grep "report" logs/health-monitoring/health_monitor_*.log

# Test manual report generation
python -c "
from tools.health_checker.production_deployment_simple import ProductionHealthMonitor
import asyncio
monitor = ProductionHealthMonitor()
asyncio.run(monitor.generate_daily_report())
"

# Check disk space
df -h reports/
```

**Solutions:**

1. **Fix File Permissions**

   ```bash
   # Ensure write permissions
   chmod 755 reports/
   chmod 755 reports/daily/
   chmod 755 reports/weekly/

   # Fix ownership
   chown -R healthmonitor:healthmonitor reports/
   ```

2. **Check Disk Space**

   ```bash
   # Clean old reports if disk is full
   find reports/ -name "*.html" -mtime +30 -delete

   # Increase disk space or add log rotation
   ```

3. **Fix Template Issues**

   ```bash
   # Check report templates
   python tools/health_checker/health_reporter.py --validate-templates

   # Regenerate templates if corrupted
   python tools/health_checker/health_reporter.py --reset-templates
   ```

#### Issue: Incomplete or Corrupted Reports

**Symptoms:**

- Reports missing sections
- HTML rendering issues
- Data inconsistencies in reports

**Diagnostic Steps:**

```bash
# Validate report data
python tools/health_checker/health_reporter.py --validate

# Check report generation process
python tools/health_checker/health_reporter.py --debug

# Compare with raw health data
python tools/health_checker/health_checker.py --raw-output
```

**Solutions:**

1. **Fix Data Collection Issues**

   ```bash
   # Ensure all health checks complete
   python tools/health_checker/health_checker.py --comprehensive

   # Check for missing components
   python tools/health_checker/health_checker.py --missing-components
   ```

2. **Update Report Templates**
   ```bash
   # Regenerate report templates
   python tools/health_checker/health_reporter.py --update-templates
   ```

### 5. Configuration Issues

#### Issue: Configuration Validation Errors

**Symptoms:**

- Service fails to start with config errors
- Invalid configuration warnings in logs
- Features not working as expected

**Diagnostic Steps:**

```bash
# Validate all configurations
python tools/config_manager/config_validator.py --all

# Check specific config files
python tools/config_manager/config_validator.py config/production-health.yaml
python tools/config_manager/config_validator.py config/alerting-config.yaml

# Check YAML syntax
python -c "import yaml; yaml.safe_load(open('config/production-health.yaml'))"
```

**Solutions:**

1. **Fix YAML Syntax Errors**

   ```bash
   # Use YAML validator
   python -c "
   import yaml
   try:
       with open('config/production-health.yaml') as f:
           yaml.safe_load(f)
       print('YAML syntax OK')
   except yaml.YAMLError as e:
       print(f'YAML Error: {e}')
   "
   ```

2. **Restore from Backup**

   ```bash
   # Restore from backup
   cp backups/config/$(date +%Y%m%d)/production-health.yaml config/

   # Or restore default configuration
   python tools/config_manager/config_unifier.py --restore-defaults
   ```

3. **Migrate Configuration**
   ```bash
   # Migrate old configuration format
   python tools/config_manager/migration_cli.py --migrate config/old-config.yaml
   ```

### 6. Performance Issues

#### Issue: High CPU/Memory Usage

**Symptoms:**

- System becomes slow during health checks
- High CPU usage by health monitoring processes
- Out of memory errors

**Diagnostic Steps:**

```bash
# Monitor resource usage
top -p $(pgrep -f health_monitor)
htop

# Check memory usage patterns
python tools/health_checker/performance_optimizer.py --memory-profile

# Analyze CPU usage
python tools/health_checker/performance_optimizer.py --cpu-profile
```

**Solutions:**

1. **Optimize Health Checks**

   ```yaml
   # config/production-health.yaml
   max_concurrent_checks: 2 # Reduce parallelism
   enable_caching: true # Enable result caching
   cache_duration_minutes: 30
   ```

2. **Implement Resource Limits**

   ```bash
   # Set systemd resource limits
   sudo systemctl edit wan22-health-monitor

   # Add:
   [Service]
   MemoryLimit=512M
   CPUQuota=50%
   ```

3. **Schedule During Low Usage**
   ```yaml
   # config/production-health.yaml
   daily_report_time: "03:00" # Run during low usage hours
   critical_check_interval: 30 # Reduce frequency
   ```

## Diagnostic Tools

### Health System Diagnostics

```bash
# Comprehensive system check
python tools/health_checker/health_checker.py --diagnose

# Performance analysis
python tools/health_checker/performance_optimizer.py --full-analysis

# Configuration validation
python tools/config_manager/config_validator.py --comprehensive

# Alert system check
python scripts/manage_alerts.py diagnose
```

### Log Analysis Tools

```bash
# Analyze health monitoring logs
python tools/health_checker/health_analytics.py --analyze-logs

# Search for specific errors
grep -r "ERROR" logs/health-monitoring/

# Check for patterns
python tools/health_checker/health_analytics.py --pattern-analysis

# Generate log summary
python tools/health_checker/health_analytics.py --log-summary
```

### System Resource Monitoring

```bash
# Real-time resource monitoring
python tools/health_checker/production_health_checks.py --monitor

# Historical resource analysis
python tools/health_checker/performance_optimizer.py --resource-history

# Disk usage analysis
python tools/health_checker/performance_optimizer.py --disk-analysis
```

## Recovery Procedures

### Complete System Recovery

If the health monitoring system is completely down:

1. **Stop All Services**

   ```bash
   sudo systemctl stop wan22-health-monitor
   pkill -f health_monitor
   ```

2. **Backup Current State**

   ```bash
   mkdir -p recovery/$(date +%Y%m%d_%H%M%S)
   cp -r config/ recovery/$(date +%Y%m%d_%H%M%S)/
   cp -r logs/ recovery/$(date +%Y%m%d_%H%M%S)/
   ```

3. **Restore Known Good Configuration**

   ```bash
   # Restore from backup
   cp backups/config/latest/* config/

   # Or reset to defaults
   python tools/config_manager/config_unifier.py --reset-to-defaults
   ```

4. **Validate Configuration**

   ```bash
   python tools/config_manager/config_validator.py --all
   ```

5. **Restart Services**

   ```bash
   sudo systemctl start wan22-health-monitor
   sudo systemctl status wan22-health-monitor
   ```

6. **Verify Operation**

   ```bash
   # Test health check
   python -m tools.health_checker.health_checker --quick

   # Test alerting
   python scripts/manage_alerts.py test critical_health_score
   ```

### Partial Recovery

For specific component failures:

1. **Identify Failed Component**

   ```bash
   python tools/health_checker/health_checker.py --component-status
   ```

2. **Restart Specific Component**

   ```bash
   # Restart alerting system
   python -c "
   from tools.health_checker.automated_alerting import AutomatedAlertingSystem
   system = AutomatedAlertingSystem()
   # Restart logic here
   "
   ```

3. **Verify Component Recovery**
   ```bash
   python tools/health_checker/health_checker.py --component specific_component
   ```

## Prevention Strategies

### Monitoring Best Practices

1. **Regular Health Checks**

   - Monitor system health daily
   - Review weekly trends
   - Update baselines monthly

2. **Proactive Maintenance**

   - Regular configuration backups
   - Performance optimization
   - Security updates

3. **Alert Tuning**
   - Regular review of alert effectiveness
   - Adjust thresholds based on trends
   - Minimize false positives

### Configuration Management

1. **Version Control**

   ```bash
   # Track configuration changes
   git add config/
   git commit -m "Update health monitoring config"
   ```

2. **Automated Backups**

   ```bash
   # Daily configuration backup
   0 2 * * * cp -r /path/to/config /path/to/backups/$(date +\%Y\%m\%d)
   ```

3. **Change Validation**
   ```bash
   # Always validate before applying
   python tools/config_manager/config_validator.py config/production-health.yaml
   ```

## Escalation Procedures

### Level 1: Self-Service Resolution

- Use this troubleshooting guide
- Check logs and system status
- Apply common fixes
- **Time Limit**: 30 minutes

### Level 2: Team Support

- Contact dev-team@company.com
- Provide diagnostic information
- Include relevant logs
- **Response Time**: 1 hour

### Level 3: Emergency Response

- Contact emergency@company.com
- For critical system failures
- Include impact assessment
- **Response Time**: 15 minutes

### Information to Include in Escalation

```bash
# System information
uname -a
python --version
systemctl status wan22-health-monitor

# Recent logs
tail -50 logs/health-monitoring/health_monitor_$(date +%Y%m%d).log

# Configuration status
python tools/config_manager/config_validator.py --summary

# Resource usage
free -m
df -h
top -b -n 1 | head -20
```

---

**Document Version**: 1.0  
**Last Updated**: 2025-09-01  
**Emergency Contact**: emergency@company.com
