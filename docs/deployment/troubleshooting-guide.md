# Troubleshooting Guide

This guide provides comprehensive troubleshooting procedures for common issues with the project health system.

## Quick Reference

### Emergency Contacts

- **On-call Engineer**: +1-555-0123
- **DevOps Team**: devops@company.com
- **System Administrator**: sysadmin@company.com

### Critical Commands

```bash
# Check service status
systemctl status project-health

# View recent logs
journalctl -u project-health -f

# Run health check
python -m tools.health_checker.cli --quick-check

# Emergency restart
systemctl restart project-health
```

## Common Issues and Solutions

### 1. Service Won't Start

#### Symptoms

- Service fails to start
- `systemctl status project-health` shows "failed" or "inactive"
- Error messages in system logs

#### Diagnostic Steps

```bash
# Check service status
systemctl status project-health

# Check detailed logs
journalctl -u project-health --no-pager

# Check configuration syntax
python -m tools.config_manager.config_cli validate

# Check file permissions
ls -la /opt/project-health/
ls -la /etc/project-health/

# Check dependencies
pip check
```

#### Common Causes and Solutions

**Configuration Syntax Error**

```bash
# Symptom: YAML parsing errors in logs
# Solution: Validate and fix configuration
python -m tools.config_manager.config_cli validate --config /etc/project-health/production.yaml

# If validation fails, check for:
# - Missing quotes around strings
# - Incorrect indentation
# - Invalid YAML syntax
```

**Permission Issues**

```bash
# Symptom: Permission denied errors
# Solution: Fix file permissions
sudo chown -R project-health:project-health /opt/project-health
sudo chmod 755 /opt/project-health
sudo chmod 640 /etc/project-health/*.yaml
```

**Missing Dependencies**

```bash
# Symptom: ImportError or ModuleNotFoundError
# Solution: Install missing dependencies
pip install -r /opt/project-health/requirements.txt

# For specific missing modules:
pip install <module-name>
```

**Port Already in Use**

```bash
# Symptom: "Address already in use" error
# Solution: Find and stop conflicting process
sudo netstat -tulpn | grep :8080
sudo kill <PID>

# Or change port in configuration
# Edit /etc/project-health/production.yaml
```

### 2. High Memory Usage

#### Symptoms

- System becomes slow or unresponsive
- Out of memory errors in logs
- High memory usage in monitoring

#### Diagnostic Steps

```bash
# Check memory usage
free -h
ps aux --sort=-%mem | head -20

# Check for memory leaks
python -m tools.health_checker.cli --memory-analysis

# Monitor memory over time
watch -n 5 'free -h && ps aux --sort=-%mem | head -10'
```

#### Solutions

**Memory Leak in Application**

```bash
# Restart service to free memory
sudo systemctl restart project-health

# Enable memory profiling
export PYTHONMALLOC=debug
systemctl restart project-health

# Check for memory leaks in code
python -m tools.health_checker.cli --profile-memory
```

**Insufficient System Memory**

```bash
# Add swap space (temporary solution)
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Permanent solution: Upgrade system memory
```

**Large Log Files**

```bash
# Check log file sizes
du -sh /var/log/project-health/*

# Rotate logs immediately
sudo logrotate -f /etc/logrotate.d/project-health

# Adjust log retention in configuration
```

### 3. Slow Performance

#### Symptoms

- Health checks take longer than expected
- Web dashboard is slow to load
- Test execution times out

#### Diagnostic Steps

```bash
# Check system load
uptime
htop

# Check I/O wait
iostat -x 1 5

# Profile application performance
python -m tools.health_checker.cli --performance-profile

# Check database performance
sqlite3 /var/lib/project-health/health.db "PRAGMA optimize;"
```

#### Solutions

**High CPU Usage**

```bash
# Identify CPU-intensive processes
top -o %CPU

# Reduce parallel workers in configuration
# Edit /etc/project-health/production.yaml:
# test_execution:
#   max_workers: 2  # Reduce from default

# Enable CPU throttling if needed
echo 80 > /sys/devices/system/cpu/intel_pstate/max_perf_pct
```

**Disk I/O Bottleneck**

```bash
# Check disk usage and performance
df -h
iotop -o

# Move to faster storage (SSD)
# Or optimize database
sqlite3 /var/lib/project-health/health.db "VACUUM; ANALYZE;"

# Enable database WAL mode for better performance
sqlite3 /var/lib/project-health/health.db "PRAGMA journal_mode=WAL;"
```

**Network Latency**

```bash
# Check network connectivity
ping -c 5 google.com

# Test local network performance
iperf3 -c <server-ip>

# Optimize network settings
echo 'net.core.rmem_max = 16777216' >> /etc/sysctl.conf
echo 'net.core.wmem_max = 16777216' >> /etc/sysctl.conf
sysctl -p
```

### 4. Database Issues

#### Symptoms

- Database corruption errors
- SQLite locking errors
- Data inconsistencies

#### Diagnostic Steps

```bash
# Check database integrity
sqlite3 /var/lib/project-health/health.db "PRAGMA integrity_check;"

# Check database size and statistics
sqlite3 /var/lib/project-health/health.db ".dbinfo"

# Check for locks
lsof /var/lib/project-health/health.db

# Backup database before repairs
cp /var/lib/project-health/health.db /var/lib/project-health/health.db.backup
```

#### Solutions

**Database Corruption**

```bash
# Stop service first
sudo systemctl stop project-health

# Attempt repair
sqlite3 /var/lib/project-health/health.db "PRAGMA integrity_check;"

# If corruption is found, try to recover
sqlite3 /var/lib/project-health/health.db ".recover" | sqlite3 /var/lib/project-health/health_recovered.db

# Restore from backup if recovery fails
cp /backup/project-health/latest/health.db /var/lib/project-health/health.db

# Restart service
sudo systemctl start project-health
```

**Database Locking**

```bash
# Find processes using the database
lsof /var/lib/project-health/health.db

# Kill processes if necessary
sudo kill -9 <PID>

# Enable WAL mode to reduce locking
sqlite3 /var/lib/project-health/health.db "PRAGMA journal_mode=WAL;"

# Restart service
sudo systemctl restart project-health
```

**Large Database Size**

```bash
# Vacuum database to reclaim space
sqlite3 /var/lib/project-health/health.db "VACUUM;"

# Archive old data
python -m tools.health_checker.cli --archive-old-data --days 90

# Set up automatic cleanup
# Add to cron: 0 2 * * 0 python -m tools.health_checker.cli --cleanup-old-data
```

### 5. Configuration Issues

#### Symptoms

- Configuration validation errors
- Settings not taking effect
- Service fails to load configuration

#### Diagnostic Steps

```bash
# Validate configuration syntax
python -m tools.config_manager.config_cli validate

# Check configuration file permissions
ls -la /etc/project-health/

# Test configuration loading
python -c "
import yaml
with open('/etc/project-health/production.yaml', 'r') as f:
    config = yaml.safe_load(f)
    print('Configuration loaded successfully')
    print(f'Keys: {list(config.keys())}')
"

# Check environment variables
env | grep PROJECT
```

#### Solutions

**YAML Syntax Errors**

```bash
# Use YAML validator
python -c "
import yaml
try:
    with open('/etc/project-health/production.yaml', 'r') as f:
        yaml.safe_load(f)
    print('YAML is valid')
except yaml.YAMLError as e:
    print(f'YAML error: {e}')
"

# Common fixes:
# - Add quotes around strings with special characters
# - Fix indentation (use spaces, not tabs)
# - Escape backslashes in paths
```

**Configuration Not Loading**

```bash
# Check file path in service configuration
grep ExecStart /etc/systemd/system/project-health.service

# Check environment variables
systemctl show project-health | grep Environment

# Reload systemd configuration
sudo systemctl daemon-reload
sudo systemctl restart project-health
```

**Invalid Configuration Values**

```bash
# Check configuration schema
python -m tools.config_manager.config_cli validate --schema

# Reset to default configuration
cp /opt/project-health/config/default.yaml /etc/project-health/production.yaml

# Gradually add custom settings
```

### 6. Network and Connectivity Issues

#### Symptoms

- Cannot access web dashboard
- Health check endpoints not responding
- External service timeouts

#### Diagnostic Steps

```bash
# Check if service is listening on correct port
netstat -tulpn | grep :8080

# Test local connectivity
curl -I http://localhost:8080/health

# Check firewall rules
sudo ufw status
sudo iptables -L

# Test external connectivity
curl -I http://external-service.com
```

#### Solutions

**Service Not Listening**

```bash
# Check configuration for correct port
grep -i port /etc/project-health/production.yaml

# Check if port is available
sudo netstat -tulpn | grep :8080

# Start service if not running
sudo systemctl start project-health
```

**Firewall Blocking Connections**

```bash
# Allow port through firewall
sudo ufw allow 8080/tcp

# For iptables:
sudo iptables -A INPUT -p tcp --dport 8080 -j ACCEPT
sudo iptables-save > /etc/iptables/rules.v4
```

**DNS Resolution Issues**

```bash
# Test DNS resolution
nslookup external-service.com

# Use alternative DNS servers
echo "nameserver 8.8.8.8" >> /etc/resolv.conf

# Check /etc/hosts for overrides
cat /etc/hosts
```

### 7. Test Execution Issues

#### Symptoms

- Tests fail to run
- Test timeouts
- Inconsistent test results

#### Diagnostic Steps

```bash
# Run tests manually
python -m pytest tests/ -v

# Check test configuration
cat tests/config/test-config.yaml

# Check test environment
python -m tools.test_runner.cli --check-environment

# Run specific test category
python -m tools.test_runner.cli --category unit
```

#### Solutions

**Test Dependencies Missing**

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov

# Check for missing test files
find tests/ -name "*.py" -exec python -m py_compile {} \;

# Fix import errors
export PYTHONPATH=/opt/project-health:$PYTHONPATH
```

**Test Timeouts**

```bash
# Increase timeout in configuration
# Edit tests/config/test-config.yaml:
# test_categories:
#   unit:
#     timeout: 60  # Increase from 30

# Run tests with verbose output
python -m pytest tests/ -v -s --tb=short
```

**Flaky Tests**

```bash
# Identify flaky tests
python -m tools.test_runner.cli --identify-flaky

# Run tests multiple times
python -m pytest tests/ --count=5

# Fix or skip flaky tests
python -m pytest tests/ -m "not flaky"
```

### 8. Documentation Issues

#### Symptoms

- Documentation generation fails
- Broken links in documentation
- Missing documentation files

#### Diagnostic Steps

```bash
# Test documentation generation
python -m tools.doc_generator.cli --generate

# Check for broken links
python -m tools.doc_generator.cli --check-links

# Validate documentation structure
find docs/ -name "*.md" -exec python -c "
import sys
try:
    with open(sys.argv[1], 'r') as f:
        content = f.read()
    print(f'✓ {sys.argv[1]}')
except Exception as e:
    print(f'✗ {sys.argv[1]}: {e}')
" {} \;
```

#### Solutions

**Documentation Generation Fails**

```bash
# Check for missing source files
ls -la docs/

# Install documentation dependencies
pip install mkdocs mkdocs-material

# Generate with verbose output
python -m tools.doc_generator.cli --generate --verbose
```

**Broken Links**

```bash
# Fix broken internal links
python -m tools.doc_generator.cli --fix-links

# Update external links
# Manually review and update broken external links

# Exclude problematic links
# Add to .linkcheck-ignore file
```

### 9. Monitoring and Alerting Issues

#### Symptoms

- No alerts being sent
- Monitoring data missing
- Dashboard not updating

#### Diagnostic Steps

```bash
# Test alert system
python -m tools.health_checker.cli --test-alerts

# Check monitoring configuration
cat /etc/project-health/production.yaml | grep -A 10 monitoring

# Test notification channels
python -m tools.health_checker.cli --test-notifications

# Check dashboard logs
tail -f /var/log/project-health/dashboard.log
```

#### Solutions

**Alerts Not Sending**

```bash
# Check email configuration
python -c "
import smtplib
try:
    server = smtplib.SMTP('smtp.company.com', 587)
    server.starttls()
    print('SMTP connection successful')
    server.quit()
except Exception as e:
    print(f'SMTP error: {e}')
"

# Check Slack webhook
curl -X POST -H 'Content-type: application/json' \
  --data '{"text":"Test message"}' \
  https://hooks.slack.com/services/YOUR/WEBHOOK/URL
```

**Monitoring Data Missing**

```bash
# Check data collection
python -m tools.health_checker.cli --collect-metrics

# Verify database writes
sqlite3 /var/lib/project-health/health.db "SELECT COUNT(*) FROM health_reports WHERE date(timestamp) = date('now');"

# Check file permissions
ls -la /var/lib/project-health/
```

## Advanced Troubleshooting

### Performance Profiling

```bash
# CPU profiling
python -m cProfile -o profile.stats -m tools.health_checker.cli --full-check

# Memory profiling
python -m memory_profiler -m tools.health_checker.cli --full-check

# I/O profiling
strace -e trace=file python -m tools.health_checker.cli --full-check
```

### Log Analysis

```bash
# Search for specific errors
grep -r "ERROR" /var/log/project-health/ | tail -20

# Analyze error patterns
awk '/ERROR/ {print $1, $2, $NF}' /var/log/project-health/*.log | sort | uniq -c

# Generate log summary
python -c "
import re
from collections import Counter

errors = Counter()
with open('/var/log/project-health/health.log', 'r') as f:
    for line in f:
        if 'ERROR' in line:
            # Extract error type
            match = re.search(r'ERROR.*?(\w+Error|\w+Exception)', line)
            if match:
                errors[match.group(1)] += 1

print('Top errors:')
for error, count in errors.most_common(5):
    print(f'  {error}: {count}')
"
```

### System Resource Analysis

```bash
# Comprehensive system check
python -c "
import psutil
import json

system_info = {
    'cpu_percent': psutil.cpu_percent(interval=1),
    'memory': dict(psutil.virtual_memory()._asdict()),
    'disk': dict(psutil.disk_usage('/')._asdict()),
    'network': dict(psutil.net_io_counters()._asdict()),
    'processes': len(psutil.pids())
}

print(json.dumps(system_info, indent=2, default=str))
"

# Check for resource limits
ulimit -a

# Monitor resource usage over time
sar -u -r -d 1 60  # CPU, memory, disk for 60 seconds
```

## Escalation Procedures

### Level 1: Self-Service

1. Check this troubleshooting guide
2. Review recent logs
3. Try basic restart procedures
4. Check system resources

### Level 2: Team Support

1. Contact DevOps team: devops@company.com
2. Provide incident details:
   - Symptoms observed
   - Steps already taken
   - Log excerpts
   - System information

### Level 3: Emergency Response

1. Call on-call engineer: +1-555-0123
2. For critical issues:
   - Service completely down
   - Data corruption detected
   - Security breach suspected

### Level 4: Vendor Support

1. Contact system vendor support
2. Escalate to cloud provider if infrastructure issue
3. Engage external consultants if needed

## Prevention Strategies

### Proactive Monitoring

```bash
# Set up comprehensive monitoring
python -m tools.health_checker.cli --setup-monitoring

# Configure predictive alerts
# Edit monitoring configuration to alert on trends
```

### Regular Maintenance

```bash
# Schedule regular health checks
echo "0 */6 * * * python -m tools.health_checker.cli --full-check" | crontab -

# Automate log rotation
sudo logrotate -f /etc/logrotate.d/project-health

# Regular backups
echo "0 2 * * * /opt/project-health/scripts/backup.sh" | crontab -
```

### Documentation Updates

- Keep troubleshooting guide current
- Document new issues and solutions
- Update contact information regularly
- Review procedures quarterly

## Tools and Utilities

### Diagnostic Scripts

```bash
# System health check
/opt/project-health/scripts/system-health-check.sh

# Performance analysis
/opt/project-health/scripts/performance-analysis.sh

# Log analyzer
/opt/project-health/scripts/log-analyzer.py

# Configuration validator
python -m tools.config_manager.config_cli validate --all
```

### Monitoring Commands

```bash
# Real-time monitoring
watch -n 5 'python -m tools.health_checker.cli --quick-check'

# Resource monitoring
htop
iotop
nethogs

# Service monitoring
systemctl status project-health
journalctl -u project-health -f
```

### Recovery Tools

```bash
# Database recovery
/opt/project-health/scripts/database-recovery.sh

# Configuration recovery
/opt/project-health/scripts/config-recovery.sh

# Service recovery
/opt/project-health/scripts/service-recovery.sh
```

## Appendix

### Error Code Reference

| Code | Description                | Severity | Action            |
| ---- | -------------------------- | -------- | ----------------- |
| E001 | Configuration syntax error | High     | Fix YAML syntax   |
| E002 | Database connection failed | Critical | Check database    |
| E003 | Permission denied          | Medium   | Fix permissions   |
| E004 | Service timeout            | Medium   | Check performance |
| E005 | Memory exhausted           | High     | Restart service   |

### Log File Locations

| Component     | Log File                             | Purpose                  |
| ------------- | ------------------------------------ | ------------------------ |
| Main Service  | `/var/log/project-health/health.log` | General application logs |
| Test Runner   | `/var/log/project-health/tests.log`  | Test execution logs      |
| Documentation | `/var/log/project-health/docs.log`   | Documentation generation |
| Configuration | `/var/log/project-health/config.log` | Configuration changes    |
| System        | `/var/log/syslog`                    | System-level messages    |

### Configuration File Locations

| File                                         | Purpose             |
| -------------------------------------------- | ------------------- |
| `/etc/project-health/production.yaml`        | Main configuration  |
| `/etc/project-health/monitoring.yaml`        | Monitoring settings |
| `/etc/systemd/system/project-health.service` | Service definition  |
| `/etc/logrotate.d/project-health`            | Log rotation config |

This troubleshooting guide should help resolve most common issues with the project health system. For issues not covered here, please contact the support team with detailed information about the problem.
