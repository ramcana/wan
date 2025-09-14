# Model Orchestrator Operational Runbook

## Overview

This runbook provides step-by-step procedures for common operational tasks with the WAN2.2 Model Orchestrator. It serves as a reference for system administrators, DevOps engineers, and on-call personnel.

## Table of Contents

1. [Daily Operations](#daily-operations)
2. [Weekly Maintenance](#weekly-maintenance)
3. [Monthly Tasks](#monthly-tasks)
4. [Emergency Procedures](#emergency-procedures)
5. [Deployment Procedures](#deployment-procedures)
6. [Backup and Recovery](#backup-and-recovery)
7. [Monitoring and Alerting](#monitoring-and-alerting)
8. [Capacity Planning](#capacity-planning)

## Daily Operations

### Morning Health Check

**Frequency:** Daily at 9:00 AM  
**Duration:** 5-10 minutes  
**Owner:** Operations Team

#### Procedure

1. **Check Service Status**

   ```bash
   # Verify all services are running
   systemctl status wan22
   systemctl status postgresql
   systemctl status nginx  # If using reverse proxy

   # Expected: All services should be "active (running)"
   ```

2. **Verify Health Endpoints**

   ```bash
   # Check main health endpoint
   curl -f http://localhost:8000/health

   # Check detailed health
   curl -s http://localhost:8000/health/models | jq '.summary'

   # Expected: HTTP 200, all components healthy
   ```

3. **Review Overnight Activity**

   ```bash
   # Check for errors in the last 24 hours
   journalctl -u wan22 --since "24 hours ago" --grep ERROR | wc -l

   # Review download activity
   grep "download.*completed" /var/log/wan22/orchestrator.log | tail -10

   # Expected: <10 errors, successful downloads
   ```

4. **Check Resource Usage**

   ```bash
   # Disk usage
   df -h /data/models

   # Memory usage
   free -h

   # CPU load
   uptime

   # Expected: <80% disk, <70% memory, load <2.0
   ```

5. **Verify Model Availability**

   ```bash
   # Check critical models
   wan models status --only t2v-A14B@2.2.0,i2v-A14B@2.2.0,ti2v-5b@2.2.0

   # Expected: All critical models in COMPLETE state
   ```

#### Escalation Criteria

- Any service not running → Immediate escalation
- Health endpoint returning 5xx → Immediate escalation
- > 50 errors in 24 hours → Escalate to engineering
- > 90% disk usage → Immediate action required
- Critical models not available → Escalate to engineering

### Evening Cleanup

**Frequency:** Daily at 11:00 PM  
**Duration:** 2-3 minutes  
**Owner:** Automated (cron job)

#### Procedure

1. **Automated Cleanup Script**

   ```bash
   #!/bin/bash
   # /opt/wan22/scripts/daily-cleanup.sh

   # Clean temporary files older than 1 day
   find /data/models/.tmp -type f -mtime +1 -delete

   # Clean old lock files
   find /data/models/.locks -name "*.lock" -mtime +0.04 -delete

   # Rotate logs
   logrotate /etc/logrotate.d/wan22

   # Run garbage collection if needed
   wan models gc --auto

   # Send daily report
   wan models report --daily | mail -s "WAN22 Daily Report" ops@company.com
   ```

2. **Cron Configuration**
   ```bash
   # Add to /etc/crontab
   0 23 * * * wan22 /opt/wan22/scripts/daily-cleanup.sh
   ```

## Weekly Maintenance

### System Maintenance Window

**Frequency:** Weekly, Sunday 2:00 AM - 4:00 AM  
**Duration:** 2 hours  
**Owner:** Operations Team

#### Pre-Maintenance Checklist

1. **Notify Stakeholders**

   ```bash
   # Send maintenance notification
   echo "Maintenance window starting at 2:00 AM" | \
   mail -s "WAN22 Maintenance Window" users@company.com
   ```

2. **Create Backup**

   ```bash
   # Full system backup
   /opt/wan22/scripts/backup-system.sh

   # Verify backup integrity
   /opt/wan22/scripts/verify-backup.sh
   ```

3. **Check System Health**
   ```bash
   # Pre-maintenance health check
   /opt/wan22/scripts/health-check-comprehensive.sh > /tmp/pre-maintenance-health.log
   ```

#### Maintenance Procedures

1. **Update System Packages**

   ```bash
   # Update OS packages
   sudo apt update && sudo apt upgrade -y

   # Update Python packages
   cd /opt/wan22
   source venv/bin/activate
   pip list --outdated
   # Review and update as needed
   ```

2. **Database Maintenance**

   ```bash
   # Stop application
   sudo systemctl stop wan22

   # Database maintenance
   sudo -u postgres psql wan22 -c "VACUUM ANALYZE;"
   sudo -u postgres psql wan22 -c "REINDEX DATABASE wan22;"

   # Update statistics
   sudo -u postgres psql wan22 -c "ANALYZE;"

   # Start application
   sudo systemctl start wan22
   ```

3. **Storage Optimization**

   ```bash
   # Run comprehensive garbage collection
   wan models gc --deep-clean

   # Defragment if needed (ext4)
   e4defrag /data/models

   # Check filesystem
   fsck -n /dev/sdb1  # Read-only check
   ```

4. **Security Updates**

   ```bash
   # Check for security updates
   apt list --upgradable | grep -i security

   # Update certificates if needed
   certbot renew --dry-run

   # Review access logs
   grep "401\|403\|404" /var/log/nginx/access.log | tail -20
   ```

#### Post-Maintenance Verification

1. **System Health Check**

   ```bash
   # Comprehensive health check
   /opt/wan22/scripts/health-check-comprehensive.sh > /tmp/post-maintenance-health.log

   # Compare with pre-maintenance
   diff /tmp/pre-maintenance-health.log /tmp/post-maintenance-health.log
   ```

2. **Performance Verification**

   ```bash
   # Test download performance
   wan models ensure --only test-model@1.0.0 --benchmark

   # Check response times
   curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8000/health
   ```

3. **Notify Completion**
   ```bash
   # Send completion notification
   echo "Maintenance window completed successfully" | \
   mail -s "WAN22 Maintenance Complete" users@company.com
   ```

## Monthly Tasks

### Capacity Planning Review

**Frequency:** First Monday of each month  
**Duration:** 1-2 hours  
**Owner:** Infrastructure Team

#### Data Collection

1. **Storage Growth Analysis**

   ```bash
   # Generate storage report
   wan models report --storage --last-30-days > /tmp/storage-report.txt

   # Analyze growth trends
   awk '{print $1, $2}' /var/log/wan22/storage-daily.log | \
   gnuplot -e "set terminal png; set output 'storage-trend.png'; plot '-' with lines"
   ```

2. **Performance Metrics**

   ```bash
   # Download performance trends
   curl -s http://localhost:9090/api/v1/query_range?query=rate(model_download_duration_seconds_sum[1h])&start=$(date -d '30 days ago' +%s)&end=$(date +%s)&step=3600

   # Error rate trends
   curl -s http://localhost:9090/api/v1/query_range?query=rate(model_errors_total[1h])&start=$(date -d '30 days ago' +%s)&end=$(date +%s)&step=3600
   ```

3. **Usage Patterns**

   ```bash
   # Most requested models
   grep "ensure.*completed" /var/log/wan22/orchestrator.log | \
   awk '{print $5}' | sort | uniq -c | sort -nr | head -10

   # Peak usage times
   grep "ensure" /var/log/wan22/orchestrator.log | \
   awk '{print $1, $2}' | cut -c1-13 | sort | uniq -c
   ```

#### Capacity Recommendations

1. **Storage Scaling**

   ```bash
   # Calculate projected storage needs
   CURRENT_USAGE=$(du -sb /data/models | cut -f1)
   GROWTH_RATE=$(echo "scale=2; $CURRENT_USAGE / 30" | bc)  # Per day
   PROJECTED_90_DAYS=$(echo "scale=0; $CURRENT_USAGE + ($GROWTH_RATE * 90)" | bc)

   echo "Current usage: $(numfmt --to=iec $CURRENT_USAGE)"
   echo "Projected 90-day usage: $(numfmt --to=iec $PROJECTED_90_DAYS)"
   ```

2. **Performance Scaling**

   ```bash
   # Analyze concurrent request patterns
   grep "concurrent_requests" /var/log/wan22/metrics.log | \
   awk '{sum+=$3; count++} END {print "Average concurrent requests:", sum/count}'

   # Recommend scaling if average > 80% of capacity
   ```

### Security Review

**Frequency:** Monthly  
**Duration:** 2-3 hours  
**Owner:** Security Team

#### Security Audit

1. **Access Review**

   ```bash
   # Review user access
   sudo lastlog | grep wan22

   # Check SSH keys
   sudo find /home -name "authorized_keys" -exec ls -la {} \;

   # Review sudo access
   sudo grep wan22 /etc/sudoers /etc/sudoers.d/*
   ```

2. **Certificate Management**

   ```bash
   # Check certificate expiration
   openssl x509 -in /etc/ssl/certs/wan22.crt -noout -dates

   # Verify certificate chain
   openssl verify -CAfile /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/wan22.crt
   ```

3. **Vulnerability Scanning**

   ```bash
   # System vulnerability scan
   sudo apt install -y lynis
   sudo lynis audit system

   # Application dependency scan
   cd /opt/wan22
   pip-audit
   ```

4. **Log Analysis**
   ```bash
   # Security-related log analysis
   grep -E "(failed|denied|unauthorized)" /var/log/auth.log | tail -20
   grep -E "(401|403)" /var/log/nginx/access.log | tail -20
   ```

## Emergency Procedures

### Service Outage Response

**Trigger:** Service health check fails or alerts fire  
**Response Time:** 5 minutes  
**Owner:** On-call Engineer

#### Immediate Response (0-5 minutes)

1. **Assess Situation**

   ```bash
   # Check service status
   systemctl status wan22

   # Check system resources
   top -bn1 | head -20
   df -h
   free -h

   # Check recent logs
   journalctl -u wan22 --since "10 minutes ago" | tail -20
   ```

2. **Quick Recovery Attempts**

   ```bash
   # Try service restart
   sudo systemctl restart wan22
   sleep 30

   # Verify recovery
   curl -f http://localhost:8000/health

   # If successful, monitor for 10 minutes
   ```

#### Detailed Investigation (5-15 minutes)

1. **Root Cause Analysis**

   ```bash
   # Check for obvious issues
   dmesg | tail -20  # System messages
   journalctl --since "1 hour ago" | grep -i error

   # Check disk space
   df -h | grep -E "(9[0-9]%|100%)"

   # Check memory
   free -h | grep Mem

   # Check database
   sudo -u postgres psql -c "SELECT 1;" 2>&1
   ```

2. **Escalation Decision**
   ```bash
   # If issue not resolved in 15 minutes:
   # 1. Page senior engineer
   # 2. Notify incident channel
   # 3. Begin detailed troubleshooting
   ```

### Data Corruption Response

**Trigger:** Integrity verification failures or corrupted model reports  
**Response Time:** 10 minutes  
**Owner:** Senior Engineer

#### Assessment Phase

1. **Scope Assessment**

   ```bash
   # Check affected models
   wan models verify --all --report-only

   # Check filesystem integrity
   sudo fsck -n /dev/sdb1

   # Check recent backup status
   ls -la /backup/wan22/ | tail -10
   ```

2. **Immediate Containment**

   ```bash
   # Stop service to prevent further corruption
   sudo systemctl stop wan22

   # Mark affected models as corrupted
   wan models mark-corrupted --model-list /tmp/corrupted-models.txt

   # Notify users
   echo "Service temporarily unavailable due to data integrity issue" | \
   wall
   ```

#### Recovery Phase

1. **Selective Recovery**

   ```bash
   # Restore from backup (specific models)
   /opt/wan22/scripts/restore-models.sh --models t2v-A14B@2.2.0,i2v-A14B@2.2.0

   # Re-download corrupted models
   wan models ensure --force-redownload --model-list /tmp/corrupted-models.txt

   # Verify integrity
   wan models verify --model-list /tmp/corrupted-models.txt
   ```

2. **Service Recovery**

   ```bash
   # Start service
   sudo systemctl start wan22

   # Verify functionality
   curl -f http://localhost:8000/health
   wan models status --critical-only

   # Monitor for stability
   watch -n 30 'curl -s http://localhost:8000/health | jq .status'
   ```

### Security Incident Response

**Trigger:** Security alerts or suspicious activity  
**Response Time:** Immediate  
**Owner:** Security Team + On-call Engineer

#### Immediate Actions

1. **Containment**

   ```bash
   # Block suspicious IPs (if identified)
   sudo iptables -A INPUT -s SUSPICIOUS_IP -j DROP

   # Disable compromised accounts (if any)
   sudo usermod -L compromised_user

   # Change critical passwords
   # (Follow security team procedures)
   ```

2. **Evidence Collection**

   ```bash
   # Preserve logs
   cp /var/log/wan22/* /tmp/incident-logs/
   cp /var/log/auth.log /tmp/incident-logs/
   cp /var/log/nginx/access.log /tmp/incident-logs/

   # Network connections
   netstat -tulpn > /tmp/incident-logs/netstat.txt

   # Process list
   ps auxf > /tmp/incident-logs/processes.txt
   ```

## Deployment Procedures

### Application Deployment

**Frequency:** As needed (typically bi-weekly)  
**Duration:** 30-60 minutes  
**Owner:** DevOps Team

#### Pre-Deployment

1. **Preparation**

   ```bash
   # Create deployment branch
   git checkout -b deploy-$(date +%Y%m%d)

   # Run tests
   python -m pytest tests/ -v

   # Build deployment package
   /opt/wan22/scripts/build-deployment.sh
   ```

2. **Staging Deployment**

   ```bash
   # Deploy to staging
   /opt/wan22/scripts/deploy-staging.sh

   # Run integration tests
   python -m pytest tests/integration/ --staging

   # Performance testing
   /opt/wan22/scripts/performance-test.sh --staging
   ```

#### Production Deployment

1. **Blue-Green Deployment**

   ```bash
   # Prepare green environment
   /opt/wan22/scripts/prepare-green.sh

   # Deploy to green
   /opt/wan22/scripts/deploy-green.sh

   # Health check green
   curl -f http://green.internal:8000/health

   # Switch traffic
   /opt/wan22/scripts/switch-to-green.sh

   # Monitor for 15 minutes
   /opt/wan22/scripts/monitor-deployment.sh
   ```

2. **Rollback Procedure (if needed)**

   ```bash
   # Switch back to blue
   /opt/wan22/scripts/switch-to-blue.sh

   # Verify rollback
   curl -f http://localhost:8000/health

   # Investigate issues
   /opt/wan22/scripts/deployment-postmortem.sh
   ```

### Configuration Updates

**Frequency:** As needed  
**Duration:** 10-15 minutes  
**Owner:** Operations Team

#### Configuration Change Process

1. **Preparation**

   ```bash
   # Backup current configuration
   cp /opt/wan22/.env /opt/wan22/.env.backup.$(date +%s)

   # Validate new configuration
   /opt/wan22/scripts/validate-config.sh /path/to/new/config
   ```

2. **Deployment**

   ```bash
   # Apply configuration
   cp /path/to/new/config /opt/wan22/.env

   # Reload service (if hot-reload supported)
   sudo systemctl reload wan22

   # Or restart if needed
   sudo systemctl restart wan22

   # Verify changes
   curl -s http://localhost:8000/config | jq .
   ```

## Backup and Recovery

### Backup Procedures

#### Daily Automated Backup

**Schedule:** 2:00 AM daily  
**Retention:** 7 days local, 30 days remote  
**Owner:** Automated system

```bash
#!/bin/bash
# /opt/wan22/scripts/daily-backup.sh

BACKUP_DIR="/backup/wan22"
DATE=$(date +%Y%m%d_%H%M%S)
MODELS_ROOT="/data/models"

# Create backup directory
mkdir -p "$BACKUP_DIR/daily"

# Backup model metadata and small files
rsync -av \
    --include="*.json" \
    --include="*.yaml" \
    --include="*.toml" \
    --exclude="*.safetensors" \
    --exclude="*.bin" \
    "$MODELS_ROOT/" "$BACKUP_DIR/daily/metadata_$DATE/"

# Backup database
pg_dump wan22 | gzip > "$BACKUP_DIR/daily/database_$DATE.sql.gz"

# Backup configuration
tar -czf "$BACKUP_DIR/daily/config_$DATE.tar.gz" /opt/wan22/.env /opt/wan22/config/

# Upload to remote storage
aws s3 sync "$BACKUP_DIR/daily/" s3://wan22-backups/daily/

# Cleanup old local backups
find "$BACKUP_DIR/daily" -name "*_*" -mtime +7 -delete

# Send backup report
echo "Daily backup completed: $DATE" | \
mail -s "WAN22 Backup Report" ops@company.com
```

#### Weekly Full Backup

**Schedule:** Sunday 1:00 AM  
**Retention:** 4 weeks local, 12 weeks remote  
**Owner:** Automated system

```bash
#!/bin/bash
# /opt/wan22/scripts/weekly-backup.sh

BACKUP_DIR="/backup/wan22"
DATE=$(date +%Y%m%d)
MODELS_ROOT="/data/models"

# Create full backup
mkdir -p "$BACKUP_DIR/weekly"

# Backup all model data
tar -czf "$BACKUP_DIR/weekly/models_full_$DATE.tar.gz" \
    --exclude=".tmp" \
    --exclude=".locks" \
    "$MODELS_ROOT"

# Backup system configuration
tar -czf "$BACKUP_DIR/weekly/system_$DATE.tar.gz" \
    /opt/wan22 \
    /etc/systemd/system/wan22.service \
    /etc/nginx/sites-available/wan22

# Upload to remote storage
aws s3 sync "$BACKUP_DIR/weekly/" s3://wan22-backups/weekly/

# Cleanup old backups
find "$BACKUP_DIR/weekly" -name "*_*" -mtime +28 -delete
```

### Recovery Procedures

#### Point-in-Time Recovery

```bash
#!/bin/bash
# /opt/wan22/scripts/point-in-time-recovery.sh

RECOVERY_DATE="$1"  # Format: YYYYMMDD_HHMMSS
BACKUP_DIR="/backup/wan22"

if [ -z "$RECOVERY_DATE" ]; then
    echo "Usage: $0 YYYYMMDD_HHMMSS"
    exit 1
fi

echo "Starting point-in-time recovery to $RECOVERY_DATE"

# Stop services
sudo systemctl stop wan22
sudo systemctl stop nginx

# Backup current state
mv /data/models /data/models.pre-recovery.$(date +%s)
mv /opt/wan22 /opt/wan22.pre-recovery.$(date +%s)

# Restore from backup
tar -xzf "$BACKUP_DIR/weekly/models_full_$RECOVERY_DATE.tar.gz" -C /data/
tar -xzf "$BACKUP_DIR/weekly/system_$RECOVERY_DATE.tar.gz" -C /

# Restore database
sudo -u postgres dropdb wan22
sudo -u postgres createdb wan22
gunzip -c "$BACKUP_DIR/daily/database_$RECOVERY_DATE.sql.gz" | \
sudo -u postgres psql wan22

# Fix permissions
sudo chown -R wan22:wan22 /data/models /opt/wan22

# Start services
sudo systemctl start wan22
sudo systemctl start nginx

# Verify recovery
sleep 30
curl -f http://localhost:8000/health

echo "Point-in-time recovery completed"
```

## Monitoring and Alerting

### Alert Response Procedures

#### Critical Alerts

**Service Down Alert**

```bash
# Response time: 2 minutes
# 1. Check service status
systemctl status wan22

# 2. Check system resources
df -h && free -h && uptime

# 3. Attempt restart
sudo systemctl restart wan22

# 4. Verify recovery
curl -f http://localhost:8000/health

# 5. If not recovered, escalate immediately
```

**High Error Rate Alert**

```bash
# Response time: 5 minutes
# 1. Check recent errors
journalctl -u wan22 --since "15 minutes ago" | grep ERROR

# 2. Identify error patterns
grep ERROR /var/log/wan22/orchestrator.log | tail -20 | \
awk '{print $NF}' | sort | uniq -c

# 3. Check affected models
wan models status --failed-only

# 4. Take corrective action based on error type
```

**Disk Space Alert**

```bash
# Response time: 10 minutes
# 1. Check current usage
df -h /data/models

# 2. Run garbage collection
wan models gc --aggressive

# 3. Check for large temporary files
find /data/models/.tmp -size +1G -ls

# 4. Clean up if safe
rm -rf /data/models/.tmp/*

# 5. If still critical, escalate for storage expansion
```

#### Warning Alerts

**High Memory Usage**

```bash
# Response time: 15 minutes
# 1. Check memory usage
free -h && ps aux --sort=-%mem | head -10

# 2. Check for memory leaks
pmap $(pgrep -f wan22) | tail -1

# 3. Restart service if memory usage > 90%
if [ $(free | grep Mem | awk '{print $3/$2 * 100.0}') -gt 90 ]; then
    sudo systemctl restart wan22
fi
```

### Monitoring Dashboard Setup

#### Grafana Dashboard Configuration

```json
{
  "dashboard": {
    "title": "WAN2.2 Model Orchestrator Operations",
    "panels": [
      {
        "title": "Service Health",
        "type": "stat",
        "targets": [{ "expr": "up{job=\"wan22-orchestrator\"}" }]
      },
      {
        "title": "Download Success Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(model_downloads_total{status=\"success\"}[5m]) / rate(model_downloads_total[5m]) * 100"
          }
        ]
      },
      {
        "title": "Storage Usage",
        "type": "gauge",
        "targets": [
          {
            "expr": "model_storage_bytes_used / model_storage_bytes_total * 100"
          }
        ]
      },
      {
        "title": "Active Downloads",
        "type": "graph",
        "targets": [{ "expr": "model_downloads_active" }]
      }
    ]
  }
}
```

## Capacity Planning

### Growth Projections

#### Monthly Capacity Review

```bash
#!/bin/bash
# /opt/wan22/scripts/capacity-review.sh

echo "=== Monthly Capacity Review ==="
echo "Date: $(date)"

# Storage growth analysis
echo "Storage Growth:"
CURRENT_SIZE=$(du -sb /data/models | cut -f1)
LAST_MONTH_SIZE=$(cat /var/log/wan22/capacity-$(date -d '1 month ago' +%Y%m).log 2>/dev/null || echo "0")
GROWTH=$((CURRENT_SIZE - LAST_MONTH_SIZE))
GROWTH_PERCENT=$(echo "scale=2; $GROWTH / $LAST_MONTH_SIZE * 100" | bc 2>/dev/null || echo "N/A")

echo "Current size: $(numfmt --to=iec $CURRENT_SIZE)"
echo "Growth this month: $(numfmt --to=iec $GROWTH) ($GROWTH_PERCENT%)"

# Save current size for next month
echo "$CURRENT_SIZE" > /var/log/wan22/capacity-$(date +%Y%m).log

# Performance trends
echo "Performance Trends:"
AVG_DOWNLOAD_TIME=$(grep "download.*completed" /var/log/wan22/orchestrator.log | \
awk '{print $NF}' | awk '{sum+=$1; count++} END {print sum/count}')
echo "Average download time: ${AVG_DOWNLOAD_TIME}s"

# Usage patterns
echo "Usage Patterns:"
DAILY_REQUESTS=$(grep "ensure" /var/log/wan22/orchestrator.log | \
grep "$(date +%Y-%m-%d)" | wc -l)
echo "Requests today: $DAILY_REQUESTS"

# Recommendations
echo "Recommendations:"
if [ $GROWTH_PERCENT -gt 20 ]; then
    echo "- Consider storage expansion (>20% growth)"
fi

if [ $(echo "$AVG_DOWNLOAD_TIME > 300" | bc) -eq 1 ]; then
    echo "- Investigate download performance (>5min average)"
fi
```

### Scaling Recommendations

#### Horizontal Scaling Triggers

```bash
# CPU utilization > 70% for 1 hour
# Memory utilization > 80% for 30 minutes
# Concurrent requests > 80% of capacity
# Download queue length > 50

# Scaling actions:
# 1. Add additional application instances
# 2. Implement load balancing
# 3. Scale storage backend
# 4. Optimize database connections
```

#### Vertical Scaling Triggers

```bash
# Single instance CPU > 90% for 15 minutes
# Memory usage approaching limits
# I/O wait time > 20%
# Network bandwidth utilization > 80%

# Scaling actions:
# 1. Increase CPU cores
# 2. Add more RAM
# 3. Upgrade to faster storage
# 4. Increase network bandwidth
```

This operational runbook provides comprehensive procedures for managing the Model Orchestrator in production environments. It should be regularly updated based on operational experience and system changes.
