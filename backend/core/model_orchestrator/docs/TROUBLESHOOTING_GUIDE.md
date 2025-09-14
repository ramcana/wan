# Model Orchestrator Troubleshooting Guide

## Overview

This guide provides comprehensive troubleshooting procedures for common issues with the WAN2.2 Model Orchestrator. It includes diagnostic steps, solutions, and preventive measures for operational problems.

## Quick Diagnostic Commands

### System Health Check

```bash
# Overall system status
wan models diagnose

# Service status
systemctl status wan22

# Health endpoint check
curl -f http://localhost:8000/health

# Model status overview
wan models status --json | jq '.summary'
```

### Log Analysis

```bash
# Recent errors
journalctl -u wan22 --since "1 hour ago" --grep ERROR

# Download failures
grep "download.*failed" /var/log/wan22/orchestrator.log | tail -10

# Lock timeouts
grep "lock.*timeout" /var/log/wan22/orchestrator.log | tail -10
```

## Common Issues and Solutions

### 1. Model Download Failures

#### Symptoms

- Models stuck in "DOWNLOADING" state
- Repeated download timeout errors
- Network connection errors
- Authentication failures

#### Diagnostic Steps

**Check Network Connectivity:**

```bash
# Test HuggingFace connectivity
curl -I https://huggingface.co

# Test S3 connectivity (if using S3)
aws s3 ls s3://your-bucket --endpoint-url $AWS_ENDPOINT_URL

# Check DNS resolution
nslookup huggingface.co
```

**Verify Credentials:**

```bash
# Test HuggingFace token
curl -H "Authorization: Bearer $HF_TOKEN" https://huggingface.co/api/whoami

# Test S3 credentials
aws sts get-caller-identity --endpoint-url $AWS_ENDPOINT_URL
```

**Check Download Status:**

```bash
# Detailed model status
wan models status --model t2v-A14B@2.2.0 --verbose

# Check for partial downloads
ls -la $MODELS_ROOT/.tmp/

# Check lock status
wan models locks --list
```

#### Solutions

**Network Issues:**

```bash
# Increase timeout values
export DOWNLOAD_TIMEOUT=7200  # 2 hours
export RETRY_ATTEMPTS=5

# Enable connection pooling
export MAX_CONNECTIONS_PER_HOST=10

# Use proxy if needed
export HTTP_PROXY=http://proxy.company.com:8080
export HTTPS_PROXY=http://proxy.company.com:8080
```

**Authentication Issues:**

```bash
# Refresh HuggingFace token
huggingface-cli login

# Update S3 credentials
aws configure set aws_access_key_id YOUR_KEY
aws configure set aws_secret_access_key YOUR_SECRET
```

**Cleanup Partial Downloads:**

```bash
# Remove stuck downloads
wan models cleanup --partial-downloads

# Force re-download
wan models ensure --only t2v-A14B@2.2.0 --force-redownload
```

### 2. Disk Space Issues

#### Symptoms

- "Insufficient disk space" errors
- Downloads failing with NO_SPACE error
- System running out of storage

#### Diagnostic Steps

**Check Disk Usage:**

```bash
# Overall disk usage
df -h

# Model directory usage
du -sh $MODELS_ROOT/*

# Largest files
find $MODELS_ROOT -type f -exec du -h {} + | sort -rh | head -20

# Temporary files
du -sh $MODELS_ROOT/.tmp/
```

**Check Garbage Collection Status:**

```bash
# GC configuration
wan models gc --status

# Reclaimable space
wan models gc --dry-run --verbose
```

#### Solutions

**Immediate Space Recovery:**

```bash
# Run garbage collection
wan models gc --aggressive

# Clean temporary files
rm -rf $MODELS_ROOT/.tmp/*
rm -rf $MODELS_ROOT/.locks/*

# Remove old log files
find /var/log/wan22 -name "*.log.*" -mtime +7 -delete
```

**Configure Automatic Cleanup:**

```bash
# Set storage limits
export MAX_TOTAL_SIZE=$((1024 * 1024**3))  # 1TB
export MAX_MODEL_AGE=2592000  # 30 days

# Enable automatic GC
export ENABLE_GARBAGE_COLLECTION=true
export GC_CHECK_INTERVAL=3600  # 1 hour
```

**Expand Storage:**

```bash
# Add new disk (example)
sudo fdisk /dev/sdb
sudo mkfs.ext4 /dev/sdb1
sudo mount /dev/sdb1 /data/models-new

# Migrate models
rsync -av $MODELS_ROOT/ /data/models-new/
sudo umount /data/models
sudo mount /dev/sdb1 /data/models
```

### 3. Lock Contention and Deadlocks

#### Symptoms

- Lock timeout errors
- Multiple processes waiting indefinitely
- High CPU usage with no progress

#### Diagnostic Steps

**Check Active Locks:**

```bash
# List all locks
wan models locks --list --verbose

# Check lock ages
find $MODELS_ROOT/.locks -name "*.lock" -exec stat -c "%Y %n" {} \; | sort -n

# Check processes holding locks
lsof +D $MODELS_ROOT/.locks
```

**Monitor Lock Contention:**

```bash
# Real-time lock monitoring
watch -n 1 'wan models locks --list'

# Check metrics
curl -s http://localhost:9090/metrics | grep lock_
```

#### Solutions

**Clean Stale Locks:**

```bash
# Remove old locks (older than 1 hour)
find $MODELS_ROOT/.locks -name "*.lock" -mtime +0.04 -delete

# Force cleanup all locks (use with caution)
wan models locks --cleanup --force
```

**Adjust Lock Settings:**

```bash
# Increase lock timeout
export LOCK_TIMEOUT=1800  # 30 minutes

# Reduce lock granularity
export LOCK_PER_VARIANT=false  # Lock per model, not per variant
```

**Restart Services:**

```bash
# Graceful restart
sudo systemctl reload wan22

# Force restart if needed
sudo systemctl restart wan22
```

### 4. Memory Issues

#### Symptoms

- Out of memory (OOM) kills
- Slow performance
- High swap usage
- Memory leaks

#### Diagnostic Steps

**Check Memory Usage:**

```bash
# Current memory usage
free -h
ps aux --sort=-%mem | head -10

# Memory usage over time
sar -r 1 10

# Check for memory leaks
valgrind --tool=memcheck --leak-check=full python -m backend.main
```

**Monitor Application Memory:**

```bash
# Process memory details
cat /proc/$(pgrep -f wan22)/status | grep -E "(VmSize|VmRSS|VmSwap)"

# Memory maps
pmap $(pgrep -f wan22)
```

#### Solutions

**Immediate Memory Relief:**

```bash
# Clear system caches
sudo sync && sudo sysctl vm.drop_caches=3

# Restart application
sudo systemctl restart wan22

# Reduce concurrent operations
export MAX_CONCURRENT_DOWNLOADS=2
```

**Optimize Memory Usage:**

```bash
# Tune garbage collection
export PYTHONHASHSEED=0
export MALLOC_TRIM_THRESHOLD_=100000

# Limit worker processes
export WORKERS=2  # Reduce from default

# Enable memory profiling
export PYTHONMALLOC=debug
```

**System-Level Tuning:**

```bash
# Adjust swappiness
echo 'vm.swappiness=10' >> /etc/sysctl.conf

# Increase memory limits
echo 'wan22 soft memlock unlimited' >> /etc/security/limits.conf
echo 'wan22 hard memlock unlimited' >> /etc/security/limits.conf
```

### 5. Performance Issues

#### Symptoms

- Slow download speeds
- High response times
- CPU bottlenecks
- I/O wait times

#### Diagnostic Steps

**System Performance:**

```bash
# CPU usage
top -p $(pgrep -f wan22)
htop

# I/O performance
iostat -x 1 5
iotop

# Network performance
iftop -i eth0
nethogs
```

**Application Performance:**

```bash
# Response time testing
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8000/health

# Download speed testing
wan models ensure --only small-test-model --benchmark

# Database performance
EXPLAIN ANALYZE SELECT * FROM model_status WHERE model_id = 'test';
```

#### Solutions

**Network Optimization:**

```bash
# Enable HF transfer acceleration
export HF_HUB_ENABLE_HF_TRANSFER=1

# Tune network buffers
echo 'net.core.rmem_max = 134217728' >> /etc/sysctl.conf
echo 'net.core.wmem_max = 134217728' >> /etc/sysctl.conf
sysctl -p

# Increase connection limits
export MAX_CONNECTIONS_PER_HOST=20
```

**I/O Optimization:**

```bash
# Use faster storage
# Move MODELS_ROOT to SSD/NVMe

# Optimize mount options
mount -o remount,noatime,nodiratime /data/models

# Tune I/O scheduler
echo mq-deadline > /sys/block/sda/queue/scheduler
```

**Application Tuning:**

```bash
# Increase worker processes
export WORKERS=8

# Optimize database connections
export DB_POOL_SIZE=20
export DB_MAX_OVERFLOW=30

# Enable caching
export ENABLE_REDIS_CACHE=true
export REDIS_URL=redis://localhost:6379/0
```

### 6. Configuration Issues

#### Symptoms

- Service fails to start
- Invalid configuration errors
- Environment variable issues
- Manifest parsing errors

#### Diagnostic Steps

**Validate Configuration:**

```bash
# Check environment variables
env | grep -E "(MODELS_ROOT|HF_TOKEN|AWS_)"

# Validate manifest
wan models validate-manifest

# Test configuration
wan models test-config
```

**Check File Permissions:**

```bash
# Models directory permissions
ls -la $MODELS_ROOT
stat $MODELS_ROOT

# Configuration file permissions
ls -la /opt/wan22/.env
ls -la /opt/wan22/config/
```

#### Solutions

**Fix Environment Variables:**

```bash
# Set required variables
export MODELS_ROOT=/data/models
export WAN_MODELS_MANIFEST=/opt/wan22/config/models.toml

# Validate paths exist
mkdir -p $MODELS_ROOT
touch $WAN_MODELS_MANIFEST
```

**Fix Permissions:**

```bash
# Fix ownership
sudo chown -R wan22:wan22 /data/models
sudo chown -R wan22:wan22 /opt/wan22

# Fix permissions
sudo chmod -R 755 /data/models
sudo chmod 600 /opt/wan22/.env
```

**Validate Manifest:**

```bash
# Check TOML syntax
python -c "import toml; toml.load('$WAN_MODELS_MANIFEST')"

# Validate schema
wan models validate-manifest --strict
```

### 7. Database Issues

#### Symptoms

- Database connection errors
- Slow queries
- Lock timeouts
- Corruption errors

#### Diagnostic Steps

**Check Database Status:**

```bash
# PostgreSQL status
sudo systemctl status postgresql
sudo -u postgres psql -c "\l"

# Connection testing
psql $DATABASE_URL -c "SELECT 1;"

# Check connections
sudo -u postgres psql -c "SELECT * FROM pg_stat_activity;"
```

**Query Performance:**

```bash
# Slow queries
sudo -u postgres psql -c "SELECT query, mean_time, calls FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 10;"

# Lock analysis
sudo -u postgres psql -c "SELECT * FROM pg_locks WHERE NOT granted;"
```

#### Solutions

**Connection Issues:**

```bash
# Restart PostgreSQL
sudo systemctl restart postgresql

# Increase connection limits
echo "max_connections = 200" >> /etc/postgresql/13/main/postgresql.conf

# Tune connection pooling
export DB_POOL_SIZE=20
export DB_MAX_OVERFLOW=30
```

**Performance Tuning:**

```bash
# Optimize PostgreSQL settings
echo "shared_buffers = 256MB" >> /etc/postgresql/13/main/postgresql.conf
echo "effective_cache_size = 1GB" >> /etc/postgresql/13/main/postgresql.conf
echo "work_mem = 4MB" >> /etc/postgresql/13/main/postgresql.conf

# Restart to apply changes
sudo systemctl restart postgresql
```

**Maintenance:**

```bash
# Vacuum and analyze
sudo -u postgres psql wan22 -c "VACUUM ANALYZE;"

# Reindex if needed
sudo -u postgres psql wan22 -c "REINDEX DATABASE wan22;"
```

## Platform-Specific Issues

### Windows Issues

#### Long Path Problems

```cmd
REM Enable long paths via registry
reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled /t REG_DWORD /d 1

REM Or via Group Policy
gpedit.msc
REM Navigate to: Computer Configuration > Administrative Templates > System > Filesystem
REM Enable "Enable Win32 long paths"
```

#### Permission Issues

```cmd
REM Fix directory permissions
icacls C:\data\models /grant wan22:(OI)(CI)F /T

REM Run as administrator if needed
runas /user:Administrator "wan models ensure --all"
```

#### Antivirus Interference

```cmd
REM Add exclusions to Windows Defender
powershell Add-MpPreference -ExclusionPath "C:\data\models"
powershell Add-MpPreference -ExclusionProcess "python.exe"
```

### Linux Issues

#### SELinux Problems

```bash
# Check SELinux status
getenforce

# Set permissive mode temporarily
sudo setenforce 0

# Create custom policy
sudo setsebool -P httpd_can_network_connect 1
sudo setsebool -P httpd_execmem 1
```

#### Systemd Service Issues

```bash
# Check service logs
journalctl -u wan22 -f

# Reload systemd configuration
sudo systemctl daemon-reload

# Check service file syntax
systemd-analyze verify /etc/systemd/system/wan22.service
```

### macOS Issues

#### Gatekeeper Problems

```bash
# Allow unsigned binaries
sudo spctl --master-disable

# Add specific exceptions
sudo spctl --add /path/to/wan22
```

#### File System Case Sensitivity

```bash
# Check case sensitivity
touch test TEST
ls -la test*  # Should show both files on case-sensitive FS

# Create case-sensitive volume if needed
hdiutil create -size 100g -fs "Case-sensitive HFS+" -volname "Models" models.dmg
```

## Emergency Procedures

### Service Recovery

#### Complete Service Restart

```bash
#!/bin/bash
# emergency-restart.sh

echo "Starting emergency service restart..."

# Stop all related services
sudo systemctl stop wan22
sudo systemctl stop nginx  # If using reverse proxy
sudo systemctl stop postgresql

# Clear temporary files
sudo rm -rf /data/models/.tmp/*
sudo rm -rf /data/models/.locks/*

# Check and repair filesystem if needed
sudo fsck /dev/sdb1  # Adjust device as needed

# Start services in order
sudo systemctl start postgresql
sleep 10
sudo systemctl start wan22
sleep 5
sudo systemctl start nginx

# Verify services
sudo systemctl is-active wan22 postgresql nginx

# Test functionality
curl -f http://localhost:8000/health || echo "Health check failed"

echo "Emergency restart completed"
```

#### Database Recovery

```bash
#!/bin/bash
# emergency-db-recovery.sh

echo "Starting database recovery..."

# Stop application
sudo systemctl stop wan22

# Backup current database
sudo -u postgres pg_dump wan22 > /tmp/wan22_emergency_backup.sql

# Check database integrity
sudo -u postgres psql wan22 -c "SELECT pg_database_size('wan22');"

# Repair if needed
sudo -u postgres vacuumdb --analyze --verbose wan22

# Restart services
sudo systemctl start wan22

echo "Database recovery completed"
```

### Data Recovery

#### Model Data Recovery

```bash
#!/bin/bash
# recover-models.sh

BACKUP_DIR="/backup/wan22"
MODELS_ROOT="/data/models"

echo "Starting model data recovery..."

# Find latest backup
LATEST_BACKUP=$(ls -t $BACKUP_DIR/models_*.tar.gz | head -1)

if [ -z "$LATEST_BACKUP" ]; then
    echo "No backup found!"
    exit 1
fi

echo "Restoring from: $LATEST_BACKUP"

# Stop service
sudo systemctl stop wan22

# Backup current state
sudo mv $MODELS_ROOT $MODELS_ROOT.corrupted.$(date +%s)

# Restore from backup
sudo mkdir -p $MODELS_ROOT
sudo tar -xzf "$LATEST_BACKUP" -C $(dirname $MODELS_ROOT)

# Fix permissions
sudo chown -R wan22:wan22 $MODELS_ROOT
sudo chmod -R 755 $MODELS_ROOT

# Start service
sudo systemctl start wan22

# Verify recovery
sleep 10
curl -f http://localhost:8000/health

echo "Model data recovery completed"
```

## Monitoring and Alerting

### Health Check Scripts

#### Comprehensive Health Check

```bash
#!/bin/bash
# health-check-comprehensive.sh

EXIT_CODE=0

echo "=== WAN2.2 Model Orchestrator Health Check ==="
echo "Timestamp: $(date)"

# Check service status
echo "Checking service status..."
if ! systemctl is-active --quiet wan22; then
    echo "ERROR: Service is not running"
    EXIT_CODE=1
else
    echo "OK: Service is running"
fi

# Check HTTP endpoint
echo "Checking HTTP endpoint..."
if ! curl -f -s http://localhost:8000/health > /dev/null; then
    echo "ERROR: Health endpoint not responding"
    EXIT_CODE=1
else
    echo "OK: Health endpoint responding"
fi

# Check disk space
echo "Checking disk space..."
DISK_USAGE=$(df $MODELS_ROOT | tail -1 | awk '{print $5}' | sed 's/%//')
if [ $DISK_USAGE -gt 90 ]; then
    echo "WARNING: Disk usage is ${DISK_USAGE}%"
    EXIT_CODE=1
else
    echo "OK: Disk usage is ${DISK_USAGE}%"
fi

# Check memory usage
echo "Checking memory usage..."
MEM_USAGE=$(free | grep Mem | awk '{printf "%.0f", $3/$2 * 100.0}')
if [ $MEM_USAGE -gt 90 ]; then
    echo "WARNING: Memory usage is ${MEM_USAGE}%"
    EXIT_CODE=1
else
    echo "OK: Memory usage is ${MEM_USAGE}%"
fi

# Check database connectivity
echo "Checking database connectivity..."
if ! psql $DATABASE_URL -c "SELECT 1;" > /dev/null 2>&1; then
    echo "ERROR: Cannot connect to database"
    EXIT_CODE=1
else
    echo "OK: Database connection successful"
fi

# Check model status
echo "Checking model status..."
FAILED_MODELS=$(wan models status --json | jq -r '.models[] | select(.state == "FAILED") | .model_id' | wc -l)
if [ $FAILED_MODELS -gt 0 ]; then
    echo "WARNING: $FAILED_MODELS models in failed state"
    EXIT_CODE=1
else
    echo "OK: No failed models"
fi

echo "=== Health Check Complete ==="
exit $EXIT_CODE
```

### Alerting Configuration

#### Prometheus Alerts

```yaml
# alerts.yml
groups:
  - name: wan22-orchestrator
    rules:
      - alert: ServiceDown
        expr: up{job="wan22-orchestrator"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "WAN2.2 Orchestrator service is down"
          description: "The WAN2.2 Model Orchestrator service has been down for more than 1 minute"

      - alert: HighErrorRate
        expr: rate(model_errors_total[5m]) > 0.1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High error rate in model operations"
          description: "Error rate is {{ $value }} errors per second"

      - alert: DiskSpaceHigh
        expr: (model_storage_bytes_used / model_storage_bytes_total) > 0.9
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Model storage disk space is running low"
          description: "Disk usage is {{ $value | humanizePercentage }}"

      - alert: DownloadTimeout
        expr: increase(model_download_timeouts_total[10m]) > 5
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "Multiple download timeouts detected"
          description: "{{ $value }} download timeouts in the last 10 minutes"
```

#### Nagios Checks

```bash
#!/bin/bash
# check_wan22_orchestrator.sh

# Nagios plugin for WAN2.2 Orchestrator
# Usage: check_wan22_orchestrator.sh -H hostname -p port

while getopts "H:p:" opt; do
    case $opt in
        H) HOST=$OPTARG ;;
        p) PORT=$OPTARG ;;
    esac
done

HOST=${HOST:-localhost}
PORT=${PORT:-8000}

# Check health endpoint
RESPONSE=$(curl -s -w "%{http_code}" http://$HOST:$PORT/health)
HTTP_CODE=${RESPONSE: -3}
BODY=${RESPONSE%???}

if [ "$HTTP_CODE" = "200" ]; then
    echo "OK - Service is healthy | response_time=0.123s"
    exit 0
elif [ "$HTTP_CODE" = "503" ]; then
    echo "WARNING - Service degraded: $BODY"
    exit 1
else
    echo "CRITICAL - Service unavailable (HTTP $HTTP_CODE)"
    exit 2
fi
```

## Performance Optimization

### Profiling and Analysis

#### CPU Profiling

```bash
# Profile CPU usage
py-spy top --pid $(pgrep -f wan22)

# Generate flame graph
py-spy record -o profile.svg --pid $(pgrep -f wan22) --duration 60

# Profile specific function
python -m cProfile -o profile.stats -m backend.main
python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"
```

#### Memory Profiling

```bash
# Memory usage over time
memory_profiler python -m backend.main

# Detailed memory analysis
valgrind --tool=massif python -m backend.main
ms_print massif.out.* > memory_profile.txt
```

#### I/O Profiling

```bash
# Monitor file I/O
strace -e trace=file -p $(pgrep -f wan22)

# Monitor network I/O
tcpdump -i any -w network_trace.pcap host huggingface.co

# Analyze I/O patterns
iotop -a -o -d 1
```

### Optimization Recommendations

#### System Level

```bash
# CPU governor
echo performance > /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# I/O scheduler
echo mq-deadline > /sys/block/sda/queue/scheduler

# Network tuning
echo 'net.core.rmem_max = 134217728' >> /etc/sysctl.conf
echo 'net.core.wmem_max = 134217728' >> /etc/sysctl.conf
echo 'net.ipv4.tcp_congestion_control = bbr' >> /etc/sysctl.conf
sysctl -p
```

#### Application Level

```bash
# Optimize Python
export PYTHONOPTIMIZE=2
export PYTHONDONTWRITEBYTECODE=1

# Tune garbage collection
export PYTHONHASHSEED=0
export MALLOC_TRIM_THRESHOLD_=100000

# Database optimization
export DB_POOL_SIZE=20
export DB_POOL_RECYCLE=3600
export DB_POOL_PRE_PING=true
```

This troubleshooting guide provides comprehensive coverage of common issues and their solutions, along with emergency procedures and monitoring recommendations. It should help operators quickly diagnose and resolve problems with the Model Orchestrator system.
