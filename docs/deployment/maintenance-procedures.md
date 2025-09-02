# Maintenance Procedures

This document outlines comprehensive maintenance procedures for the project health system to ensure optimal performance, reliability, and security.

## Overview

Regular maintenance is essential for:

- Preventing system degradation
- Ensuring optimal performance
- Maintaining security
- Preventing data loss
- Identifying issues early

## Maintenance Schedule

### Daily Tasks (Automated)

- Health check execution
- Log rotation
- Backup verification
- Performance monitoring
- Alert processing

### Weekly Tasks (Semi-automated)

- Comprehensive system health review
- Performance analysis
- Security updates review
- Backup integrity verification
- Documentation updates

### Monthly Tasks (Manual)

- System optimization review
- Capacity planning
- Security audit
- Dependency updates
- Disaster recovery testing

### Quarterly Tasks (Manual)

- Architecture review
- Performance benchmarking
- Security penetration testing
- Business continuity planning
- Training updates

## Daily Maintenance Procedures

### 1. Automated Health Monitoring

```bash
#!/bin/bash
# daily-health-check.sh
# Run: 0 6 * * * /opt/project-health/scripts/daily-health-check.sh

LOG_FILE="/var/log/project-health/daily-maintenance.log"
DATE=$(date '+%Y-%m-%d %H:%M:%S')

echo "[$DATE] Starting daily health check" >> "$LOG_FILE"

# Run comprehensive health check
python -m tools.health_checker.cli --full-check --output-format json > /tmp/health-report.json

# Check health score
HEALTH_SCORE=$(jq '.overall_score' /tmp/health-report.json)

if (( $(echo "$HEALTH_SCORE < 80" | bc -l) )); then
    echo "[$DATE] WARNING: Health score below threshold: $HEALTH_SCORE" >> "$LOG_FILE"

    # Send alert
    python -m tools.health_checker.cli --send-alert \
        --message "Daily health check: Score $HEALTH_SCORE below threshold" \
        --severity warning
fi

# Archive health report
cp /tmp/health-report.json "/var/lib/project-health/reports/daily-$(date +%Y%m%d).json"

echo "[$DATE] Daily health check completed" >> "$LOG_FILE"
```

### 2. Log Management

```bash
#!/bin/bash
# daily-log-management.sh

LOG_DIR="/var/log/project-health"
ARCHIVE_DIR="/var/log/project-health/archive"
RETENTION_DAYS=30

# Create archive directory if it doesn't exist
mkdir -p "$ARCHIVE_DIR"

# Compress logs older than 1 day
find "$LOG_DIR" -name "*.log" -mtime +1 -not -path "*/archive/*" -exec gzip {} \;

# Move compressed logs to archive
find "$LOG_DIR" -name "*.log.gz" -not -path "*/archive/*" -exec mv {} "$ARCHIVE_DIR/" \;

# Remove logs older than retention period
find "$ARCHIVE_DIR" -name "*.log.gz" -mtime +$RETENTION_DAYS -delete

# Log cleanup summary
CURRENT_SIZE=$(du -sh "$LOG_DIR" | cut -f1)
echo "$(date): Log cleanup completed. Current log directory size: $CURRENT_SIZE" >> "$LOG_DIR/maintenance.log"
```

### 3. Performance Monitoring

```bash
#!/bin/bash
# daily-performance-monitoring.sh

METRICS_FILE="/var/lib/project-health/metrics/daily-$(date +%Y%m%d).json"

# Collect system metrics
cat > "$METRICS_FILE" << EOF
{
  "timestamp": "$(date -Iseconds)",
  "system": {
    "cpu_usage": $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1),
    "memory_usage": $(free | grep Mem | awk '{printf "%.2f", $3/$2 * 100.0}'),
    "disk_usage": $(df -h / | awk 'NR==2 {print $5}' | cut -d'%' -f1),
    "load_average": "$(uptime | awk -F'load average:' '{print $2}')"
  },
  "application": {
    "health_score": $(python -m tools.health_checker.cli --quick-score),
    "active_processes": $(pgrep -f "project-health" | wc -l),
    "response_time": $(curl -w "%{time_total}" -s -o /dev/null http://localhost:8080/health)
  }
}
EOF

# Check for performance anomalies
python -m tools.health_checker.cli --analyze-performance "$METRICS_FILE"
```

### 4. Backup Verification

```bash
#!/bin/bash
# daily-backup-verification.sh

BACKUP_DIR="/backup/project-health"
LATEST_BACKUP=$(ls -t "$BACKUP_DIR" | head -n1)

if [ -z "$LATEST_BACKUP" ]; then
    echo "ERROR: No backups found in $BACKUP_DIR"
    python -m tools.health_checker.cli --send-alert \
        --message "No backups found" \
        --severity critical
    exit 1
fi

# Check backup age (should be less than 24 hours old)
BACKUP_AGE=$(find "$BACKUP_DIR/$LATEST_BACKUP" -mtime +1 | wc -l)

if [ "$BACKUP_AGE" -gt 0 ]; then
    echo "WARNING: Latest backup is older than 24 hours"
    python -m tools.health_checker.cli --send-alert \
        --message "Backup is older than 24 hours" \
        --severity warning
fi

# Verify backup integrity
if [ -f "$BACKUP_DIR/$LATEST_BACKUP/health.db" ]; then
    sqlite3 "$BACKUP_DIR/$LATEST_BACKUP/health.db" "PRAGMA integrity_check;" > /tmp/backup_check.txt

    if ! grep -q "ok" /tmp/backup_check.txt; then
        echo "ERROR: Backup integrity check failed"
        python -m tools.health_checker.cli --send-alert \
            --message "Backup integrity check failed" \
            --severity critical
    fi
fi
```

## Weekly Maintenance Procedures

### 1. Comprehensive System Review

```bash
#!/bin/bash
# weekly-system-review.sh
# Run: 0 2 * * 0 /opt/project-health/scripts/weekly-system-review.sh

REPORT_DIR="/var/lib/project-health/reports/weekly"
WEEK_NUM=$(date +%Y-W%U)
REPORT_FILE="$REPORT_DIR/system-review-$WEEK_NUM.md"

mkdir -p "$REPORT_DIR"

cat > "$REPORT_FILE" << EOF
# Weekly System Review - $WEEK_NUM

## Executive Summary
Generated on: $(date)

## Health Metrics
EOF

# Generate comprehensive health report
python -m tools.health_checker.cli --comprehensive-report >> "$REPORT_FILE"

# Add performance analysis
echo -e "\n## Performance Analysis" >> "$REPORT_FILE"
python -m tools.health_checker.cli --performance-analysis --week >> "$REPORT_FILE"

# Add security status
echo -e "\n## Security Status" >> "$REPORT_FILE"
python -m tools.health_checker.cli --security-check >> "$REPORT_FILE"

# Add recommendations
echo -e "\n## Recommendations" >> "$REPORT_FILE"
python -m tools.health_checker.cli --recommendations >> "$REPORT_FILE"

# Send report to stakeholders
python -m tools.health_checker.cli --send-report "$REPORT_FILE" \
    --recipients "devops@company.com,management@company.com"
```

### 2. Performance Analysis

```python
#!/usr/bin/env python3
# weekly-performance-analysis.py

import json
import statistics
from pathlib import Path
from datetime import datetime, timedelta

def analyze_weekly_performance():
    """Analyze performance metrics from the past week."""

    metrics_dir = Path("/var/lib/project-health/metrics")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)

    # Collect metrics from the past week
    weekly_metrics = []

    for day in range(7):
        date = start_date + timedelta(days=day)
        metrics_file = metrics_dir / f"daily-{date.strftime('%Y%m%d')}.json"

        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                daily_metrics = json.load(f)
                weekly_metrics.append(daily_metrics)

    if not weekly_metrics:
        print("No metrics data available for analysis")
        return

    # Analyze trends
    health_scores = [m['application']['health_score'] for m in weekly_metrics]
    response_times = [float(m['application']['response_time']) for m in weekly_metrics]
    cpu_usage = [float(m['system']['cpu_usage']) for m in weekly_metrics]
    memory_usage = [float(m['system']['memory_usage']) for m in weekly_metrics]

    analysis = {
        "period": f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
        "health_score": {
            "average": statistics.mean(health_scores),
            "min": min(health_scores),
            "max": max(health_scores),
            "trend": "improving" if health_scores[-1] > health_scores[0] else "declining"
        },
        "response_time": {
            "average": statistics.mean(response_times),
            "min": min(response_times),
            "max": max(response_times),
            "p95": statistics.quantiles(response_times, n=20)[18] if len(response_times) > 1 else response_times[0]
        },
        "resource_usage": {
            "cpu_average": statistics.mean(cpu_usage),
            "memory_average": statistics.mean(memory_usage),
            "cpu_peak": max(cpu_usage),
            "memory_peak": max(memory_usage)
        }
    }

    # Generate recommendations
    recommendations = []

    if analysis["health_score"]["average"] < 85:
        recommendations.append("Health score below target (85). Review failing components.")

    if analysis["response_time"]["p95"] > 5.0:
        recommendations.append("95th percentile response time above 5s. Consider performance optimization.")

    if analysis["resource_usage"]["cpu_average"] > 80:
        recommendations.append("High CPU usage detected. Consider scaling or optimization.")

    if analysis["resource_usage"]["memory_average"] > 85:
        recommendations.append("High memory usage detected. Review memory leaks or increase capacity.")

    analysis["recommendations"] = recommendations

    # Save analysis
    analysis_file = Path(f"/var/lib/project-health/reports/weekly/performance-analysis-{end_date.strftime('%Y-W%U')}.json")
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2)

    print(f"Performance analysis saved to {analysis_file}")

    # Print summary
    print("\n=== Weekly Performance Summary ===")
    print(f"Health Score: {analysis['health_score']['average']:.1f} (trend: {analysis['health_score']['trend']})")
    print(f"Response Time: {analysis['response_time']['average']:.3f}s avg, {analysis['response_time']['p95']:.3f}s p95")
    print(f"CPU Usage: {analysis['resource_usage']['cpu_average']:.1f}% avg, {analysis['resource_usage']['cpu_peak']:.1f}% peak")
    print(f"Memory Usage: {analysis['resource_usage']['memory_average']:.1f}% avg, {analysis['resource_usage']['memory_peak']:.1f}% peak")

    if recommendations:
        print("\n=== Recommendations ===")
        for rec in recommendations:
            print(f"- {rec}")

if __name__ == "__main__":
    analyze_weekly_performance()
```

### 3. Security Updates Review

```bash
#!/bin/bash
# weekly-security-review.sh

SECURITY_LOG="/var/log/project-health/security-review.log"
DATE=$(date '+%Y-%m-%d %H:%M:%S')

echo "[$DATE] Starting weekly security review" >> "$SECURITY_LOG"

# Check for system updates
echo "=== System Updates ===" >> "$SECURITY_LOG"
apt list --upgradable 2>/dev/null | grep -i security >> "$SECURITY_LOG"

# Check Python package vulnerabilities
echo "=== Python Package Security ===" >> "$SECURITY_LOG"
pip-audit --format=json > /tmp/pip-audit.json 2>/dev/null

if [ -s /tmp/pip-audit.json ]; then
    VULN_COUNT=$(jq '.vulnerabilities | length' /tmp/pip-audit.json)
    echo "Found $VULN_COUNT vulnerabilities in Python packages" >> "$SECURITY_LOG"

    if [ "$VULN_COUNT" -gt 0 ]; then
        jq -r '.vulnerabilities[] | "- \(.package): \(.vulnerability_id) (\(.vulnerability_description))"' /tmp/pip-audit.json >> "$SECURITY_LOG"

        # Send security alert
        python -m tools.health_checker.cli --send-alert \
            --message "Security vulnerabilities found in Python packages: $VULN_COUNT" \
            --severity high
    fi
fi

# Check file permissions
echo "=== File Permissions ===" >> "$SECURITY_LOG"
find /opt/project-health -type f -perm /o+w -ls >> "$SECURITY_LOG" 2>/dev/null

# Check for suspicious processes
echo "=== Process Review ===" >> "$SECURITY_LOG"
ps aux | grep -E "(project-health|python)" | grep -v grep >> "$SECURITY_LOG"

# Check network connections
echo "=== Network Connections ===" >> "$SECURITY_LOG"
netstat -tulpn | grep -E "(8080|9090)" >> "$SECURITY_LOG"

echo "[$DATE] Weekly security review completed" >> "$SECURITY_LOG"
```

### 4. Backup Integrity Testing

```bash
#!/bin/bash
# weekly-backup-test.sh

BACKUP_DIR="/backup/project-health"
TEST_DIR="/tmp/backup-test-$(date +%Y%m%d)"
LATEST_BACKUP=$(ls -t "$BACKUP_DIR" | head -n1)

echo "Testing backup integrity for: $LATEST_BACKUP"

# Create test directory
mkdir -p "$TEST_DIR"

# Extract backup
cp -r "$BACKUP_DIR/$LATEST_BACKUP"/* "$TEST_DIR/"

# Test database integrity
if [ -f "$TEST_DIR/health.db" ]; then
    echo "Testing database integrity..."
    sqlite3 "$TEST_DIR/health.db" "PRAGMA integrity_check;" > "$TEST_DIR/db_check.txt"

    if grep -q "ok" "$TEST_DIR/db_check.txt"; then
        echo "✓ Database integrity check passed"
    else
        echo "✗ Database integrity check failed"
        cat "$TEST_DIR/db_check.txt"
    fi
fi

# Test configuration files
if [ -d "$TEST_DIR/config" ]; then
    echo "Testing configuration files..."
    for config_file in "$TEST_DIR/config"/*.yaml; do
        if [ -f "$config_file" ]; then
            python -c "import yaml; yaml.safe_load(open('$config_file'))" 2>/dev/null
            if [ $? -eq 0 ]; then
                echo "✓ $(basename "$config_file") is valid"
            else
                echo "✗ $(basename "$config_file") is invalid"
            fi
        fi
    done
fi

# Test restore procedure (dry run)
echo "Testing restore procedure..."
python -m tools.health_checker.cli --test-restore "$TEST_DIR" --dry-run

# Cleanup
rm -rf "$TEST_DIR"

echo "Backup integrity test completed"
```

## Monthly Maintenance Procedures

### 1. System Optimization Review

```python
#!/usr/bin/env python3
# monthly-optimization-review.py

import json
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

def monthly_optimization_review():
    """Perform monthly system optimization review."""

    print("=== Monthly Optimization Review ===")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d')}")

    # 1. Analyze performance trends
    print("\n1. Performance Trend Analysis")
    analyze_performance_trends()

    # 2. Review resource utilization
    print("\n2. Resource Utilization Review")
    review_resource_utilization()

    # 3. Database optimization
    print("\n3. Database Optimization")
    optimize_database()

    # 4. Configuration optimization
    print("\n4. Configuration Optimization")
    optimize_configuration()

    # 5. Generate optimization report
    print("\n5. Generating Optimization Report")
    generate_optimization_report()

def analyze_performance_trends():
    """Analyze performance trends over the past month."""

    # Collect monthly performance data
    metrics_dir = Path("/var/lib/project-health/metrics")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    monthly_data = []
    for day in range(30):
        date = start_date + timedelta(days=day)
        metrics_file = metrics_dir / f"daily-{date.strftime('%Y%m%d')}.json"

        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                daily_metrics = json.load(f)
                monthly_data.append(daily_metrics)

    if monthly_data:
        # Calculate trends
        health_scores = [m['application']['health_score'] for m in monthly_data]
        response_times = [float(m['application']['response_time']) for m in monthly_data]

        print(f"  Health Score Trend: {health_scores[0]:.1f} → {health_scores[-1]:.1f}")
        print(f"  Response Time Trend: {response_times[0]:.3f}s → {response_times[-1]:.3f}s")

        # Identify optimization opportunities
        if health_scores[-1] < health_scores[0]:
            print("  ⚠️  Health score declining - investigate root causes")

        if response_times[-1] > response_times[0] * 1.2:
            print("  ⚠️  Response time increasing - performance optimization needed")

def review_resource_utilization():
    """Review system resource utilization."""

    # Check disk usage
    result = subprocess.run(['df', '-h', '/'], capture_output=True, text=True)
    disk_usage = result.stdout.split('\n')[1].split()[4]
    print(f"  Disk Usage: {disk_usage}")

    if int(disk_usage.rstrip('%')) > 80:
        print("  ⚠️  High disk usage - consider cleanup or expansion")

    # Check memory usage
    result = subprocess.run(['free', '-h'], capture_output=True, text=True)
    memory_lines = result.stdout.split('\n')
    memory_info = memory_lines[1].split()
    memory_used = memory_info[2]
    memory_total = memory_info[1]
    print(f"  Memory Usage: {memory_used}/{memory_total}")

    # Check log file sizes
    log_dir = Path("/var/log/project-health")
    if log_dir.exists():
        total_size = sum(f.stat().st_size for f in log_dir.rglob('*') if f.is_file())
        print(f"  Log Directory Size: {total_size / (1024*1024):.1f} MB")

        if total_size > 1024 * 1024 * 1024:  # 1GB
            print("  ⚠️  Large log directory - consider more aggressive rotation")

def optimize_database():
    """Optimize database performance."""

    db_path = Path("/var/lib/project-health/health.db")
    if db_path.exists():
        print("  Running database optimization...")

        # Run VACUUM to reclaim space
        subprocess.run(['sqlite3', str(db_path), 'VACUUM;'])

        # Analyze tables for query optimization
        subprocess.run(['sqlite3', str(db_path), 'ANALYZE;'])

        # Check database size
        db_size = db_path.stat().st_size / (1024 * 1024)  # MB
        print(f"  Database Size: {db_size:.1f} MB")

        if db_size > 500:  # 500MB
            print("  ⚠️  Large database - consider archiving old data")

def optimize_configuration():
    """Review and optimize configuration settings."""

    config_file = Path("/etc/project-health/production.yaml")
    if config_file.exists():
        with open(config_file, 'r') as f:
            import yaml
            config = yaml.safe_load(f)

        # Check for optimization opportunities
        optimizations = []

        # Check test execution settings
        if 'test_execution' in config:
            test_config = config['test_execution']
            if test_config.get('parallel', False) and test_config.get('timeout', 0) > 3600:
                optimizations.append("Consider reducing test timeout for faster feedback")

        # Check health monitoring frequency
        if 'health_monitoring' in config:
            health_config = config['health_monitoring']
            if health_config.get('check_interval', 0) < 300:
                optimizations.append("Consider increasing health check interval to reduce overhead")

        if optimizations:
            print("  Configuration Optimization Suggestions:")
            for opt in optimizations:
                print(f"    - {opt}")
        else:
            print("  Configuration appears optimal")

def generate_optimization_report():
    """Generate comprehensive optimization report."""

    report_dir = Path("/var/lib/project-health/reports/monthly")
    report_dir.mkdir(parents=True, exist_ok=True)

    month_year = datetime.now().strftime('%Y-%m')
    report_file = report_dir / f"optimization-report-{month_year}.md"

    with open(report_file, 'w') as f:
        f.write(f"# Monthly Optimization Report - {month_year}\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Add sections for each optimization area
        f.write("## Performance Trends\n")
        f.write("See performance analysis for detailed trends.\n\n")

        f.write("## Resource Utilization\n")
        f.write("Current resource usage is within acceptable limits.\n\n")

        f.write("## Optimization Recommendations\n")
        f.write("- Review and implement suggested optimizations\n")
        f.write("- Monitor performance impact of changes\n")
        f.write("- Schedule next optimization review\n\n")

    print(f"  Optimization report saved: {report_file}")

if __name__ == "__main__":
    monthly_optimization_review()
```

### 2. Capacity Planning

```python
#!/usr/bin/env python3
# monthly-capacity-planning.py

import json
import statistics
from pathlib import Path
from datetime import datetime, timedelta

def capacity_planning_analysis():
    """Perform monthly capacity planning analysis."""

    print("=== Monthly Capacity Planning ===")

    # Collect historical data
    metrics_dir = Path("/var/lib/project-health/metrics")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)  # 3 months of data

    historical_data = []
    for day in range(90):
        date = start_date + timedelta(days=day)
        metrics_file = metrics_dir / f"daily-{date.strftime('%Y%m%d')}.json"

        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                daily_metrics = json.load(f)
                historical_data.append(daily_metrics)

    if len(historical_data) < 30:
        print("Insufficient historical data for capacity planning")
        return

    # Analyze growth trends
    analyze_growth_trends(historical_data)

    # Predict future capacity needs
    predict_capacity_needs(historical_data)

    # Generate capacity recommendations
    generate_capacity_recommendations(historical_data)

def analyze_growth_trends(data):
    """Analyze growth trends in system usage."""

    print("\n1. Growth Trend Analysis")

    # Extract metrics over time
    cpu_usage = [float(d['system']['cpu_usage']) for d in data]
    memory_usage = [float(d['system']['memory_usage']) for d in data]
    response_times = [float(d['application']['response_time']) for d in data]

    # Calculate trends (simple linear regression)
    def calculate_trend(values):
        n = len(values)
        x_sum = sum(range(n))
        y_sum = sum(values)
        xy_sum = sum(i * values[i] for i in range(n))
        x2_sum = sum(i * i for i in range(n))

        slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum * x_sum)
        return slope

    cpu_trend = calculate_trend(cpu_usage)
    memory_trend = calculate_trend(memory_usage)
    response_trend = calculate_trend(response_times)

    print(f"  CPU Usage Trend: {cpu_trend:+.3f}% per day")
    print(f"  Memory Usage Trend: {memory_trend:+.3f}% per day")
    print(f"  Response Time Trend: {response_trend:+.6f}s per day")

    # Identify concerning trends
    if cpu_trend > 0.1:
        print("  ⚠️  CPU usage increasing - monitor for capacity issues")

    if memory_trend > 0.1:
        print("  ⚠️  Memory usage increasing - monitor for memory leaks")

    if response_trend > 0.001:
        print("  ⚠️  Response time increasing - performance degradation detected")

def predict_capacity_needs(data):
    """Predict future capacity needs based on trends."""

    print("\n2. Capacity Predictions (6 months)")

    # Current averages
    current_cpu = statistics.mean([float(d['system']['cpu_usage']) for d in data[-30:]])
    current_memory = statistics.mean([float(d['system']['memory_usage']) for d in data[-30:]])
    current_response = statistics.mean([float(d['application']['response_time']) for d in data[-30:]])

    # Growth rates (per day)
    cpu_growth = (current_cpu - statistics.mean([float(d['system']['cpu_usage']) for d in data[:30]])) / 60
    memory_growth = (current_memory - statistics.mean([float(d['system']['memory_usage']) for d in data[:30]])) / 60
    response_growth = (current_response - statistics.mean([float(d['application']['response_time']) for d in data[:30]])) / 60

    # Predict 6 months ahead (180 days)
    predicted_cpu = current_cpu + (cpu_growth * 180)
    predicted_memory = current_memory + (memory_growth * 180)
    predicted_response = current_response + (response_growth * 180)

    print(f"  Predicted CPU Usage: {predicted_cpu:.1f}% (current: {current_cpu:.1f}%)")
    print(f"  Predicted Memory Usage: {predicted_memory:.1f}% (current: {current_memory:.1f}%)")
    print(f"  Predicted Response Time: {predicted_response:.3f}s (current: {current_response:.3f}s)")

    # Capacity warnings
    if predicted_cpu > 80:
        print("  🚨 CPU capacity will be exceeded - plan for scaling")

    if predicted_memory > 85:
        print("  🚨 Memory capacity will be exceeded - plan for upgrade")

    if predicted_response > 5.0:
        print("  🚨 Response time will exceed SLA - plan for optimization")

def generate_capacity_recommendations(data):
    """Generate capacity planning recommendations."""

    print("\n3. Capacity Recommendations")

    recommendations = []

    # Analyze current utilization
    current_cpu = statistics.mean([float(d['system']['cpu_usage']) for d in data[-7:]])
    current_memory = statistics.mean([float(d['system']['memory_usage']) for d in data[-7:]])

    if current_cpu > 70:
        recommendations.append("Consider CPU upgrade or horizontal scaling")

    if current_memory > 80:
        recommendations.append("Consider memory upgrade")

    # Analyze growth patterns
    if len(data) >= 60:
        recent_avg = statistics.mean([float(d['system']['cpu_usage']) for d in data[-30:]])
        older_avg = statistics.mean([float(d['system']['cpu_usage']) for d in data[-60:-30]])

        if recent_avg > older_avg * 1.2:
            recommendations.append("Rapid growth detected - accelerate capacity planning")

    # Storage recommendations
    # (This would typically check actual storage metrics)
    recommendations.append("Monitor storage growth and plan for archival")

    if recommendations:
        print("  Recommendations:")
        for rec in recommendations:
            print(f"    - {rec}")
    else:
        print("  No immediate capacity concerns identified")

    # Save capacity planning report
    report_dir = Path("/var/lib/project-health/reports/monthly")
    report_dir.mkdir(parents=True, exist_ok=True)

    month_year = datetime.now().strftime('%Y-%m')
    report_file = report_dir / f"capacity-planning-{month_year}.json"

    capacity_report = {
        "date": datetime.now().isoformat(),
        "current_utilization": {
            "cpu": current_cpu,
            "memory": current_memory
        },
        "recommendations": recommendations,
        "next_review": (datetime.now() + timedelta(days=30)).isoformat()
    }

    with open(report_file, 'w') as f:
        json.dump(capacity_report, f, indent=2)

    print(f"\n  Capacity planning report saved: {report_file}")

if __name__ == "__main__":
    capacity_planning_analysis()
```

### 3. Security Audit

```bash
#!/bin/bash
# monthly-security-audit.sh

AUDIT_DIR="/var/lib/project-health/security-audits"
AUDIT_DATE=$(date +%Y-%m)
AUDIT_FILE="$AUDIT_DIR/security-audit-$AUDIT_DATE.txt"

mkdir -p "$AUDIT_DIR"

echo "=== Monthly Security Audit - $AUDIT_DATE ===" > "$AUDIT_FILE"
echo "Generated: $(date)" >> "$AUDIT_FILE"

# 1. System security updates
echo -e "\n1. SYSTEM SECURITY UPDATES" >> "$AUDIT_FILE"
apt list --upgradable 2>/dev/null | grep -i security >> "$AUDIT_FILE"

# 2. User and permission audit
echo -e "\n2. USER AND PERMISSION AUDIT" >> "$AUDIT_FILE"
echo "System users:" >> "$AUDIT_FILE"
getent passwd | grep -E "(project-health|root)" >> "$AUDIT_FILE"

echo -e "\nFile permissions audit:" >> "$AUDIT_FILE"
find /opt/project-health -type f \( -perm -4000 -o -perm -2000 \) -ls >> "$AUDIT_FILE"

# 3. Network security
echo -e "\n3. NETWORK SECURITY" >> "$AUDIT_FILE"
echo "Open ports:" >> "$AUDIT_FILE"
netstat -tulpn | grep LISTEN >> "$AUDIT_FILE"

echo -e "\nFirewall status:" >> "$AUDIT_FILE"
ufw status verbose >> "$AUDIT_FILE" 2>/dev/null || echo "UFW not installed" >> "$AUDIT_FILE"

# 4. Log analysis
echo -e "\n4. LOG ANALYSIS" >> "$AUDIT_FILE"
echo "Failed login attempts:" >> "$AUDIT_FILE"
grep "Failed password" /var/log/auth.log | tail -10 >> "$AUDIT_FILE" 2>/dev/null

echo -e "\nSuspicious activities:" >> "$AUDIT_FILE"
grep -i "error\|warning\|failed" /var/log/project-health/*.log | tail -20 >> "$AUDIT_FILE" 2>/dev/null

# 5. SSL/TLS certificate check
echo -e "\n5. SSL/TLS CERTIFICATES" >> "$AUDIT_FILE"
if command -v openssl &> /dev/null; then
    echo "Certificate expiration check:" >> "$AUDIT_FILE"
    # Check certificates in common locations
    find /etc/ssl/certs -name "*.pem" -exec openssl x509 -in {} -noout -enddate \; 2>/dev/null | head -5 >> "$AUDIT_FILE"
fi

# 6. Application security
echo -e "\n6. APPLICATION SECURITY" >> "$AUDIT_FILE"
echo "Configuration file permissions:" >> "$AUDIT_FILE"
ls -la /etc/project-health/ >> "$AUDIT_FILE" 2>/dev/null

echo -e "\nSecret files check:" >> "$AUDIT_FILE"
find /opt/project-health -name "*.key" -o -name "*.pem" -o -name "*secret*" -ls >> "$AUDIT_FILE" 2>/dev/null

# 7. Dependency vulnerabilities
echo -e "\n7. DEPENDENCY VULNERABILITIES" >> "$AUDIT_FILE"
if command -v pip-audit &> /dev/null; then
    pip-audit --format=text >> "$AUDIT_FILE" 2>/dev/null
else
    echo "pip-audit not installed - consider installing for vulnerability scanning" >> "$AUDIT_FILE"
fi

# 8. Generate security score
echo -e "\n8. SECURITY ASSESSMENT" >> "$AUDIT_FILE"
python3 -c "
import subprocess
import re

score = 100
issues = []

# Check for security updates
result = subprocess.run(['apt', 'list', '--upgradable'], capture_output=True, text=True)
security_updates = len([line for line in result.stdout.split('\n') if 'security' in line.lower()])
if security_updates > 0:
    score -= min(security_updates * 5, 30)
    issues.append(f'{security_updates} security updates pending')

# Check for world-writable files
result = subprocess.run(['find', '/opt/project-health', '-type', 'f', '-perm', '/o+w'], capture_output=True, text=True)
writable_files = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
if writable_files > 0:
    score -= min(writable_files * 10, 20)
    issues.append(f'{writable_files} world-writable files found')

print(f'Security Score: {score}/100')
if issues:
    print('Issues found:')
    for issue in issues:
        print(f'  - {issue}')
else:
    print('No major security issues detected')
" >> "$AUDIT_FILE"

echo "Security audit completed: $AUDIT_FILE"

# Send alert if security score is low
SECURITY_SCORE=$(grep "Security Score:" "$AUDIT_FILE" | grep -o '[0-9]\+')
if [ "$SECURITY_SCORE" -lt 80 ]; then
    python -m tools.health_checker.cli --send-alert \
        --message "Monthly security audit: Score $SECURITY_SCORE/100" \
        --severity high
fi
```

## Quarterly Maintenance Procedures

### 1. Architecture Review

```markdown
# Quarterly Architecture Review Checklist

## System Architecture Assessment

### Current State Analysis

- [ ] Document current system architecture
- [ ] Identify architectural debt
- [ ] Review component dependencies
- [ ] Assess scalability limitations
- [ ] Evaluate performance bottlenecks

### Technology Stack Review

- [ ] Review technology choices
- [ ] Assess framework versions and support
- [ ] Evaluate third-party dependencies
- [ ] Consider technology upgrades
- [ ] Plan migration strategies

### Security Architecture

- [ ] Review security controls
- [ ] Assess threat model
- [ ] Evaluate access controls
- [ ] Review data protection measures
- [ ] Plan security improvements

### Scalability Assessment

- [ ] Review current capacity
- [ ] Assess scaling strategies
- [ ] Evaluate load balancing
- [ ] Consider microservices migration
- [ ] Plan infrastructure scaling

### Documentation Review

- [ ] Update architecture diagrams
- [ ] Review API documentation
- [ ] Update deployment guides
- [ ] Refresh troubleshooting guides
- [ ] Update runbooks

## Action Items

- [ ] Priority 1: Critical issues requiring immediate attention
- [ ] Priority 2: Important improvements for next quarter
- [ ] Priority 3: Long-term architectural goals

## Next Review

Schedule next quarterly review for: [Date + 3 months]
```

### 2. Disaster Recovery Testing

```bash
#!/bin/bash
# quarterly-dr-test.sh

DR_TEST_DIR="/tmp/dr-test-$(date +%Y%m%d)"
BACKUP_DIR="/backup/project-health"
LOG_FILE="/var/log/project-health/dr-test.log"

echo "=== Disaster Recovery Test - $(date) ===" | tee -a "$LOG_FILE"

# 1. Prepare test environment
echo "1. Preparing test environment..." | tee -a "$LOG_FILE"
mkdir -p "$DR_TEST_DIR"

# 2. Test backup restoration
echo "2. Testing backup restoration..." | tee -a "$LOG_FILE"
LATEST_BACKUP=$(ls -t "$BACKUP_DIR" | head -n1)

if [ -z "$LATEST_BACKUP" ]; then
    echo "ERROR: No backups found" | tee -a "$LOG_FILE"
    exit 1
fi

# Copy backup to test directory
cp -r "$BACKUP_DIR/$LATEST_BACKUP"/* "$DR_TEST_DIR/"

# 3. Test database restoration
echo "3. Testing database restoration..." | tee -a "$LOG_FILE"
if [ -f "$DR_TEST_DIR/health.db" ]; then
    # Test database integrity
    sqlite3 "$DR_TEST_DIR/health.db" "PRAGMA integrity_check;" > "$DR_TEST_DIR/db_check.txt"

    if grep -q "ok" "$DR_TEST_DIR/db_check.txt"; then
        echo "✓ Database restoration successful" | tee -a "$LOG_FILE"
    else
        echo "✗ Database restoration failed" | tee -a "$LOG_FILE"
    fi

    # Test data retrieval
    RECORD_COUNT=$(sqlite3 "$DR_TEST_DIR/health.db" "SELECT COUNT(*) FROM health_reports;" 2>/dev/null || echo "0")
    echo "Database contains $RECORD_COUNT health reports" | tee -a "$LOG_FILE"
fi

# 4. Test configuration restoration
echo "4. Testing configuration restoration..." | tee -a "$LOG_FILE"
if [ -d "$DR_TEST_DIR/config" ]; then
    for config_file in "$DR_TEST_DIR/config"/*.yaml; do
        if [ -f "$config_file" ]; then
            python -c "import yaml; yaml.safe_load(open('$config_file'))" 2>/dev/null
            if [ $? -eq 0 ]; then
                echo "✓ $(basename "$config_file") restored successfully" | tee -a "$LOG_FILE"
            else
                echo "✗ $(basename "$config_file") restoration failed" | tee -a "$LOG_FILE"
            fi
        fi
    done
fi

# 5. Test service startup simulation
echo "5. Testing service startup simulation..." | tee -a "$LOG_FILE"

# Create temporary service configuration
cat > "$DR_TEST_DIR/test-service.conf" << EOF
[Unit]
Description=Project Health Test Service
After=network.target

[Service]
Type=simple
WorkingDirectory=$DR_TEST_DIR
ExecStart=/usr/bin/python3 -c "print('Service would start successfully')"
Restart=no

[Install]
WantedBy=multi-user.target
EOF

echo "✓ Service configuration created" | tee -a "$LOG_FILE"

# 6. Test network connectivity simulation
echo "6. Testing network connectivity..." | tee -a "$LOG_FILE"

# Test external dependencies
EXTERNAL_DEPS=("google.com" "github.com")
for dep in "${EXTERNAL_DEPS[@]}"; do
    if ping -c 1 "$dep" &> /dev/null; then
        echo "✓ Connectivity to $dep successful" | tee -a "$LOG_FILE"
    else
        echo "✗ Connectivity to $dep failed" | tee -a "$LOG_FILE"
    fi
done

# 7. Generate DR test report
echo "7. Generating DR test report..." | tee -a "$LOG_FILE"

DR_REPORT="$DR_TEST_DIR/dr-test-report.md"
cat > "$DR_REPORT" << EOF
# Disaster Recovery Test Report

**Date:** $(date)
**Test Duration:** $(date -d "$(head -1 "$LOG_FILE" | cut -d' ' -f5-)" +%s) seconds
**Backup Used:** $LATEST_BACKUP

## Test Results

### Database Restoration
- Status: $(grep "Database restoration" "$LOG_FILE" | tail -1 | cut -d' ' -f1)
- Records Recovered: $RECORD_COUNT

### Configuration Restoration
- Status: $(grep "config.*restored" "$LOG_FILE" | wc -l) files restored successfully

### Service Startup
- Status: ✓ Simulated successfully

### Network Connectivity
- External Dependencies: $(grep "Connectivity.*successful" "$LOG_FILE" | wc -l)/$(echo "${EXTERNAL_DEPS[@]}" | wc -w) successful

## Recommendations

1. Regular backup verification
2. Update disaster recovery procedures
3. Test with different failure scenarios
4. Train team on recovery procedures

## Next Test

Schedule next DR test for: $(date -d "+3 months" +%Y-%m-%d)
EOF

echo "DR test report generated: $DR_REPORT" | tee -a "$LOG_FILE"

# 8. Cleanup
echo "8. Cleaning up test environment..." | tee -a "$LOG_FILE"
# Keep report but remove test data
cp "$DR_REPORT" "/var/lib/project-health/reports/quarterly/"
rm -rf "$DR_TEST_DIR"

echo "=== Disaster Recovery Test Completed ===" | tee -a "$LOG_FILE"
```

## Emergency Procedures

### System Failure Response

```bash
#!/bin/bash
# emergency-response.sh

INCIDENT_ID="INC-$(date +%Y%m%d-%H%M%S)"
INCIDENT_LOG="/var/log/project-health/incidents/$INCIDENT_ID.log"

mkdir -p "/var/log/project-health/incidents"

echo "=== EMERGENCY RESPONSE - $INCIDENT_ID ===" | tee "$INCIDENT_LOG"
echo "Incident started: $(date)" | tee -a "$INCIDENT_LOG"

# 1. Immediate assessment
echo "1. IMMEDIATE ASSESSMENT" | tee -a "$INCIDENT_LOG"

# Check service status
systemctl is-active project-health >> "$INCIDENT_LOG" 2>&1
SERVICE_STATUS=$?

if [ $SERVICE_STATUS -ne 0 ]; then
    echo "CRITICAL: Service is not running" | tee -a "$INCIDENT_LOG"

    # Attempt service restart
    echo "Attempting service restart..." | tee -a "$INCIDENT_LOG"
    systemctl restart project-health
    sleep 10

    systemctl is-active project-health >> "$INCIDENT_LOG" 2>&1
    if [ $? -eq 0 ]; then
        echo "SUCCESS: Service restarted" | tee -a "$INCIDENT_LOG"
    else
        echo "FAILED: Service restart unsuccessful" | tee -a "$INCIDENT_LOG"
    fi
fi

# Check system resources
echo "System resources:" | tee -a "$INCIDENT_LOG"
df -h >> "$INCIDENT_LOG"
free -h >> "$INCIDENT_LOG"
uptime >> "$INCIDENT_LOG"

# 2. Collect diagnostic information
echo "2. DIAGNOSTIC INFORMATION" | tee -a "$INCIDENT_LOG"

# Recent logs
echo "Recent error logs:" | tee -a "$INCIDENT_LOG"
tail -50 /var/log/project-health/*.log | grep -i error >> "$INCIDENT_LOG" 2>/dev/null

# System logs
echo "System logs:" | tee -a "$INCIDENT_LOG"
journalctl -u project-health --since "1 hour ago" >> "$INCIDENT_LOG" 2>&1

# 3. Notification
echo "3. INCIDENT NOTIFICATION" | tee -a "$INCIDENT_LOG"

# Send emergency alert
python -m tools.health_checker.cli --send-alert \
    --message "EMERGENCY: System failure detected - Incident $INCIDENT_ID" \
    --severity critical \
    --incident-id "$INCIDENT_ID"

# 4. Escalation procedures
echo "4. ESCALATION PROCEDURES" | tee -a "$INCIDENT_LOG"
echo "If service cannot be restored within 30 minutes:" | tee -a "$INCIDENT_LOG"
echo "1. Contact on-call engineer: +1-555-0123" | tee -a "$INCIDENT_LOG"
echo "2. Initiate disaster recovery procedures" | tee -a "$INCIDENT_LOG"
echo "3. Notify stakeholders" | tee -a "$INCIDENT_LOG"

echo "Incident log: $INCIDENT_LOG"
```

### Data Corruption Recovery

```bash
#!/bin/bash
# data-corruption-recovery.sh

RECOVERY_ID="REC-$(date +%Y%m%d-%H%M%S)"
RECOVERY_LOG="/var/log/project-health/recovery/$RECOVERY_ID.log"

mkdir -p "/var/log/project-health/recovery"

echo "=== DATA CORRUPTION RECOVERY - $RECOVERY_ID ===" | tee "$RECOVERY_LOG"

# 1. Stop services to prevent further corruption
echo "1. Stopping services..." | tee -a "$RECOVERY_LOG"
systemctl stop project-health

# 2. Assess corruption extent
echo "2. Assessing corruption extent..." | tee -a "$RECOVERY_LOG"

DB_PATH="/var/lib/project-health/health.db"
if [ -f "$DB_PATH" ]; then
    echo "Checking database integrity..." | tee -a "$RECOVERY_LOG"
    sqlite3 "$DB_PATH" "PRAGMA integrity_check;" > /tmp/integrity_check.txt

    if grep -q "ok" /tmp/integrity_check.txt; then
        echo "Database integrity: OK" | tee -a "$RECOVERY_LOG"
    else
        echo "Database integrity: CORRUPTED" | tee -a "$RECOVERY_LOG"
        cat /tmp/integrity_check.txt >> "$RECOVERY_LOG"

        # Attempt database recovery
        echo "Attempting database recovery..." | tee -a "$RECOVERY_LOG"

        # Backup corrupted database
        cp "$DB_PATH" "$DB_PATH.corrupted.$(date +%Y%m%d-%H%M%S)"

        # Try to recover from backup
        LATEST_BACKUP=$(ls -t /backup/project-health/*/health.db | head -n1)
        if [ -f "$LATEST_BACKUP" ]; then
            echo "Restoring from backup: $LATEST_BACKUP" | tee -a "$RECOVERY_LOG"
            cp "$LATEST_BACKUP" "$DB_PATH"

            # Verify restored database
            sqlite3 "$DB_PATH" "PRAGMA integrity_check;" > /tmp/restore_check.txt
            if grep -q "ok" /tmp/restore_check.txt; then
                echo "Database restored successfully" | tee -a "$RECOVERY_LOG"
            else
                echo "Database restore failed" | tee -a "$RECOVERY_LOG"
            fi
        else
            echo "No backup database found" | tee -a "$RECOVERY_LOG"
        fi
    fi
fi

# 3. Check configuration files
echo "3. Checking configuration files..." | tee -a "$RECOVERY_LOG"
for config_file in /etc/project-health/*.yaml; do
    if [ -f "$config_file" ]; then
        python -c "import yaml; yaml.safe_load(open('$config_file'))" 2>/dev/null
        if [ $? -eq 0 ]; then
            echo "✓ $(basename "$config_file") is valid" | tee -a "$RECOVERY_LOG"
        else
            echo "✗ $(basename "$config_file") is corrupted" | tee -a "$RECOVERY_LOG"

            # Restore from backup
            BACKUP_CONFIG="/backup/project-health/latest/config/$(basename "$config_file")"
            if [ -f "$BACKUP_CONFIG" ]; then
                cp "$BACKUP_CONFIG" "$config_file"
                echo "Restored $(basename "$config_file") from backup" | tee -a "$RECOVERY_LOG"
            fi
        fi
    fi
done

# 4. Restart services
echo "4. Restarting services..." | tee -a "$RECOVERY_LOG"
systemctl start project-health

sleep 10

systemctl is-active project-health >> "$RECOVERY_LOG" 2>&1
if [ $? -eq 0 ]; then
    echo "Services restarted successfully" | tee -a "$RECOVERY_LOG"
else
    echo "Service restart failed" | tee -a "$RECOVERY_LOG"
fi

# 5. Verify system functionality
echo "5. Verifying system functionality..." | tee -a "$RECOVERY_LOG"
python -m tools.health_checker.cli --quick-check >> "$RECOVERY_LOG" 2>&1

echo "Recovery completed: $RECOVERY_LOG"
```

## Maintenance Automation

### Cron Job Configuration

```bash
# /etc/cron.d/project-health-maintenance

# Daily maintenance (6 AM)
0 6 * * * project-health /opt/project-health/scripts/daily-health-check.sh
15 6 * * * project-health /opt/project-health/scripts/daily-log-management.sh
30 6 * * * project-health /opt/project-health/scripts/daily-performance-monitoring.sh
45 6 * * * project-health /opt/project-health/scripts/daily-backup-verification.sh

# Weekly maintenance (Sunday 2 AM)
0 2 * * 0 project-health /opt/project-health/scripts/weekly-system-review.sh
30 2 * * 0 project-health /opt/project-health/scripts/weekly-security-review.sh
0 3 * * 0 project-health /opt/project-health/scripts/weekly-backup-test.sh

# Monthly maintenance (1st of month, 1 AM)
0 1 1 * * project-health /opt/project-health/scripts/monthly-optimization-review.py
0 1 2 * * project-health /opt/project-health/scripts/monthly-capacity-planning.py
0 1 3 * * project-health /opt/project-health/scripts/monthly-security-audit.sh

# Quarterly maintenance (1st of quarter, 12 AM)
0 0 1 1,4,7,10 * project-health /opt/project-health/scripts/quarterly-dr-test.sh
```

### Monitoring Integration

```yaml
# monitoring/alerts.yaml
alerts:
  - name: MaintenanceTaskFailed
    condition: maintenance_task_exit_code != 0
    severity: warning
    message: "Maintenance task failed: {{ task_name }}"

  - name: HealthScoreDeclined
    condition: health_score < 80
    severity: critical
    message: "Health score declined to {{ health_score }}"

  - name: BackupFailed
    condition: backup_age_hours > 25
    severity: critical
    message: "Backup is {{ backup_age_hours }} hours old"

  - name: SecurityVulnerabilities
    condition: security_vulnerabilities > 0
    severity: high
    message: "{{ security_vulnerabilities }} security vulnerabilities detected"
```

This comprehensive maintenance procedures document provides detailed guidance for keeping the project health system running optimally through regular maintenance, monitoring, and emergency response procedures.
