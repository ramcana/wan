# Enhanced Model Availability - Administrator Guide

## Overview

This guide provides comprehensive information for system administrators managing the Enhanced Model Availability system, including configuration, monitoring, maintenance, and troubleshooting at the system level.

## System Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────┐
│ Enhanced Model Availability System                      │
├─────────────────────────────────────────────────────────┤
│ • ModelAvailabilityManager (Coordination)              │
│ • EnhancedModelDownloader (Download Management)        │
│ • ModelHealthMonitor (Integrity & Performance)         │
│ • ModelUsageAnalytics (Usage Tracking)                 │
│ • IntelligentFallbackManager (Smart Alternatives)      │
│ • EnhancedErrorRecovery (Advanced Recovery)            │
└─────────────────────────────────────────────────────────┘
    ↓ (Integrates with)
┌─────────────────────────────────────────────────────────┐
│ Core Infrastructure                                     │
├─────────────────────────────────────────────────────────┤
│ • ModelManager (Base Model Management)                 │
│ • ModelDownloader (Basic Download)                     │
│ • FallbackRecoverySystem (Basic Recovery)              │
│ • WebSocket Manager (Real-time Updates)                │
└─────────────────────────────────────────────────────────┘
```

### Service Dependencies

- **FastAPI Backend**: Core API and service layer
- **WebSocket Manager**: Real-time notifications
- **Database**: Analytics and usage tracking (optional)
- **File System**: Model storage and caching
- **Network**: Model downloads and updates

## Installation and Setup

### System Requirements

**Minimum Requirements**:

- Python 3.8+
- 8GB RAM
- 50GB free disk space
- Stable internet connection (10 Mbps+)

**Recommended Requirements**:

- Python 3.10+
- 16GB+ RAM
- 200GB+ free disk space (SSD preferred)
- High-speed internet (100 Mbps+)
- GPU with 8GB+ VRAM (for model loading)

### Installation Steps

1. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   pip install enhanced-model-availability
   ```

2. **Initialize Configuration**:

   ```bash
   python -m enhanced_model_availability init-config
   ```

3. **Set Up Storage**:

   ```bash
   mkdir -p /data/models
   chown -R app:app /data/models
   chmod 755 /data/models
   ```

4. **Configure Database** (optional):

   ```bash
   python -m enhanced_model_availability init-db
   ```

5. **Start Services**:
   ```bash
   systemctl enable enhanced-model-availability
   systemctl start enhanced-model-availability
   ```

## Configuration Management

### Main Configuration File

Location: `config/enhanced_model_config.json`

```json
{
  "storage": {
    "models_directory": "/data/models",
    "cache_directory": "/data/cache",
    "max_storage_gb": 500,
    "cleanup_threshold_percent": 90
  },
  "downloads": {
    "max_concurrent_downloads": 3,
    "max_retries": 3,
    "retry_delay_seconds": 30,
    "bandwidth_limit_mbps": 0,
    "resume_enabled": true,
    "integrity_check_enabled": true
  },
  "health_monitoring": {
    "enabled": true,
    "check_interval_hours": 24,
    "auto_repair_enabled": true,
    "performance_monitoring_enabled": true,
    "corruption_detection_enabled": true
  },
  "fallback": {
    "intelligent_fallback_enabled": true,
    "suggestion_threshold": 0.7,
    "queue_requests_enabled": true,
    "mock_fallback_enabled": true
  },
  "analytics": {
    "enabled": true,
    "retention_days": 90,
    "detailed_tracking": true,
    "performance_metrics": true
  },
  "features": {
    "enhanced_downloads": true,
    "health_monitoring": true,
    "intelligent_fallback": true,
    "usage_analytics": true,
    "auto_updates": false,
    "websocket_notifications": true
  },
  "security": {
    "verify_downloads": true,
    "quarantine_suspicious": true,
    "max_file_size_gb": 50,
    "allowed_sources": ["huggingface.co"]
  }
}
```

### Environment Variables

```bash
# Core Settings
ENHANCED_MODEL_CONFIG_PATH=/etc/enhanced-model-availability/config.json
ENHANCED_MODEL_STORAGE_PATH=/data/models
ENHANCED_MODEL_LOG_LEVEL=INFO

# Database (optional)
ENHANCED_MODEL_DB_URL=postgresql://user:pass@localhost/models
ENHANCED_MODEL_DB_POOL_SIZE=10

# Security
ENHANCED_MODEL_API_KEY=your-api-key
ENHANCED_MODEL_ALLOWED_HOSTS=localhost,your-domain.com

# Performance
ENHANCED_MODEL_WORKER_THREADS=4
ENHANCED_MODEL_MAX_MEMORY_GB=8
```

### Feature Flags

Control feature availability through configuration:

```json
{
  "feature_flags": {
    "enhanced_downloads": {
      "enabled": true,
      "rollout_percentage": 100,
      "user_groups": ["admin", "power_user"]
    },
    "health_monitoring": {
      "enabled": true,
      "auto_repair": true,
      "scheduled_checks": true
    },
    "intelligent_fallback": {
      "enabled": true,
      "suggestion_engine": true,
      "queue_management": true
    },
    "usage_analytics": {
      "enabled": true,
      "detailed_tracking": false,
      "retention_days": 30
    }
  }
}
```

## User Management

### Access Control

Configure user permissions for model management:

```json
{
  "access_control": {
    "roles": {
      "admin": {
        "permissions": ["*"],
        "description": "Full system access"
      },
      "power_user": {
        "permissions": [
          "models.download",
          "models.manage",
          "models.analytics.view",
          "models.health.view"
        ]
      },
      "user": {
        "permissions": ["models.view", "models.download.basic"]
      }
    },
    "default_role": "user",
    "require_authentication": false
  }
}
```

### User Quotas

Set limits per user or group:

```json
{
  "quotas": {
    "default": {
      "max_models": 10,
      "max_storage_gb": 50,
      "max_concurrent_downloads": 2
    },
    "power_user": {
      "max_models": 50,
      "max_storage_gb": 200,
      "max_concurrent_downloads": 5
    },
    "admin": {
      "max_models": -1,
      "max_storage_gb": -1,
      "max_concurrent_downloads": -1
    }
  }
}
```

## Monitoring and Alerting

### System Health Monitoring

#### Health Check Endpoints

```bash
# Overall system health
curl http://localhost:8000/api/v1/admin/health

# Component-specific health
curl http://localhost:8000/api/v1/admin/health/downloads
curl http://localhost:8000/api/v1/admin/health/storage
curl http://localhost:8000/api/v1/admin/health/models

# Detailed system status
curl http://localhost:8000/api/v1/admin/status/detailed
```

#### Key Metrics to Monitor

1. **Download Performance**:

   - Success rate (target: >95%)
   - Average download speed
   - Retry frequency
   - Queue length

2. **Storage Health**:

   - Disk usage percentage
   - Available space
   - Model count
   - Cleanup frequency

3. **Model Health**:

   - Integrity check results
   - Corruption detection rate
   - Performance scores
   - Health check frequency

4. **System Performance**:
   - API response times
   - Memory usage
   - CPU utilization
   - WebSocket connections

### Alerting Configuration

#### Prometheus Integration

```yaml
# prometheus.yml
scrape_configs:
  - job_name: "enhanced-model-availability"
    static_configs:
      - targets: ["localhost:8000"]
    metrics_path: "/api/v1/admin/metrics"
    scrape_interval: 30s
```

#### Alert Rules

```yaml
# alerts.yml
groups:
  - name: enhanced_model_availability
    rules:
      - alert: HighDownloadFailureRate
        expr: download_failure_rate > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High download failure rate detected"

      - alert: LowDiskSpace
        expr: disk_usage_percent > 90
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Low disk space for model storage"

      - alert: ModelCorruptionDetected
        expr: corrupted_models_count > 0
        for: 0m
        labels:
          severity: warning
        annotations:
          summary: "Model corruption detected"
```

### Log Management

#### Log Configuration

```json
{
  "logging": {
    "level": "INFO",
    "format": "json",
    "rotation": {
      "max_size_mb": 100,
      "backup_count": 10,
      "compress": true
    },
    "outputs": {
      "file": {
        "enabled": true,
        "path": "/var/log/enhanced-model-availability/"
      },
      "syslog": {
        "enabled": false,
        "facility": "local0"
      },
      "elasticsearch": {
        "enabled": false,
        "host": "localhost:9200",
        "index": "model-availability"
      }
    },
    "components": {
      "downloads": "INFO",
      "health_monitor": "INFO",
      "fallback": "INFO",
      "analytics": "WARNING",
      "api": "INFO"
    }
  }
}
```

#### Log Analysis

Key log patterns to monitor:

```bash
# Download failures
grep "download.*failed" /var/log/enhanced-model-availability/downloads.log

# Model corruption
grep "corruption.*detected" /var/log/enhanced-model-availability/health.log

# Performance issues
grep "slow.*response" /var/log/enhanced-model-availability/api.log

# Security events
grep "security.*violation" /var/log/enhanced-model-availability/security.log
```

## Performance Optimization

### Storage Optimization

#### Storage Tiering

Configure different storage tiers for models:

```json
{
  "storage_tiers": {
    "hot": {
      "path": "/fast-ssd/models",
      "criteria": "frequently_used",
      "max_size_gb": 100
    },
    "warm": {
      "path": "/ssd/models",
      "criteria": "occasionally_used",
      "max_size_gb": 500
    },
    "cold": {
      "path": "/hdd/models",
      "criteria": "rarely_used",
      "max_size_gb": 2000
    }
  }
}
```

#### Cleanup Policies

```json
{
  "cleanup_policies": {
    "automatic": {
      "enabled": true,
      "trigger_threshold_percent": 85,
      "target_threshold_percent": 70,
      "min_unused_days": 30
    },
    "scheduled": {
      "enabled": true,
      "schedule": "0 2 * * 0",
      "aggressive_cleanup": false
    },
    "retention": {
      "keep_recent_days": 7,
      "keep_frequently_used": true,
      "min_usage_count": 5
    }
  }
}
```

### Download Optimization

#### Bandwidth Management

```json
{
  "bandwidth_management": {
    "global_limit_mbps": 100,
    "per_download_limit_mbps": 50,
    "time_based_limits": {
      "business_hours": {
        "start": "09:00",
        "end": "17:00",
        "limit_mbps": 20
      },
      "off_hours": {
        "limit_mbps": 100
      }
    },
    "adaptive_limiting": {
      "enabled": true,
      "monitor_system_load": true,
      "reduce_on_high_load": true
    }
  }
}
```

#### Connection Optimization

```json
{
  "connection_settings": {
    "max_connections_per_host": 4,
    "connection_timeout_seconds": 30,
    "read_timeout_seconds": 300,
    "retry_backoff_factor": 2.0,
    "max_retry_delay_seconds": 300,
    "keep_alive_enabled": true,
    "compression_enabled": true
  }
}
```

### System Resource Management

#### Memory Management

```json
{
  "memory_management": {
    "max_memory_usage_gb": 8,
    "model_cache_size_gb": 4,
    "download_buffer_size_mb": 64,
    "gc_threshold": 0.8,
    "preload_frequently_used": true
  }
}
```

#### CPU Optimization

```json
{
  "cpu_optimization": {
    "worker_threads": 4,
    "io_threads": 8,
    "background_tasks_threads": 2,
    "cpu_affinity": [0, 1, 2, 3],
    "priority": "normal"
  }
}
```

## Security Configuration

### Download Security

```json
{
  "download_security": {
    "verify_checksums": true,
    "verify_signatures": true,
    "quarantine_suspicious": true,
    "scan_for_malware": false,
    "allowed_file_types": [".bin", ".safetensors", ".json", ".txt"],
    "max_file_size_gb": 50,
    "trusted_sources": ["huggingface.co", "github.com"]
  }
}
```

### API Security

```json
{
  "api_security": {
    "authentication_required": false,
    "api_key_required": false,
    "rate_limiting": {
      "enabled": true,
      "requests_per_minute": 60,
      "burst_size": 10
    },
    "cors": {
      "enabled": true,
      "allowed_origins": ["*"],
      "allowed_methods": ["GET", "POST", "PUT", "DELETE"],
      "allowed_headers": ["*"]
    },
    "request_validation": {
      "strict_mode": false,
      "max_request_size_mb": 10,
      "sanitize_inputs": true
    }
  }
}
```

### File System Security

```json
{
  "filesystem_security": {
    "restrict_paths": true,
    "allowed_directories": ["/data/models", "/tmp/downloads"],
    "file_permissions": "644",
    "directory_permissions": "755",
    "owner": "app",
    "group": "app",
    "prevent_symlink_traversal": true
  }
}
```

## Backup and Recovery

### Backup Strategy

#### Model Backup

```bash
#!/bin/bash
# backup_models.sh

BACKUP_DIR="/backup/models"
MODELS_DIR="/data/models"
DATE=$(date +%Y%m%d_%H%M%S)

# Create incremental backup
rsync -av --link-dest="$BACKUP_DIR/latest" \
  "$MODELS_DIR/" "$BACKUP_DIR/$DATE/"

# Update latest symlink
ln -sfn "$BACKUP_DIR/$DATE" "$BACKUP_DIR/latest"

# Cleanup old backups (keep 7 days)
find "$BACKUP_DIR" -maxdepth 1 -type d -mtime +7 -exec rm -rf {} \;
```

#### Configuration Backup

```bash
#!/bin/bash
# backup_config.sh

CONFIG_DIR="/etc/enhanced-model-availability"
BACKUP_DIR="/backup/config"
DATE=$(date +%Y%m%d_%H%M%S)

tar -czf "$BACKUP_DIR/config_$DATE.tar.gz" "$CONFIG_DIR"

# Keep 30 days of config backups
find "$BACKUP_DIR" -name "config_*.tar.gz" -mtime +30 -delete
```

### Disaster Recovery

#### Recovery Procedures

1. **Service Recovery**:

   ```bash
   # Stop service
   systemctl stop enhanced-model-availability

   # Restore configuration
   tar -xzf /backup/config/config_latest.tar.gz -C /

   # Restore models (if needed)
   rsync -av /backup/models/latest/ /data/models/

   # Start service
   systemctl start enhanced-model-availability
   ```

2. **Database Recovery** (if using analytics DB):

   ```bash
   # Restore database
   pg_restore -d models /backup/db/models_latest.dump

   # Verify data integrity
   curl http://localhost:8000/api/v1/admin/analytics/verify
   ```

3. **Partial Recovery**:

   ```bash
   # Recover specific model
   rsync -av /backup/models/latest/MODEL_NAME/ /data/models/MODEL_NAME/

   # Trigger integrity check
   curl -X POST http://localhost:8000/api/v1/models/health/MODEL_NAME/check
   ```

## Maintenance Procedures

### Regular Maintenance Tasks

#### Daily Tasks

```bash
#!/bin/bash
# daily_maintenance.sh

# Check system health
curl -f http://localhost:8000/api/v1/admin/health || exit 1

# Rotate logs
logrotate /etc/logrotate.d/enhanced-model-availability

# Update usage statistics
curl -X POST http://localhost:8000/api/v1/admin/analytics/update

# Check for corrupted models
curl http://localhost:8000/api/v1/admin/health/models/corruption
```

#### Weekly Tasks

```bash
#!/bin/bash
# weekly_maintenance.sh

# Full system health check
curl -X POST http://localhost:8000/api/v1/admin/health/full-check

# Cleanup unused models
curl -X POST http://localhost:8000/api/v1/admin/cleanup/auto

# Generate health report
curl http://localhost:8000/api/v1/admin/reports/health > /reports/weekly_health.json

# Backup configuration
/scripts/backup_config.sh
```

#### Monthly Tasks

```bash
#!/bin/bash
# monthly_maintenance.sh

# Full model integrity check
curl -X POST http://localhost:8000/api/v1/admin/integrity/full-check

# Optimize storage
curl -X POST http://localhost:8000/api/v1/admin/storage/optimize

# Update system dependencies
pip install --upgrade enhanced-model-availability

# Performance analysis
curl http://localhost:8000/api/v1/admin/performance/analysis > /reports/monthly_performance.json
```

### Update Procedures

#### System Updates

1. **Preparation**:

   ```bash
   # Backup current system
   /scripts/backup_config.sh
   /scripts/backup_models.sh

   # Check current version
   curl http://localhost:8000/api/v1/admin/version
   ```

2. **Update Process**:

   ```bash
   # Stop service
   systemctl stop enhanced-model-availability

   # Update package
   pip install --upgrade enhanced-model-availability

   # Migrate configuration
   python -m enhanced_model_availability migrate-config

   # Start service
   systemctl start enhanced-model-availability
   ```

3. **Verification**:

   ```bash
   # Verify service health
   curl -f http://localhost:8000/api/v1/admin/health

   # Check feature availability
   curl http://localhost:8000/api/v1/admin/features

   # Run integration tests
   python -m enhanced_model_availability test-integration
   ```

## Troubleshooting

### Common Admin Issues

#### Service Won't Start

**Diagnostic Steps**:

```bash
# Check service status
systemctl status enhanced-model-availability

# Check logs
journalctl -u enhanced-model-availability -f

# Verify configuration
python -m enhanced_model_availability validate-config

# Check dependencies
pip check enhanced-model-availability
```

#### High Resource Usage

**Diagnostic Steps**:

```bash
# Check resource usage
curl http://localhost:8000/api/v1/admin/resources

# Monitor system resources
htop
iotop
nvidia-smi

# Check active downloads
curl http://localhost:8000/api/v1/admin/downloads/active
```

#### Database Issues

**Diagnostic Steps**:

```bash
# Check database connectivity
curl http://localhost:8000/api/v1/admin/db/health

# Check database size
du -sh /var/lib/postgresql/data

# Analyze slow queries
curl http://localhost:8000/api/v1/admin/db/slow-queries
```

### Performance Troubleshooting

#### Slow Downloads

1. **Network Analysis**:

   ```bash
   # Test network speed
   speedtest-cli

   # Check DNS resolution
   nslookup huggingface.co

   # Test connectivity
   curl -w "@curl-format.txt" -o /dev/null -s https://huggingface.co/
   ```

2. **System Analysis**:

   ```bash
   # Check disk I/O
   iotop -a

   # Check network I/O
   iftop

   # Check system load
   uptime
   ```

#### Memory Issues

1. **Memory Analysis**:

   ```bash
   # Check memory usage
   free -h

   # Check for memory leaks
   curl http://localhost:8000/api/v1/admin/memory/analysis

   # Monitor memory over time
   vmstat 5
   ```

2. **Optimization**:

   ```bash
   # Reduce cache size
   curl -X POST http://localhost:8000/api/v1/admin/cache/reduce

   # Force garbage collection
   curl -X POST http://localhost:8000/api/v1/admin/gc/force
   ```

## Best Practices

### Configuration Management

1. **Version Control**: Keep configuration files in version control
2. **Environment Separation**: Use different configs for dev/staging/prod
3. **Validation**: Always validate configuration before applying
4. **Documentation**: Document all configuration changes
5. **Rollback Plan**: Have rollback procedures for configuration changes

### Monitoring and Alerting

1. **Proactive Monitoring**: Monitor trends, not just current state
2. **Alert Fatigue**: Tune alerts to reduce false positives
3. **Escalation**: Define clear escalation procedures
4. **Documentation**: Document all alerts and their resolution
5. **Regular Review**: Review and update monitoring regularly

### Security

1. **Principle of Least Privilege**: Grant minimum necessary permissions
2. **Regular Updates**: Keep system and dependencies updated
3. **Security Scanning**: Regular security scans and audits
4. **Incident Response**: Have clear incident response procedures
5. **Backup Security**: Secure backup storage and access

### Performance

1. **Capacity Planning**: Plan for growth and peak usage
2. **Resource Monitoring**: Monitor all system resources
3. **Performance Testing**: Regular performance testing
4. **Optimization**: Continuous performance optimization
5. **Scaling**: Plan for horizontal and vertical scaling

This administrator guide provides comprehensive coverage of system management, configuration, monitoring, and maintenance for the Enhanced Model Availability system.
