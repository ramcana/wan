# Enhanced Model Availability - Migration Guide

## Overview

This guide provides step-by-step instructions for migrating from the basic model management system to the Enhanced Model Availability system. It covers data migration, configuration updates, API changes, and best practices for a smooth transition.

## Pre-Migration Assessment

### System Requirements Check

Before starting the migration, verify your system meets the enhanced requirements:

```bash
# Check Python version (3.8+ required, 3.10+ recommended)
python --version

# Check available disk space (minimum 50GB, recommended 200GB+)
df -h

# Check memory (minimum 8GB, recommended 16GB+)
free -h

# Check network connectivity
ping huggingface.co
```

### Current System Analysis

Analyze your current model management setup:

```bash
# Check current models
ls -la models/
du -sh models/*

# Check current configuration
cat config.json

# Check current API usage
curl http://localhost:8000/api/v1/models/status
```

### Backup Current System

Create a complete backup before migration:

```bash
#!/bin/bash
# backup_current_system.sh

BACKUP_DIR="/backup/pre-migration-$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Backup models
echo "Backing up models..."
cp -r models/ "$BACKUP_DIR/models/"

# Backup configuration
echo "Backing up configuration..."
cp -r config/ "$BACKUP_DIR/config/" 2>/dev/null || true
cp config.json "$BACKUP_DIR/" 2>/dev/null || true

# Backup logs
echo "Backing up logs..."
cp -r logs/ "$BACKUP_DIR/logs/" 2>/dev/null || true

# Backup database (if exists)
echo "Backing up database..."
if [ -f "models.db" ]; then
    cp models.db "$BACKUP_DIR/"
fi

# Create system info snapshot
echo "Creating system snapshot..."
{
    echo "=== System Info ==="
    uname -a
    echo "=== Python Version ==="
    python --version
    echo "=== Installed Packages ==="
    pip list
    echo "=== Disk Usage ==="
    df -h
    echo "=== Memory Info ==="
    free -h
} > "$BACKUP_DIR/system_info.txt"

echo "Backup completed: $BACKUP_DIR"
```

## Migration Paths

### Path 1: In-Place Migration (Recommended)

Upgrade the existing installation while preserving data and configuration.

#### Step 1: Stop Current Services

```bash
# Stop the current service
systemctl stop model-management-service
# or
pkill -f "python.*app.py"

# Verify services are stopped
ps aux | grep -E "(app.py|model)"
```

#### Step 2: Install Enhanced System

```bash
# Update package
pip install --upgrade enhanced-model-availability

# Install additional dependencies
pip install -r requirements-enhanced.txt

# Verify installation
python -c "import enhanced_model_availability; print('Installation successful')"
```

#### Step 3: Migrate Configuration

```bash
# Run configuration migration tool
python -m enhanced_model_availability migrate-config \
    --source config.json \
    --target config/enhanced_model_config.json \
    --backup

# Verify configuration
python -m enhanced_model_availability validate-config \
    --config config/enhanced_model_config.json
```

#### Step 4: Migrate Model Data

```bash
# Run model data migration
python -m enhanced_model_availability migrate-models \
    --models-dir models/ \
    --verify-integrity \
    --create-metadata

# Check migration results
python -m enhanced_model_availability verify-migration \
    --models-dir models/
```

#### Step 5: Start Enhanced Services

```bash
# Start enhanced services
systemctl start enhanced-model-availability

# Verify services are running
curl http://localhost:8000/api/v1/admin/health

# Check enhanced features
curl http://localhost:8000/api/v1/models/status/detailed
```

### Path 2: Side-by-Side Migration

Run both systems in parallel during transition.

#### Step 1: Install Enhanced System in New Location

```bash
# Create new installation directory
mkdir -p /opt/enhanced-model-availability
cd /opt/enhanced-model-availability

# Install enhanced system
python -m venv venv
source venv/bin/activate
pip install enhanced-model-availability

# Configure for different port
export ENHANCED_MODEL_PORT=8001
```

#### Step 2: Configure Data Sharing

```bash
# Create configuration for shared model directory
cat > config/enhanced_model_config.json << EOF
{
  "storage": {
    "models_directory": "/data/models",
    "cache_directory": "/data/cache-enhanced",
    "shared_storage": true
  },
  "api": {
    "port": 8001,
    "host": "0.0.0.0"
  },
  "migration": {
    "compatibility_mode": true,
    "legacy_api_support": true
  }
}
EOF
```

#### Step 3: Start Enhanced System

```bash
# Start enhanced system on different port
python -m enhanced_model_availability start \
    --config config/enhanced_model_config.json \
    --port 8001

# Verify both systems running
curl http://localhost:8000/api/v1/models/status  # Legacy
curl http://localhost:8001/api/v1/models/status/detailed  # Enhanced
```

#### Step 4: Gradual Migration

```bash
# Migrate clients gradually
# Update client configurations to use port 8001
# Test enhanced features with subset of users
# Monitor both systems during transition
```

#### Step 5: Complete Migration

```bash
# Stop legacy system
systemctl stop model-management-service

# Update enhanced system to use standard port
# Update configuration and restart
systemctl restart enhanced-model-availability
```

## Configuration Migration

### Automatic Configuration Migration

The migration tool automatically converts most settings:

```bash
# Run automatic migration
python -m enhanced_model_availability migrate-config \
    --source config.json \
    --target config/enhanced_model_config.json \
    --interactive

# Review generated configuration
cat config/enhanced_model_config.json
```

### Manual Configuration Updates

Some settings require manual review and adjustment:

#### Legacy Configuration Example

```json
{
  "models_directory": "/data/models",
  "download_retries": 3,
  "max_concurrent_downloads": 2,
  "health_check_enabled": false,
  "fallback_to_mock": true
}
```

#### Enhanced Configuration Equivalent

```json
{
  "storage": {
    "models_directory": "/data/models",
    "cache_directory": "/data/cache",
    "max_storage_gb": 500,
    "cleanup_threshold_percent": 90
  },
  "downloads": {
    "max_concurrent_downloads": 2,
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
    "performance_monitoring_enabled": true
  },
  "fallback": {
    "intelligent_fallback_enabled": true,
    "suggestion_threshold": 0.7,
    "queue_requests_enabled": true,
    "mock_fallback_enabled": true
  },
  "features": {
    "enhanced_downloads": true,
    "health_monitoring": true,
    "intelligent_fallback": true,
    "usage_analytics": true
  }
}
```

### Configuration Mapping Reference

| Legacy Setting             | Enhanced Setting                        | Notes                        |
| -------------------------- | --------------------------------------- | ---------------------------- |
| `models_directory`         | `storage.models_directory`              | Direct mapping               |
| `download_retries`         | `downloads.max_retries`                 | Direct mapping               |
| `max_concurrent_downloads` | `downloads.max_concurrent_downloads`    | Direct mapping               |
| `health_check_enabled`     | `health_monitoring.enabled`             | Enhanced with more options   |
| `fallback_to_mock`         | `fallback.mock_fallback_enabled`        | Part of intelligent fallback |
| N/A                        | `analytics.enabled`                     | New feature                  |
| N/A                        | `downloads.resume_enabled`              | New feature                  |
| N/A                        | `fallback.intelligent_fallback_enabled` | New feature                  |

## Data Migration

### Model Metadata Migration

The enhanced system creates additional metadata for each model:

```bash
# Run metadata generation
python -m enhanced_model_availability generate-metadata \
    --models-dir models/ \
    --force-regenerate

# Verify metadata creation
find models/ -name "*.metadata.json" | head -5
```

### Database Migration (if using analytics)

If enabling analytics features, set up the database:

```bash
# Initialize analytics database
python -m enhanced_model_availability init-analytics-db \
    --db-url postgresql://user:pass@localhost/models

# Migrate existing usage data (if available)
python -m enhanced_model_availability migrate-usage-data \
    --source logs/usage.log \
    --target-db postgresql://user:pass@localhost/models
```

### Model Integrity Verification

Verify all models after migration:

```bash
# Run comprehensive integrity check
python -m enhanced_model_availability verify-all-models \
    --models-dir models/ \
    --repair-corrupted \
    --generate-report

# Review integrity report
cat reports/model_integrity_report.json
```

## API Migration

### API Endpoint Changes

#### Legacy API Endpoints

```bash
# Legacy endpoints
GET /api/v1/models/status
POST /api/v1/models/download
GET /api/v1/models/health
```

#### Enhanced API Endpoints

```bash
# Enhanced endpoints (backward compatible)
GET /api/v1/models/status/detailed
POST /api/v1/models/download/manage
GET /api/v1/models/health
GET /api/v1/models/analytics
GET /api/v1/models/fallback/suggest
```

### Client Code Migration

#### Legacy Client Code

```python
import requests

# Legacy API usage
response = requests.get('http://localhost:8000/api/v1/models/status')
models = response.json()

for model_id, status in models.items():
    if not status['available']:
        # Start download
        requests.post(f'http://localhost:8000/api/v1/models/download',
                     json={'model_id': model_id})
```

#### Enhanced Client Code

```python
import requests

# Enhanced API usage with backward compatibility
response = requests.get('http://localhost:8000/api/v1/models/status/detailed')
data = response.json()

for model_id, status in data['models'].items():
    if not status['is_available']:
        # Start download with retry logic
        requests.post('http://localhost:8000/api/v1/models/download/manage',
                     json={
                         'model_id': model_id,
                         'action': 'start',
                         'max_retries': 3
                     })

        # Get suggestions if download fails
        suggestions = requests.post('http://localhost:8000/api/v1/models/fallback/suggest',
                                  json={
                                      'requested_model': model_id,
                                      'requirements': {'quality': 'high'}
                                  })
```

### WebSocket Integration Migration

#### Adding WebSocket Support

```javascript
// Legacy: Polling for updates
setInterval(() => {
  fetch("/api/v1/models/status")
    .then((response) => response.json())
    .then((data) => updateUI(data));
}, 5000);

// Enhanced: WebSocket real-time updates
const ws = new WebSocket("ws://localhost:8000/ws");

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);

  switch (data.type) {
    case "model_download_progress":
      updateDownloadProgress(data.model_id, data.progress_percent);
      break;
    case "model_availability_change":
      updateModelStatus(data.model_id, data.new_status);
      break;
    case "model_health_alert":
      showHealthAlert(data.model_id, data.message);
      break;
  }
};
```

## Testing Migration

### Pre-Migration Testing

Test the enhanced system before full migration:

```bash
# Test enhanced system with sample data
python -m enhanced_model_availability test-installation \
    --sample-models \
    --run-all-tests

# Test API compatibility
python -m enhanced_model_availability test-api-compatibility \
    --legacy-endpoints \
    --enhanced-endpoints
```

### Post-Migration Validation

Validate the migration was successful:

```bash
# Run comprehensive validation
python -m enhanced_model_availability validate-migration \
    --check-models \
    --check-config \
    --check-api \
    --check-features

# Test all enhanced features
python -m enhanced_model_availability test-enhanced-features \
    --download-management \
    --health-monitoring \
    --intelligent-fallback \
    --usage-analytics
```

### Performance Testing

Compare performance before and after migration:

```bash
# Run performance benchmarks
python -m enhanced_model_availability benchmark \
    --test-downloads \
    --test-api-response \
    --test-model-loading \
    --compare-with-baseline
```

## Rollback Procedures

### Automatic Rollback

If migration fails, use automatic rollback:

```bash
# Rollback to previous version
python -m enhanced_model_availability rollback \
    --backup-dir /backup/pre-migration-20240101_120000 \
    --verify-rollback

# Restart legacy services
systemctl start model-management-service
```

### Manual Rollback

For manual rollback if automatic fails:

```bash
# Stop enhanced services
systemctl stop enhanced-model-availability

# Restore from backup
BACKUP_DIR="/backup/pre-migration-20240101_120000"

# Restore models
rm -rf models/
cp -r "$BACKUP_DIR/models/" models/

# Restore configuration
cp "$BACKUP_DIR/config.json" config.json

# Restore database (if exists)
cp "$BACKUP_DIR/models.db" models.db

# Downgrade package
pip install model-management==1.0.0

# Start legacy services
systemctl start model-management-service
```

## Common Migration Issues

### Issue 1: Configuration Validation Errors

**Problem**: Configuration migration fails validation

**Solution**:

```bash
# Check specific validation errors
python -m enhanced_model_availability validate-config \
    --config config/enhanced_model_config.json \
    --verbose

# Fix common issues
python -m enhanced_model_availability fix-config \
    --config config/enhanced_model_config.json \
    --auto-fix
```

### Issue 2: Model Metadata Generation Fails

**Problem**: Cannot generate metadata for existing models

**Solution**:

```bash
# Check model integrity first
python -m enhanced_model_availability check-models \
    --models-dir models/ \
    --repair-if-needed

# Regenerate metadata with force flag
python -m enhanced_model_availability generate-metadata \
    --models-dir models/ \
    --force-regenerate \
    --ignore-errors
```

### Issue 3: API Compatibility Issues

**Problem**: Legacy clients cannot connect to enhanced API

**Solution**:

```bash
# Enable legacy compatibility mode
python -m enhanced_model_availability configure \
    --enable-legacy-api \
    --legacy-endpoints-only

# Or run both APIs in parallel
python -m enhanced_model_availability start \
    --legacy-port 8000 \
    --enhanced-port 8001
```

### Issue 4: Performance Degradation

**Problem**: System slower after migration

**Solution**:

```bash
# Disable resource-intensive features temporarily
python -m enhanced_model_availability configure \
    --disable-health-monitoring \
    --disable-analytics \
    --basic-mode

# Optimize configuration
python -m enhanced_model_availability optimize-config \
    --for-performance \
    --system-specs auto-detect
```

### Issue 5: Storage Issues

**Problem**: Insufficient storage or permission errors

**Solution**:

```bash
# Check and fix permissions
sudo chown -R app:app /data/models
sudo chmod -R 755 /data/models

# Clean up unnecessary files
python -m enhanced_model_availability cleanup \
    --remove-temp-files \
    --remove-old-logs \
    --compress-backups

# Move models to larger storage
python -m enhanced_model_availability move-models \
    --source /data/models \
    --target /large-storage/models \
    --update-config
```

## Post-Migration Optimization

### Performance Tuning

After successful migration, optimize the system:

```bash
# Run performance analysis
python -m enhanced_model_availability analyze-performance \
    --generate-recommendations

# Apply recommended optimizations
python -m enhanced_model_availability optimize \
    --apply-recommendations \
    --restart-if-needed
```

### Feature Enablement

Gradually enable enhanced features:

```bash
# Enable features one by one
python -m enhanced_model_availability enable-feature \
    --feature health-monitoring \
    --test-first

python -m enhanced_model_availability enable-feature \
    --feature intelligent-fallback \
    --test-first

python -m enhanced_model_availability enable-feature \
    --feature usage-analytics \
    --test-first
```

### Monitoring Setup

Set up monitoring for the enhanced system:

```bash
# Configure monitoring
python -m enhanced_model_availability setup-monitoring \
    --prometheus \
    --grafana \
    --alerting

# Test monitoring
python -m enhanced_model_availability test-monitoring \
    --generate-test-alerts
```

## Migration Checklist

### Pre-Migration Checklist

- [ ] System requirements verified
- [ ] Current system backed up
- [ ] Migration plan reviewed
- [ ] Downtime window scheduled
- [ ] Rollback procedures tested
- [ ] Team notified

### Migration Checklist

- [ ] Services stopped
- [ ] Enhanced system installed
- [ ] Configuration migrated
- [ ] Model data migrated
- [ ] Database migrated (if applicable)
- [ ] Services started
- [ ] Basic functionality tested

### Post-Migration Checklist

- [ ] All models accessible
- [ ] API endpoints responding
- [ ] Enhanced features working
- [ ] Performance acceptable
- [ ] Monitoring configured
- [ ] Documentation updated
- [ ] Team trained on new features
- [ ] Rollback procedures verified

## Support and Resources

### Getting Help During Migration

- **Documentation**: Refer to user guide and troubleshooting guide
- **Migration Tool Help**: `python -m enhanced_model_availability migrate --help`
- **Validation Tools**: Use built-in validation and testing tools
- **Community Support**: Check community forums and discussions
- **Professional Support**: Contact support team for complex migrations

### Additional Resources

- **Migration Scripts**: Available in `scripts/migration/`
- **Test Data**: Sample configurations and models for testing
- **Performance Baselines**: Reference performance metrics
- **Best Practices**: Recommended configurations and optimizations

This comprehensive migration guide ensures a smooth transition from the basic model management system to the Enhanced Model Availability system while minimizing downtime and preserving data integrity.
