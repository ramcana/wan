# Enhanced Model Availability Deployment Tools

This directory contains comprehensive deployment and migration tools for the Enhanced Model Availability System. These tools provide automated deployment, validation, rollback capabilities, and monitoring setup.

## Overview

The deployment system consists of several interconnected components:

- **Deployment Orchestrator** (`deploy.py`) - Main deployment automation
- **Deployment Validator** (`deployment_validator.py`) - Pre/post deployment validation
- **Rollback Manager** (`rollback_manager.py`) - Rollback and recovery capabilities
- **Model Migration** (`model_migration.py`) - Migrate existing model installations
- **Configuration Backup/Restore** (`config_backup_restore.py`) - Configuration management
- **Monitoring Setup** (`monitoring_setup.py`) - Production monitoring and alerting

## Quick Start

### 1. Basic Deployment

```bash
# Full deployment with all validations
python backend/scripts/deployment/deploy.py

# Dry run to see what would be deployed
python backend/scripts/deployment/deploy.py --dry-run

# Force deployment even with warnings
python backend/scripts/deployment/deploy.py --force
```

### 2. Pre-deployment Validation

```bash
# Validate system readiness
python backend/scripts/deployment/deployment_validator.py

# Check specific component
python backend/scripts/deployment/deployment_validator.py --component enhanced_downloader
```

### 3. Create Rollback Point

```bash
# Create full system backup
python backend/scripts/deployment/rollback_manager.py create --description "Pre-deployment backup"

# Create configuration-only backup
python backend/scripts/deployment/rollback_manager.py create --type configuration_only --description "Config backup"
```

### 4. Model Migration

```bash
# Migrate all existing models
python backend/scripts/deployment/model_migration.py

# Check migration status
python backend/scripts/deployment/model_migration.py --dry-run
```

## Deployment Process

The deployment follows these phases:

1. **Pre-validation** - Verify system requirements and dependencies
2. **Backup Creation** - Create rollback point for safety
3. **Migration** - Migrate existing models to enhanced system
4. **Core Deployment** - Deploy enhanced model availability components
5. **Post-validation** - Verify deployment success
6. **Monitoring Setup** - Configure production monitoring
7. **Health Check** - Final system health verification
8. **Cleanup** - Remove temporary files and optimize system

## Configuration

### Deployment Configuration

Edit `deployment_config.json` to customize deployment behavior:

```json
{
  "deployment_settings": {
    "setup_monitoring": true,
    "cleanup_after_deployment": true,
    "auto_rollback_on_failure": true
  },
  "validation_settings": {
    "require_all_components": true,
    "check_disk_space_gb": 5
  },
  "backup_settings": {
    "create_pre_deployment_backup": true,
    "backup_type": "full_system"
  }
}
```

### Monitoring Configuration

The monitoring system can be configured via `monitoring_config.json`:

```json
{
  "monitoring_interval_seconds": 30,
  "alert_cooldown_minutes": 5,
  "notification_channels": ["console", "log", "file"]
}
```

## Advanced Usage

### Custom Deployment

```bash
# Skip validation (not recommended for production)
python backend/scripts/deployment/deploy.py --skip-validation

# Skip backup creation
python backend/scripts/deployment/deploy.py --skip-backup

# Use custom configuration
python backend/scripts/deployment/deploy.py --config custom_deployment_config.json
```

### Rollback Operations

```bash
# List available rollback points
python backend/scripts/deployment/rollback_manager.py list

# Execute rollback
python backend/scripts/deployment/rollback_manager.py rollback --rollback-id rollback_20240827_143022

# Cleanup old rollback points
python backend/scripts/deployment/rollback_manager.py cleanup --keep 5
```

### Configuration Management

```bash
# Create configuration backup
python backend/scripts/deployment/config_backup_restore.py backup --type configuration_only

# Restore configuration
python backend/scripts/deployment/config_backup_restore.py restore --backup-id config_backup_20240827_143022

# Export backup for external storage
python backend/scripts/deployment/config_backup_restore.py export --backup-id config_backup_20240827_143022 --export-path /external/backup/location
```

### Monitoring Setup

```bash
# Setup monitoring configuration
python backend/scripts/deployment/monitoring_setup.py setup

# Start monitoring for testing
python backend/scripts/deployment/monitoring_setup.py start --duration 300

# Export current metrics
python backend/scripts/deployment/monitoring_setup.py export --format json
```

## Health Check API

The deployment includes health check endpoints for production monitoring:

- `GET /api/v1/deployment/health` - Overall system health
- `GET /api/v1/deployment/health/{component}` - Specific component health
- `GET /api/v1/deployment/validate` - Deployment validation
- `GET /api/v1/deployment/metrics` - Performance metrics
- `GET /api/v1/deployment/readiness` - Kubernetes readiness probe
- `GET /api/v1/deployment/liveness` - Kubernetes liveness probe

## Troubleshooting

### Common Issues

1. **Validation Failures**

   ```bash
   # Check specific validation issues
   python backend/scripts/deployment/deployment_validator.py

   # Fix missing directories
   mkdir -p backend/core backend/api backend/services models
   ```

2. **Migration Issues**

   ```bash
   # Check migration status
   python backend/scripts/deployment/model_migration.py --dry-run

   # View migration logs
   cat backups/model_migration/migration_report.md
   ```

3. **Rollback Issues**

   ```bash
   # Verify rollback point integrity
   python backend/scripts/deployment/rollback_manager.py verify --rollback-id <rollback_id>

   # List available rollback points
   python backend/scripts/deployment/rollback_manager.py list
   ```

4. **Monitoring Issues**

   ```bash
   # Check monitoring status
   python backend/scripts/deployment/monitoring_setup.py status

   # Test monitoring setup
   python backend/scripts/deployment/monitoring_setup.py start --duration 60
   ```

### Log Files

- `logs/deployment.log` - Main deployment log
- `logs/deployment_history.json` - Deployment history
- `logs/alerts.log` - Monitoring alerts
- `backups/rollback_points/rollback_log.json` - Rollback operations
- `backups/configuration/backup_manifests.json` - Configuration backups

### Recovery Procedures

1. **Failed Deployment Recovery**

   ```bash
   # Automatic rollback (if enabled)
   # Manual rollback
   python backend/scripts/deployment/rollback_manager.py rollback --rollback-id <pre_deployment_backup_id>
   ```

2. **Configuration Recovery**

   ```bash
   # Restore from backup
   python backend/scripts/deployment/config_backup_restore.py restore --backup-id <backup_id>
   ```

3. **Model Recovery**
   ```bash
   # Re-run migration
   python backend/scripts/deployment/model_migration.py
   ```

## Testing

Run the deployment system tests:

```bash
# Run all deployment tests
python -m pytest backend/tests/test_deployment_system.py -v

# Run specific test categories
python -m pytest backend/tests/test_deployment_system.py::TestDeploymentValidator -v
python -m pytest backend/tests/test_deployment_system.py::TestRollbackManager -v
```

## Security Considerations

- All backups include integrity checksums
- Rollback points are verified before execution
- Configuration files are validated before restoration
- Deployment logs include audit trails
- Health check endpoints require appropriate authentication in production

## Performance Optimization

- Parallel operations where possible
- Incremental backups for large configurations
- Cached validation results
- Optimized rollback point storage
- Efficient monitoring data collection

## Production Deployment Checklist

- [ ] Review deployment configuration
- [ ] Verify system requirements
- [ ] Create pre-deployment backup
- [ ] Test rollback procedures
- [ ] Configure monitoring and alerting
- [ ] Validate health check endpoints
- [ ] Review security settings
- [ ] Plan maintenance window
- [ ] Prepare rollback plan
- [ ] Test deployment in staging environment

## Support

For issues with the deployment system:

1. Check the troubleshooting section above
2. Review deployment logs
3. Verify system requirements
4. Test individual components
5. Create GitHub issue with deployment logs and configuration
