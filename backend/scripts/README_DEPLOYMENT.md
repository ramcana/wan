# Deployment and Migration Scripts

This directory contains scripts to help deploy and migrate the real AI model integration system.

## Scripts Overview

### 1. `migrate_to_real_generation.py`

Migrates the system from mock to real AI generation mode.

**Features:**

- Backs up current configuration
- Updates configuration for real generation mode
- Migrates database schema for enhanced tracking
- Validates all system components
- Provides rollback capability on failure

**Usage:**

```bash
cd backend
python scripts/migrate_to_real_generation.py
```

### 2. `deployment_validator.py`

Validates that all components are working correctly after deployment.

**Features:**

- Validates system configuration
- Checks database connectivity and schema
- Tests system integration components
- Validates model management functionality
- Checks API endpoints accessibility
- Verifies performance requirements

**Usage:**

```bash
cd backend
python scripts/deployment_validator.py
```

### 3. `config_migration.py`

Migrates and merges configurations from existing systems.

**Features:**

- Merges WAN2.2 configuration settings
- Integrates local installation configurations
- Preserves existing FastAPI settings
- Validates merged configuration
- Creates backup of existing config

**Usage:**

```bash
cd backend
python scripts/config_migration.py
```

## Deployment Process

Follow these steps for a complete deployment:

### Step 1: Configuration Migration

```bash
# Migrate and merge configurations from existing systems
python scripts/config_migration.py
```

### Step 2: System Migration

```bash
# Migrate from mock to real generation mode
python scripts/migrate_to_real_generation.py
```

### Step 3: Validation

```bash
# Validate the deployment
python scripts/deployment_validator.py
```

### Step 4: Start the System

```bash
# Start the FastAPI server
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

## Configuration Sources

The migration scripts will look for configurations in these locations:

1. **FastAPI Config**: `backend/config.json`
2. **WAN2.2 Config**: `backend/main_config.json`
3. **Local Install Config**: `local_installation/config.json`
4. **Model Config**: `backend/models/model_config.json`

## Migration Features

### Database Schema Updates

The migration adds these new columns to the `generation_tasks` table:

- `model_used`: VARCHAR(100) - Which model was used for generation
- `generation_time_seconds`: FLOAT - Time taken for generation
- `peak_vram_usage_mb`: FLOAT - Peak VRAM usage during generation
- `optimizations_applied`: TEXT - JSON string of applied optimizations
- `error_category`: VARCHAR(50) - Category of any errors that occurred
- `recovery_suggestions`: TEXT - JSON string of recovery suggestions

### Configuration Updates

The migration updates the configuration to enable:

- Real AI model generation (mode: "real")
- Automatic model downloading
- Hardware optimization
- VRAM management
- Detailed WebSocket progress updates

## Validation Checks

The deployment validator performs these checks:

### Configuration Validation

- ✅ Configuration file exists and is valid
- ✅ Required sections are present
- ✅ Real generation mode is enabled
- ✅ Model and hardware settings are configured

### Database Validation

- ✅ Database connectivity
- ✅ Required tables exist
- ✅ New columns are present

### System Integration Validation

- ✅ System components are initialized
- ✅ Model bridge is functional
- ✅ Hardware optimizer is available

### Model Management Validation

- ✅ Model status checking works
- ✅ Model availability can be queried
- ✅ Hardware profile is accessible

### API Validation

- ✅ Main application can be imported
- ✅ API modules are accessible
- ✅ WebSocket manager is available

### Performance Validation

- ✅ Sufficient RAM (≥8GB recommended)
- ✅ Sufficient disk space (≥50GB for models)
- ✅ Adequate CPU cores (≥4 recommended)
- ✅ GPU availability (optional but recommended)

## Troubleshooting

### Common Issues

1. **Configuration Validation Failed**

   - Check that all required configuration sections are present
   - Verify model paths exist and are accessible
   - Ensure database URL is valid

2. **Database Migration Failed**

   - Check database permissions
   - Verify database file is not locked
   - Ensure sufficient disk space

3. **Model Bridge Initialization Failed**

   - Verify existing WAN2.2 infrastructure is available
   - Check model paths in configuration
   - Ensure required Python packages are installed

4. **Performance Requirements Not Met**
   - Free up RAM and disk space
   - Consider using model quantization for lower VRAM usage
   - Upgrade hardware if necessary

### Rollback Procedure

If migration fails, you can rollback:

1. **Configuration Rollback**:

   ```bash
   # Restore from backup (created automatically)
   cp config.backup.json config.json
   ```

2. **Database Rollback**:
   ```bash
   # Remove new columns (if needed)
   sqlite3 generation_tasks.db "ALTER TABLE generation_tasks DROP COLUMN model_used;"
   # Repeat for other new columns
   ```

### Log Files

Check these log files for detailed error information:

- Migration logs: Console output during script execution
- Application logs: `logs/fastapi_backend.log`
- System logs: Check system logs for hardware/GPU issues

## Environment Requirements

### Python Dependencies

```bash
pip install fastapi uvicorn sqlalchemy aiosqlite psutil torch
```

### System Requirements

- **RAM**: 8GB minimum, 16GB+ recommended
- **Disk**: 50GB+ free space for models
- **CPU**: 4+ cores recommended
- **GPU**: CUDA-compatible GPU recommended (optional)

### Optional Dependencies

```bash
# For GPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For advanced monitoring
pip install nvidia-ml-py3 gpustat
```

## Security Considerations

1. **Configuration Security**

   - Update the `secret_key` in configuration
   - Use environment variables for sensitive settings
   - Restrict file permissions on configuration files

2. **API Security**

   - Configure CORS origins appropriately
   - Enable rate limiting in production
   - Use HTTPS in production deployments

3. **Model Security**
   - Verify model file integrity
   - Use trusted model sources
   - Monitor model download sources

## Production Deployment

For production deployment:

1. **Use Environment Variables**:

   ```bash
   export SECRET_KEY="your-production-secret-key"
   export DATABASE_URL="postgresql://user:pass@localhost/db"
   export CORS_ORIGINS="https://yourdomain.com"
   ```

2. **Use Production WSGI Server**:

   ```bash
   gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker
   ```

3. **Set Up Reverse Proxy**:
   Configure nginx or similar for SSL termination and load balancing.

4. **Monitor Resources**:
   Set up monitoring for CPU, RAM, GPU usage and generation performance.

## Support

If you encounter issues:

1. Run the deployment validator to identify problems
2. Check log files for detailed error messages
3. Verify system requirements are met
4. Consult the troubleshooting section above
