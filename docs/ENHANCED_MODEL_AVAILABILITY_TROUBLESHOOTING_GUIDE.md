# Enhanced Model Availability - Troubleshooting Guide

## Overview

This guide provides solutions for common issues with the Enhanced Model Availability system, including download failures, model corruption, performance problems, and configuration issues.

## Quick Diagnostic Steps

### 1. System Health Check

```bash
# Check overall system status
curl http://localhost:8000/api/v1/models/health

# Get detailed model status
curl http://localhost:8000/api/v1/models/status/detailed

# Check recent logs
tail -f logs/model_management.log
```

### 2. Basic Troubleshooting Checklist

- [ ] Internet connection is stable
- [ ] Sufficient disk space available (at least 10GB free)
- [ ] No antivirus blocking downloads
- [ ] System resources not overloaded
- [ ] Model management service is running

## Download Issues

### Download Keeps Failing

**Symptoms**:

- Downloads fail repeatedly
- Error messages about network timeouts
- Partial downloads that don't resume

**Diagnostic Steps**:

1. Check network connectivity:

   ```bash
   ping huggingface.co
   curl -I https://huggingface.co/models
   ```

2. Verify disk space:

   ```bash
   df -h
   # Ensure at least 10GB free space
   ```

3. Check download logs:
   ```bash
   tail -100 logs/downloads.log | grep ERROR
   ```

**Solutions**:

**Network Issues**:

- Switch to a more stable internet connection
- Try downloading during off-peak hours
- Configure proxy settings if behind corporate firewall
- Increase timeout settings in configuration

**Disk Space Issues**:

- Free up disk space using cleanup recommendations
- Move existing models to external storage
- Configure models directory to different drive

**Permission Issues**:

- Check write permissions on models directory
- Run with appropriate user privileges
- Verify antivirus isn't blocking file creation

### Download Stuck at 0%

**Symptoms**:

- Download shows as started but no progress
- Status remains "DOWNLOADING" indefinitely
- No network activity visible

**Diagnostic Steps**:

1. Check download status:

   ```bash
   curl http://localhost:8000/api/v1/models/download/status/MODEL_ID
   ```

2. Verify model exists:
   ```bash
   curl -I https://huggingface.co/MODEL_ID/resolve/main/config.json
   ```

**Solutions**:

- Cancel and restart download
- Clear download cache
- Verify model ID is correct
- Check if model requires authentication

### Partial Downloads Not Resuming

**Symptoms**:

- Downloads restart from beginning after interruption
- Resume functionality not working
- Wasted bandwidth on repeated downloads

**Diagnostic Steps**:

1. Check partial file existence:

   ```bash
   ls -la models/MODEL_ID/
   ```

2. Verify resume capability:
   ```bash
   curl http://localhost:8000/api/v1/models/download/manage \
     -X POST -d '{"model_id": "MODEL_ID", "action": "resume"}'
   ```

**Solutions**:

- Enable resume functionality in settings
- Clear corrupted partial files
- Restart download with resume enabled
- Check file system supports partial files

## Model Corruption Issues

### Model Files Corrupted

**Symptoms**:

- Generation produces poor quality results
- Model loading fails with checksum errors
- Integrity checks fail

**Diagnostic Steps**:

1. Run integrity check:

   ```bash
   curl http://localhost:8000/api/v1/models/health/MODEL_ID
   ```

2. Check file sizes:

   ```bash
   ls -la models/MODEL_ID/
   # Compare with expected sizes
   ```

3. Verify checksums:
   ```bash
   # System automatically verifies, but manual check:
   sha256sum models/MODEL_ID/*.bin
   ```

**Solutions**:

**Automatic Repair**:

- Use built-in repair function
- System will re-download corrupted files
- Monitor repair progress in logs

**Manual Repair**:

1. Delete corrupted files:

   ```bash
   rm models/MODEL_ID/corrupted_file.bin
   ```

2. Trigger re-download:
   ```bash
   curl http://localhost:8000/api/v1/models/download/manage \
     -X POST -d '{"model_id": "MODEL_ID", "action": "repair"}'
   ```

**Prevention**:

- Enable automatic integrity checking
- Use stable storage (avoid network drives)
- Ensure adequate power supply
- Regular system health checks

### Model Loading Failures

**Symptoms**:

- Models fail to load despite being downloaded
- Error messages about missing files
- Generation requests timeout

**Diagnostic Steps**:

1. Check model completeness:

   ```bash
   curl http://localhost:8000/api/v1/models/status/detailed/MODEL_ID
   ```

2. Verify required files:

   ```bash
   ls models/MODEL_ID/
   # Should contain: config.json, model files, tokenizer files
   ```

3. Check loading logs:
   ```bash
   grep "MODEL_ID" logs/model_management.log | tail -20
   ```

**Solutions**:

- Re-download missing files
- Clear model cache and reload
- Check model compatibility with system
- Verify sufficient VRAM/RAM available

## Performance Issues

### Slow Model Downloads

**Symptoms**:

- Downloads much slower than expected
- Bandwidth not fully utilized
- Downloads taking hours for small models

**Diagnostic Steps**:

1. Check download speed:

   ```bash
   curl http://localhost:8000/api/v1/models/download/progress/MODEL_ID
   ```

2. Test network speed:

   ```bash
   speedtest-cli
   # Or use online speed test
   ```

3. Check bandwidth limits:
   ```bash
   curl http://localhost:8000/api/v1/config/download_settings
   ```

**Solutions**:

- Remove or increase bandwidth limits
- Close other network-intensive applications
- Try different download servers if available
- Schedule downloads during off-peak hours

### Model Generation Performance Degraded

**Symptoms**:

- Generation takes much longer than before
- Quality of outputs decreased
- Frequent generation failures

**Diagnostic Steps**:

1. Check model health:

   ```bash
   curl http://localhost:8000/api/v1/models/health/MODEL_ID
   ```

2. Review performance metrics:

   ```bash
   curl http://localhost:8000/api/v1/models/analytics/performance/MODEL_ID
   ```

3. Check system resources:
   ```bash
   nvidia-smi  # For GPU usage
   htop        # For CPU/RAM usage
   ```

**Solutions**:

- Run model integrity check and repair
- Update to latest model version
- Clear model cache and reload
- Check for system resource constraints
- Consider using alternative model temporarily

## Storage and Space Issues

### Insufficient Disk Space

**Symptoms**:

- Downloads fail with "No space left" errors
- System becomes unresponsive
- Cannot save new models

**Diagnostic Steps**:

1. Check available space:

   ```bash
   df -h
   du -sh models/
   ```

2. Identify large unused models:
   ```bash
   curl http://localhost:8000/api/v1/models/analytics/usage
   ```

**Solutions**:

**Immediate Actions**:

- Use cleanup recommendations to remove unused models
- Move infrequently used models to external storage
- Clear temporary files and caches

**Long-term Solutions**:

- Configure automatic cleanup policies
- Set up external storage for model archive
- Monitor storage usage regularly
- Implement storage quotas

### Models Directory Issues

**Symptoms**:

- Cannot find downloaded models
- Models appear downloaded but not accessible
- Permission denied errors

**Diagnostic Steps**:

1. Check models directory:

   ```bash
   ls -la models/
   ```

2. Verify permissions:

   ```bash
   ls -ld models/
   # Should be writable by application user
   ```

3. Check configuration:
   ```bash
   curl http://localhost:8000/api/v1/config/storage_settings
   ```

**Solutions**:

- Fix directory permissions
- Verify models directory path in configuration
- Create missing directories
- Check disk mount status

## Configuration Issues

### Settings Not Applied

**Symptoms**:

- Configuration changes don't take effect
- System uses default settings despite changes
- Settings revert after restart

**Diagnostic Steps**:

1. Check current configuration:

   ```bash
   curl http://localhost:8000/api/v1/config/current
   ```

2. Verify configuration file:

   ```bash
   cat config/enhanced_model_config.json
   ```

3. Check for configuration errors:
   ```bash
   grep "config" logs/model_management.log | grep ERROR
   ```

**Solutions**:

- Restart service after configuration changes
- Validate configuration file syntax
- Check file permissions on configuration files
- Use configuration validation endpoint

### Feature Flags Not Working

**Symptoms**:

- Enhanced features not available
- Fallback to basic functionality
- Missing UI elements

**Diagnostic Steps**:

1. Check feature flag status:

   ```bash
   curl http://localhost:8000/api/v1/config/features
   ```

2. Verify license/permissions:
   ```bash
   curl http://localhost:8000/api/v1/system/capabilities
   ```

**Solutions**:

- Enable required feature flags
- Check license validity
- Restart service after flag changes
- Verify user permissions

## API and Integration Issues

### API Endpoints Not Responding

**Symptoms**:

- 404 errors on model management endpoints
- Timeout errors on API calls
- Inconsistent API responses

**Diagnostic Steps**:

1. Check API availability:

   ```bash
   curl http://localhost:8000/api/v1/health
   ```

2. Verify endpoint paths:

   ```bash
   curl http://localhost:8000/docs
   # Check OpenAPI documentation
   ```

3. Check service logs:
   ```bash
   tail -f logs/api.log
   ```

**Solutions**:

- Verify correct API endpoint URLs
- Check service is running and healthy
- Validate request format and parameters
- Review API documentation for changes

### WebSocket Connection Issues

**Symptoms**:

- No real-time updates
- WebSocket connection drops frequently
- Missing progress notifications

**Diagnostic Steps**:

1. Test WebSocket connection:

   ```javascript
   const ws = new WebSocket("ws://localhost:8000/ws");
   ws.onopen = () => console.log("Connected");
   ws.onerror = (error) => console.log("Error:", error);
   ```

2. Check WebSocket logs:
   ```bash
   grep "websocket" logs/model_management.log
   ```

**Solutions**:

- Check firewall settings for WebSocket ports
- Verify WebSocket endpoint configuration
- Test with different browsers/clients
- Check for proxy interference

## System Integration Issues

### Service Dependencies

**Symptoms**:

- Enhanced features partially working
- Inconsistent behavior across features
- Service startup failures

**Diagnostic Steps**:

1. Check service status:

   ```bash
   curl http://localhost:8000/api/v1/system/status
   ```

2. Verify dependencies:

   ```bash
   pip list | grep -E "(torch|transformers|huggingface)"
   ```

3. Check integration logs:
   ```bash
   grep "integration" logs/model_management.log
   ```

**Solutions**:

- Update dependencies to compatible versions
- Restart services in correct order
- Check configuration compatibility
- Verify system requirements

## Advanced Troubleshooting

### Debug Mode

Enable debug logging for detailed troubleshooting:

1. Update configuration:

   ```json
   {
     "logging": {
       "level": "DEBUG",
       "detailed_model_operations": true
     }
   }
   ```

2. Restart service and monitor logs:
   ```bash
   tail -f logs/model_management.log | grep DEBUG
   ```

### Performance Profiling

For performance issues, enable profiling:

1. Enable performance monitoring:

   ```bash
   curl http://localhost:8000/api/v1/config/performance \
     -X POST -d '{"enable_profiling": true}'
   ```

2. Generate performance report:
   ```bash
   curl http://localhost:8000/api/v1/models/performance/report
   ```

### Database Issues

If using analytics database:

1. Check database connectivity:

   ```bash
   curl http://localhost:8000/api/v1/analytics/health
   ```

2. Reset analytics database:
   ```bash
   curl http://localhost:8000/api/v1/analytics/reset \
     -X POST -d '{"confirm": true}'
   ```

## Getting Additional Help

### Log Collection

When contacting support, collect these logs:

```bash
# Create support bundle
tar -czf support_logs.tar.gz \
  logs/model_management.log \
  logs/downloads.log \
  logs/health_monitor.log \
  config/enhanced_model_config.json \
  models/*/status.json
```

### System Information

Include system information:

```bash
# System specs
uname -a
python --version
pip list > installed_packages.txt
nvidia-smi > gpu_info.txt
df -h > disk_usage.txt
```

### Support Channels

- **Documentation**: Check user guide and API docs
- **Community**: Search existing issues and discussions
- **Support**: Contact with logs and system information
- **Emergency**: Use fallback to basic functionality

## Prevention and Maintenance

### Regular Maintenance Tasks

1. **Weekly**:

   - Review model usage analytics
   - Clean up unused models
   - Check system health reports

2. **Monthly**:

   - Update models to latest versions
   - Review and optimize configuration
   - Backup important models

3. **Quarterly**:
   - Full system health audit
   - Performance optimization review
   - Update system dependencies

### Monitoring Setup

Set up monitoring for:

- Disk space usage
- Download success rates
- Model health scores
- API response times
- System resource usage

This proactive approach helps prevent issues before they impact users.
