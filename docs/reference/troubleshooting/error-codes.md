---
title: Error Code Reference
category: reference
tags: [troubleshooting, errors, reference]
last_updated: 2024-01-01
status: published
---

# Error Code Reference

Complete reference for WAN22 error codes and their meanings.

## System Error Codes

### ERR_001 - Configuration File Not Found

**Message**: "Configuration file not found"  
**Cause**: The system cannot locate the required configuration file.  
**Solution**: Ensure the configuration file exists in the expected location.

### ERR_002 - Model Loading Failed

**Message**: "Model loading failed"  
**Cause**: AI model files are missing, corrupted, or incompatible.  
**Solution**: Verify model files and re-download if necessary.

### ERR_003 - Insufficient Memory

**Message**: "Insufficient memory for operation"  
**Cause**: Not enough system memory available for the requested operation.  
**Solution**: Close other applications or upgrade system memory.

## API Error Codes

### API_001 - Invalid Request Format

**HTTP Status**: 400  
**Message**: "Invalid request format"  
**Cause**: Request body does not match expected schema.

### API_002 - Authentication Failed

**HTTP Status**: 401  
**Message**: "Authentication failed"  
**Cause**: Invalid or missing authentication credentials.

### API_003 - Resource Not Found

**HTTP Status**: 404  
**Message**: "Requested resource not found"  
**Cause**: The requested resource does not exist.

## Model Error Codes

### MDL_001 - Model Not Available

**Message**: "Requested model is not available"  
**Cause**: Model is not installed or not properly configured.

### MDL_002 - Model Compatibility Issue

**Message**: "Model compatibility issue detected"  
**Cause**: Model version is incompatible with current system.

## Network Error Codes

### NET_001 - Connection Timeout

**Message**: "Connection timeout"  
**Cause**: Network connection timed out.

### NET_002 - Port Already in Use

**Message**: "Port already in use"  
**Cause**: The required port is already occupied by another process.

## File System Error Codes

### FS_001 - Permission Denied

**Message**: "Permission denied"  
**Cause**: Insufficient file system permissions.

### FS_002 - Disk Space Insufficient

**Message**: "Insufficient disk space"  
**Cause**: Not enough free disk space for the operation.

---

**Last Updated**: 2024-01-01  
**See Also**: [Troubleshooting Guide](../../user-guide/troubleshooting.md)
