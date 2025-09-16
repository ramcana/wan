---
title: core.model_availability_manager
category: api
tags: [api, core]
---

# core.model_availability_manager

Model Availability Manager
Central coordination system for model availability, lifecycle management, and download prioritization.
Integrates with existing ModelManager, EnhancedModelDownloader, and ModelHealthMonitor.

## Classes

### ModelAvailabilityStatus

Enhanced model availability status

### ModelPriority

Model download priority levels

### DetailedModelStatus

Comprehensive model status information

### ModelRequestResult

Result of a model availability request

### CleanupRecommendation

Model cleanup recommendation

### CleanupResult

Result of cleanup operation

### RetentionPolicy

Policy for model retention and cleanup

### ModelAvailabilityManager

Central coordination system for model availability, lifecycle management,
and download prioritization. Integrates with existing ModelManager,
EnhancedModelDownloader, and ModelHealthMonitor.

#### Methods

##### __init__(self: Any, model_manager: <ast.Subscript object at 0x000001942FE3B760>, downloader: <ast.Subscript object at 0x000001942FE3B6A0>, health_monitor: <ast.Subscript object at 0x000001942FE3B5E0>, models_dir: <ast.Subscript object at 0x000001942FE3B520>)

Initialize the Model Availability Manager.

Args:
    model_manager: Existing ModelManager instance
    downloader: Enhanced model downloader instance
    health_monitor: Model health monitor instance
    models_dir: Directory for storing models

##### add_status_callback(self: Any, callback: Callable)

Add a callback for status updates

##### add_download_callback(self: Any, callback: Callable)

Add a callback for download updates

##### _convert_status_to_integrity_result(self: Any, status: DetailedModelStatus) -> IntegrityResult

Convert DetailedModelStatus to IntegrityResult for compatibility

## Constants

### AVAILABLE

Type: `str`

Value: `available`

### DOWNLOADING

Type: `str`

Value: `downloading`

### MISSING

Type: `str`

Value: `missing`

### CORRUPTED

Type: `str`

Value: `corrupted`

### UPDATING

Type: `str`

Value: `updating`

### QUEUED

Type: `str`

Value: `queued`

### PAUSED

Type: `str`

Value: `paused`

### FAILED

Type: `str`

Value: `failed`

### UNKNOWN

Type: `str`

Value: `unknown`

### CRITICAL

Type: `str`

Value: `critical`

### HIGH

Type: `str`

Value: `high`

### MEDIUM

Type: `str`

Value: `medium`

### LOW

Type: `str`

Value: `low`

