---
title: core.enhanced_model_downloader
category: api
tags: [api, core]
---

# core.enhanced_model_downloader

Enhanced Model Downloader with Retry Logic
Provides intelligent retry mechanisms, exponential backoff, partial download recovery,
and advanced download management features for WAN2.2 models.

## Classes

### DownloadStatus

Download status enumeration

### DownloadError

Custom exception for download errors

#### Methods

##### __init__(self: Any, message: str, error_type: str, retry_after: <ast.Subscript object at 0x00000194318D2CE0>)



### DownloadProgress

Enhanced download progress tracking

### DownloadResult

Result of a download operation

### RetryConfig

Configuration for retry logic

### BandwidthConfig

Configuration for bandwidth management

### EnhancedModelDownloader

Enhanced model downloader with intelligent retry mechanisms,
exponential backoff, partial download recovery, and bandwidth management.

#### Methods

##### __init__(self: Any, base_downloader: Any, models_dir: <ast.Subscript object at 0x0000019431926470>)

Initialize the enhanced model downloader.

Args:
    base_downloader: Optional existing ModelDownloader instance
    models_dir: Directory for storing models

##### add_progress_callback(self: Any, callback: <ast.Subscript object at 0x00000194318ECBB0>)

Add a progress callback function

##### remove_progress_callback(self: Any, callback: <ast.Subscript object at 0x00000194318EC850>)

Remove a progress callback function

##### _calculate_chunk_size(self: Any, current_speed_mbps: float) -> int

Calculate optimal chunk size based on current speed

##### set_bandwidth_limit(self: Any, limit_mbps: <ast.Subscript object at 0x00000194300ACA60>) -> bool

Set bandwidth limit for downloads.

Args:
    limit_mbps: Maximum download speed in Mbps, None for unlimited
    
Returns:
    True if successfully set

##### update_retry_config(self: Any)

Update retry configuration

##### update_bandwidth_config(self: Any)

Update bandwidth configuration

## Constants

### PERFORMANCE_MONITORING_AVAILABLE

Type: `bool`

Value: `True`

### QUEUED

Type: `str`

Value: `queued`

### DOWNLOADING

Type: `str`

Value: `downloading`

### PAUSED

Type: `str`

Value: `paused`

### COMPLETED

Type: `str`

Value: `completed`

### FAILED

Type: `str`

Value: `failed`

### CANCELLED

Type: `str`

Value: `cancelled`

### VERIFYING

Type: `str`

Value: `verifying`

### RESUMING

Type: `str`

Value: `resuming`

### PERFORMANCE_MONITORING_AVAILABLE

Type: `bool`

Value: `False`

