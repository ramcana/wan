# Enhanced Model Management API Documentation

## Overview

The Enhanced Model Management API provides comprehensive model status monitoring, download management, health monitoring, usage analytics, storage cleanup, and intelligent fallback suggestions for WAN2.2 video generation models.

## Endpoints

### 1. Get Detailed Model Status

**Endpoint:** `GET /api/v1/models/status/detailed`

**Description:** Get comprehensive model status with enhanced information including availability, health, usage statistics, and download progress.

**Response:**

```json
{
  "models": {
    "T2V-A14B": {
      "model_id": "T2V-A14B",
      "availability_status": "available",
      "is_available": true,
      "is_loaded": false,
      "size_mb": 8192.5,
      "download_progress": null,
      "missing_files": [],
      "integrity_score": 1.0,
      "last_health_check": "2024-01-15T10:30:00",
      "performance_score": 0.95,
      "corruption_detected": false,
      "usage_frequency": 2.5,
      "last_used": "2024-01-15T09:00:00",
      "average_generation_time": 45.2,
      "can_pause_download": false,
      "can_resume_download": false,
      "estimated_download_time": null,
      "current_version": "1.0.0",
      "latest_version": "1.0.0",
      "update_available": false
    }
  },
  "system_statistics": {
    "total_models": 3,
    "available_models": 2,
    "downloading_models": 1,
    "corrupted_models": 0,
    "total_size_gb": 24.5,
    "last_updated": "2024-01-15T10:30:00"
  },
  "timestamp": "2024-01-15T10:30:00"
}
```

### 2. Manage Model Downloads

**Endpoint:** `POST /api/v1/models/download/manage`

**Description:** Manage download operations including pause, resume, cancel, and priority setting.

**Request Body:**

```json
{
  "model_id": "I2V-A14B",
  "action": "pause",
  "priority": "high",
  "bandwidth_limit_mbps": 50.0
}
```

**Parameters:**

- `model_id` (required): Model identifier
- `action` (required): Action to perform - "pause", "resume", "cancel", "priority", "bandwidth"
- `priority` (optional): Priority level for priority action - "critical", "high", "normal", "low"
- `bandwidth_limit_mbps` (optional): Bandwidth limit in Mbps for bandwidth action

**Response:**

```json
{
  "success": true,
  "message": "Download paused for I2V-A14B",
  "model_id": "I2V-A14B",
  "action": "pause",
  "current_status": "paused",
  "progress_percent": 45.5,
  "timestamp": "2024-01-15T10:30:00"
}
```

### 3. Get Health Monitoring Data

**Endpoint:** `GET /api/v1/models/health`

**Description:** Get comprehensive health monitoring data for all models including integrity checks, corruption detection, and system health metrics.

**Response:**

```json
{
  "system_health": {
    "overall_health_score": 0.85,
    "models_healthy": 2,
    "models_degraded": 0,
    "models_corrupted": 0,
    "storage_usage_percent": 65.5,
    "last_updated": "2024-01-15T10:30:00"
  },
  "model_health": {
    "T2V-A14B": {
      "model_id": "T2V-A14B",
      "health_status": "healthy",
      "is_healthy": true,
      "integrity_score": 1.0,
      "issues": [],
      "corruption_types": [],
      "last_check": "2024-01-15T10:30:00",
      "repair_suggestions": [],
      "can_auto_repair": false
    }
  },
  "recommendations": [
    {
      "type": "maintenance",
      "priority": "low",
      "message": "Consider running integrity check on TI2V-5B",
      "action": "check_integrity",
      "model_id": "TI2V-5B"
    }
  ],
  "timestamp": "2024-01-15T10:30:00"
}
```

### 4. Get Usage Analytics

**Endpoint:** `GET /api/v1/models/analytics?time_period_days=30`

**Description:** Get usage analytics and statistics for all models over a specified time period.

**Query Parameters:**

- `time_period_days` (optional, default: 30): Number of days to analyze

**Response:**

```json
{
  "time_period": {
    "start_date": "2023-12-16T10:30:00",
    "end_date": "2024-01-15T10:30:00",
    "days": 30
  },
  "system_analytics": {
    "total_uses": 150,
    "average_uses_per_day": 5.0,
    "most_used_model": "T2V-A14B",
    "active_models": 2
  },
  "model_analytics": {
    "T2V-A14B": {
      "model_id": "T2V-A14B",
      "total_uses": 100,
      "uses_per_day": 3.33,
      "average_generation_time": 45.2,
      "success_rate": 0.96,
      "last_30_days_usage": [
        {
          "date": "2024-01-15",
          "uses": 5,
          "avg_time": 44.1,
          "success_rate": 1.0
        }
      ],
      "peak_usage_hours": [14, 15, 16]
    }
  },
  "timestamp": "2024-01-15T10:30:00"
}
```

### 5. Manage Storage Cleanup

**Endpoint:** `POST /api/v1/models/cleanup`

**Description:** Manage storage cleanup operations with recommendations and execution options.

**Request Body:**

```json
{
  "target_space_gb": 20.0,
  "keep_recent_days": 30,
  "dry_run": true
}
```

**Parameters:**

- `target_space_gb` (optional): Target free space in GB
- `keep_recent_days` (optional, default: 30): Keep models used in last N days
- `dry_run` (optional, default: true): Preview cleanup without executing

**Response:**

```json
{
  "dry_run": true,
  "recommendations": {
    "total_space_available_gb": 50.0,
    "target_space_gb": 20.0,
    "space_to_free_gb": 10.0,
    "cleanup_actions": [
      {
        "action_type": "remove_model",
        "model_id": "TI2V-5B",
        "space_freed_gb": 6.0,
        "reason": "Unused for 60 days",
        "last_used": "2023-11-15T10:30:00"
      }
    ]
  },
  "executed_actions": [],
  "total_space_freed_gb": 0.0,
  "timestamp": "2024-01-15T10:30:00"
}
```

### 6. Suggest Fallback Alternatives

**Endpoint:** `POST /api/v1/models/fallback/suggest`

**Description:** Suggest alternative models and fallback strategies when requested models are unavailable.

**Request Body:**

```json
{
  "requested_model": "T2V-A14B",
  "quality": "high",
  "speed": "medium",
  "resolution": "1280x720",
  "max_wait_minutes": 30
}
```

**Parameters:**

- `requested_model` (required): Originally requested model
- `quality` (optional, default: "medium"): Quality requirement - "low", "medium", "high"
- `speed` (optional, default: "medium"): Speed requirement - "fast", "medium", "slow"
- `resolution` (optional, default: "1280x720"): Target resolution
- `max_wait_minutes` (optional): Maximum acceptable wait time in minutes

**Response:**

```json
{
  "requested_model": "T2V-A14B",
  "alternative_suggestion": {
    "suggested_model": "I2V-A14B",
    "compatibility_score": 0.85,
    "performance_difference": -0.1,
    "availability_status": "available",
    "reason": "Similar capabilities with image input support",
    "estimated_quality_difference": "slightly_lower"
  },
  "fallback_strategy": {
    "strategy_type": "alternative_model",
    "recommended_action": "Use I2V-A14B as alternative",
    "alternative_model": "I2V-A14B",
    "estimated_wait_time": null,
    "user_message": "I2V-A14B is available and provides similar functionality",
    "can_queue_request": false
  },
  "wait_time_estimate": null,
  "timestamp": "2024-01-15T10:30:00"
}
```

## Error Responses

All endpoints return standard HTTP error responses:

### 400 Bad Request

```json
{
  "detail": "Invalid request parameters"
}
```

### 404 Not Found

```json
{
  "detail": "Model not found"
}
```

### 422 Unprocessable Entity

```json
{
  "detail": "Validation error: missing required field"
}
```

### 500 Internal Server Error

```json
{
  "detail": "Failed to process request: internal error"
}
```

## Status Codes

- **200 OK**: Request successful
- **400 Bad Request**: Invalid request parameters
- **404 Not Found**: Resource not found
- **422 Unprocessable Entity**: Validation error
- **500 Internal Server Error**: Server error

## Model Availability Status Values

- `available`: Model is fully downloaded and ready
- `downloading`: Model is currently being downloaded
- `missing`: Model is not available locally
- `corrupted`: Model files are corrupted
- `updating`: Model is being updated to newer version
- `queued`: Model download is queued
- `paused`: Model download is paused
- `failed`: Model download failed
- `unknown`: Status cannot be determined

## Health Status Values

- `healthy`: Model is functioning normally
- `degraded`: Model has minor issues but is functional
- `corrupted`: Model has serious integrity issues
- `missing`: Model files are missing
- `unknown`: Health status cannot be determined

## Fallback Strategy Types

- `alternative_model`: Use a different available model
- `queue_and_wait`: Queue request and wait for model download
- `mock_generation`: Use mock generation as fallback
- `download_and_retry`: Download model and retry request
- `reduce_requirements`: Reduce quality/resolution requirements
- `hybrid_approach`: Combination of multiple strategies

## Usage Examples

### Check Model Status

```bash
curl -X GET "http://localhost:8000/api/v1/models/status/detailed"
```

### Pause Model Download

```bash
curl -X POST "http://localhost:8000/api/v1/models/download/manage" \
  -H "Content-Type: application/json" \
  -d '{"model_id": "I2V-A14B", "action": "pause"}'
```

### Get Health Data

```bash
curl -X GET "http://localhost:8000/api/v1/models/health"
```

### Get Analytics for Last 7 Days

```bash
curl -X GET "http://localhost:8000/api/v1/models/analytics?time_period_days=7"
```

### Preview Cleanup

```bash
curl -X POST "http://localhost:8000/api/v1/models/cleanup" \
  -H "Content-Type: application/json" \
  -d '{"target_space_gb": 20.0, "dry_run": true}'
```

### Get Fallback Suggestions

```bash
curl -X POST "http://localhost:8000/api/v1/models/fallback/suggest" \
  -H "Content-Type: application/json" \
  -d '{"requested_model": "T2V-A14B", "quality": "high"}'
```
