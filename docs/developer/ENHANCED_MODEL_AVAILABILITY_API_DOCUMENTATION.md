---
category: developer
last_updated: '2025-09-15T22:49:59.921342'
original_path: docs\ENHANCED_MODEL_AVAILABILITY_API_DOCUMENTATION.md
tags:
- configuration
- api
- troubleshooting
- installation
- security
- performance
title: Enhanced Model Availability - API Documentation
---

# Enhanced Model Availability - API Documentation

## Overview

This document provides comprehensive API documentation for the Enhanced Model Availability system, including all endpoints, request/response formats, authentication, and integration examples.

## Base URL

```
http://localhost:8000/api/v1
```

## Authentication

Currently, the API supports optional authentication. When enabled, include the API key in the header:

```http
Authorization: Bearer YOUR_API_KEY
```

## Content Types

All API endpoints accept and return JSON unless otherwise specified.

```http
Content-Type: application/json
Accept: application/json
```

## Error Handling

### Standard Error Response

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human readable error message",
    "details": {
      "field": "Additional error details"
    },
    "timestamp": "2024-01-01T12:00:00Z",
    "request_id": "req_123456789"
  }
}
```

### HTTP Status Codes

- `200` - Success
- `201` - Created
- `400` - Bad Request
- `401` - Unauthorized
- `403` - Forbidden
- `404` - Not Found
- `409` - Conflict
- `422` - Validation Error
- `429` - Rate Limited
- `500` - Internal Server Error
- `503` - Service Unavailable

## Model Status Endpoints

### Get Detailed Model Status

Get comprehensive status information for all models.

```http
GET /models/status/detailed
```

**Response:**

```json
{
  "models": {
    "model-id": {
      "model_id": "model-id",
      "is_available": true,
      "is_loaded": false,
      "size_mb": 1024.5,
      "availability_status": "AVAILABLE",
      "download_progress": null,
      "missing_files": [],
      "integrity_score": 1.0,
      "last_health_check": "2024-01-01T12:00:00Z",
      "performance_score": 0.95,
      "corruption_detected": false,
      "usage_frequency": 5.2,
      "last_used": "2024-01-01T11:30:00Z",
      "average_generation_time": 2.5,
      "can_pause_download": false,
      "can_resume_download": false,
      "estimated_download_time": null,
      "current_version": "1.0.0",
      "latest_version": "1.0.0",
      "update_available": false
    }
  },
  "summary": {
    "total_models": 5,
    "available_models": 4,
    "downloading_models": 1,
    "corrupted_models": 0,
    "total_storage_gb": 25.6
  }
}
```

### Get Single Model Status

Get detailed status for a specific model.

```http
GET /models/status/detailed/{model_id}
```

**Parameters:**

- `model_id` (path): The model identifier

**Response:**

```json
{
  "model_id": "model-id",
  "is_available": true,
  "is_loaded": false,
  "size_mb": 1024.5,
  "availability_status": "AVAILABLE",
  "download_progress": null,
  "missing_files": [],
  "integrity_score": 1.0,
  "last_health_check": "2024-01-01T12:00:00Z",
  "performance_score": 0.95,
  "corruption_detected": false,
  "usage_frequency": 5.2,
  "last_used": "2024-01-01T11:30:00Z",
  "average_generation_time": 2.5,
  "can_pause_download": false,
  "can_resume_download": false,
  "estimated_download_time": null,
  "current_version": "1.0.0",
  "latest_version": "1.0.0",
  "update_available": false
}
```

## Download Management Endpoints

### Start Model Download

Initiate download for a specific model with retry logic.

```http
POST /models/download/manage
```

**Request Body:**

```json
{
  "model_id": "model-id",
  "action": "start",
  "max_retries": 3,
  "priority": "normal",
  "bandwidth_limit_mbps": 0
}
```

**Response:**

```json
{
  "success": true,
  "message": "Download started successfully",
  "download_id": "dl_123456789",
  "estimated_completion": "2024-01-01T12:30:00Z"
}
```

### Control Download

Pause, resume, or cancel an active download.

```http
POST /models/download/manage
```

**Request Body:**

```json
{
  "model_id": "model-id",
  "action": "pause|resume|cancel"
}
```

**Response:**

```json
{
  "success": true,
  "message": "Download paused successfully",
  "new_status": "PAUSED"
}
```

### Get Download Progress

Get real-time download progress for a model.

```http
GET /models/download/progress/{model_id}
```

**Response:**

```json
{
  "model_id": "model-id",
  "status": "DOWNLOADING",
  "progress_percent": 45.2,
  "downloaded_mb": 462.8,
  "total_mb": 1024.0,
  "speed_mbps": 12.5,
  "eta_seconds": 45,
  "retry_count": 1,
  "max_retries": 3,
  "error_message": null,
  "can_pause": true,
  "can_resume": true,
  "can_cancel": true
}
```

### List Active Downloads

Get all currently active downloads.

```http
GET /models/download/active
```

**Response:**

```json
{
  "active_downloads": [
    {
      "model_id": "model-1",
      "status": "DOWNLOADING",
      "progress_percent": 45.2,
      "speed_mbps": 12.5,
      "eta_seconds": 45
    },
    {
      "model_id": "model-2",
      "status": "QUEUED",
      "progress_percent": 0,
      "queue_position": 2
    }
  ],
  "total_active": 2,
  "total_bandwidth_usage_mbps": 12.5
}
```

## Model Health Endpoints

### Get Model Health Status

Get health information for all models or a specific model.

```http
GET /models/health
GET /models/health/{model_id}
```

**Response:**

```json
{
  "models": {
    "model-id": {
      "model_id": "model-id",
      "overall_health": "healthy",
      "integrity_score": 1.0,
      "performance_score": 0.95,
      "last_check": "2024-01-01T12:00:00Z",
      "issues": [],
      "recommendations": []
    }
  },
  "system_health": {
    "overall_score": 0.97,
    "healthy_models": 4,
    "degraded_models": 1,
    "corrupted_models": 0,
    "last_system_check": "2024-01-01T12:00:00Z"
  }
}
```

### Trigger Health Check

Manually trigger health check for a model.

```http
POST /models/health/{model_id}/check
```

**Request Body:**

```json
{
  "check_type": "full|integrity|performance",
  "force_check": false
}
```

**Response:**

```json
{
  "success": true,
  "check_id": "check_123456789",
  "estimated_completion": "2024-01-01T12:05:00Z"
}
```

### Repair Model

Attempt to repair a corrupted or degraded model.

```http
POST /models/health/{model_id}/repair
```

**Request Body:**

```json
{
  "repair_type": "auto|redownload|integrity_fix",
  "force_repair": false
}
```

**Response:**

```json
{
  "success": true,
  "repair_id": "repair_123456789",
  "actions_taken": ["integrity_check", "partial_redownload"],
  "estimated_completion": "2024-01-01T12:15:00Z"
}
```

## Fallback and Suggestions Endpoints

### Get Model Suggestions

Get alternative model suggestions when preferred model is unavailable.

```http
POST /models/fallback/suggest
```

**Request Body:**

```json
{
  "requested_model": "unavailable-model",
  "requirements": {
    "quality": "high",
    "speed": "medium",
    "model_type": "text-to-video"
  },
  "user_preferences": {
    "prefer_smaller_models": false,
    "max_wait_time_minutes": 30
  }
}
```

**Response:**

```json
{
  "suggestions": [
    {
      "suggested_model": "alternative-model-1",
      "compatibility_score": 0.85,
      "performance_difference": -0.1,
      "availability_status": "AVAILABLE",
      "reason": "Similar architecture and performance",
      "estimated_quality_difference": "slightly_lower"
    },
    {
      "suggested_model": "alternative-model-2",
      "compatibility_score": 0.75,
      "performance_difference": 0.05,
      "availability_status": "DOWNLOADING",
      "reason": "Better performance, currently downloading",
      "estimated_quality_difference": "similar"
    }
  ],
  "fallback_strategy": {
    "strategy_type": "ALTERNATIVE_MODEL",
    "recommended_action": "Use alternative-model-1 immediately",
    "estimated_wait_time": null,
    "user_message": "Alternative model available with similar quality",
    "can_queue_request": true
  }
}
```

### Queue Request

Queue a generation request while model downloads.

```http
POST /models/fallback/queue
```

**Request Body:**

```json
{
  "model_id": "downloading-model",
  "request": {
    "prompt": "Generate a video of...",
    "parameters": {
      "resolution": "1024x576",
      "duration": 5
    }
  },
  "priority": "normal",
  "max_wait_time_minutes": 60
}
```

**Response:**

```json
{
  "success": true,
  "queue_id": "queue_123456789",
  "position": 3,
  "estimated_wait_time_minutes": 15,
  "estimated_start_time": "2024-01-01T12:15:00Z"
}
```

## Analytics Endpoints

### Get Usage Statistics

Get model usage analytics and statistics.

```http
GET /models/analytics
GET /models/analytics/{model_id}
```

**Query Parameters:**

- `period` (optional): `day|week|month|year`
- `start_date` (optional): ISO date string
- `end_date` (optional): ISO date string

**Response:**

```json
{
  "models": {
    "model-id": {
      "model_id": "model-id",
      "total_uses": 150,
      "uses_per_day": 5.2,
      "average_generation_time": 2.5,
      "success_rate": 0.98,
      "last_30_days_usage": [
        { "date": "2024-01-01", "uses": 8, "avg_time": 2.3 },
        { "date": "2024-01-02", "uses": 12, "avg_time": 2.7 }
      ],
      "peak_usage_hours": [14, 15, 16, 20, 21]
    }
  },
  "summary": {
    "total_generations": 1250,
    "total_models_used": 5,
    "average_success_rate": 0.96,
    "most_used_model": "model-1",
    "least_used_model": "model-5"
  }
}
```

### Get Cleanup Recommendations

Get recommendations for model cleanup based on usage patterns.

```http
GET /models/analytics/cleanup
```

**Query Parameters:**

- `storage_threshold_gb` (optional): Trigger cleanup when storage exceeds this
- `min_unused_days` (optional): Consider models unused after this many days

**Response:**

```json
{
  "recommendations": [
    {
      "model_id": "unused-model",
      "reason": "Not used in 45 days",
      "last_used": "2023-11-15T10:30:00Z",
      "size_mb": 2048,
      "usage_count": 2,
      "confidence": 0.9,
      "action": "safe_to_remove"
    }
  ],
  "potential_savings_gb": 15.6,
  "current_usage_gb": 45.2,
  "target_usage_gb": 29.6
}
```

### Get Performance Analytics

Get performance analytics and trends.

```http
GET /models/analytics/performance
```

**Response:**

```json
{
  "performance_trends": {
    "model-id": {
      "current_score": 0.95,
      "trend": "stable",
      "last_7_days_avg": 0.94,
      "last_30_days_avg": 0.96,
      "performance_issues": [],
      "recommendations": ["Consider updating to latest version"]
    }
  },
  "system_performance": {
    "overall_score": 0.93,
    "download_success_rate": 0.98,
    "average_download_speed_mbps": 25.6,
    "health_check_success_rate": 1.0
  }
}
```

## Storage Management Endpoints

### Get Storage Information

Get detailed storage usage information.

```http
GET /models/storage
```

**Response:**

```json
{
  "storage_summary": {
    "total_capacity_gb": 500,
    "used_space_gb": 125.6,
    "available_space_gb": 374.4,
    "usage_percent": 25.1,
    "models_count": 8
  },
  "models_storage": [
    {
      "model_id": "large-model",
      "size_gb": 15.2,
      "last_accessed": "2024-01-01T10:30:00Z",
      "usage_frequency": 8.5,
      "storage_tier": "hot"
    }
  ],
  "cleanup_suggestions": {
    "potential_savings_gb": 25.6,
    "safe_to_remove_count": 3,
    "candidates": ["unused-model-1", "unused-model-2"]
  }
}
```

### Cleanup Models

Remove unused models to free up storage space.

```http
POST /models/cleanup
```

**Request Body:**

```json
{
  "cleanup_type": "auto|manual",
  "models_to_remove": ["model-1", "model-2"],
  "min_unused_days": 30,
  "preserve_frequently_used": true,
  "dry_run": false
}
```

**Response:**

```json
{
  "success": true,
  "cleanup_id": "cleanup_123456789",
  "models_removed": ["unused-model-1"],
  "space_freed_gb": 12.5,
  "models_preserved": ["frequently-used-model"],
  "summary": {
    "total_processed": 5,
    "removed": 1,
    "preserved": 4,
    "errors": 0
  }
}
```

## Configuration Endpoints

### Get Current Configuration

Get current system configuration.

```http
GET /config/current
```

**Response:**

```json
{
  "storage": {
    "models_directory": "/data/models",
    "max_storage_gb": 500,
    "cleanup_threshold_percent": 90
  },
  "downloads": {
    "max_concurrent_downloads": 3,
    "max_retries": 3,
    "bandwidth_limit_mbps": 0
  },
  "features": {
    "enhanced_downloads": true,
    "health_monitoring": true,
    "intelligent_fallback": true,
    "usage_analytics": true
  }
}
```

### Update Configuration

Update system configuration (admin only).

```http
PUT /config/update
```

**Request Body:**

```json
{
  "downloads": {
    "max_concurrent_downloads": 5,
    "bandwidth_limit_mbps": 50
  },
  "health_monitoring": {
    "check_interval_hours": 12,
    "auto_repair_enabled": true
  }
}
```

**Response:**

```json
{
  "success": true,
  "updated_settings": [
    "downloads.max_concurrent_downloads",
    "downloads.bandwidth_limit_mbps",
    "health_monitoring.check_interval_hours"
  ],
  "restart_required": false
}
```

## WebSocket API

### Connection

Connect to WebSocket for real-time updates:

```javascript
const ws = new WebSocket("ws://localhost:8000/ws");
```

### Message Types

#### Download Progress Updates

```json
{
  "type": "model_download_progress",
  "model_id": "model-id",
  "progress_percent": 45.2,
  "speed_mbps": 12.5,
  "eta_seconds": 45,
  "status": "DOWNLOADING"
}
```

#### Health Monitoring Alerts

```json
{
  "type": "model_health_alert",
  "model_id": "model-id",
  "alert_type": "corruption_detected",
  "severity": "warning",
  "message": "Corruption detected in model files",
  "recommended_action": "auto_repair",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

#### Model Availability Changes

```json
{
  "type": "model_availability_change",
  "model_id": "model-id",
  "old_status": "DOWNLOADING",
  "new_status": "AVAILABLE",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

#### Fallback Notifications

```json
{
  "type": "fallback_suggestion",
  "requested_model": "unavailable-model",
  "suggested_model": "alternative-model",
  "compatibility_score": 0.85,
  "user_action_required": true,
  "timeout_seconds": 30
}
```

## Rate Limiting

API endpoints are rate limited to prevent abuse:

- **Default**: 60 requests per minute per IP
- **Download endpoints**: 10 requests per minute per IP
- **Admin endpoints**: 30 requests per minute per authenticated user

Rate limit headers are included in responses:

```http
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1640995200
```

## Pagination

List endpoints support pagination:

**Query Parameters:**

- `page` (default: 1): Page number
- `limit` (default: 20, max: 100): Items per page
- `sort` (optional): Sort field
- `order` (default: asc): Sort order (asc|desc)

**Response Format:**

```json
{
  "data": [...],
  "pagination": {
    "page": 1,
    "limit": 20,
    "total": 150,
    "pages": 8,
    "has_next": true,
    "has_prev": false
  }
}
```

## SDK Examples

### Python SDK

```python
import requests
from typing import Dict, List, Optional

class EnhancedModelAvailabilityClient:
    def __init__(self, base_url: str = "http://localhost:8000/api/v1", api_key: Optional[str] = None):
        self.base_url = base_url
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def get_model_status(self, model_id: Optional[str] = None) -> Dict:
        """Get model status for all models or specific model."""
        url = f"{self.base_url}/models/status/detailed"
        if model_id:
            url += f"/{model_id}"

        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def start_download(self, model_id: str, max_retries: int = 3, priority: str = "normal") -> Dict:
        """Start model download with retry logic."""
        url = f"{self.base_url}/models/download/manage"
        data = {
            "model_id": model_id,
            "action": "start",
            "max_retries": max_retries,
            "priority": priority
        }

        response = requests.post(url, json=data, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def get_download_progress(self, model_id: str) -> Dict:
        """Get download progress for a model."""
        url = f"{self.base_url}/models/download/progress/{model_id}"

        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def get_model_suggestions(self, requested_model: str, requirements: Dict) -> Dict:
        """Get alternative model suggestions."""
        url = f"{self.base_url}/models/fallback/suggest"
        data = {
            "requested_model": requested_model,
            "requirements": requirements
        }

        response = requests.post(url, json=data, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def get_usage_analytics(self, model_id: Optional[str] = None, period: str = "week") -> Dict:
        """Get usage analytics."""
        url = f"{self.base_url}/models/analytics"
        if model_id:
            url += f"/{model_id}"

        params = {"period": period}
        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()
        return response.json()

# Usage example
client = EnhancedModelAvailabilityClient()

# Check model status
status = client.get_model_status("my-model")
print(f"Model available: {status['is_available']}")

# Start download if not available
if not status['is_available']:
    result = client.start_download("my-model")
    print(f"Download started: {result['success']}")

# Get suggestions for unavailable model
suggestions = client.get_model_suggestions(
    "unavailable-model",
    {"quality": "high", "speed": "medium"}
)
print(f"Suggested alternative: {suggestions['suggestions'][0]['suggested_model']}")
```

### JavaScript SDK

```javascript
class EnhancedModelAvailabilityClient {
  constructor(baseUrl = "http://localhost:8000/api/v1", apiKey = null) {
    this.baseUrl = baseUrl;
    this.headers = {
      "Content-Type": "application/json",
    };
    if (apiKey) {
      this.headers["Authorization"] = `Bearer ${apiKey}`;
    }
  }

  async getModelStatus(modelId = null) {
    let url = `${this.baseUrl}/models/status/detailed`;
    if (modelId) {
      url += `/${modelId}`;
    }

    const response = await fetch(url, {
      headers: this.headers,
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  }

  async startDownload(modelId, maxRetries = 3, priority = "normal") {
    const url = `${this.baseUrl}/models/download/manage`;
    const data = {
      model_id: modelId,
      action: "start",
      max_retries: maxRetries,
      priority: priority,
    };

    const response = await fetch(url, {
      method: "POST",
      headers: this.headers,
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  }

  async getDownloadProgress(modelId) {
    const url = `${this.baseUrl}/models/download/progress/${modelId}`;

    const response = await fetch(url, {
      headers: this.headers,
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  }

  connectWebSocket() {
    const ws = new WebSocket("ws://localhost:8000/ws");

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      this.handleWebSocketMessage(data);
    };

    return ws;
  }

  handleWebSocketMessage(data) {
    switch (data.type) {
      case "model_download_progress":
        this.onDownloadProgress(data);
        break;
      case "model_health_alert":
        this.onHealthAlert(data);
        break;
      case "model_availability_change":
        this.onAvailabilityChange(data);
        break;
      default:
        console.log("Unknown message type:", data.type);
    }
  }

  onDownloadProgress(data) {
    console.log(
      `Download progress for ${data.model_id}: ${data.progress_percent}%`
    );
  }

  onHealthAlert(data) {
    console.warn(`Health alert for ${data.model_id}: ${data.message}`);
  }

  onAvailabilityChange(data) {
    console.log(
      `Model ${data.model_id} status changed: ${data.old_status} -> ${data.new_status}`
    );
  }
}

// Usage example
const client = new EnhancedModelAvailabilityClient();

// Check model status
client
  .getModelStatus("my-model")
  .then((status) => {
    console.log("Model available:", status.is_available);

    // Start download if not available
    if (!status.is_available) {
      return client.startDownload("my-model");
    }
  })
  .then((result) => {
    if (result) {
      console.log("Download started:", result.success);
    }
  })
  .catch((error) => {
    console.error("Error:", error);
  });

// Connect to WebSocket for real-time updates
const ws = client.connectWebSocket();
```

## Error Handling Examples

### Handling Download Failures

```python
try:
    result = client.start_download("large-model")
    download_id = result["download_id"]

    # Monitor progress
    while True:
        progress = client.get_download_progress("large-model")
        if progress["status"] == "COMPLETED":
            break
        elif progress["status"] == "FAILED":
            # Handle failure with retry
            if progress["retry_count"] < progress["max_retries"]:
                print("Download failed, retrying...")
                time.sleep(30)  # Wait before checking again
            else:
                print("Download failed after all retries")
                break
        time.sleep(5)

except requests.exceptions.HTTPError as e:
    if e.response.status_code == 409:
        print("Download already in progress")
    elif e.response.status_code == 422:
        error_data = e.response.json()
        print(f"Validation error: {error_data['error']['message']}")
    else:
        print(f"HTTP error: {e}")
```

### Handling Model Unavailability

```python
try:
    # Try to use preferred model
    status = client.get_model_status("preferred-model")

    if not status["is_available"]:
        # Get suggestions for alternatives
        suggestions = client.get_model_suggestions(
            "preferred-model",
            {"quality": "high", "speed": "medium"}
        )

        best_alternative = suggestions["suggestions"][0]

        if best_alternative["availability_status"] == "AVAILABLE":
            print(f"Using alternative: {best_alternative['suggested_model']}")
            # Use alternative model
        else:
            # Queue request for preferred model
            queue_result = client.queue_request(
                "preferred-model",
                {"prompt": "Generate video..."}
            )
            print(f"Request queued, position: {queue_result['position']}")

except Exception as e:
    print(f"Error handling model unavailability: {e}")
```

This comprehensive API documentation provides all the information needed to integrate with the Enhanced Model Availability system, including detailed endpoint descriptions, request/response formats, error handling, and practical examples in multiple programming languages.
