---
category: reference
last_updated: '2025-09-15T22:49:59.838707'
original_path: backend\websocket\README_MODEL_NOTIFICATIONS.md
tags:
- configuration
- api
- troubleshooting
- installation
- security
- performance
title: WebSocket Model Notifications Integration
---

# WebSocket Model Notifications Integration

This document describes the WebSocket integration for real-time model availability updates, download progress notifications, health monitoring alerts, and fallback strategy notifications.

## Overview

The WebSocket model notifications system provides real-time updates for all enhanced model availability features, enabling responsive user interfaces and immediate feedback for model-related operations.

## Architecture

```
Frontend (React/Vue/etc.)
    ↓ WebSocket Connection
WebSocket Manager (Enhanced)
    ↓ Topic-based Subscriptions
Model Notification Integrator
    ↓ Event Handlers
Enhanced Model Components
    ↓ Callbacks
Model Operations (Download, Health, etc.)
```

## Subscription Topics

The system supports the following WebSocket subscription topics:

### Core Topics (Existing)

- `system_stats` - System performance metrics
- `generation_progress` - AI generation progress
- `queue_updates` - Generation queue status
- `alerts` - System alerts

### Enhanced Model Topics (New)

- `model_status` - Model availability status changes
- `download_progress` - Real-time download progress
- `health_monitoring` - Model health alerts and monitoring
- `fallback_notifications` - Fallback strategy notifications
- `analytics_updates` - Usage analytics and recommendations

## Message Types

### Model Status Updates

#### `model_status_update`

```json
{
  "type": "model_status_update",
  "data": {
    "model_id": "t2v-a14b",
    "availability_status": "available",
    "is_available": true,
    "is_loaded": false,
    "size_mb": 1024.0,
    "integrity_score": 0.95,
    "performance_score": 0.88,
    "usage_frequency": 2.5,
    "update_available": false,
    "timestamp": "2025-08-27T10:30:00Z"
  }
}
```

#### `model_availability_change`

```json
{
  "type": "model_availability_change",
  "data": {
    "model_id": "i2v-a14b",
    "old_availability": "missing",
    "new_availability": "downloading",
    "reason": "User requested download",
    "timestamp": "2025-08-27T10:30:00Z"
  }
}
```

#### `batch_model_status_update`

```json
{
  "type": "batch_model_status_update",
  "data": {
    "models": [
      {
        "model_id": "t2v-a14b",
        "availability_status": "available",
        "integrity_score": 0.98
      }
    ],
    "count": 1,
    "timestamp": "2025-08-27T10:30:00Z"
  }
}
```

### Download Progress Updates

#### `download_progress_update`

```json
{
  "type": "download_progress_update",
  "data": {
    "model_id": "t2v-a14b",
    "status": "downloading",
    "progress_percent": 45.5,
    "downloaded_mb": 455.0,
    "total_mb": 1000.0,
    "speed_mbps": 15.2,
    "eta_seconds": 36.0,
    "can_pause": true,
    "can_resume": true,
    "can_cancel": true,
    "timestamp": "2025-08-27T10:30:00Z"
  }
}
```

#### `download_status_change`

```json
{
  "type": "download_status_change",
  "data": {
    "model_id": "t2v-a14b",
    "old_status": "queued",
    "new_status": "downloading",
    "progress_percent": 0.0,
    "reason": "Download started",
    "timestamp": "2025-08-27T10:30:00Z"
  }
}
```

#### `download_retry`

```json
{
  "type": "download_retry",
  "data": {
    "model_id": "t2v-a14b",
    "retry_count": 2,
    "max_retries": 3,
    "error_message": "Network timeout",
    "next_retry_in_seconds": 4,
    "timestamp": "2025-08-27T10:30:00Z"
  }
}
```

### Health Monitoring Alerts

#### `health_monitoring_alert`

```json
{
  "type": "health_monitoring_alert",
  "data": {
    "model_id": "t2v-a14b",
    "health_status": "healthy",
    "is_healthy": true,
    "integrity_score": 0.98,
    "file_count": 5,
    "total_size_mb": 1024.0,
    "timestamp": "2025-08-27T10:30:00Z"
  }
}
```

#### `corruption_detection`

```json
{
  "type": "corruption_detection",
  "data": {
    "model_id": "t2v-a14b",
    "corruption_type": "checksum_mismatch",
    "severity": "high",
    "repair_action": "Re-download corrupted files",
    "requires_user_action": true,
    "timestamp": "2025-08-27T10:30:00Z"
  }
}
```

### Fallback Strategy Notifications

#### `fallback_strategy`

```json
{
  "type": "fallback_strategy",
  "data": {
    "original_model": "unavailable-model",
    "strategy_type": "alternative_model",
    "recommended_action": "Use alternative model",
    "alternative_model": "backup-model",
    "user_message": "Suggested alternative model available",
    "user_interaction_required": false,
    "timestamp": "2025-08-27T10:30:00Z"
  }
}
```

#### `alternative_model_suggestion`

```json
{
  "type": "alternative_model_suggestion",
  "data": {
    "original_model": "requested-model",
    "suggested_model": "alternative-model",
    "compatibility_score": 0.85,
    "performance_difference": -0.1,
    "reason": "Similar capabilities with slightly lower performance",
    "timestamp": "2025-08-27T10:30:00Z"
  }
}
```

#### `model_queue_update`

```json
{
  "type": "model_queue_update",
  "data": {
    "model_id": "queued-model",
    "queue_position": 2,
    "estimated_wait_time": 300.0,
    "timestamp": "2025-08-27T10:30:00Z"
  }
}
```

### Analytics Updates

#### `usage_statistics_update`

```json
{
  "type": "usage_statistics_update",
  "data": {
    "total_models": 5,
    "active_models": 3,
    "most_used_model": "t2v-a14b",
    "usage_trends": {
      "daily_usage": 25,
      "weekly_usage": 150
    },
    "timestamp": "2025-08-27T10:30:00Z"
  }
}
```

#### `cleanup_recommendation`

```json
{
  "type": "cleanup_recommendation",
  "data": {
    "recommended_cleanup": ["unused-model-1", "unused-model-2"],
    "potential_space_saved_mb": 2048.0,
    "cleanup_priority": "medium",
    "timestamp": "2025-08-27T10:30:00Z"
  }
}
```

## Usage Examples

### Frontend Integration (JavaScript)

```javascript
// Connect to WebSocket
const ws = new WebSocket("ws://localhost:8000/ws/connection_id");

// Subscribe to model notifications
ws.onopen = () => {
  ws.send(
    JSON.stringify({
      action: "subscribe",
      topic: "model_status",
    })
  );

  ws.send(
    JSON.stringify({
      action: "subscribe",
      topic: "download_progress",
    })
  );
};

// Handle incoming messages
ws.onmessage = (event) => {
  const message = JSON.parse(event.data);

  switch (message.type) {
    case "model_status_update":
      updateModelStatus(message.data);
      break;

    case "download_progress_update":
      updateDownloadProgress(message.data);
      break;

    case "health_monitoring_alert":
      showHealthAlert(message.data);
      break;

    case "fallback_strategy":
      handleFallbackStrategy(message.data);
      break;
  }
};

// Update UI functions
function updateModelStatus(data) {
  const modelElement = document.getElementById(`model-${data.model_id}`);
  modelElement.classList.toggle("available", data.is_available);
  modelElement.querySelector(".status").textContent = data.availability_status;
}

function updateDownloadProgress(data) {
  const progressBar = document.getElementById(`progress-${data.model_id}`);
  progressBar.style.width = `${data.progress_percent}%`;
  progressBar.textContent = `${data.progress_percent.toFixed(1)}%`;
}

function showHealthAlert(data) {
  if (data.health_status === "corrupted") {
    showNotification(`Model ${data.model_id} corruption detected`, "error");
  }
}

function handleFallbackStrategy(data) {
  if (data.user_interaction_required) {
    showFallbackDialog(data);
  } else {
    showNotification(data.user_message, "info");
  }
}
```

### Backend Integration (Python)

```python
from backend.websocket.model_notifications import get_model_notification_integrator

# Get the integrator instance
integrator = get_model_notification_integrator()

# Setup component integrations
await integrator.setup_component_integrations(
    enhanced_downloader=enhanced_downloader,
    health_monitor=health_monitor,
    availability_manager=availability_manager,
    fallback_manager=fallback_manager
)

# Manual notifications (if needed)
download_notifier = integrator.get_download_notifier()
await download_notifier.on_download_progress(model_id, progress_data)

health_notifier = integrator.get_health_notifier()
await health_notifier.on_corruption_detected(model_id, "checksum_mismatch", "high")

fallback_notifier = integrator.get_fallback_notifier()
await fallback_notifier.on_alternative_model_suggested(original_model, suggestion)
```

## Component Integration

### Enhanced Model Downloader Integration

The enhanced model downloader automatically sends WebSocket notifications for:

- Download started/paused/resumed/cancelled
- Progress updates (every few seconds)
- Retry attempts with exponential backoff
- Download completion or failure

### Model Health Monitor Integration

The health monitor sends notifications for:

- Scheduled health check results
- Corruption detection and severity assessment
- Performance degradation alerts
- Automatic repair start/completion

### Model Availability Manager Integration

The availability manager coordinates notifications for:

- Model availability status changes
- Batch status updates for multiple models
- Model lifecycle events (loading, unloading)
- Priority and cleanup recommendations

### Intelligent Fallback Manager Integration

The fallback manager provides notifications for:

- Fallback strategy activation
- Alternative model suggestions
- Queue position updates
- Wait time estimates

## Error Handling

The WebSocket system includes robust error handling:

```python
# Connection errors are handled gracefully
try:
    await websocket.send_text(json.dumps(message))
except WebSocketDisconnect:
    await self.disconnect(connection_id)
except Exception as e:
    logger.error(f"Error sending message: {e}")
    await self.disconnect(connection_id)
```

## Performance Considerations

- **Batching**: Multiple status updates are batched when possible
- **Throttling**: Progress updates are throttled to prevent spam
- **Selective Broadcasting**: Messages are only sent to subscribed clients
- **Connection Management**: Automatic cleanup of disconnected clients

## Security Considerations

- **Authentication**: WebSocket connections should be authenticated
- **Rate Limiting**: Prevent abuse with connection rate limits
- **Data Validation**: All incoming messages are validated
- **Access Control**: Topic subscriptions can be restricted by user role

## Testing

Comprehensive tests are provided in `backend/tests/test_websocket_model_notifications.py`:

```bash
# Run all WebSocket notification tests
python -m pytest backend/tests/test_websocket_model_notifications.py -v

# Run specific test categories
python -m pytest backend/tests/test_websocket_model_notifications.py::TestModelDownloadNotifier -v
python -m pytest backend/tests/test_websocket_model_notifications.py::TestModelHealthNotifier -v
```

## Configuration

WebSocket behavior can be configured through environment variables:

```bash
# WebSocket settings
WEBSOCKET_HEARTBEAT_INTERVAL=30
WEBSOCKET_MAX_CONNECTIONS=100
WEBSOCKET_MESSAGE_QUEUE_SIZE=1000

# Model notification settings
MODEL_STATUS_UPDATE_INTERVAL=5
DOWNLOAD_PROGRESS_THROTTLE=2
HEALTH_CHECK_NOTIFICATION_LEVEL=warning
```

## Monitoring and Debugging

Enable debug logging for WebSocket operations:

```python
import logging
logging.getLogger('backend.websocket').setLevel(logging.DEBUG)
```

Monitor WebSocket metrics:

- Active connection count
- Message throughput
- Subscription distribution
- Error rates

## Future Enhancements

Planned improvements include:

- Message compression for large payloads
- WebSocket clustering for horizontal scaling
- Advanced filtering and subscription management
- Real-time analytics dashboard integration
- Mobile push notification integration
