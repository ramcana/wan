---
title: websocket.model_notifications
category: api
tags: [api, websocket]
---

# websocket.model_notifications

Model Notifications WebSocket Integration
Integrates enhanced model availability components with WebSocket real-time notifications.

## Classes

### ModelDownloadNotifier

Handles WebSocket notifications for model download events

#### Methods

##### __init__(self: Any, connection_manager: ConnectionManager)



### ModelHealthNotifier

Handles WebSocket notifications for model health monitoring events

#### Methods

##### __init__(self: Any, connection_manager: ConnectionManager)



### ModelAvailabilityNotifier

Handles WebSocket notifications for model availability changes

#### Methods

##### __init__(self: Any, connection_manager: ConnectionManager)



### FallbackNotifier

Handles WebSocket notifications for fallback strategy events

#### Methods

##### __init__(self: Any, connection_manager: ConnectionManager)



### AnalyticsNotifier

Handles WebSocket notifications for analytics updates

#### Methods

##### __init__(self: Any, connection_manager: ConnectionManager)



### ModelNotificationIntegrator

Main integration class that coordinates all model-related WebSocket notifications

#### Methods

##### __init__(self: Any, connection_manager: <ast.Subscript object at 0x00000194285AC880>)



##### get_download_notifier(self: Any) -> ModelDownloadNotifier

Get the download notifier instance

##### get_health_notifier(self: Any) -> ModelHealthNotifier

Get the health notifier instance

##### get_availability_notifier(self: Any) -> ModelAvailabilityNotifier

Get the availability notifier instance

##### get_fallback_notifier(self: Any) -> FallbackNotifier

Get the fallback notifier instance

##### get_analytics_notifier(self: Any) -> AnalyticsNotifier

Get the analytics notifier instance

