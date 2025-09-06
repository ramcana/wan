# API Reference

This section contains comprehensive API documentation for the WAN22 Video Generation System.

## Contents

- [Backend API](backend-api.md) - REST API endpoints and WebSocket connections
- [Frontend Components](frontend-components.md) - React component API reference
- [Configuration API](configuration-api.md) - Configuration management API
- [Model Management API](model-management-api.md) - AI model management endpoints

## API Overview

The WAN22 system provides multiple API interfaces:

### REST API

- **Base URL**: `http://localhost:8000/api/v1`
- **Authentication**: Token-based authentication
- **Content Type**: `application/json`
- **Rate Limiting**: Configurable per endpoint

### WebSocket API

- **Base URL**: `ws://localhost:8000/ws`
- **Real-time Updates**: Progress tracking, system status
- **Event-driven**: Subscription-based event system

### Configuration API

- **Management Interface**: System configuration and settings
- **Environment Support**: Development, staging, production
- **Validation**: Comprehensive configuration validation

## Quick Start

### Authentication

```javascript
// Get authentication token
const response = await fetch("/api/v1/auth/token", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ username, password }),
});
const { token } = await response.json();
```

### Video Generation

```javascript
// Start video generation
const response = await fetch("/api/v1/generate/video", {
  method: "POST",
  headers: {
    Authorization: `Bearer ${token}`,
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    prompt: "A beautiful sunset over mountains",
    model: "wan22-base",
    duration: 5,
  }),
});
```

### WebSocket Connection

```javascript
// Connect to WebSocket for real-time updates
const ws = new WebSocket("ws://localhost:8000/ws");
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log("Progress update:", data);
};
```

## Error Handling

All API endpoints return standardized error responses:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input parameters",
    "details": {
      "field": "prompt",
      "issue": "Prompt cannot be empty"
    }
  }
}
```

## Rate Limits

API endpoints have configurable rate limits:

- **Generation endpoints**: 10 requests per minute
- **Status endpoints**: 100 requests per minute
- **Configuration endpoints**: 50 requests per minute
