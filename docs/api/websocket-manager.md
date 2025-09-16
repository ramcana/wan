---
title: websocket.manager
category: api
tags: [api, websocket]
---

# websocket.manager

WebSocket manager for real-time system monitoring and updates
Provides sub-second updates for system stats and generation progress

## Classes

### ConnectionManager

Manages WebSocket connections for real-time updates

#### Methods

##### __init__(self: Any)



##### _is_generation_active(self: Any) -> bool

Check if any generation is currently active

##### get_connection_count(self: Any) -> int

Get number of active connections

##### get_subscription_count(self: Any, topic: str) -> int

Get number of subscribers for a topic

