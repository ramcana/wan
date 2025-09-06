"""
WebSocket endpoints for real-time updates
"""

import json
import uuid
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from websocket.manager import connection_manager, get_connection_manager, ConnectionManager

logger = logging.getLogger(__name__)

router = APIRouter()

@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    manager: ConnectionManager = Depends(get_connection_manager)
):
    """
    Main WebSocket endpoint for real-time updates
    Supports subscriptions to: system_stats, generation_progress, queue_updates, alerts
    """
    connection_id = str(uuid.uuid4())
    
    # Accept connection
    connected = await manager.connect(websocket, connection_id)
    if not connected:
        return
    
    try:
        # Send welcome message
        await manager.send_personal_message({
            "type": "connection_established",
            "data": {
                "connection_id": connection_id,
                "available_topics": list(manager.subscriptions.keys()),
                "message": "WebSocket connection established"
            }
        }, connection_id)
        
        # Handle incoming messages
        while True:
            try:
                # Receive message from client
                data = await websocket.receive_text()
                message = json.loads(data)
                
                message_type = message.get("type")
                payload = message.get("data", {})
                
                if message_type == "subscribe":
                    # Subscribe to topic
                    topic = payload.get("topic")
                    if topic:
                        success = await manager.subscribe(connection_id, topic)
                        await manager.send_personal_message({
                            "type": "subscription_response",
                            "data": {
                                "topic": topic,
                                "subscribed": success,
                                "message": f"{'Subscribed to' if success else 'Failed to subscribe to'} {topic}"
                            }
                        }, connection_id)
                
                elif message_type == "unsubscribe":
                    # Unsubscribe from topic
                    topic = payload.get("topic")
                    if topic:
                        success = await manager.unsubscribe(connection_id, topic)
                        await manager.send_personal_message({
                            "type": "unsubscription_response",
                            "data": {
                                "topic": topic,
                                "unsubscribed": success,
                                "message": f"{'Unsubscribed from' if success else 'Failed to unsubscribe from'} {topic}"
                            }
                        }, connection_id)
                
                elif message_type == "ping":
                    # Respond to ping
                    await manager.send_personal_message({
                        "type": "pong",
                        "data": {
                            "timestamp": payload.get("timestamp"),
                            "message": "pong"
                        }
                    }, connection_id)
                
                elif message_type == "get_status":
                    # Send connection status
                    await manager.send_personal_message({
                        "type": "status_response",
                        "data": {
                            "connection_id": connection_id,
                            "active_connections": manager.get_connection_count(),
                            "subscriptions": {
                                topic: connection_id in subscribers 
                                for topic, subscribers in manager.subscriptions.items()
                            },
                            "subscription_counts": {
                                topic: len(subscribers)
                                for topic, subscribers in manager.subscriptions.items()
                            }
                        }
                    }, connection_id)
                
                else:
                    # Unknown message type
                    await manager.send_personal_message({
                        "type": "error",
                        "data": {
                            "message": f"Unknown message type: {message_type}",
                            "received_message": message
                        }
                    }, connection_id)
                    
            except json.JSONDecodeError:
                await manager.send_personal_message({
                    "type": "error",
                    "data": {
                        "message": "Invalid JSON format",
                        "received_data": data
                    }
                }, connection_id)
            
            except Exception as e:
                logger.error(f"Error processing WebSocket message from {connection_id}: {e}")
                await manager.send_personal_message({
                    "type": "error",
                    "data": {
                        "message": f"Error processing message: {str(e)}"
                    }
                }, connection_id)
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket client {connection_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error for {connection_id}: {e}")
    finally:
        await manager.disconnect(connection_id)

@router.get("/ws/stats")
async def get_websocket_stats(
    manager: ConnectionManager = Depends(get_connection_manager)
):
    """Get WebSocket connection statistics"""
    return {
        "active_connections": manager.get_connection_count(),
        "subscription_counts": {
            topic: manager.get_subscription_count(topic)
            for topic in manager.subscriptions.keys()
        },
        "available_topics": list(manager.subscriptions.keys())
    }