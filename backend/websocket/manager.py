"""
WebSocket manager for real-time system monitoring and updates
Provides sub-second updates for system stats and generation progress
"""

import asyncio
import json
import logging
from typing import Dict, List, Set, Optional, Any
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

class ConnectionManager:
    """Manages WebSocket connections for real-time updates"""
    
    def __init__(self):
        # Active connections by connection ID
        self.active_connections: Dict[str, WebSocket] = {}
        # Subscriptions by topic
        self.subscriptions: Dict[str, Set[str]] = {
            "system_stats": set(),
            "generation_progress": set(),
            "queue_updates": set(),
            "alerts": set()
        }
        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        self._stats_update_task: Optional[asyncio.Task] = None
        self._is_running = False
    
    async def connect(self, websocket: WebSocket, connection_id: str) -> bool:
        """Accept a new WebSocket connection"""
        try:
            await websocket.accept()
            self.active_connections[connection_id] = websocket
            logger.info(f"WebSocket connection established: {connection_id}")
            
            # Start background tasks if this is the first connection
            if len(self.active_connections) == 1 and not self._is_running:
                await self._start_background_tasks()
            
            return True
        except Exception as e:
            logger.error(f"Failed to accept WebSocket connection {connection_id}: {e}")
            return False
    
    async def disconnect(self, connection_id: str):
        """Remove a WebSocket connection"""
        if connection_id in self.active_connections:
            # Remove from all subscriptions
            for topic_subscribers in self.subscriptions.values():
                topic_subscribers.discard(connection_id)
            
            # Remove connection
            del self.active_connections[connection_id]
            logger.info(f"WebSocket connection removed: {connection_id}")
            
            # Stop background tasks if no connections remain
            if len(self.active_connections) == 0:
                await self._stop_background_tasks()
    
    async def subscribe(self, connection_id: str, topic: str) -> bool:
        """Subscribe a connection to a topic"""
        if connection_id not in self.active_connections:
            return False
        
        if topic not in self.subscriptions:
            logger.warning(f"Unknown subscription topic: {topic}")
            return False
        
        self.subscriptions[topic].add(connection_id)
        logger.info(f"Connection {connection_id} subscribed to {topic}")
        return True
    
    async def unsubscribe(self, connection_id: str, topic: str) -> bool:
        """Unsubscribe a connection from a topic"""
        if topic in self.subscriptions:
            self.subscriptions[topic].discard(connection_id)
            logger.info(f"Connection {connection_id} unsubscribed from {topic}")
            return True
        return False
    
    async def send_personal_message(self, message: dict, connection_id: str):
        """Send a message to a specific connection"""
        if connection_id in self.active_connections:
            websocket = self.active_connections[connection_id]
            try:
                await websocket.send_text(json.dumps(message))
            except WebSocketDisconnect:
                await self.disconnect(connection_id)
            except Exception as e:
                logger.error(f"Error sending message to {connection_id}: {e}")
                await self.disconnect(connection_id)
    
    async def broadcast_to_topic(self, message: dict, topic: str):
        """Broadcast a message to all subscribers of a topic"""
        if topic not in self.subscriptions:
            return
        
        subscribers = self.subscriptions[topic].copy()
        disconnected = []
        
        for connection_id in subscribers:
            if connection_id in self.active_connections:
                websocket = self.active_connections[connection_id]
                try:
                    await websocket.send_text(json.dumps(message))
                except WebSocketDisconnect:
                    disconnected.append(connection_id)
                except Exception as e:
                    logger.error(f"Error broadcasting to {connection_id}: {e}")
                    disconnected.append(connection_id)
        
        # Clean up disconnected clients
        for connection_id in disconnected:
            await self.disconnect(connection_id)
    
    async def broadcast_to_all(self, message: dict):
        """Broadcast a message to all active connections"""
        disconnected = []
        
        for connection_id, websocket in self.active_connections.items():
            try:
                await websocket.send_text(json.dumps(message))
            except WebSocketDisconnect:
                disconnected.append(connection_id)
            except Exception as e:
                logger.error(f"Error broadcasting to {connection_id}: {e}")
                disconnected.append(connection_id)
        
        # Clean up disconnected clients
        for connection_id in disconnected:
            await self.disconnect(connection_id)
    
    async def _start_background_tasks(self):
        """Start background tasks for real-time updates"""
        if self._is_running:
            return
        
        self._is_running = True
        
        # Start system stats update task (sub-second updates)
        self._stats_update_task = asyncio.create_task(self._system_stats_updater())
        self._background_tasks.add(self._stats_update_task)
        
        logger.info("WebSocket background tasks started")
    
    async def _stop_background_tasks(self):
        """Stop all background tasks"""
        self._is_running = False
        
        # Cancel all background tasks
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        self._background_tasks.clear()
        self._stats_update_task = None
        
        logger.info("WebSocket background tasks stopped")
    
    async def _system_stats_updater(self):
        """Background task to send system stats updates every 500ms"""
        from backend.core.system_integration import get_system_integration
        
        try:
            integration = await get_system_integration()
            
            while self._is_running and self.subscriptions["system_stats"]:
                try:
                    # Get current system stats
                    stats = await integration.get_enhanced_system_stats()
                    
                    if stats:
                        message = {
                            "type": "system_stats_update",
                            "data": {
                                "cpu_percent": stats.get("cpu_percent", 0.0),
                                "ram_used_gb": stats.get("ram_used_gb", 0.0),
                                "ram_total_gb": stats.get("ram_total_gb", 0.0),
                                "ram_percent": stats.get("ram_percent", 0.0),
                                "gpu_percent": stats.get("gpu_percent", 0.0),
                                "vram_used_mb": stats.get("vram_used_mb", 0.0),
                                "vram_total_mb": stats.get("vram_total_mb", 0.0),
                                "vram_percent": stats.get("vram_percent", 0.0),
                                "timestamp": datetime.utcnow().isoformat()
                            }
                        }
                        
                        await self.broadcast_to_topic(message, "system_stats")
                    
                    # Wait 500ms for sub-second updates
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"Error in system stats updater: {e}")
                    await asyncio.sleep(1.0)  # Wait longer on error
                    
        except Exception as e:
            logger.error(f"System stats updater failed: {e}")
    
    async def send_generation_progress(self, task_id: str, progress: int, status: str, **kwargs):
        """Send generation progress update"""
        message = {
            "type": "generation_progress",
            "data": {
                "task_id": task_id,
                "progress": progress,
                "status": status,
                "timestamp": datetime.utcnow().isoformat(),
                **kwargs
            }
        }
        await self.broadcast_to_topic(message, "generation_progress")
    
    async def send_queue_update(self, queue_data: dict):
        """Send queue status update"""
        message = {
            "type": "queue_update",
            "data": {
                **queue_data,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        await self.broadcast_to_topic(message, "queue_updates")
    
    async def send_alert(self, alert_type: str, message: str, severity: str = "info", **kwargs):
        """Send system alert"""
        alert_message = {
            "type": "alert",
            "data": {
                "alert_type": alert_type,
                "message": message,
                "severity": severity,
                "timestamp": datetime.utcnow().isoformat(),
                **kwargs
            }
        }
        await self.broadcast_to_topic(alert_message, "alerts")
    
    def get_connection_count(self) -> int:
        """Get number of active connections"""
        return len(self.active_connections)
    
    def get_subscription_count(self, topic: str) -> int:
        """Get number of subscribers for a topic"""
        return len(self.subscriptions.get(topic, set()))

# Global connection manager instance
connection_manager = ConnectionManager()

def get_connection_manager() -> ConnectionManager:
    """Dependency to get the connection manager"""
    return connection_manager