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
            "alerts": set(),
            # Enhanced model availability topics
            "model_status": set(),
            "download_progress": set(),
            "health_monitoring": set(),
            "fallback_notifications": set(),
            "analytics_updates": set()
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
    
    async def broadcast(self, message: dict):
        """Broadcast a message to all active connections (alias for broadcast_to_all)"""
        await self.broadcast_to_all(message)
    
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
        """Background task to send system stats updates every 500ms with enhanced VRAM monitoring"""
        from backend.core.system_integration import get_system_integration
        
        try:
            integration = await get_system_integration()
            
            while self._is_running and self.subscriptions["system_stats"]:
                try:
                    # Get current system stats
                    stats = await integration.get_enhanced_system_stats()
                    
                    if stats:
                        # Enhanced system stats with detailed VRAM monitoring
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
                        
                        # Send detailed VRAM monitoring if generation is active
                        if self._is_generation_active():
                            vram_details = await self._get_detailed_vram_stats(stats)
                            if vram_details:
                                await self.send_vram_monitoring_update(vram_details)
                    
                    # Wait 500ms for sub-second updates
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"Error in system stats updater: {e}")
                    await asyncio.sleep(1.0)  # Wait longer on error
                    
        except Exception as e:
            logger.error(f"System stats updater failed: {e}")
    
    def _is_generation_active(self) -> bool:
        """Check if any generation is currently active"""
        try:
            # Check if there are any subscribers to generation progress
            return len(self.subscriptions.get("generation_progress", set())) > 0
        except Exception:
            return False
    
    async def _get_detailed_vram_stats(self, base_stats: dict) -> Optional[dict]:
        """Get detailed VRAM statistics for real-time monitoring"""
        try:
            import torch
            if not torch.cuda.is_available():
                return None
            
            device = torch.cuda.current_device()
            allocated_bytes = torch.cuda.memory_allocated(device)
            reserved_bytes = torch.cuda.memory_reserved(device)
            total_bytes = torch.cuda.get_device_properties(device).total_memory
            
            allocated_mb = allocated_bytes / (1024 * 1024)
            reserved_mb = reserved_bytes / (1024 * 1024)
            total_mb = total_bytes / (1024 * 1024)
            free_mb = total_mb - allocated_mb
            
            # Calculate usage percentages
            allocated_percent = (allocated_mb / total_mb) * 100
            reserved_percent = (reserved_mb / total_mb) * 100
            
            # Determine warning levels
            warning_level = "normal"
            if allocated_percent > 90:
                warning_level = "critical"
            elif allocated_percent > 75:
                warning_level = "warning"
            
            return {
                "allocated_mb": round(allocated_mb, 1),
                "reserved_mb": round(reserved_mb, 1),
                "free_mb": round(free_mb, 1),
                "total_mb": round(total_mb, 1),
                "allocated_percent": round(allocated_percent, 1),
                "reserved_percent": round(reserved_percent, 1),
                "warning_level": warning_level,
                "device_name": torch.cuda.get_device_name(device),
                "device_id": device
            }
            
        except Exception as e:
            logger.warning(f"Failed to get detailed VRAM stats: {e}")
            return None
    
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
    
    async def send_detailed_generation_progress(self, task_id: str, stage: str, progress: int, 
                                              message: str, **kwargs):
        """Send detailed generation progress with stage information"""
        progress_message = {
            "type": "detailed_generation_progress",
            "data": {
                "task_id": task_id,
                "stage": stage,
                "progress": progress,
                "message": message,
                "timestamp": datetime.utcnow().isoformat(),
                **kwargs
            }
        }
        await self.broadcast_to_topic(progress_message, "generation_progress")
    
    async def send_model_loading_progress(self, task_id: str, model_type: str, progress: int, 
                                        status: str, **kwargs):
        """Send model loading progress updates"""
        message = {
            "type": "model_loading_progress",
            "data": {
                "task_id": task_id,
                "model_type": model_type,
                "progress": progress,
                "status": status,
                "timestamp": datetime.utcnow().isoformat(),
                **kwargs
            }
        }
        await self.broadcast_to_topic(message, "generation_progress")
    
    async def send_vram_monitoring_update(self, vram_data: dict, **kwargs):
        """Send real-time VRAM monitoring updates"""
        message = {
            "type": "vram_monitoring",
            "data": {
                **vram_data,
                "timestamp": datetime.utcnow().isoformat(),
                **kwargs
            }
        }
        await self.broadcast_to_topic(message, "system_stats")
    
    async def send_generation_stage_notification(self, task_id: str, stage: str, 
                                                stage_progress: int, **kwargs):
        """Send generation stage notifications (model loading, processing, post-processing)"""
        message = {
            "type": "generation_stage",
            "data": {
                "task_id": task_id,
                "stage": stage,
                "stage_progress": stage_progress,
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
    
    # Enhanced Model Availability WebSocket Methods
    
    async def send_model_status_update(self, model_id: str, **kwargs):
        """Send model availability status update"""
        message = {
            "type": "model_status_update",
            "data": {
                "model_id": model_id,
                "timestamp": datetime.utcnow().isoformat(),
                **kwargs
            }
        }
        await self.broadcast_to_topic(message, "model_status")
    
    async def send_download_progress_update(self, model_id: str, **kwargs):
        """Send real-time download progress notifications"""
        message = {
            "type": "download_progress_update",
            "data": {
                "model_id": model_id,
                "timestamp": datetime.utcnow().isoformat(),
                **kwargs
            }
        }
        await self.broadcast_to_topic(message, "download_progress")
    
    async def send_download_status_change(self, model_id: str, old_status: str, new_status: str, 
                                        progress_percent: float = 0.0, **kwargs):
        """Send download status change notification"""
        message = {
            "type": "download_status_change",
            "data": {
                "model_id": model_id,
                "old_status": old_status,
                "new_status": new_status,
                "progress_percent": progress_percent,
                "timestamp": datetime.utcnow().isoformat(),
                **kwargs
            }
        }
        await self.broadcast_to_topic(message, "download_progress")
    
    async def send_download_retry_notification(self, model_id: str, retry_count: int, 
                                             max_retries: int, error_message: str = "", **kwargs):
        """Send download retry attempt notification"""
        message = {
            "type": "download_retry",
            "data": {
                "model_id": model_id,
                "retry_count": retry_count,
                "max_retries": max_retries,
                "error_message": error_message,
                "timestamp": datetime.utcnow().isoformat(),
                **kwargs
            }
        }
        await self.broadcast_to_topic(message, "download_progress")
    
    async def send_health_monitoring_alert(self, model_id: str, health_status: str, **kwargs):
        """Send health monitoring alerts and notifications"""
        message = {
            "type": "health_monitoring_alert",
            "data": {
                "model_id": model_id,
                "health_status": health_status,
                "timestamp": datetime.utcnow().isoformat(),
                **kwargs
            }
        }
        await self.broadcast_to_topic(message, "health_monitoring")
    
    async def send_corruption_detection_alert(self, model_id: str, corruption_type: str, 
                                            severity: str, repair_action: str = "", **kwargs):
        """Send corruption detection alert"""
        message = {
            "type": "corruption_detection",
            "data": {
                "model_id": model_id,
                "corruption_type": corruption_type,
                "severity": severity,
                "repair_action": repair_action,
                "timestamp": datetime.utcnow().isoformat(),
                **kwargs
            }
        }
        await self.broadcast_to_topic(message, "health_monitoring")
    
    async def send_model_availability_change(self, model_id: str, old_availability: str, 
                                           new_availability: str, reason: str = "", **kwargs):
        """Send model availability change notifications"""
        message = {
            "type": "model_availability_change",
            "data": {
                "model_id": model_id,
                "old_availability": old_availability,
                "new_availability": new_availability,
                "reason": reason,
                "timestamp": datetime.utcnow().isoformat(),
                **kwargs
            }
        }
        await self.broadcast_to_topic(message, "model_status")
    
    async def send_fallback_strategy_notification(self, original_model: str, 
                                                user_interaction_required: bool = False, **kwargs):
        """Send fallback strategy notifications with user interaction options"""
        message = {
            "type": "fallback_strategy",
            "data": {
                "original_model": original_model,
                "user_interaction_required": user_interaction_required,
                "timestamp": datetime.utcnow().isoformat(),
                **kwargs
            }
        }
        await self.broadcast_to_topic(message, "fallback_notifications")
    
    async def send_alternative_model_suggestion(self, original_model: str, suggested_model: str, 
                                              compatibility_score: float, reason: str, **kwargs):
        """Send alternative model suggestion notification"""
        message = {
            "type": "alternative_model_suggestion",
            "data": {
                "original_model": original_model,
                "suggested_model": suggested_model,
                "compatibility_score": compatibility_score,
                "reason": reason,
                "timestamp": datetime.utcnow().isoformat(),
                **kwargs
            }
        }
        await self.broadcast_to_topic(message, "fallback_notifications")
    
    async def send_model_queue_notification(self, model_id: str, queue_position: int, 
                                          estimated_wait_time: Optional[float] = None, **kwargs):
        """Send model download queue notification"""
        message = {
            "type": "model_queue_update",
            "data": {
                "model_id": model_id,
                "queue_position": queue_position,
                "estimated_wait_time": estimated_wait_time,
                "timestamp": datetime.utcnow().isoformat(),
                **kwargs
            }
        }
        await self.broadcast_to_topic(message, "fallback_notifications")
    
    async def send_analytics_update(self, analytics_type: str, analytics_data: dict, **kwargs):
        """Send analytics updates for dashboard integration"""
        message = {
            "type": "analytics_update",
            "data": {
                "analytics_type": analytics_type,
                "timestamp": datetime.utcnow().isoformat(),
                **analytics_data,
                **kwargs
            }
        }
        await self.broadcast_to_topic(message, "analytics_updates")
    
    async def send_usage_statistics_update(self, model_usage_data: dict, **kwargs):
        """Send model usage statistics update"""
        message = {
            "type": "usage_statistics_update",
            "data": {
                "timestamp": datetime.utcnow().isoformat(),
                **model_usage_data,
                **kwargs
            }
        }
        await self.broadcast_to_topic(message, "analytics_updates")
    
    async def send_cleanup_recommendation(self, recommendation_data: dict, **kwargs):
        """Send model cleanup recommendation notification"""
        message = {
            "type": "cleanup_recommendation",
            "data": {
                "timestamp": datetime.utcnow().isoformat(),
                **recommendation_data,
                **kwargs
            }
        }
        await self.broadcast_to_topic(message, "analytics_updates")
    
    async def send_model_update_notification(self, model_id: str, update_type: str, 
                                           update_data: dict, **kwargs):
        """Send model update notifications (version updates, etc.)"""
        message = {
            "type": "model_update_notification",
            "data": {
                "model_id": model_id,
                "update_type": update_type,
                "timestamp": datetime.utcnow().isoformat(),
                **update_data,
                **kwargs
            }
        }
        await self.broadcast_to_topic(message, "model_status")
    
    async def send_batch_model_status_update(self, models_data: List[dict], **kwargs):
        """Send batch model status update for multiple models"""
        message = {
            "type": "batch_model_status_update",
            "data": {
                "models": models_data,
                "count": len(models_data),
                "timestamp": datetime.utcnow().isoformat(),
                **kwargs
            }
        }
        await self.broadcast_to_topic(message, "model_status")
    
    async def send_system_health_report(self, health_report: dict, **kwargs):
        """Send comprehensive system health report"""
        message = {
            "type": "system_health_report",
            "data": {
                "timestamp": datetime.utcnow().isoformat(),
                **health_report,
                **kwargs
            }
        }
        await self.broadcast_to_topic(message, "health_monitoring")

# Global connection manager instance
connection_manager = ConnectionManager()

def get_connection_manager() -> ConnectionManager:
    """Dependency to get the connection manager"""
    return connection_manager