"""
Generation Queue Management System for Wan2.2 Video Generation

This module provides queue management for multiple generation requests,
priority handling, resource allocation, and concurrent processing.
"""

import logging
import threading
import time
import uuid
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from queue import PriorityQueue, Queue, Empty
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class QueuePriority(Enum):
    """Queue priority levels"""
    LOW = 3
    NORMAL = 2
    HIGH = 1
    URGENT = 0

class RequestStatus(Enum):
    """Status of generation requests"""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

@dataclass
class GenerationRequest:
    """Represents a generation request in the queue"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = "default"
    model_type: str = "t2v"
    prompt: str = ""
    image_path: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: QueuePriority = QueuePriority.NORMAL
    status: RequestStatus = RequestStatus.QUEUED
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    estimated_duration: float = 0.0  # seconds
    actual_duration: Optional[float] = None
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    progress: float = 0.0
    retry_count: int = 0
    max_retries: int = 3
    callback: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other):
        """For priority queue ordering"""
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        return self.created_at < other.created_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "model_type": self.model_type,
            "prompt": self.prompt,
            "image_path": self.image_path,
            "parameters": self.parameters,
            "priority": self.priority.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "estimated_duration": self.estimated_duration,
            "actual_duration": self.actual_duration,
            "output_path": self.output_path,
            "error_message": self.error_message,
            "progress": self.progress,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "metadata": self.metadata
        }

@dataclass
class QueueStatistics:
    """Queue statistics and metrics"""
    total_requests: int = 0
    queued_requests: int = 0
    processing_requests: int = 0
    completed_requests: int = 0
    failed_requests: int = 0
    cancelled_requests: int = 0
    average_wait_time: float = 0.0
    average_processing_time: float = 0.0
    throughput_per_hour: float = 0.0
    success_rate: float = 0.0
    queue_length: int = 0
    estimated_wait_time: float = 0.0

class GenerationQueueManager:
    """Manages generation request queue with priority and resource management"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Queue configuration
        self.max_concurrent_requests = self.config.get("max_concurrent_requests", 1)
        self.max_queue_size = self.config.get("max_queue_size", 100)
        self.auto_retry_failed = self.config.get("auto_retry_failed", True)
        self.save_queue_state = self.config.get("save_queue_state", True)
        
        # Queue and request tracking
        self.request_queue = PriorityQueue(maxsize=self.max_queue_size)
        self.active_requests: Dict[str, GenerationRequest] = {}
        self.completed_requests: List[GenerationRequest] = []
        self.failed_requests: List[GenerationRequest] = []
        
        # Threading
        self.queue_lock = threading.RLock()
        self.worker_threads: List[threading.Thread] = []
        self.shutdown_event = threading.Event()
        self.pause_event = threading.Event()
        
        # Statistics
        self.statistics = QueueStatistics()
        self.request_history: List[Dict[str, Any]] = []
        
        # Callbacks
        self.status_callbacks: List[Callable] = []
        self.progress_callbacks: List[Callable] = []
        
        # Load saved state
        if self.save_queue_state:
            self._load_queue_state()
        
        # Start worker threads
        self._start_workers()
        
        logger.info(f"Queue manager initialized with {self.max_concurrent_requests} workers")
    
    def add_status_callback(self, callback: Callable):
        """Add callback for status updates"""
        self.status_callbacks.append(callback)
    
    def add_progress_callback(self, callback: Callable):
        """Add callback for progress updates"""
        self.progress_callbacks.append(callback)
    
    def submit_request(self, model_type: str, prompt: str, 
                      image_path: Optional[str] = None,
                      parameters: Optional[Dict[str, Any]] = None,
                      priority: QueuePriority = QueuePriority.NORMAL,
                      user_id: str = "default",
                      callback: Optional[Callable] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Submit a new generation request to the queue
        Returns: request ID
        """
        try:
            # Estimate duration based on parameters
            estimated_duration = self._estimate_duration(model_type, parameters or {})
            
            request = GenerationRequest(
                user_id=user_id,
                model_type=model_type,
                prompt=prompt,
                image_path=image_path,
                parameters=parameters or {},
                priority=priority,
                estimated_duration=estimated_duration,
                callback=callback,
                metadata=metadata or {}
            )
            
            # Check queue capacity
            if self.request_queue.qsize() >= self.max_queue_size:
                raise Exception("Queue is full")
            
            # Add to queue
            self.request_queue.put(request)
            
            with self.queue_lock:
                self.statistics.total_requests += 1
                self.statistics.queued_requests += 1
                self.statistics.queue_length = self.request_queue.qsize()
            
            # Notify callbacks
            self._notify_status_callbacks("request_queued", request.id)
            
            logger.info(f"Request {request.id} queued with priority {priority.name}")
            return request.id
            
        except Exception as e:
            logger.error(f"Failed to submit request: {e}")
            return ""
    
    def get_request_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific request"""
        try:
            # Check active requests
            with self.queue_lock:
                if request_id in self.active_requests:
                    request = self.active_requests[request_id]
                    return {
                        "id": request.id,
                        "status": request.status.value,
                        "progress": request.progress,
                        "estimated_duration": request.estimated_duration,
                        "elapsed_time": (
                            (datetime.now() - request.started_at).total_seconds()
                            if request.started_at else 0
                        ),
                        "position_in_queue": None
                    }
            
            # Check completed requests
            for request in self.completed_requests:
                if request.id == request_id:
                    return {
                        "id": request.id,
                        "status": request.status.value,
                        "progress": 100.0,
                        "actual_duration": request.actual_duration,
                        "output_path": request.output_path,
                        "error_message": request.error_message
                    }
            
            # Check failed requests
            for request in self.failed_requests:
                if request.id == request_id:
                    return {
                        "id": request.id,
                        "status": request.status.value,
                        "progress": request.progress,
                        "error_message": request.error_message,
                        "retry_count": request.retry_count
                    }
            
            # Check queue (this is expensive, so do it last)
            queue_position = self._get_queue_position(request_id)
            if queue_position is not None:
                return {
                    "id": request_id,
                    "status": "queued",
                    "progress": 0.0,
                    "position_in_queue": queue_position,
                    "estimated_wait_time": self._estimate_wait_time(queue_position)
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get request status: {e}")
            return None
    
    def cancel_request(self, request_id: str) -> bool:
        """Cancel a queued or active request"""
        try:
            # Check if request is active
            with self.queue_lock:
                if request_id in self.active_requests:
                    request = self.active_requests[request_id]
                    request.status = RequestStatus.CANCELLED
                    request.completed_at = datetime.now()
                    
                    # Move to completed list
                    del self.active_requests[request_id]
                    self.completed_requests.append(request)
                    
                    self.statistics.cancelled_requests += 1
                    self.statistics.processing_requests -= 1
                    
                    self._notify_status_callbacks("request_cancelled", request_id)
                    logger.info(f"Active request {request_id} cancelled")
                    return True
            
            # Check if request is in queue
            if self._remove_from_queue(request_id):
                self.statistics.queued_requests -= 1
                self.statistics.cancelled_requests += 1
                self.statistics.queue_length = self.request_queue.qsize()
                
                self._notify_status_callbacks("request_cancelled", request_id)
                logger.info(f"Queued request {request_id} cancelled")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to cancel request: {e}")
            return False
    
    def pause_queue(self):
        """Pause queue processing"""
        self.pause_event.set()
        logger.info("Queue processing paused")
    
    def resume_queue(self):
        """Resume queue processing"""
        self.pause_event.clear()
        logger.info("Queue processing resumed")
    
    def clear_queue(self, keep_active: bool = True):
        """Clear all queued requests"""
        try:
            cleared_count = 0
            
            # Clear the queue
            while not self.request_queue.empty():
                try:
                    self.request_queue.get_nowait()
                    cleared_count += 1
                except Empty:
                    break
            
            with self.queue_lock:
                self.statistics.queued_requests = 0
                self.statistics.queue_length = 0
                
                # Optionally cancel active requests
                if not keep_active:
                    for request_id in list(self.active_requests.keys()):
                        self.cancel_request(request_id)
            
            logger.info(f"Cleared {cleared_count} requests from queue")
            
        except Exception as e:
            logger.error(f"Failed to clear queue: {e}")
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get comprehensive queue status"""
        try:
            with self.queue_lock:
                # Update statistics
                self.statistics.queue_length = self.request_queue.qsize()
                self.statistics.estimated_wait_time = self._estimate_wait_time(0)
                
                # Calculate success rate
                total_finished = self.statistics.completed_requests + self.statistics.failed_requests
                if total_finished > 0:
                    self.statistics.success_rate = (
                        self.statistics.completed_requests / total_finished * 100
                    )
                
                return {
                    "statistics": {
                        "total_requests": self.statistics.total_requests,
                        "queued_requests": self.statistics.queued_requests,
                        "processing_requests": self.statistics.processing_requests,
                        "completed_requests": self.statistics.completed_requests,
                        "failed_requests": self.statistics.failed_requests,
                        "cancelled_requests": self.statistics.cancelled_requests,
                        "success_rate": self.statistics.success_rate,
                        "queue_length": self.statistics.queue_length,
                        "estimated_wait_time": self.statistics.estimated_wait_time
                    },
                    "active_requests": [
                        {
                            "id": req.id,
                            "model_type": req.model_type,
                            "progress": req.progress,
                            "elapsed_time": (
                                (datetime.now() - req.started_at).total_seconds()
                                if req.started_at else 0
                            )
                        }
                        for req in self.active_requests.values()
                    ],
                    "worker_status": {
                        "active_workers": len([t for t in self.worker_threads if t.is_alive()]),
                        "max_workers": self.max_concurrent_requests,
                        "paused": self.pause_event.is_set()
                    }
                }
                
        except Exception as e:
            logger.error(f"Failed to get queue status: {e}")
            return {"error": str(e)}
    
    def get_user_requests(self, user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get requests for a specific user"""
        try:
            user_requests = []
            
            # Active requests
            with self.queue_lock:
                for request in self.active_requests.values():
                    if request.user_id == user_id:
                        user_requests.append(request.to_dict())
            
            # Completed requests
            for request in self.completed_requests:
                if request.user_id == user_id:
                    user_requests.append(request.to_dict())
            
            # Failed requests
            for request in self.failed_requests:
                if request.user_id == user_id:
                    user_requests.append(request.to_dict())
            
            # Sort by creation time (newest first)
            user_requests.sort(key=lambda x: x["created_at"], reverse=True)
            
            return user_requests[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get user requests: {e}")
            return []
    
    def retry_failed_request(self, request_id: str) -> Optional[str]:
        """Retry a failed request"""
        try:
            # Find the failed request
            failed_request = None
            for i, request in enumerate(self.failed_requests):
                if request.id == request_id:
                    failed_request = self.failed_requests.pop(i)
                    break
            
            if not failed_request:
                return None
            
            # Check retry limit
            if failed_request.retry_count >= failed_request.max_retries:
                logger.warning(f"Request {request_id} exceeded max retries")
                self.failed_requests.append(failed_request)
                return None
            
            # Create new request with incremented retry count
            new_request = GenerationRequest(
                user_id=failed_request.user_id,
                model_type=failed_request.model_type,
                prompt=failed_request.prompt,
                image_path=failed_request.image_path,
                parameters=failed_request.parameters,
                priority=failed_request.priority,
                estimated_duration=failed_request.estimated_duration,
                callback=failed_request.callback,
                metadata=failed_request.metadata,
                retry_count=failed_request.retry_count + 1,
                max_retries=failed_request.max_retries
            )
            
            # Add to queue
            self.request_queue.put(new_request)
            
            with self.queue_lock:
                self.statistics.queued_requests += 1
                self.statistics.failed_requests -= 1
                self.statistics.queue_length = self.request_queue.qsize()
            
            logger.info(f"Retry request {new_request.id} created for failed {request_id}")
            return new_request.id
            
        except Exception as e:
            logger.error(f"Failed to retry request: {e}")
            return None
    
    def _start_workers(self):
        """Start worker threads"""
        for i in range(self.max_concurrent_requests):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"GenerationWorker-{i}",
                daemon=True
            )
            worker.start()
            self.worker_threads.append(worker)
    
    def _worker_loop(self):
        """Main worker loop"""
        while not self.shutdown_event.is_set():
            try:
                # Wait if paused
                if self.pause_event.is_set():
                    time.sleep(1)
                    continue
                
                # Get next request
                try:
                    request = self.request_queue.get(timeout=1)
                except Empty:
                    continue
                
                # Process the request
                self._process_request(request)
                
            except Exception as e:
                logger.error(f"Worker error: {e}")
                time.sleep(1)
    
    def _process_request(self, request: GenerationRequest):
        """Process a single generation request"""
        try:
            # Move to active requests
            with self.queue_lock:
                self.active_requests[request.id] = request
                request.status = RequestStatus.PROCESSING
                request.started_at = datetime.now()
                
                self.statistics.queued_requests -= 1
                self.statistics.processing_requests += 1
                self.statistics.queue_length = self.request_queue.qsize()
            
            self._notify_status_callbacks("request_started", request.id)
            
            # Simulate generation process (replace with actual generation)
            success, output_path, error_message = self._execute_generation(request)
            
            # Update request status
            request.completed_at = datetime.now()
            request.actual_duration = (
                request.completed_at - request.started_at
            ).total_seconds()
            
            with self.queue_lock:
                del self.active_requests[request.id]
                self.statistics.processing_requests -= 1
                
                if success:
                    request.status = RequestStatus.COMPLETED
                    request.output_path = output_path
                    request.progress = 100.0
                    
                    self.completed_requests.append(request)
                    self.statistics.completed_requests += 1
                    
                    self._notify_status_callbacks("request_completed", request.id)
                else:
                    request.status = RequestStatus.FAILED
                    request.error_message = error_message
                    
                    self.failed_requests.append(request)
                    self.statistics.failed_requests += 1
                    
                    self._notify_status_callbacks("request_failed", request.id)
                    
                    # Auto-retry if enabled
                    if (self.auto_retry_failed and 
                        request.retry_count < request.max_retries):
                        self.retry_failed_request(request.id)
            
            # Call user callback if provided
            if request.callback:
                try:
                    request.callback(request.id, success, output_path, error_message)
                except Exception as e:
                    logger.error(f"Callback error: {e}")
            
            # Update performance history
            self._update_performance_history(request)
            
        except Exception as e:
            logger.error(f"Request processing failed: {e}")
            
            # Mark as failed
            with self.queue_lock:
                if request.id in self.active_requests:
                    del self.active_requests[request.id]
                    self.statistics.processing_requests -= 1
                
                request.status = RequestStatus.FAILED
                request.error_message = str(e)
                request.completed_at = datetime.now()
                
                self.failed_requests.append(request)
                self.statistics.failed_requests += 1
            
            self._notify_status_callbacks("request_failed", request.id)
    
    def _execute_generation(self, request: GenerationRequest) -> Tuple[bool, Optional[str], Optional[str]]:
        """Execute the actual generation (placeholder implementation)"""
        try:
            # This is a placeholder - replace with actual generation logic
            import time
            import random
            
            # Simulate generation time
            generation_time = request.estimated_duration
            steps = 10
            step_time = generation_time / steps
            
            for i in range(steps):
                if self.shutdown_event.is_set():
                    return False, None, "Generation cancelled"
                
                time.sleep(step_time)
                request.progress = (i + 1) / steps * 100
                self._notify_progress_callbacks(request.id, request.progress)
            
            # Simulate success/failure
            if random.random() > 0.1:  # 90% success rate
                output_path = f"outputs/video_{request.id}.mp4"
                return True, output_path, None
            else:
                return False, None, "Simulated generation failure"
                
        except Exception as e:
            return False, None, str(e)
    
    def _estimate_duration(self, model_type: str, parameters: Dict[str, Any]) -> float:
        """Estimate generation duration based on parameters"""
        base_time = 120.0  # 2 minutes base
        
        # Adjust based on steps
        steps = parameters.get("steps", 50)
        time_multiplier = steps / 50.0
        
        # Adjust based on resolution
        resolution = parameters.get("resolution", "1280x720")
        if "1920x1080" in resolution:
            time_multiplier *= 1.5
        
        return base_time * time_multiplier
    
    def _estimate_wait_time(self, position: int) -> float:
        """Estimate wait time based on queue position"""
        if position == 0:
            return 0.0
        
        # Calculate average processing time
        avg_time = 120.0  # Default 2 minutes
        if self.completed_requests:
            total_time = sum(
                req.actual_duration or req.estimated_duration 
                for req in self.completed_requests[-10:]  # Last 10 requests
            )
            avg_time = total_time / min(len(self.completed_requests), 10)
        
        # Account for concurrent processing
        concurrent_factor = 1.0 / self.max_concurrent_requests
        
        return position * avg_time * concurrent_factor
    
    def _get_queue_position(self, request_id: str) -> Optional[int]:
        """Get position of request in queue (expensive operation)"""
        try:
            # This is expensive as it requires examining the entire queue
            temp_queue = []
            position = None
            
            # Extract all items to find position
            while not self.request_queue.empty():
                try:
                    request = self.request_queue.get_nowait()
                    temp_queue.append(request)
                    if request.id == request_id:
                        position = len(temp_queue)
                except Empty:
                    break
            
            # Put items back
            for request in temp_queue:
                self.request_queue.put(request)
            
            return position
            
        except Exception as e:
            logger.error(f"Failed to get queue position: {e}")
            return None
    
    def _remove_from_queue(self, request_id: str) -> bool:
        """Remove specific request from queue"""
        try:
            temp_queue = []
            found = False
            
            # Extract all items except the target
            while not self.request_queue.empty():
                try:
                    request = self.request_queue.get_nowait()
                    if request.id == request_id:
                        found = True
                    else:
                        temp_queue.append(request)
                except Empty:
                    break
            
            # Put remaining items back
            for request in temp_queue:
                self.request_queue.put(request)
            
            return found
            
        except Exception as e:
            logger.error(f"Failed to remove from queue: {e}")
            return False
    
    def _notify_status_callbacks(self, event: str, request_id: str):
        """Notify status callbacks"""
        for callback in self.status_callbacks:
            try:
                callback(event, request_id)
            except Exception as e:
                logger.error(f"Status callback error: {e}")
    
    def _notify_progress_callbacks(self, request_id: str, progress: float):
        """Notify progress callbacks"""
        for callback in self.progress_callbacks:
            try:
                callback(request_id, progress)
            except Exception as e:
                logger.error(f"Progress callback error: {e}")
    
    def _update_performance_history(self, request: GenerationRequest):
        """Update performance history for analytics"""
        try:
            history_entry = {
                "timestamp": time.time(),
                "model_type": request.model_type,
                "parameters": request.parameters,
                "estimated_duration": request.estimated_duration,
                "actual_duration": request.actual_duration,
                "success": request.status == RequestStatus.COMPLETED,
                "retry_count": request.retry_count
            }
            
            self.request_history.append(history_entry)
            
            # Keep only recent history
            cutoff_time = time.time() - (7 * 24 * 3600)  # 7 days
            self.request_history = [
                h for h in self.request_history 
                if h["timestamp"] > cutoff_time
            ]
            
        except Exception as e:
            logger.error(f"Failed to update performance history: {e}")
    
    def _load_queue_state(self):
        """Load saved queue state"""
        try:
            state_file = Path("queue_state.json")
            if state_file.exists():
                with open(state_file, 'r') as f:
                    state = json.load(f)
                
                # Restore statistics
                stats_data = state.get("statistics", {})
                for key, value in stats_data.items():
                    if hasattr(self.statistics, key):
                        setattr(self.statistics, key, value)
                
                logger.info("Queue state loaded")
                
        except Exception as e:
            logger.error(f"Failed to load queue state: {e}")
    
    def _save_queue_state(self):
        """Save current queue state"""
        try:
            if not self.save_queue_state:
                return
            
            state = {
                "timestamp": datetime.now().isoformat(),
                "statistics": {
                    "total_requests": self.statistics.total_requests,
                    "completed_requests": self.statistics.completed_requests,
                    "failed_requests": self.statistics.failed_requests,
                    "cancelled_requests": self.statistics.cancelled_requests
                }
            }
            
            state_file = Path("queue_state.json")
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save queue state: {e}")
    
    def shutdown(self):
        """Shutdown the queue manager"""
        try:
            logger.info("Shutting down queue manager...")
            
            # Signal shutdown
            self.shutdown_event.set()
            
            # Wait for workers to finish
            for worker in self.worker_threads:
                worker.join(timeout=5)
            
            # Save state
            self._save_queue_state()
            
            logger.info("Queue manager shutdown complete")
            
        except Exception as e:
            logger.error(f"Shutdown error: {e}")

# Global queue manager instance
_queue_manager = None

def get_queue_manager(config: Optional[Dict[str, Any]] = None) -> GenerationQueueManager:
    """Get or create global queue manager instance"""
    global _queue_manager
    if _queue_manager is None:
        _queue_manager = GenerationQueueManager(config)
    return _queue_manager