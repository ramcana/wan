#!/usr/bin/env python3
"""
Minimal test script for queue management system
Tests the core queue functionality without heavy dependencies
"""

import sys
import time
import threading
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
from queue import Queue
from enum import Enum

# Copy the core classes for testing without importing utils.py
class TaskStatus(Enum):
    """Enumeration for task status values"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class GenerationTask:
    """Data structure for video generation tasks"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model_type: str = ""  # 't2v-A14B', 'i2v-A14B', 'ti2v-5B'
    prompt: str = ""
    image: Optional[Any] = None  # Using Any instead of PIL.Image for testing
    resolution: str = "1280x720"
    steps: int = 50
    lora_path: Optional[str] = None
    lora_strength: float = 1.0
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    progress: float = 0.0  # Progress percentage (0.0 to 100.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for serialization"""
        return {
            "id": self.id,
            "model_type": self.model_type,
            "prompt": self.prompt,
            "image": None if self.image is None else "Image object",
            "resolution": self.resolution,
            "steps": self.steps,
            "lora_path": self.lora_path,
            "lora_strength": self.lora_strength,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "output_path": self.output_path,
            "error_message": self.error_message,
            "progress": self.progress
        }
    
    def update_status(self, status: TaskStatus, error_message: Optional[str] = None):
        """Update task status with optional error message"""
        self.status = status
        if error_message:
            self.error_message = error_message
        if status == TaskStatus.COMPLETED or status == TaskStatus.FAILED:
            self.completed_at = datetime.now()

class TaskQueue:
    """Thread-safe FIFO queue for managing generation tasks"""
    
    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self._queue = Queue(maxsize=max_size)
        self._tasks = {}  # Task ID -> GenerationTask mapping
        self._lock = threading.RLock()  # Reentrant lock for thread safety
        self._current_task: Optional[GenerationTask] = None
        self._processing = False
        
    def add_task(self, task: GenerationTask) -> bool:
        """Add a task to the queue. Returns True if successful, False if queue is full"""
        with self._lock:
            try:
                # Check if queue is full
                if self._queue.full():
                    print(f"Queue is full (max size: {self.max_size}), cannot add task {task.id}")
                    return False
                
                # Add task to queue and tracking dict
                self._queue.put(task, block=False)
                self._tasks[task.id] = task
                
                print(f"Added task {task.id} to queue (model: {task.model_type}, prompt: '{task.prompt[:50]}...')")
                return True
                
            except Exception as e:
                print(f"Failed to add task to queue: {e}")
                return False
    
    def get_next_task(self) -> Optional[GenerationTask]:
        """Get the next task from the queue (FIFO order). Returns None if queue is empty"""
        with self._lock:
            try:
                if self._queue.empty():
                    return None
                
                task = self._queue.get(block=False)
                self._current_task = task
                task.update_status(TaskStatus.PROCESSING)
                
                print(f"Retrieved task {task.id} from queue")
                return task
                
            except Exception as e:
                print(f"Failed to get next task from queue: {e}")
                return None
    
    def complete_task(self, task_id: str, output_path: Optional[str] = None, error_message: Optional[str] = None):
        """Mark a task as completed or failed"""
        with self._lock:
            if task_id not in self._tasks:
                print(f"Task {task_id} not found in queue")
                return
            
            task = self._tasks[task_id]
            
            if error_message:
                task.update_status(TaskStatus.FAILED, error_message)
                print(f"Task {task_id} failed: {error_message}")
            else:
                task.update_status(TaskStatus.COMPLETED)
                task.output_path = output_path
                print(f"Task {task_id} completed successfully")
            
            # Clear current task if this was it
            if self._current_task and self._current_task.id == task_id:
                self._current_task = None
    
    def update_task_progress(self, task_id: str, progress: float):
        """Update the progress of a task (0.0 to 100.0)"""
        with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id].progress = max(0.0, min(100.0, progress))
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a specific task"""
        with self._lock:
            if task_id in self._tasks:
                return self._tasks[task_id].to_dict()
            return None
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get comprehensive queue status information"""
        with self._lock:
            pending_tasks = []
            completed_tasks = []
            failed_tasks = []
            
            for task in self._tasks.values():
                task_dict = task.to_dict()
                if task.status == TaskStatus.PENDING:
                    pending_tasks.append(task_dict)
                elif task.status == TaskStatus.COMPLETED:
                    completed_tasks.append(task_dict)
                elif task.status == TaskStatus.FAILED:
                    failed_tasks.append(task_dict)
            
            # Sort by creation time
            pending_tasks.sort(key=lambda x: x['created_at'])
            completed_tasks.sort(key=lambda x: x['created_at'], reverse=True)
            failed_tasks.sort(key=lambda x: x['created_at'], reverse=True)
            
            current_task_dict = None
            if self._current_task:
                current_task_dict = self._current_task.to_dict()
            
            return {
                "queue_size": self._queue.qsize(),
                "max_size": self.max_size,
                "is_processing": self._processing,
                "current_task": current_task_dict,
                "pending_tasks": pending_tasks,
                "completed_tasks": completed_tasks[:10],  # Limit to last 10
                "failed_tasks": failed_tasks[:10],  # Limit to last 10
                "total_pending": len(pending_tasks),
                "total_completed": len(completed_tasks),
                "total_failed": len(failed_tasks)
            }
    
    def is_empty(self) -> bool:
        """Check if the queue is empty"""
        return self._queue.empty()
    
    def is_full(self) -> bool:
        """Check if the queue is full"""
        return self._queue.full()
    
    def size(self) -> int:
        """Get the current queue size"""
        return self._queue.qsize()

def test_generation_task():
    """Test GenerationTask class functionality"""
    print("Testing GenerationTask class...")
    
    # Create a basic task
    task = GenerationTask(
        model_type="t2v-A14B",
        prompt="A beautiful sunset over mountains",
        resolution="1280x720",
        steps=50
    )
    
    # Test initial state
    assert task.status == TaskStatus.PENDING
    assert task.progress == 0.0
    assert task.error_message is None
    assert task.output_path is None
    
    # Test status update
    task.update_status(TaskStatus.PROCESSING)
    assert task.status == TaskStatus.PROCESSING
    
    # Test completion
    task.update_status(TaskStatus.COMPLETED)
    assert task.status == TaskStatus.COMPLETED
    assert task.completed_at is not None
    
    # Test serialization
    task_dict = task.to_dict()
    assert task_dict["model_type"] == "t2v-A14B"
    assert task_dict["prompt"] == "A beautiful sunset over mountains"
    assert task_dict["status"] == "completed"
    
    print("✓ GenerationTask tests passed")

def test_task_queue():
    """Test TaskQueue class functionality"""
    print("Testing TaskQueue class...")
    
    # Create a small queue for testing
    queue = TaskQueue(max_size=3)
    
    # Test empty queue
    assert queue.is_empty()
    assert not queue.is_full()
    assert queue.size() == 0
    
    # Add tasks
    task1 = GenerationTask(model_type="t2v-A14B", prompt="Test 1")
    task2 = GenerationTask(model_type="i2v-A14B", prompt="Test 2")
    task3 = GenerationTask(model_type="ti2v-5B", prompt="Test 3")
    
    assert queue.add_task(task1)
    assert queue.add_task(task2)
    assert queue.add_task(task3)
    
    # Queue should be full now
    assert queue.is_full()
    assert queue.size() == 3
    
    # Try to add another task (should fail)
    task4 = GenerationTask(model_type="t2v-A14B", prompt="Test 4")
    assert not queue.add_task(task4)
    
    # Get tasks in FIFO order
    next_task = queue.get_next_task()
    assert next_task is not None
    assert next_task.prompt == "Test 1"
    assert next_task.status == TaskStatus.PROCESSING
    
    # Complete the task
    queue.complete_task(next_task.id, output_path="/test/output.mp4")
    
    # Check queue status
    status = queue.get_queue_status()
    assert status["queue_size"] == 2
    assert status["total_pending"] == 2
    assert status["total_completed"] == 1
    
    print("✓ TaskQueue tests passed")

def test_thread_safety():
    """Test thread safety of TaskQueue"""
    print("Testing TaskQueue thread safety...")
    
    queue = TaskQueue(max_size=100)
    results = []
    
    def add_tasks(start_id, count):
        """Add tasks from multiple threads"""
        for i in range(count):
            task = GenerationTask(
                model_type="t2v-A14B",
                prompt=f"Thread task {start_id}-{i}"
            )
            success = queue.add_task(task)
            results.append(success)
    
    def process_tasks():
        """Process tasks from queue"""
        processed = 0
        while processed < 20:  # Process 20 tasks
            task = queue.get_next_task()
            if task:
                # Simulate processing
                time.sleep(0.01)
                queue.complete_task(task.id, output_path=f"/test/{task.id}.mp4")
                processed += 1
            else:
                time.sleep(0.01)
    
    # Start multiple threads
    threads = []
    
    # Add tasks from multiple threads
    for i in range(4):
        thread = threading.Thread(target=add_tasks, args=(i, 5))
        threads.append(thread)
        thread.start()
    
    # Process tasks
    processor_thread = threading.Thread(target=process_tasks)
    threads.append(processor_thread)
    processor_thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Check results
    assert len(results) == 20  # 4 threads * 5 tasks each
    assert all(results)  # All tasks should be added successfully
    
    status = queue.get_queue_status()
    assert status["total_completed"] == 20
    
    print("✓ Thread safety tests passed")

def run_all_tests():
    """Run all queue management tests"""
    print("Running Queue Management System Tests")
    print("=" * 50)
    
    try:
        test_generation_task()
        test_task_queue()
        test_thread_safety()
        
        print("=" * 50)
        print("✓ All queue management tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)