#!/usr/bin/env python3
"""
Test script for queue management system
Tests the GenerationTask, TaskQueue, and QueueProcessor functionality
"""

import sys
import time
import threading
from PIL import Image
from utils import (
    GenerationTask, TaskStatus, TaskQueue, QueueProcessor, QueueManager,
    add_to_generation_queue, get_queue_comprehensive_status,
    start_queue_processing, stop_queue_processing
)

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

def test_queue_manager_basic():
    """Test basic QueueManager functionality"""
    print("Testing QueueManager basic functionality...")
    
    manager = QueueManager()
    
    # Add a task
    task_id = manager.add_task(
        model_type="t2v-A14B",
        prompt="Test video generation",
        resolution="1280x720"
    )
    
    assert task_id is not None
    
    # Get status
    status = manager.get_comprehensive_status()
    assert status["queue"]["total_pending"] >= 1
    
    # Get task details
    task_details = manager.get_task_details(task_id)
    assert task_details is not None
    assert task_details["prompt"] == "Test video generation"
    
    # Clean up
    manager.clear_all_tasks()
    
    print("✓ QueueManager basic tests passed")

def test_convenience_functions():
    """Test convenience functions"""
    print("Testing convenience functions...")
    
    # Add task using convenience function
    task_id = add_to_generation_queue(
        model_type="i2v-A14B",
        prompt="Test convenience function",
        resolution="1920x1080",
        steps=30
    )
    
    assert task_id is not None
    
    # Get status using convenience function
    status = get_queue_comprehensive_status()
    assert "queue" in status
    assert "processing" in status
    assert "system" in status
    
    # Stop processing to clean up
    stop_queue_processing()
    
    print("✓ Convenience function tests passed")

def run_all_tests():
    """Run all queue management tests"""
    print("Running Queue Management System Tests")
    print("=" * 50)
    
    try:
        test_generation_task()
        test_task_queue()
        test_queue_manager_basic()
        test_convenience_functions()
        
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