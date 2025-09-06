from unittest.mock import Mock, patch
#!/usr/bin/env python3
"""
Basic test script for the maintenance scheduler components.
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)

def test_models():
    """Test the models module."""
    print("Testing models...")
    try:
        from models import MaintenanceTask, TaskCategory, TaskPriority
        
        task = MaintenanceTask(
            name="Test Task",
            category=TaskCategory.CODE_QUALITY,
            priority=TaskPriority.HIGH
        )
        
        print(f"✓ Created task: {task.name} ({task.id})")
        return True
    except Exception as e:
        print(f"✗ Models test failed: {e}")
        return False

def test_task_manager():
    """Test the task manager."""
    print("Testing task manager...")
    try:
        from task_manager import MaintenanceTaskManager
        from models import MaintenanceTask, TaskCategory, TaskPriority
        
        # Create temp storage
        temp_storage = "data/test_tasks.json"
        Path("data").mkdir(exist_ok=True)
        
        manager = MaintenanceTaskManager(temp_storage)
        
        task = MaintenanceTask(
            name="Test Task Manager",
            category=TaskCategory.CODE_QUALITY,
            priority=TaskPriority.MEDIUM
        )
        
        manager.add_task(task)
        retrieved_task = manager.get_task(task.id)
        
        if retrieved_task and retrieved_task.name == task.name:
            print(f"✓ Task manager working: {retrieved_task.name}")
            return True
        else:
            print("✗ Task manager failed: task not retrieved correctly")
            return False
            
    except Exception as e:
        print(f"✗ Task manager test failed: {e}")
        return False

def test_priority_engine():
    """Test the priority engine."""
    print("Testing priority engine...")
    try:
        from priority_engine import TaskPriorityEngine
        from models import MaintenanceTask, TaskCategory, TaskPriority
        
        engine = TaskPriorityEngine()
        
        task = MaintenanceTask(
            name="Test Priority Task",
            category=TaskCategory.SECURITY_SCAN,
            priority=TaskPriority.CRITICAL
        )
        
        score = engine.get_priority_score(task)
        impact = engine.analyze_impact(task)
        
        print(f"✓ Priority engine working: score={score:.1f}, impact={impact.impact_category}")
        return True
        
    except Exception as e:
        print(f"✗ Priority engine test failed: {e}")
        return False

def test_history_tracker():
    """Test the history tracker."""
    print("Testing history tracker...")
    try:
        from history_tracker import MaintenanceHistoryTracker
        from models import MaintenanceTask, MaintenanceResult, TaskStatus, TaskCategory
        
        # Create temp storage
        temp_storage = "data/test_history.json"
        Path("data").mkdir(exist_ok=True)
        
        tracker = MaintenanceHistoryTracker(temp_storage)
        
        task = MaintenanceTask(
            name="Test History Task",
            category=TaskCategory.CODE_QUALITY
        )
        
        result = MaintenanceResult(
            task_id=task.id,
            status=TaskStatus.COMPLETED,
            started_at=datetime.now(),
            success=True,
            files_modified=5,
            issues_fixed=3
        )
        
        tracker.record_execution(task, result)
        history = tracker.get_task_history(task.id)
        
        if history and len(history) == 1:
            print(f"✓ History tracker working: recorded {len(history)} execution(s)")
            return True
        else:
            print("✗ History tracker failed: execution not recorded")
            return False
            
    except Exception as e:
        print(f"✗ History tracker test failed: {e}")
        return False

async def test_basic_scheduler():
    """Test basic scheduler functionality."""
    print("Testing basic scheduler...")
    try:
        from scheduler import MaintenanceScheduler
        from models import MaintenanceTask, TaskCategory, TaskPriority
        
        scheduler = MaintenanceScheduler({'max_concurrent_tasks': 1})
        
        # Create a simple task
        task = MaintenanceTask(
            name="Test Scheduler Task",
            category=TaskCategory.CODE_QUALITY,
            priority=TaskPriority.MEDIUM
        )
        
        # Mock executor
        async def mock_executor(config):
            await asyncio.sleep(0.1)
            return {
                'success': True,
                'output': 'Test task completed',
                'files_modified': 1,
                'issues_fixed': 1
            }
        
        task.executor = mock_executor
        
        # Add task and run
        scheduler.add_task(task)
        result = scheduler.run_task_now(task.id)
        
        if result and result.success:
            print(f"✓ Basic scheduler working: {result.output}")
            return True
        else:
            print("✗ Basic scheduler failed: task execution failed")
            return False
            
    except Exception as e:
        print(f"✗ Basic scheduler test failed: {e}")
        return False

async def main():
    """Run all basic tests."""
    print("Running basic maintenance scheduler tests...")
    print("=" * 50)
    
    # Ensure data directory exists
    Path("data").mkdir(exist_ok=True)
    
    tests = [
        test_models,
        test_task_manager,
        test_priority_engine,
        test_history_tracker,
        test_basic_scheduler
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if asyncio.iscoroutinefunction(test):
                result = await test()
            else:
                result = test()
            
            if result:
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
        
        print()  # Add spacing between tests
    
    print("=" * 50)
    print(f"Tests completed: {passed}/{total} passed")
    
    if passed == total:
        print("✓ All basic tests passed!")
        return True
    else:
        print("✗ Some tests failed")
        return False

if __name__ == '__main__':
    success = asyncio.run(main())
    exit(0 if success else 1)