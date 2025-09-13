"""
Demo script for TimeoutManager - timeout management and resource cleanup system.

This script demonstrates the key features of the TimeoutManager:
- Context-aware timeout calculation
- Automatic resource cleanup
- Resource exhaustion detection
- Graceful operation cancellation
- Disk space monitoring
"""

import os
import sys
import time
import logging
from pathlib import Path

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from scripts.timeout_manager import (
    TimeoutManager, OperationType, ResourceType, 
    TimeoutException, ResourceExhaustionException
)


def setup_logging():
    """Set up logging for the demo."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def demo_timeout_calculation(timeout_manager, logger):
    """Demonstrate context-aware timeout calculation."""
    logger.info("=== Timeout Calculation Demo ===")
    
    # Different operation types with various contexts
    test_cases = [
        {
            'operation': OperationType.MODEL_DOWNLOAD,
            'context': {'file_size_gb': 5.0, 'network_speed': 'slow', 'retry_count': 0},
            'description': 'Large model download on slow network'
        },
        {
            'operation': OperationType.DEPENDENCY_INSTALL,
            'context': {'retry_count': 2, 'complexity_level': 'complex'},
            'description': 'Complex dependency installation (3rd retry)'
        },
        {
            'operation': OperationType.SYSTEM_DETECTION,
            'context': {'complexity_level': 'simple'},
            'description': 'Simple system detection'
        },
        {
            'operation': OperationType.NETWORK_TEST,
            'context': {'network_speed': 'fast'},
            'description': 'Network connectivity test on fast connection'
        }
    ]
    
    for case in test_cases:
        timeout = timeout_manager.calculate_timeout(case['operation'], case['context'])
        logger.info(f"{case['description']}: {timeout}s timeout")
    
    logger.info("")


def demo_resource_management(timeout_manager, logger):
    """Demonstrate automatic resource management."""
    logger.info("=== Resource Management Demo ===")
    
    # Create some temporary resources
    temp_file = timeout_manager.create_temp_file(suffix='.demo', prefix='demo_')
    temp_dir = timeout_manager.create_temp_directory(suffix='_demo', prefix='demo_')
    
    logger.info(f"Created temporary file: {temp_file}")
    logger.info(f"Created temporary directory: {temp_dir}")
    
    # Verify they exist
    logger.info(f"File exists: {os.path.exists(temp_file)}")
    logger.info(f"Directory exists: {os.path.exists(temp_dir)}")
    
    # Get resource summary
    summary = timeout_manager.get_resource_summary()
    logger.info(f"Tracked resources: {summary['tracked_resources']['total_count']}")
    
    # Manual cleanup
    timeout_manager.cleanup_all_resources()
    
    # Verify cleanup
    logger.info(f"After cleanup - File exists: {os.path.exists(temp_file)}")
    logger.info(f"After cleanup - Directory exists: {os.path.exists(temp_dir)}")
    logger.info("")


def demo_timeout_context(timeout_manager, logger):
    """Demonstrate timeout context manager."""
    logger.info("=== Timeout Context Demo ===")
    
    # Successful operation
    try:
        context = {'file_size_gb': 0.1, 'network_speed': 'fast'}
        with timeout_manager.timeout_context(OperationType.VALIDATION, context) as op_context:
            logger.info(f"Operation {op_context.operation_id} started with {op_context.timeout_seconds}s timeout")
            
            # Create resources within the operation
            temp_file = timeout_manager.create_temp_file(operation_id=op_context.operation_id)
            logger.info(f"Created resource: {temp_file}")
            
            # Simulate work
            time.sleep(0.5)
            logger.info("Operation completed successfully")
        
        # Resources should be automatically cleaned up
        logger.info(f"Resource cleaned up: {not os.path.exists(temp_file)}")
        
    except Exception as e:
        logger.error(f"Operation failed: {e}")
    
    logger.info("")


def demo_timeout_exception(timeout_manager, logger):
    """Demonstrate timeout exception handling."""
    logger.info("=== Timeout Exception Demo ===")
    
    try:
        # Use a very short timeout for demo
        context = {'file_size_gb': 0.1}
        
        # Patch the timeout calculation for demo purposes
        original_calculate = timeout_manager.calculate_timeout
        timeout_manager.calculate_timeout = lambda op_type, ctx: 1  # 1 second timeout
        
        with timeout_manager.timeout_context(OperationType.VALIDATION, context) as op_context:
            logger.info(f"Operation {op_context.operation_id} started with short timeout")
            
            # Create a resource
            temp_file = timeout_manager.create_temp_file(operation_id=op_context.operation_id)
            logger.info(f"Created resource: {temp_file}")
            
            # Sleep longer than timeout
            logger.info("Sleeping for 2 seconds (longer than timeout)...")
            time.sleep(2)
        
    except TimeoutException as e:
        logger.info(f"Caught timeout exception: {e}")
        logger.info(f"Operation ID: {e.operation_id}")
        logger.info(f"Timeout: {e.timeout_seconds}s")
        
        # Resource should be cleaned up automatically
        logger.info(f"Resource cleaned up: {not os.path.exists(temp_file)}")
    
    finally:
        # Restore original method
        timeout_manager.calculate_timeout = original_calculate
    
    logger.info("")


def demo_resource_monitoring(timeout_manager, logger):
    """Demonstrate resource monitoring."""
    logger.info("=== Resource Monitoring Demo ===")
    
    # Start monitoring
    timeout_manager.start_monitoring()
    logger.info("Started background resource monitoring")
    
    # Create some resources and let monitoring run
    resources = []
    for i in range(3):
        temp_file = timeout_manager.create_temp_file(prefix=f'monitor_test_{i}_')
        resources.append(temp_file)
        logger.info(f"Created resource {i+1}: {temp_file}")
    
    # Get system status
    summary = timeout_manager.get_resource_summary()
    logger.info(f"System status: {summary['system_status']}")
    
    # Let monitoring run for a bit
    time.sleep(2)
    
    # Stop monitoring
    timeout_manager.stop_monitoring()
    logger.info("Stopped background resource monitoring")
    
    # Cleanup
    timeout_manager.cleanup_all_resources()
    logger.info("Cleaned up all resources")
    logger.info("")


def demo_concurrent_operations(timeout_manager, logger):
    """Demonstrate concurrent operations."""
    logger.info("=== Concurrent Operations Demo ===")
    
    import threading
    
    results = []
    results_lock = threading.Lock()
    
    def run_operation(op_id):
        try:
            context = {'file_size_gb': 0.1}
            with timeout_manager.timeout_context(
                OperationType.FILE_OPERATION, context, operation_id=f"concurrent_op_{op_id}"
            ) as op_context:
                # Create a resource
                temp_file = timeout_manager.create_temp_file(operation_id=op_context.operation_id)
                
                # Simulate work
                time.sleep(0.2)
                
                with results_lock:
                    results.append(f"success_{op_id}")
                    logger.info(f"Operation {op_id} completed successfully")
        
        except Exception as e:
            with results_lock:
                results.append(f"error_{op_id}")
                logger.error(f"Operation {op_id} failed: {e}")
    
    # Start multiple concurrent operations
    threads = []
    for i in range(3):
        thread = threading.Thread(target=run_operation, args=(i,))
        threads.append(thread)
        thread.start()
    
    # Wait for all to complete
    for thread in threads:
        thread.join()
    
    logger.info(f"Concurrent operations results: {results}")
    logger.info("")


def main():
    """Main demo function."""
    logger = setup_logging()
    logger.info("Starting TimeoutManager Demo")
    
    # Create a temporary directory for the demo
    demo_dir = Path("temp_demo")
    demo_dir.mkdir(exist_ok=True)
    
    try:
        # Initialize TimeoutManager
        timeout_manager = TimeoutManager(str(demo_dir), logger)
        logger.info("TimeoutManager initialized")
        
        # Run demos
        demo_timeout_calculation(timeout_manager, logger)
        demo_resource_management(timeout_manager, logger)
        demo_timeout_context(timeout_manager, logger)
        demo_timeout_exception(timeout_manager, logger)
        demo_resource_monitoring(timeout_manager, logger)
        demo_concurrent_operations(timeout_manager, logger)
        
        logger.info("Demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise
    
    finally:
        # Cleanup demo directory
        import shutil
        if demo_dir.exists():
            shutil.rmtree(demo_dir, ignore_errors=True)
        logger.info("Demo cleanup completed")


if __name__ == "__main__":
    main()
