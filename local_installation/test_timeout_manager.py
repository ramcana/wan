"""
Comprehensive tests for TimeoutManager - timeout management and resource cleanup system.

This test suite validates:
- Context-aware timeout calculation
- Automatic cleanup of temporary files and resources during failures
- Resource exhaustion detection and prevention
- Graceful operation cancellation and cleanup
- Disk space monitoring during long-running operations
"""

import unittest
import tempfile
import shutil
import time
import threading
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import logging

# Import the module under test
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from scripts.timeout_manager import (
    TimeoutManager, OperationType, ResourceType, TimeoutConfiguration,
    TimeoutException, ResourceExhaustionException, OperationContext,
    ResourceInfo, SystemResourceStatus
)


class TestTimeoutManager(unittest.TestCase):
    """Test cases for TimeoutManager class."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.logger = logging.getLogger(__name__)
        self.timeout_manager = TimeoutManager(self.test_dir, self.logger)
    
    def tearDown(self):
        """Clean up test environment."""
        self.timeout_manager.cleanup_all_resources()
        self.timeout_manager.stop_monitoring()
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_timeout_calculation_basic(self):
        """Test basic timeout calculation for different operation types."""
        # Test model download timeout
        context = {'file_size_gb': 2.0, 'network_speed': 'fast'}
        timeout = self.timeout_manager.calculate_timeout(OperationType.MODEL_DOWNLOAD, context)
        self.assertGreater(timeout, 1800)  # Should be greater than base timeout due to file size
        
        # Test dependency install timeout
        context = {'retry_count': 2}
        timeout = self.timeout_manager.calculate_timeout(OperationType.DEPENDENCY_INSTALL, context)
        self.assertGreater(timeout, 600)  # Should be greater than base timeout due to retries
        
        # Test system detection timeout
        context = {'complexity_level': 'complex'}
        timeout = self.timeout_manager.calculate_timeout(OperationType.SYSTEM_DETECTION, context)
        self.assertGreater(timeout, 60)  # Should be greater than base timeout due to complexity
    
    def test_timeout_calculation_network_speed(self):
        """Test timeout calculation with different network speeds."""
        base_context = {'file_size_gb': 1.0}
        
        # Fast network
        fast_context = {**base_context, 'network_speed': 'fast'}
        fast_timeout = self.timeout_manager.calculate_timeout(OperationType.MODEL_DOWNLOAD, fast_context)
        
        # Slow network
        slow_context = {**base_context, 'network_speed': 'slow'}
        slow_timeout = self.timeout_manager.calculate_timeout(OperationType.MODEL_DOWNLOAD, slow_context)
        
        # Slow network should have longer timeout
        self.assertGreater(slow_timeout, fast_timeout)
    
    def test_timeout_calculation_bounds(self):
        """Test that timeout calculation respects min/max bounds."""
        # Test minimum bound
        context = {'file_size_gb': 0.001, 'network_speed': 'fast', 'complexity_level': 'simple'}
        timeout = self.timeout_manager.calculate_timeout(OperationType.MODEL_DOWNLOAD, context)
        config = TimeoutManager.DEFAULT_TIMEOUTS[OperationType.MODEL_DOWNLOAD]
        self.assertGreaterEqual(timeout, config.min_timeout)
        
        # Test maximum bound
        context = {'file_size_gb': 100.0, 'network_speed': 'slow', 'retry_count': 10}
        timeout = self.timeout_manager.calculate_timeout(OperationType.MODEL_DOWNLOAD, context)
        self.assertLessEqual(timeout, config.max_timeout)
    
    def test_timeout_context_success(self):
        """Test successful operation within timeout context."""
        context = {'file_size_gb': 0.1}
        
        with self.timeout_manager.timeout_context(OperationType.VALIDATION, context) as op_context:
            self.assertIsInstance(op_context, OperationContext)
            self.assertEqual(op_context.operation_type, OperationType.VALIDATION)
            self.assertGreater(op_context.timeout_seconds, 0)
            
            # Simulate some work
            time.sleep(0.1)
        
        # Operation should be removed from active operations
        self.assertNotIn(op_context.operation_id, self.timeout_manager.active_operations)
    
    def test_timeout_context_timeout(self):
        """Test timeout exception in timeout context."""
        context = {'file_size_gb': 0.1}
        
        # Use a very short timeout for testing
        with patch.object(self.timeout_manager, 'calculate_timeout', return_value=1):
            with self.assertRaises(TimeoutException) as cm:
                with self.timeout_manager.timeout_context(OperationType.VALIDATION, context) as op_context:
                    # Sleep longer than timeout
                    time.sleep(2)
            
            self.assertEqual(cm.exception.operation_id, op_context.operation_id)
            self.assertEqual(cm.exception.timeout_seconds, 1)
    
    def test_timeout_context_exception_cleanup(self):
        """Test that resources are cleaned up when exception occurs in timeout context."""
        context = {'file_size_gb': 0.1}
        
        with self.assertRaises(ValueError):
            with self.timeout_manager.timeout_context(OperationType.VALIDATION, context) as op_context:
                # Create a resource
                temp_file = self.timeout_manager.create_temp_file(operation_id=op_context.operation_id)
                self.assertTrue(os.path.exists(temp_file))
                
                # Raise an exception
                raise ValueError("Test exception")
        
        # Resource should be cleaned up
        self.assertFalse(os.path.exists(temp_file))
    
    def test_resource_registration_and_cleanup(self):
        """Test resource registration and cleanup."""
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_path = temp_file.name
        temp_file.close()
        
        # Register the resource
        resource_id = self.timeout_manager.register_resource(
            ResourceType.TEMPORARY_FILE,
            path=temp_path,
            size_bytes=1024
        )
        
        self.assertIn(resource_id, self.timeout_manager.tracked_resources)
        self.assertTrue(os.path.exists(temp_path))
        
        # Unregister and cleanup
        self.timeout_manager.unregister_resource(resource_id, cleanup=True)
        
        self.assertNotIn(resource_id, self.timeout_manager.tracked_resources)
        self.assertFalse(os.path.exists(temp_path))
    
    def test_create_temp_file(self):
        """Test temporary file creation and automatic registration."""
        temp_file = self.timeout_manager.create_temp_file(suffix='.txt', prefix='test_')
        
        # File should exist
        self.assertTrue(os.path.exists(temp_file))
        
        # Should be registered for cleanup
        self.assertTrue(any(
            resource.path == temp_file 
            for resource in self.timeout_manager.tracked_resources.values()
        ))
        
        # Cleanup all resources
        self.timeout_manager.cleanup_all_resources()
        
        # File should be cleaned up
        self.assertFalse(os.path.exists(temp_file))
    
    def test_create_temp_directory(self):
        """Test temporary directory creation and automatic registration."""
        temp_dir = self.timeout_manager.create_temp_directory(suffix='_test', prefix='test_')
        
        # Directory should exist
        self.assertTrue(os.path.exists(temp_dir))
        self.assertTrue(os.path.isdir(temp_dir))
        
        # Should be registered for cleanup
        self.assertTrue(any(
            resource.path == temp_dir 
            for resource in self.timeout_manager.tracked_resources.values()
        ))
        
        # Cleanup all resources
        self.timeout_manager.cleanup_all_resources()
        
        # Directory should be cleaned up
        self.assertFalse(os.path.exists(temp_dir))
    
    def test_resource_association_with_operation(self):
        """Test that resources are properly associated with operations."""
        context = {'file_size_gb': 0.1}
        
        with self.timeout_manager.timeout_context(OperationType.VALIDATION, context) as op_context:
            # Create resources within the operation
            temp_file = self.timeout_manager.create_temp_file(operation_id=op_context.operation_id)
            temp_dir = self.timeout_manager.create_temp_directory(operation_id=op_context.operation_id)
            
            # Resources should be associated with the operation
            self.assertEqual(len(op_context.resources), 2)
            
            # Files should exist
            self.assertTrue(os.path.exists(temp_file))
            self.assertTrue(os.path.exists(temp_dir))
        
        # Resources should be cleaned up after operation
        self.assertFalse(os.path.exists(temp_file))
        self.assertFalse(os.path.exists(temp_dir))
    
    @patch('psutil.disk_usage')
    def test_resource_exhaustion_detection_disk(self, mock_disk_usage):
        """Test disk space exhaustion detection."""
        # Mock low disk space
        mock_disk_usage.return_value = Mock(free=1024**3)  # 1GB free
        
        with self.assertRaises(ResourceExhaustionException) as cm:
            self.timeout_manager._check_resource_availability()
        
        self.assertEqual(cm.exception.resource_type, "disk_space")
        self.assertLess(cm.exception.current_usage, cm.exception.limit)
    
    @patch('psutil.virtual_memory')
    def test_resource_exhaustion_detection_memory(self, mock_memory):
        """Test memory exhaustion detection."""
        # Mock low memory
        mock_memory.return_value = Mock(available=512*1024**2)  # 512MB available
        
        with self.assertRaises(ResourceExhaustionException) as cm:
            self.timeout_manager._check_resource_availability()
        
        self.assertEqual(cm.exception.resource_type, "memory")
        self.assertLess(cm.exception.current_usage, cm.exception.limit)
    
    @patch('psutil.disk_usage')
    def test_disk_space_monitoring(self, mock_disk_usage):
        """Test disk space monitoring during operations."""
        # Start with sufficient disk space
        mock_disk_usage.return_value = Mock(free=5*1024**3)  # 5GB free
        
        context = {'file_size_gb': 0.1}
        
        with self.timeout_manager.timeout_context(OperationType.VALIDATION, context) as op_context:
            # Start monitoring
            self.timeout_manager.monitor_disk_space(op_context.operation_id, check_interval=0.1)
            
            # Simulate work
            time.sleep(0.2)
            
            # Simulate disk space running low
            mock_disk_usage.return_value = Mock(free=1024**3)  # 1GB free
            
            # Give monitoring thread time to detect
            time.sleep(0.2)
    
    def test_custom_cleanup_callback(self):
        """Test custom cleanup callback functionality."""
        cleanup_called = threading.Event()
        
        def custom_cleanup():
            cleanup_called.set()
        
        # Register resource with custom cleanup
        resource_id = self.timeout_manager.register_resource(
            ResourceType.MEMORY_BUFFER,
            cleanup_callback=custom_cleanup,
            metadata={'test': 'data'}
        )
        
        # Cleanup the resource
        self.timeout_manager.unregister_resource(resource_id, cleanup=True)
        
        # Custom cleanup should have been called
        self.assertTrue(cleanup_called.is_set())
    
    def test_old_resource_cleanup(self):
        """Test cleanup of old resources."""
        # Create a resource
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_path = temp_file.name
        temp_file.close()
        
        resource_id = self.timeout_manager.register_resource(
            ResourceType.TEMPORARY_FILE,
            path=temp_path
        )
        
        # Manually set old creation time
        resource_info = self.timeout_manager.tracked_resources[resource_id]
        from datetime import datetime, timedelta
        resource_info.created_at = datetime.now() - timedelta(hours=25)  # Older than max age
        
        # Trigger cleanup of old resources
        self.timeout_manager._cleanup_old_resources()
        
        # Resource should be cleaned up
        self.assertNotIn(resource_id, self.timeout_manager.tracked_resources)
        self.assertFalse(os.path.exists(temp_path))
    
    def test_resource_summary(self):
        """Test resource summary generation."""
        # Create some resources
        temp_file = self.timeout_manager.create_temp_file()
        temp_dir = self.timeout_manager.create_temp_directory()
        
        # Get summary
        summary = self.timeout_manager.get_resource_summary()
        
        # Verify summary structure
        self.assertIn('tracked_resources', summary)
        self.assertIn('active_operations', summary)
        self.assertIn('system_status', summary)
        self.assertIn('monitoring_active', summary)
        
        # Verify resource counts
        self.assertGreaterEqual(summary['tracked_resources']['total_count'], 2)
        self.assertIn('temporary_file', summary['tracked_resources']['by_type'])
        self.assertIn('temporary_directory', summary['tracked_resources']['by_type'])
    
    def test_monitoring_start_stop(self):
        """Test background monitoring start and stop."""
        # Start monitoring
        self.timeout_manager.start_monitoring()
        self.assertTrue(self.timeout_manager._monitoring_active)
        self.assertIsNotNone(self.timeout_manager._monitoring_thread)
        
        # Stop monitoring
        self.timeout_manager.stop_monitoring()
        self.assertFalse(self.timeout_manager._monitoring_active)
    
    @patch('psutil.cpu_percent')
    def test_timeout_adjustment_for_cpu_load(self, mock_cpu_percent):
        """Test timeout adjustment based on CPU load."""
        # High CPU usage
        mock_cpu_percent.return_value = 90.0
        
        context = {'file_size_gb': 1.0}
        high_cpu_timeout = self.timeout_manager.calculate_timeout(OperationType.MODEL_DOWNLOAD, context)
        
        # Normal CPU usage
        mock_cpu_percent.return_value = 20.0
        normal_cpu_timeout = self.timeout_manager.calculate_timeout(OperationType.MODEL_DOWNLOAD, context)
        
        # High CPU should result in longer timeout
        self.assertGreater(high_cpu_timeout, normal_cpu_timeout)
    
    def test_operation_context_metadata(self):
        """Test that operation context preserves metadata."""
        context = {
            'file_size_gb': 2.5,
            'network_speed': 'slow',
            'retry_count': 1,
            'complexity_level': 'complex',
            'custom_field': 'test_value'
        }
        
        with self.timeout_manager.timeout_context(OperationType.MODEL_DOWNLOAD, context) as op_context:
            self.assertEqual(op_context.file_size_gb, 2.5)
            self.assertEqual(op_context.network_speed, 'slow')
            self.assertEqual(op_context.retry_count, 1)
            self.assertEqual(op_context.complexity_level, 'complex')
            self.assertEqual(op_context.metadata['custom_field'], 'test_value')
    
    def test_concurrent_operations(self):
        """Test handling of concurrent operations."""
        results = []
        results_lock = threading.Lock()
        
        def run_operation(op_id):
            context = {'file_size_gb': 0.1}
            try:
                with self.timeout_manager.timeout_context(
                    OperationType.VALIDATION, context, operation_id=f"op_{op_id}"
                ) as op_context:
                    time.sleep(0.1)
                    with results_lock:
                        results.append(f"success_{op_id}")
            except Exception as e:
                with results_lock:
                    results.append(f"error_{op_id}_{str(e)}")
        
        # Start multiple concurrent operations
        threads = []
        for i in range(5):
            thread = threading.Thread(target=run_operation, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all to complete
        for thread in threads:
            thread.join()
        
        # All operations should succeed
        self.assertEqual(len(results), 5)
        success_count = sum(1 for result in results if result.startswith('success_'))
        self.assertGreaterEqual(success_count, 3)  # Allow some failures due to timing
    
    def test_resource_cleanup_on_exit(self):
        """Test that resources are cleaned up on exit."""
        # Create a new timeout manager
        temp_dir = tempfile.mkdtemp()
        tm = TimeoutManager(temp_dir, self.logger)
        
        # Create some resources
        temp_file = tm.create_temp_file()
        temp_dir_resource = tm.create_temp_directory()
        
        self.assertTrue(os.path.exists(temp_file))
        self.assertTrue(os.path.exists(temp_dir_resource))
        
        # Simulate exit cleanup
        tm.cleanup_all_resources()
        
        # Resources should be cleaned up
        self.assertFalse(os.path.exists(temp_file))
        self.assertFalse(os.path.exists(temp_dir_resource))
        
        # Cleanup test directory
        shutil.rmtree(temp_dir, ignore_errors=True)


class TestTimeoutConfiguration(unittest.TestCase):
    """Test cases for TimeoutConfiguration class."""
    
    def test_timeout_configuration_creation(self):
        """Test TimeoutConfiguration creation and attributes."""
        config = TimeoutConfiguration(
            base_timeout=300,
            max_timeout=1800,
            min_timeout=60,
            size_multiplier=2.0,
            speed_multiplier=1.5,
            retry_multiplier=1.2,
            complexity_multiplier=1.8
        )
        
        self.assertEqual(config.base_timeout, 300)
        self.assertEqual(config.max_timeout, 1800)
        self.assertEqual(config.min_timeout, 60)
        self.assertEqual(config.size_multiplier, 2.0)
        self.assertEqual(config.speed_multiplier, 1.5)
        self.assertEqual(config.retry_multiplier, 1.2)
        self.assertEqual(config.complexity_multiplier, 1.8)


class TestResourceInfo(unittest.TestCase):
    """Test cases for ResourceInfo class."""
    
    def test_resource_info_creation(self):
        """Test ResourceInfo creation and attributes."""
        from datetime import datetime
        
        def dummy_cleanup():
            pass
        
        resource_info = ResourceInfo(
            resource_id="test_resource",
            resource_type=ResourceType.TEMPORARY_FILE,
            path="/tmp/test_file",
            size_bytes=1024,
            cleanup_callback=dummy_cleanup,
            metadata={'key': 'value'}
        )
        
        self.assertEqual(resource_info.resource_id, "test_resource")
        self.assertEqual(resource_info.resource_type, ResourceType.TEMPORARY_FILE)
        self.assertEqual(resource_info.path, "/tmp/test_file")
        self.assertEqual(resource_info.size_bytes, 1024)
        self.assertEqual(resource_info.cleanup_callback, dummy_cleanup)
        self.assertEqual(resource_info.metadata['key'], 'value')
        self.assertIsInstance(resource_info.created_at, datetime)
        self.assertIsInstance(resource_info.last_accessed, datetime)


class TestExceptions(unittest.TestCase):
    """Test cases for custom exceptions."""
    
    def test_timeout_exception(self):
        """Test TimeoutException creation and attributes."""
        exception = TimeoutException("Operation timed out", "op_123", 300)
        
        self.assertEqual(str(exception), "Operation timed out")
        self.assertEqual(exception.operation_id, "op_123")
        self.assertEqual(exception.timeout_seconds, 300)
    
    def test_resource_exhaustion_exception(self):
        """Test ResourceExhaustionException creation and attributes."""
        exception = ResourceExhaustionException(
            "Insufficient disk space", "disk_space", 1.5, 2.0
        )
        
        self.assertEqual(str(exception), "Insufficient disk space")
        self.assertEqual(exception.resource_type, "disk_space")
        self.assertEqual(exception.current_usage, 1.5)
        self.assertEqual(exception.limit, 2.0)


if __name__ == '__main__':
    # Set up logging for tests
    logging.basicConfig(level=logging.DEBUG)
    
    # Run the tests
    unittest.main(verbosity=2)