"""
Unit tests for enhanced error context system.
Tests the comprehensive system state capture functionality.
"""

import unittest
import tempfile
import shutil
import json
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock

import sys
sys.path.append('scripts')

from error_handler import (
    SystemStateCollector, EnhancedErrorContext, SystemInfo, 
    ResourceSnapshot, NetworkStatus, ComprehensiveErrorHandler
)
from interfaces import InstallationError, ErrorCategory, HardwareProfile


class TestSystemStateCollector(unittest.TestCase):
    """Test system state collection functionality."""
    
    def setUp(self):
        self.collector = SystemStateCollector()
    
    def test_collect_system_info(self):
        """Test system information collection."""
        system_info = self.collector.collect_system_info()
        
        self.assertIsInstance(system_info, SystemInfo)
        self.assertIsInstance(system_info.os_version, str)
        self.assertIsInstance(system_info.python_version, str)
        self.assertIsInstance(system_info.available_memory_gb, float)
        self.assertIsInstance(system_info.available_disk_gb, float)
        self.assertIsInstance(system_info.cpu_usage_percent, float)
        self.assertIsInstance(system_info.installed_packages, dict)
        self.assertIsInstance(system_info.environment_vars, dict)
        
        # Check that we got reasonable values
        self.assertGreater(system_info.available_memory_gb, 0)
        self.assertGreater(system_info.available_disk_gb, 0)
        self.assertGreaterEqual(system_info.cpu_usage_percent, 0)
        self.assertLessEqual(system_info.cpu_usage_percent, 100)
        
        # Check OS version format
        self.assertTrue(any(os_name in system_info.os_version.lower() 
                          for os_name in ['windows', 'linux', 'darwin']))
        
        # Check Python version format
        self.assertTrue(system_info.python_version.startswith('Python'))
    
    def test_collect_resource_snapshot(self):
        """Test resource snapshot collection."""
        snapshot = self.collector.collect_resource_snapshot()
        
        self.assertIsInstance(snapshot, ResourceSnapshot)
        self.assertIsInstance(snapshot.memory_usage_mb, int)
        self.assertIsInstance(snapshot.disk_usage_gb, float)
        self.assertIsInstance(snapshot.cpu_usage_percent, float)
        self.assertIsInstance(snapshot.gpu_memory_usage_mb, int)
        self.assertIsInstance(snapshot.network_bandwidth_mbps, float)
        self.assertIsInstance(snapshot.open_file_handles, int)
        self.assertIsInstance(snapshot.process_count, int)
        
        # Check reasonable values
        self.assertGreater(snapshot.memory_usage_mb, 0)
        self.assertGreater(snapshot.disk_usage_gb, 0)
        self.assertGreaterEqual(snapshot.cpu_usage_percent, 0)
        self.assertGreaterEqual(snapshot.open_file_handles, 0)
        self.assertGreater(snapshot.process_count, 0)
    
    def test_collect_network_status(self):
        """Test network status collection."""
        network_status = self.collector.collect_network_status()
        
        self.assertIsInstance(network_status, NetworkStatus)
        self.assertIsInstance(network_status.connectivity, bool)
        self.assertIsInstance(network_status.latency_ms, float)
        self.assertIsInstance(network_status.bandwidth_mbps, float)
        self.assertIsInstance(network_status.proxy_configured, bool)
        self.assertIsInstance(network_status.dns_resolution, bool)
        
        # Latency should be reasonable if connectivity is True
        if network_status.connectivity:
            self.assertGreater(network_status.latency_ms, 0)
            self.assertLess(network_status.latency_ms, 10000)  # Less than 10 seconds
    
    @patch('psutil.virtual_memory')
    def test_collect_system_info_error_handling(self, mock_memory):
        """Test error handling in system info collection."""
        # Simulate psutil error
        mock_memory.side_effect = Exception("Test error")
        
        system_info = self.collector.collect_system_info()
        
        # Should return default values on error
        self.assertEqual(system_info.os_version, "Unknown")
        self.assertEqual(system_info.python_version, "Unknown")
        self.assertEqual(system_info.available_memory_gb, 0.0)


class TestEnhancedErrorContext(unittest.TestCase):
    """Test enhanced error context functionality."""
    
    def test_enhanced_error_context_creation(self):
        """Test creation of enhanced error context."""
        system_info = SystemInfo(
            os_version="Windows 11",
            python_version="Python 3.11.4",
            available_memory_gb=16.0,
            available_disk_gb=500.0,
            cpu_usage_percent=25.5,
            gpu_info={"name": "RTX 3070", "memory_mb": 8192},
            installed_packages={"numpy": "1.21.0"},
            environment_vars={"PATH": "/usr/bin"}
        )
        
        resources = ResourceSnapshot(
            memory_usage_mb=8192,
            disk_usage_gb=100.0,
            cpu_usage_percent=30.0,
            gpu_memory_usage_mb=4096,
            network_bandwidth_mbps=100.0,
            open_file_handles=50,
            process_count=200
        )
        
        network = NetworkStatus(
            connectivity=True,
            latency_ms=25.0,
            bandwidth_mbps=100.0,
            proxy_configured=False,
            dns_resolution=True,
            external_ip="192.168.1.100"
        )
        
        context = EnhancedErrorContext(
            phase="test_phase",
            task="test_task",
            component="test_component",
            method="test_method",
            system_info=system_info,
            system_resources=resources,
            network_status=network,
            retry_count=2,
            previous_errors=["Error 1", "Error 2"],
            recovery_attempts=["Retry", "Fallback"]
        )
        
        self.assertEqual(context.phase, "test_phase")
        self.assertEqual(context.task, "test_task")
        self.assertEqual(context.component, "test_component")
        self.assertEqual(context.method, "test_method")
        self.assertEqual(context.retry_count, 2)
        self.assertEqual(len(context.previous_errors), 2)
        self.assertEqual(len(context.recovery_attempts), 2)
        self.assertIsNotNone(context.system_info)
        self.assertIsNotNone(context.system_resources)
        self.assertIsNotNone(context.network_status)


class TestComprehensiveErrorHandlerEnhanced(unittest.TestCase):
    """Test enhanced comprehensive error handler functionality."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.handler = ComprehensiveErrorHandler(self.temp_dir)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_create_enhanced_error_context(self):
        """Test creation of enhanced error context."""
        context = self.handler.create_enhanced_error_context(
            phase="test_phase",
            task="test_task",
            component="ModelDownloader",
            method="download_models",
            retry_count=1
        )
        
        self.assertIsInstance(context, EnhancedErrorContext)
        self.assertEqual(context.phase, "test_phase")
        self.assertEqual(context.task, "test_task")
        self.assertEqual(context.component, "ModelDownloader")
        self.assertEqual(context.method, "download_models")
        self.assertEqual(context.retry_count, 1)
        self.assertIsNotNone(context.system_info)
        self.assertIsNotNone(context.system_resources)
        self.assertIsNotNone(context.network_status)
        self.assertIsInstance(context.timestamp, datetime)
    
    def test_enhanced_error_logging(self):
        """Test error logging with enhanced context."""
        error = InstallationError("Test error", ErrorCategory.NETWORK)
        context = self.handler.create_enhanced_error_context(
            phase="models",
            task="download",
            component="ModelDownloader",
            method="download_models_parallel"
        )
        
        # Log the error
        self.handler.log_error(error, context)
        
        # Check that error was logged to file
        error_log_file = Path(self.temp_dir) / "logs" / "errors.json"
        self.assertTrue(error_log_file.exists())
        
        # Read and verify log content
        with open(error_log_file, 'r', encoding='utf-8') as f:
            logged_errors = json.load(f)
        
        self.assertEqual(len(logged_errors), 1)
        logged_error = logged_errors[0]
        
        self.assertEqual(logged_error["error_message"], "Test error")
        self.assertEqual(logged_error["error_category"], "network")
        self.assertIn("context", logged_error)
        
        # Check that enhanced context was captured
        context_data = logged_error["context"]
        self.assertEqual(context_data["phase"], "models")
        self.assertEqual(context_data["task"], "download")
        self.assertEqual(context_data["component"], "ModelDownloader")
        self.assertEqual(context_data["method"], "download_models_parallel")
        self.assertIn("system_info", context_data)
        self.assertIn("system_resources", context_data)
        self.assertIn("network_status", context_data)
    
    def test_error_context_serialization(self):
        """Test serialization of enhanced error context."""
        context = self.handler.create_enhanced_error_context(
            phase="test_phase",
            task="test_task"
        )
        
        # Test serialization
        serialized = self.handler._serialize_error_context(context)
        
        self.assertIsInstance(serialized, dict)
        self.assertEqual(serialized["phase"], "test_phase")
        self.assertEqual(serialized["task"], "test_task")
        self.assertIn("timestamp", serialized)
        self.assertIn("system_info", serialized)
        self.assertIn("system_resources", serialized)
        self.assertIn("network_status", serialized)
        
        # Verify it's JSON serializable
        json_str = json.dumps(serialized)
        self.assertIsInstance(json_str, str)
    
    @patch('error_handler.SystemStateCollector.collect_system_info')
    def test_enhanced_context_error_handling(self, mock_collect):
        """Test error handling in enhanced context creation."""
        # Simulate error in system state collection
        mock_collect.side_effect = Exception("Collection failed")
        
        context = self.handler.create_enhanced_error_context(
            phase="test_phase",
            task="test_task"
        )
        
        # Should still create context with basic information
        self.assertIsInstance(context, EnhancedErrorContext)
        self.assertEqual(context.phase, "test_phase")
        self.assertEqual(context.task, "test_task")


class TestErrorContextIntegration(unittest.TestCase):
    """Integration tests for error context system."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.handler = ComprehensiveErrorHandler(self.temp_dir)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_full_error_handling_workflow(self):
        """Test complete error handling workflow with enhanced context."""
        # Create a realistic error scenario
        error = InstallationError(
            "Model validation failed: Found 3 model issues",
            ErrorCategory.VALIDATION,
            ["Check model files", "Re-download models"]
        )
        
        # Create enhanced context
        context = self.handler.create_enhanced_error_context(
            phase="validation",
            task="model_validation",
            component="InstallationValidator",
            method="validate_models",
            retry_count=0
        )
        
        # Handle the error
        recovery_action = self.handler.handle_error(error, context)
        
        # Verify error was handled
        self.assertIsNotNone(recovery_action)
        
        # Check error was logged with full context
        error_log_file = Path(self.temp_dir) / "logs" / "errors.json"
        self.assertTrue(error_log_file.exists())
        
        with open(error_log_file, 'r', encoding='utf-8') as f:
            logged_errors = json.load(f)
        
        self.assertEqual(len(logged_errors), 1)
        logged_error = logged_errors[0]
        
        # Verify comprehensive context was captured
        context_data = logged_error["context"]
        self.assertEqual(context_data["phase"], "validation")
        self.assertEqual(context_data["task"], "model_validation")
        self.assertEqual(context_data["component"], "InstallationValidator")
        self.assertEqual(context_data["method"], "validate_models")
        
        # Verify system state was captured
        self.assertIn("system_info", context_data)
        system_info = context_data["system_info"]
        self.assertIn("os_version", system_info)
        self.assertIn("python_version", system_info)
        self.assertIn("available_memory_gb", system_info)
        self.assertIn("available_disk_gb", system_info)
        
        # Verify resource snapshot was captured
        self.assertIn("system_resources", context_data)
        resources = context_data["system_resources"]
        self.assertIn("memory_usage_mb", resources)
        self.assertIn("cpu_usage_percent", resources)
        
        # Verify network status was captured
        self.assertIn("network_status", context_data)
        network = context_data["network_status"]
        self.assertIn("connectivity", network)
        self.assertIn("dns_resolution", network)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)