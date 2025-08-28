"""
Comprehensive tests for the enhanced user guidance system.

Tests cover:
- Enhanced error message formatting with recovery strategies
- Progress indicators with estimated completion times
- Recovery strategy explanations and success likelihood display
- Support ticket generation with pre-filled error reports
- Links to documentation and support resources
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
from pathlib import Path
import json
from datetime import datetime, timedelta
import time
import threading

# Import the modules to test
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))

from scripts.user_guidance import (
    UserGuidanceSystem, RecoveryStrategy, ProgressIndicator, SupportTicket,
    RecoveryStatus, SupportResourceType, SupportResource, InstallationError, ErrorCategory
)


class TestEnhancedUserGuidance(unittest.TestCase):
    """Test enhanced user guidance system functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.guidance = UserGuidanceSystem(self.test_dir)
        
        # Create logs directory
        self.logs_dir = Path(self.test_dir) / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        
        # Create sample log file
        sample_log = self.logs_dir / "test.log"
        sample_log.write_text("2024-01-01 10:00:00 - INFO - Test log entry\n"
                             "2024-01-01 10:01:00 - ERROR - Sample error\n")
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_enhanced_error_formatting(self):
        """Test enhanced error message formatting with recovery strategies."""
        # Create test error
        error = InstallationError(
            "Network connection failed",
            ErrorCategory.NETWORK,
            recovery_suggestions=["Check internet connection", "Configure proxy settings"]
        )
        
        # Create recovery strategies
        strategies = [
            RecoveryStrategy(
                name="Network Diagnostics",
                description="Run network connectivity tests",
                success_likelihood=0.8,
                estimated_time_minutes=5,
                steps=[{"description": "Ping test", "command": "ping google.com"}]
            ),
            RecoveryStrategy(
                name="Proxy Configuration",
                description="Configure proxy settings for corporate networks",
                success_likelihood=0.6,
                estimated_time_minutes=10,
                steps=[{"description": "Set proxy", "command": "set proxy"}],
                prerequisites=["Proxy server details"],
                risks=["May affect other network applications"]
            )
        ]
        
        # Test formatting
        context = {
            "timestamp": "2024-01-01 10:00:00",
            "phase": "dependency_installation",
            "component": "network_handler"
        }
        
        formatted_error = self.guidance.format_user_friendly_error(
            error, context, strategies
        )
        
        # Verify enhanced formatting
        self.assertIn("🔍 What happened:", formatted_error)
        self.assertIn("🖥️  System Context:", formatted_error)
        self.assertIn("🔧 Available Recovery Strategies:", formatted_error)
        self.assertIn("Network Diagnostics", formatted_error)
        self.assertIn("Success likelihood: 80%", formatted_error)
        self.assertIn("Estimated time: 5 min", formatted_error)
        self.assertIn("🆘 Support Resources:", formatted_error)
        self.assertIn("📚", formatted_error)  # Documentation icon
        self.assertIn("💬", formatted_error)  # Community forum icon
    
    def test_progress_indicators(self):
        """Test progress indicators with estimated completion times."""
        # Create progress indicator
        progress_id = self.guidance.create_progress_indicator("Test Operation", 5)
        
        # Verify initial state
        self.assertIn(progress_id, self.guidance.active_progress_indicators)
        indicator = self.guidance.active_progress_indicators[progress_id]
        self.assertEqual(indicator.operation_name, "Test Operation")
        self.assertEqual(indicator.total_steps, 5)
        self.assertEqual(indicator.current_step, 0)
        
        # Test progress updates
        self.guidance.update_progress(progress_id, 2, "Processing files", "10 files/sec")
        
        indicator = self.guidance.active_progress_indicators[progress_id]
        self.assertEqual(indicator.current_step, 2)
        self.assertEqual(indicator.current_step_name, "Processing files")
        self.assertEqual(indicator.speed_info, "10 files/sec")
        self.assertEqual(indicator.progress_percentage, 40.0)
        self.assertIsNotNone(indicator.time_remaining_seconds)
        
        # Test progress bar generation
        progress_bar = indicator.get_progress_bar(20)
        self.assertIn("█", progress_bar)  # Filled portion
        self.assertIn("░", progress_bar)  # Empty portion
        self.assertIn("40.0%", progress_bar)
        
        # Test time remaining formatting
        time_str = indicator.get_time_remaining_str()
        self.assertIn("remaining", time_str.lower())
        
        # Test completion
        self.guidance.complete_progress(progress_id)
        self.assertNotIn(progress_id, self.guidance.active_progress_indicators)
    
    def test_recovery_strategy_explanation(self):
        """Test recovery strategy explanations with success likelihood."""
        strategy = RecoveryStrategy(
            name="Model Validation Recovery",
            description="Automatically re-download and validate corrupted models",
            success_likelihood=0.9,
            estimated_time_minutes=15,
            steps=[
                {
                    "description": "Identify corrupted models",
                    "command": "python validate_models.py",
                    "expected_result": "List of corrupted files"
                },
                {
                    "description": "Download fresh models",
                    "command": "python download_models.py --force",
                    "expected_result": "Models downloaded successfully"
                },
                {
                    "description": "Verify model integrity",
                    "command": "python verify_models.py",
                    "expected_result": "All models validated"
                }
            ],
            prerequisites=["Internet connection", "Sufficient disk space"],
            risks=["Large download size", "May take significant time"]
        )
        
        explanation = self.guidance.explain_recovery_strategy(strategy)
        
        # Verify explanation content
        self.assertIn("🔧 Recovery Strategy: Model Validation Recovery", explanation)
        self.assertIn("Success Likelihood: 90%", explanation)
        self.assertIn("Estimated Time: 15 minutes", explanation)
        self.assertIn("Prerequisites:", explanation)
        self.assertIn("Internet connection", explanation)
        self.assertIn("⚠️  Potential Risks:", explanation)
        self.assertIn("Large download size", explanation)
        self.assertIn("Steps to Execute:", explanation)
        self.assertIn("1. Identify corrupted models", explanation)
        self.assertIn("Command: python validate_models.py", explanation)
        self.assertIn("Expected: List of corrupted files", explanation)
    
    def test_recovery_strategy_execution(self):
        """Test recovery strategy execution with progress tracking."""
        strategy = RecoveryStrategy(
            name="Test Recovery",
            description="Test recovery strategy execution",
            success_likelihood=0.8,
            estimated_time_minutes=5,
            steps=[
                {"description": "Step 1", "command": "test1"},
                {"description": "Step 2", "command": "test2"}
            ]
        )
        
        # Mock callback function
        callback_results = [True, True]  # Both steps succeed
        def mock_callback(step):
            return callback_results.pop(0) if callback_results else False
        
        # Execute strategy
        success = self.guidance.execute_recovery_strategy(strategy, mock_callback)
        
        # Verify execution
        self.assertTrue(success)
        self.assertEqual(strategy.status, RecoveryStatus.COMPLETED)
        self.assertIsNotNone(strategy.start_time)
        self.assertIsNotNone(strategy.completion_time)
        
        # Verify recovery history
        self.assertEqual(len(self.guidance.recovery_history), 1)
        history_entry = self.guidance.recovery_history[0]
        self.assertEqual(history_entry['strategy_name'], "Test Recovery")
        self.assertTrue(history_entry['success'])
        self.assertIn('duration_seconds', history_entry)
    
    def test_support_ticket_generation(self):
        """Test comprehensive support ticket generation."""
        # Create test error
        error = InstallationError(
            "Model validation failed: checksum mismatch",
            ErrorCategory.SYSTEM,
            recovery_suggestions=["Re-download models", "Check disk space"]
        )
        
        context = {
            "phase": "model_validation",
            "component": "model_validator",
            "timestamp": "2024-01-01 10:00:00"
        }
        
        steps_attempted = [
            "Ran diagnostic tool",
            "Checked system requirements",
            "Attempted model re-download"
        ]
        
        # Generate support ticket
        ticket = self.guidance.generate_support_ticket(error, context, steps_attempted)
        
        # Verify ticket content
        self.assertIn("Installation Error:", ticket.title)
        self.assertIn("Model validation failed", ticket.title)
        self.assertEqual(ticket.severity, "high")  # System errors are high severity
        self.assertEqual(ticket.category, "installation")
        self.assertEqual(len(ticket.steps_attempted), 3)
        self.assertIn("Ran diagnostic tool", ticket.steps_attempted)
        
        # Verify system info collection
        self.assertIn("OS", ticket.system_info)
        self.assertIn("Python Version", ticket.system_info)
        self.assertIn("Total RAM", ticket.system_info)
        
        # Verify environment details
        self.assertIsInstance(ticket.environment_details, dict)
        
        # Verify logs collection
        self.assertIsInstance(ticket.logs, list)
        
        # Test markdown conversion
        markdown = ticket.to_markdown()
        self.assertIn("# Support Ticket:", markdown)
        self.assertIn("## Problem Description", markdown)
        self.assertIn("## Error Details", markdown)
        self.assertIn("## System Information", markdown)
        self.assertIn("## Steps Attempted", markdown)
        
        # Test URL encoding
        url_encoded = ticket.to_url_encoded()
        self.assertIsInstance(url_encoded, str)
        self.assertGreater(len(url_encoded), 0)
    
    def test_support_ticket_saving(self):
        """Test support ticket saving and URL generation."""
        # Create test ticket
        error = InstallationError("Test error", ErrorCategory.NETWORK)
        ticket = self.guidance.generate_support_ticket(error)
        
        # Save ticket
        ticket_path = self.guidance.save_support_ticket(ticket)
        
        # Verify file was created
        self.assertTrue(Path(ticket_path).exists())
        
        # Verify content
        with open(ticket_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        self.assertIn("# Support Ticket:", content)
        self.assertIn("Test error", content)
        
        # Test URL generation
        github_url = self.guidance.create_support_ticket_url(ticket, "github")
        self.assertIn("github.com", github_url)
        self.assertIn("issues/new", github_url)
        self.assertIn("title=", github_url)
        self.assertIn("body=", github_url)
    
    def test_support_resources_integration(self):
        """Test support resources integration in error formatting."""
        # Test network error resources
        error = InstallationError("Connection timeout", ErrorCategory.NETWORK)
        formatted = self.guidance.format_user_friendly_error(error)
        
        self.assertIn("🆘 Support Resources:", formatted)
        self.assertIn("Network Troubleshooting Guide", formatted)
        self.assertIn("github.com/wan22/wiki/network-troubleshooting", formatted)
        
        # Test system error resources
        error = InstallationError("Insufficient memory", ErrorCategory.SYSTEM)
        formatted = self.guidance.format_user_friendly_error(error)
        
        self.assertIn("System Requirements Guide", formatted)
        self.assertIn("GPU Setup Guide", formatted)
    
    def test_resource_icon_mapping(self):
        """Test resource type icon mapping."""
        # Test all resource type icons
        icons = {
            SupportResourceType.DOCUMENTATION: "📚",
            SupportResourceType.COMMUNITY_FORUM: "💬",
            SupportResourceType.SUPPORT_TICKET: "🎫",
            SupportResourceType.VIDEO_TUTORIAL: "🎥",
            SupportResourceType.FAQ: "❓"
        }
        
        for resource_type, expected_icon in icons.items():
            icon = self.guidance._get_resource_icon(resource_type)
            self.assertEqual(icon, expected_icon)
    
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    @patch('psutil.cpu_count')
    @patch('psutil.cpu_percent')
    def test_system_info_collection(self, mock_cpu_percent, mock_cpu_count, 
                                   mock_disk_usage, mock_virtual_memory):
        """Test system information collection."""
        # Mock system info
        mock_virtual_memory.return_value = Mock(
            total=8 * 1024**3,  # 8GB
            available=4 * 1024**3  # 4GB
        )
        mock_disk_usage.return_value = Mock(free=100 * 1024**3)  # 100GB
        mock_cpu_count.return_value = 8
        mock_cpu_percent.return_value = 25.5
        
        system_info = self.guidance._collect_system_info()
        
        self.assertIn("OS", system_info)
        self.assertIn("Python Version", system_info)
        self.assertEqual(system_info["Total RAM"], "8 GB")
        self.assertEqual(system_info["Available RAM"], "4 GB")
        self.assertEqual(system_info["Disk Space"], "100 GB free")
        self.assertEqual(system_info["CPU Count"], 8)
        self.assertEqual(system_info["CPU Usage"], "25.5%")
    
    def test_environment_details_collection(self):
        """Test environment details collection."""
        with patch.dict('os.environ', {
            'PATH': '/usr/bin:/bin',
            'PYTHONPATH': '/usr/lib/python',
            'CUDA_PATH': '/usr/local/cuda'
        }):
            details = self.guidance._collect_environment_details()
            
            self.assertIn("ENV_PATH", details)
            self.assertIn("ENV_PYTHONPATH", details)
            self.assertIn("ENV_CUDA_PATH", details)
            self.assertEqual(details["ENV_PATH"], "/usr/bin:/bin")
    
    def test_log_collection(self):
        """Test recent log entries collection."""
        # Create additional log files
        log2 = self.logs_dir / "error.log"
        log2.write_text("2024-01-01 11:00:00 - ERROR - Critical error\n"
                       "2024-01-01 11:01:00 - INFO - Recovery attempted\n")
        
        logs = self.guidance._collect_recent_logs()
        
        self.assertIsInstance(logs, list)
        self.assertGreater(len(logs), 0)
        
        # Check that logs from both files are included
        log_content = "\n".join(logs)
        self.assertIn("test.log", log_content)
        self.assertIn("error.log", log_content)
        self.assertIn("Test log entry", log_content)
        self.assertIn("Critical error", log_content)
    
    def test_concurrent_progress_tracking(self):
        """Test concurrent progress tracking with multiple operations."""
        # Create multiple progress indicators
        progress_ids = []
        for i in range(3):
            progress_id = self.guidance.create_progress_indicator(f"Operation {i}", 5)
            progress_ids.append(progress_id)
        
        # Verify all are tracked
        self.assertEqual(len(self.guidance.active_progress_indicators), 3)
        
        # Update progress concurrently
        def update_progress(pid, steps):
            for step in range(1, steps + 1):
                self.guidance.update_progress(pid, step, f"Step {step}")
                time.sleep(0.1)
        
        threads = []
        for i, pid in enumerate(progress_ids):
            thread = threading.Thread(target=update_progress, args=(pid, 3))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Complete all progress
        for pid in progress_ids:
            self.guidance.complete_progress(pid)
        
        # Verify cleanup
        self.assertEqual(len(self.guidance.active_progress_indicators), 0)


class TestProgressIndicator(unittest.TestCase):
    """Test ProgressIndicator class functionality."""
    
    def test_progress_calculation(self):
        """Test progress percentage and time estimation."""
        indicator = ProgressIndicator("Test", 0, 10, "Starting")
        
        # Test initial state
        self.assertEqual(indicator.progress_percentage, 0.0)
        self.assertIsNone(indicator.time_remaining_seconds)
        
        # Simulate some time passing
        time.sleep(0.1)
        
        # Update progress
        indicator.update_progress(3, "Step 3", "5 items/sec")
        
        self.assertEqual(indicator.current_step, 3)
        self.assertEqual(indicator.progress_percentage, 30.0)
        self.assertEqual(indicator.speed_info, "5 items/sec")
        self.assertIsNotNone(indicator.time_remaining_seconds)
        self.assertIsNotNone(indicator.estimated_completion_time)
    
    def test_progress_bar_generation(self):
        """Test progress bar visual generation."""
        indicator = ProgressIndicator("Test", 0, 4, "Starting")
        indicator.update_progress(2, "Halfway")
        
        # Test different widths
        bar_20 = indicator.get_progress_bar(20)
        bar_40 = indicator.get_progress_bar(40)
        
        self.assertIn("50.0%", bar_20)
        self.assertIn("50.0%", bar_40)
        self.assertIn("█", bar_20)  # Filled
        self.assertIn("░", bar_20)  # Empty
        
        # Test full progress
        indicator.update_progress(4, "Complete")
        bar_full = indicator.get_progress_bar(20)
        self.assertIn("100.0%", bar_full)
    
    def test_time_formatting(self):
        """Test time remaining formatting."""
        indicator = ProgressIndicator("Test", 0, 10, "Starting")
        
        # Test different time ranges
        indicator.time_remaining_seconds = 30
        self.assertEqual(indicator.get_time_remaining_str(), "30s remaining")
        
        indicator.time_remaining_seconds = 90
        self.assertEqual(indicator.get_time_remaining_str(), "1m 30s remaining")
        
        indicator.time_remaining_seconds = 3661
        self.assertEqual(indicator.get_time_remaining_str(), "1h 1m remaining")


class TestSupportTicket(unittest.TestCase):
    """Test SupportTicket class functionality."""
    
    def test_ticket_creation(self):
        """Test support ticket creation and basic properties."""
        ticket = SupportTicket(
            title="Test Error",
            description="Test description",
            error_details="Error details",
            system_info={"OS": "Windows 10"},
            logs=["Log entry 1", "Log entry 2"],
            steps_attempted=["Step 1", "Step 2"],
            severity="high"
        )
        
        self.assertEqual(ticket.title, "Test Error")
        self.assertEqual(ticket.severity, "high")
        self.assertEqual(len(ticket.logs), 2)
        self.assertEqual(len(ticket.steps_attempted), 2)
    
    def test_markdown_conversion(self):
        """Test markdown format conversion."""
        ticket = SupportTicket(
            title="Network Error",
            description="Connection failed",
            error_details="Timeout after 30s",
            system_info={"OS": "Windows 10", "RAM": "8GB"},
            logs=["ERROR: Connection timeout"],
            steps_attempted=["Checked network", "Restarted router"],
            recovery_strategies_tried=["Network Diagnostics"],
            environment_details={"ENV_PATH": "/usr/bin"}
        )
        
        markdown = ticket.to_markdown()
        
        # Verify structure
        self.assertIn("# Support Ticket: Network Error", markdown)
        self.assertIn("## Problem Description", markdown)
        self.assertIn("Connection failed", markdown)
        self.assertIn("## Error Details", markdown)
        self.assertIn("Timeout after 30s", markdown)
        self.assertIn("## System Information", markdown)
        self.assertIn("- **OS:** Windows 10", markdown)
        self.assertIn("## Environment Details", markdown)
        self.assertIn("- **ENV_PATH:** /usr/bin", markdown)
        self.assertIn("## Steps Attempted", markdown)
        self.assertIn("1. Checked network", markdown)
        self.assertIn("## Recovery Strategies Tried", markdown)
        self.assertIn("- Network Diagnostics", markdown)
        self.assertIn("## Relevant Log Entries", markdown)
        self.assertIn("ERROR: Connection timeout", markdown)


if __name__ == "__main__":
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestEnhancedUserGuidance))
    suite.addTest(unittest.makeSuite(TestProgressIndicator))
    suite.addTest(unittest.makeSuite(TestSupportTicket))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Enhanced User Guidance System Test Results")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            error_msg = traceback.split('AssertionError: ')[-1].split('\n')[0]
            print(f"  - {test}: {error_msg}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            error_msg = traceback.split('\n')[-2]
            print(f"  - {test}: {error_msg}")