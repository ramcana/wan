"""
Tests for the Test Execution Engine

This module tests the test execution engine's functionality including:
- Timeout handling
- Retry logic with exponential backoff
- Parallel execution
- Resource management
- Test result aggregation
"""

import asyncio
import os
import tempfile
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

import pytest

# Import the test execution engine
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.test_execution_engine import (
    TestExecutionEngine, TestConfig, TestCategory, TestStatus, 
    TestResult, TestSuiteResult, ResourceMonitor
)


class TestTestExecutionEngine(unittest.TestCase):
    """Test cases for TestExecutionEngine"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = TestConfig(
            timeouts={
                TestCategory.UNIT: 5,
                TestCategory.INTEGRATION: 10,
                TestCategory.E2E: 15
            },
            max_retries=2,
            max_workers=2,
            retry_delay_base=0.1,
            retry_delay_max=1.0
        )
        self.engine = TestExecutionEngine(self.config)
        
    def test_categorize_test(self):
        """Test test categorization logic"""
        # Unit tests
        self.assertEqual(
            self.engine.categorize_test("tests/unit/test_example.py"),
            TestCategory.UNIT
        )
        self.assertEqual(
            self.engine.categorize_test("test_unit_example.py"),
            TestCategory.UNIT
        )
        
        # Integration tests
        self.assertEqual(
            self.engine.categorize_test("tests/integration/test_api.py"),
            TestCategory.INTEGRATION
        )
        self.assertEqual(
            self.engine.categorize_test("test_integration_workflow.py"),
            TestCategory.INTEGRATION
        )
        
        # E2E tests
        self.assertEqual(
            self.engine.categorize_test("tests/e2e/test_full_flow.py"),
            TestCategory.E2E
        )
        
        # Default to unit
        self.assertEqual(
            self.engine.categorize_test("tests/test_unknown.py"),
            TestCategory.UNIT
        )
        
    def test_calculate_retry_delay(self):
        """Test exponential backoff calculation"""
        # First retry
        delay1 = self.engine._calculate_retry_delay(0)
        self.assertEqual(delay1, 0.1)
        
        # Second retry
        delay2 = self.engine._calculate_retry_delay(1)
        self.assertEqual(delay2, 0.2)
        
        # Third retry
        delay3 = self.engine._calculate_retry_delay(2)
        self.assertEqual(delay3, 0.4)
        
        # Should cap at max delay
        large_delay = self.engine._calculate_retry_delay(10)
        self.assertEqual(large_delay, 1.0)
        
    def test_flaky_test_detection(self):
        """Test flaky test detection logic"""
        test_id = "test_flaky.py"
        
        # Initially not flaky
        self.assertFalse(self.engine._is_flaky_test(test_id))
        
        # Add some failures
        self.engine._update_flaky_history(test_id, False)  # Failure
        self.engine._update_flaky_history(test_id, False)  # Failure
        self.engine._update_flaky_history(test_id, True)   # Success
        
        # Should be considered flaky now (success rate = 1/3 = 0.33 < 0.7)
        self.assertTrue(self.engine._is_flaky_test(test_id))
        
        # Add more successes
        for _ in range(5):
            self.engine._update_flaky_history(test_id, True)
            
        # Should not be flaky anymore (success rate improved)
        self.assertFalse(self.engine._is_flaky_test(test_id))
        
    def test_discover_tests(self):
        """Test test discovery functionality"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            test_files = [
                "test_unit_example.py",
                "test_integration_api.py",
                "example_test.py",
                "not_a_test.py"
            ]
            
            for test_file in test_files:
                Path(temp_dir, test_file).touch()
                
            # Discover tests
            discovered = self.engine.discover_tests(temp_dir)
            
            # Should find test files but not regular files
            self.assertEqual(len(discovered), 3)
            self.assertTrue(any("test_unit_example.py" in f for f in discovered))
            self.assertTrue(any("test_integration_api.py" in f for f in discovered))
            self.assertTrue(any("example_test.py" in f for f in discovered))
            self.assertFalse(any("not_a_test.py" in f for f in discovered))


class TestResourceMonitor(unittest.TestCase):
    """Test cases for ResourceMonitor"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.monitor = ResourceMonitor()
        
    def test_resource_monitoring(self):
        """Test resource monitoring functionality"""
        # Start monitoring
        self.monitor.start_monitoring()
        
        # Wait a bit for some data
        time.sleep(2)
        
        # Stop monitoring
        self.monitor.stop_monitoring()
        
        # Should have collected some data
        self.assertGreater(len(self.monitor.cpu_usage), 0)
        self.assertGreater(len(self.monitor.memory_usage), 0)
        
        # Get average usage
        avg_cpu, avg_memory = self.monitor.get_average_usage()
        self.assertGreaterEqual(avg_cpu, 0)
        self.assertGreaterEqual(avg_memory, 0)
        self.assertLessEqual(avg_cpu, 100)
        self.assertLessEqual(avg_memory, 100)
        
    def test_throttling_decision(self):
        """Test throttling decision logic"""
        config = TestConfig(cpu_threshold=0.5, memory_threshold=0.5)
        
        # Simulate high resource usage
        self.monitor.cpu_usage = [80.0]  # 80% CPU
        self.monitor.memory_usage = [60.0]  # 60% memory
        
        # Should throttle due to high CPU
        self.assertTrue(self.monitor.should_throttle(config))
        
        # Simulate low resource usage
        self.monitor.cpu_usage = [20.0]  # 20% CPU
        self.monitor.memory_usage = [30.0]  # 30% memory
        
        # Should not throttle
        self.assertFalse(self.monitor.should_throttle(config))


class TestAsyncExecution(unittest.TestCase):
    """Test cases for async test execution"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = TestConfig(
            timeouts={TestCategory.UNIT: 2},
            max_retries=1,
            max_workers=2
        )
        self.engine = TestExecutionEngine(self.config)
        
    @patch('asyncio.create_subprocess_exec')
    async def test_successful_test_execution(self, mock_subprocess):
        """Test successful test execution"""
        # Mock successful process
        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"test output", b"")
        mock_process.returncode = 0
        mock_subprocess.return_value = mock_process
        
        # Create semaphore
        semaphore = asyncio.Semaphore(1)
        
        # Execute test
        result = await self.engine._execute_single_test("test_example.py", semaphore)
        
        # Verify result
        self.assertEqual(result.status, TestStatus.PASSED)
        self.assertEqual(result.retry_count, 0)
        self.assertIsNotNone(result.stdout)
        self.assertGreater(result.duration, 0)
        
    @patch('asyncio.create_subprocess_exec')
    async def test_failed_test_with_retry(self, mock_subprocess):
        """Test failed test with retry logic"""
        # Mock failing process
        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"", b"test failed")
        mock_process.returncode = 1
        mock_subprocess.return_value = mock_process
        
        # Create semaphore
        semaphore = asyncio.Semaphore(1)
        
        # Execute test
        result = await self.engine._execute_single_test("test_failing.py", semaphore)
        
        # Verify result
        self.assertEqual(result.status, TestStatus.FAILED)
        self.assertEqual(result.retry_count, 1)  # Should have retried once
        self.assertIsNotNone(result.error_message)
        
    @patch('asyncio.create_subprocess_exec')
    async def test_timeout_handling(self, mock_subprocess):
        """Test timeout handling"""
        # Mock process that never completes
        mock_process = AsyncMock()
        mock_process.communicate.side_effect = asyncio.TimeoutError()
        mock_process.kill = AsyncMock()
        mock_process.wait = AsyncMock()
        mock_subprocess.return_value = mock_process
        
        # Create semaphore
        semaphore = asyncio.Semaphore(1)
        
        # Execute test
        result = await self.engine._execute_single_test("test_timeout.py", semaphore)
        
        # Verify result
        self.assertEqual(result.status, TestStatus.TIMEOUT)
        self.assertIn("timed out", result.error_message.lower())
        
        # Verify process was killed
        mock_process.kill.assert_called_once()


class TestIntegration(unittest.TestCase):
    """Integration tests for the test execution engine"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = TestConfig(
            timeouts={TestCategory.UNIT: 10},
            max_retries=1,
            max_workers=2
        )
        self.engine = TestExecutionEngine(self.config)
        
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def create_test_file(self, filename: str, content: str):
        """Create a test file with given content"""
        test_file = Path(self.temp_dir) / filename
        test_file.write_text(content)
        return str(test_file)
        
    def test_run_simple_passing_test(self):
        """Test running a simple passing test"""
        # Create a simple passing test
        test_content = '''
import unittest

class TestExample(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(1 + 1, 2)

if __name__ == "__main__":
    unittest.main()
'''
        test_file = self.create_test_file("test_simple.py", test_content)
        
        # Run the test
        result = self.engine.run_tests([test_file])
        
        # Verify results
        self.assertEqual(result.total_tests, 1)
        self.assertEqual(result.passed, 1)
        self.assertEqual(result.failed, 0)
        self.assertGreater(result.total_duration, 0)
        
    def test_run_failing_test(self):
        """Test running a failing test"""
        # Create a failing test
        test_content = '''
import unittest

class TestExample(unittest.TestCase):
    def test_failing(self):
        self.assertEqual(1 + 1, 3)  # This will fail

if __name__ == "__main__":
    unittest.main()
'''
        test_file = self.create_test_file("test_failing.py", test_content)
        
        # Run the test
        result = self.engine.run_tests([test_file])
        
        # Verify results
        self.assertEqual(result.total_tests, 1)
        self.assertEqual(result.passed, 0)
        self.assertEqual(result.failed, 1)
        
    def test_report_generation(self):
        """Test report generation"""
        # Create a simple test
        test_content = '''
import unittest

class TestExample(unittest.TestCase):
    def test_simple(self):
        self.assertTrue(True)

if __name__ == "__main__":
    unittest.main()
'''
        test_file = self.create_test_file("test_report.py", test_content)
        
        # Run the test
        result = self.engine.run_tests([test_file])
        
        # Generate report
        report = self.engine.generate_report(result)
        
        # Verify report content
        self.assertIn("Test Execution Report", report)
        self.assertIn("Total Tests: 1", report)
        self.assertIn("Passed: 1", report)
        self.assertIn("Success Rate:", report)


if __name__ == "__main__":
    # Run the tests
    unittest.main(verbosity=2)