from unittest.mock import Mock, patch
#!/usr/bin/env python3
"""
Unit tests for Integration Tester components
Tests the integration testing orchestrator, UI tester, and API tester functionality.
"""

import unittest
import unittest.mock as mock
import tempfile
import shutil
import os
import sys
import subprocess
import requests
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integration_tester import IntegrationTester, UITester, APITester
from models.configuration import LocalTestConfiguration
from models.test_results import TestStatus, ValidationStatus, ValidationResult


class TestIntegrationTester(unittest.TestCase):
    """Test cases for IntegrationTester class"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp(prefix="test_integration_")
        
        # Create mock configuration
        self.mock_config = mock.MagicMock(spec=LocalTestConfiguration)
        self.mock_config.performance_targets = mock.MagicMock()
        self.mock_config.performance_targets.target_720p_time_minutes = 9.0
        self.mock_config.performance_targets.target_1080p_time_minutes = 17.0
        
        self.integration_tester = IntegrationTester(self.mock_config)
    
    def tearDown(self):
        """Clean up test environment"""
        if hasattr(self, 'integration_tester'):
            self.integration_tester.cleanup()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test IntegrationTester initialization"""
        self.assertIsNotNone(self.integration_tester)
        self.assertEqual(self.integration_tester.config, self.mock_config)
        self.assertIsNotNone(self.integration_tester.temp_dir)
        self.assertTrue(os.path.exists(self.integration_tester.temp_dir))
    
    def test_temp_environment_setup(self):
        """Test temporary environment setup"""
        # Check that required directories are created
        required_dirs = ['outputs', 'models', 'loras']
        for dir_name in required_dirs:
            dir_path = os.path.join(self.integration_tester.temp_dir, dir_name)
            self.assertTrue(os.path.exists(dir_path))
            self.assertTrue(os.path.isdir(dir_path))
    
    @mock.patch('subprocess.Popen')
    def test_execute_generation_test_success(self, mock_popen):
        """Test successful generation test execution"""
        # Mock successful subprocess execution
        mock_process = mock.MagicMock()
        mock_process.communicate.return_value = ("Generation completed successfully", "")
        mock_process.returncode = 0
        mock_popen.return_value = mock_process
        
        test_config = {
            'model_type': 't2v-A14B',
            'prompt': 'Test prompt',
            'resolution': '1280x720',
            'expected_time_limit': 540
        }
        
        result = self.integration_tester._execute_generation_test(test_config)
        
        self.assertTrue(result['success'])
        self.assertIn('stdout', result)
        self.assertIn('stderr', result)
    
    @mock.patch('subprocess.Popen')
    def test_execute_generation_test_failure(self, mock_popen):
        """Test failed generation test execution"""
        # Mock failed subprocess execution
        mock_process = mock.MagicMock()
        mock_process.communicate.return_value = ("", "Generation failed")
        mock_process.returncode = 1
        mock_popen.return_value = mock_process
        
        test_config = {
            'model_type': 't2v-A14B',
            'prompt': 'Test prompt',
            'resolution': '1280x720',
            'expected_time_limit': 540
        }
        
        result = self.integration_tester._execute_generation_test(test_config)
        
        self.assertFalse(result['success'])
        self.assertIn('error_message', result)
    
    @mock.patch('subprocess.Popen')
    def test_execute_generation_test_timeout(self, mock_popen):
        """Test generation test timeout handling"""
        # Mock timeout scenario
        mock_process = mock.MagicMock()
        mock_process.communicate.side_effect = subprocess.TimeoutExpired('cmd', 10)
        mock_popen.return_value = mock_process
        
        test_config = {
            'model_type': 't2v-A14B',
            'prompt': 'Test prompt',
            'resolution': '1280x720',
            'expected_time_limit': 1  # Very short timeout for test
        }
        
        result = self.integration_tester._execute_generation_test(test_config)
        
        self.assertFalse(result['success'])
        self.assertIn('timed out', result['error_message'])
    
    @mock.patch('subprocess.Popen')
    def test_error_handling_and_recovery_success(self, mock_popen):
        """Test successful error handling test execution"""
        # Mock successful error handling tests
        mock_process = mock.MagicMock()
        mock_process.communicate.return_value = ("✓ Test passed\n✓ Another test passed", "")
        mock_process.returncode = 0
        mock_popen.return_value = mock_process
        
        result = self.integration_tester.test_error_handling_and_recovery()
        
        self.assertEqual(result.status, TestStatus.PASSED)
        self.assertEqual(result.component, "error_handling")
        self.assertIn("passed", result.message)
    
    @mock.patch('subprocess.Popen')
    def test_error_handling_and_recovery_failure(self, mock_popen):
        """Test failed error handling test execution"""
        # Mock failed error handling tests
        mock_process = mock.MagicMock()
        mock_process.communicate.return_value = ("✗ Test failed\n✓ Test passed", "")
        mock_process.returncode = 1
        mock_popen.return_value = mock_process
        
        result = self.integration_tester.test_error_handling_and_recovery()
        
        self.assertEqual(result.status, TestStatus.FAILED)
        self.assertEqual(result.component, "error_handling")
        self.assertGreater(len(result.remediation_steps), 0)
    
    @mock.patch('subprocess.Popen')
    def test_resource_monitoring_accuracy_success(self, mock_popen):
        """Test successful resource monitoring accuracy test"""
        # Mock successful resource monitoring tests
        mock_process = mock.MagicMock()
        mock_process.communicate.return_value = ("Resource monitoring tests passed", "")
        mock_process.returncode = 0
        mock_popen.return_value = mock_process
        
        result = self.integration_tester.test_resource_monitoring_accuracy()
        
        self.assertEqual(result.status, TestStatus.PASSED)
        self.assertEqual(result.component, "resource_monitoring")
    
    def test_capture_resource_snapshot(self):
        """Test resource snapshot capture"""
        snapshot = self.integration_tester._capture_resource_snapshot()
        
        self.assertIsNotNone(snapshot)
        self.assertGreaterEqual(snapshot.cpu_percent, 0.0)
        self.assertGreaterEqual(snapshot.memory_percent, 0.0)
        self.assertGreaterEqual(snapshot.memory_used_gb, 0.0)
        self.assertGreater(snapshot.memory_total_gb, 0.0)
        self.assertIsInstance(snapshot.timestamp, datetime)
    
    def test_calculate_resource_delta(self):
        """Test resource usage delta calculation"""
        start_snapshot = self.integration_tester._capture_resource_snapshot()
        
        # Create a mock end snapshot with different values
        end_snapshot = start_snapshot
        end_snapshot.cpu_percent = start_snapshot.cpu_percent + 10.0
        end_snapshot.memory_used_gb = start_snapshot.memory_used_gb + 1.0
        
        delta = self.integration_tester._calculate_resource_delta(start_snapshot, end_snapshot)
        
        self.assertEqual(delta.cpu_percent, 10.0)
        self.assertEqual(delta.memory_used_gb, 1.0)
    
    def test_extract_output_path(self):
        """Test output path extraction from stdout"""
        stdout_with_path = "Generation completed. Output saved to /path/to/output.mp4"
        path = self.integration_tester._extract_output_path(stdout_with_path)
        self.assertEqual(path, "/path/to/output.mp4")
        
        stdout_without_path = "Generation completed successfully"
        path = self.integration_tester._extract_output_path(stdout_without_path)
        self.assertIsNone(path)
    
    def test_determine_overall_status_all_passed(self):
        """Test overall status determination when all tests pass"""
        # Mock all successful results
        generation_results = [
            mock.MagicMock(success=True),
            mock.MagicMock(success=True)
        ]
        error_handling_result = mock.MagicMock(status=TestStatus.PASSED)
        ui_results = mock.MagicMock(overall_status=TestStatus.PASSED)
        api_results = mock.MagicMock(overall_status=TestStatus.PASSED)
        resource_monitoring_result = mock.MagicMock(status=TestStatus.PASSED)
        
        status = self.integration_tester._determine_overall_status(
            generation_results, error_handling_result, ui_results,
            api_results, resource_monitoring_result
        )
        
        self.assertEqual(status, TestStatus.PASSED)
    
    def test_determine_overall_status_some_failed(self):
        """Test overall status determination when some tests fail"""
        # Mock mixed results
        generation_results = [
            mock.MagicMock(success=True),
            mock.MagicMock(success=False)
        ]
        error_handling_result = mock.MagicMock(status=TestStatus.PASSED)
        ui_results = mock.MagicMock(overall_status=TestStatus.FAILED)
        api_results = mock.MagicMock(overall_status=TestStatus.PASSED)
        resource_monitoring_result = mock.MagicMock(status=TestStatus.PASSED)
        
        status = self.integration_tester._determine_overall_status(
            generation_results, error_handling_result, ui_results,
            api_results, resource_monitoring_result
        )
        
        self.assertEqual(status, TestStatus.WARNING)
    
    def test_determine_overall_status_all_failed(self):
        """Test overall status determination when all tests fail"""
        # Mock all failed results
        generation_results = [
            mock.MagicMock(success=False),
            mock.MagicMock(success=False)
        ]
        error_handling_result = mock.MagicMock(status=TestStatus.FAILED)
        ui_results = mock.MagicMock(overall_status=TestStatus.FAILED)
        api_results = mock.MagicMock(overall_status=TestStatus.FAILED)
        resource_monitoring_result = mock.MagicMock(status=TestStatus.FAILED)
        
        status = self.integration_tester._determine_overall_status(
            generation_results, error_handling_result, ui_results,
            api_results, resource_monitoring_result
        )
        
        self.assertEqual(status, TestStatus.FAILED)


class TestUITester(unittest.TestCase):
    """Test cases for UITester class"""
    
    def setUp(self):
        """Set up test environment"""
        self.mock_config = mock.MagicMock(spec=LocalTestConfiguration)
        
        # Only create UITester if Selenium is available
        try:
            self.ui_tester = UITester(self.mock_config)
        except ImportError:
            self.ui_tester = None
    
    def tearDown(self):
        """Clean up test environment"""
        if self.ui_tester:
            self.ui_tester.cleanup()
    
    @unittest.skipIf(not hasattr(UITester, '__init__'), "Selenium not available")
    def test_initialization(self):
        """Test UITester initialization"""
        if self.ui_tester:
            self.assertIsNotNone(self.ui_tester)
            self.assertEqual(self.ui_tester.config, self.mock_config)
            self.assertEqual(self.ui_tester.base_url, "http://localhost:7860")
    
    @mock.patch('subprocess.Popen')
    def test_launch_application_success(self, mock_popen):
        """Test successful application launch"""
        if not self.ui_tester:
            self.skipTest("Selenium not available")
        
        # Mock successful application launch
        mock_process = mock.MagicMock()
        mock_process.poll.return_value = None  # Process still running
        mock_popen.return_value = mock_process
        
        with mock.patch('time.sleep'):  # Speed up test
            result = self.ui_tester.launch_application()
        
        self.assertTrue(result)
        self.assertEqual(self.ui_tester.app_process, mock_process)
    
    @mock.patch('subprocess.Popen')
    def test_launch_application_failure(self, mock_popen):
        """Test failed application launch"""
        if not self.ui_tester:
            self.skipTest("Selenium not available")
        
        # Mock failed application launch
        mock_process = mock.MagicMock()
        mock_process.poll.return_value = 1  # Process exited with error
        mock_popen.return_value = mock_process
        
        with mock.patch('time.sleep'):  # Speed up test
            result = self.ui_tester.launch_application()
        
        self.assertFalse(result)
    
    @mock.patch('local_testing_framework.integration_tester.webdriver')
    def test_setup_headless_browser_chrome(self, mock_webdriver):
        """Test headless browser setup with Chrome"""
        if not self.ui_tester:
            self.skipTest("Selenium not available")
        
        # Mock Chrome driver creation
        mock_driver = mock.MagicMock()
        mock_webdriver.Chrome.return_value = mock_driver
        
        driver = self.ui_tester.setup_headless_browser()
        
        self.assertEqual(driver, mock_driver)
        mock_webdriver.Chrome.assert_called_once()
    
    @mock.patch('local_testing_framework.integration_tester.webdriver')
    def test_setup_headless_browser_firefox_fallback(self, mock_webdriver):
        """Test headless browser setup with Firefox fallback"""
        if not self.ui_tester:
            self.skipTest("Selenium not available")
        
        # Mock Chrome failure and Firefox success
        mock_webdriver.Chrome.side_effect = Exception("Chrome not available")
        mock_firefox_driver = mock.MagicMock()
        mock_webdriver.Firefox.return_value = mock_firefox_driver
        
        driver = self.ui_tester.setup_headless_browser()
        
        self.assertEqual(driver, mock_firefox_driver)
        mock_webdriver.Firefox.assert_called_once()
    
    def test_browser_access_no_driver(self):
        """Test browser access test without driver"""
        if not self.ui_tester:
            self.skipTest("Selenium not available")
        
        # Ensure no driver is set
        self.ui_tester.driver = None
        
        result = self.ui_tester.test_browser_access()
        
        self.assertEqual(result.status, TestStatus.FAILED)
        self.assertEqual(result.component, "browser_access")


class TestAPITester(unittest.TestCase):
    """Test cases for APITester class"""
    
    def setUp(self):
        """Set up test environment"""
        self.mock_config = mock.MagicMock(spec=LocalTestConfiguration)
        self.api_tester = APITester(self.mock_config)
    
    def test_initialization(self):
        """Test APITester initialization"""
        self.assertIsNotNone(self.api_tester)
        self.assertEqual(self.api_tester.config, self.mock_config)
        self.assertEqual(self.api_tester.base_url, "http://localhost:7860")
        self.assertIsNotNone(self.api_tester.session)
    
    @mock.patch('requests.Session.get')
    def test_health_endpoint_success(self, mock_get):
        """Test successful health endpoint validation"""
        # Mock successful health endpoint response
        mock_response = mock.MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "healthy",
            "gpu_available": True
        }
        mock_get.return_value = mock_response
        
        result = self.api_tester.test_health_endpoint()
        
        self.assertEqual(result.status, TestStatus.PASSED)
        self.assertEqual(result.component, "health_endpoint")
        self.assertIn("passed", result.message)
    
    @mock.patch('requests.Session.get')
    def test_health_endpoint_unhealthy(self, mock_get):
        """Test unhealthy health endpoint response"""
        # Mock unhealthy health endpoint response
        mock_response = mock.MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "unhealthy",
            "gpu_available": False
        }
        mock_get.return_value = mock_response
        
        result = self.api_tester.test_health_endpoint()
        
        self.assertEqual(result.status, TestStatus.FAILED)
        self.assertEqual(result.component, "health_endpoint")
    
    @mock.patch('requests.Session.get')
    def test_health_endpoint_invalid_json(self, mock_get):
        """Test health endpoint with invalid JSON response"""
        # Mock invalid JSON response
        mock_response = mock.MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_response.text = "Invalid response"
        mock_get.return_value = mock_response
        
        result = self.api_tester.test_health_endpoint()
        
        self.assertEqual(result.status, TestStatus.FAILED)
        self.assertEqual(result.component, "health_endpoint")
        self.assertIn("invalid JSON", result.message)
    
    @mock.patch('requests.Session.get')
    def test_health_endpoint_connection_error(self, mock_get):
        """Test health endpoint connection error"""
        # Mock connection error
        mock_get.side_effect = requests.ConnectionError("Connection failed")
        
        result = self.api_tester.test_health_endpoint()
        
        self.assertEqual(result.status, TestStatus.FAILED)
        self.assertEqual(result.component, "health_endpoint")
        self.assertIn("request failed", result.message)
    
    @mock.patch('requests.Session.get')
    def test_authentication_endpoints_success(self, mock_get):
        """Test successful authentication endpoint validation"""
        # Mock appropriate auth response
        mock_response = mock.MagicMock()
        mock_response.status_code = 401  # Unauthorized is expected for auth endpoint
        mock_get.return_value = mock_response
        
        result = self.api_tester.test_authentication_endpoints()
        
        self.assertEqual(result.status, TestStatus.PASSED)
        self.assertEqual(result.component, "authentication")
    
    @mock.patch('requests.Session.get')
    def test_authentication_endpoints_not_available(self, mock_get):
        """Test authentication endpoint not available"""
        # Mock connection error (endpoint doesn't exist)
        mock_get.side_effect = requests.ConnectionError("Connection failed")
        
        result = self.api_tester.test_authentication_endpoints()
        
        self.assertEqual(result.status, TestStatus.WARNING)
        self.assertEqual(result.component, "authentication")
        self.assertIn("not available", result.message)
    
    @mock.patch('requests.Session.get')
    def test_rate_limiting_no_limits(self, mock_get):
        """Test rate limiting when no limits are enforced"""
        # Mock successful responses (no rate limiting)
        mock_response = mock.MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        with mock.patch('time.sleep'):  # Speed up test
            result = self.api_tester.test_api_rate_limiting()
        
        self.assertEqual(result.status, TestStatus.PASSED)
        self.assertEqual(result.component, "rate_limiting")
        self.assertIn("not active", result.message)
    
    @mock.patch('requests.Session.get')
    def test_rate_limiting_active(self, mock_get):
        """Test rate limiting when limits are enforced"""
        # Mock rate limited responses
        responses = [mock.MagicMock(status_code=200) for _ in range(5)]
        responses.extend([mock.MagicMock(status_code=429) for _ in range(5)])  # Rate limited
        mock_get.side_effect = responses
        
        with mock.patch('time.sleep'):  # Speed up test
            result = self.api_tester.test_api_rate_limiting()
        
        self.assertEqual(result.status, TestStatus.PASSED)
        self.assertEqual(result.component, "rate_limiting")
        self.assertIn("active", result.message)
    
    @mock.patch('requests.Session.get')
    def test_error_response_formats_json(self, mock_get):
        """Test error response format with JSON"""
        # Mock 404 response with JSON error
        mock_response = mock.MagicMock()
        mock_response.status_code = 404
        mock_response.json.return_value = {"error": "Not found"}
        mock_get.return_value = mock_response
        
        result = self.api_tester.test_error_response_formats()
        
        self.assertEqual(result.status, TestStatus.PASSED)
        self.assertEqual(result.component, "error_responses")
        self.assertIn("valid", result.message)
    
    @mock.patch('requests.Session.get')
    def test_error_response_formats_non_json(self, mock_get):
        """Test error response format without JSON"""
        # Mock 404 response with non-JSON error
        mock_response = mock.MagicMock()
        mock_response.status_code = 404
        mock_response.json.side_effect = ValueError("Not JSON")
        mock_response.text = "Not found"
        mock_get.return_value = mock_response
        
        result = self.api_tester.test_error_response_formats()
        
        self.assertEqual(result.status, TestStatus.WARNING)
        self.assertEqual(result.component, "error_responses")
        self.assertIn("not in JSON format", result.message)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
