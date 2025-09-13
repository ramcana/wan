"""
Comprehensive tests for the Network Failure Recovery System.

Tests all aspects of network failure recovery including:
- Error detection and classification
- Alternative source selection
- Resume capability for downloads
- Proxy and authentication handling
- Network connectivity testing
- Recovery strategy execution
"""

import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock, mock_open
import tempfile
import shutil
import json
import socket
import urllib.error
import ssl
from pathlib import Path
from datetime import datetime, timedelta
import threading
import time

# Import the modules to test
from scripts.network_failure_recovery import (
    NetworkFailureRecovery, NetworkErrorDetector, NetworkConnectivityTester,
    DownloadMirrorManager, ResumableDownloader, NetworkErrorType, DownloadSource,
    NetworkConfiguration, ProxyConfiguration, AuthenticationConfiguration,
    DownloadMirror, DownloadProgress, NetworkTestResult
)
from scripts.interfaces import ValidationResult


class TestNetworkErrorDetector(unittest.TestCase):
    """Test network error detection and classification."""
    
    def setUp(self):
        self.detector = NetworkErrorDetector()
    
    def test_detect_connection_timeout(self):
        """Test detection of connection timeout errors."""
        error = socket.timeout("Connection timed out")
        error_type = self.detector.detect_error_type(error)
        self.assertEqual(error_type, NetworkErrorType.CONNECTION_TIMEOUT)
    
    def test_detect_dns_resolution_error(self):
        """Test detection of DNS resolution errors."""
        error = socket.gaierror("Name resolution failed")
        error_type = self.detector.detect_error_type(error)
        self.assertEqual(error_type, NetworkErrorType.DNS_RESOLUTION)
    
    def test_detect_ssl_error(self):
        """Test detection of SSL certificate errors."""
        error = ssl.SSLError("Certificate verification failed")
        error_type = self.detector.detect_error_type(error)
        self.assertEqual(error_type, NetworkErrorType.SSL_CERTIFICATE)
    
    def test_detect_authentication_error_by_code(self):
        """Test detection of authentication errors by response code."""
        error = Exception("Unauthorized access")
        error_type = self.detector.detect_error_type(error, response_code=401)
        self.assertEqual(error_type, NetworkErrorType.AUTHENTICATION)
    
    def test_detect_rate_limiting_error(self):
        """Test detection of rate limiting errors."""
        error = Exception("Too many requests")
        error_type = self.detector.detect_error_type(error, response_code=429)
        self.assertEqual(error_type, NetworkErrorType.RATE_LIMITING)
    
    def test_detect_proxy_error(self):
        """Test detection of proxy errors."""
        error = Exception("Proxy authentication required")
        error_type = self.detector.detect_error_type(error, response_code=407)
        self.assertEqual(error_type, NetworkErrorType.PROXY_ERROR)
    
    def test_detect_server_error(self):
        """Test detection of server errors."""
        error = Exception("Internal server error")
        error_type = self.detector.detect_error_type(error, response_code=500)
        self.assertEqual(error_type, NetworkErrorType.SERVER_ERROR)
    
    def test_is_retryable_error(self):
        """Test determination of retryable errors."""
        retryable_errors = [
            NetworkErrorType.CONNECTION_TIMEOUT,
            NetworkErrorType.SERVER_ERROR,
            NetworkErrorType.PARTIAL_DOWNLOAD
        ]
        
        non_retryable_errors = [
            NetworkErrorType.AUTHENTICATION,
            NetworkErrorType.SSL_CERTIFICATE
        ]
        
        for error_type in retryable_errors:
            self.assertTrue(self.detector.is_retryable_error(error_type))
        
        for error_type in non_retryable_errors:
            self.assertFalse(self.detector.is_retryable_error(error_type))
    
    def test_requires_alternative_source(self):
        """Test determination of errors requiring alternative sources."""
        alt_source_errors = [
            NetworkErrorType.AUTHENTICATION,
            NetworkErrorType.RATE_LIMITING,
            NetworkErrorType.SERVER_ERROR
        ]
        
        for error_type in alt_source_errors:
            self.assertTrue(self.detector.requires_alternative_source(error_type))


class TestNetworkConnectivityTester(unittest.TestCase):
    """Test network connectivity testing functionality."""
    
    def setUp(self):
        self.config = NetworkConfiguration()
        self.tester = NetworkConnectivityTester(self.config)
    
    @patch('socket.create_connection')
    @patch('time.time')
    def test_basic_connectivity_success(self, mock_time, mock_connection):
        """Test successful basic connectivity test."""
        mock_socket = Mock()
        mock_connection.return_value = mock_socket
        
        # Mock time to simulate latency - need enough values for all time.time() calls
        mock_time.side_effect = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
        
        result = self.tester.test_basic_connectivity(timeout=5)
        
        self.assertTrue(result.success)
        self.assertGreaterEqual(result.latency_ms, 0)
        mock_socket.close.assert_called()
    
    @patch('socket.create_connection')
    def test_basic_connectivity_failure(self, mock_connection):
        """Test failed basic connectivity test."""
        mock_connection.side_effect = socket.timeout("Connection timed out")
        
        result = self.tester.test_basic_connectivity(timeout=5)
        
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error_message)
    
    @patch('urllib.request.urlopen')
    @patch('time.time')
    def test_bandwidth_test_success(self, mock_time, mock_urlopen):
        """Test successful bandwidth measurement."""
        mock_response = Mock()
        mock_response.read.return_value = b'x' * (1024 * 1024)  # 1MB of data
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        # Mock time to simulate download duration
        mock_time.side_effect = [0.0, 0.0, 1.0, 1.0]  # Start and end times
        
        result = self.tester.test_bandwidth(timeout=10)
        
        self.assertTrue(result.success)
        self.assertGreaterEqual(result.bandwidth_mbps, 0)
    
    @patch('urllib.request.urlopen')
    def test_bandwidth_test_failure(self, mock_urlopen):
        """Test failed bandwidth measurement."""
        mock_urlopen.side_effect = urllib.error.URLError("Network error")
        
        result = self.tester.test_bandwidth(timeout=10)
        
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error_message)
    
    @patch('socket.gethostbyname')
    def test_dns_resolution_test(self, mock_gethostbyname):
        """Test DNS resolution testing."""
        mock_gethostbyname.return_value = "8.8.8.8"
        
        result = self.tester._test_dns_resolution()
        self.assertTrue(result)
        
        mock_gethostbyname.side_effect = socket.gaierror("DNS error")
        result = self.tester._test_dns_resolution()
        self.assertFalse(result)


class TestDownloadMirrorManager(unittest.TestCase):
    """Test download mirror management functionality."""
    
    def setUp(self):
        self.config = NetworkConfiguration()
        self.manager = DownloadMirrorManager(self.config)
    
    def test_initialize_default_mirrors(self):
        """Test initialization of default mirrors."""
        self.assertIn("huggingface", self.manager.mirrors)
        self.assertIn("github", self.manager.mirrors)
        self.assertIn("pypi", self.manager.mirrors)
        
        # Check that mirrors are sorted by priority
        hf_mirrors = self.manager.mirrors["huggingface"]
        priorities = [mirror.priority for mirror in hf_mirrors]
        self.assertEqual(priorities, sorted(priorities))
    
    def test_add_mirror(self):
        """Test adding a new mirror."""
        new_mirror = DownloadMirror(
            name="test_mirror",
            base_url="https://test.example.com",
            priority=1
        )
        
        self.manager.add_mirror("test_service", new_mirror)
        
        self.assertIn("test_service", self.manager.mirrors)
        self.assertIn(new_mirror, self.manager.mirrors["test_service"])
    
    def test_get_available_mirrors(self):
        """Test getting available mirrors."""
        mirrors = self.manager.get_available_mirrors("huggingface")
        
        # Should return only available mirrors
        for mirror in mirrors:
            self.assertTrue(mirror.available)
        
        # Should be sorted by priority
        priorities = [mirror.priority for mirror in mirrors]
        self.assertEqual(priorities, sorted(priorities))
    
    def test_mark_mirror_failure(self):
        """Test marking mirror as failed."""
        mirror = self.manager.mirrors["huggingface"][0]
        original_failure_count = mirror.failure_count
        
        self.manager.mark_mirror_failure(mirror, Exception("Test error"))
        
        self.assertEqual(mirror.failure_count, original_failure_count + 1)
        self.assertIsNotNone(mirror.last_check)
    
    def test_mark_mirror_success(self):
        """Test marking mirror as successful."""
        mirror = self.manager.mirrors["huggingface"][0]
        original_success_count = mirror.success_count
        
        self.manager.mark_mirror_success(mirror)
        
        self.assertEqual(mirror.success_count, original_success_count + 1)
        self.assertTrue(mirror.available)
        self.assertIsNotNone(mirror.last_check)
    
    def test_mirror_disable_after_failures(self):
        """Test that mirrors are disabled after too many failures."""
        mirror = self.manager.mirrors["huggingface"][0]
        
        # Mark multiple failures
        for _ in range(3):
            self.manager.mark_mirror_failure(mirror, Exception("Test error"))
        
        self.assertFalse(mirror.available)
    
    def test_get_mirror_statistics(self):
        """Test getting mirror statistics."""
        stats = self.manager.get_mirror_statistics()
        
        self.assertIn("huggingface", stats)
        self.assertIn("total_mirrors", stats["huggingface"])
        self.assertIn("available_mirrors", stats["huggingface"])
        self.assertIn("mirrors", stats["huggingface"])
        
        # Check mirror details
        mirror_stats = stats["huggingface"]["mirrors"][0]
        self.assertIn("name", mirror_stats)
        self.assertIn("success_rate", mirror_stats)


class TestResumableDownloader(unittest.TestCase):
    """Test resumable download functionality."""
    
    def setUp(self):
        self.config = NetworkConfiguration()
        self.downloader = ResumableDownloader(self.config)
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_download_progress_tracking(self):
        """Test download progress tracking."""
        url = "https://example.com/test.txt"
        destination = self.temp_dir / "test.txt"
        
        progress = DownloadProgress(
            url=url,
            filename=destination.name,
            total_size=1000,
            downloaded_size=0,
            start_time=datetime.now(),
            current_speed_mbps=0.0,
            average_speed_mbps=0.0,
            eta_seconds=None
        )
        
        download_id = f"{url}_{destination.name}"
        self.downloader.active_downloads[download_id] = progress
        
        retrieved_progress = self.downloader.get_download_progress(download_id)
        self.assertEqual(retrieved_progress.url, url)
        self.assertEqual(retrieved_progress.filename, destination.name)
    
    def test_get_active_downloads(self):
        """Test getting active downloads."""
        # Add some mock downloads
        for i in range(3):
            url = f"https://example.com/test{i}.txt"
            destination = self.temp_dir / f"test{i}.txt"
            download_id = f"{url}_{destination.name}"
            
            progress = DownloadProgress(
                url=url,
                filename=destination.name,
                total_size=1000,
                downloaded_size=0,
                start_time=datetime.now(),
                current_speed_mbps=0.0,
                average_speed_mbps=0.0,
                eta_seconds=None
            )
            
            self.downloader.active_downloads[download_id] = progress
        
        active_downloads = self.downloader.get_active_downloads()
        self.assertEqual(len(active_downloads), 3)
    
    def test_cancel_download(self):
        """Test canceling a download."""
        url = "https://example.com/test.txt"
        destination = self.temp_dir / "test.txt"
        download_id = f"{url}_{destination.name}"
        
        progress = DownloadProgress(
            url=url,
            filename=destination.name,
            total_size=1000,
            downloaded_size=0,
            start_time=datetime.now(),
            current_speed_mbps=0.0,
            average_speed_mbps=0.0,
            eta_seconds=None
        )
        
        self.downloader.active_downloads[download_id] = progress
        
        # Cancel the download
        result = self.downloader.cancel_download(download_id)
        self.assertTrue(result)
        self.assertNotIn(download_id, self.downloader.active_downloads)
        
        # Try to cancel non-existent download
        result = self.downloader.cancel_download("non_existent")
        self.assertFalse(result)
    
    @patch('hashlib.new')
    @patch('builtins.open', new_callable=mock_open, read_data=b'test data')
    def test_verify_checksum_success(self, mock_file, mock_hasher):
        """Test successful checksum verification."""
        mock_hash = Mock()
        mock_hash.hexdigest.return_value = "abcd1234"
        mock_hasher.return_value = mock_hash
        
        file_path = self.temp_dir / "test.txt"
        result = self.downloader._verify_checksum(file_path, "sha256:abcd1234")
        
        self.assertTrue(result)
        mock_hasher.assert_called_with('sha256')
    
    @patch('hashlib.new')
    @patch('builtins.open', new_callable=mock_open, read_data=b'test data')
    def test_verify_checksum_failure(self, mock_file, mock_hasher):
        """Test failed checksum verification."""
        mock_hash = Mock()
        mock_hash.hexdigest.return_value = "abcd1234"
        mock_hasher.return_value = mock_hash
        
        file_path = self.temp_dir / "test.txt"
        result = self.downloader._verify_checksum(file_path, "sha256:different_hash")
        
        self.assertFalse(result)


class TestNetworkFailureRecovery(unittest.TestCase):
    """Test the main network failure recovery system."""
    
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.recovery = NetworkFailureRecovery(str(self.temp_dir))
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test proper initialization of the recovery system."""
        self.assertIsNotNone(self.recovery.network_config)
        self.assertIsNotNone(self.recovery.error_detector)
        self.assertIsNotNone(self.recovery.connectivity_tester)
        self.assertIsNotNone(self.recovery.mirror_manager)
        self.assertIsNotNone(self.recovery.downloader)
        
        # Check that statistics are initialized
        self.assertIn('total_attempts', self.recovery.recovery_stats)
        self.assertEqual(self.recovery.recovery_stats['total_attempts'], 0)
    
    @patch('scripts.network_failure_recovery.NetworkConnectivityTester.test_basic_connectivity')
    def test_test_network_connectivity(self, mock_test):
        """Test network connectivity testing."""
        mock_result = NetworkTestResult(
            success=True,
            latency_ms=50.0,
            bandwidth_mbps=10.0
        )
        mock_test.return_value = mock_result
        
        result = self.recovery.test_network_connectivity()
        
        self.assertTrue(result.success)
        self.assertEqual(result.latency_ms, 50.0)
        mock_test.assert_called_once()
    
    def test_get_recovery_strategies_authentication(self):
        """Test getting recovery strategies for authentication errors."""
        strategies = self.recovery._get_recovery_strategies(
            NetworkErrorType.AUTHENTICATION, "https://example.com/file.txt"
        )
        
        strategy_names = [name for name, _ in strategies]
        self.assertIn("fix_authentication", strategy_names)
        self.assertIn("try_alternative_source", strategy_names)
    
    def test_get_recovery_strategies_rate_limiting(self):
        """Test getting recovery strategies for rate limiting errors."""
        strategies = self.recovery._get_recovery_strategies(
            NetworkErrorType.RATE_LIMITING, "https://example.com/file.txt"
        )
        
        strategy_names = [name for name in strategies]
        self.assertIn("wait_for_rate_limit", [name for name, _ in strategies])
        self.assertIn("try_alternative_source", [name for name, _ in strategies])
    
    def test_get_recovery_strategies_proxy_error(self):
        """Test getting recovery strategies for proxy errors."""
        strategies = self.recovery._get_recovery_strategies(
            NetworkErrorType.PROXY_ERROR, "https://example.com/file.txt"
        )
        
        strategy_names = [name for name, _ in strategies]
        self.assertIn("fix_proxy_configuration", strategy_names)
        self.assertIn("bypass_proxy", strategy_names)
    
    def test_detect_service_from_url(self):
        """Test service detection from URLs."""
        test_cases = [
            ("https://huggingface.co/model/file.bin", "huggingface"),
            ("https://github.com/user/repo/releases/file.zip", "github"),
            ("https://pypi.org/simple/package/", "pypi"),
            ("https://unknown.com/file.txt", None)
        ]
        
        for url, expected_service in test_cases:
            result = self.recovery._detect_service_from_url(url)
            self.assertEqual(result, expected_service)
    
    def test_configure_authentication(self):
        """Test authentication configuration."""
        self.recovery.configure_authentication(
            username="testuser",
            password="testpass",
            token="testtoken",
            api_key="testapikey"
        )
        
        self.assertEqual(self.recovery.auth_config.username, "testuser")
        self.assertEqual(self.recovery.auth_config.password, "testpass")
        self.assertEqual(self.recovery.auth_config.token, "testtoken")
        self.assertEqual(self.recovery.auth_config.api_key, "testapikey")
    
    def test_configure_proxy(self):
        """Test proxy configuration."""
        self.recovery.configure_proxy(
            http_proxy="http://proxy.example.com:8080",
            https_proxy="https://proxy.example.com:8080",
            no_proxy="localhost,127.0.0.1"
        )
        
        self.assertEqual(self.recovery.proxy_config.http_proxy, "http://proxy.example.com:8080")
        self.assertEqual(self.recovery.proxy_config.https_proxy, "https://proxy.example.com:8080")
        self.assertEqual(self.recovery.proxy_config.no_proxy, "localhost,127.0.0.1")
    
    @patch('scripts.network_failure_recovery.NetworkConnectivityTester.test_basic_connectivity')
    def test_validate_network_configuration_success(self, mock_test):
        """Test successful network configuration validation."""
        mock_result = NetworkTestResult(
            success=True,
            latency_ms=50.0,
            bandwidth_mbps=10.0,
            dns_working=True,
            proxy_working=True
        )
        mock_test.return_value = mock_result
        
        result = self.recovery.validate_network_configuration()
        
        self.assertTrue(result.success)
        self.assertEqual(result.message, "Network configuration is valid")
    
    @patch('scripts.network_failure_recovery.NetworkConnectivityTester.test_basic_connectivity')
    def test_validate_network_configuration_failure(self, mock_test):
        """Test failed network configuration validation."""
        mock_result = NetworkTestResult(
            success=False,
            latency_ms=0.0,
            bandwidth_mbps=0.0,
            error_message="Network unreachable",
            dns_working=False,
            proxy_working=False
        )
        mock_test.return_value = mock_result
        
        result = self.recovery.validate_network_configuration()
        
        self.assertFalse(result.success)
        self.assertIn("issues", result.details)
        self.assertGreater(len(result.details["issues"]), 0)
    
    def test_get_recovery_statistics(self):
        """Test getting recovery statistics."""
        # Simulate some recovery attempts
        self.recovery.recovery_stats['total_attempts'] = 10
        self.recovery.recovery_stats['successful_recoveries'] = 7
        self.recovery.recovery_stats['failed_recoveries'] = 3
        
        stats = self.recovery.get_recovery_statistics()
        
        self.assertEqual(stats['total_attempts'], 10)
        self.assertEqual(stats['successful_recoveries'], 7)
        self.assertEqual(stats['failed_recoveries'], 3)
        self.assertEqual(stats['success_rate'], 0.7)
        self.assertIn('mirror_statistics', stats)
        self.assertIn('active_downloads', stats)
    
    def test_retry_with_backoff_strategy(self):
        """Test retry with backoff strategy."""
        call_count = 0
        
        def mock_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return "success"
        
        with patch('time.sleep'):  # Mock sleep to speed up test
            result = self.recovery._retry_with_backoff(
                mock_operation, "test_operation", "https://example.com", Exception("test")
            )
        
        self.assertEqual(result, "success")
        self.assertEqual(call_count, 3)
    
    def test_wait_for_rate_limit_strategy(self):
        """Test wait for rate limit strategy."""
        call_count = 0
        
        def mock_operation():
            nonlocal call_count
            call_count += 1
            return "success"
        
        with patch('time.sleep') as mock_sleep:
            result = self.recovery._wait_for_rate_limit(
                mock_operation, "test_operation", "https://example.com", Exception("rate limited")
            )
        
        self.assertEqual(result, "success")
        mock_sleep.assert_called_once_with(60)  # Default rate limit wait
    
    def test_increase_timeout_strategy(self):
        """Test increase timeout strategy."""
        original_timeout = self.recovery.network_config.read_timeout
        
        def mock_operation():
            # Verify timeout was increased
            self.assertEqual(self.recovery.network_config.read_timeout, original_timeout * 2)
            return "success"
        
        result = self.recovery._increase_timeout(
            mock_operation, "test_operation", "https://example.com", Exception("timeout")
        )
        
        self.assertEqual(result, "success")
        # Verify timeout was restored
        self.assertEqual(self.recovery.network_config.read_timeout, original_timeout)
    
    def test_bypass_proxy_strategy(self):
        """Test bypass proxy strategy."""
        # Set up proxy configuration
        self.recovery.proxy_config = ProxyConfiguration(
            http_proxy="http://proxy.example.com:8080"
        )
        original_proxy = self.recovery.proxy_config
        
        def mock_operation():
            # Verify proxy was bypassed
            self.assertIsNone(self.recovery.proxy_config.http_proxy)
            return "success"
        
        result = self.recovery._bypass_proxy(
            mock_operation, "test_operation", "https://example.com", Exception("proxy error")
        )
        
        self.assertEqual(result, "success")
        # Verify proxy configuration was restored
        self.assertEqual(self.recovery.proxy_config, original_proxy)
    
    @patch('scripts.network_failure_recovery.ResumableDownloader.download_with_resume')
    def test_download_with_recovery_success(self, mock_download):
        """Test successful download with recovery."""
        mock_download.return_value = True
        
        destination = self.temp_dir / "test_file.bin"
        result = self.recovery.download_with_recovery(
            "https://example.com/test_file.bin",
            destination,
            expected_size=1024,
            checksum="sha256:abcd1234"
        )
        
        self.assertTrue(result)
        mock_download.assert_called_once()
    
    @patch('scripts.network_failure_recovery.ResumableDownloader.download_with_resume')
    def test_download_with_recovery_failure(self, mock_download):
        """Test failed download with recovery."""
        mock_download.side_effect = Exception("Download failed")
        
        destination = self.temp_dir / "test_file.bin"
        result = self.recovery.download_with_recovery(
            "https://example.com/test_file.bin",
            destination
        )
        
        self.assertFalse(result)


class TestNetworkFailureRecoveryIntegration(unittest.TestCase):
    """Integration tests for network failure recovery."""
    
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.recovery = NetworkFailureRecovery(str(self.temp_dir))
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_full_recovery_workflow_authentication_error(self):
        """Test full recovery workflow for authentication errors."""
        call_count = 0
        
        def failing_operation():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call fails with authentication error
                raise urllib.error.HTTPError(
                    "https://example.com", 401, "Unauthorized", {}, None
                )
            else:
                # Subsequent calls succeed (simulating fixed authentication)
                return "success"
        
        with patch('time.sleep'):  # Speed up test
            result = self.recovery.recover_from_network_failure(
                failing_operation, "test_download", "https://example.com/file.txt",
                urllib.error.HTTPError("https://example.com", 401, "Unauthorized", {}, None),
                max_attempts=2
            )
        
        self.assertEqual(result, "success")
        self.assertGreater(self.recovery.recovery_stats['total_attempts'], 0)
    
    def test_full_recovery_workflow_rate_limiting(self):
        """Test full recovery workflow for rate limiting errors."""
        call_count = 0
        
        def rate_limited_operation():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call fails with rate limiting
                error = urllib.error.HTTPError(
                    "https://example.com", 429, "Too Many Requests", 
                    {"Retry-After": "1"}, None
                )
                error.headers = {"Retry-After": "1"}
                raise error
            else:
                # Subsequent calls succeed
                return "success"
        
        with patch('time.sleep'):  # Speed up test
            result = self.recovery.recover_from_network_failure(
                rate_limited_operation, "test_download", "https://example.com/file.txt",
                urllib.error.HTTPError("https://example.com", 429, "Too Many Requests", {}, None),
                max_attempts=2
            )
        
        self.assertEqual(result, "success")
    
    def test_recovery_failure_after_max_attempts(self):
        """Test recovery failure after maximum attempts."""
        def always_failing_operation():
            raise Exception("Persistent failure")
        
        with patch('time.sleep'):  # Speed up test
            with self.assertRaises(Exception):
                self.recovery.recover_from_network_failure(
                    always_failing_operation, "test_download", "https://example.com/file.txt",
                    Exception("Initial failure"),
                    max_attempts=2
                )
        
        self.assertGreater(self.recovery.recovery_stats['failed_recoveries'], 0)
    
    @patch('scripts.network_failure_recovery.NetworkConnectivityTester.test_basic_connectivity')
    def test_network_validation_with_warnings(self, mock_test):
        """Test network validation with warnings for poor performance."""
        mock_result = NetworkTestResult(
            success=True,
            latency_ms=1500.0,  # High latency
            bandwidth_mbps=0.5,  # Low bandwidth
            dns_working=True,
            proxy_working=True
        )
        mock_test.return_value = mock_result
        
        result = self.recovery.validate_network_configuration()
        
        self.assertTrue(result.success)  # Still successful but with warnings
        self.assertIsNotNone(result.warnings)
        self.assertGreater(len(result.warnings), 0)
        
        # Check for specific warnings
        warning_text = " ".join(result.warnings)
        self.assertIn("latency", warning_text.lower())
        self.assertIn("bandwidth", warning_text.lower())


if __name__ == '__main__':
    # Create a test suite
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestNetworkErrorDetector,
        TestNetworkConnectivityTester,
        TestDownloadMirrorManager,
        TestResumableDownloader,
        TestNetworkFailureRecovery,
        TestNetworkFailureRecoveryIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Network Failure Recovery Tests Summary")
    print(f"{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            error_msg = traceback.split('AssertionError: ')[-1].split('\n')[0]
            print(f"- {test}: {error_msg}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            error_msg = traceback.split('\n')[-2]
            print(f"- {test}: {error_msg}")
