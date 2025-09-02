"""
Shared test fixtures and utilities for consistent testing across the test suite.

This module provides common fixtures, utilities, and helper functions that can be
used across all test files to ensure consistency and reduce code duplication.
"""

import pytest
import tempfile
import shutil
import json
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, Optional, List
from contextlib import contextmanager
import threading
import time


# Add project root to Python path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def project_root():
    """Get the project root directory."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def test_root():
    """Get the test root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def temp_file():
    """Create a temporary file for testing."""
    fd, temp_path = tempfile.mkstemp()
    os.close(fd)
    temp_file_path = Path(temp_path)
    yield temp_file_path
    if temp_file_path.exists():
        temp_file_path.unlink()


@pytest.fixture
def sample_config():
    """Create a sample configuration for testing."""
    return {
        "backend": {
            "port": 8000,
            "host": "localhost",
            "log_level": "info",
            "timeout": 30,
            "reload": False
        },
        "frontend": {
            "port": 3000,
            "host": "localhost",
            "open_browser": True,
            "hot_reload": True
        },
        "logging": {
            "level": "info",
            "file_enabled": True,
            "console_enabled": True,
            "max_file_size": 10485760,
            "backup_count": 5
        },
        "recovery": {
            "enabled": True,
            "max_retry_attempts": 3,
            "retry_delay": 2.0,
            "exponential_backoff": True,
            "auto_kill_processes": False,
            "fallback_ports": [8080, 8081, 8082, 3001, 3002, 3003]
        },
        "environment": {
            "python_min_version": "3.8.0",
            "node_min_version": "16.0.0",
            "npm_min_version": "8.0.0",
            "check_virtual_env": True,
            "validate_dependencies": True,
            "auto_install_missing": False
        },
        "security": {
            "allow_admin_elevation": True,
            "firewall_auto_exception": False,
            "trusted_port_range": [8000, 9000]
        }
    }


@pytest.fixture
def config_file(temp_dir, sample_config):
    """Create a temporary configuration file."""
    config_path = temp_dir / "test_config.json"
    with open(config_path, 'w') as f:
        json.dump(sample_config, f, indent=2)
    return config_path


@pytest.fixture
def mock_process():
    """Create a mock process for testing."""
    process = Mock()
    process.pid = 1234
    process.returncode = None
    process.poll.return_value = None
    process.communicate.return_value = ("output", "")
    process.terminate.return_value = None
    process.kill.return_value = None
    return process


@pytest.fixture
def mock_subprocess_run():
    """Mock subprocess.run for testing."""
    with patch('subprocess.run') as mock_run:
        mock_run.return_value = Mock(
            returncode=0,
            stdout="success",
            stderr=""
        )
        yield mock_run


@pytest.fixture
def mock_psutil():
    """Mock psutil for testing."""
    with patch('psutil.Process') as mock_process_class:
        mock_process = Mock()
        mock_process.pid = 1234
        mock_process.name.return_value = "test_process"
        mock_process.status.return_value = "running"
        mock_process.cpu_percent.return_value = 10.0
        mock_process.memory_info.return_value = Mock(rss=1024*1024*100)  # 100MB
        mock_process.terminate.return_value = None
        mock_process.kill.return_value = None
        
        mock_process_class.return_value = mock_process
        
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value = Mock(
                total=8*1024**3,  # 8GB
                available=4*1024**3,  # 4GB
                percent=50.0
            )
            
            with patch('psutil.disk_usage') as mock_disk:
                mock_disk.return_value = Mock(
                    total=100*1024**3,  # 100GB
                    free=50*1024**3,    # 50GB
                    used=50*1024**3     # 50GB
                )
                
                yield {
                    'process': mock_process,
                    'process_class': mock_process_class,
                    'memory': mock_memory,
                    'disk': mock_disk
                }


@pytest.fixture
def mock_socket():
    """Mock socket operations for testing."""
    with patch('socket.socket') as mock_socket_class:
        mock_sock = Mock()
        mock_sock.connect.return_value = None
        mock_sock.close.return_value = None
        mock_sock.bind.return_value = None
        mock_sock.listen.return_value = None
        
        mock_socket_class.return_value = mock_sock
        yield mock_sock


@pytest.fixture
def isolated_environment():
    """Create an isolated environment for testing."""
    original_env = os.environ.copy()
    original_path = sys.path.copy()
    
    # Clear test-related environment variables
    test_env_vars = [key for key in os.environ.keys() if key.startswith('TEST_') or key.startswith('WAN22_')]
    for var in test_env_vars:
        os.environ.pop(var, None)
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)
    sys.path[:] = original_path


@pytest.fixture
def capture_logs():
    """Capture log output for testing."""
    import logging
    from io import StringIO
    
    log_capture = StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.DEBUG)
    
    # Add handler to root logger
    root_logger = logging.getLogger()
    original_level = root_logger.level
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(handler)
    
    yield log_capture
    
    # Clean up
    root_logger.removeHandler(handler)
    root_logger.setLevel(original_level)


@contextmanager
def timeout_context(seconds: float):
    """Context manager for timing out operations."""
    def timeout_handler():
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    
    timer = threading.Timer(seconds, timeout_handler)
    timer.start()
    
    try:
        yield
    finally:
        timer.cancel()


class MockServer:
    """Mock server for testing network operations."""
    
    def __init__(self, port: int = 8000):
        self.port = port
        self.running = False
        self.requests = []
    
    def start(self):
        """Start the mock server."""
        self.running = True
        return True
    
    def stop(self):
        """Stop the mock server."""
        self.running = False
        return True
    
    def add_request(self, request: Dict[str, Any]):
        """Add a mock request."""
        self.requests.append(request)
    
    def get_requests(self) -> List[Dict[str, Any]]:
        """Get all mock requests."""
        return self.requests.copy()


@pytest.fixture
def mock_server():
    """Create a mock server for testing."""
    server = MockServer()
    yield server
    server.stop()


class TestDataFactory:
    """Factory for creating test data."""
    
    @staticmethod
    def create_user_data(name: str = "test_user", email: str = "test@example.com") -> Dict[str, Any]:
        """Create test user data."""
        return {
            "name": name,
            "email": email,
            "id": hash(name) % 10000,
            "created_at": "2023-01-01T00:00:00Z",
            "active": True
        }
    
    @staticmethod
    def create_config_data(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create test configuration data."""
        base_config = {
            "backend": {"port": 8000, "host": "localhost"},
            "frontend": {"port": 3000, "host": "localhost"},
            "logging": {"level": "info"},
            "recovery": {"enabled": True, "max_retry_attempts": 3}
        }
        
        if overrides:
            base_config.update(overrides)
        
        return base_config
    
    @staticmethod
    def create_process_data(name: str = "test_process", pid: int = 1234) -> Dict[str, Any]:
        """Create test process data."""
        return {
            "name": name,
            "pid": pid,
            "status": "running",
            "cpu_percent": 10.0,
            "memory_mb": 100,
            "start_time": time.time()
        }


@pytest.fixture
def test_data_factory():
    """Provide the test data factory."""
    return TestDataFactory


class AssertionHelpers:
    """Helper methods for common test assertions."""
    
    @staticmethod
    def assert_file_exists(file_path: Path, message: str = ""):
        """Assert that a file exists."""
        assert file_path.exists(), f"File {file_path} does not exist. {message}"
    
    @staticmethod
    def assert_file_contains(file_path: Path, content: str, message: str = ""):
        """Assert that a file contains specific content."""
        AssertionHelpers.assert_file_exists(file_path)
        with open(file_path, 'r') as f:
            file_content = f.read()
        assert content in file_content, f"File {file_path} does not contain '{content}'. {message}"
    
    @staticmethod
    def assert_json_file_valid(file_path: Path, message: str = ""):
        """Assert that a JSON file is valid."""
        AssertionHelpers.assert_file_exists(file_path)
        try:
            with open(file_path, 'r') as f:
                json.load(f)
        except json.JSONDecodeError as e:
            pytest.fail(f"File {file_path} is not valid JSON: {e}. {message}")
    
    @staticmethod
    def assert_port_available(port: int, message: str = ""):
        """Assert that a port is available."""
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            result = sock.connect_ex(('localhost', port))
            assert result != 0, f"Port {port} is not available (already in use). {message}"
    
    @staticmethod
    def assert_process_running(pid: int, message: str = ""):
        """Assert that a process is running."""
        try:
            import psutil
            process = psutil.Process(pid)
            assert process.is_running(), f"Process {pid} is not running. {message}"
        except psutil.NoSuchProcess:
            pytest.fail(f"Process {pid} does not exist. {message}")
    
    @staticmethod
    def assert_dict_subset(subset: Dict[str, Any], superset: Dict[str, Any], message: str = ""):
        """Assert that one dict is a subset of another."""
        for key, value in subset.items():
            assert key in superset, f"Key '{key}' not found in superset. {message}"
            assert superset[key] == value, f"Value mismatch for key '{key}': expected {value}, got {superset[key]}. {message}"


@pytest.fixture
def assert_helpers():
    """Provide assertion helpers."""
    return AssertionHelpers


# Performance testing utilities
class PerformanceTimer:
    """Context manager for timing operations."""
    
    def __init__(self, name: str = "operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.duration = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time
    
    def assert_duration_less_than(self, max_duration: float):
        """Assert that the operation took less than the specified duration."""
        assert self.duration is not None, "Timer was not used as context manager"
        assert self.duration < max_duration, f"{self.name} took {self.duration:.2f}s, expected < {max_duration}s"


@pytest.fixture
def performance_timer():
    """Provide performance timer."""
    return PerformanceTimer


# Cleanup utilities
@pytest.fixture(autouse=True)
def cleanup_test_artifacts():
    """Automatically clean up test artifacts after each test."""
    yield
    
    # Clean up any temporary files or processes created during tests
    import tempfile
    import shutil
    
    # Clean up temporary directories that might have been created
    temp_dirs = []
    for item in Path(tempfile.gettempdir()).iterdir():
        if item.is_dir() and item.name.startswith('tmp') and 'test' in item.name.lower():
            temp_dirs.append(item)
    
    for temp_dir in temp_dirs:
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except:
            pass


# Test markers for categorization
def requires_network():
    """Mark test as requiring network access."""
    return pytest.mark.network


def requires_admin():
    """Mark test as requiring admin privileges."""
    return pytest.mark.admin


def slow_test():
    """Mark test as slow running."""
    return pytest.mark.slow


def integration_test():
    """Mark test as integration test."""
    return pytest.mark.integration


def performance_test():
    """Mark test as performance test."""
    return pytest.mark.performance


# Export commonly used fixtures and utilities
__all__ = [
    'project_root', 'test_root', 'temp_dir', 'temp_file',
    'sample_config', 'config_file', 'mock_process', 'mock_subprocess_run',
    'mock_psutil', 'mock_socket', 'isolated_environment', 'capture_logs',
    'mock_server', 'test_data_factory', 'assert_helpers', 'performance_timer',
    'timeout_context', 'MockServer', 'TestDataFactory', 'AssertionHelpers',
    'PerformanceTimer', 'requires_network', 'requires_admin', 'slow_test',
    'integration_test', 'performance_test'
]