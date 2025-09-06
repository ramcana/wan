"""
Test isolation utilities for ensuring tests don't interfere with each other.

This module provides fixtures and utilities for isolating tests from each other,
preventing side effects and ensuring reliable test execution.
"""

import pytest
import tempfile
import shutil
import os
import sys
from pathlib import Path
from typing import Generator, Dict, Any
from unittest.mock import patch


@pytest.fixture
def isolated_temp_dir() -> Generator[Path, None, None]:
    """
    Create an isolated temporary directory for test use.
    
    This fixture creates a temporary directory that is automatically
    cleaned up after the test completes.
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="wan22_test_"))
    try:
        yield temp_dir
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def isolated_working_dir(isolated_temp_dir: Path) -> Generator[Path, None, None]:
    """
    Change to an isolated working directory for the test.
    
    This fixture changes the current working directory to a temporary
    directory and restores the original directory after the test.
    """
    original_cwd = Path.cwd()
    os.chdir(isolated_temp_dir)
    try:
        yield isolated_temp_dir
    finally:
        os.chdir(original_cwd)


@pytest.fixture
def isolated_environment() -> Generator[Dict[str, str], None, None]:
    """
    Provide an isolated environment variable context.
    
    This fixture captures the current environment variables and
    restores them after the test, allowing tests to modify
    environment variables without affecting other tests.
    """
    original_env = os.environ.copy()
    try:
        yield os.environ
    finally:
        os.environ.clear()
        os.environ.update(original_env)


@pytest.fixture
def isolated_sys_path() -> Generator[list, None, None]:
    """
    Provide an isolated sys.path context.
    
    This fixture captures the current sys.path and restores it
    after the test, allowing tests to modify the Python path
    without affecting other tests.
    """
    original_path = sys.path.copy()
    try:
        yield sys.path
    finally:
        sys.path.clear()
        sys.path.extend(original_path)


@pytest.fixture
def isolated_modules() -> Generator[Dict[str, Any], None, None]:
    """
    Provide an isolated module import context.
    
    This fixture allows tests to import modules without affecting
    the global module cache, useful for testing module initialization.
    """
    original_modules = sys.modules.copy()
    modules_to_remove = set()
    
    try:
        yield sys.modules
        # Track new modules added during test
        modules_to_remove = set(sys.modules.keys()) - set(original_modules.keys())
    finally:
        # Remove modules that were imported during the test
        for module_name in modules_to_remove:
            if module_name in sys.modules:
                del sys.modules[module_name]


@pytest.fixture
def mock_file_system(isolated_temp_dir: Path):
    """
    Create a mock file system structure for testing.
    
    This fixture creates a temporary directory structure that
    mimics the project layout for testing file operations.
    """
    # Create mock project structure
    project_dirs = [
        "backend",
        "frontend", 
        "scripts",
        "tools",
        "tests",
        "config",
        "data",
        "logs"
    ]
    
    for dir_name in project_dirs:
        (isolated_temp_dir / dir_name).mkdir(exist_ok=True)
    
    # Create some mock files
    mock_files = {
        "config.json": '{"test": true}',
        "requirements.txt": "pytest>=7.0.0\nfastapi>=0.100.0",
        "README.md": "# Test Project",
        "backend/__init__.py": "",
        "tests/__init__.py": "",
        "scripts/test_script.py": "#!/usr/bin/env python3\nprint('test')"
    }
    
    for file_path, content in mock_files.items():
        full_path = isolated_temp_dir / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)
    
    return isolated_temp_dir


@pytest.fixture
def isolated_config():
    """
    Provide isolated configuration for tests.
    
    This fixture ensures that configuration changes made during
    tests don't affect other tests or the global configuration.
    """
    # Mock configuration that can be safely modified
    test_config = {
        "test_mode": True,
        "debug": True,
        "log_level": "DEBUG",
        "temp_dir": "/tmp/wan22_test",
        "max_workers": 2,
        "timeout": 30
    }
    
    return test_config


@pytest.fixture
def isolated_logger():
    """
    Provide an isolated logger for tests.
    
    This fixture creates a test-specific logger that doesn't
    interfere with the application's logging configuration.
    """
    import logging
    
    # Create a test-specific logger
    logger = logging.getLogger(f"wan22_test_{id(object())}")
    logger.setLevel(logging.DEBUG)
    
    # Add a memory handler for capturing log messages
    from logging.handlers import MemoryHandler
    memory_handler = MemoryHandler(capacity=1000)
    logger.addHandler(memory_handler)
    
    return logger


class ProcessIsolation:
    """
    Context manager for process isolation during tests.
    
    This class provides utilities for isolating process-related
    operations during testing.
    """
    
    def __init__(self):
        self.original_processes = []
        self.test_processes = []
    
    def __enter__(self):
        # Capture current processes (simplified)
        try:
            import psutil
            self.original_processes = [p.pid for p in psutil.process_iter()]
        except ImportError:
            pass
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Clean up any processes started during test
        try:
            import psutil
            current_processes = [p.pid for p in psutil.process_iter()]
            new_processes = set(current_processes) - set(self.original_processes)
            
            for pid in new_processes:
                try:
                    process = psutil.Process(pid)
                    # Only terminate processes that look like test processes
                    cmdline = " ".join(process.cmdline()).lower()
                    if any(keyword in cmdline for keyword in ["test", "mock", "temp"]):
                        process.terminate()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except ImportError:
            pass


@pytest.fixture
def process_isolation():
    """
    Provide process isolation for tests.
    
    This fixture ensures that processes started during tests
    are properly cleaned up.
    """
    with ProcessIsolation() as isolation:
        yield isolation


@pytest.fixture
def network_isolation():
    """
    Provide network isolation for tests.
    
    This fixture can be used to mock network operations
    and prevent tests from making actual network calls.
    """
    with patch('socket.socket'), \
         patch('urllib.request.urlopen'), \
         patch('requests.get'), \
         patch('requests.post'), \
         patch('aiohttp.ClientSession'):
        yield


@pytest.fixture
def database_isolation():
    """
    Provide database isolation for tests.
    
    This fixture ensures that database operations during tests
    don't affect the real database.
    """
    # Mock database connections and operations
    with patch('sqlalchemy.create_engine'), \
         patch('sqlite3.connect'):
        yield


def cleanup_test_artifacts():
    """
    Clean up test artifacts and temporary files.
    
    This function can be called to clean up any test artifacts
    that might have been left behind.
    """
    import glob
    
    # Clean up common test artifact patterns
    patterns = [
        "/tmp/wan22_test*",
        "*.test.log",
        "test_*.tmp",
        ".pytest_cache",
        "__pycache__",
        "*.pyc"
    ]
    
    for pattern in patterns:
        for path in glob.glob(pattern, recursive=True):
            try:
                if os.path.isfile(path):
                    os.remove(path)
                elif os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)
            except (OSError, PermissionError):
                pass


# Auto-cleanup fixture that runs after each test
@pytest.fixture(autouse=True)
def auto_cleanup():
    """
    Automatically clean up test artifacts after each test.
    """
    yield
    cleanup_test_artifacts()