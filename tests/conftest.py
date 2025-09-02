"""
Main pytest configuration and shared fixtures for the WAN22 test suite.

This file provides project-wide test configuration and imports shared fixtures
from the utils module to make them available to all test files.
"""

import pytest
import sys
import os
from pathlib import Path

# Add the project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import all shared fixtures to make them available project-wide
from tests.utils.shared_fixtures import *
from tests.utils.test_isolation import *
from tests.utils.test_execution_engine import *
from tests.fixtures.fixture_manager import *


def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Add custom markers
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "network: Tests requiring network access")
    config.addinivalue_line("markers", "admin: Tests requiring admin privileges")
    config.addinivalue_line("markers", "windows: Windows-specific tests")
    config.addinivalue_line("markers", "unix: Unix/Linux-specific tests")
    config.addinivalue_line("markers", "docker: Tests requiring Docker")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add markers based on test file paths
        test_path = str(item.fspath)
        
        if "integration" in test_path:
            item.add_marker(pytest.mark.integration)
        elif "performance" in test_path:
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)
        elif "e2e" in test_path:
            item.add_marker(pytest.mark.e2e)
            item.add_marker(pytest.mark.slow)
        elif "stress" in test_path:
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)
        elif "reliability" in test_path:
            item.add_marker(pytest.mark.integration)
            item.add_marker(pytest.mark.slow)
        else:
            item.add_marker(pytest.mark.unit)
        
        # Add slow marker for tests that take a long time
        if any(keyword in item.name.lower() for keyword in ["stress", "scalability", "benchmark", "load"]):
            item.add_marker(pytest.mark.slow)
        
        # Add network marker for tests that need network
        if any(keyword in item.name.lower() for keyword in ["network", "download", "upload", "http", "api"]):
            item.add_marker(pytest.mark.network)
        
        # Add admin marker for tests that need admin privileges
        if any(keyword in item.name.lower() for keyword in ["admin", "privilege", "elevation", "firewall"]):
            item.add_marker(pytest.mark.admin)
        
        # Add platform-specific markers
        if "windows" in test_path.lower() or "win" in item.name.lower():
            item.add_marker(pytest.mark.windows)
        elif "unix" in test_path.lower() or "linux" in test_path.lower():
            item.add_marker(pytest.mark.unix)


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up the test environment before running tests."""
    # Ensure test directories exist
    test_dirs = [
        "tests/fixtures/data",
        "tests/fixtures/configs",
        "tests/fixtures/mocks",
        "tests/fixtures/temp",
        "tests/logs",
        "tests/reports"
    ]
    
    for test_dir in test_dirs:
        Path(test_dir).mkdir(parents=True, exist_ok=True)
    
    # Set test environment variables
    os.environ["TESTING"] = "true"
    os.environ["WAN22_TEST_MODE"] = "true"
    
    yield
    
    # Cleanup after all tests
    # Remove test environment variables
    os.environ.pop("TESTING", None)
    os.environ.pop("WAN22_TEST_MODE", None)


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset any singleton instances between tests."""
    yield
    
    # Reset any module-level caches or singletons
    # This ensures tests don't interfere with each other
    import importlib
    
    # List of modules that might have singletons to reset
    modules_to_reset = [
        'scripts.startup_manager.config',
        'tools.health-checker.health_checker',
        'tools.config_manager.config_unifier'
    ]
    
    for module_name in modules_to_reset:
        if module_name in sys.modules:
            try:
                module = sys.modules[module_name]
                # Reset common singleton patterns
                if hasattr(module, '_instance'):
                    module._instance = None
                if hasattr(module, '_instances'):
                    module._instances.clear()
                if hasattr(module, 'reset'):
                    module.reset()
            except:
                pass


@pytest.fixture(autouse=True)
def cleanup_processes():
    """Automatically clean up any processes created during tests."""
    yield
    
    # Clean up any test processes that might be running
    try:
        import psutil
        current_process = psutil.Process()
        
        children = current_process.children(recursive=True)
        for child in children:
            try:
                cmdline = " ".join(child.cmdline()).lower()
                if any(keyword in cmdline for keyword in ["test", "pytest", "mock", "temp"]):
                    child.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except ImportError:
        # psutil not available, skip cleanup
        pass
    except Exception:
        # Other errors, skip cleanup to avoid test failures
        pass


# Platform-specific fixtures
@pytest.fixture
def skip_if_not_windows():
    """Skip test if not running on Windows."""
    if os.name != 'nt':
        pytest.skip("Windows-only test")


@pytest.fixture
def skip_if_not_unix():
    """Skip test if not running on Unix-like system."""
    if os.name == 'nt':
        pytest.skip("Unix-only test")


@pytest.fixture(scope="session")
def docker_available():
    """Check if Docker is available for container-based tests."""
    try:
        import subprocess
        result = subprocess.run(
            ["docker", "--version"], 
            capture_output=True, 
            timeout=10
        )
        return result.returncode == 0
    except:
        return False


@pytest.fixture
def skip_if_no_docker(docker_available):
    """Skip test if Docker is not available."""
    if not docker_available:
        pytest.skip("Docker not available")


@pytest.fixture(scope="session")
def network_available():
    """Check if network is available for network-dependent tests."""
    try:
        import socket
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except:
        return False


@pytest.fixture
def skip_if_no_network(network_available):
    """Skip test if network is not available."""
    if not network_available:
        pytest.skip("Network not available")


# Test data management
@pytest.fixture(scope="session")
def test_data_dir():
    """Get the test data directory."""
    data_dir = Path(__file__).parent / "fixtures" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


@pytest.fixture(scope="session")
def test_config_dir():
    """Get the test configuration directory."""
    config_dir = Path(__file__).parent / "fixtures" / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


@pytest.fixture(scope="session")
def test_mock_dir():
    """Get the test mock directory."""
    mock_dir = Path(__file__).parent / "fixtures" / "mocks"
    mock_dir.mkdir(parents=True, exist_ok=True)
    return mock_dir