"""
Pytest configuration and shared fixtures for the test suite.
"""

import pytest
import sys
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def test_data_dir():
    """Get the test data directory."""
    return Path(__file__).parent / "test_data"


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def mock_config():
    """Create a mock startup configuration."""
    from scripts.startup_manager.config import StartupConfig
    return StartupConfig()


@pytest.fixture
def mock_environment_validator():
    """Create a mock environment validator."""
    from scripts.startup_manager.environment_validator import EnvironmentValidator
    validator = EnvironmentValidator()
    
    # Mock the validation methods to return success by default
    with patch.object(validator, 'validate_all') as mock_validate:
        mock_validate.return_value = Mock(
            is_valid=True,
            issues=[],
            warnings=[],
            system_info={"platform": "test", "python_version": "3.9.0"}
        )
        yield validator


@pytest.fixture
def mock_port_manager():
    """Create a mock port manager."""
    from scripts.startup_manager.port_manager import PortManager, PortAllocation
    manager = PortManager()
    
    # Mock port allocation to return default ports
    with patch.object(manager, 'allocate_ports') as mock_allocate:
        mock_allocate.return_value = PortAllocation(
            backend=8000,
            frontend=3000,
            conflicts_resolved=[],
            alternative_ports_used=False
        )
        yield manager


@pytest.fixture
def mock_process_manager():
    """Create a mock process manager."""
    from scripts.startup_manager.process_manager import ProcessManager, ProcessResult, ProcessInfo
    from scripts.startup_manager.config import StartupConfig
    
    config = StartupConfig()
    manager = ProcessManager(config)
    
    # Mock process startup methods
    with patch.object(manager, 'start_backend') as mock_backend:
        with patch.object(manager, 'start_frontend') as mock_frontend:
            mock_backend.return_value = ProcessResult.success_result(
                ProcessInfo(name="backend", port=8000, pid=1234)
            )
            mock_frontend.return_value = ProcessResult.success_result(
                ProcessInfo(name="frontend", port=3000, pid=5678)
            )
            yield manager


@pytest.fixture
def mock_recovery_engine():
    """Create a mock recovery engine."""
    from scripts.startup_manager.recovery_engine import RecoveryEngine, RecoveryResult
    
    with patch('pathlib.Path.exists', return_value=False):
        engine = RecoveryEngine()
    
    # Mock recovery attempts to succeed by default
    with patch.object(engine, 'attempt_recovery') as mock_recovery:
        mock_recovery.return_value = RecoveryResult(
            success=True,
            action_taken="mock_recovery",
            message="Mock recovery successful",
            retry_recommended=False
        )
        yield engine


@pytest.fixture
def mock_startup_manager(mock_config, mock_environment_validator, mock_port_manager, 
                        mock_process_manager, mock_recovery_engine):
    """Create a fully mocked startup manager."""
    from scripts.startup_manager import StartupManager
    
    manager = StartupManager(mock_config)
    manager.environment_validator = mock_environment_validator
    manager.port_manager = mock_port_manager
    manager.process_manager = mock_process_manager
    manager.recovery_engine = mock_recovery_engine
    
    return manager


@pytest.fixture(autouse=True)
def cleanup_processes():
    """Automatically clean up any processes created during tests."""
    yield
    
    # Clean up any test processes that might be running
    import psutil
    current_process = psutil.Process()
    
    try:
        children = current_process.children(recursive=True)
        for child in children:
            try:
                if "test" in " ".join(child.cmdline()).lower():
                    child.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except:
        pass


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset any singleton instances between tests."""
    yield
    
    # Reset any module-level caches or singletons
    # This ensures tests don't interfere with each other


def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Add custom markers
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "slow: Slow running tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add markers based on test file names
        if "integration" in item.fspath.basename:
            item.add_marker(pytest.mark.integration)
        elif "performance" in item.fspath.basename:
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)
        else:
            item.add_marker(pytest.mark.unit)
        
        # Add slow marker for tests that take a long time
        if any(keyword in item.name.lower() for keyword in ["stress", "scalability", "benchmark"]):
            item.add_marker(pytest.mark.slow)


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


@pytest.fixture
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


# Platform-specific fixtures
@pytest.fixture
def windows_only():
    """Skip test if not running on Windows."""
    if os.name != 'nt':
        pytest.skip("Windows-only test")


@pytest.fixture
def unix_only():
    """Skip test if not running on Unix-like system."""
    if os.name == 'nt':
        pytest.skip("Unix-only test")