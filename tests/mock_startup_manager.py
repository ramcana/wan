"""
Mock StartupManager for testing purposes.
"""

from unittest.mock import Mock
from pathlib import Path
import sys

# Add the startup_manager package to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from startup_manager.config import StartupConfig
from startup_manager.environment_validator import EnvironmentValidator
from startup_manager.port_manager import PortManager
from startup_manager.process_manager import ProcessManager
from startup_manager.recovery_engine import RecoveryEngine


class MockStartupManager:
    """Mock StartupManager for testing."""
    
    def __init__(self, config=None):
        self.config = config or StartupConfig()
        self.environment_validator = EnvironmentValidator()
        self.port_manager = PortManager()
        self.process_manager = ProcessManager(self.config)
        self.recovery_engine = RecoveryEngine()
    
    def start_servers(self):
        """Mock start servers method."""
        return Mock(
            success=True,
            backend_port=8000,
            frontend_port=3000,
            errors=[],
            conflicts_resolved=False,
            startup_duration=1.0
        )
    
    def stop_servers(self):
        """Mock stop servers method."""
        return True


# Use this as StartupManager in tests
StartupManager = MockStartupManager