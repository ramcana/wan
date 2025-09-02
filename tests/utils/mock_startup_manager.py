"""
Mock StartupManager for testing purposes.
"""

from unittest.mock import Mock, MagicMock
from pathlib import Path
from typing import Optional, Dict, Any


class MockStartupManager:
    """Mock implementation of StartupManager for testing."""
    
    def __init__(self, config=None):
        self.config = config or Mock()
        self.environment_validator = Mock()
        self.port_manager = Mock()
        self.process_manager = Mock()
        self.recovery_engine = Mock()
        
        # Set up default mock behaviors
        self._setup_default_mocks()
    
    def _setup_default_mocks(self):
        """Set up default mock behaviors."""
        # Environment validator
        self.environment_validator.validate_all.return_value = Mock(
            is_valid=True,
            issues=[],
            warnings=[],
            system_info={"platform": "test", "python_version": "3.9.0"}
        )
        
        # Port manager
        from unittest.mock import Mock as MockClass
        port_allocation = MockClass()
        port_allocation.backend = 8000
        port_allocation.frontend = 3000
        port_allocation.conflicts_resolved = []
        port_allocation.alternative_ports_used = False
        
        self.port_manager.allocate_ports.return_value = port_allocation
        
        # Process manager
        process_info = MockClass()
        process_info.name = "test"
        process_info.port = 8000
        process_info.pid = 1234
        
        process_result = MockClass()
        process_result.success = True
        process_result.process_info = process_info
        
        self.process_manager.start_backend.return_value = process_result
        self.process_manager.start_frontend.return_value = process_result
        
        # Recovery engine
        recovery_result = MockClass()
        recovery_result.success = True
        recovery_result.action_taken = "mock_recovery"
        recovery_result.message = "Mock recovery successful"
        recovery_result.retry_recommended = False
        
        self.recovery_engine.attempt_recovery.return_value = recovery_result
    
    def start_servers(self):
        """Mock start_servers method."""
        result = Mock()
        result.success = True
        result.backend_info = Mock(port=8000, pid=1234)
        result.frontend_info = Mock(port=3000, pid=5678)
        return result
    
    def stop_servers(self):
        """Mock stop_servers method."""
        result = Mock()
        result.success = True
        return result


# For backward compatibility
StartupManager = MockStartupManager
