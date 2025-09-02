"""
Comprehensive integration tests for the Server Startup Management System.

Tests complete startup workflows, error recovery scenarios, and system integration
with realistic conditions and Docker container simulations.
"""

import pytest
import json
import time
import threading
import tempfile
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import mock StartupManager for testing
from .mock_startup_manager import StartupManager
from scripts.startup_manager.config import StartupConfig
from scripts.startup_manager.port_manager import PortManager, PortStatus
from scripts.startup_manager.process_manager import ProcessManager, ProcessStatus
from scripts.startup_manager.environment_validator import EnvironmentValidator
from scripts.startup_manager.recovery_engine import RecoveryEngine, ErrorType, StartupError


class MockDockerContainer:
    """Mock Docker container for testing startup scenarios."""
    
    def __init__(self, name: str, ports: dict):
        self.name = name
        self.ports = ports  # {internal_port: external_port}
        self.running = False
        self.processes = {}
    
    def start(self):
        """Start the mock container."""
        self.running = True
        # Simulate processes binding to ports
        for internal_port, external_port in self.ports.items():
            self.processes[internal_port] = {
                "pid": 1000 + internal_port,
                "cmd": f"mock-server --port {internal_port}",
                "external_port": external_port
            }
    
    def stop(self):
        """Stop the mock container."""
        self.running = False
        self.processes.clear()
    
    def is_port_occupied(self, port: int) -> bool:
        """Check if a port is occupied in the container."""
        return port in self.processes
    
    def get_process_info(self, port: int):
        """Get process information for a port."""
        return self.processes.get(port)


class TestCompleteStartupWorkflows:
    """Test complete startup workflows from start to finish."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = StartupConfig()
        self.startup_manager = StartupManager(self.config)
        self.mock_containers = []
    
    def teardown_method(self):
        """Clean up test fixtures."""
        for container in self.mock_containers:
            container.stop()
        self.mock_containers.clear()
    
    def create_mock_container(self, name: str, ports: dict) -> MockDockerContainer:
        """Create and start a mock Docker container."""
        container = MockDockerContainer(name, ports)
        container.start()
        self.mock_containers.append(container)
        return container
    
    def test_successful_startup_workflow(self):
        """Test successful startup workflow with no conflicts."""
        with patch.object(self.startup_manager.environment_validator, 'validate_all') as mock_validate:
            mock_validate.return_value = Mock(is_valid=True, issues=[], warnings=[])
            
            with patch.object(self.startup_manager.port_manager, 'allocate_ports') as mock_allocate:
                from scripts.startup_manager.port_manager import PortAllocation
                mock_allocate.return_value = PortAllocation(
                    backend=8000, frontend=3000, 
                    conflicts_resolved=[], alternative_ports_used=False
                )
                
                with patch.object(self.startup_manager.process_manager, 'start_backend') as mock_start_backend:
                    with patch.object(self.startup_manager.process_manager, 'start_frontend') as mock_start_frontend:
                        from scripts.startup_manager.process_manager import ProcessResult, ProcessInfo
                        
                        mock_start_backend.return_value = ProcessResult.success_result(
                            ProcessInfo(name="backend", port=8000, pid=1234)
                        )
                        mock_start_frontend.return_value = ProcessResult.success_result(
                            ProcessInfo(name="frontend", port=3000, pid=5678)
                        )
                        
                        result = self.startup_manager.start_servers()
                        
                        assert result.success is True
                        assert result.backend_port == 8000
                        assert result.frontend_port == 3000
                        assert len(result.errors) == 0
    
    def test_startup_with_port_conflicts_resolution(self):
        """Test startup workflow with port conflicts that get resolved."""
        # Create mock containers occupying default ports
        backend_container = self.create_mock_container("existing-backend", {8000: 8000})
        frontend_container = self.create_mock_container("existing-frontend", {3000: 3000})
        
        with patch.object(self.startup_manager.environment_validator, 'validate_all') as mock_validate:
            mock_validate.return_value = Mock(is_valid=True, issues=[], warnings=[])
            
            # Mock port manager to detect conflicts and resolve them
            with patch.object(self.startup_manager.port_manager, 'check_port_availability') as mock_check:
                def check_side_effect(port):
                    from scripts.startup_manager.port_manager import PortCheckResult
                    if port in [8000, 3000]:
                        return PortCheckResult(port, PortStatus.OCCUPIED)
                    else:
                        return PortCheckResult(port, PortStatus.AVAILABLE)
                
                mock_check.side_effect = check_side_effect
                
                with patch.object(self.startup_manager.port_manager, 'find_available_port') as mock_find:
                    mock_find.side_effect = lambda start_port, **kwargs: start_port + 1
                    
                    with patch.object(self.startup_manager.process_manager, 'start_backend') as mock_start_backend:
                        with patch.object(self.startup_manager.process_manager, 'start_frontend') as mock_start_frontend:
                            from scripts.startup_manager.process_manager import ProcessResult, ProcessInfo
                            
                            mock_start_backend.return_value = ProcessResult.success_result(
                                ProcessInfo(name="backend", port=8001, pid=1234)
                            )
                            mock_start_frontend.return_value = ProcessResult.success_result(
                                ProcessInfo(name="frontend", port=3001, pid=5678)
                            )
                            
                            result = self.startup_manager.start_servers()
                            
                            assert result.success is True
                            assert result.backend_port == 8001  # Alternative port
                            assert result.frontend_port == 3001  # Alternative port
                            assert result.conflicts_resolved is True
    
    def test_startup_with_environment_validation_failures(self):
        """Test startup workflow with environment validation failures."""
        from scripts.startup_manager.environment_validator import ValidationIssue, ValidationStatus
        
        # Mock validation failures
        validation_issues = [
            ValidationIssue(
                component="python",
                issue_type="version_too_old",
                message="Python version too old",
                status=ValidationStatus.FAILED,
                auto_fixable=False
            ),
            ValidationIssue(
                component="nodejs",
                issue_type="not_installed",
                message="Node.js not installed",
                status=ValidationStatus.FAILED,
                auto_fixable=True,
                fix_command="Install Node.js from nodejs.org"
            )
        ]
        
        with patch.object(self.startup_manager.environment_validator, 'validate_all') as mock_validate:
            mock_validate.return_value = Mock(
                is_valid=False, 
                issues=validation_issues, 
                warnings=[]
            )
            
            with patch.object(self.startup_manager.environment_validator, 'auto_fix_issues') as mock_fix:
                mock_fix.return_value = ["Fixed Node.js installation"]
                
                result = self.startup_manager.start_servers()
                
                # Should attempt auto-fix but still fail due to non-fixable issues
                assert result.success is False
                assert len(result.errors) > 0
                assert any("Python version" in error for error in result.errors)
    
    def test_startup_with_process_failures_and_recovery(self):
        """Test startup workflow with process failures and recovery."""
        with patch.object(self.startup_manager.environment_validator, 'validate_all') as mock_validate:
            mock_validate.return_value = Mock(is_valid=True, issues=[], warnings=[])
            
            with patch.object(self.startup_manager.port_manager, 'allocate_ports') as mock_allocate:
                from scripts.startup_manager.port_manager import PortAllocation
                mock_allocate.return_value = PortAllocation(
                    backend=8000, frontend=3000,
                    conflicts_resolved=[], alternative_ports_used=False
                )
                
                # Mock backend startup failure followed by success
                backend_call_count = 0
                def mock_start_backend(*args, **kwargs):
                    nonlocal backend_call_count
                    backend_call_count += 1
                    
                    from scripts.startup_manager.process_manager import ProcessResult, ProcessInfo
                    if backend_call_count == 1:
                        return ProcessResult.failure_result("Process failed to start")
                    else:
                        return ProcessResult.success_result(
                            ProcessInfo(name="backend", port=8000, pid=1234)
                        )
                
                with patch.object(self.startup_manager.process_manager, 'start_backend', side_effect=mock_start_backend):
                    with patch.object(self.startup_manager.process_manager, 'start_frontend') as mock_start_frontend:
                        from scripts.startup_manager.process_manager import ProcessResult, ProcessInfo
                        mock_start_frontend.return_value = ProcessResult.success_result(
                            ProcessInfo(name="frontend", port=3000, pid=5678)
                        )
                        
                        # Mock recovery engine to retry
                        with patch.object(self.startup_manager.recovery_engine, 'attempt_recovery') as mock_recovery:
                            from scripts.startup_manager.recovery_engine import RecoveryResult
                            mock_recovery.return_value = RecoveryResult(
                                success=True,
                                action_taken="retry_startup",
                                message="Retrying process startup",
                                retry_recommended=True
                            )
                            
                            result = self.startup_manager.start_servers()
                            
                            assert result.success is True
                            assert backend_call_count == 2  # Should have retried
    
    def test_startup_performance_benchmarking(self):
        """Test startup performance stays within acceptable limits."""
        with patch.object(self.startup_manager.environment_validator, 'validate_all') as mock_validate:
            mock_validate.return_value = Mock(is_valid=True, issues=[], warnings=[])
            
            with patch.object(self.startup_manager.port_manager, 'allocate_ports') as mock_allocate:
                from scripts.startup_manager.port_manager import PortAllocation
                mock_allocate.return_value = PortAllocation(
                    backend=8000, frontend=3000,
                    conflicts_resolved=[], alternative_ports_used=False
                )
                
                with patch.object(self.startup_manager.process_manager, 'start_backend') as mock_start_backend:
                    with patch.object(self.startup_manager.process_manager, 'start_frontend') as mock_start_frontend:
                        from scripts.startup_manager.process_manager import ProcessResult, ProcessInfo
                        
                        # Add realistic delays
                        def delayed_backend(*args, **kwargs):
                            time.sleep(0.1)  # Simulate startup time
                            return ProcessResult.success_result(
                                ProcessInfo(name="backend", port=8000, pid=1234)
                            )
                        
                        def delayed_frontend(*args, **kwargs):
                            time.sleep(0.15)  # Simulate startup time
                            return ProcessResult.success_result(
                                ProcessInfo(name="frontend", port=3000, pid=5678)
                            )
                        
                        mock_start_backend.side_effect = delayed_backend
                        mock_start_frontend.side_effect = delayed_frontend
                        
                        start_time = time.time()
                        result = self.startup_manager.start_servers()
                        end_time = time.time()
                        
                        startup_duration = end_time - start_time
                        
                        assert result.success is True
                        assert startup_duration < 10.0  # Should complete within 10 seconds
                        assert result.startup_duration is not None


class TestErrorRecoveryScenarios:
    """Test complex error recovery scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = StartupConfig()
        self.startup_manager = StartupManager(self.config)
    
    def test_cascading_failure_recovery(self):
        """Test recovery from cascading failures."""
        # Simulate a scenario where fixing one issue causes another
        errors_sequence = [
            StartupError(ErrorType.PORT_CONFLICT, "Port 8000 in use"),
            StartupError(ErrorType.PERMISSION_DENIED, "Permission denied on port 8001"),
            StartupError(ErrorType.FIREWALL_BLOCKED, "Firewall blocking port 8002"),
        ]
        
        recovery_results = []
        for error in errors_sequence:
            result = self.startup_manager.recovery_engine.attempt_recovery(
                error, {"port": 8000 + len(recovery_results)}
            )
            recovery_results.append(result)
        
        # Should handle the cascade of failures
        assert len(recovery_results) == 3
        # At least some recoveries should succeed or provide fallback
        successful_recoveries = [r for r in recovery_results if r.success]
        assert len(successful_recoveries) > 0
    
    def test_recovery_with_limited_resources(self):
        """Test recovery when system resources are limited."""
        # Simulate low memory/disk space scenario
        error = StartupError(ErrorType.PROCESS_FAILED, "Cannot allocate memory")
        context = {"available_memory": "100MB", "disk_space": "50MB"}
        
        with patch.object(self.startup_manager.recovery_engine, '_check_system_resources') as mock_check:
            mock_check.return_value = {
                "memory_available": False,
                "disk_space_available": False,
                "cpu_available": True
            }
            
            result = self.startup_manager.recovery_engine.attempt_recovery(error, context)
            
            # Should provide appropriate guidance for resource constraints
            assert result is not None
            if not result.success:
                assert "resource" in result.message.lower() or "memory" in result.message.lower()
    
    def test_recovery_with_network_issues(self):
        """Test recovery when network connectivity is limited."""
        error = StartupError(ErrorType.NETWORK_ERROR, "Cannot connect to external services")
        context = {"network_available": False, "offline_mode": True}
        
        result = self.startup_manager.recovery_engine.attempt_recovery(error, context)
        
        # Should suggest offline/local-only configuration
        assert result is not None
        if result.success:
            assert "local" in result.message.lower() or "offline" in result.message.lower()
    
    def test_recovery_learning_and_adaptation(self):
        """Test that recovery engine learns from repeated failures."""
        # Simulate the same error occurring multiple times
        error = StartupError(ErrorType.PORT_CONFLICT, "Port 8000 always in use")
        context = {"port": 8000, "user": "test_user"}
        
        results = []
        for i in range(5):
            result = self.startup_manager.recovery_engine.attempt_recovery(error, context)
            results.append(result)
            
            # Simulate learning by updating success rates
            if hasattr(self.startup_manager.recovery_engine, 'intelligent_handler'):
                pattern = self.startup_manager.recovery_engine.intelligent_handler.detect_failure_pattern(error, context)
                if pattern:
                    # Simulate some actions working better than others
                    pattern.add_recovery_result("find_alternative_port", i % 2 == 0)
                    pattern.add_recovery_result("kill_process_on_port", i % 3 == 0)
        
        # Later attempts should be more successful due to learning
        if len(results) >= 3:
            later_results = results[-2:]
            earlier_results = results[:2]
            
            later_success_rate = sum(1 for r in later_results if r.success) / len(later_results)
            earlier_success_rate = sum(1 for r in earlier_results if r.success) / len(earlier_results)
            
            # Learning should improve success rate (or at least not make it worse)
            assert later_success_rate >= earlier_success_rate


class TestStressTestScenarios:
    """Test system behavior under stress conditions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = StartupConfig()
        self.startup_manager = StartupManager(self.config)
    
    def test_multiple_simultaneous_startup_attempts(self):
        """Test handling multiple simultaneous startup attempts."""
        results = []
        threads = []
        
        def startup_worker():
            try:
                with patch.object(self.startup_manager.environment_validator, 'validate_all') as mock_validate:
                    mock_validate.return_value = Mock(is_valid=True, issues=[], warnings=[])
                    
                    with patch.object(self.startup_manager.port_manager, 'allocate_ports') as mock_allocate:
                        from scripts.startup_manager.port_manager import PortAllocation
                        # Use different ports for each thread to avoid conflicts
                        thread_id = threading.current_thread().ident
                        base_port = 8000 + (thread_id % 1000)
                        
                        mock_allocate.return_value = PortAllocation(
                            backend=base_port, frontend=base_port + 1000,
                            conflicts_resolved=[], alternative_ports_used=False
                        )
                        
                        with patch.object(self.startup_manager.process_manager, 'start_backend') as mock_start_backend:
                            with patch.object(self.startup_manager.process_manager, 'start_frontend') as mock_start_frontend:
                                from scripts.startup_manager.process_manager import ProcessResult, ProcessInfo
                                
                                mock_start_backend.return_value = ProcessResult.success_result(
                                    ProcessInfo(name="backend", port=base_port, pid=1234 + thread_id % 1000)
                                )
                                mock_start_frontend.return_value = ProcessResult.success_result(
                                    ProcessInfo(name="frontend", port=base_port + 1000, pid=5678 + thread_id % 1000)
                                )
                                
                                result = self.startup_manager.start_servers()
                                results.append(result)
            except Exception as e:
                results.append(Mock(success=False, error=str(e)))
        
        # Start multiple threads
        for _ in range(5):
            thread = threading.Thread(target=startup_worker)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=30)
        
        # All attempts should complete (though not necessarily successfully)
        assert len(results) == 5
        
        # At least some should succeed
        successful_results = [r for r in results if r.success]
        assert len(successful_results) > 0
    
    def test_rapid_startup_shutdown_cycles(self):
        """Test rapid startup and shutdown cycles."""
        cycle_results = []
        
        for cycle in range(10):
            # Mock successful startup
            with patch.object(self.startup_manager.environment_validator, 'validate_all') as mock_validate:
                mock_validate.return_value = Mock(is_valid=True, issues=[], warnings=[])
                
                with patch.object(self.startup_manager.port_manager, 'allocate_ports') as mock_allocate:
                    from scripts.startup_manager.port_manager import PortAllocation
                    mock_allocate.return_value = PortAllocation(
                        backend=8000 + cycle, frontend=3000 + cycle,
                        conflicts_resolved=[], alternative_ports_used=False
                    )
                    
                    with patch.object(self.startup_manager.process_manager, 'start_backend') as mock_start_backend:
                        with patch.object(self.startup_manager.process_manager, 'start_frontend') as mock_start_frontend:
                            from scripts.startup_manager.process_manager import ProcessResult, ProcessInfo
                            
                            mock_start_backend.return_value = ProcessResult.success_result(
                                ProcessInfo(name="backend", port=8000 + cycle, pid=1234 + cycle)
                            )
                            mock_start_frontend.return_value = ProcessResult.success_result(
                                ProcessInfo(name="frontend", port=3000 + cycle, pid=5678 + cycle)
                            )
                            
                            # Start servers
                            start_result = self.startup_manager.start_servers()
                            
                            # Immediately stop servers
                            with patch.object(self.startup_manager.process_manager, 'stop_all_processes') as mock_stop:
                                mock_stop.return_value = True
                                stop_result = self.startup_manager.stop_servers()
                            
                            cycle_results.append({
                                "cycle": cycle,
                                "start_success": start_result.success,
                                "stop_success": stop_result
                            })
        
        # All cycles should complete
        assert len(cycle_results) == 10
        
        # Most cycles should succeed
        successful_cycles = [r for r in cycle_results if r["start_success"] and r["stop_success"]]
        assert len(successful_cycles) >= 8  # Allow for some failures under stress
    
    def test_memory_usage_under_stress(self):
        """Test memory usage doesn't grow excessively under stress."""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Perform many operations
        for i in range(100):
            # Create and destroy startup manager instances
            config = StartupConfig()
            manager = StartupManager(config)
            
            # Simulate some operations
            with patch.object(manager.environment_validator, 'validate_all') as mock_validate:
                mock_validate.return_value = Mock(is_valid=True, issues=[], warnings=[])
                manager.environment_validator.validate_all()
            
            # Clean up
            del manager
            
            # Force garbage collection every 10 iterations
            if i % 10 == 0:
                gc.collect()
        
        final_memory = process.memory_info().rss
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be reasonable (less than 50MB)
        assert memory_growth < 50 * 1024 * 1024  # 50MB
    
    def test_file_handle_management_under_stress(self):
        """Test that file handles are properly managed under stress."""
        import psutil
        
        process = psutil.Process()
        initial_handles = len(process.open_files())
        
        # Perform many file operations
        for i in range(50):
            config = StartupConfig()
            manager = StartupManager(config)
            
            # Simulate configuration file operations
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump({"test": f"config_{i}"}, f)
                config_path = Path(f.name)
            
            try:
                # Simulate reading/writing config files
                with patch.object(manager.port_manager, '_update_backend_config'):
                    manager.port_manager._update_backend_config(config_path, 8000 + i)
            finally:
                config_path.unlink(missing_ok=True)
            
            del manager
        
        final_handles = len(process.open_files())
        handle_growth = final_handles - initial_handles
        
        # File handle growth should be minimal
        assert handle_growth < 10  # Allow for some growth but not excessive


class TestEndToEndIntegration:
    """Test complete end-to-end integration scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = StartupConfig()
        self.startup_manager = StartupManager(self.config)
    
    def test_complete_development_workflow(self):
        """Test complete development workflow from first run to daily use."""
        # Phase 1: First-time setup
        with patch.object(self.startup_manager.environment_validator, 'validate_all') as mock_validate:
            # Simulate missing dependencies on first run
            from scripts.startup_manager.environment_validator import ValidationIssue, ValidationStatus
            
            first_run_issues = [
                ValidationIssue(
                    component="backend",
                    issue_type="missing_dependencies",
                    message="Missing backend dependencies",
                    status=ValidationStatus.FAILED,
                    auto_fixable=True,
                    fix_command="pip install -r requirements.txt"
                )
            ]
            
            mock_validate.return_value = Mock(
                is_valid=False,
                issues=first_run_issues,
                warnings=[]
            )
            
            with patch.object(self.startup_manager.environment_validator, 'auto_fix_issues') as mock_fix:
                mock_fix.return_value = ["Fixed backend dependencies"]
                
                # After fix, validation should pass
                mock_validate.return_value = Mock(is_valid=True, issues=[], warnings=[])
                
                # Phase 2: Successful startup after fixes
                with patch.object(self.startup_manager.port_manager, 'allocate_ports') as mock_allocate:
                    from scripts.startup_manager.port_manager import PortAllocation
                    mock_allocate.return_value = PortAllocation(
                        backend=8000, frontend=3000,
                        conflicts_resolved=[], alternative_ports_used=False
                    )
                    
                    with patch.object(self.startup_manager.process_manager, 'start_backend') as mock_start_backend:
                        with patch.object(self.startup_manager.process_manager, 'start_frontend') as mock_start_frontend:
                            from scripts.startup_manager.process_manager import ProcessResult, ProcessInfo
                            
                            mock_start_backend.return_value = ProcessResult.success_result(
                                ProcessInfo(name="backend", port=8000, pid=1234)
                            )
                            mock_start_frontend.return_value = ProcessResult.success_result(
                                ProcessInfo(name="frontend", port=3000, pid=5678)
                            )
                            
                            result = self.startup_manager.start_servers()
                            
                            assert result.success is True
                            assert len(result.fixes_applied) > 0  # Should have applied fixes
    
    def test_production_deployment_scenario(self):
        """Test production deployment scenario with optimizations."""
        # Configure for production mode
        self.config.environment = "production"
        self.config.backend.reload = False
        self.config.frontend.hot_reload = False
        
        with patch.object(self.startup_manager.environment_validator, 'validate_all') as mock_validate:
            mock_validate.return_value = Mock(is_valid=True, issues=[], warnings=[])
            
            with patch.object(self.startup_manager.port_manager, 'allocate_ports') as mock_allocate:
                from scripts.startup_manager.port_manager import PortAllocation
                mock_allocate.return_value = PortAllocation(
                    backend=8000, frontend=3000,
                    conflicts_resolved=[], alternative_ports_used=False
                )
                
                with patch.object(self.startup_manager.process_manager, 'start_backend') as mock_start_backend:
                    with patch.object(self.startup_manager.process_manager, 'start_frontend') as mock_start_frontend:
                        from scripts.startup_manager.process_manager import ProcessResult, ProcessInfo
                        
                        # Production processes should have different characteristics
                        mock_start_backend.return_value = ProcessResult.success_result(
                            ProcessInfo(name="backend", port=8000, pid=1234, restart_count=0)
                        )
                        mock_start_frontend.return_value = ProcessResult.success_result(
                            ProcessInfo(name="frontend", port=3000, pid=5678, restart_count=0)
                        )
                        
                        result = self.startup_manager.start_servers()
                        
                        assert result.success is True
                        assert result.production_mode is True
    
    def test_debugging_workflow(self):
        """Test debugging workflow with detailed logging and diagnostics."""
        # Enable debug mode
        self.config.verbose_logging = True
        self.config.debug_mode = True
        
        with patch.object(self.startup_manager.logger, 'set_debug_mode') as mock_debug:
            with patch.object(self.startup_manager.environment_validator, 'validate_all') as mock_validate:
                mock_validate.return_value = Mock(is_valid=True, issues=[], warnings=[])
                
                result = self.startup_manager.start_servers()
                
                # Debug mode should be enabled
                mock_debug.assert_called_once_with(True)
                
                # Should collect diagnostic information
                assert hasattr(result, 'diagnostic_info')
    
    def test_configuration_migration_scenario(self):
        """Test configuration migration between versions."""
        # Simulate old configuration format
        old_config = {
            "server": {"port": 8000},
            "client": {"port": 3000}  # Old naming
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(old_config, f)
            config_path = Path(f.name)
        
        try:
            # Should migrate configuration automatically
            with patch.object(self.startup_manager, '_migrate_configuration') as mock_migrate:
                mock_migrate.return_value = {
                    "backend": {"port": 8000},
                    "frontend": {"port": 3000}  # New naming
                }
                
                migrated_config = self.startup_manager._load_and_migrate_config(config_path)
                
                assert "backend" in migrated_config
                assert "frontend" in migrated_config
                assert "server" not in migrated_config
                assert "client" not in migrated_config
                
        finally:
            config_path.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])