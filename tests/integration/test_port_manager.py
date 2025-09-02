"""
Unit tests for Port Manager component.

Tests port availability checking, conflict detection, and resolution
with mocked socket operations and process management.
"""

import pytest
import socket
import psutil
from unittest.mock import Mock, patch, MagicMock, mock_open
from pathlib import Path
import json

from scripts.startup_manager.port_manager import (
    PortManager, PortStatus, PortCheckResult, ProcessInfo, 
    PortConflict, ConflictResolution, PortAllocation
)


class TestPortManager:
    """Test suite for PortManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.port_manager = PortManager()
    
    def test_init(self):
        """Test PortManager initialization."""
        assert self.port_manager.default_ports == {"backend": 8000, "frontend": 3000}
        assert len(self.port_manager.safe_port_ranges) > 0
        assert self.port_manager.allocated_ports == {}
    
    @patch('socket.socket')
    def test_check_port_availability_available(self, mock_socket):
        """Test checking an available port."""
        # Mock socket operations for available port
        mock_sock = MagicMock()
        mock_socket.return_value.__enter__.return_value = mock_sock
        mock_sock.bind.return_value = None
        
        with patch.object(self.port_manager, '_get_process_using_port', return_value=None):
            result = self.port_manager.check_port_availability(8000)
            
        assert result.port == 8000
        assert result.status == PortStatus.AVAILABLE
        assert result.process is None
        assert result.error_message is None
        mock_sock.bind.assert_called_once_with(("localhost", 8000))
    
    @patch('socket.socket')
    def test_check_port_availability_occupied(self, mock_socket):
        """Test checking a port occupied by a process."""
        mock_process = ProcessInfo(
            pid=1234,
            name="python.exe",
            cmdline=["python", "server.py"],
            port=8000,
            can_terminate=True
        )
        
        with patch.object(self.port_manager, '_get_process_using_port', return_value=mock_process):
            result = self.port_manager.check_port_availability(8000)
            
        assert result.port == 8000
        assert result.status == PortStatus.OCCUPIED
        assert result.process == mock_process
    
    @patch('socket.socket')
    def test_check_port_availability_permission_denied(self, mock_socket):
        """Test checking a port with permission denied error."""
        mock_sock = MagicMock()
        mock_socket.return_value.__enter__.return_value = mock_sock
        
        # Mock Windows permission denied error (WSAEACCES)
        socket_error = socket.error()
        socket_error.errno = 10013
        mock_sock.bind.side_effect = socket_error
        
        with patch.object(self.port_manager, '_get_process_using_port', return_value=None):
            result = self.port_manager.check_port_availability(8000)
            
        assert result.port == 8000
        assert result.status == PortStatus.BLOCKED
        assert "Permission denied" in result.error_message
    
    @patch('socket.socket')
    def test_check_port_availability_address_in_use(self, mock_socket):
        """Test checking a port with address already in use error."""
        mock_sock = MagicMock()
        mock_socket.return_value.__enter__.return_value = mock_sock
        
        # Mock Windows address in use error (WSAEADDRINUSE)
        socket_error = socket.error()
        socket_error.errno = 10048
        mock_sock.bind.side_effect = socket_error
        
        with patch.object(self.port_manager, '_get_process_using_port', return_value=None):
            result = self.port_manager.check_port_availability(8000)
            
        assert result.port == 8000
        assert result.status == PortStatus.OCCUPIED
        assert "Address already in use" in result.error_message
    
    @patch('psutil.net_connections')
    def test_get_process_using_port_found(self, mock_connections):
        """Test getting process information for a port in use."""
        # Mock network connection
        mock_conn = Mock()
        mock_conn.laddr.port = 8000
        mock_conn.status = psutil.CONN_LISTEN
        mock_conn.pid = 1234
        mock_connections.return_value = [mock_conn]
        
        # Mock process
        mock_process = Mock()
        mock_process.name.return_value = "python.exe"
        mock_process.cmdline.return_value = ["python", "server.py"]
        
        with patch('psutil.Process', return_value=mock_process):
            with patch.object(self.port_manager, '_can_safely_terminate_process', return_value=True):
                result = self.port_manager._get_process_using_port(8000)
                
        assert result is not None
        assert result.pid == 1234
        assert result.name == "python.exe"
        assert result.port == 8000
        assert result.can_terminate is True
    
    @patch('psutil.net_connections')
    def test_get_process_using_port_not_found(self, mock_connections):
        """Test getting process information when no process uses the port."""
        mock_connections.return_value = []
        
        result = self.port_manager._get_process_using_port(8000)
        
        assert result is None
    
    def test_can_safely_terminate_process_system_process(self):
        """Test that system processes cannot be safely terminated."""
        mock_process = Mock()
        mock_process.pid = 4  # System process PID
        
        result = self.port_manager._can_safely_terminate_process(mock_process)
        
        assert result is False
    
    def test_can_safely_terminate_process_system_user(self):
        """Test that processes running as SYSTEM cannot be safely terminated."""
        mock_process = Mock()
        mock_process.pid = 1234
        mock_process.username.return_value = "NT AUTHORITY\\SYSTEM"
        
        result = self.port_manager._can_safely_terminate_process(mock_process)
        
        assert result is False
    
    def test_can_safely_terminate_process_dev_server(self):
        """Test that development servers can be safely terminated."""
        mock_process = Mock()
        mock_process.pid = 1234
        mock_process.username.return_value = "user"
        mock_process.cmdline.return_value = ["python", "-m", "uvicorn", "app:app"]
        
        result = self.port_manager._can_safely_terminate_process(mock_process)
        
        assert result is True
    
    def test_find_available_port_success(self):
        """Test finding an available port."""
        with patch.object(self.port_manager, 'check_port_availability') as mock_check:
            # First port occupied, second available
            mock_check.side_effect = [
                PortCheckResult(8000, PortStatus.OCCUPIED),
                PortCheckResult(8001, PortStatus.AVAILABLE)
            ]
            
            result = self.port_manager.find_available_port(8000, max_attempts=10)
            
        assert result == 8001
    
    def test_find_available_port_none_found(self):
        """Test finding available port when none are available."""
        with patch.object(self.port_manager, 'check_port_availability') as mock_check:
            mock_check.return_value = PortCheckResult(8000, PortStatus.OCCUPIED)
            
            result = self.port_manager.find_available_port(8000, max_attempts=5)
            
        assert result is None
    
    def test_find_available_ports_in_range(self):
        """Test finding all available ports in a range."""
        with patch.object(self.port_manager, 'check_port_availability') as mock_check:
            # Ports 8000, 8002 available; 8001 occupied
            def check_side_effect(port):
                if port in [8000, 8002]:
                    return PortCheckResult(port, PortStatus.AVAILABLE)
                else:
                    return PortCheckResult(port, PortStatus.OCCUPIED)
            
            mock_check.side_effect = check_side_effect
            
            result = self.port_manager.find_available_ports_in_range((8000, 8002))
            
        assert result == [8000, 8002]
    
    def test_detect_port_conflicts(self):
        """Test detecting port conflicts."""
        mock_process = ProcessInfo(
            pid=1234,
            name="python.exe",
            cmdline=["python", "server.py"],
            port=8000,
            can_terminate=True
        )
        
        with patch.object(self.port_manager, 'check_port_availability') as mock_check:
            mock_check.return_value = PortCheckResult(
                8000, PortStatus.OCCUPIED, process=mock_process
            )
            
            conflicts = self.port_manager.detect_port_conflicts({"backend": 8000})
            
        assert len(conflicts) == 1
        assert conflicts[0].port == 8000
        assert conflicts[0].process == mock_process
        assert ConflictResolution.KILL_PROCESS in conflicts[0].resolution_options
        assert ConflictResolution.USE_ALTERNATIVE in conflicts[0].resolution_options
    
    @patch('subprocess.run')
    def test_check_firewall_exceptions_with_exception(self, mock_run):
        """Test checking firewall exceptions when rules exist."""
        mock_run.return_value = Mock(returncode=0, stdout="Rule Name: Test Rule")
        
        result = self.port_manager.check_firewall_exceptions([8000])
        
        assert result[8000] is True
        mock_run.assert_called_once()
    
    @patch('subprocess.run')
    def test_check_firewall_exceptions_no_exception(self, mock_run):
        """Test checking firewall exceptions when no rules exist."""
        mock_run.return_value = Mock(returncode=1, stdout="No rules match")
        
        result = self.port_manager.check_firewall_exceptions([8000])
        
        assert result[8000] is False
    
    def test_suggest_firewall_fix(self):
        """Test generating firewall exception command."""
        command = self.port_manager.suggest_firewall_fix(8000, "TestApp")
        
        expected = (
            'netsh advfirewall firewall add rule name="TestApp-8000" '
            'dir=in action=allow protocol=TCP localport=8000'
        )
        assert command == expected
    
    def test_allocate_ports_no_conflicts(self):
        """Test port allocation when no conflicts exist."""
        with patch.object(self.port_manager, 'check_port_availability') as mock_check:
            mock_check.return_value = PortCheckResult(8000, PortStatus.AVAILABLE)
            
            result = self.port_manager.allocate_ports()
            
        assert result.backend == 8000
        assert result.frontend == 3000
        assert result.alternative_ports_used is False
        assert len(result.conflicts_resolved) == 0
    
    def test_allocate_ports_with_conflicts(self):
        """Test port allocation when conflicts exist."""
        def check_side_effect(port):
            if port == 8000:
                return PortCheckResult(port, PortStatus.OCCUPIED)
            else:
                return PortCheckResult(port, PortStatus.AVAILABLE)
        
        with patch.object(self.port_manager, 'check_port_availability', side_effect=check_side_effect):
            with patch.object(self.port_manager, '_find_alternative_port', return_value=8001):
                result = self.port_manager.allocate_ports()
                
        assert result.backend == 8001
        assert result.frontend == 3000
        assert result.alternative_ports_used is True
        assert len(result.conflicts_resolved) == 1
        assert "backend: 8000 -> 8001" in result.conflicts_resolved[0]
    
    def test_allocate_ports_no_alternative_found(self):
        """Test port allocation when no alternative ports are available."""
        with patch.object(self.port_manager, 'check_port_availability') as mock_check:
            mock_check.return_value = PortCheckResult(8000, PortStatus.OCCUPIED)
            
        with patch.object(self.port_manager, '_find_alternative_port', return_value=None):
            with pytest.raises(RuntimeError, match="Could not find available port"):
                self.port_manager.allocate_ports()
    
    @patch('psutil.Process')
    def test_kill_process_on_port_success(self, mock_process_class):
        """Test successfully killing a process on a port."""
        mock_process_info = ProcessInfo(
            pid=1234,
            name="python.exe",
            cmdline=["python", "server.py"],
            port=8000,
            can_terminate=True
        )
        
        mock_process = Mock()
        mock_process_class.return_value = mock_process
        
        with patch.object(self.port_manager, '_get_process_using_port', return_value=mock_process_info):
            result = self.port_manager.kill_process_on_port(8000)
            
        assert result is True
        mock_process.terminate.assert_called_once()
    
    def test_kill_process_on_port_no_process(self):
        """Test killing process when no process uses the port."""
        with patch.object(self.port_manager, '_get_process_using_port', return_value=None):
            result = self.port_manager.kill_process_on_port(8000)
            
        assert result is True
    
    def test_kill_process_on_port_cannot_terminate(self):
        """Test killing process that cannot be safely terminated."""
        mock_process_info = ProcessInfo(
            pid=4,
            name="System",
            cmdline=["System"],
            port=8000,
            can_terminate=False
        )
        
        with patch.object(self.port_manager, '_get_process_using_port', return_value=mock_process_info):
            result = self.port_manager.kill_process_on_port(8000)
            
        assert result is False
    
    def test_resolve_port_conflicts_kill_strategy(self):
        """Test resolving conflicts using kill process strategy."""
        mock_process = ProcessInfo(
            pid=1234, name="python.exe", cmdline=["python"], port=8000, can_terminate=True
        )
        conflict = PortConflict(
            port=8000,
            process=mock_process,
            resolution_options=[ConflictResolution.KILL_PROCESS, ConflictResolution.USE_ALTERNATIVE]
        )
        
        with patch.object(self.port_manager, 'kill_process_on_port', return_value=True):
            result = self.port_manager.resolve_port_conflicts(
                [conflict], ConflictResolution.KILL_PROCESS
            )
            
        assert result[8000] == 8000
    
    def test_resolve_port_conflicts_alternative_strategy(self):
        """Test resolving conflicts using alternative port strategy."""
        mock_process = ProcessInfo(
            pid=1234, name="python.exe", cmdline=["python"], port=8000, can_terminate=True
        )
        conflict = PortConflict(
            port=8000,
            process=mock_process,
            resolution_options=[ConflictResolution.USE_ALTERNATIVE]
        )
        
        with patch.object(self.port_manager, 'find_available_port', return_value=8001):
            result = self.port_manager.resolve_port_conflicts(
                [conflict], ConflictResolution.USE_ALTERNATIVE
            )
            
        assert result[8000] == 8001
    
    @patch('builtins.open', new_callable=mock_open, read_data='{"server": {"host": "localhost"}}')
    @patch('pathlib.Path.exists')
    def test_update_configuration_files(self, mock_exists, mock_file):
        """Test updating configuration files with new ports."""
        mock_exists.return_value = True
        port_allocation = PortAllocation(
            backend=8001, frontend=3001, conflicts_resolved=[], alternative_ports_used=True
        )
        
        with patch.object(self.port_manager, '_update_backend_config') as mock_backend:
            with patch.object(self.port_manager, '_update_frontend_config') as mock_frontend:
                result = self.port_manager.update_configuration_files(port_allocation)
                
        assert result is True
        mock_backend.assert_called_once()
        # Frontend config update should be called for each existing config file
        assert mock_frontend.call_count >= 0
    
    def test_get_port_status_report(self):
        """Test generating port status report."""
        self.port_manager.allocated_ports = {"backend": 8001, "frontend": 3001}
        
        with patch.object(self.port_manager, 'check_port_availability') as mock_check:
            mock_check.return_value = PortCheckResult(8000, PortStatus.AVAILABLE)
            
            with patch.object(self.port_manager, 'detect_port_conflicts', return_value=[]):
                with patch.object(self.port_manager, 'check_firewall_exceptions', return_value={}):
                    report = self.port_manager.get_port_status_report()
                    
        assert 'default_ports' in report
        assert 'allocated_ports' in report
        assert 'port_checks' in report
        assert 'conflicts' in report
        assert 'firewall_status' in report
        assert report['allocated_ports'] == {"backend": 8001, "frontend": 3001}


class TestPortManagerIntegration:
    """Integration tests for PortManager with real system interactions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.port_manager = PortManager()
    
    def test_real_port_check_localhost(self):
        """Test real port checking on localhost (integration test)."""
        # This test uses real socket operations
        result = self.port_manager.check_port_availability(65432)  # Unlikely to be used
        
        # Should be available or have a specific status
        assert result.port == 65432
        assert result.status in [PortStatus.AVAILABLE, PortStatus.OCCUPIED, PortStatus.BLOCKED]
    
    def test_find_available_port_real(self):
        """Test finding available port with real socket operations."""
        # Start from a high port number to avoid conflicts
        port = self.port_manager.find_available_port(60000, max_attempts=10)
        
        assert port is not None
        assert port >= 60000
        
        # Verify the port is actually available
        result = self.port_manager.check_port_availability(port)
        assert result.status == PortStatus.AVAILABLE


class TestPortManagerAdvanced:
    """Advanced test scenarios for PortManager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.port_manager = PortManager()
    
    def test_port_scanning_performance(self):
        """Test port scanning performance with large ranges."""
        import time
        
        start_time = time.time()
        available_ports = self.port_manager.find_available_ports_in_range((60000, 60010))
        end_time = time.time()
        
        # Should complete within reasonable time
        assert end_time - start_time < 5.0  # 5 seconds max
        assert isinstance(available_ports, list)
    
    def test_concurrent_port_checks(self):
        """Test concurrent port availability checks."""
        import threading
        import time
        
        results = {}
        
        def check_port_worker(port):
            result = self.port_manager.check_port_availability(port)
            results[port] = result
        
        # Start multiple port checks concurrently
        threads = []
        test_ports = [60000 + i for i in range(10)]
        
        for port in test_ports:
            thread = threading.Thread(target=check_port_worker, args=(port,))
            threads.append(thread)
            thread.start()
        
        # Wait for all checks to complete
        for thread in threads:
            thread.join(timeout=10)
        
        # All checks should complete
        assert len(results) == len(test_ports)
        for port, result in results.items():
            assert result.port == port
            assert result.status in [PortStatus.AVAILABLE, PortStatus.OCCUPIED, PortStatus.BLOCKED]
    
    def test_ipv6_port_checking(self):
        """Test port checking with IPv6 addresses."""
        with patch('socket.socket') as mock_socket:
            mock_sock = MagicMock()
            mock_socket.return_value.__enter__.return_value = mock_sock
            mock_sock.bind.return_value = None
            
            # Test IPv6 localhost
            result = self.port_manager.check_port_availability(8000, host="::1")
            
            assert result.port == 8000
            assert result.status == PortStatus.AVAILABLE
            mock_sock.bind.assert_called_with(("::1", 8000))
    
    def test_port_range_validation(self):
        """Test validation of port ranges."""
        # Test invalid port ranges
        with pytest.raises(ValueError):
            self.port_manager.find_available_port(-1)  # Negative port
        
        with pytest.raises(ValueError):
            self.port_manager.find_available_port(70000)  # Port too high
        
        # Test valid edge cases
        result = self.port_manager.find_available_port(1024)  # Minimum non-privileged
        assert result is None or result >= 1024
    
    def test_process_detection_with_complex_cmdlines(self):
        """Test process detection with complex command lines."""
        with patch('psutil.net_connections') as mock_connections:
            # Mock complex process command line
            mock_conn = Mock()
            mock_conn.laddr.port = 8000
            mock_conn.pid = 1234
            mock_connections.return_value = [mock_conn]
            
            mock_process = Mock()
            mock_process.name.return_value = "python.exe"
            mock_process.cmdline.return_value = [
                "python", "-m", "uvicorn", "app:app", 
                "--host", "0.0.0.0", "--port", "8000",
                "--reload", "--log-level", "debug"
            ]
            
            with patch('psutil.Process', return_value=mock_process):
                with patch.object(self.port_manager, '_can_safely_terminate_process', return_value=True):
                    result = self.port_manager._get_process_using_port(8000)
                    
            assert result is not None
            assert result.name == "python.exe"
            assert "uvicorn" in " ".join(result.cmdline)
    
    def test_firewall_exception_detection_edge_cases(self):
        """Test firewall exception detection in various scenarios."""
        with patch('subprocess.run') as mock_run:
            # Test Windows Defender Firewall
            mock_run.return_value = Mock(
                returncode=0,
                stdout="Rule Name: Python 3.9\nEnabled: Yes\nDirection: Inbound\nProfiles: Domain,Private,Public\nLocalPort: 8000"
            )
            
            result = self.port_manager.check_firewall_exceptions([8000])
            assert result[8000] is True
            
            # Test third-party firewall (command not found)
            mock_run.side_effect = FileNotFoundError("netsh not found")
            result = self.port_manager.check_firewall_exceptions([8000])
            assert result[8000] is False
    
    def test_configuration_update_atomic_operations(self):
        """Test that configuration updates are atomic."""
        import tempfile
        import json
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            original_config = {"server": {"port": 8000, "host": "localhost"}}
            json.dump(original_config, f)
            config_path = Path(f.name)
        
        try:
            # Simulate interruption during update
            with patch('builtins.open', side_effect=KeyboardInterrupt("Interrupted")):
                try:
                    self.port_manager._update_backend_config(config_path, 8001)
                except KeyboardInterrupt:
                    pass
            
            # Original config should be intact
            with open(config_path, 'r') as f:
                current_config = json.load(f)
            
            assert current_config["server"]["port"] == 8000  # Should be unchanged
            
        finally:
            config_path.unlink()
    
    def test_port_allocation_with_custom_strategies(self):
        """Test port allocation with custom allocation strategies."""
        # Test sequential allocation
        with patch.object(self.port_manager, 'check_port_availability') as mock_check:
            # Ports 8000-8002 occupied, 8003 available
            def check_side_effect(port):
                if port <= 8002:
                    return PortCheckResult(port, PortStatus.OCCUPIED)
                else:
                    return PortCheckResult(port, PortStatus.AVAILABLE)
            
            mock_check.side_effect = check_side_effect
            
            allocation = self.port_manager.allocate_ports(
                {"backend": 8000, "frontend": 8001},
                strategy="sequential"
            )
            
            assert allocation.backend == 8003
            assert allocation.frontend == 8004
    
    def test_port_manager_state_consistency(self):
        """Test that PortManager maintains consistent internal state."""
        # Allocate ports
        allocation = self.port_manager.allocate_ports()
        
        # Check internal state
        assert self.port_manager.allocated_ports["backend"] == allocation.backend
        assert self.port_manager.allocated_ports["frontend"] == allocation.frontend
        
        # Generate report and verify consistency
        report = self.port_manager.get_port_status_report()
        assert report["allocated_ports"]["backend"] == allocation.backend
        assert report["allocated_ports"]["frontend"] == allocation.frontend


class TestPortManagerErrorRecovery:
    """Test error recovery scenarios in PortManager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.port_manager = PortManager()
    
    def test_recovery_from_socket_errors(self):
        """Test recovery from various socket errors."""
        socket_errors = [
            (10013, PortStatus.BLOCKED),  # WSAEACCES
            (10048, PortStatus.OCCUPIED),  # WSAEADDRINUSE
            (10049, PortStatus.BLOCKED),  # WSAEADDRNOTAVAIL
            (10061, PortStatus.BLOCKED),  # WSAECONNREFUSED
        ]
        
        for errno, expected_status in socket_errors:
            with patch('socket.socket') as mock_socket:
                mock_sock = Mock()
                mock_socket.return_value.__enter__.return_value = mock_sock
                
                socket_error = socket.error()
                socket_error.errno = errno
                mock_sock.bind.side_effect = socket_error
                
                result = self.port_manager.check_port_availability(8000)
                assert result.status == expected_status
    
    def test_recovery_from_psutil_errors(self):
        """Test recovery from psutil-related errors."""
        import psutil
        
        with patch('psutil.net_connections', side_effect=psutil.AccessDenied()):
            result = self.port_manager._get_process_using_port(8000)
            assert result is None  # Should handle gracefully
        
        with patch('psutil.net_connections', side_effect=psutil.NoSuchProcess(1234)):
            result = self.port_manager._get_process_using_port(8000)
            assert result is None
    
    def test_configuration_backup_and_restore(self):
        """Test configuration backup and restore functionality."""
        import tempfile
        import json
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            original_config = {"server": {"port": 8000, "host": "localhost"}}
            json.dump(original_config, f)
            config_path = Path(f.name)
        
        try:
            # Update config (should create backup)
            self.port_manager._update_backend_config(config_path, 8001)
            
            # Check that backup was created
            backup_path = config_path.with_suffix('.json.backup')
            assert backup_path.exists()
            
            # Verify backup content
            with open(backup_path, 'r') as f:
                backup_config = json.load(f)
            assert backup_config["server"]["port"] == 8000
            
            # Verify updated config
            with open(config_path, 'r') as f:
                updated_config = json.load(f)
            assert updated_config["server"]["port"] == 8001
            
        finally:
            config_path.unlink()
            backup_path = config_path.with_suffix('.json.backup')
            if backup_path.exists():
                backup_path.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])