"""
Unit tests for the Process Manager component.
Tests process startup, health monitoring, and lifecycle management.
"""

import os
import sys
import time
import pytest
import subprocess
from unittest.mock import Mock, patch, MagicMock, mock_open
from pathlib import Path
from datetime import datetime

# Add the project root to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.startup_manager.process_manager import (
    ProcessManager, ProcessInfo, ProcessResult, ProcessStatus, HealthMonitor
)
from scripts.startup_manager.config import StartupConfig, BackendConfig, FrontendConfig


class TestProcessInfo:
    """Test ProcessInfo dataclass."""
    
    def test_process_info_creation(self):
        """Test ProcessInfo creation with default values."""
        info = ProcessInfo(name="test")
        assert info.name == "test"
        assert info.pid is None
        assert info.port == 0
        assert info.status == ProcessStatus.STARTING
        assert info.start_time is None
        assert info.health_check_url == ""
        assert info.log_file == ""
        assert info.process is None
        assert info.working_directory == ""
        assert info.command == []
        assert info.environment == {}
    
    def test_process_info_with_values(self):
        """Test ProcessInfo creation with specific values."""
        start_time = datetime.now()
        info = ProcessInfo(
            name="backend",
            pid=1234,
            port=8000,
            status=ProcessStatus.RUNNING,
            start_time=start_time,
            health_check_url="http://localhost:8000/health",
            log_file="/logs/backend.log",
            working_directory="/app/backend",
            command=["python", "main.py"],
            environment={"PORT": "8000"}
        )
        
        assert info.name == "backend"
        assert info.pid == 1234
        assert info.port == 8000
        assert info.status == ProcessStatus.RUNNING
        assert info.start_time == start_time
        assert info.health_check_url == "http://localhost:8000/health"
        assert info.log_file == "/logs/backend.log"
        assert info.working_directory == "/app/backend"
        assert info.command == ["python", "main.py"]
        assert info.environment == {"PORT": "8000"}


class TestProcessResult:
    """Test ProcessResult dataclass."""
    
    def test_success_result(self):
        """Test creating successful ProcessResult."""
        process_info = ProcessInfo(name="test")
        result = ProcessResult.success_result(process_info)
        
        assert result.success is True
        assert result.process_info == process_info
        assert result.error_message == ""
        assert result.details == {}
    
    def test_failure_result(self):
        """Test creating failed ProcessResult."""
        error_msg = "Process failed to start"
        details = {"code": 1, "reason": "port_conflict"}
        result = ProcessResult.failure_result(error_msg, details)
        
        assert result.success is False
        assert result.process_info is None
        assert result.error_message == error_msg
        assert result.details == details
    
    def test_failure_result_no_details(self):
        """Test creating failed ProcessResult without details."""
        error_msg = "Process failed"
        result = ProcessResult.failure_result(error_msg)
        
        assert result.success is False
        assert result.error_message == error_msg
        assert result.details == {}


class TestHealthMonitor:
    """Test HealthMonitor functionality."""
    
    def test_health_monitor_creation(self):
        """Test HealthMonitor creation."""
        monitor = HealthMonitor(check_interval=2.0)
        assert monitor.check_interval == 2.0
        assert monitor.monitoring is False
        assert monitor.monitor_thread is None
        assert monitor.processes == {}
    
    def test_add_remove_process(self):
        """Test adding and removing processes from monitoring."""
        monitor = HealthMonitor()
        process_info = ProcessInfo(name="test")
        
        monitor.add_process(process_info)
        assert "test" in monitor.processes
        assert monitor.processes["test"] == process_info
        
        monitor.remove_process("test")
        assert "test" not in monitor.processes
    
    def test_start_stop_monitoring(self):
        """Test starting and stopping monitoring."""
        monitor = HealthMonitor(check_interval=0.1)
        
        # Start monitoring
        monitor.start_monitoring()
        assert monitor.monitoring is True
        assert monitor.monitor_thread is not None
        
        # Stop monitoring
        monitor.stop_monitoring()
        assert monitor.monitoring is False
    
    @patch('requests.get')
    def test_check_process_health_success(self, mock_get):
        """Test successful health check."""
        monitor = HealthMonitor()
        mock_process = Mock()
        mock_process.poll.return_value = None  # Process still running
        
        process_info = ProcessInfo(
            name="test",
            process=mock_process,
            health_check_url="http://localhost:8000/health"
        )
        
        # Mock successful HTTP response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        monitor._check_process_health(process_info)
        assert process_info.status == ProcessStatus.RUNNING
    
    @patch('requests.get')
    def test_check_process_health_http_failure(self, mock_get):
        """Test health check with HTTP failure."""
        monitor = HealthMonitor()
        mock_process = Mock()
        mock_process.poll.return_value = None  # Process still running
        
        process_info = ProcessInfo(
            name="test",
            process=mock_process,
            health_check_url="http://localhost:8000/health",
            status=ProcessStatus.RUNNING
        )
        
        # Mock failed HTTP response
        mock_response = Mock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response
        
        monitor._check_process_health(process_info)
        assert process_info.status == ProcessStatus.FAILED
    
    def test_check_process_health_process_dead(self):
        """Test health check when process is dead."""
        monitor = HealthMonitor()
        mock_process = Mock()
        mock_process.poll.return_value = 1  # Process exited with code 1
        
        process_info = ProcessInfo(
            name="test",
            process=mock_process
        )
        
        monitor._check_process_health(process_info)
        assert process_info.status == ProcessStatus.FAILED


class TestProcessManager:
    """Test ProcessManager functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = StartupConfig()
        self.process_manager = ProcessManager(self.config)
    
    def teardown_method(self):
        """Clean up after tests."""
        self.process_manager.cleanup()
    
    def test_process_manager_creation(self):
        """Test ProcessManager creation."""
        assert self.process_manager.config == self.config
        assert self.process_manager.processes == {}
        assert isinstance(self.process_manager.health_monitor, HealthMonitor)
        assert self.process_manager.project_root == Path.cwd()
    
    @patch('subprocess.Popen')
    @patch('builtins.open', new_callable=mock_open)
    @patch('pathlib.Path.exists')
    def test_start_backend_success(self, mock_exists, mock_file_open, mock_popen):
        """Test successful backend startup."""
        # Mock file system
        mock_exists.return_value = True
        
        # Mock process
        mock_process = Mock()
        mock_process.pid = 1234
        mock_popen.return_value = mock_process
        
        result = self.process_manager.start_backend(8000)
        
        assert result.success is True
        assert result.process_info is not None
        assert result.process_info.name == "backend"
        assert result.process_info.port == 8000
        assert result.process_info.pid == 1234
        assert "backend" in self.process_manager.processes
    
    @patch('pathlib.Path.exists')
    def test_start_backend_no_script(self, mock_exists):
        """Test backend startup when main script doesn't exist."""
        mock_exists.return_value = False
        
        result = self.process_manager.start_backend(8000)
        
        assert result.success is False
        assert "Backend main script not found" in result.error_message
    
    @patch('subprocess.Popen')
    @patch('builtins.open', new_callable=mock_open)
    @patch('pathlib.Path.exists')
    def test_start_backend_process_failure(self, mock_exists, mock_file_open, mock_popen):
        """Test backend startup when process creation fails."""
        mock_exists.return_value = True
        mock_popen.side_effect = OSError("Permission denied")
        
        result = self.process_manager.start_backend(8000)
        
        assert result.success is False
        assert "Failed to start backend" in result.error_message
    
    @patch('subprocess.run')
    @patch('subprocess.Popen')
    @patch('builtins.open', new_callable=mock_open)
    def test_start_frontend_with_npm(self, mock_file_open, mock_popen, mock_run):
        """Test successful frontend startup with npm."""
        # Mock npm availability
        mock_run.return_value = Mock(returncode=0)
        
        # Mock process
        mock_process = Mock()
        mock_process.pid = 5678
        mock_popen.return_value = mock_process
        
        # Mock Path.exists to return False for yarn.lock (so npm is used)
        with patch('pathlib.Path.exists') as mock_exists:
            mock_exists.side_effect = lambda: str(self).endswith('yarn.lock') == False
            
            result = self.process_manager.start_frontend(3000)
        
        assert result.success is True
        assert result.process_info is not None
        assert result.process_info.name == "frontend"
        assert result.process_info.port == 3000
        assert result.process_info.pid == 5678
        assert "frontend" in self.process_manager.processes
    
    @patch('subprocess.run')
    def test_start_frontend_no_package_manager(self, mock_run):
        """Test frontend startup when no package manager is available."""
        mock_run.side_effect = FileNotFoundError("npm not found")
        
        with patch('pathlib.Path.exists') as mock_exists:
            mock_exists.return_value = False  # No yarn.lock
            
            result = self.process_manager.start_frontend(3000)
        
        assert result.success is False
        assert "No package manager found" in result.error_message
    
    @patch('subprocess.run')
    def test_detect_package_manager_yarn(self, mock_run):
        """Test package manager detection with yarn."""
        frontend_dir = Path("frontend")
        
        with patch.object(Path, 'exists') as mock_exists:
            # Mock yarn.lock exists and yarn is available
            mock_exists.return_value = True
            mock_run.return_value = Mock(returncode=0)
            
            result = self.process_manager._detect_package_manager(frontend_dir)
            assert result == "yarn"
    
    @patch('subprocess.run')
    def test_detect_package_manager_npm(self, mock_run):
        """Test package manager detection with npm."""
        frontend_dir = Path("frontend")
        
        with patch.object(Path, 'exists') as mock_exists:
            mock_exists.return_value = False  # No yarn.lock
            # Only npm call (succeeds)
            mock_run.return_value = Mock(returncode=0)
            
            result = self.process_manager._detect_package_manager(frontend_dir)
            assert result == "npm"
    
    @patch('subprocess.run')
    def test_detect_package_manager_none(self, mock_run):
        """Test package manager detection when none available."""
        frontend_dir = Path("frontend")
        
        with patch.object(Path, 'exists') as mock_exists:
            mock_exists.return_value = False
            mock_run.side_effect = FileNotFoundError("Command not found")
            
            result = self.process_manager._detect_package_manager(frontend_dir)
            assert result is None
    
    def test_get_process_status(self):
        """Test getting process status."""
        process_info = ProcessInfo(name="test")
        self.process_manager.processes["test"] = process_info
        
        result = self.process_manager.get_process_status("test")
        assert result == process_info
        
        result = self.process_manager.get_process_status("nonexistent")
        assert result is None
    
    def test_is_process_healthy(self):
        """Test checking if process is healthy."""
        # Healthy process
        healthy_process = ProcessInfo(name="healthy", status=ProcessStatus.RUNNING)
        self.process_manager.processes["healthy"] = healthy_process
        assert self.process_manager.is_process_healthy("healthy") is True
        
        # Unhealthy process
        unhealthy_process = ProcessInfo(name="unhealthy", status=ProcessStatus.FAILED)
        self.process_manager.processes["unhealthy"] = unhealthy_process
        assert self.process_manager.is_process_healthy("unhealthy") is False
        
        # Nonexistent process
        assert self.process_manager.is_process_healthy("nonexistent") is False
    
    def test_wait_for_health_success(self):
        """Test waiting for process health - success case."""
        process_info = ProcessInfo(name="test", status=ProcessStatus.RUNNING)
        self.process_manager.processes["test"] = process_info
        
        result = self.process_manager.wait_for_health("test", timeout=1.0)
        assert result is True
    
    def test_wait_for_health_timeout(self):
        """Test waiting for process health - timeout case."""
        process_info = ProcessInfo(name="test", status=ProcessStatus.STARTING)
        self.process_manager.processes["test"] = process_info
        
        result = self.process_manager.wait_for_health("test", timeout=0.1)
        assert result is False
    
    def test_get_all_processes(self):
        """Test getting all processes."""
        process1 = ProcessInfo(name="test1")
        process2 = ProcessInfo(name="test2")
        self.process_manager.processes["test1"] = process1
        self.process_manager.processes["test2"] = process2
        
        all_processes = self.process_manager.get_all_processes()
        assert len(all_processes) == 2
        assert all_processes["test1"] == process1
        assert all_processes["test2"] == process2
        
        # Ensure it's a copy
        all_processes["test3"] = ProcessInfo(name="test3")
        assert "test3" not in self.process_manager.processes
    
    def test_stop_process_success(self):
        """Test successful process stopping."""
        mock_process = Mock()
        mock_process.wait.return_value = 0
        
        process_info = ProcessInfo(name="test", process=mock_process)
        self.process_manager.processes["test"] = process_info
        
        result = self.process_manager.stop_process("test")
        assert result is True
        assert "test" not in self.process_manager.processes
        mock_process.terminate.assert_called_once()
    
    def test_stop_process_nonexistent(self):
        """Test stopping nonexistent process."""
        result = self.process_manager.stop_process("nonexistent")
        assert result is True  # Should succeed for nonexistent processes
    
    def test_stop_process_force_kill(self):
        """Test force killing process when graceful shutdown fails."""
        mock_process = Mock()
        # First wait() call times out, second wait() call after kill() succeeds
        mock_process.wait.side_effect = [
            subprocess.TimeoutExpired("cmd", 10),  # First timeout
            None  # Second call succeeds
        ]
        
        process_info = ProcessInfo(name="test", process=mock_process)
        self.process_manager.processes["test"] = process_info
        
        result = self.process_manager.stop_process("test", force=True)
        assert result is True
        mock_process.terminate.assert_called_once()
        mock_process.kill.assert_called_once()


class TestProcessManagerAdvanced:
    """Advanced test scenarios for ProcessManager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = StartupConfig()
        self.process_manager = ProcessManager(self.config)
    
    def teardown_method(self):
        """Clean up after tests."""
        self.process_manager.cleanup()
    
    def test_process_startup_with_environment_variables(self):
        """Test process startup with custom environment variables."""
        custom_env = {
            "NODE_ENV": "development",
            "PORT": "3000",
            "DEBUG": "true"
        }
        
        with patch('subprocess.Popen') as mock_popen:
            mock_process = Mock()
            mock_process.pid = 1234
            mock_popen.return_value = mock_process
            
            with patch('pathlib.Path.exists', return_value=True):
                with patch('builtins.open', mock_open()):
                    result = self.process_manager.start_backend(8000, env_vars=custom_env)
            
            assert result.success is True
            
            # Check that environment variables were passed
            call_args = mock_popen.call_args
            env_arg = call_args[1].get('env', {})
            for key, value in custom_env.items():
                assert env_arg.get(key) == value
    
    def test_process_startup_with_working_directory(self):
        """Test process startup with custom working directory."""
        custom_wd = Path("custom/backend/path")
        
        with patch('subprocess.Popen') as mock_popen:
            mock_process = Mock()
            mock_process.pid = 1234
            mock_popen.return_value = mock_process
            
            with patch('pathlib.Path.exists', return_value=True):
                with patch('builtins.open', mock_open()):
                    result = self.process_manager.start_backend(8000, working_dir=custom_wd)
            
            assert result.success is True
            
            # Check that working directory was set
            call_args = mock_popen.call_args
            assert call_args[1]['cwd'] == str(custom_wd)
    
    def test_health_monitoring_with_custom_endpoints(self):
        """Test health monitoring with custom health check endpoints."""
        process_info = ProcessInfo(
            name="backend",
            health_check_url="http://localhost:8000/api/health",
            status=ProcessStatus.RUNNING
        )
        
        mock_process = Mock()
        mock_process.poll.return_value = None
        process_info.process = mock_process
        
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"status": "healthy", "uptime": 3600}
            mock_get.return_value = mock_response
            
            self.process_manager.health_monitor._check_process_health(process_info)
            
            assert process_info.status == ProcessStatus.RUNNING
            mock_get.assert_called_with(
                "http://localhost:8000/api/health",
                timeout=5
            )
    
    def test_process_resource_monitoring(self):
        """Test monitoring of process resource usage."""
        with patch('psutil.Process') as mock_psutil:
            mock_proc = Mock()
            mock_proc.memory_info.return_value = Mock(rss=1024000, vms=2048000)
            mock_proc.memory_percent.return_value = 5.5
            mock_proc.cpu_percent.return_value = 10.2
            mock_proc.num_threads.return_value = 4
            mock_proc.open_files.return_value = [Mock(), Mock()]  # 2 open files
            mock_proc.connections.return_value = [Mock()]  # 1 connection
            mock_psutil.return_value = mock_proc
            
            process_info = ProcessInfo(name="test", pid=1234)
            self.process_manager.processes["test"] = process_info
            
            metrics = self.process_manager.get_process_metrics("test")
            
            assert metrics["memory_rss"] == 1024000
            assert metrics["cpu_percent"] == 10.2
            assert metrics["num_threads"] == 4
            assert metrics["open_files"] == 2
            assert metrics["connections"] == 1
    
    def test_process_log_rotation(self):
        """Test process log file rotation."""
        import tempfile

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            # Create a large log file
            large_content = "Log line\n" * 10000  # ~100KB
            f.write(large_content)
            log_path = Path(f.name)
        
        try:
            process_info = ProcessInfo(name="test", log_file=str(log_path))
            self.process_manager.processes["test"] = process_info
            
            # Trigger log rotation
            self.process_manager.rotate_process_logs("test", max_size_mb=0.05)  # 50KB limit
            
            # Check that log was rotated
            rotated_path = log_path.with_suffix('.log.1')
            assert rotated_path.exists() or log_path.stat().st_size < 60000  # Either rotated or truncated
            
        finally:
            log_path.unlink(missing_ok=True)
            rotated_path = log_path.with_suffix('.log.1')
            rotated_path.unlink(missing_ok=True)
    
    def test_process_dependency_management(self):
        """Test process startup with dependency checking."""
        # Mock dependency chain: frontend depends on backend
        backend_info = ProcessInfo(name="backend", status=ProcessStatus.RUNNING)
        self.process_manager.processes["backend"] = backend_info
        
        with patch('subprocess.Popen') as mock_popen:
            mock_process = Mock()
            mock_process.pid = 5678
            mock_popen.return_value = mock_process
            
            with patch('subprocess.run', return_value=Mock(returncode=0)):
                with patch('pathlib.Path.exists', return_value=False):  # No yarn.lock
                    result = self.process_manager.start_frontend(
                        3000, 
                        dependencies=["backend"]
                    )
            
            assert result.success is True
    
    def test_process_startup_retry_logic(self):
        """Test process startup with retry logic on failure."""
        attempt_count = 0
        
        def mock_popen_side_effect(*args, **kwargs):
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise OSError("Temporary failure")
            
            mock_process = Mock()
            mock_process.pid = 1234
            return mock_process
        
        with patch('subprocess.Popen', side_effect=mock_popen_side_effect):
            with patch('pathlib.Path.exists', return_value=True):
                with patch('builtins.open', mock_open()):
                    with patch('time.sleep'):  # Speed up test
                        result = self.process_manager.start_backend(8000, max_retries=3)
        
        assert result.success is True
        assert attempt_count == 3
    
    def test_process_cleanup_on_manager_shutdown(self):
        """Test that all processes are cleaned up when manager shuts down."""
        # Add some mock processes
        mock_processes = []
        for i in range(3):
            mock_process = Mock()
            mock_process.poll.return_value = None  # Running
            mock_processes.append(mock_process)
            
            process_info = ProcessInfo(name=f"test{i}", process=mock_process)
            self.process_manager.processes[f"test{i}"] = process_info
        
        # Cleanup should terminate all processes
        self.process_manager.cleanup()
        
        for mock_process in mock_processes:
            mock_process.terminate.assert_called_once()

        assert True  # TODO: Add proper assertion
    
    def test_process_signal_handling(self):
        """Test process signal handling on different platforms."""
        import signal

        mock_process = Mock()
        mock_process.poll.return_value = None
        
        process_info = ProcessInfo(name="test", process=mock_process)
        self.process_manager.processes["test"] = process_info
        
        # Test graceful shutdown signal
        if os.name == 'nt':
            # Windows
            self.process_manager.send_signal("test", signal.SIGTERM)
            mock_process.terminate.assert_called_once()
        else:
            # Unix-like
            self.process_manager.send_signal("test", signal.SIGTERM)
            mock_process.send_signal.assert_called_with(signal.SIGTERM)


        assert True  # TODO: Add proper assertion

class TestProcessManagerErrorHandling:
    """Test error handling scenarios in ProcessManager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = StartupConfig()
        self.process_manager = ProcessManager(self.config)
    
    def teardown_method(self):
        """Clean up after tests."""
        self.process_manager.cleanup()
    
    def test_handle_process_crash_during_startup(self):
        """Test handling of process crash during startup."""
        with patch('subprocess.Popen') as mock_popen:
            mock_process = Mock()
            mock_process.pid = 1234
            mock_process.poll.return_value = 1  # Process crashed
            mock_popen.return_value = mock_process
            
            with patch('pathlib.Path.exists', return_value=True):
                with patch('builtins.open', mock_open()):
                    result = self.process_manager.start_backend(8000)
            
            # Should detect crash and report failure
            assert result.success is False
            assert "crashed" in result.error_message.lower()
    
    def test_handle_zombie_process_cleanup(self):
        """Test cleanup of zombie processes."""
        import psutil

        # Create a zombie process scenario
        mock_process = Mock()
        mock_process.poll.return_value = 0  # Process is dead
        
        process_info = ProcessInfo(name="zombie", process=mock_process, pid=1234)
        self.process_manager.processes["zombie"] = process_info
        
        with patch('psutil.Process') as mock_psutil:
            mock_psutil_proc = Mock()
            mock_psutil_proc.is_running.return_value = False
            mock_psutil_proc.status.return_value = psutil.STATUS_ZOMBIE
            mock_psutil.return_value = mock_psutil_proc
            
            cleaned = self.process_manager.cleanup_zombie_processes()
            
            assert "zombie" in cleaned
            assert process_info.status == ProcessStatus.STOPPED
    
    def test_handle_permission_denied_on_process_kill(self):
        """Test handling permission denied when killing processes."""
        import psutil

        mock_process = Mock()
        mock_process.terminate.side_effect = psutil.AccessDenied()
        
        process_info = ProcessInfo(name="protected", process=mock_process)
        self.process_manager.processes["protected"] = process_info
        
        result = self.process_manager.stop_process("protected")
        
        # Should handle gracefully and report failure
        assert result is False
    
    def test_handle_health_check_failures(self):
        """Test handling of health check failures."""
        import requests

        process_info = ProcessInfo(
            name="unhealthy",
            health_check_url="http://localhost:8000/health",
            status=ProcessStatus.RUNNING
        )
        
        mock_process = Mock()
        mock_process.poll.return_value = None  # Process still running
        process_info.process = mock_process
        
        # Test various health check failures
        failure_scenarios = [
            requests.ConnectionError("Connection refused"),
            requests.Timeout("Request timeout"),
            requests.HTTPError("500 Internal Server Error"),
        ]
        
        for exception in failure_scenarios:
            with patch('requests.get', side_effect=exception):
                self.process_manager.health_monitor._check_process_health(process_info)
                assert process_info.status == ProcessStatus.FAILED
                
                # Reset for next test
                process_info.status = ProcessStatus.RUNNING
    
    def test_handle_log_file_access_errors(self):
        """Test handling of log file access errors."""
        process_info = ProcessInfo(name="test", log_file="/protected/path/test.log")
        self.process_manager.processes["test"] = process_info
        
        # Mock permission denied on log file access
        with patch('builtins.open', side_effect=PermissionError("Access denied")):
            # Should handle gracefully without crashing
            logs = self.process_manager.get_process_logs("test", lines=10)
            assert logs == []  # Empty logs due to access error
    
    def test_handle_resource_exhaustion(self):
        """Test handling of system resource exhaustion."""
        with patch('subprocess.Popen', side_effect=OSError("Cannot allocate memory")):
            with patch('pathlib.Path.exists', return_value=True):
                with patch('builtins.open', mock_open()):
                    result = self.process_manager.start_backend(8000)
            
            assert result.success is False
            assert "resource" in result.error_message.lower() or "memory" in result.error_message.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])