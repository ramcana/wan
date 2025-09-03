"""
Integration tests for Process Manager lifecycle management.
Tests graceful shutdown, process cleanup, and restart functionality.
"""

import os
import sys
import time
import pytest
import subprocess
import threading
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from datetime import datetime, timedelta

# Add the project root to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.startup_manager.process_manager import (
    ProcessManager, ProcessInfo, ProcessResult, ProcessStatus
)
from scripts.startup_manager.config import StartupConfig


class TestProcessLifecycleIntegration:
    """Integration tests for process lifecycle management."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = StartupConfig()
        self.process_manager = ProcessManager(self.config)
    
    def teardown_method(self):
        """Clean up after tests."""
        self.process_manager.cleanup()
    
    def test_graceful_shutdown_success(self):
        """Test successful graceful shutdown."""
        mock_process = Mock()
        mock_process.poll.return_value = None  # Process is running
        mock_process.wait.return_value = 0  # Graceful shutdown succeeds
        
        process_info = ProcessInfo(name="test", process=mock_process)
        self.process_manager.processes["test"] = process_info
        
        result = self.process_manager.graceful_shutdown("test", timeout=5.0)
        
        assert result is True
        assert process_info.status == ProcessStatus.STOPPED
        
        if os.name == 'nt':
            mock_process.terminate.assert_called_once()
        else:
            mock_process.send_signal.assert_called_once()
    
    def test_graceful_shutdown_force_kill(self):
        """Test graceful shutdown with force kill fallback."""
        mock_process = Mock()
        mock_process.poll.return_value = None  # Process is running
        mock_process.wait.side_effect = [
            subprocess.TimeoutExpired("cmd", 5),  # Graceful shutdown times out
            None  # Force kill succeeds
        ]
        
        process_info = ProcessInfo(name="test", process=mock_process)
        self.process_manager.processes["test"] = process_info
        
        result = self.process_manager.graceful_shutdown("test", timeout=5.0)
        
        assert result is True
        assert process_info.status == ProcessStatus.STOPPED
        mock_process.kill.assert_called_once()
    
    def test_graceful_shutdown_stuck_process(self):
        """Test graceful shutdown when process is completely stuck."""
        mock_process = Mock()
        mock_process.poll.return_value = None  # Process is running
        mock_process.wait.side_effect = subprocess.TimeoutExpired("cmd", 5)  # Always times out
        
        process_info = ProcessInfo(name="test", process=mock_process)
        self.process_manager.processes["test"] = process_info
        
        result = self.process_manager.graceful_shutdown("test", timeout=5.0)
        
        assert result is False
        assert process_info.status == ProcessStatus.FAILED
    
    def test_graceful_shutdown_nonexistent_process(self):
        """Test graceful shutdown of nonexistent process."""
        result = self.process_manager.graceful_shutdown("nonexistent")
        assert result is True  # Should succeed for nonexistent processes
    
    @patch('psutil.Process')
    def test_cleanup_zombie_processes(self, mock_psutil_process):
        """Test cleanup of zombie processes."""
        # Set up mock processes
        mock_process1 = Mock()
        mock_process1.poll.return_value = 0  # Dead process
        mock_process1.stdout = Mock()
        mock_process1.stderr = Mock()
        mock_process1.stdin = Mock()
        
        mock_process2 = Mock()
        mock_process2.poll.return_value = None  # Live process
        
        process_info1 = ProcessInfo(name="dead", process=mock_process1, pid=1234)
        process_info2 = ProcessInfo(name="alive", process=mock_process2, pid=5678)
        
        self.process_manager.processes["dead"] = process_info1
        self.process_manager.processes["alive"] = process_info2
        
        # Mock psutil - different behavior for different PIDs
        def mock_psutil_factory(pid):
            mock_instance = Mock()
            if pid == 1234:  # Dead process
                mock_instance.is_running.return_value = False
            else:  # Live process
                mock_instance.is_running.return_value = True
            return mock_instance
        
        mock_psutil_process.side_effect = mock_psutil_factory
        
        cleaned = self.process_manager.cleanup_zombie_processes()
        
        assert "dead" in cleaned
        assert "alive" not in cleaned
        assert process_info1.status == ProcessStatus.STOPPED
        assert process_info2.status != ProcessStatus.STOPPED
    
    @patch('time.sleep')
    def test_restart_process_success(self, mock_sleep):
        """Test successful process restart."""
        # Set up initial process
        mock_process = Mock()
        mock_process.poll.return_value = 0  # Process is dead
        
        process_info = ProcessInfo(
            name="backend",
            process=mock_process,
            port=8000,
            restart_count=0
        )
        self.process_manager.processes["backend"] = process_info
        
        # Mock the start_backend method
        new_process_info = ProcessInfo(name="backend", port=8000, pid=9999)
        mock_result = ProcessResult.success_result(new_process_info)
        
        with patch.object(self.process_manager, 'start_backend', return_value=mock_result):
            result = self.process_manager.restart_process("backend")
        
        assert result.success is True
        assert result.process_info.restart_count == 1
        assert result.process_info.last_restart is not None
        mock_sleep.assert_not_called()  # No sleep on first restart
    
    @patch('time.sleep')
    def test_restart_process_with_backoff(self, mock_sleep):
        """Test process restart with exponential backoff."""
        # Set up process that has already been restarted once
        mock_process = Mock()
        mock_process.poll.return_value = 0
        
        process_info = ProcessInfo(
            name="backend",
            process=mock_process,
            port=8000,
            restart_count=1  # Already restarted once
        )
        self.process_manager.processes["backend"] = process_info
        
        # Mock the start_backend method
        new_process_info = ProcessInfo(name="backend", port=8000, pid=9999)
        mock_result = ProcessResult.success_result(new_process_info)
        
        with patch.object(self.process_manager, 'start_backend', return_value=mock_result):
            result = self.process_manager.restart_process("backend")
        
        assert result.success is True
        assert result.process_info.restart_count == 2
        mock_sleep.assert_called_once_with(2)  # 2^1 = 2 seconds backoff
    
    def test_restart_process_max_attempts_exceeded(self):
        """Test restart failure when max attempts exceeded."""
        process_info = ProcessInfo(
            name="backend",
            restart_count=3  # Already at max attempts
        )
        self.process_manager.processes["backend"] = process_info
        
        result = self.process_manager.restart_process("backend", max_attempts=3)
        
        assert result.success is False
        assert "exceeded maximum restart attempts" in result.error_message
    
    def test_restart_nonexistent_process(self):
        """Test restart of nonexistent process."""
        result = self.process_manager.restart_process("nonexistent")
        
        assert result.success is False
        assert "not found" in result.error_message
    
    def test_auto_restart_failed_processes(self):
        """Test automatic restart of failed processes."""
        # Set up failed process with auto_restart enabled
        failed_process = ProcessInfo(
            name="backend",
            status=ProcessStatus.FAILED,
            auto_restart=True,
            restart_count=0,
            port=8000
        )
        self.process_manager.processes["backend"] = failed_process
        
        # Set up healthy process (should not be restarted)
        healthy_process = ProcessInfo(
            name="frontend",
            status=ProcessStatus.RUNNING,
            auto_restart=True
        )
        self.process_manager.processes["frontend"] = healthy_process
        
        # Mock the restart_process method
        mock_result = ProcessResult.success_result(ProcessInfo(name="backend"))
        with patch.object(self.process_manager, 'restart_process', return_value=mock_result) as mock_restart:
            results = self.process_manager.auto_restart_failed_processes()
        
        assert "backend" in results
        assert "frontend" not in results
        mock_restart.assert_called_once_with("backend")
    
    def test_auto_restart_respects_time_limit(self):
        """Test that auto-restart respects minimum time between restarts."""
        # Set up failed process that was recently restarted
        failed_process = ProcessInfo(
            name="backend",
            status=ProcessStatus.FAILED,
            auto_restart=True,
            restart_count=1,
            last_restart=datetime.now() - timedelta(seconds=30)  # 30 seconds ago
        )
        self.process_manager.processes["backend"] = failed_process
        
        with patch.object(self.process_manager, 'restart_process') as mock_restart:
            results = self.process_manager.auto_restart_failed_processes()
        
        assert "backend" not in results  # Should not restart too soon
        mock_restart.assert_not_called()
    
    @patch('psutil.Process')
    def test_get_process_metrics(self, mock_psutil_process):
        """Test getting process metrics."""
        # Set up mock psutil process
        mock_proc = Mock()
        mock_proc.memory_info.return_value = Mock(rss=1024000, vms=2048000)
        mock_proc.memory_percent.return_value = 5.5
        mock_proc.cpu_percent.return_value = 10.2
        mock_proc.status.return_value = "running"
        mock_psutil_process.return_value = mock_proc
        
        # Set up process info
        start_time = datetime.now() - timedelta(minutes=5)
        process_info = ProcessInfo(
            name="test",
            pid=1234,
            port=8000,
            start_time=start_time,
            restart_count=2,
            last_restart=datetime.now() - timedelta(minutes=2),
            health_check_url="http://localhost:8000/health"
        )
        self.process_manager.processes["test"] = process_info
        
        metrics = self.process_manager.get_process_metrics("test")
        
        assert metrics is not None
        assert metrics["pid"] == 1234
        assert metrics["status"] == "running"
        assert metrics["cpu_percent"] == 10.2
        assert metrics["memory_rss"] == 1024000
        assert metrics["memory_vms"] == 2048000
        assert metrics["memory_percent"] == 5.5
        assert metrics["uptime_seconds"] is not None
        assert metrics["restart_count"] == 2
        assert metrics["port"] == 8000
        assert metrics["health_check_url"] == "http://localhost:8000/health"
    
    def test_get_process_metrics_nonexistent(self):
        """Test getting metrics for nonexistent process."""
        metrics = self.process_manager.get_process_metrics("nonexistent")
        assert metrics is None
    
    @patch('psutil.Process')
    def test_get_process_metrics_dead_process(self, mock_psutil_process):
        """Test getting metrics for dead process."""
        import psutil
mock_psutil_process.side_effect = psutil.NoSuchProcess(1234)
        
        process_info = ProcessInfo(name="test", pid=1234)
        self.process_manager.processes["test"] = process_info
        
        metrics = self.process_manager.get_process_metrics("test")
        assert metrics is None
    
    def test_set_auto_restart(self):
        """Test setting auto-restart flag."""
        process_info = ProcessInfo(name="test", auto_restart=True)
        self.process_manager.processes["test"] = process_info
        
        # Disable auto-restart
        result = self.process_manager.set_auto_restart("test", False)
        assert result is True
        assert process_info.auto_restart is False
        
        # Enable auto-restart
        result = self.process_manager.set_auto_restart("test", True)
        assert result is True
        assert process_info.auto_restart is True
        
        # Test nonexistent process
        result = self.process_manager.set_auto_restart("nonexistent", True)
        assert result is False
    
    def test_reset_restart_count(self):
        """Test resetting restart count."""
        process_info = ProcessInfo(
            name="test",
            restart_count=5,
            last_restart=datetime.now()
        )
        self.process_manager.processes["test"] = process_info
        
        result = self.process_manager.reset_restart_count("test")
        assert result is True
        assert process_info.restart_count == 0
        assert process_info.last_restart is None
        
        # Test nonexistent process
        result = self.process_manager.reset_restart_count("nonexistent")
        assert result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])