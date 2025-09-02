"""
Unit tests for the performance monitoring system.

Tests timing metrics collection, success/failure rate tracking,
resource usage monitoring, and trend analysis.
"""

import pytest
import time
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from startup_manager.performance_monitor import (
    PerformanceMonitor,
    TimingMetric,
    ResourceSnapshot,
    StartupSession,
    PerformanceStats,
    StartupPhase,
    MetricType,
    get_performance_monitor
)


class TestTimingMetric:
    """Test TimingMetric functionality."""
    
    def test_timing_metric_creation(self):
        """Test creating a timing metric."""
        start_time = time.time()
        metric = TimingMetric(
            operation="test_operation",
            start_time=start_time,
            metadata={"test": "data"}
        )
        
        assert metric.operation == "test_operation"
        assert metric.start_time == start_time
        assert metric.end_time is None
        assert metric.duration is None
        assert metric.success is True
        assert metric.metadata == {"test": "data"}
    
    def test_timing_metric_finish(self):
        """Test finishing a timing metric."""
        start_time = time.time()
        metric = TimingMetric(operation="test", start_time=start_time)
        
        time.sleep(0.01)  # Small delay
        metric.finish(success=True)
        
        assert metric.end_time is not None
        assert metric.duration is not None
        assert metric.duration > 0
        assert metric.success is True
        assert metric.error_message is None
    
    def test_timing_metric_finish_with_error(self):
        """Test finishing a timing metric with error."""
        metric = TimingMetric(operation="test", start_time=time.time())
        
        metric.finish(success=False, error_message="Test error")
        
        assert metric.success is False
        assert metric.error_message == "Test error"


class TestResourceSnapshot:
    """Test ResourceSnapshot functionality."""
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_io_counters')
    @patch('psutil.net_io_counters')
    @patch('psutil.pids')
    def test_resource_snapshot_capture(
        self,
        mock_pids,
        mock_net_io,
        mock_disk_io,
        mock_memory,
        mock_cpu
    ):
        """Test capturing resource snapshot."""
        # Mock system metrics
        mock_cpu.return_value = 25.5
        mock_memory.return_value = Mock(percent=60.0, used=8 * 1024 * 1024 * 1024)  # 8GB
        mock_disk_io.return_value = Mock(read_bytes=1024 * 1024, write_bytes=2 * 1024 * 1024)
        mock_net_io.return_value = Mock(bytes_sent=512 * 1024, bytes_recv=1024 * 1024)
        mock_pids.return_value = list(range(100))  # 100 processes
        
        snapshot = ResourceSnapshot.capture()
        
        assert snapshot.cpu_percent == 25.5
        assert snapshot.memory_percent == 60.0
        assert snapshot.memory_used_mb == 8 * 1024  # 8GB in MB
        assert snapshot.disk_io_read_mb == 1.0  # 1MB
        assert snapshot.disk_io_write_mb == 2.0  # 2MB
        assert snapshot.network_sent_mb == 0.5  # 0.5MB
        assert snapshot.network_recv_mb == 1.0  # 1MB
        assert snapshot.process_count == 100
        assert snapshot.timestamp > 0


class TestStartupSession:
    """Test StartupSession functionality."""
    
    def test_startup_session_creation(self):
        """Test creating a startup session."""
        start_time = time.time()
        session = StartupSession(
            session_id="test_session",
            start_time=start_time,
            metadata={"test": "data"}
        )
        
        assert session.session_id == "test_session"
        assert session.start_time == start_time
        assert session.end_time is None
        assert session.success is False
        assert session.total_duration is None
        assert session.metadata == {"test": "data"}
    
    def test_startup_session_finish(self):
        """Test finishing a startup session."""
        session = StartupSession(
            session_id="test",
            start_time=time.time()
        )
        
        time.sleep(0.01)  # Small delay
        session.finish(success=True)
        
        assert session.end_time is not None
        assert session.success is True
        assert session.total_duration is not None
        assert session.total_duration > 0


class TestPerformanceMonitor:
    """Test PerformanceMonitor functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.monitor = PerformanceMonitor(
            data_dir=self.temp_dir,
            max_sessions=10,
            resource_sampling_interval=0.1
        )
    
    def test_performance_monitor_initialization(self):
        """Test performance monitor initialization."""
        assert self.monitor.data_dir == Path(self.temp_dir)
        assert self.monitor.max_sessions == 10
        assert self.monitor.resource_sampling_interval == 0.1
        assert self.monitor.current_session is None
        assert len(self.monitor.active_timings) == 0
        assert len(self.monitor.sessions) == 0
    
    def test_start_session(self):
        """Test starting a performance monitoring session."""
        metadata = {"test": "data"}
        session_id = self.monitor.start_session(metadata)
        
        assert session_id is not None
        assert self.monitor.current_session is not None
        assert self.monitor.current_session.session_id == session_id
        assert self.monitor.current_session.metadata == metadata
        assert self.monitor.current_session.start_time > 0
    
    def test_finish_session(self):
        """Test finishing a performance monitoring session."""
        session_id = self.monitor.start_session()
        
        time.sleep(0.01)  # Small delay
        self.monitor.finish_session(success=True)
        
        assert self.monitor.current_session is None
        assert len(self.monitor.sessions) == 1
        
        session = self.monitor.sessions[0]
        assert session.session_id == session_id
        assert session.success is True
        assert session.total_duration is not None
        assert session.total_duration > 0
    
    def test_start_timing(self):
        """Test starting operation timing."""
        self.monitor.start_session()
        
        metadata = {"phase": "test"}
        timing_id = self.monitor.start_timing("test_operation", metadata)
        
        assert timing_id is not None
        assert timing_id in self.monitor.active_timings
        
        timing = self.monitor.active_timings[timing_id]
        assert timing.operation == "test_operation"
        assert timing.metadata == metadata
        assert timing.start_time > 0
        assert timing.end_time is None
    
    def test_finish_timing(self):
        """Test finishing operation timing."""
        self.monitor.start_session()
        timing_id = self.monitor.start_timing("test_operation")
        
        time.sleep(0.01)  # Small delay
        self.monitor.finish_timing(timing_id, success=True)
        
        assert timing_id not in self.monitor.active_timings
        
        # Check if timing was added to current session
        session = self.monitor.current_session
        assert "test_operation" in session.phase_timings
        
        timing = session.phase_timings["test_operation"]
        assert timing.success is True
        assert timing.duration is not None
        assert timing.duration > 0
    
    def test_finish_timing_with_error(self):
        """Test finishing operation timing with error."""
        self.monitor.start_session()
        timing_id = self.monitor.start_timing("test_operation")
        
        self.monitor.finish_timing(timing_id, success=False, error_message="Test error")
        
        session = self.monitor.current_session
        timing = session.phase_timings["test_operation"]
        assert timing.success is False
        assert timing.error_message == "Test error"
        assert "Test error" in session.errors
    
    def test_time_operation_context_manager(self):
        """Test timing operation using context manager."""
        self.monitor.start_session()
        
        with self.monitor.time_operation("test_context", {"test": "data"}):
            time.sleep(0.01)  # Small delay
        
        session = self.monitor.current_session
        assert "test_context" in session.phase_timings
        
        timing = session.phase_timings["test_context"]
        assert timing.success is True
        assert timing.duration is not None
        assert timing.duration > 0
        assert timing.metadata == {"test": "data"}
    
    def test_time_operation_context_manager_with_exception(self):
        """Test timing operation context manager with exception."""
        self.monitor.start_session()
        
        with pytest.raises(ValueError):
            with self.monitor.time_operation("test_error"):
                raise ValueError("Test exception")
        
        session = self.monitor.current_session
        timing = session.phase_timings["test_error"]
        assert timing.success is False
        assert "Test exception" in timing.error_message
    
    def test_record_error(self):
        """Test recording errors."""
        self.monitor.start_session()
        
        self.monitor.record_error("Test error", "test_operation")
        
        session = self.monitor.current_session
        assert "test_operation: Test error" in session.errors
    
    @patch('startup_manager.performance_monitor.ResourceSnapshot.capture')
    def test_resource_monitoring(self, mock_capture):
        """Test resource monitoring during session."""
        # Mock resource snapshots
        mock_snapshot = Mock()
        mock_snapshot.timestamp = time.time()
        mock_snapshot.cpu_percent = 50.0
        mock_capture.return_value = mock_snapshot
        
        self.monitor.resource_sampling_interval = 0.05  # Fast sampling for test
        self.monitor.start_session()
        
        time.sleep(0.15)  # Allow some samples
        
        self.monitor.finish_session()
        
        session = self.monitor.sessions[0]
        assert len(session.resource_snapshots) > 0
        assert mock_capture.called
    
    def test_get_performance_stats_empty(self):
        """Test getting performance stats with no data."""
        stats = self.monitor.get_performance_stats()
        
        assert stats.total_sessions == 0
        assert stats.success_rate == 0.0
        assert stats.average_duration == 0.0
        assert stats.trend_direction == "stable"
    
    def test_get_performance_stats_with_data(self):
        """Test getting performance stats with session data."""
        # Create some test sessions
        for i in range(5):
            session = StartupSession(
                session_id=f"session_{i}",
                start_time=time.time() - (i * 3600),  # 1 hour apart
                success=i % 2 == 0,  # Alternate success/failure
                total_duration=1.0 + (i * 0.1)  # Increasing duration
            )
            session.end_time = session.start_time + session.total_duration
            self.monitor.sessions.append(session)
        
        stats = self.monitor.get_performance_stats()
        
        assert stats.total_sessions == 5
        assert stats.success_rate == 0.6  # 3 out of 5 successful
        assert stats.average_duration > 0
        assert stats.min_duration > 0
        assert stats.max_duration > stats.min_duration
    
    def test_get_resource_usage_summary_empty(self):
        """Test getting resource usage summary with no data."""
        summary = self.monitor.get_resource_usage_summary()
        assert summary == {}
    
    def test_get_resource_usage_summary_with_data(self):
        """Test getting resource usage summary with data."""
        # Create session with resource snapshots
        session = StartupSession(session_id="test", start_time=time.time())
        
        # Add mock resource snapshots
        for i in range(3):
            snapshot = Mock()
            snapshot.cpu_percent = 50.0 + i * 10
            snapshot.memory_percent = 60.0 + i * 5
            snapshot.memory_used_mb = 1000.0 + i * 100
            snapshot.disk_io_read_mb = 10.0 + i
            snapshot.disk_io_write_mb = 20.0 + i
            snapshot.network_sent_mb = 5.0 + i
            snapshot.network_recv_mb = 15.0 + i
            session.resource_snapshots.append(snapshot)
        
        self.monitor.sessions.append(session)
        
        summary = self.monitor.get_resource_usage_summary()
        
        assert "cpu_usage" in summary
        assert "memory_usage" in summary
        assert "disk_io" in summary
        assert "network_io" in summary
        
        assert summary["cpu_usage"]["average"] == 60.0  # (50+60+70)/3
        assert summary["cpu_usage"]["peak"] == 70.0
        assert summary["memory_usage"]["peak_percent"] == 70.0
    
    def test_trend_analysis_stable(self):
        """Test trend analysis with stable performance."""
        # Create sessions with stable durations
        for i in range(10):
            session = StartupSession(
                session_id=f"session_{i}",
                start_time=time.time() - (i * 3600),
                total_duration=1.0  # Constant duration
            )
            self.monitor.sessions.append(session)
        
        trend = self.monitor._analyze_trend(list(self.monitor.sessions))
        assert trend == "stable"
    
    def test_trend_analysis_improving(self):
        """Test trend analysis with improving performance."""
        # Create sessions with decreasing durations over time (improving)
        base_time = time.time() - (10 * 3600)  # Start 10 hours ago
        for i in range(10):
            session = StartupSession(
                session_id=f"session_{i}",
                start_time=base_time + (i * 3600),  # Increasing timestamps (newer sessions)
                total_duration=2.0 - (i * 0.15)  # Decreasing duration (improving performance)
            )
            self.monitor.sessions.append(session)
        
        trend = self.monitor._analyze_trend(list(self.monitor.sessions))
        assert trend == "improving"
    
    def test_trend_analysis_degrading(self):
        """Test trend analysis with degrading performance."""
        # Create sessions with increasing durations over time (degrading)
        base_time = time.time() - (10 * 3600)  # Start 10 hours ago
        for i in range(10):
            session = StartupSession(
                session_id=f"session_{i}",
                start_time=base_time + (i * 3600),  # Increasing timestamps (newer sessions)
                total_duration=1.0 + (i * 0.15)  # Increasing duration (degrading performance)
            )
            self.monitor.sessions.append(session)
        
        trend = self.monitor._analyze_trend(list(self.monitor.sessions))
        assert trend == "degrading"
    
    def test_data_persistence(self):
        """Test saving and loading performance data."""
        # Create a session
        self.monitor.start_session({"test": "data"})
        
        with self.monitor.time_operation("test_op"):
            time.sleep(0.01)
        
        self.monitor.finish_session(success=True)
        
        # Create new monitor instance with same data directory
        new_monitor = PerformanceMonitor(data_dir=self.temp_dir)
        
        # Check if data was loaded
        assert len(new_monitor.sessions) == 1
        session = new_monitor.sessions[0]
        assert session.metadata == {"test": "data"}
        assert session.success is True
        assert "test_op" in session.phase_timings


class TestGlobalPerformanceMonitor:
    """Test global performance monitor functionality."""
    
    def test_get_performance_monitor_singleton(self):
        """Test that get_performance_monitor returns singleton."""
        monitor1 = get_performance_monitor()
        monitor2 = get_performance_monitor()
        
        assert monitor1 is monitor2
    
    def test_get_performance_monitor_with_args(self):
        """Test get_performance_monitor with custom arguments."""
        # Reset global monitor
        import startup_manager.performance_monitor
        startup_manager.performance_monitor._global_monitor = None
        
        monitor = get_performance_monitor(max_sessions=50)
        assert monitor.max_sessions == 50


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.monitor = PerformanceMonitor(data_dir=self.temp_dir)
    
    def test_complete_startup_scenario(self):
        """Test complete startup monitoring scenario."""
        # Start session
        session_id = self.monitor.start_session({
            "user": "test_user",
            "environment": "development"
        })
        
        # Simulate startup phases
        phases = [
            ("environment_validation", 0.5),
            ("port_management", 0.2),
            ("process_startup", 1.0),
            ("health_verification", 0.3)
        ]
        
        for phase, duration in phases:
            with self.monitor.time_operation(phase):
                time.sleep(duration * 0.01)  # Scale down for test
        
        # Finish session
        self.monitor.finish_session(success=True)
        
        # Verify results
        assert len(self.monitor.sessions) == 1
        session = self.monitor.sessions[0]
        
        assert session.session_id == session_id
        assert session.success is True
        assert len(session.phase_timings) == 4
        
        for phase, _ in phases:
            assert phase in session.phase_timings
            timing = session.phase_timings[phase]
            assert timing.success is True
            assert timing.duration is not None
    
    def test_startup_with_errors(self):
        """Test startup monitoring with errors."""
        self.monitor.start_session()
        
        # Simulate successful phase
        with self.monitor.time_operation("environment_validation"):
            time.sleep(0.01)
        
        # Simulate failed phase
        try:
            with self.monitor.time_operation("port_management"):
                raise RuntimeError("Port conflict detected")
        except RuntimeError:
            pass
        
        # Record additional error
        self.monitor.record_error("Configuration invalid", "config_validation")
        
        self.monitor.finish_session(success=False)
        
        # Verify error tracking
        session = self.monitor.sessions[0]
        assert session.success is False
        assert len(session.errors) >= 1
        assert "Port conflict detected" in str(session.errors)
        assert "config_validation: Configuration invalid" in session.errors
        
        # Check phase timing success/failure
        env_timing = session.phase_timings["environment_validation"]
        port_timing = session.phase_timings["port_management"]
        
        assert env_timing.success is True
        assert port_timing.success is False
        assert "Port conflict detected" in port_timing.error_message
    
    def test_performance_degradation_detection(self):
        """Test detection of performance degradation over time."""
        # Simulate sessions with gradually increasing duration
        base_time = time.time() - (24 * 3600)  # 24 hours ago
        
        for i in range(20):
            session = StartupSession(
                session_id=f"session_{i}",
                start_time=base_time + (i * 3600),  # 1 hour apart
                success=True,
                total_duration=1.0 + (i * 0.1)  # Gradually increasing
            )
            session.end_time = session.start_time + session.total_duration
            self.monitor.sessions.append(session)
        
        # Get performance stats
        stats = self.monitor.get_performance_stats()
        
        assert stats.total_sessions == 20
        assert stats.success_rate == 1.0
        assert stats.trend_direction == "degrading"
        assert stats.max_duration > stats.min_duration


if __name__ == "__main__":
    pytest.main([__file__])