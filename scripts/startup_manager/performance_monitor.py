"""
Performance monitoring system for startup manager.

This module provides comprehensive timing metrics collection, success/failure rate tracking,
resource usage monitoring, and trend analysis for the startup process.
"""

import time
import psutil
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict, field
from collections import defaultdict, deque
import json
import statistics
from enum import Enum


class MetricType(Enum):
    """Types of metrics collected."""
    TIMING = "timing"
    SUCCESS_RATE = "success_rate"
    RESOURCE_USAGE = "resource_usage"
    ERROR_COUNT = "error_count"
    PHASE_DURATION = "phase_duration"


class StartupPhase(Enum):
    """Startup phases for timing measurement."""
    ENVIRONMENT_VALIDATION = "environment_validation"
    PORT_MANAGEMENT = "port_management"
    PROCESS_STARTUP = "process_startup"
    HEALTH_VERIFICATION = "health_verification"
    TOTAL_STARTUP = "total_startup"


@dataclass
class TimingMetric:
    """Individual timing measurement."""
    operation: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def finish(self, success: bool = True, error_message: Optional[str] = None):
        """Mark the timing as finished."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.success = success
        self.error_message = error_message


@dataclass
class ResourceSnapshot:
    """System resource usage snapshot."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_sent_mb: float
    network_recv_mb: float
    process_count: int
    
    @classmethod
    def capture(cls) -> 'ResourceSnapshot':
        """Capture current system resource usage."""
        # Get system-wide metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk_io = psutil.disk_io_counters()
        network_io = psutil.net_io_counters()
        
        return cls(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_mb=memory.used / (1024 * 1024),
            disk_io_read_mb=(disk_io.read_bytes if disk_io else 0) / (1024 * 1024),
            disk_io_write_mb=(disk_io.write_bytes if disk_io else 0) / (1024 * 1024),
            network_sent_mb=(network_io.bytes_sent if network_io else 0) / (1024 * 1024),
            network_recv_mb=(network_io.bytes_recv if network_io else 0) / (1024 * 1024),
            process_count=len(psutil.pids())
        )


@dataclass
class StartupSession:
    """Complete startup session data."""
    session_id: str
    start_time: float
    end_time: Optional[float] = None
    success: bool = False
    total_duration: Optional[float] = None
    phase_timings: Dict[str, TimingMetric] = field(default_factory=dict)
    resource_snapshots: List[ResourceSnapshot] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def finish(self, success: bool = True):
        """Mark the session as finished."""
        self.end_time = time.time()
        self.success = success
        self.total_duration = self.end_time - self.start_time


@dataclass
class PerformanceStats:
    """Aggregated performance statistics."""
    total_sessions: int
    success_rate: float
    average_duration: float
    median_duration: float
    min_duration: float
    max_duration: float
    phase_averages: Dict[str, float]
    error_frequency: Dict[str, int]
    trend_direction: str  # "improving", "degrading", "stable"
    last_updated: str


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system.
    
    Features:
    - Timing metrics collection for each startup phase
    - Success/failure rate tracking with trend analysis
    - Resource usage monitoring during startup
    - Historical data storage and analysis
    - Performance optimization suggestions
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path] = "logs/performance",
        max_sessions: int = 1000,
        resource_sampling_interval: float = 1.0
    ):
        """
        Initialize performance monitor.
        
        Args:
            data_dir: Directory to store performance data
            max_sessions: Maximum number of sessions to keep in memory
            resource_sampling_interval: Interval for resource sampling in seconds
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_sessions = max_sessions
        self.resource_sampling_interval = resource_sampling_interval
        
        # Current session tracking
        self.current_session: Optional[StartupSession] = None
        self.active_timings: Dict[str, TimingMetric] = {}
        
        # Historical data
        self.sessions: deque = deque(maxlen=max_sessions)
        
        # Resource monitoring
        self.resource_monitor_active = False
        self.resource_monitor_thread: Optional[threading.Thread] = None
        
        # Load existing data
        self._load_historical_data()
    
    def start_session(self, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Start a new startup session.
        
        Args:
            metadata: Optional metadata for the session
            
        Returns:
            Session ID
        """
        session_id = f"session_{int(time.time())}_{id(self)}"
        
        self.current_session = StartupSession(
            session_id=session_id,
            start_time=time.time(),
            metadata=metadata or {}
        )
        
        # Start resource monitoring
        self._start_resource_monitoring()
        
        return session_id
    
    def finish_session(self, success: bool = True):
        """
        Finish the current startup session.
        
        Args:
            success: Whether the startup was successful
        """
        if not self.current_session:
            return
        
        # Stop resource monitoring
        self._stop_resource_monitoring()
        
        # Finish the session
        self.current_session.finish(success)
        
        # Add to historical data
        self.sessions.append(self.current_session)
        
        # Save to disk
        self._save_session_data(self.current_session)
        
        # Clear current session
        self.current_session = None
        self.active_timings.clear()
    
    def start_timing(
        self,
        operation: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start timing an operation.
        
        Args:
            operation: Name of the operation
            metadata: Optional metadata for the timing
            
        Returns:
            Timing ID
        """
        timing = TimingMetric(
            operation=operation,
            start_time=time.time(),
            metadata=metadata or {}
        )
        
        timing_id = f"{operation}_{int(time.time())}"
        self.active_timings[timing_id] = timing
        
        # Add to current session if active
        if self.current_session:
            self.current_session.phase_timings[operation] = timing
        
        return timing_id
    
    def finish_timing(
        self,
        timing_id: str,
        success: bool = True,
        error_message: Optional[str] = None
    ):
        """
        Finish timing an operation.
        
        Args:
            timing_id: ID returned from start_timing
            success: Whether the operation was successful
            error_message: Error message if operation failed
        """
        if timing_id not in self.active_timings:
            return
        
        timing = self.active_timings[timing_id]
        timing.finish(success, error_message)
        
        # Add error to current session if failed
        if not success and self.current_session and error_message:
            self.current_session.errors.append(error_message)
        
        # Remove from active timings
        del self.active_timings[timing_id]
    
    def time_operation(self, operation: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Context manager for timing operations.
        
        Args:
            operation: Name of the operation
            metadata: Optional metadata for the timing
            
        Usage:
            with monitor.time_operation("environment_validation"):
                # Do validation work
                pass
        """
        return TimingContext(self, operation, metadata)
    
    def record_error(self, error_message: str, operation: Optional[str] = None):
        """
        Record an error during startup.
        
        Args:
            error_message: Error message
            operation: Operation where error occurred
        """
        if self.current_session:
            error_entry = error_message
            if operation:
                error_entry = f"{operation}: {error_message}"
            self.current_session.errors.append(error_entry)
    
    def get_performance_stats(self, days: int = 30) -> PerformanceStats:
        """
        Get aggregated performance statistics.
        
        Args:
            days: Number of days to include in statistics
            
        Returns:
            Performance statistics
        """
        cutoff_time = time.time() - (days * 24 * 3600)
        recent_sessions = [
            session for session in self.sessions
            if session.start_time >= cutoff_time and session.total_duration is not None
        ]
        
        if not recent_sessions:
            return PerformanceStats(
                total_sessions=0,
                success_rate=0.0,
                average_duration=0.0,
                median_duration=0.0,
                min_duration=0.0,
                max_duration=0.0,
                phase_averages={},
                error_frequency={},
                trend_direction="stable",
                last_updated=datetime.now().isoformat()
            )
        
        # Calculate basic stats
        durations = [session.total_duration for session in recent_sessions]
        successful_sessions = [session for session in recent_sessions if session.success]
        
        # Phase averages
        phase_averages = {}
        for phase in StartupPhase:
            phase_durations = []
            for session in recent_sessions:
                if phase.value in session.phase_timings:
                    timing = session.phase_timings[phase.value]
                    if timing.duration is not None:
                        phase_durations.append(timing.duration)
            
            if phase_durations:
                phase_averages[phase.value] = statistics.mean(phase_durations)
        
        # Error frequency
        error_frequency = defaultdict(int)
        for session in recent_sessions:
            for error in session.errors:
                # Extract error type from message
                error_type = error.split(":")[0] if ":" in error else error
                error_frequency[error_type] += 1
        
        # Trend analysis
        trend_direction = self._analyze_trend(recent_sessions)
        
        return PerformanceStats(
            total_sessions=len(recent_sessions),
            success_rate=len(successful_sessions) / len(recent_sessions),
            average_duration=statistics.mean(durations),
            median_duration=statistics.median(durations),
            min_duration=min(durations),
            max_duration=max(durations),
            phase_averages=phase_averages,
            error_frequency=dict(error_frequency),
            trend_direction=trend_direction,
            last_updated=datetime.now().isoformat()
        )
    
    def get_resource_usage_summary(self) -> Dict[str, Any]:
        """
        Get resource usage summary for current or last session.
        
        Returns:
            Resource usage summary
        """
        session = self.current_session or (self.sessions[-1] if self.sessions else None)
        
        if not session or not session.resource_snapshots:
            return {}
        
        snapshots = session.resource_snapshots
        
        return {
            "cpu_usage": {
                "average": statistics.mean([s.cpu_percent for s in snapshots]),
                "peak": max([s.cpu_percent for s in snapshots]),
                "samples": len(snapshots)
            },
            "memory_usage": {
                "average_percent": statistics.mean([s.memory_percent for s in snapshots]),
                "peak_percent": max([s.memory_percent for s in snapshots]),
                "average_mb": statistics.mean([s.memory_used_mb for s in snapshots]),
                "peak_mb": max([s.memory_used_mb for s in snapshots])
            },
            "disk_io": {
                "total_read_mb": max([s.disk_io_read_mb for s in snapshots]) - min([s.disk_io_read_mb for s in snapshots]),
                "total_write_mb": max([s.disk_io_write_mb for s in snapshots]) - min([s.disk_io_write_mb for s in snapshots])
            },
            "network_io": {
                "total_sent_mb": max([s.network_sent_mb for s in snapshots]) - min([s.network_sent_mb for s in snapshots]),
                "total_recv_mb": max([s.network_recv_mb for s in snapshots]) - min([s.network_recv_mb for s in snapshots])
            }
        }
    
    def _start_resource_monitoring(self):
        """Start background resource monitoring."""
        if self.resource_monitor_active:
            return
        
        self.resource_monitor_active = True
        self.resource_monitor_thread = threading.Thread(
            target=self._resource_monitor_loop,
            daemon=True
        )
        self.resource_monitor_thread.start()
    
    def _stop_resource_monitoring(self):
        """Stop background resource monitoring."""
        self.resource_monitor_active = False
        if self.resource_monitor_thread:
            self.resource_monitor_thread.join(timeout=2.0)
    
    def _resource_monitor_loop(self):
        """Background loop for resource monitoring."""
        while self.resource_monitor_active and self.current_session:
            try:
                snapshot = ResourceSnapshot.capture()
                self.current_session.resource_snapshots.append(snapshot)
                time.sleep(self.resource_sampling_interval)
            except Exception:
                # Ignore errors in resource monitoring
                pass
    
    def _analyze_trend(self, sessions: List[StartupSession]) -> str:
        """
        Analyze performance trend from recent sessions.
        
        Args:
            sessions: List of recent sessions
            
        Returns:
            Trend direction: "improving", "degrading", or "stable"
        """
        if len(sessions) < 5:
            return "stable"
        
        # Sort by start time (oldest first)
        sorted_sessions = sorted(sessions, key=lambda s: s.start_time)
        
        # Filter sessions with valid durations
        valid_sessions = [s for s in sorted_sessions if s.total_duration is not None]
        
        if len(valid_sessions) < 5:
            return "stable"
        
        # Split into two halves (older vs newer)
        mid_point = len(valid_sessions) // 2
        older_half = valid_sessions[:mid_point]
        newer_half = valid_sessions[mid_point:]
        
        # Calculate average durations
        older_avg = statistics.mean([s.total_duration for s in older_half])
        newer_avg = statistics.mean([s.total_duration for s in newer_half])
        
        # Determine trend (newer sessions compared to older sessions)
        improvement_threshold = 0.1  # 10% improvement
        if newer_avg < older_avg * (1 - improvement_threshold):
            return "improving"  # Newer sessions are faster
        elif newer_avg > older_avg * (1 + improvement_threshold):
            return "degrading"  # Newer sessions are slower
        else:
            return "stable"
    
    def _load_historical_data(self):
        """Load historical performance data from disk."""
        data_file = self.data_dir / "performance_history.json"
        
        if not data_file.exists():
            return
        
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Load sessions
            for session_data in data.get('sessions', []):
                session = StartupSession(**session_data)
                self.sessions.append(session)
        
        except (json.JSONDecodeError, KeyError, TypeError):
            # Ignore corrupted data
            pass
    
    def _save_session_data(self, session: StartupSession):
        """Save session data to disk."""
        # Save individual session
        session_file = self.data_dir / f"{session.session_id}.json"
        
        try:
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(session), f, indent=2, default=str)
        except Exception:
            # Ignore save errors
            pass
        
        # Update history file
        self._save_performance_history()
    
    def _save_performance_history(self):
        """Save performance history to disk."""
        data_file = self.data_dir / "performance_history.json"
        
        try:
            # Keep only recent sessions for the history file
            recent_sessions = list(self.sessions)[-100:]  # Last 100 sessions
            
            data = {
                'sessions': [asdict(session) for session in recent_sessions],
                'last_updated': datetime.now().isoformat()
            }
            
            with open(data_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
        
        except Exception:
            # Ignore save errors
            pass


class TimingContext:
    """Context manager for timing operations."""
    
    def __init__(
        self,
        monitor: PerformanceMonitor,
        operation: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.monitor = monitor
        self.operation = operation
        self.metadata = metadata
        self.timing_id: Optional[str] = None
    
    def __enter__(self):
        self.timing_id = self.monitor.start_timing(self.operation, self.metadata)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.timing_id:
            success = exc_type is None
            error_message = str(exc_val) if exc_val else None
            self.monitor.finish_timing(self.timing_id, success, error_message)


# Global performance monitor instance
_global_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor(**kwargs) -> PerformanceMonitor:
    """
    Get or create global performance monitor instance.
    
    Args:
        **kwargs: Additional arguments for monitor initialization
    
    Returns:
        PerformanceMonitor instance
    """
    global _global_monitor
    
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor(**kwargs)
    
    return _global_monitor