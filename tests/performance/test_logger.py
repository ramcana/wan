"""
Unit tests for the startup manager logging system.
"""

import pytest
import json
import tempfile
import logging
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from scripts.startup_manager.logger import (
    StartupLogger,
    LogLevel,
    LogEntry,
    ColoredFormatter,
    JSONFormatter,
    get_logger,
    configure_logging
)


class TestLogEntry:
    """Test LogEntry dataclass."""
    
    def test_log_entry_creation(self):
        """Test creating a log entry."""
        entry = LogEntry(
            timestamp="2023-01-01T12:00:00",
            level="INFO",
            logger_name="test",
            message="Test message"
        )
        
        assert entry.timestamp == "2023-01-01T12:00:00"
        assert entry.level == "INFO"
        assert entry.logger_name == "test"
        assert entry.message == "Test message"
        assert entry.extra_data is None


class TestColoredFormatter:
    """Test ColoredFormatter class."""
    
    def test_colored_formatter_creation(self):
        """Test creating a colored formatter."""
        formatter = ColoredFormatter()
        assert formatter is not None
    
    def test_format_info_message(self):
        """Test formatting an info message."""
        formatter = ColoredFormatter("%(levelname)s - %(message)s")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        formatted = formatter.format(record)
        assert "INFO" in formatted
        assert "Test message" in formatted
    
    def test_format_error_message(self):
        """Test formatting an error message with color."""
        formatter = ColoredFormatter("%(levelname)s - %(message)s")
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="Error message",
            args=(),
            exc_info=None
        )
        
        formatted = formatter.format(record)
        assert "ERROR" in formatted
        assert "Error message" in formatted


class TestJSONFormatter:
    """Test JSONFormatter class."""
    
    def test_json_formatter_creation(self):
        """Test creating a JSON formatter."""
        formatter = JSONFormatter()
        assert formatter is not None
    
    def test_format_json_message(self):
        """Test formatting a message as JSON."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        formatted = formatter.format(record)
        data = json.loads(formatted)
        
        assert data["level"] == "INFO"
        assert data["logger_name"] == "test"
        assert data["message"] == "Test message"
        assert "timestamp" in data
    
    def test_format_json_with_extra_data(self):
        """Test formatting JSON with extra data."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.extra_data = {"key": "value"}
        
        formatted = formatter.format(record)
        data = json.loads(formatted)
        
        assert data["extra_data"] == {"key": "value"}


class TestStartupLogger:
    """Test StartupLogger class."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_dir = Path(self.temp_dir) / "logs"
    
    def teardown_method(self):
        """Cleanup test environment."""
        import shutil
shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_logger_creation(self):
        """Test creating a startup logger."""
        logger = StartupLogger(log_dir=self.log_dir)
        assert logger.name == "StartupManager"
        assert logger.log_dir == self.log_dir
        assert self.log_dir.exists()
    
    def test_logger_with_custom_settings(self):
        """Test creating logger with custom settings."""
        logger = StartupLogger(
            name="TestLogger",
            log_dir=self.log_dir,
            console_level="DEBUG",
            file_level="WARNING",
            max_file_size=1024,
            backup_count=3
        )
        
        assert logger.name == "TestLogger"
        assert logger.console_level == logging.DEBUG
        assert logger.file_level == logging.WARNING
        assert logger.max_file_size == 1024
        assert logger.backup_count == 3
    
    def test_log_methods(self):
        """Test different log level methods."""
        logger = StartupLogger(log_dir=self.log_dir)
        
        # Test all log levels
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.critical("Critical message")
        
        # Check that log files were created
        log_files = logger.get_log_files()
        assert len(log_files) >= 2  # At least .log and .json files
    
    def test_log_with_extra_data(self):
        """Test logging with extra data."""
        logger = StartupLogger(log_dir=self.log_dir)
        
        extra_data = {"key": "value", "number": 42}
        logger.info("Test message", extra_data=extra_data)
        
        # Check JSON log contains extra data
        json_files = list(self.log_dir.glob("*.json"))
        assert len(json_files) > 0
        
        with open(json_files[0], 'r') as f:
            log_line = f.readline()
            data = json.loads(log_line)
            assert data["extra_data"] == extra_data
    
    def test_log_startup_phase(self):
        """Test logging startup phase."""
        logger = StartupLogger(log_dir=self.log_dir)
        
        phase_details = {"step": 1, "description": "Validation"}
        logger.log_startup_phase("Environment Validation", phase_details)
        
        # Check that phase was logged
        json_files = list(self.log_dir.glob("*.json"))
        assert len(json_files) > 0
        
        with open(json_files[0], 'r') as f:
            log_line = f.readline()
            data = json.loads(log_line)
            assert "PHASE: Environment Validation" in data["message"]
            assert data["extra_data"]["phase"] == "Environment Validation"
    
    def test_log_error_with_context(self):
        """Test logging error with context."""
        logger = StartupLogger(log_dir=self.log_dir)
        
        try:
            raise ValueError("Test error")
        except ValueError as e:
            context = {"operation": "test", "step": 1}
            logger.log_error_with_context(e, context)
        
        # Check that error was logged with context
        json_files = list(self.log_dir.glob("*.json"))
        assert len(json_files) > 0
        
        with open(json_files[0], 'r') as f:
            log_line = f.readline()
            data = json.loads(log_line)
            assert "ERROR: Test error" in data["message"]
            assert data["extra_data"]["error_context"] == context
            assert data["extra_data"]["error_type"] == "ValueError"
    
    def test_log_performance_metric(self):
        """Test logging performance metrics."""
        logger = StartupLogger(log_dir=self.log_dir)
        
        details = {"cpu_usage": 50, "memory_usage": 100}
        logger.log_performance_metric("startup", 2.5, details)
        
        # Check that metric was logged
        json_files = list(self.log_dir.glob("*.json"))
        assert len(json_files) > 0
        
        with open(json_files[0], 'r') as f:
            log_line = f.readline()
            data = json.loads(log_line)
            assert "PERFORMANCE: startup completed in 2.50s" in data["message"]
            assert data["extra_data"]["metric_type"] == "performance"
            assert data["extra_data"]["duration"] == 2.5
    
    def test_log_system_info(self):
        """Test logging system information."""
        logger = StartupLogger(log_dir=self.log_dir)
        
        system_info = {"os": "Windows", "python_version": "3.8.0"}
        logger.log_system_info(system_info)
        
        # Check that system info was logged
        json_files = list(self.log_dir.glob("*.json"))
        assert len(json_files) > 0
        
        with open(json_files[0], 'r') as f:
            log_line = f.readline()
            data = json.loads(log_line)
            assert "SYSTEM_INFO: System information collected" in data["message"]
            assert data["extra_data"]["system_info"] == system_info
    
    def test_set_console_level(self):
        """Test dynamically changing console level."""
        logger = StartupLogger(log_dir=self.log_dir, console_level="INFO")
        
        # Change to DEBUG level
        logger.set_console_level("DEBUG")
        
        # Find console handler and check level
        console_handler = None
        for handler in logger.logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                console_handler = handler
                break
        
        assert console_handler is not None
        assert console_handler.level == logging.DEBUG
    
    def test_get_log_files(self):
        """Test getting list of log files."""
        logger = StartupLogger(log_dir=self.log_dir)
        
        # Log something to create files
        logger.info("Test message")
        
        log_files = logger.get_log_files()
        assert len(log_files) >= 2  # At least .log and .json files
        
        # Check file extensions
        extensions = {f.suffix for f in log_files}
        assert ".log" in extensions
        assert ".json" in extensions
    
    @patch('scripts.startup_manager.logger.datetime')
    def test_cleanup_old_logs(self, mock_datetime):
        """Test cleanup of old log files."""
        # Mock current time
        current_time = datetime(2023, 1, 15, 12, 0, 0)
        mock_datetime.now.return_value = current_time
        mock_datetime.fromtimestamp = datetime.fromtimestamp
        
        logger = StartupLogger(log_dir=self.log_dir, cleanup_days=7)
        
        # Create some old log files
        old_log = self.log_dir / "startup_20230101.log"
        old_json = self.log_dir / "startup_20230101.json"
        recent_log = self.log_dir / "startup_20230114.log"
        
        old_log.touch()
        old_json.touch()
        recent_log.touch()
        
        # Set modification times
        old_time = (current_time - timedelta(days=10)).timestamp()
        recent_time = (current_time - timedelta(days=1)).timestamp()
        
        import os
os.utime(old_log, (old_time, old_time))
        os.utime(old_json, (old_time, old_time))
        os.utime(recent_log, (recent_time, recent_time))
        
        # Trigger cleanup
        logger._cleanup_old_logs()
        
        # Check that old files were removed and recent files remain
        assert not old_log.exists()
        assert not old_json.exists()
        assert recent_log.exists()


class TestGlobalLogger:
    """Test global logger functions."""
    
    def setup_method(self):
        """Setup test environment."""
        # Reset global logger
        import scripts.startup_manager.logger
scripts.startup_manager.logger._global_logger = None
    
    def test_get_logger(self):
        """Test getting global logger instance."""
        logger1 = get_logger()
        logger2 = get_logger()
        
        # Should return same instance
        assert logger1 is logger2
        assert logger1.name == "StartupManager"
    
    def test_get_logger_with_custom_name(self):
        """Test getting logger with custom name."""
        logger = get_logger(name="CustomLogger")
        assert logger.name == "CustomLogger"
    
    def test_configure_logging(self):
        """Test configuring global logging."""
        logger = configure_logging(
            console_level="DEBUG",
            file_level="WARNING",
            verbose=False
        )
        
        assert logger.console_level == logging.DEBUG
        assert logger.file_level == logging.WARNING
    
    def test_configure_logging_verbose(self):
        """Test configuring logging with verbose mode."""
        logger = configure_logging(
            console_level="INFO",
            file_level="INFO",
            verbose=True
        )
        
        # Verbose should override levels to DEBUG
        assert logger.console_level == logging.DEBUG
        assert logger.file_level == logging.DEBUG


class TestLogRotation:
    """Test log rotation functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_dir = Path(self.temp_dir) / "logs"
    
    def teardown_method(self):
        """Cleanup test environment."""
        import shutil
shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_log_rotation_setup(self):
        """Test that log rotation is properly configured."""
        logger = StartupLogger(
            log_dir=self.log_dir,
            max_file_size=1024,  # Small size to trigger rotation
            backup_count=3
        )
        
        # Check that handlers are configured with rotation
        for handler in logger.logger.handlers:
            if hasattr(handler, 'maxBytes'):
                assert handler.maxBytes == 1024
                assert handler.backupCount == 3


if __name__ == "__main__":
    pytest.main([__file__])