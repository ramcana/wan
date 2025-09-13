"""
Comprehensive logging system for the startup manager.

This module provides structured logging with multiple outputs, rotation policies,
and different verbosity levels for debugging and troubleshooting.
"""

import logging
import logging.handlers
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from enum import Enum
from dataclasses import dataclass, asdict
import colorama
from colorama import Fore, Back, Style

# Initialize colorama for Windows color support
colorama.init()


class LogLevel(Enum):
    """Log levels with color mapping."""
    DEBUG = ("DEBUG", Fore.CYAN)
    INFO = ("INFO", Fore.GREEN)
    WARNING = ("WARNING", Fore.YELLOW)
    ERROR = ("ERROR", Fore.RED)
    CRITICAL = ("CRITICAL", Fore.MAGENTA + Style.BRIGHT)


@dataclass
class LogEntry:
    """Structured log entry for JSON logging."""
    timestamp: str
    level: str
    logger_name: str
    message: str
    module: Optional[str] = None
    function: Optional[str] = None
    line_number: Optional[int] = None
    extra_data: Optional[Dict[str, Any]] = None
    error_type: Optional[str] = None
    stack_trace: Optional[str] = None


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to console output."""
    
    def __init__(self, fmt: str = None, datefmt: str = None):
        super().__init__(fmt, datefmt)
        self.level_colors = {level.name: level.value[1] for level in LogLevel}
    
    def format(self, record: logging.LogRecord) -> str:
        # Add color to level name
        level_color = self.level_colors.get(record.levelname, "")
        colored_level = f"{level_color}{record.levelname}{Style.RESET_ALL}"
        
        # Create a copy of the record to avoid modifying the original
        record_copy = logging.makeLogRecord(record.__dict__)
        record_copy.levelname = colored_level
        
        # Format the message
        formatted = super().format(record_copy)
        
        # Add color to the entire message for errors and critical
        if record.levelname in ['ERROR', 'CRITICAL']:
            formatted = f"{self.level_colors[record.levelname]}{formatted}{Style.RESET_ALL}"
        
        return formatted


class JSONFormatter(logging.Formatter):
    """Custom formatter for JSON output."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = LogEntry(
            timestamp=datetime.fromtimestamp(record.created).isoformat(),
            level=record.levelname,
            logger_name=record.name,
            message=record.getMessage(),
            module=getattr(record, 'module', None),
            function=getattr(record, 'funcName', None),
            line_number=getattr(record, 'lineno', None),
            extra_data=getattr(record, 'extra_data', None),
            error_type=getattr(record, 'error_type', None),
            stack_trace=getattr(record, 'stack_trace', None)
        )
        
        return json.dumps(asdict(log_entry), ensure_ascii=False)


class StartupLogger:
    """
    Comprehensive logging system for startup manager.
    
    Features:
    - Multiple output formats (console, file, JSON)
    - Log rotation and cleanup
    - Colored console output
    - Structured logging for programmatic analysis
    - Different verbosity levels
    """
    
    def __init__(
        self,
        name: str = "StartupManager",
        log_dir: Union[str, Path] = "logs",
        console_level: str = "INFO",
        file_level: str = "DEBUG",
        json_level: str = "INFO",
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        cleanup_days: int = 30
    ):
        """
        Initialize the logging system.
        
        Args:
            name: Logger name
            log_dir: Directory for log files
            console_level: Console logging level
            file_level: File logging level
            json_level: JSON logging level
            max_file_size: Maximum size per log file in bytes
            backup_count: Number of backup files to keep
            cleanup_days: Days after which to clean up old logs
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.console_level = getattr(logging, console_level.upper())
        self.file_level = getattr(logging, file_level.upper())
        self.json_level = getattr(logging, json_level.upper())
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        self.cleanup_days = cleanup_days
        
        # Create log directory
        self.log_dir.mkdir(exist_ok=True)
        
        # Initialize logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)  # Set to lowest level, handlers will filter
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # Setup handlers
        self._setup_console_handler()
        self._setup_file_handler()
        self._setup_json_handler()
        
        # Cleanup old logs
        self._cleanup_old_logs()
    
    def _setup_console_handler(self):
        """Setup colored console handler."""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.console_level)
        
        # Use colored formatter for console
        console_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        console_formatter = ColoredFormatter(
            fmt=console_format,
            datefmt="%H:%M:%S"
        )
        console_handler.setFormatter(console_formatter)
        
        self.logger.addHandler(console_handler)
    
    def _setup_file_handler(self):
        """Setup rotating file handler."""
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = self.log_dir / f"startup_{timestamp}.log"
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=self.max_file_size,
            backupCount=self.backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(self.file_level)
        
        # Use detailed format for file
        file_format = (
            "%(asctime)s | %(levelname)-8s | %(name)s | %(module)s:%(funcName)s:%(lineno)d | "
            "%(message)s"
        )
        file_formatter = logging.Formatter(
            fmt=file_format,
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        
        self.logger.addHandler(file_handler)
    
    def _setup_json_handler(self):
        """Setup JSON handler for structured logging."""
        timestamp = datetime.now().strftime("%Y%m%d")
        json_file = self.log_dir / f"startup_{timestamp}.json"
        
        json_handler = logging.handlers.RotatingFileHandler(
            json_file,
            maxBytes=self.max_file_size,
            backupCount=self.backup_count,
            encoding='utf-8'
        )
        json_handler.setLevel(self.json_level)
        
        json_formatter = JSONFormatter()
        json_handler.setFormatter(json_formatter)
        
        self.logger.addHandler(json_handler)
    
    def _cleanup_old_logs(self):
        """Clean up log files older than cleanup_days."""
        if not self.log_dir.exists():
            return
        
        cutoff_time = datetime.now().timestamp() - (self.cleanup_days * 24 * 3600)
        
        for log_file in self.log_dir.glob("startup_*.log*"):
            try:
                if log_file.stat().st_mtime < cutoff_time:
                    log_file.unlink()
            except (OSError, FileNotFoundError):
                pass  # Ignore errors during cleanup
        
        for json_file in self.log_dir.glob("startup_*.json*"):
            try:
                if json_file.stat().st_mtime < cutoff_time:
                    json_file.unlink()
            except (OSError, FileNotFoundError):
                pass  # Ignore errors during cleanup
    
    def debug(self, message: str, extra_data: Optional[Dict[str, Any]] = None, **kwargs):
        """Log debug message."""
        self._log(logging.DEBUG, message, extra_data, **kwargs)
    
    def info(self, message: str, extra_data: Optional[Dict[str, Any]] = None, **kwargs):
        """Log info message."""
        self._log(logging.INFO, message, extra_data, **kwargs)
    
    def warning(self, message: str, extra_data: Optional[Dict[str, Any]] = None, **kwargs):
        """Log warning message."""
        self._log(logging.WARNING, message, extra_data, **kwargs)
    
    def error(self, message: str, extra_data: Optional[Dict[str, Any]] = None, **kwargs):
        """Log error message."""
        self._log(logging.ERROR, message, extra_data, **kwargs)
    
    def critical(self, message: str, extra_data: Optional[Dict[str, Any]] = None, **kwargs):
        """Log critical message."""
        self._log(logging.CRITICAL, message, extra_data, **kwargs)
    
    def _log(self, level: int, message: str, extra_data: Optional[Dict[str, Any]] = None, **kwargs):
        """Internal logging method with extra data support."""
        extra = {
            'extra_data': extra_data,
            **kwargs
        }
        self.logger.log(level, message, extra=extra)
    
    def log_startup_phase(self, phase: str, details: Dict[str, Any]):
        """Log startup phase with structured data."""
        self.info(f"PHASE: {phase}", extra_data={"phase": phase, "details": details})
    
    def log_error_with_context(self, error: Exception, context: Dict[str, Any]):
        """Log error with full context for debugging."""
        import traceback
        
        self.error(
            f"ERROR: {str(error)}",
            extra_data={
                "error_context": context,
                "error_type": type(error).__name__,
                "stack_trace": traceback.format_exc()
            }
        )
    
    def log_performance_metric(self, operation: str, duration: float, details: Optional[Dict[str, Any]] = None):
        """Log performance metrics."""
        self.info(
            f"PERFORMANCE: {operation} completed in {duration:.2f}s",
            extra_data={
                "metric_type": "performance",
                "operation": operation,
                "duration": duration,
                "details": details or {}
            }
        )
    
    def log_system_info(self, info: Dict[str, Any]):
        """Log system information."""
        self.info(
            "SYSTEM_INFO: System information collected",
            extra_data={
                "metric_type": "system_info",
                "system_info": info
            }
        )
    
    def set_console_level(self, level: str):
        """Dynamically change console logging level."""
        new_level = getattr(logging, level.upper())
        for handler in self.logger.handlers:
            if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
                handler.setLevel(new_level)
                break
    
    def get_log_files(self) -> List[Path]:
        """Get list of current log files."""
        log_files = []
        if self.log_dir.exists():
            log_files.extend(self.log_dir.glob("startup_*.log*"))
            log_files.extend(self.log_dir.glob("startup_*.json*"))
        return sorted(log_files)


# Global logger instance
_global_logger: Optional[StartupLogger] = None


def get_logger(
    name: str = "StartupManager",
    **kwargs
) -> StartupLogger:
    """
    Get or create global logger instance.
    
    Args:
        name: Logger name
        **kwargs: Additional arguments for logger initialization
    
    Returns:
        StartupLogger instance
    """
    global _global_logger
    
    if _global_logger is None:
        _global_logger = StartupLogger(name=name, **kwargs)
    
    return _global_logger


def configure_logging(
    console_level: str = "INFO",
    file_level: str = "DEBUG",
    json_level: str = "INFO",
    log_dir: str = "logs",
    verbose: bool = False
) -> StartupLogger:
    """
    Configure global logging with specified parameters.
    
    Args:
        console_level: Console logging level
        file_level: File logging level
        json_level: JSON logging level
        log_dir: Log directory
        verbose: Enable verbose logging
    
    Returns:
        Configured StartupLogger instance
    """
    global _global_logger
    
    if verbose:
        console_level = "DEBUG"
        file_level = "DEBUG"
    
    _global_logger = StartupLogger(
        console_level=console_level,
        file_level=file_level,
        json_level=json_level,
        log_dir=log_dir
    )
    
    return _global_logger
