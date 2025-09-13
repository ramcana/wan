"""
Enhanced Logging System
Provides configurable logging with multiple levels, file rotation, and structured output.
"""

import os
import sys
import json
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from enum import Enum


class LogLevel(Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogCategory(Enum):
    """Log categories for structured logging."""
    SYSTEM = "system"
    INSTALLATION = "installation"
    HARDWARE = "hardware"
    NETWORK = "network"
    CONFIGURATION = "configuration"
    VALIDATION = "validation"
    USER_ACTION = "user_action"


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging."""
    
    def format(self, record):
        # Create structured log entry
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, 'category'):
            log_entry['category'] = record.category
        if hasattr(record, 'phase'):
            log_entry['phase'] = record.phase
        if hasattr(record, 'progress'):
            log_entry['progress'] = record.progress
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry, ensure_ascii=False)


class ColoredConsoleFormatter(logging.Formatter):
    """Colored console formatter for better readability."""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        # Add color to level name
        level_color = self.COLORS.get(record.levelname, '')
        reset_color = self.COLORS['RESET']
        
        # Format the message
        formatted = super().format(record)
        
        # Apply color only to the level name
        if level_color:
            formatted = formatted.replace(
                record.levelname,
                f"{level_color}{record.levelname}{reset_color}"
            )
        
        return formatted


class InstallationLogger:
    """
    Enhanced logging system for the installation process.
    """
    
    def __init__(self, installation_path: str, log_level: str = "INFO", 
                 enable_console: bool = True, enable_structured: bool = True):
        self.installation_path = Path(installation_path)
        self.logs_dir = self.installation_path / "logs"
        self.log_level = getattr(logging, log_level.upper())
        self.enable_console = enable_console
        self.enable_structured = enable_structured
        
        # Create logs directory
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize loggers
        self.loggers = {}
        self._setup_root_logger()
        self._setup_specialized_loggers()
    
    def _setup_root_logger(self) -> None:
        """Setup the root logger."""
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Main installation log file (rotating)
        main_handler = logging.handlers.RotatingFileHandler(
            self.logs_dir / "installation.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        main_handler.setLevel(logging.DEBUG)
        main_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        main_handler.setFormatter(main_formatter)
        root_logger.addHandler(main_handler)
        
        # Error log file
        error_handler = logging.FileHandler(
            self.logs_dir / "errors.log",
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(main_formatter)
        root_logger.addHandler(error_handler)
        
        # Console handler
        if self.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.log_level)
            
            # Use colored formatter for console
            console_formatter = ColoredConsoleFormatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)
            root_logger.addHandler(console_handler)
        
        # Structured log file (JSON format)
        if self.enable_structured:
            structured_handler = logging.FileHandler(
                self.logs_dir / "structured.log",
                encoding='utf-8'
            )
            structured_handler.setLevel(logging.DEBUG)
            structured_handler.setFormatter(StructuredFormatter())
            root_logger.addHandler(structured_handler)
    
    def _setup_specialized_loggers(self) -> None:
        """Setup specialized loggers for different components."""
        
        # Performance logger
        perf_logger = logging.getLogger('performance')
        perf_handler = logging.FileHandler(
            self.logs_dir / "performance.log",
            encoding='utf-8'
        )
        perf_formatter = logging.Formatter(
            '%(asctime)s - %(message)s'
        )
        perf_handler.setFormatter(perf_formatter)
        perf_logger.addHandler(perf_handler)
        perf_logger.setLevel(logging.INFO)
        self.loggers['performance'] = perf_logger
        
        # Network logger
        network_logger = logging.getLogger('network')
        network_handler = logging.FileHandler(
            self.logs_dir / "network.log",
            encoding='utf-8'
        )
        network_handler.setFormatter(perf_formatter)
        network_logger.addHandler(network_handler)
        network_logger.setLevel(logging.DEBUG)
        self.loggers['network'] = network_logger
        
        # User actions logger
        user_logger = logging.getLogger('user_actions')
        user_handler = logging.FileHandler(
            self.logs_dir / "user_actions.log",
            encoding='utf-8'
        )
        user_handler.setFormatter(perf_formatter)
        user_logger.addHandler(user_handler)
        user_logger.setLevel(logging.INFO)
        self.loggers['user_actions'] = user_logger
    
    def get_logger(self, name: str = None) -> logging.Logger:
        """Get a logger instance."""
        if name and name in self.loggers:
            return self.loggers[name]
        return logging.getLogger(name or __name__)
    
    def log_structured(self, level: str, message: str, category: LogCategory = None,
                      phase: str = None, progress: float = None, **kwargs) -> None:
        """Log a structured message with additional metadata."""
        logger = self.get_logger()
        
        # Create log record
        record = logger.makeRecord(
            logger.name, getattr(logging, level.upper()), 
            __file__, 0, message, (), None
        )
        
        # Add structured data
        if category:
            record.category = category.value
        if phase:
            record.phase = phase
        if progress is not None:
            record.progress = progress
        
        # Add custom fields
        for key, value in kwargs.items():
            setattr(record, key, value)
        
        logger.handle(record)
    
    def log_performance(self, operation: str, duration: float, 
                       details: Dict[str, Any] = None) -> None:
        """Log performance metrics."""
        perf_logger = self.get_logger('performance')
        
        message = f"PERF: {operation} took {duration:.3f}s"
        if details:
            message += f" - {json.dumps(details)}"
        
        perf_logger.info(message)
    
    def log_network_activity(self, operation: str, url: str = None, 
                           status: str = None, size: int = None) -> None:
        """Log network activity."""
        network_logger = self.get_logger('network')
        
        message = f"NET: {operation}"
        if url:
            message += f" - {url}"
        if status:
            message += f" - Status: {status}"
        if size:
            message += f" - Size: {size} bytes"
        
        network_logger.info(message)
    
    def log_user_action(self, action: str, details: Dict[str, Any] = None) -> None:
        """Log user actions."""
        user_logger = self.get_logger('user_actions')
        
        message = f"USER: {action}"
        if details:
            message += f" - {json.dumps(details)}"
        
        user_logger.info(message)
    
    def log_phase_start(self, phase: str, description: str = None) -> None:
        """Log the start of an installation phase."""
        self.log_structured(
            'INFO', 
            f"Starting phase: {phase}" + (f" - {description}" if description else ""),
            category=LogCategory.INSTALLATION,
            phase=phase,
            progress=0.0
        )
    
    def log_phase_end(self, phase: str, success: bool = True, 
                     duration: float = None) -> None:
        """Log the end of an installation phase."""
        status = "completed" if success else "failed"
        message = f"Phase {phase} {status}"
        
        if duration:
            message += f" in {duration:.2f}s"
        
        level = 'INFO' if success else 'ERROR'
        self.log_structured(
            level, message,
            category=LogCategory.INSTALLATION,
            phase=phase,
            progress=1.0 if success else None
        )
    
    def log_progress(self, phase: str, progress: float, task: str) -> None:
        """Log progress updates."""
        self.log_structured(
            'DEBUG',
            f"Progress: {task}",
            category=LogCategory.INSTALLATION,
            phase=phase,
            progress=progress
        )
    
    def log_error_with_context(self, error: Exception, context: Dict[str, Any]) -> None:
        """Log an error with additional context."""
        logger = self.get_logger()
        
        logger.error(
            f"Error: {str(error)}",
            extra={
                'category': LogCategory.SYSTEM.value,
                'context': context,
                'error_type': type(error).__name__
            },
            exc_info=True
        )
    
    def create_log_summary(self) -> Dict[str, Any]:
        """Create a summary of log files and their contents."""
        summary = {
            'log_directory': str(self.logs_dir),
            'files': [],
            'statistics': {
                'total_size': 0,
                'error_count': 0,
                'warning_count': 0
            }
        }
        
        # Analyze log files
        for log_file in self.logs_dir.glob("*.log"):
            file_info = {
                'name': log_file.name,
                'size': log_file.stat().st_size,
                'modified': datetime.fromtimestamp(log_file.stat().st_mtime).isoformat(),
                'lines': 0,
                'errors': 0,
                'warnings': 0
            }
            
            # Count lines and log levels
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        file_info['lines'] += 1
                        if 'ERROR' in line:
                            file_info['errors'] += 1
                        elif 'WARNING' in line:
                            file_info['warnings'] += 1
            except Exception:
                pass  # Skip files that can't be read
            
            summary['files'].append(file_info)
            summary['statistics']['total_size'] += file_info['size']
            summary['statistics']['error_count'] += file_info['errors']
            summary['statistics']['warning_count'] += file_info['warnings']
        
        return summary
    
    def cleanup_old_logs(self, keep_days: int = 7) -> int:
        """Clean up old log files."""
        cutoff_time = datetime.now().timestamp() - (keep_days * 24 * 60 * 60)
        deleted_count = 0
        
        for log_file in self.logs_dir.glob("*.log.*"):  # Rotated log files
            if log_file.stat().st_mtime < cutoff_time:
                try:
                    log_file.unlink()
                    deleted_count += 1
                except Exception as e:
                    logging.warning(f"Failed to delete old log file {log_file}: {e}")
        
        return deleted_count
    
    def export_logs(self, output_path: str, include_structured: bool = True) -> bool:
        """Export logs to a zip file for support purposes."""
        import zipfile
        
        try:
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for log_file in self.logs_dir.glob("*.log"):
                    # Skip structured logs if not requested
                    if not include_structured and log_file.name == "structured.log":
                        continue
                    
                    zipf.write(log_file, log_file.name)
                
                # Add log summary
                summary = self.create_log_summary()
                zipf.writestr("log_summary.json", json.dumps(summary, indent=2))
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to export logs: {e}")
            return False


def setup_installation_logging(installation_path: str, log_level: str = "INFO",
                             enable_console: bool = True, 
                             enable_structured: bool = True) -> InstallationLogger:
    """
    Setup the installation logging system.
    
    Args:
        installation_path: Path to the installation directory
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        enable_console: Whether to enable console output
        enable_structured: Whether to enable structured JSON logging
        
    Returns:
        InstallationLogger instance
    """
    return InstallationLogger(
        installation_path=installation_path,
        log_level=log_level,
        enable_console=enable_console,
        enable_structured=enable_structured
    )


def setup_logging(level: str = "INFO", log_file: str = "installation.log") -> None:
    """
    Simple logging setup function for compatibility.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Log file name
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Configure basic logging
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
