"""
Structured logging configuration for the Model Orchestrator system.
"""

import json
import logging
import sys
import time
import uuid
from contextvars import ContextVar
from typing import Any, Dict, Optional

# Context variable to store correlation ID across async operations
correlation_id_var: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)


class StructuredFormatter(logging.Formatter):
    """Custom formatter that outputs structured JSON logs."""
    
    def __init__(self, include_extra: bool = True):
        super().__init__()
        self.include_extra = include_extra
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        # Base log structure
        message = record.getMessage()
        # Mask sensitive information in the message
        from .credential_manager import CredentialMasker
        masked_message = CredentialMasker.mask_sensitive_data(message)
        
        log_entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": masked_message,
        }
        
        # Add correlation ID if available
        correlation_id = getattr(record, 'correlation_id', None) or correlation_id_var.get()
        if correlation_id:
            log_entry["correlation_id"] = correlation_id
        
        # Add exception information if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info) if record.exc_info else None
            }
        
        # Add extra fields if enabled
        if self.include_extra:
            # Standard fields to exclude from extra
            standard_fields = {
                'name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 'filename',
                'module', 'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                'thread', 'threadName', 'processName', 'process', 'getMessage',
                'exc_info', 'exc_text', 'stack_info', 'correlation_id'
            }
            
            # Add any extra fields from the log record
            for key, value in record.__dict__.items():
                if key not in standard_fields and not key.startswith('_'):
                    # Sanitize sensitive information
                    if self._is_sensitive_field(key):
                        log_entry[key] = self._mask_sensitive_value(str(value))
                    else:
                        log_entry[key] = value
        
        return json.dumps(log_entry, default=str, ensure_ascii=False)
    
    def formatTime(self, record: logging.LogRecord, datefmt: Optional[str] = None) -> str:
        """Format timestamp in ISO format."""
        return time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(record.created)) + \
               f'.{int(record.msecs):03d}Z'
    
    def _is_sensitive_field(self, field_name: str) -> bool:
        """Check if a field contains sensitive information."""
        from .credential_manager import CredentialMasker
        return CredentialMasker._is_sensitive_key(field_name)
    
    def _mask_sensitive_value(self, value: str) -> str:
        """Mask sensitive values in logs."""
        from .credential_manager import CredentialMasker
        return CredentialMasker._mask_value(value)


class CorrelationIdFilter(logging.Filter):
    """Filter to add correlation ID to log records."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add correlation ID to the log record if not already present."""
        if not hasattr(record, 'correlation_id'):
            correlation_id = correlation_id_var.get()
            if correlation_id:
                record.correlation_id = correlation_id
        return True


class ModelOrchestratorLogger:
    """Centralized logger configuration for the Model Orchestrator."""
    
    def __init__(self, name: str = "model_orchestrator"):
        self.name = name
        self._logger = None
        self._configured = False
    
    def configure(
        self,
        level: str = "INFO",
        structured: bool = True,
        output_file: Optional[str] = None
    ) -> logging.Logger:
        """Configure the logger with structured output."""
        if self._configured:
            return self._logger
        
        logger = logging.getLogger(self.name)
        logger.setLevel(getattr(logging, level.upper()))
        
        # Clear any existing handlers
        logger.handlers.clear()
        
        # Create handler
        if output_file:
            handler = logging.FileHandler(output_file)
        else:
            handler = logging.StreamHandler(sys.stdout)
        
        # Set formatter
        if structured:
            formatter = StructuredFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        handler.setFormatter(formatter)
        
        # Add correlation ID filter
        handler.addFilter(CorrelationIdFilter())
        
        logger.addHandler(handler)
        
        # Prevent propagation to root logger to avoid duplicate logs
        logger.propagate = False
        
        self._logger = logger
        self._configured = True
        
        return logger
    
    def get_logger(self, name: Optional[str] = None) -> logging.Logger:
        """Get a logger instance."""
        if not self._configured:
            self.configure()
        
        if name:
            return logging.getLogger(f"{self.name}.{name}")
        return self._logger or logging.getLogger(self.name)


# Global logger instance
_orchestrator_logger = ModelOrchestratorLogger()


def configure_logging(
    level: str = "INFO",
    structured: bool = True,
    output_file: Optional[str] = None
) -> logging.Logger:
    """Configure logging for the Model Orchestrator."""
    return _orchestrator_logger.configure(level, structured, output_file)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a logger instance for the Model Orchestrator."""
    return _orchestrator_logger.get_logger(name)


def set_correlation_id(correlation_id: str):
    """Set the correlation ID for the current context."""
    correlation_id_var.set(correlation_id)


def get_correlation_id() -> Optional[str]:
    """Get the current correlation ID."""
    return correlation_id_var.get()


def generate_correlation_id() -> str:
    """Generate a new correlation ID."""
    return str(uuid.uuid4())


class LogContext:
    """Context manager for setting correlation ID and additional context."""
    
    def __init__(self, correlation_id: Optional[str] = None, **context):
        self.correlation_id = correlation_id or generate_correlation_id()
        self.context = context
        self.previous_correlation_id = None
    
    def __enter__(self):
        self.previous_correlation_id = correlation_id_var.get()
        correlation_id_var.set(self.correlation_id)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        correlation_id_var.set(self.previous_correlation_id)
    
    def log(self, level: str, message: str, **extra):
        """Log a message with the current context."""
        logger = get_logger()
        log_method = getattr(logger, level.lower())
        log_method(message, extra={**self.context, **extra})


# Convenience functions for common logging patterns

def log_operation_start(operation: str, **context):
    """Log the start of an operation."""
    logger = get_logger()
    logger.info(
        f"Starting {operation}",
        extra={"operation": operation, "phase": "start", **context}
    )


def log_operation_success(operation: str, duration: Optional[float] = None, **context):
    """Log successful completion of an operation."""
    logger = get_logger()
    extra = {"operation": operation, "phase": "success", **context}
    if duration is not None:
        extra["duration_seconds"] = duration
    logger.info(f"Completed {operation}", extra=extra)


def log_operation_failure(operation: str, error: Exception, duration: Optional[float] = None, **context):
    """Log failure of an operation."""
    logger = get_logger()
    extra = {
        "operation": operation,
        "phase": "failure",
        "error_type": type(error).__name__,
        "error_message": str(error),
        **context
    }
    if duration is not None:
        extra["duration_seconds"] = duration
    if hasattr(error, 'error_code'):
        extra["error_code"] = error.error_code.value
    logger.error(f"Failed {operation}", extra=extra, exc_info=True)


def log_retry_attempt(operation: str, attempt: int, error: Exception, delay: float, **context):
    """Log a retry attempt."""
    logger = get_logger()
    logger.warning(
        f"Retrying {operation} (attempt {attempt})",
        extra={
            "operation": operation,
            "phase": "retry",
            "attempt": attempt,
            "delay_seconds": delay,
            "error_type": type(error).__name__,
            "error_message": str(error),
            **context
        }
    )


class PerformanceTimer:
    """Context manager for measuring and logging operation performance."""
    
    def __init__(self, operation: str, logger: Optional[logging.Logger] = None, **context):
        self.operation = operation
        self.logger = logger or get_logger()
        self.context = context
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        log_operation_start(self.operation, **self.context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        
        if exc_type is None:
            log_operation_success(self.operation, duration=duration, **self.context)
        else:
            log_operation_failure(self.operation, exc_val, duration=duration, **self.context)
    
    @property
    def duration(self) -> Optional[float]:
        """Get the duration of the operation if completed."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


def performance_timer(operation: str, **context) -> PerformanceTimer:
    """Create a performance timer context manager."""
    return PerformanceTimer(operation, **context)


def log_metrics(metrics: Dict[str, Any], **context):
    """Log performance metrics."""
    logger = get_logger()
    logger.info(
        "Performance metrics",
        extra={"metrics": metrics, "type": "performance", **context}
    )