"""
Production logging system with structured logging and monitoring.
"""

import logging
import logging.handlers
import json
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from contextlib import contextmanager
import functools

from ..config.environment import get_config

class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
        if hasattr(record, 'task_id'):
            log_entry['task_id'] = record.task_id
        if hasattr(record, 'duration'):
            log_entry['duration'] = record.duration
        if hasattr(record, 'error_type'):
            log_entry['error_type'] = record.error_type
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        return json.dumps(log_entry)

class PerformanceLogger:
    """Logger for performance metrics and monitoring."""
    
    def __init__(self):
        self.logger = logging.getLogger('performance')
        self.config = get_config()
    
    def log_request(self, method: str, path: str, status_code: int, 
                   duration: float, user_id: Optional[str] = None) -> None:
        """Log HTTP request metrics."""
        self.logger.info(
            f"HTTP {method} {path} - {status_code}",
            extra={
                'method': method,
                'path': path,
                'status_code': status_code,
                'duration': duration,
                'user_id': user_id,
                'metric_type': 'http_request'
            }
        )
    
    def log_generation(self, task_id: str, model_type: str, 
                      duration: float, success: bool, 
                      error_type: Optional[str] = None) -> None:
        """Log generation task metrics."""
        status = "success" if success else "error"
        message = f"Generation {status} - {model_type} - {duration:.2f}s"
        
        extra = {
            'task_id': task_id,
            'model_type': model_type,
            'duration': duration,
            'success': success,
            'metric_type': 'generation'
        }
        
        if error_type:
            extra['error_type'] = error_type
        
        if success:
            self.logger.info(message, extra=extra)
        else:
            self.logger.error(message, extra=extra)
    
    def log_system_metrics(self, cpu_percent: float, ram_used_gb: float, 
                          gpu_percent: float, vram_used_mb: float) -> None:
        """Log system resource metrics."""
        self.logger.info(
            f"System metrics - CPU: {cpu_percent}%, RAM: {ram_used_gb}GB, "
            f"GPU: {gpu_percent}%, VRAM: {vram_used_mb}MB",
            extra={
                'cpu_percent': cpu_percent,
                'ram_used_gb': ram_used_gb,
                'gpu_percent': gpu_percent,
                'vram_used_mb': vram_used_mb,
                'metric_type': 'system_metrics'
            }
        )
    
    def log_queue_metrics(self, queue_size: int, processing_count: int, 
                         completed_count: int, failed_count: int) -> None:
        """Log queue metrics."""
        self.logger.info(
            f"Queue metrics - Size: {queue_size}, Processing: {processing_count}, "
            f"Completed: {completed_count}, Failed: {failed_count}",
            extra={
                'queue_size': queue_size,
                'processing_count': processing_count,
                'completed_count': completed_count,
                'failed_count': failed_count,
                'metric_type': 'queue_metrics'
            }
        )

class ErrorLogger:
    """Logger for error tracking and monitoring."""
    
    def __init__(self):
        self.logger = logging.getLogger('errors')
    
    def log_error(self, error: Exception, context: Optional[Dict[str, Any]] = None,
                 user_id: Optional[str] = None, request_id: Optional[str] = None) -> None:
        """Log error with context information."""
        extra = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'metric_type': 'error'
        }
        
        if context:
            extra.update(context)
        if user_id:
            extra['user_id'] = user_id
        if request_id:
            extra['request_id'] = request_id
        
        self.logger.error(
            f"Error: {type(error).__name__} - {str(error)}",
            exc_info=True,
            extra=extra
        )
    
    def log_validation_error(self, field: str, value: Any, message: str,
                           request_id: Optional[str] = None) -> None:
        """Log validation error."""
        self.logger.warning(
            f"Validation error - {field}: {message}",
            extra={
                'error_type': 'ValidationError',
                'field': field,
                'value': str(value),
                'message': message,
                'request_id': request_id,
                'metric_type': 'validation_error'
            }
        )
    
    def log_generation_error(self, task_id: str, model_type: str, 
                           error: Exception, prompt: Optional[str] = None) -> None:
        """Log generation-specific error."""
        extra = {
            'task_id': task_id,
            'model_type': model_type,
            'error_type': type(error).__name__,
            'metric_type': 'generation_error'
        }
        
        if prompt:
            extra['prompt'] = prompt[:100]  # Truncate long prompts
        
        self.logger.error(
            f"Generation error - Task {task_id} - {type(error).__name__}: {str(error)}",
            exc_info=True,
            extra=extra
        )

def setup_logging() -> None:
    """Set up logging configuration based on environment."""
    config = get_config()
    log_config = config.get_log_config()
    
    # Create log directory if it doesn't exist
    log_file = config.get("logging_settings", "log_file")
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.config.dictConfig(log_config)
    
    # Set up structured logging for production
    if config.is_production():
        # Add structured formatter to file handler
        for handler in logging.getLogger().handlers:
            if isinstance(handler, logging.handlers.RotatingFileHandler):
                handler.setFormatter(StructuredFormatter())
    
    logging.info(f"Logging configured for environment: {config.env.value}")

@contextmanager
def log_performance(operation: str, logger: Optional[logging.Logger] = None,
                   extra_context: Optional[Dict[str, Any]] = None):
    """Context manager for logging operation performance."""
    if logger is None:
        logger = logging.getLogger('performance')
    
    start_time = time.time()
    context = {'operation': operation}
    if extra_context:
        context.update(extra_context)
    
    try:
        yield context
        duration = time.time() - start_time
        logger.info(
            f"Operation completed: {operation} - {duration:.3f}s",
            extra={**context, 'duration': duration, 'success': True}
        )
    except Exception as e:
        duration = time.time() - start_time
        logger.error(
            f"Operation failed: {operation} - {duration:.3f}s - {type(e).__name__}: {str(e)}",
            exc_info=True,
            extra={**context, 'duration': duration, 'success': False, 'error_type': type(e).__name__}
        )
        raise

def log_function_performance(func):
    """Decorator for logging function performance."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger('performance')
        operation = f"{func.__module__}.{func.__name__}"
        
        with log_performance(operation, logger):
            return func(*args, **kwargs)
    
    return wrapper

def get_performance_logger() -> PerformanceLogger:
    """Get performance logger instance."""
    return PerformanceLogger()

def get_error_logger() -> ErrorLogger:
    """Get error logger instance."""
    return ErrorLogger()

# Global logger instances
performance_logger = PerformanceLogger()
error_logger = ErrorLogger()
