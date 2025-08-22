#!/usr/bin/env python3
"""
Wan2.2 Logging System - Comprehensive logging and debugging
Provides structured logging, debugging capabilities, and performance tracking
"""

import logging
import logging.handlers
import sys
import json
import time
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import threading
from datetime import datetime


@dataclass
class LogConfig:
    """Configuration for the logging system"""
    log_level: str = "INFO"
    log_dir: str = "logs"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    enable_console: bool = True
    enable_file: bool = True
    enable_json_logs: bool = True
    enable_performance_logs: bool = True
    enable_debug_mode: bool = False


@dataclass
class PerformanceMetrics:
    """Performance metrics for operations"""
    operation: str
    start_time: float
    end_time: float
    duration: float
    memory_before: float
    memory_after: float
    memory_peak: float
    success: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class StructuredLogger:
    """Structured logger with JSON output support"""
    
    def __init__(self, name: str, config: LogConfig):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, config.log_level.upper()))
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup logging handlers"""
        log_dir = Path(self.config.log_dir)
        log_dir.mkdir(exist_ok=True)
        
        # Console handler
        if self.config.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if self.config.enable_file:
            file_handler = logging.handlers.RotatingFileHandler(
                log_dir / f"{self.name}.log",
                maxBytes=self.config.max_file_size,
                backupCount=self.config.backup_count
            )
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
        
        # JSON handler
        if self.config.enable_json_logs:
            json_handler = logging.handlers.RotatingFileHandler(
                log_dir / f"{self.name}_structured.json",
                maxBytes=self.config.max_file_size,
                backupCount=self.config.backup_count
            )
            json_handler.setFormatter(JSONFormatter())
            self.logger.addHandler(json_handler)
    
    def log_structured(self, level: str, message: str, **kwargs):
        """Log structured data"""
        extra_data = {
            'structured_data': kwargs,
            'timestamp': datetime.utcnow().isoformat(),
            'logger_name': self.name
        }
        
        log_level = getattr(logging, level.upper())
        self.logger.log(log_level, message, extra=extra_data)
    
    def debug(self, message: str, **kwargs):
        """Debug level logging"""
        self.log_structured('DEBUG', message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Info level logging"""
        self.log_structured('INFO', message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Warning level logging"""
        self.log_structured('WARNING', message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Error level logging"""
        self.log_structured('ERROR', message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Critical level logging"""
        self.log_structured('CRITICAL', message, **kwargs)
c
lass JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add structured data if available
        if hasattr(record, 'structured_data'):
            log_entry['data'] = record.structured_data
        
        # Add exception info if available
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        return json.dumps(log_entry)


class PerformanceTracker:
    """Tracks performance metrics for operations"""
    
    def __init__(self, logger: StructuredLogger):
        self.logger = logger
        self.active_operations = {}
        self.completed_operations = []
        self._lock = threading.Lock()
    
    @contextmanager
    def track_operation(self, operation_name: str, **metadata):
        """Context manager for tracking operation performance"""
        start_time = time.time()
        memory_before = self._get_memory_usage()
        
        operation_id = f"{operation_name}_{int(start_time * 1000)}"
        
        with self._lock:
            self.active_operations[operation_id] = {
                'name': operation_name,
                'start_time': start_time,
                'memory_before': memory_before,
                'metadata': metadata
            }
        
        try:
            self.logger.debug(f"Starting operation: {operation_name}", 
                            operation_id=operation_id, **metadata)
            yield operation_id
            success = True
            error = None
        except Exception as e:
            success = False
            error = str(e)
            self.logger.error(f"Operation failed: {operation_name}", 
                            operation_id=operation_id, error=error, **metadata)
            raise
        finally:
            end_time = time.time()
            memory_after = self._get_memory_usage()
            duration = end_time - start_time
            
            with self._lock:
                if operation_id in self.active_operations:
                    op_data = self.active_operations.pop(operation_id)
                    
                    metrics = PerformanceMetrics(
                        operation=operation_name,
                        start_time=start_time,
                        end_time=end_time,
                        duration=duration,
                        memory_before=op_data['memory_before'],
                        memory_after=memory_after,
                        memory_peak=max(op_data['memory_before'], memory_after),
                        success=success,
                        error=error,
                        metadata=metadata
                    )
                    
                    self.completed_operations.append(metrics)
                    
                    self.logger.info(f"Operation completed: {operation_name}",
                                   operation_id=operation_id,
                                   duration=duration,
                                   success=success,
                                   memory_delta=memory_after - op_data['memory_before'],
                                   **metadata)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    def get_operation_stats(self) -> Dict[str, Any]:
        """Get statistics for completed operations"""
        if not self.completed_operations:
            return {}
        
        operations_by_name = {}
        for op in self.completed_operations:
            if op.operation not in operations_by_name:
                operations_by_name[op.operation] = []
            operations_by_name[op.operation].append(op)
        
        stats = {}
        for op_name, ops in operations_by_name.items():
            durations = [op.duration for op in ops]
            memory_deltas = [op.memory_after - op.memory_before for op in ops]
            success_count = sum(1 for op in ops if op.success)
            
            stats[op_name] = {
                'total_count': len(ops),
                'success_count': success_count,
                'failure_count': len(ops) - success_count,
                'success_rate': success_count / len(ops) if ops else 0,
                'avg_duration': sum(durations) / len(durations) if durations else 0,
                'min_duration': min(durations) if durations else 0,
                'max_duration': max(durations) if durations else 0,
                'avg_memory_delta': sum(memory_deltas) / len(memory_deltas) if memory_deltas else 0,
                'total_memory_delta': sum(memory_deltas)
            }
        
        return stats


class DebugContext:
    """Debug context for detailed debugging information"""
    
    def __init__(self, logger: StructuredLogger):
        self.logger = logger
        self.debug_data = {}
        self.call_stack = []
        self._lock = threading.Lock()
    
    @contextmanager
    def debug_scope(self, scope_name: str, **context):
        """Create a debug scope with context"""
        with self._lock:
            self.call_stack.append(scope_name)
            scope_id = f"{scope_name}_{len(self.call_stack)}"
            self.debug_data[scope_id] = {
                'scope': scope_name,
                'context': context,
                'start_time': time.time(),
                'variables': {},
                'checkpoints': []
            }
        
        try:
            self.logger.debug(f"Entering debug scope: {scope_name}", 
                            scope_id=scope_id, **context)
            yield DebugScope(self, scope_id)
        finally:
            with self._lock:
                if scope_id in self.debug_data:
                    scope_data = self.debug_data[scope_id]
                    duration = time.time() - scope_data['start_time']
                    
                    self.logger.debug(f"Exiting debug scope: {scope_name}",
                                    scope_id=scope_id,
                                    duration=duration,
                                    checkpoints=len(scope_data['checkpoints']),
                                    variables=len(scope_data['variables']))
                
                if self.call_stack and self.call_stack[-1] == scope_name:
                    self.call_stack.pop()
    
    def get_debug_summary(self) -> Dict[str, Any]:
        """Get summary of debug information"""
        with self._lock:
            return {
                'active_scopes': len(self.call_stack),
                'call_stack': self.call_stack.copy(),
                'total_scopes': len(self.debug_data),
                'debug_data_keys': list(self.debug_data.keys())
            }


class DebugScope:
    """Individual debug scope for tracking variables and checkpoints"""
    
    def __init__(self, debug_context: DebugContext, scope_id: str):
        self.debug_context = debug_context
        self.scope_id = scope_id
    
    def set_variable(self, name: str, value: Any):
        """Set a debug variable"""
        with self.debug_context._lock:
            if self.scope_id in self.debug_context.debug_data:
                # Convert complex objects to string representation
                if isinstance(value, (dict, list, tuple)):
                    str_value = json.dumps(value, default=str, indent=2)[:1000]
                else:
                    str_value = str(value)[:1000]
                
                self.debug_context.debug_data[self.scope_id]['variables'][name] = str_value
    
    def checkpoint(self, message: str, **data):
        """Add a debug checkpoint"""
        with self.debug_context._lock:
            if self.scope_id in self.debug_context.debug_data:
                checkpoint = {
                    'timestamp': time.time(),
                    'message': message,
                    'data': data
                }
                self.debug_context.debug_data[self.scope_id]['checkpoints'].append(checkpoint)
                
                self.debug_context.logger.debug(f"Debug checkpoint: {message}",
                                              scope_id=self.scope_id, **data)


class Wan22LoggingSystem:
    """Main logging system for Wan2.2 compatibility"""
    
    def __init__(self, config: Optional[LogConfig] = None):
        self.config = config or LogConfig()
        self.loggers = {}
        self.performance_tracker = None
        self.debug_context = None
        
        # Create logs directory
        Path(self.config.log_dir).mkdir(exist_ok=True)
        
        # Initialize main logger
        self.main_logger = self.get_logger("wan22_compatibility")
        
        # Initialize performance tracking if enabled
        if self.config.enable_performance_logs:
            self.performance_tracker = PerformanceTracker(self.main_logger)
        
        # Initialize debug context if enabled
        if self.config.enable_debug_mode:
            self.debug_context = DebugContext(self.main_logger)
        
        self.main_logger.info("Wan2.2 Logging System initialized",
                            config=asdict(self.config))
    
    def get_logger(self, name: str) -> StructuredLogger:
        """Get or create a structured logger"""
        if name not in self.loggers:
            self.loggers[name] = StructuredLogger(name, self.config)
        return self.loggers[name]
    
    def track_performance(self, operation_name: str, **metadata):
        """Get performance tracking context manager"""
        if self.performance_tracker:
            return self.performance_tracker.track_operation(operation_name, **metadata)
        else:
            # Return a no-op context manager
            @contextmanager
            def no_op():
                yield None
            return no_op()
    
    def debug_scope(self, scope_name: str, **context):
        """Get debug scope context manager"""
        if self.debug_context:
            return self.debug_context.debug_scope(scope_name, **context)
        else:
            # Return a no-op context manager
            @contextmanager
            def no_op():
                yield None
            return no_op()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get logging system status"""
        status = {
            'config': asdict(self.config),
            'active_loggers': list(self.loggers.keys()),
            'log_directory': str(Path(self.config.log_dir).absolute())
        }
        
        if self.performance_tracker:
            status['performance_stats'] = self.performance_tracker.get_operation_stats()
        
        if self.debug_context:
            status['debug_summary'] = self.debug_context.get_debug_summary()
        
        return status
    
    def cleanup(self):
        """Cleanup logging system"""
        self.main_logger.info("Cleaning up logging system")
        
        # Close all handlers
        for logger in self.loggers.values():
            for handler in logger.logger.handlers[:]:
                handler.close()
                logger.logger.removeHandler(handler)
        
        # Save final performance stats
        if self.performance_tracker:
            stats_file = Path(self.config.log_dir) / "performance_stats.json"
            try:
                with open(stats_file, 'w') as f:
                    json.dump(self.performance_tracker.get_operation_stats(), f, indent=2)
                self.main_logger.info(f"Saved performance stats to {stats_file}")
            except Exception as e:
                self.main_logger.error(f"Failed to save performance stats: {e}")


# Global logging system instance
_logging_system: Optional[Wan22LoggingSystem] = None


def get_logging_system(config: Optional[LogConfig] = None) -> Wan22LoggingSystem:
    """Get or create the global logging system"""
    global _logging_system
    
    if _logging_system is None:
        _logging_system = Wan22LoggingSystem(config)
    
    return _logging_system


def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger"""
    return get_logging_system().get_logger(name)


def cleanup_logging_system():
    """Cleanup the global logging system"""
    global _logging_system
    
    if _logging_system is not None:
        _logging_system.cleanup()
        _logging_system = None