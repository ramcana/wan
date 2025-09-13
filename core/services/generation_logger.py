"""
Comprehensive logging and diagnostics system for video generation pipeline.

This module provides detailed logging capabilities for all stages of the video
generation process, including error tracking, performance monitoring, and
diagnostic information collection.
"""

import logging
import logging.handlers
import json
import traceback
import time
import psutil
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import threading
import os


@dataclass
class GenerationContext:
    """Context information for a generation session."""
    session_id: str
    model_type: str
    generation_mode: str
    prompt: str
    parameters: Dict[str, Any]
    start_time: float
    user_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return asdict(self)


@dataclass
class SystemDiagnostics:
    """System diagnostic information."""
    timestamp: str
    cpu_usage: float
    memory_usage: float
    gpu_memory_used: Optional[float]
    gpu_memory_total: Optional[float]
    disk_usage: float
    python_version: str
    torch_version: str
    cuda_available: bool
    cuda_version: Optional[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def collect(cls) -> 'SystemDiagnostics':
        """Collect current system diagnostics."""
        # CPU and memory info
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # GPU info
        gpu_memory_used = None
        gpu_memory_total = None
        cuda_version = None
        
        cuda_available = False
        try:
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                cuda_available = True
                try:
                    gpu_memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
                    gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                    cuda_version = torch.version.cuda
                except Exception:
                    pass
        except Exception:
            # Handle cases where torch.cuda is not available
            pass
        
        return cls(
            timestamp=datetime.now().isoformat(),
            cpu_usage=cpu_usage,
            memory_usage=memory.percent,
            gpu_memory_used=gpu_memory_used,
            gpu_memory_total=gpu_memory_total,
            disk_usage=disk.percent,
            python_version=f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}",
            torch_version=getattr(torch, '__version__', 'unknown'),
            cuda_available=cuda_available,
            cuda_version=cuda_version
        )


@dataclass
class ErrorContext:
    """Detailed error context for troubleshooting."""
    error_type: str
    error_message: str
    stack_trace: str
    generation_context: Optional[Dict[str, Any]]
    system_diagnostics: Dict[str, Any]
    timestamp: str
    recovery_attempted: bool = False
    recovery_successful: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return asdict(self)


class GenerationLogger:
    """Comprehensive logging system for video generation pipeline."""
    
    def __init__(self, 
                 log_dir: str = "logs",
                 max_log_size: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5,
                 log_level: str = "INFO"):
        """
        Initialize the generation logger.
        
        Args:
            log_dir: Directory to store log files
            max_log_size: Maximum size of each log file in bytes
            backup_count: Number of backup log files to keep
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.max_log_size = max_log_size
        self.backup_count = backup_count
        self.log_level = getattr(logging, log_level.upper())
        
        # Thread-local storage for generation context
        self._local = threading.local()
        
        # Initialize loggers
        self._setup_loggers()
        
        # Performance tracking
        self._performance_metrics = {}
        self._metrics_lock = threading.Lock()
    
    def _setup_loggers(self):
        """Set up different loggers for different purposes."""
        # Main generation logger
        self.generation_logger = self._create_logger(
            'generation',
            self.log_dir / 'generation.log',
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Error logger with detailed context
        self.error_logger = self._create_logger(
            'generation.errors',
            self.log_dir / 'errors.log',
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Performance logger
        self.performance_logger = self._create_logger(
            'generation.performance',
            self.log_dir / 'performance.log',
            '%(asctime)s - %(message)s'
        )
        
        # Diagnostics logger
        self.diagnostics_logger = self._create_logger(
            'generation.diagnostics',
            self.log_dir / 'diagnostics.log',
            '%(asctime)s - %(message)s'
        )
    
    def _create_logger(self, name: str, log_file: Path, format_string: str) -> logging.Logger:
        """Create a logger with rotating file handler."""
        logger = logging.getLogger(name)
        logger.setLevel(self.log_level)
        
        # Remove existing handlers to avoid duplicates
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Create rotating file handler
        handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=self.max_log_size,
            backupCount=self.backup_count,
            encoding='utf-8'
        )
        
        # Set formatter
        formatter = logging.Formatter(format_string)
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        logger.propagate = False
        
        return logger
    
    @contextmanager
    def generation_session(self, context: GenerationContext):
        """Context manager for tracking a generation session."""
        self._local.context = context
        
        # Log session start
        self.log_generation_start(context)
        
        # Collect initial diagnostics
        diagnostics = SystemDiagnostics.collect()
        self.log_diagnostics(diagnostics, f"Session {context.session_id} start")
        
        start_time = time.time()
        
        try:
            yield context
            
            # Log successful completion
            duration = time.time() - start_time
            self.log_generation_success(context, duration)
            
        except Exception as e:
            # Log error with full context
            duration = time.time() - start_time
            self.log_generation_error(e, context, duration)
            raise
        
        finally:
            # Clean up context
            if hasattr(self._local, 'context'):
                delattr(self._local, 'context')
    
    def log_generation_start(self, context: GenerationContext):
        """Log the start of a generation session."""
        self.generation_logger.info(
            f"Starting generation session {context.session_id} - "
            f"Model: {context.model_type}, Mode: {context.generation_mode}"
        )
        
        # Log detailed parameters
        self.generation_logger.debug(
            f"Session {context.session_id} parameters: {json.dumps(context.parameters, indent=2)}"
        )
    
    def log_generation_success(self, context: GenerationContext, duration: float):
        """Log successful completion of generation."""
        self.generation_logger.info(
            f"Generation session {context.session_id} completed successfully in {duration:.2f}s"
        )
        
        # Log performance metrics
        self.performance_logger.info(
            json.dumps({
                'session_id': context.session_id,
                'model_type': context.model_type,
                'generation_mode': context.generation_mode,
                'duration': duration,
                'status': 'success',
                'timestamp': datetime.now().isoformat()
            })
        )
    
    def log_generation_error(self, error: Exception, context: GenerationContext, duration: float):
        """Log generation error with full context."""
        error_context = ErrorContext(
            error_type=type(error).__name__,
            error_message=str(error),
            stack_trace=traceback.format_exc(),
            generation_context=context.to_dict(),
            system_diagnostics=SystemDiagnostics.collect().to_dict(),
            timestamp=datetime.now().isoformat()
        )
        
        # Log to error logger
        self.error_logger.error(
            f"Generation session {context.session_id} failed: {error_context.error_message}"
        )
        
        # Log detailed error context
        self.error_logger.error(
            f"Error context for session {context.session_id}: {json.dumps(error_context.to_dict(), indent=2)}"
        )
        
        # Log performance metrics for failed generation
        self.performance_logger.info(
            json.dumps({
                'session_id': context.session_id,
                'model_type': context.model_type,
                'generation_mode': context.generation_mode,
                'duration': duration,
                'status': 'error',
                'error_type': error_context.error_type,
                'timestamp': datetime.now().isoformat()
            })
        )
    
    def log_pipeline_stage(self, stage: str, message: str, level: str = "INFO"):
        """Log a pipeline stage with context."""
        context = getattr(self._local, 'context', None)
        session_id = context.session_id if context else "unknown"
        
        log_message = f"[{session_id}] {stage}: {message}"
        
        logger_method = getattr(self.generation_logger, level.lower())
        logger_method(log_message)
    
    def log_diagnostics(self, diagnostics: SystemDiagnostics, context: str = ""):
        """Log system diagnostics."""
        self.diagnostics_logger.info(
            f"{context} - System diagnostics: {json.dumps(diagnostics.to_dict(), indent=2)}"
        )
    
    def log_model_loading(self, model_type: str, model_path: str, success: bool, duration: float, error: Optional[str] = None):
        """Log model loading events."""
        status = "success" if success else "failed"
        message = f"Model loading {status} - Type: {model_type}, Path: {model_path}, Duration: {duration:.2f}s"
        
        if success:
            self.generation_logger.info(message)
        else:
            self.generation_logger.error(f"{message}, Error: {error}")
    
    def log_vram_usage(self, stage: str, used_gb: float, total_gb: float, percentage: float):
        """Log VRAM usage at different stages."""
        context = getattr(self._local, 'context', None)
        session_id = context.session_id if context else "unknown"
        
        self.performance_logger.info(
            json.dumps({
                'session_id': session_id,
                'stage': stage,
                'vram_used_gb': used_gb,
                'vram_total_gb': total_gb,
                'vram_percentage': percentage,
                'timestamp': datetime.now().isoformat()
            })
        )
    
    def log_parameter_optimization(self, original_params: Dict[str, Any], optimized_params: Dict[str, Any], reason: str):
        """Log parameter optimization events."""
        context = getattr(self._local, 'context', None)
        session_id = context.session_id if context else "unknown"
        
        self.generation_logger.info(
            f"[{session_id}] Parameter optimization applied - Reason: {reason}"
        )
        
        self.generation_logger.debug(
            f"[{session_id}] Parameter changes: "
            f"Original: {json.dumps(original_params, indent=2)}, "
            f"Optimized: {json.dumps(optimized_params, indent=2)}"
        )
    
    def log_recovery_attempt(self, error_type: str, recovery_strategy: str, success: bool):
        """Log error recovery attempts."""
        context = getattr(self._local, 'context', None)
        session_id = context.session_id if context else "unknown"
        
        status = "successful" if success else "failed"
        self.generation_logger.info(
            f"[{session_id}] Recovery attempt {status} - Error: {error_type}, Strategy: {recovery_strategy}"
        )
    
    def get_session_logs(self, session_id: str) -> Dict[str, List[str]]:
        """Retrieve all logs for a specific session."""
        logs = {
            'generation': [],
            'errors': [],
            'performance': [],
            'diagnostics': []
        }
        
        log_files = {
            'generation': self.log_dir / 'generation.log',
            'errors': self.log_dir / 'errors.log',
            'performance': self.log_dir / 'performance.log',
            'diagnostics': self.log_dir / 'diagnostics.log'
        }
        
        for log_type, log_file in log_files.items():
            if log_file.exists():
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            if session_id in line:
                                logs[log_type].append(line.strip())
                except Exception as e:
                    self.error_logger.error(f"Error reading log file {log_file}: {e}")
        
        return logs
    
    def generate_diagnostic_report(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Generate a comprehensive diagnostic report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'system_diagnostics': SystemDiagnostics.collect().to_dict(),
            'log_summary': self._get_log_summary(),
        }
        
        if session_id:
            report['session_logs'] = self.get_session_logs(session_id)
        
        return report
    
    def _get_log_summary(self) -> Dict[str, Any]:
        """Get summary statistics from log files."""
        summary = {
            'total_sessions': 0,
            'successful_sessions': 0,
            'failed_sessions': 0,
            'error_types': {},
            'average_duration': 0.0
        }
        
        performance_log = self.log_dir / 'performance.log'
        if performance_log.exists():
            try:
                durations = []
                with open(performance_log, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            # Extract JSON from log line
                            json_start = line.find('{')
                            if json_start != -1:
                                data = json.loads(line[json_start:])
                                summary['total_sessions'] += 1
                                
                                if data.get('status') == 'success':
                                    summary['successful_sessions'] += 1
                                elif data.get('status') == 'error':
                                    summary['failed_sessions'] += 1
                                    error_type = data.get('error_type', 'unknown')
                                    summary['error_types'][error_type] = summary['error_types'].get(error_type, 0) + 1
                                
                                if 'duration' in data:
                                    durations.append(data['duration'])
                        except json.JSONDecodeError:
                            continue
                
                if durations:
                    summary['average_duration'] = sum(durations) / len(durations)
                    
            except Exception as e:
                self.error_logger.error(f"Error generating log summary: {e}")
        
        return summary
    
    def cleanup_old_logs(self, days_to_keep: int = 30):
        """Clean up log files older than specified days."""
        cutoff_time = time.time() - (days_to_keep * 24 * 60 * 60)
        
        for log_file in self.log_dir.glob('*.log*'):
            try:
                if log_file.stat().st_mtime < cutoff_time:
                    log_file.unlink()
                    self.generation_logger.info(f"Cleaned up old log file: {log_file}")
            except Exception as e:
                self.error_logger.error(f"Error cleaning up log file {log_file}: {e}")


# Global logger instance
_logger_instance = None
_logger_lock = threading.Lock()


def get_logger() -> GenerationLogger:
    """Get the global logger instance (singleton pattern)."""
    global _logger_instance
    
    if _logger_instance is None:
        with _logger_lock:
            if _logger_instance is None:
                _logger_instance = GenerationLogger()
    
    return _logger_instance


def configure_logger(log_dir: str = "logs", 
                    max_log_size: int = 10 * 1024 * 1024,
                    backup_count: int = 5,
                    log_level: str = "INFO") -> GenerationLogger:
    """Configure the global logger instance."""
    global _logger_instance
    
    with _logger_lock:
        _logger_instance = GenerationLogger(
            log_dir=log_dir,
            max_log_size=max_log_size,
            backup_count=backup_count,
            log_level=log_level
        )
    
    return _logger_instance
