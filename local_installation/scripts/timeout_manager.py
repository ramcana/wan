"""
TimeoutManager - Context-aware timeout management and resource cleanup system

This module provides comprehensive timeout management with context-aware timeout calculation,
automatic cleanup of temporary files and resources during failures, resource exhaustion
detection and prevention, graceful operation cancellation, and disk space monitoring.

Requirements addressed: 5.4, 6.1, 6.2
"""

import logging
import time
import threading
import signal
import os
import shutil
import psutil
import tempfile
from typing import Dict, Any, Optional, List, Callable, Union, ContextManager
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
import json
from contextlib import contextmanager
import weakref
import atexit

from interfaces import InstallationError, ErrorCategory
from base_classes import BaseInstallationComponent


class OperationType(Enum):
    """Types of operations that can be timed out."""
    MODEL_DOWNLOAD = "model_download"
    DEPENDENCY_INSTALL = "dependency_install"
    SYSTEM_DETECTION = "system_detection"
    VALIDATION = "validation"
    NETWORK_TEST = "network_test"
    FILE_OPERATION = "file_operation"
    CONFIGURATION = "configuration"
    CLEANUP = "cleanup"
    UNKNOWN = "unknown"


class ResourceType(Enum):
    """Types of resources that can be tracked and cleaned up."""
    TEMPORARY_FILE = "temporary_file"
    TEMPORARY_DIRECTORY = "temporary_directory"
    DOWNLOAD_CACHE = "download_cache"
    PROCESS = "process"
    NETWORK_CONNECTION = "network_connection"
    FILE_HANDLE = "file_handle"
    MEMORY_BUFFER = "memory_buffer"


@dataclass
class TimeoutConfiguration:
    """Configuration for timeout management."""
    base_timeout: int  # Base timeout in seconds
    max_timeout: int   # Maximum timeout in seconds
    min_timeout: int   # Minimum timeout in seconds
    size_multiplier: float = 1.0  # Multiplier based on file/data size
    speed_multiplier: float = 1.0  # Multiplier based on network/system speed
    retry_multiplier: float = 1.2  # Multiplier for retry attempts
    complexity_multiplier: float = 1.0  # Multiplier based on operation complexity


@dataclass
class ResourceInfo:
    """Information about a tracked resource."""
    resource_id: str
    resource_type: ResourceType
    path: Optional[str] = None
    size_bytes: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    cleanup_callback: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OperationContext:
    """Context information for an operation."""
    operation_id: str
    operation_type: OperationType
    start_time: datetime
    timeout_seconds: int
    file_size_gb: float = 0.0
    network_speed: str = "unknown"  # fast, medium, slow
    retry_count: int = 0
    complexity_level: str = "normal"  # simple, normal, complex
    resources: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemResourceStatus:
    """Current system resource status."""
    available_memory_gb: float
    available_disk_gb: float
    cpu_usage_percent: float
    disk_io_usage_percent: float
    network_usage_percent: float
    open_file_handles: int
    running_processes: int
    timestamp: datetime = field(default_factory=datetime.now)


class TimeoutException(Exception):
    """Exception raised when an operation times out."""
    def __init__(self, message: str, operation_id: str, timeout_seconds: int):
        super().__init__(message)
        self.operation_id = operation_id
        self.timeout_seconds = timeout_seconds


class ResourceExhaustionException(Exception):
    """Exception raised when system resources are exhausted."""
    def __init__(self, message: str, resource_type: str, current_usage: float, limit: float):
        super().__init__(message)
        self.resource_type = resource_type
        self.current_usage = current_usage
        self.limit = limit


class TimeoutManager(BaseInstallationComponent):
    """
    Context-aware timeout management and resource cleanup system.
    
    This class provides:
    - Context-aware timeout calculation based on operation type, file size, network speed, etc.
    - Automatic cleanup of temporary files and resources during failures
    - Resource exhaustion detection and prevention
    - Graceful operation cancellation and cleanup
    - Disk space monitoring during long-running operations
    """
    
    # Default timeout configurations for different operation types
    DEFAULT_TIMEOUTS = {
        OperationType.MODEL_DOWNLOAD: TimeoutConfiguration(
            base_timeout=1800,  # 30 minutes
            max_timeout=7200,   # 2 hours
            min_timeout=300,    # 5 minutes
            size_multiplier=2.0,
            speed_multiplier=1.5
        ),
        OperationType.DEPENDENCY_INSTALL: TimeoutConfiguration(
            base_timeout=600,   # 10 minutes
            max_timeout=1800,   # 30 minutes
            min_timeout=60,     # 1 minute
            size_multiplier=1.5,
            speed_multiplier=1.2
        ),
        OperationType.SYSTEM_DETECTION: TimeoutConfiguration(
            base_timeout=60,    # 1 minute
            max_timeout=300,    # 5 minutes
            min_timeout=10,     # 10 seconds
            complexity_multiplier=1.5
        ),
        OperationType.VALIDATION: TimeoutConfiguration(
            base_timeout=300,   # 5 minutes
            max_timeout=900,    # 15 minutes
            min_timeout=30,     # 30 seconds
            complexity_multiplier=2.0
        ),
        OperationType.NETWORK_TEST: TimeoutConfiguration(
            base_timeout=30,    # 30 seconds
            max_timeout=120,    # 2 minutes
            min_timeout=5,      # 5 seconds
            speed_multiplier=2.0
        ),
        OperationType.FILE_OPERATION: TimeoutConfiguration(
            base_timeout=120,   # 2 minutes
            max_timeout=600,    # 10 minutes
            min_timeout=10,     # 10 seconds
            size_multiplier=1.8
        ),
        OperationType.CONFIGURATION: TimeoutConfiguration(
            base_timeout=180,   # 3 minutes
            max_timeout=600,    # 10 minutes
            min_timeout=30,     # 30 seconds
            complexity_multiplier=1.5
        ),
        OperationType.CLEANUP: TimeoutConfiguration(
            base_timeout=300,   # 5 minutes
            max_timeout=900,    # 15 minutes
            min_timeout=60,     # 1 minute
        ),
        OperationType.UNKNOWN: TimeoutConfiguration(
            base_timeout=300,   # 5 minutes
            max_timeout=1800,   # 30 minutes
            min_timeout=60,     # 1 minute
        )
    }
    
    def __init__(self, installation_path: str, logger: Optional[logging.Logger] = None):
        super().__init__(installation_path, logger)
        
        # Resource tracking
        self.tracked_resources: Dict[str, ResourceInfo] = {}
        self.active_operations: Dict[str, OperationContext] = {}
        self.cleanup_callbacks: List[Callable] = []
        
        # Resource limits and thresholds
        self.min_free_disk_gb = 2.0  # Minimum free disk space in GB
        self.min_free_memory_gb = 1.0  # Minimum free memory in GB
        self.max_cpu_usage_percent = 95.0  # Maximum CPU usage percentage
        self.max_open_files = 1000  # Maximum open file handles
        
        # Monitoring configuration
        self.resource_check_interval = 30  # Check resources every 30 seconds
        self.cleanup_interval = 300  # Cleanup old resources every 5 minutes
        self.max_resource_age_hours = 24  # Clean up resources older than 24 hours
        
        # Threading for background monitoring
        self._monitoring_thread = None
        self._monitoring_active = False
        self._resource_lock = threading.Lock()
        
        # Temporary directory for managed resources
        self.temp_dir = Path(installation_path) / "temp"
        self.temp_dir.mkdir(exist_ok=True)
        
        # Register cleanup on exit
        atexit.register(self.cleanup_all_resources)
        
        self.logger.info("TimeoutManager initialized successfully")
    
    def calculate_timeout(self, operation_type: OperationType, context: Dict[str, Any]) -> int:
        """
        Calculate context-aware timeout for an operation.
        
        Args:
            operation_type: Type of operation
            context: Context information including file size, network speed, retry count, etc.
            
        Returns:
            Calculated timeout in seconds
        """
        # Get base configuration
        config = self.DEFAULT_TIMEOUTS.get(operation_type, self.DEFAULT_TIMEOUTS[OperationType.UNKNOWN])
        
        # Start with base timeout
        timeout = config.base_timeout
        
        # Adjust based on file size
        file_size_gb = context.get('file_size_gb', 0.0)
        if file_size_gb > 0:
            timeout = int(timeout * (1 + file_size_gb * config.size_multiplier))
        
        # Adjust based on network speed
        network_speed = context.get('network_speed', 'unknown')
        if network_speed == 'slow':
            timeout = int(timeout * config.speed_multiplier * 2)
        elif network_speed == 'medium':
            timeout = int(timeout * config.speed_multiplier)
        
        # Adjust based on retry count
        retry_count = context.get('retry_count', 0)
        if retry_count > 0:
            timeout = int(timeout * (config.retry_multiplier ** retry_count))
        
        # Adjust based on complexity
        complexity = context.get('complexity_level', 'normal')
        if complexity == 'complex':
            timeout = int(timeout * config.complexity_multiplier * 2)
        elif complexity == 'simple':
            timeout = int(timeout / config.complexity_multiplier)
        
        # Adjust based on system load
        try:
            cpu_usage = psutil.cpu_percent(interval=1)
            if cpu_usage > 80:
                timeout = int(timeout * 1.5)  # Increase timeout on high CPU usage
        except Exception:
            pass
        
        # Ensure timeout is within bounds
        timeout = max(config.min_timeout, min(timeout, config.max_timeout))
        
        self.logger.debug(f"Calculated timeout for {operation_type.value}: {timeout}s (context: {context})")
        return timeout
    
    @contextmanager
    def timeout_context(self, operation_type: OperationType, context: Dict[str, Any] = None,
                       operation_id: str = None):
        """
        Context manager for timeout-aware operations with automatic resource cleanup.
        
        Args:
            operation_type: Type of operation
            context: Context information for timeout calculation
            operation_id: Optional operation identifier
            
        Yields:
            OperationContext object with operation details
        """
        if context is None:
            context = {}
        
        if operation_id is None:
            operation_id = f"{operation_type.value}_{int(time.time())}"
        
        # Calculate timeout
        timeout_seconds = self.calculate_timeout(operation_type, context)
        
        # Create operation context
        op_context = OperationContext(
            operation_id=operation_id,
            operation_type=operation_type,
            start_time=datetime.now(),
            timeout_seconds=timeout_seconds,
            file_size_gb=context.get('file_size_gb', 0.0),
            network_speed=context.get('network_speed', 'unknown'),
            retry_count=context.get('retry_count', 0),
            complexity_level=context.get('complexity_level', 'normal'),
            metadata=context
        )
        
        # Register operation
        self.active_operations[operation_id] = op_context
        
        # Set up timeout handler
        timeout_triggered = threading.Event()
        
        def timeout_handler():
            time.sleep(timeout_seconds)
            if not timeout_triggered.is_set():
                timeout_triggered.set()
        
        timeout_thread = threading.Thread(target=timeout_handler, daemon=True)
        timeout_thread.start()
        
        try:
            # Check resource availability before starting
            self._check_resource_availability()
            
            self.logger.info(f"Starting operation {operation_id} with {timeout_seconds}s timeout")
            yield op_context
            
            # Check if timeout was triggered
            if timeout_triggered.is_set():
                raise TimeoutException(
                    f"Operation {operation_id} timed out after {timeout_seconds} seconds",
                    operation_id,
                    timeout_seconds
                )
            
        except Exception as e:
            self.logger.error(f"Operation {operation_id} failed: {e}")
            # Cleanup resources associated with this operation
            self._cleanup_operation_resources(operation_id)
            raise
        finally:
            # Mark timeout as handled
            timeout_triggered.set()
            
            # Always cleanup resources associated with this operation
            self._cleanup_operation_resources(operation_id)
            
            # Remove from active operations
            self.active_operations.pop(operation_id, None)
            
            self.logger.info(f"Operation {operation_id} completed")
    
    def register_resource(self, resource_type: ResourceType, path: str = None,
                         size_bytes: int = 0, cleanup_callback: Callable = None,
                         metadata: Dict[str, Any] = None) -> str:
        """
        Register a resource for tracking and automatic cleanup.
        
        Args:
            resource_type: Type of resource
            path: Path to the resource (for files/directories)
            size_bytes: Size of the resource in bytes
            cleanup_callback: Optional callback for custom cleanup
            metadata: Additional metadata about the resource
            
        Returns:
            Resource ID for tracking
        """
        resource_id = f"{resource_type.value}_{int(time.time())}_{id(self)}"
        
        resource_info = ResourceInfo(
            resource_id=resource_id,
            resource_type=resource_type,
            path=path,
            size_bytes=size_bytes,
            cleanup_callback=cleanup_callback,
            metadata=metadata or {}
        )
        
        with self._resource_lock:
            self.tracked_resources[resource_id] = resource_info
        
        self.logger.debug(f"Registered resource {resource_id}: {resource_type.value}")
        return resource_id
    
    def unregister_resource(self, resource_id: str, cleanup: bool = True):
        """
        Unregister a resource and optionally clean it up.
        
        Args:
            resource_id: ID of the resource to unregister
            cleanup: Whether to perform cleanup
        """
        with self._resource_lock:
            resource_info = self.tracked_resources.pop(resource_id, None)
        
        if resource_info and cleanup:
            self._cleanup_resource(resource_info)
        
        self.logger.debug(f"Unregistered resource {resource_id}")
    
    def create_temp_file(self, suffix: str = "", prefix: str = "wan22_", 
                        operation_id: str = None) -> str:
        """
        Create a temporary file that will be automatically cleaned up.
        
        Args:
            suffix: File suffix
            prefix: File prefix
            operation_id: Optional operation ID to associate with
            
        Returns:
            Path to the temporary file
        """
        temp_file = tempfile.NamedTemporaryFile(
            suffix=suffix,
            prefix=prefix,
            dir=str(self.temp_dir),
            delete=False
        )
        temp_path = temp_file.name
        temp_file.close()
        
        # Register for cleanup
        resource_id = self.register_resource(
            ResourceType.TEMPORARY_FILE,
            path=temp_path,
            metadata={'operation_id': operation_id}
        )
        
        # Associate with operation if provided
        if operation_id and operation_id in self.active_operations:
            self.active_operations[operation_id].resources.append(resource_id)
        
        self.logger.debug(f"Created temporary file: {temp_path}")
        return temp_path
    
    def create_temp_directory(self, suffix: str = "", prefix: str = "wan22_",
                             operation_id: str = None) -> str:
        """
        Create a temporary directory that will be automatically cleaned up.
        
        Args:
            suffix: Directory suffix
            prefix: Directory prefix
            operation_id: Optional operation ID to associate with
            
        Returns:
            Path to the temporary directory
        """
        temp_dir = tempfile.mkdtemp(
            suffix=suffix,
            prefix=prefix,
            dir=str(self.temp_dir)
        )
        
        # Register for cleanup
        resource_id = self.register_resource(
            ResourceType.TEMPORARY_DIRECTORY,
            path=temp_dir,
            metadata={'operation_id': operation_id}
        )
        
        # Associate with operation if provided
        if operation_id and operation_id in self.active_operations:
            self.active_operations[operation_id].resources.append(resource_id)
        
        self.logger.debug(f"Created temporary directory: {temp_dir}")
        return temp_dir
    
    def monitor_disk_space(self, operation_id: str, check_interval: int = 30):
        """
        Monitor disk space during a long-running operation.
        
        Args:
            operation_id: ID of the operation to monitor
            check_interval: How often to check disk space (seconds)
        """
        def monitor():
            while operation_id in self.active_operations:
                try:
                    status = self._get_system_resource_status()
                    
                    if status.available_disk_gb < self.min_free_disk_gb:
                        self.logger.warning(
                            f"Low disk space detected: {status.available_disk_gb:.2f}GB available"
                        )
                        
                        # Try to free up space by cleaning old resources
                        self._cleanup_old_resources()
                        
                        # Check again
                        status = self._get_system_resource_status()
                        if status.available_disk_gb < self.min_free_disk_gb:
                            raise ResourceExhaustionException(
                                f"Insufficient disk space: {status.available_disk_gb:.2f}GB available, "
                                f"minimum required: {self.min_free_disk_gb}GB",
                                "disk_space",
                                status.available_disk_gb,
                                self.min_free_disk_gb
                            )
                    
                    time.sleep(check_interval)
                    
                except Exception as e:
                    self.logger.error(f"Disk space monitoring error: {e}")
                    break
        
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        
        self.logger.info(f"Started disk space monitoring for operation {operation_id}")
    
    def _check_resource_availability(self):
        """Check if system resources are available for operation."""
        status = self._get_system_resource_status()
        
        # Check disk space
        if status.available_disk_gb < self.min_free_disk_gb:
            raise ResourceExhaustionException(
                f"Insufficient disk space: {status.available_disk_gb:.2f}GB available, "
                f"minimum required: {self.min_free_disk_gb}GB",
                "disk_space",
                status.available_disk_gb,
                self.min_free_disk_gb
            )
        
        # Check memory
        if status.available_memory_gb < self.min_free_memory_gb:
            raise ResourceExhaustionException(
                f"Insufficient memory: {status.available_memory_gb:.2f}GB available, "
                f"minimum required: {self.min_free_memory_gb}GB",
                "memory",
                status.available_memory_gb,
                self.min_free_memory_gb
            )
        
        # Check CPU usage
        if status.cpu_usage_percent > self.max_cpu_usage_percent:
            self.logger.warning(
                f"High CPU usage detected: {status.cpu_usage_percent:.1f}%"
            )
        
        # Check open file handles
        if status.open_file_handles > self.max_open_files:
            self.logger.warning(
                f"High number of open files: {status.open_file_handles}"
            )
    
    def _get_system_resource_status(self) -> SystemResourceStatus:
        """Get current system resource status."""
        try:
            # Memory information
            memory = psutil.virtual_memory()
            available_memory_gb = memory.available / (1024**3)
            
            # Disk information
            disk = psutil.disk_usage(str(self.installation_path))
            available_disk_gb = disk.free / (1024**3)
            
            # CPU usage
            cpu_usage_percent = psutil.cpu_percent(interval=1)
            
            # Disk I/O usage
            disk_io = psutil.disk_io_counters()
            disk_io_usage_percent = 0.0  # Placeholder - would need baseline measurement
            
            # Network usage
            network_io = psutil.net_io_counters()
            network_usage_percent = 0.0  # Placeholder - would need baseline measurement
            
            # Open file handles
            try:
                process = psutil.Process()
                open_file_handles = len(process.open_files())
            except Exception:
                open_file_handles = 0
            
            # Running processes
            running_processes = len(psutil.pids())
            
            return SystemResourceStatus(
                available_memory_gb=available_memory_gb,
                available_disk_gb=available_disk_gb,
                cpu_usage_percent=cpu_usage_percent,
                disk_io_usage_percent=disk_io_usage_percent,
                network_usage_percent=network_usage_percent,
                open_file_handles=open_file_handles,
                running_processes=running_processes
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to get system resource status: {e}")
            return SystemResourceStatus(
                available_memory_gb=0.0,
                available_disk_gb=0.0,
                cpu_usage_percent=0.0,
                disk_io_usage_percent=0.0,
                network_usage_percent=0.0,
                open_file_handles=0,
                running_processes=0
            )
    
    def _cleanup_operation_resources(self, operation_id: str):
        """Clean up all resources associated with an operation."""
        if operation_id not in self.active_operations:
            return
        
        operation = self.active_operations[operation_id]
        resources_to_cleanup = operation.resources.copy()
        
        self.logger.info(f"Cleaning up {len(resources_to_cleanup)} resources for operation {operation_id}")
        
        for resource_id in resources_to_cleanup:
            try:
                self.unregister_resource(resource_id, cleanup=True)
            except Exception as e:
                self.logger.error(f"Failed to cleanup resource {resource_id}: {e}")
    
    def _cleanup_resource(self, resource_info: ResourceInfo):
        """Clean up a specific resource."""
        try:
            if resource_info.cleanup_callback:
                # Use custom cleanup callback
                resource_info.cleanup_callback()
            elif resource_info.path:
                # Standard file/directory cleanup
                path = Path(resource_info.path)
                if path.exists():
                    if resource_info.resource_type == ResourceType.TEMPORARY_DIRECTORY:
                        shutil.rmtree(str(path), ignore_errors=True)
                    else:
                        path.unlink(missing_ok=True)
            
            self.logger.debug(f"Cleaned up resource: {resource_info.resource_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup resource {resource_info.resource_id}: {e}")
    
    def _cleanup_old_resources(self):
        """Clean up resources older than the maximum age."""
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(hours=self.max_resource_age_hours)
        
        resources_to_cleanup = []
        
        with self._resource_lock:
            for resource_id, resource_info in self.tracked_resources.items():
                if resource_info.created_at < cutoff_time:
                    resources_to_cleanup.append(resource_id)
        
        self.logger.info(f"Cleaning up {len(resources_to_cleanup)} old resources")
        
        for resource_id in resources_to_cleanup:
            try:
                self.unregister_resource(resource_id, cleanup=True)
            except Exception as e:
                self.logger.error(f"Failed to cleanup old resource {resource_id}: {e}")
    
    def cleanup_all_resources(self):
        """Clean up all tracked resources."""
        self.logger.info("Cleaning up all tracked resources")
        
        with self._resource_lock:
            resource_ids = list(self.tracked_resources.keys())
        
        for resource_id in resource_ids:
            try:
                self.unregister_resource(resource_id, cleanup=True)
            except Exception as e:
                self.logger.error(f"Failed to cleanup resource {resource_id}: {e}")
    
    def start_monitoring(self):
        """Start background resource monitoring."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        
        def monitor():
            while self._monitoring_active:
                try:
                    # Check system resources
                    status = self._get_system_resource_status()
                    
                    # Log warnings for resource exhaustion
                    if status.available_disk_gb < self.min_free_disk_gb * 2:
                        self.logger.warning(f"Low disk space: {status.available_disk_gb:.2f}GB")
                    
                    if status.available_memory_gb < self.min_free_memory_gb * 2:
                        self.logger.warning(f"Low memory: {status.available_memory_gb:.2f}GB")
                    
                    # Cleanup old resources periodically
                    self._cleanup_old_resources()
                    
                    time.sleep(self.resource_check_interval)
                    
                except Exception as e:
                    self.logger.error(f"Resource monitoring error: {e}")
                    time.sleep(self.resource_check_interval)
        
        self._monitoring_thread = threading.Thread(target=monitor, daemon=True)
        self._monitoring_thread.start()
        
        self.logger.info("Started background resource monitoring")
    
    def stop_monitoring(self):
        """Stop background resource monitoring."""
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
        
        self.logger.info("Stopped background resource monitoring")
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """Get summary of tracked resources and system status."""
        with self._resource_lock:
            resource_count_by_type = {}
            total_size_bytes = 0
            
            for resource_info in self.tracked_resources.values():
                resource_type = resource_info.resource_type.value
                resource_count_by_type[resource_type] = resource_count_by_type.get(resource_type, 0) + 1
                total_size_bytes += resource_info.size_bytes
        
        system_status = self._get_system_resource_status()
        
        return {
            "tracked_resources": {
                "total_count": len(self.tracked_resources),
                "by_type": resource_count_by_type,
                "total_size_mb": total_size_bytes / (1024**2)
            },
            "active_operations": len(self.active_operations),
            "system_status": {
                "available_memory_gb": system_status.available_memory_gb,
                "available_disk_gb": system_status.available_disk_gb,
                "cpu_usage_percent": system_status.cpu_usage_percent,
                "open_file_handles": system_status.open_file_handles
            },
            "monitoring_active": self._monitoring_active
        }