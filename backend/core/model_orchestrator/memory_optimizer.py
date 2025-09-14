"""
Memory optimization utilities for large model downloads and processing.
"""

import gc
import os
import mmap
import threading
import time
import statistics
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Dict, Any, BinaryIO
import logging

from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    total_memory: int
    available_memory: int
    used_memory: int
    memory_percent: float
    process_memory: int


class MemoryMonitor:
    """Monitor and track memory usage during downloads."""
    
    def __init__(self, warning_threshold: float = 0.8, critical_threshold: float = 0.9):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._callbacks: Dict[str, callable] = {}
        
        # Try to import psutil for detailed memory monitoring
        try:
            import psutil
            self.psutil = psutil
            self._psutil_available = True
        except ImportError:
            logger.warning("psutil not available, using basic memory monitoring")
            self.psutil = None
            self._psutil_available = False
    
    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        if self._psutil_available:
            # Use psutil for accurate memory information
            memory = self.psutil.virtual_memory()
            process = self.psutil.Process()
            
            return MemoryStats(
                total_memory=memory.total,
                available_memory=memory.available,
                used_memory=memory.used,
                memory_percent=memory.percent,
                process_memory=process.memory_info().rss
            )
        else:
            # Fallback to basic memory information
            try:
                # Try to get memory info from /proc/meminfo on Linux
                if os.path.exists('/proc/meminfo'):
                    with open('/proc/meminfo', 'r') as f:
                        meminfo = f.read()
                    
                    total = 0
                    available = 0
                    for line in meminfo.split('\n'):
                        if line.startswith('MemTotal:'):
                            total = int(line.split()[1]) * 1024  # Convert KB to bytes
                        elif line.startswith('MemAvailable:'):
                            available = int(line.split()[1]) * 1024
                    
                    used = total - available
                    percent = (used / total) * 100 if total > 0 else 0
                    
                    return MemoryStats(
                        total_memory=total,
                        available_memory=available,
                        used_memory=used,
                        memory_percent=percent,
                        process_memory=0  # Not available without psutil
                    )
            except Exception as e:
                logger.debug(f"Failed to read memory info: {e}")
            
            # Ultimate fallback - return zeros
            return MemoryStats(
                total_memory=0,
                available_memory=0,
                used_memory=0,
                memory_percent=0.0,
                process_memory=0
            )
    
    def start_monitoring(self, interval: float = 5.0):
        """Start background memory monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            name="memory-monitor",
            daemon=True
        )
        self._monitor_thread.start()
        logger.info("Memory monitoring started")
    
    def stop_monitoring(self):
        """Stop background memory monitoring."""
        self._monitoring = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)
        logger.info("Memory monitoring stopped")
    
    def add_callback(self, name: str, callback: callable):
        """Add callback for memory threshold events."""
        self._callbacks[name] = callback
    
    def remove_callback(self, name: str):
        """Remove memory threshold callback."""
        self._callbacks.pop(name, None)
    
    def _monitor_loop(self, interval: float):
        """Background monitoring loop."""
        while self._monitoring:
            try:
                stats = self.get_memory_stats()
                
                # Check thresholds and trigger callbacks
                if stats.memory_percent >= self.critical_threshold * 100:
                    self._trigger_callbacks('critical', stats)
                elif stats.memory_percent >= self.warning_threshold * 100:
                    self._trigger_callbacks('warning', stats)
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in memory monitoring: {e}")
                time.sleep(interval)
    
    def _trigger_callbacks(self, level: str, stats: MemoryStats):
        """Trigger registered callbacks for memory events."""
        for name, callback in self._callbacks.items():
            try:
                callback(level, stats)
            except Exception as e:
                logger.error(f"Memory callback {name} failed: {e}")


class StreamingFileHandler:
    """Memory-efficient file handler for large downloads."""
    
    def __init__(self, chunk_size: int = 64 * 1024, use_mmap: bool = True):
        self.chunk_size = chunk_size
        self.use_mmap = use_mmap
    
    @contextmanager
    def open_for_streaming_write(self, file_path: Path, expected_size: Optional[int] = None):
        """
        Open file for streaming write with memory optimization.
        
        Args:
            file_path: Path to the file to write
            expected_size: Expected file size for pre-allocation
        """
        try:
            # Pre-allocate file space if size is known (helps with fragmentation)
            if expected_size and expected_size > 0:
                self._preallocate_file(file_path, expected_size)
            
            # Open file for writing
            with open(file_path, 'wb') as f:
                yield StreamingWriter(f, self.chunk_size)
                
        except Exception as e:
            logger.error(f"Error in streaming write: {e}")
            raise
    
    @contextmanager
    def open_for_streaming_read(self, file_path: Path):
        """Open file for streaming read with memory optimization."""
        try:
            file_size = file_path.stat().st_size
            
            # Use memory mapping for large files if available
            if self.use_mmap and file_size > 100 * 1024 * 1024:  # 100MB threshold
                with open(file_path, 'rb') as f:
                    with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                        yield StreamingReader(mm, self.chunk_size)
            else:
                # Use regular file reading for smaller files
                with open(file_path, 'rb') as f:
                    yield StreamingReader(f, self.chunk_size)
                    
        except Exception as e:
            logger.error(f"Error in streaming read: {e}")
            raise
    
    def _preallocate_file(self, file_path: Path, size: int):
        """Pre-allocate file space to reduce fragmentation."""
        try:
            # Create parent directories
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Pre-allocate space using fallocate on Linux or similar
            if hasattr(os, 'posix_fallocate'):
                with open(file_path, 'wb') as f:
                    os.posix_fallocate(f.fileno(), 0, size)
            else:
                # Fallback: create file with zeros (less efficient)
                with open(file_path, 'wb') as f:
                    f.seek(size - 1)
                    f.write(b'\0')
                    
        except Exception as e:
            logger.debug(f"File pre-allocation failed (non-critical): {e}")


class StreamingWriter:
    """Memory-efficient streaming writer."""
    
    def __init__(self, file_handle: BinaryIO, chunk_size: int):
        self.file_handle = file_handle
        self.chunk_size = chunk_size
        self.bytes_written = 0
        self._buffer = bytearray()
    
    def write(self, data: bytes) -> int:
        """Write data with buffering for efficiency."""
        self._buffer.extend(data)
        
        # Flush buffer when it reaches chunk size
        if len(self._buffer) >= self.chunk_size:
            self._flush_buffer()
        
        return len(data)
    
    def write_chunk(self, chunk: bytes):
        """Write a chunk directly (bypassing buffer)."""
        if self._buffer:
            self._flush_buffer()
        
        self.file_handle.write(chunk)
        self.bytes_written += len(chunk)
        
        # Force OS to write to disk periodically for large files
        if self.bytes_written % (10 * 1024 * 1024) == 0:  # Every 10MB
            self.file_handle.flush()
            os.fsync(self.file_handle.fileno())
    
    def flush(self):
        """Flush any remaining buffered data."""
        if self._buffer:
            self._flush_buffer()
        self.file_handle.flush()
    
    def _flush_buffer(self):
        """Flush internal buffer to file."""
        if self._buffer:
            self.file_handle.write(self._buffer)
            self.bytes_written += len(self._buffer)
            self._buffer.clear()


class StreamingReader:
    """Memory-efficient streaming reader."""
    
    def __init__(self, file_handle, chunk_size: int):
        self.file_handle = file_handle
        self.chunk_size = chunk_size
        self.bytes_read = 0
    
    def read_chunks(self) -> Iterator[bytes]:
        """Read file in chunks to minimize memory usage."""
        while True:
            chunk = self.file_handle.read(self.chunk_size)
            if not chunk:
                break
            
            self.bytes_read += len(chunk)
            yield chunk
    
    def read_chunk(self) -> Optional[bytes]:
        """Read a single chunk."""
        chunk = self.file_handle.read(self.chunk_size)
        if chunk:
            self.bytes_read += len(chunk)
        return chunk if chunk else None


class MemoryOptimizer:
    """
    Main memory optimization coordinator for model downloads.
    """
    
    def __init__(
        self,
        max_memory_usage: Optional[int] = None,
        gc_threshold: float = 0.8,
        streaming_threshold: int = 100 * 1024 * 1024  # 100MB
    ):
        self.max_memory_usage = max_memory_usage
        self.gc_threshold = gc_threshold
        self.streaming_threshold = streaming_threshold
        
        # Initialize components
        self.memory_monitor = MemoryMonitor()
        self.file_handler = StreamingFileHandler()
        
        # Register memory callbacks
        self.memory_monitor.add_callback('gc_trigger', self._handle_memory_pressure)
        
        # Track active downloads for memory management
        self._active_downloads: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
    
    def start_monitoring(self):
        """Start memory monitoring."""
        self.memory_monitor.start_monitoring()
    
    def stop_monitoring(self):
        """Stop memory monitoring."""
        self.memory_monitor.stop_monitoring()
    
    @contextmanager
    def optimized_download_context(
        self,
        model_id: str,
        total_size: int,
        file_count: int
    ):
        """
        Context manager for memory-optimized downloads.
        
        Args:
            model_id: Model identifier
            total_size: Total download size
            file_count: Number of files to download
        """
        # Register download
        with self._lock:
            self._active_downloads[model_id] = {
                'total_size': total_size,
                'file_count': file_count,
                'start_time': time.time(),
                'bytes_downloaded': 0
            }
        
        try:
            # Check if we should use streaming for large downloads
            use_streaming = total_size > self.streaming_threshold
            
            # Trigger garbage collection before large downloads
            if use_streaming:
                self._force_garbage_collection()
            
            logger.info(
                f"Starting optimized download: {model_id}",
                extra={
                    'total_size': total_size,
                    'file_count': file_count,
                    'use_streaming': use_streaming
                }
            )
            
            yield {
                'use_streaming': use_streaming,
                'file_handler': self.file_handler,
                'memory_monitor': self.memory_monitor
            }
            
        finally:
            # Unregister download
            with self._lock:
                self._active_downloads.pop(model_id, None)
            
            # Clean up after download
            self._force_garbage_collection()
    
    def update_download_progress(self, model_id: str, bytes_downloaded: int):
        """Update download progress for memory tracking."""
        with self._lock:
            if model_id in self._active_downloads:
                self._active_downloads[model_id]['bytes_downloaded'] = bytes_downloaded
    
    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        return self.memory_monitor.get_memory_stats()
    
    def get_download_stats(self) -> Dict[str, Any]:
        """Get current download statistics."""
        with self._lock:
            return {
                'active_downloads': len(self._active_downloads),
                'downloads': dict(self._active_downloads)
            }
    
    def _handle_memory_pressure(self, level: str, stats: MemoryStats):
        """Handle memory pressure events."""
        logger.warning(
            f"Memory pressure detected: {level}",
            extra={
                'memory_percent': stats.memory_percent,
                'available_memory': stats.available_memory,
                'active_downloads': len(self._active_downloads)
            }
        )
        
        if level == 'critical':
            # Force garbage collection
            self._force_garbage_collection()
            
            # Consider pausing downloads if memory is critically low
            if stats.memory_percent > 95:
                logger.critical("Critical memory usage, consider pausing downloads")
    
    def _force_garbage_collection(self):
        """Force garbage collection to free memory."""
        try:
            # Run garbage collection
            collected = gc.collect()
            
            # Get memory stats after GC
            stats = self.memory_monitor.get_memory_stats()
            
            logger.info(
                f"Garbage collection completed",
                extra={
                    'objects_collected': collected,
                    'memory_percent': stats.memory_percent,
                    'available_memory': stats.available_memory
                }
            )
            
        except Exception as e:
            logger.error(f"Garbage collection failed: {e}")
    
    def should_use_streaming(self, file_size: int) -> bool:
        """Determine if streaming should be used for a file."""
        return file_size > self.streaming_threshold
    
    def get_optimal_chunk_size(self, file_size: int, available_memory: int) -> int:
        """Calculate optimal chunk size based on file size and available memory."""
        # Base chunk size
        base_chunk = 64 * 1024  # 64KB
        
        # Adjust based on available memory
        if available_memory > 1024 * 1024 * 1024:  # > 1GB available
            chunk_size = min(1024 * 1024, file_size // 100)  # Up to 1MB chunks
        elif available_memory > 512 * 1024 * 1024:  # > 512MB available
            chunk_size = min(512 * 1024, file_size // 100)  # Up to 512KB chunks
        else:
            chunk_size = min(256 * 1024, file_size // 100)  # Up to 256KB chunks
        
        # Ensure minimum chunk size
        return max(base_chunk, chunk_size)
    
    def cleanup(self):
        """Clean up resources."""
        self.stop_monitoring()
        self._force_garbage_collection()