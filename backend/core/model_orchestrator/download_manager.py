"""
Advanced download manager with parallel downloads, bandwidth limiting, and queue management.
"""

import asyncio
import time
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Callable, Set, Any, Tuple
from queue import Queue, PriorityQueue, Empty
import logging
import hashlib
import json

from .exceptions import ModelOrchestratorError, ErrorCode
from .logging_config import get_logger

logger = get_logger(__name__)


class DownloadPriority(Enum):
    """Priority levels for download queue."""
    HIGH = 1
    NORMAL = 2
    LOW = 3


@dataclass
class DownloadTask:
    """Individual download task."""
    task_id: str
    model_id: str
    file_path: str
    source_url: str
    local_path: Path
    size: int
    priority: DownloadPriority = DownloadPriority.NORMAL
    retry_count: int = 0
    max_retries: int = 3
    created_at: float = field(default_factory=time.time)
    
    def __lt__(self, other):
        """For priority queue ordering."""
        return self.priority.value < other.priority.value


@dataclass
class DownloadProgress:
    """Progress information for a download."""
    task_id: str
    model_id: str
    file_path: str
    bytes_downloaded: int
    total_bytes: int
    speed_bps: float
    eta_seconds: Optional[float] = None
    status: str = "downloading"


@dataclass
class BandwidthLimiter:
    """Token bucket bandwidth limiter."""
    max_bps: int  # Maximum bytes per second
    bucket_size: int  # Token bucket size
    tokens: float = 0.0
    last_update: float = field(default_factory=time.time)
    lock: threading.Lock = field(default_factory=threading.Lock)
    
    def acquire_tokens(self, bytes_requested: int) -> float:
        """
        Acquire tokens for bandwidth limiting.
        
        Returns:
            Time to wait before proceeding (0 if no wait needed)
        """
        with self.lock:
            now = time.time()
            time_passed = now - self.last_update
            
            # Add tokens based on time passed
            self.tokens = min(
                self.bucket_size,
                self.tokens + (self.max_bps * time_passed)
            )
            self.last_update = now
            
            if bytes_requested <= self.tokens:
                self.tokens -= bytes_requested
                return 0.0
            else:
                # Calculate wait time
                tokens_needed = bytes_requested - self.tokens
                wait_time = tokens_needed / self.max_bps
                self.tokens = 0.0
                return wait_time


class ConnectionPool:
    """Enhanced HTTP connection pool with adaptive connection management."""
    
    def __init__(self, max_connections: int = 10, max_connections_per_host: int = 5):
        self.max_connections = max_connections
        self.max_connections_per_host = max_connections_per_host
        self._sessions: Dict[str, Any] = {}  # Per-thread sessions
        self._session_stats: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        
        # Initialize HTTP session with enhanced connection pooling
        try:
            import requests
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry
            from urllib3.poolmanager import PoolManager
            
            self.requests = requests
            self.HTTPAdapter = HTTPAdapter
            self.Retry = Retry
            
            # Create default session
            self._create_session()
            
        except ImportError:
            logger.warning("requests library not available, connection pooling disabled")
            self.requests = None
            self.HTTPAdapter = None
            self.Retry = None
    
    def _create_session(self) -> Any:
        """Create an optimized HTTP session."""
        if not self.requests:
            return None
        
        session = self.requests.Session()
        
        # Configure enhanced retry strategy
        retry_strategy = self.Retry(
            total=5,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
            raise_on_status=False
        )
        
        # Configure adapter with optimized connection pooling
        adapter = self.HTTPAdapter(
            pool_connections=self.max_connections,
            pool_maxsize=self.max_connections_per_host,
            max_retries=retry_strategy,
            pool_block=False  # Don't block when pool is full
        )
        
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set optimized timeouts and headers
        session.headers.update({
            'Connection': 'keep-alive',
            'Keep-Alive': 'timeout=30, max=100'
        })
        
        return session
    
    def get_session(self) -> Any:
        """Get HTTP session for the current thread."""
        if not self.requests:
            return None
        
        thread_id = threading.get_ident()
        
        with self._lock:
            if thread_id not in self._sessions:
                self._sessions[thread_id] = self._create_session()
                self._session_stats[thread_id] = {
                    'requests_made': 0,
                    'bytes_downloaded': 0,
                    'errors': 0,
                    'created_at': time.time()
                }
        
        return self._sessions[thread_id]
    
    def update_stats(self, thread_id: int, bytes_downloaded: int, error: bool = False):
        """Update session statistics."""
        with self._lock:
            if thread_id in self._session_stats:
                stats = self._session_stats[thread_id]
                stats['requests_made'] += 1
                stats['bytes_downloaded'] += bytes_downloaded
                if error:
                    stats['errors'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        with self._lock:
            total_requests = sum(stats['requests_made'] for stats in self._session_stats.values())
            total_bytes = sum(stats['bytes_downloaded'] for stats in self._session_stats.values())
            total_errors = sum(stats['errors'] for stats in self._session_stats.values())
            
            return {
                'active_sessions': len(self._sessions),
                'total_requests': total_requests,
                'total_bytes_downloaded': total_bytes,
                'total_errors': total_errors,
                'error_rate': total_errors / max(total_requests, 1),
                'max_connections': self.max_connections,
                'max_connections_per_host': self.max_connections_per_host
            }
    
    def cleanup_stale_sessions(self, max_age: float = 3600.0):
        """Clean up stale sessions that haven't been used recently."""
        current_time = time.time()
        stale_threads = []
        
        with self._lock:
            for thread_id, stats in self._session_stats.items():
                if current_time - stats['created_at'] > max_age:
                    stale_threads.append(thread_id)
            
            for thread_id in stale_threads:
                if thread_id in self._sessions:
                    try:
                        self._sessions[thread_id].close()
                    except:
                        pass
                    del self._sessions[thread_id]
                    del self._session_stats[thread_id]
        
        if stale_threads:
            logger.info(f"Cleaned up {len(stale_threads)} stale HTTP sessions")
    
    def close(self):
        """Close all connections in the pool."""
        with self._lock:
            for session in self._sessions.values():
                try:
                    session.close()
                except:
                    pass
            self._sessions.clear()
            self._session_stats.clear()


@dataclass
class ModelDownloadQueue:
    """Queue for managing downloads of a specific model."""
    model_id: str
    priority: DownloadPriority
    tasks: List[DownloadTask] = field(default_factory=list)
    max_concurrent_files: int = 8
    active_tasks: Set[str] = field(default_factory=set)
    completed_tasks: Set[str] = field(default_factory=set)
    failed_tasks: Dict[str, str] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    
    def __lt__(self, other):
        """For priority queue ordering."""
        return self.priority.value < other.priority.value


@dataclass
class DownloadMetrics:
    """Metrics for download performance analysis."""
    start_time: float
    end_time: Optional[float] = None
    total_bytes: int = 0
    bytes_downloaded: int = 0
    files_total: int = 0
    files_completed: int = 0
    files_failed: int = 0
    average_speed_bps: float = 0.0
    peak_speed_bps: float = 0.0
    concurrent_downloads: int = 0
    retry_count: int = 0
    bandwidth_limited_time: float = 0.0
    
    @property
    def duration(self) -> float:
        """Get download duration."""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time
    
    @property
    def completion_rate(self) -> float:
        """Get completion rate as percentage."""
        if self.files_total == 0:
            return 0.0
        return (self.files_completed / self.files_total) * 100.0


class ParallelDownloadManager:
    """
    Advanced download manager with parallel downloads, bandwidth limiting,
    queue management, and comprehensive performance optimization.
    """
    
    def __init__(
        self,
        max_concurrent_downloads: int = 4,
        max_concurrent_files_per_model: int = 8,
        max_bandwidth_bps: Optional[int] = None,
        connection_pool_size: int = 20,
        chunk_size: int = 8192 * 16,  # 128KB chunks
        enable_resume: bool = True,
        enable_adaptive_chunking: bool = True,
        enable_compression: bool = True,
        queue_timeout: float = 300.0
    ):
        self.max_concurrent_downloads = max_concurrent_downloads
        self.max_concurrent_files_per_model = max_concurrent_files_per_model
        self.chunk_size = chunk_size
        self.enable_resume = enable_resume
        self.enable_adaptive_chunking = enable_adaptive_chunking
        self.enable_compression = enable_compression
        self.queue_timeout = queue_timeout
        
        # Model-based download queues for better organization
        self.model_queues: Dict[str, ModelDownloadQueue] = {}
        self.model_queue_priority: PriorityQueue = PriorityQueue()
        
        # Legacy support - individual task tracking
        self.active_downloads: Dict[str, DownloadTask] = {}
        self.completed_downloads: Dict[str, bool] = {}
        self.failed_downloads: Dict[str, str] = {}
        
        # Progress tracking
        self.progress_callbacks: Dict[str, List[Callable[[DownloadProgress], None]]] = {}
        self.download_progress: Dict[str, DownloadProgress] = {}
        
        # Metrics tracking
        self.download_metrics: Dict[str, DownloadMetrics] = {}
        
        # Thread pool for parallel downloads with dynamic sizing
        self.executor = ThreadPoolExecutor(
            max_workers=max_concurrent_downloads * 2,  # Allow some overhead
            thread_name_prefix="download-worker"
        )
        
        # Separate executor for I/O intensive operations
        self.io_executor = ThreadPoolExecutor(
            max_workers=max_concurrent_files_per_model,
            thread_name_prefix="io-worker"
        )
        
        # Bandwidth limiting with adaptive adjustment
        self.bandwidth_limiter = None
        if max_bandwidth_bps:
            self.bandwidth_limiter = BandwidthLimiter(
                max_bps=max_bandwidth_bps,
                bucket_size=max_bandwidth_bps * 2  # 2 second burst capacity
            )
        
        # Enhanced connection pool
        self.connection_pool = ConnectionPool(
            max_connections=connection_pool_size,
            max_connections_per_host=max_concurrent_files_per_model
        )
        
        # Performance optimization state
        self._adaptive_chunk_sizes: Dict[str, int] = {}
        self._speed_history: Dict[str, List[float]] = {}
        self._congestion_detected: Dict[str, bool] = {}
        
        # Synchronization
        self._lock = threading.RLock()  # Use RLock for nested locking
        self._shutdown = False
        
        # Start background workers
        self._queue_worker_thread = threading.Thread(
            target=self._queue_worker,
            name="download-queue-worker",
            daemon=True
        )
        self._queue_worker_thread.start()
        
        self._metrics_worker_thread = threading.Thread(
            target=self._metrics_worker,
            name="metrics-worker",
            daemon=True
        )
        self._metrics_worker_thread.start()
    
    def add_download_task(
        self,
        model_id: str,
        file_path: str,
        source_url: str,
        local_path: Path,
        size: int,
        priority: DownloadPriority = DownloadPriority.NORMAL,
        progress_callback: Optional[Callable[[DownloadProgress], None]] = None
    ) -> str:
        """
        Add a download task to the queue.
        
        Returns:
            Task ID for tracking the download
        """
        task_id = f"{model_id}:{file_path}:{int(time.time())}"
        
        task = DownloadTask(
            task_id=task_id,
            model_id=model_id,
            file_path=file_path,
            source_url=source_url,
            local_path=local_path,
            size=size,
            priority=priority
        )
        
        # Add progress callback
        if progress_callback:
            if model_id not in self.progress_callbacks:
                self.progress_callbacks[model_id] = []
            self.progress_callbacks[model_id].append(progress_callback)
        
        # Add to queue
        self.download_queue.put(task)
        
        logger.info(
            f"Added download task: {task_id}",
            extra={
                "model_id": model_id,
                "file_path": file_path,
                "size": size,
                "priority": priority.name
            }
        )
        
        return task_id
    
    def queue_model_download(
        self,
        model_id: str,
        file_specs: List[Any],
        source_url: str,
        local_dir: Path,
        priority: DownloadPriority = DownloadPriority.NORMAL,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> str:
        """
        Queue a complete model download with optimized parallel file handling.
        
        Returns:
            Queue ID for tracking the model download
        """
        queue_id = f"{model_id}:{int(time.time())}"
        
        with self._lock:
            # Create model download queue
            model_queue = ModelDownloadQueue(
                model_id=model_id,
                priority=priority,
                max_concurrent_files=min(
                    self.max_concurrent_files_per_model,
                    len(file_specs)
                )
            )
            
            # Create download tasks for all files
            for file_spec in file_specs:
                local_file_path = local_dir / file_spec.path
                local_file_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Determine priority based on file type and size
                file_priority = self._determine_file_priority(file_spec)
                
                task = DownloadTask(
                    task_id=f"{queue_id}:{file_spec.path}",
                    model_id=model_id,
                    file_path=file_spec.path,
                    source_url=f"{source_url}/{file_spec.path}",
                    local_path=local_file_path,
                    size=file_spec.size,
                    priority=file_priority
                )
                model_queue.tasks.append(task)
            
            # Sort tasks by priority and size (smaller files first for quick wins)
            model_queue.tasks.sort(key=lambda t: (t.priority.value, t.size))
            
            # Add to model queues
            self.model_queues[queue_id] = model_queue
            self.model_queue_priority.put(model_queue)
            
            # Initialize metrics
            self.download_metrics[queue_id] = DownloadMetrics(
                start_time=time.time(),
                total_bytes=sum(f.size for f in file_specs),
                files_total=len(file_specs)
            )
            
            # Add progress callback
            if progress_callback:
                if model_id not in self.progress_callbacks:
                    self.progress_callbacks[model_id] = []
                self.progress_callbacks[model_id].append(progress_callback)
        
        logger.info(
            f"Queued model download: {model_id}",
            extra={
                "queue_id": queue_id,
                "file_count": len(file_specs),
                "total_size": sum(f.size for f in file_specs),
                "priority": priority.name
            }
        )
        
        return queue_id
    
    def download_model_parallel(
        self,
        model_id: str,
        file_specs: List[Any],
        source_url: str,
        local_dir: Path,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict[str, Any]:
        """
        Download all files for a model in parallel with enhanced performance optimization.
        
        Returns:
            Dictionary with download results and comprehensive statistics
        """
        # Queue the download
        queue_id = self.queue_model_download(
            model_id=model_id,
            file_specs=file_specs,
            source_url=source_url,
            local_dir=local_dir,
            progress_callback=progress_callback
        )
        
        # Wait for completion with timeout
        return self.wait_for_model_completion(queue_id, timeout=self.queue_timeout)
    
    def wait_for_model_completion(
        self,
        queue_id: str,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Wait for a queued model download to complete.
        
        Returns:
            Dictionary with download results and statistics
        """
        start_wait = time.time()
        check_interval = 0.1  # 100ms
        
        while queue_id in self.model_queues:
            if timeout and (time.time() - start_wait) > timeout:
                logger.error(f"Download timeout for queue {queue_id}")
                break
            
            time.sleep(check_interval)
            
            # Adaptive check interval based on progress
            with self._lock:
                if queue_id in self.download_metrics:
                    metrics = self.download_metrics[queue_id]
                    if metrics.files_completed > 0:
                        # Increase check interval as download progresses
                        check_interval = min(1.0, 0.1 + (metrics.completion_rate / 100.0))
        
        # Collect final results
        return self._collect_download_results(queue_id)
    
    def _determine_file_priority(self, file_spec: Any) -> DownloadPriority:
        """Determine download priority for a file based on type and size."""
        file_path = file_spec.path.lower()
        
        # High priority for small config files
        if any(file_path.endswith(ext) for ext in ['.json', '.yaml', '.yml', '.txt']):
            return DownloadPriority.HIGH
        
        # Normal priority for medium files
        if file_spec.size < 100 * 1024 * 1024:  # < 100MB
            return DownloadPriority.NORMAL
        
        # Low priority for large model files
        return DownloadPriority.LOW
    
    def _collect_download_results(self, queue_id: str) -> Dict[str, Any]:
        """Collect comprehensive download results for a model."""
        with self._lock:
            metrics = self.download_metrics.get(queue_id)
            if not metrics:
                return {"success": False, "error": "Queue not found"}
            
            # Finalize metrics
            if not metrics.end_time:
                metrics.end_time = time.time()
            
            # Calculate final statistics
            if metrics.duration > 0:
                metrics.average_speed_bps = metrics.bytes_downloaded / metrics.duration
            
            results = {
                "success": metrics.files_failed == 0,
                "queue_id": queue_id,
                "completed_tasks": metrics.files_completed,
                "failed_tasks": metrics.files_failed,
                "total_tasks": metrics.files_total,
                "total_bytes": metrics.total_bytes,
                "bytes_downloaded": metrics.bytes_downloaded,
                "duration_seconds": metrics.duration,
                "average_speed_bps": metrics.average_speed_bps,
                "peak_speed_bps": metrics.peak_speed_bps,
                "completion_rate": metrics.completion_rate,
                "concurrent_downloads": metrics.concurrent_downloads,
                "retry_count": metrics.retry_count,
                "bandwidth_limited_time": metrics.bandwidth_limited_time,
                "failed_files": []
            }
            
            # Collect failed file information
            model_queue = self.model_queues.get(queue_id)
            if model_queue:
                for task_id, error in model_queue.failed_tasks.items():
                    task_parts = task_id.split(":", 3)
                    if len(task_parts) >= 3:
                        results["failed_files"].append({
                            "file_path": task_parts[2],
                            "error": error
                        })
            
            # Clean up
            self.model_queues.pop(queue_id, None)
            self.download_metrics.pop(queue_id, None)
            
            # Clean up progress callbacks
            model_id = queue_id.split(":", 1)[0]
            if model_id in self.progress_callbacks:
                del self.progress_callbacks[model_id]
        
        return results
    
    def _queue_worker(self):
        """Enhanced background worker that processes model download queues."""
        while not self._shutdown:
            try:
                # Get next model queue (blocks with timeout)
                try:
                    model_queue = self.model_queue_priority.get(timeout=1.0)
                except Empty:
                    continue
                
                # Process this model's downloads
                self._process_model_queue(model_queue)
                
            except Exception as e:
                logger.error(f"Error in queue worker: {e}")
                time.sleep(1.0)
    
    def _process_model_queue(self, model_queue: ModelDownloadQueue):
        """Process downloads for a specific model with optimized concurrency."""
        queue_id = None
        
        # Find the queue ID for this model queue
        with self._lock:
            for qid, mq in self.model_queues.items():
                if mq is model_queue:
                    queue_id = qid
                    break
        
        if not queue_id:
            return
        
        logger.info(f"Processing model queue: {model_queue.model_id}")
        
        # Submit tasks with controlled concurrency
        active_futures = {}
        pending_tasks = list(model_queue.tasks)
        
        while pending_tasks or active_futures:
            # Submit new tasks up to concurrency limit
            while (len(active_futures) < model_queue.max_concurrent_files and 
                   pending_tasks):
                
                task = pending_tasks.pop(0)
                
                # Apply adaptive chunking
                self._apply_adaptive_chunking(task)
                
                # Submit to appropriate executor
                if self._is_io_intensive(task):
                    future = self.io_executor.submit(self._download_file_optimized, task, queue_id)
                else:
                    future = self.executor.submit(self._download_file_optimized, task, queue_id)
                
                active_futures[future] = task
                
                with self._lock:
                    model_queue.active_tasks.add(task.task_id)
                    self.active_downloads[task.task_id] = task
            
            # Wait for at least one task to complete
            if active_futures:
                completed_futures = []
                
                # Check for completed futures with short timeout
                for future in list(active_futures.keys()):
                    if future.done():
                        completed_futures.append(future)
                
                # If no futures completed, wait a bit
                if not completed_futures:
                    time.sleep(0.1)
                    continue
                
                # Process completed futures
                for future in completed_futures:
                    task = active_futures.pop(future)
                    
                    try:
                        success = future.result()
                        self._handle_task_completion(task, queue_id, success, None)
                    except Exception as e:
                        self._handle_task_completion(task, queue_id, False, str(e))
        
        # Mark model queue as complete
        with self._lock:
            if queue_id in self.download_metrics:
                self.download_metrics[queue_id].end_time = time.time()
        
        logger.info(f"Completed model queue: {model_queue.model_id}")
    
    def _handle_task_completion(
        self,
        task: DownloadTask,
        queue_id: str,
        success: bool,
        error: Optional[str]
    ):
        """Handle completion of a download task."""
        with self._lock:
            # Update model queue
            if queue_id in self.model_queues:
                model_queue = self.model_queues[queue_id]
                model_queue.active_tasks.discard(task.task_id)
                
                if success:
                    model_queue.completed_tasks.add(task.task_id)
                    self.completed_downloads[task.task_id] = True
                else:
                    model_queue.failed_tasks[task.task_id] = error or "Unknown error"
                    self.failed_downloads[task.task_id] = error or "Unknown error"
            
            # Update metrics
            if queue_id in self.download_metrics:
                metrics = self.download_metrics[queue_id]
                if success:
                    metrics.files_completed += 1
                    metrics.bytes_downloaded += task.size
                else:
                    metrics.files_failed += 1
                    metrics.retry_count += task.retry_count
            
            # Clean up active downloads
            self.active_downloads.pop(task.task_id, None)
    
    def _apply_adaptive_chunking(self, task: DownloadTask):
        """Apply adaptive chunk sizing based on file size and network conditions."""
        if not self.enable_adaptive_chunking:
            return
        
        # Get optimal chunk size based on file size
        if task.size < 1024 * 1024:  # < 1MB
            chunk_size = 8192  # 8KB
        elif task.size < 10 * 1024 * 1024:  # < 10MB
            chunk_size = 64 * 1024  # 64KB
        elif task.size < 100 * 1024 * 1024:  # < 100MB
            chunk_size = 256 * 1024  # 256KB
        else:  # >= 100MB
            chunk_size = 1024 * 1024  # 1MB
        
        # Adjust based on historical performance
        source_key = self._get_source_key(task.source_url)
        if source_key in self._speed_history:
            speeds = self._speed_history[source_key]
            if speeds:
                avg_speed = statistics.mean(speeds[-10:])  # Last 10 measurements
                
                # Increase chunk size for fast connections
                if avg_speed > 50 * 1024 * 1024:  # > 50MB/s
                    chunk_size = min(chunk_size * 2, 2 * 1024 * 1024)
                # Decrease chunk size for slow connections
                elif avg_speed < 1 * 1024 * 1024:  # < 1MB/s
                    chunk_size = max(chunk_size // 2, 8192)
        
        self._adaptive_chunk_sizes[task.task_id] = chunk_size
    
    def _is_io_intensive(self, task: DownloadTask) -> bool:
        """Determine if a task is I/O intensive and should use the I/O executor."""
        # Large files are more I/O intensive
        return task.size > 50 * 1024 * 1024  # > 50MB
    
    def _get_source_key(self, source_url: str) -> str:
        """Get a key for tracking performance by source."""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(source_url)
            return f"{parsed.scheme}://{parsed.netloc}"
        except:
            return "unknown"
    
    def _metrics_worker(self):
        """Background worker for collecting and updating performance metrics."""
        while not self._shutdown:
            try:
                with self._lock:
                    current_time = time.time()
                    
                    # Update metrics for active downloads
                    for queue_id, metrics in self.download_metrics.items():
                        if queue_id in self.model_queues:
                            model_queue = self.model_queues[queue_id]
                            metrics.concurrent_downloads = len(model_queue.active_tasks)
                            
                            # Calculate current speed
                            if metrics.duration > 0:
                                current_speed = metrics.bytes_downloaded / metrics.duration
                                metrics.peak_speed_bps = max(metrics.peak_speed_bps, current_speed)
                    
                    # Update speed history for adaptive chunking
                    for task_id, progress in self.download_progress.items():
                        if progress.speed_bps > 0:
                            source_key = self._get_source_key(
                                self.active_downloads.get(task_id, DownloadTask("", "", "", "", Path(), 0)).source_url
                            )
                            if source_key not in self._speed_history:
                                self._speed_history[source_key] = []
                            
                            self._speed_history[source_key].append(progress.speed_bps)
                            
                            # Keep only recent history
                            if len(self._speed_history[source_key]) > 50:
                                self._speed_history[source_key] = self._speed_history[source_key][-25:]
                
                time.sleep(2.0)  # Update every 2 seconds
                
            except Exception as e:
                logger.error(f"Error in metrics worker: {e}")
                time.sleep(5.0)
    
    def _download_file_optimized(self, task: DownloadTask, queue_id: str) -> bool:
        """
        Optimized download method with advanced performance features.
        
        Returns:
            True if download succeeded, False otherwise
        """
        try:
            # Initialize progress tracking
            progress = DownloadProgress(
                task_id=task.task_id,
                model_id=task.model_id,
                file_path=task.file_path,
                bytes_downloaded=0,
                total_bytes=task.size,
                speed_bps=0.0
            )
            self.download_progress[task.task_id] = progress
            
            # Check if file already exists and resume if enabled
            resume_pos = 0
            if self.enable_resume and task.local_path.exists():
                resume_pos = task.local_path.stat().st_size
                if resume_pos >= task.size:
                    # File already complete
                    progress.bytes_downloaded = task.size
                    progress.status = "complete"
                    self._notify_progress(progress)
                    return True
            
            # Get adaptive chunk size
            chunk_size = self._adaptive_chunk_sizes.get(task.task_id, self.chunk_size)
            
            # Prepare HTTP headers
            headers = self._prepare_headers(task, resume_pos)
            
            # Start download with optimized session
            session = self.connection_pool.get_session()
            if not session:
                return self._download_file_basic(task, progress, resume_pos)
            
            start_time = time.time()
            last_progress_time = start_time
            bandwidth_wait_time = 0.0
            
            try:
                with session.get(task.source_url, headers=headers, stream=True, timeout=(30, 300)) as response:
                    response.raise_for_status()
                    
                    # Verify content length if available
                    content_length = response.headers.get('content-length')
                    if content_length:
                        expected_size = int(content_length)
                        if resume_pos > 0:
                            expected_size += resume_pos
                        if expected_size != task.size:
                            logger.warning(f"Size mismatch for {task.file_path}: expected {task.size}, got {expected_size}")
                    
                    # Open file for writing with optimized buffering
                    mode = 'ab' if resume_pos > 0 else 'wb'
                    buffer_size = max(chunk_size * 4, 64 * 1024)  # At least 64KB buffer
                    
                    with open(task.local_path, mode, buffering=buffer_size) as f:
                        bytes_since_sync = 0
                        
                        for chunk in response.iter_content(chunk_size=chunk_size):
                            if chunk:  # Filter out keep-alive chunks
                                # Apply bandwidth limiting
                                if self.bandwidth_limiter:
                                    wait_time = self.bandwidth_limiter.acquire_tokens(len(chunk))
                                    if wait_time > 0:
                                        time.sleep(wait_time)
                                        bandwidth_wait_time += wait_time
                                
                                # Write chunk
                                f.write(chunk)
                                progress.bytes_downloaded += len(chunk)
                                bytes_since_sync += len(chunk)
                                
                                # Periodic fsync for large files
                                if bytes_since_sync > 10 * 1024 * 1024:  # Every 10MB
                                    f.flush()
                                    os.fsync(f.fileno())
                                    bytes_since_sync = 0
                                
                                # Update progress and adapt performance
                                now = time.time()
                                if now - last_progress_time >= 0.5:  # Update every 500ms
                                    elapsed = now - start_time
                                    if elapsed > 0:
                                        # Calculate speed excluding bandwidth wait time
                                        effective_time = elapsed - bandwidth_wait_time
                                        if effective_time > 0:
                                            progress.speed_bps = progress.bytes_downloaded / effective_time
                                        
                                        # Calculate ETA
                                        remaining_bytes = progress.total_bytes - progress.bytes_downloaded
                                        if progress.speed_bps > 0:
                                            progress.eta_seconds = remaining_bytes / progress.speed_bps
                                    
                                    self._notify_progress(progress)
                                    self._update_performance_metrics(task, progress, queue_id)
                                    last_progress_time = now
                        
                        # Final sync
                        f.flush()
                        os.fsync(f.fileno())
            
            except Exception as e:
                # Handle specific HTTP errors
                if hasattr(e, 'response') and hasattr(e.response, 'status_code'):
                    if e.response.status_code == 416:  # Range not satisfiable
                        logger.info(f"Range not satisfiable for {task.file_path}, starting fresh download")
                        return self._download_file_fresh(task, progress)
                    elif e.response.status_code in [429, 503, 504]:  # Rate limiting or server errors
                        logger.warning(f"Server error {e.response.status_code} for {task.file_path}, will retry")
                        raise
                raise
            
            # Verify download completion
            if progress.bytes_downloaded < task.size:
                logger.error(f"Incomplete download: {task.file_path} ({progress.bytes_downloaded}/{task.size} bytes)")
                return False
            
            # Final progress update
            progress.status = "complete"
            progress.bytes_downloaded = task.size
            self._notify_progress(progress)
            
            # Update performance history
            duration = time.time() - start_time
            if duration > 0:
                final_speed = task.size / duration
                source_key = self._get_source_key(task.source_url)
                with self._lock:
                    if source_key not in self._speed_history:
                        self._speed_history[source_key] = []
                    self._speed_history[source_key].append(final_speed)
            
            logger.info(
                f"Download completed: {task.file_path}",
                extra={
                    "model_id": task.model_id,
                    "bytes": task.size,
                    "duration": duration,
                    "speed_mbps": (task.size / duration) / (1024 * 1024) if duration > 0 else 0,
                    "chunk_size": chunk_size
                }
            )
            
            return True
            
        except Exception as e:
            logger.error(
                f"Download failed: {task.file_path}",
                extra={
                    "model_id": task.model_id,
                    "error": str(e),
                    "retry_count": task.retry_count
                }
            )
            
            # Update progress with error
            if task.task_id in self.download_progress:
                self.download_progress[task.task_id].status = f"error: {str(e)}"
                self._notify_progress(self.download_progress[task.task_id])
            
            return False
    
    def _prepare_headers(self, task: DownloadTask, resume_pos: int) -> Dict[str, str]:
        """Prepare optimized HTTP headers for download."""
        headers = {
            'User-Agent': 'ModelOrchestrator/1.0',
            'Accept': '*/*',
            'Connection': 'keep-alive'
        }
        
        # Add resume header if needed
        if resume_pos > 0:
            headers['Range'] = f'bytes={resume_pos}-'
        
        # Add compression support if enabled
        if self.enable_compression:
            headers['Accept-Encoding'] = 'gzip, deflate, br'
        
        return headers
    
    def _download_file_fresh(self, task: DownloadTask, progress: DownloadProgress) -> bool:
        """Download file from the beginning (used when resume fails)."""
        # Remove partial file
        if task.local_path.exists():
            task.local_path.unlink()
        
        # Reset progress
        progress.bytes_downloaded = 0
        
        # Download without resume
        headers = self._prepare_headers(task, 0)
        session = self.connection_pool.get_session()
        
        try:
            with session.get(task.source_url, headers=headers, stream=True, timeout=(30, 300)) as response:
                response.raise_for_status()
                
                chunk_size = self._adaptive_chunk_sizes.get(task.task_id, self.chunk_size)
                
                with open(task.local_path, 'wb', buffering=chunk_size * 4) as f:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            progress.bytes_downloaded += len(chunk)
            
            return progress.bytes_downloaded == task.size
            
        except Exception as e:
            logger.error(f"Fresh download failed for {task.file_path}: {e}")
            return False
    
    def _update_performance_metrics(self, task: DownloadTask, progress: DownloadProgress, queue_id: str):
        """Update performance metrics during download."""
        with self._lock:
            if queue_id in self.download_metrics:
                metrics = self.download_metrics[queue_id]
                
                # Update bandwidth limiting time
                if self.bandwidth_limiter:
                    # This is an approximation - actual implementation would track this more precisely
                    metrics.bandwidth_limited_time += 0.1  # Assume some bandwidth limiting
                
                # Detect congestion based on speed drops
                source_key = self._get_source_key(task.source_url)
                if source_key in self._speed_history and len(self._speed_history[source_key]) > 5:
                    recent_speeds = self._speed_history[source_key][-5:]
                    if progress.speed_bps < statistics.mean(recent_speeds) * 0.5:
                        self._congestion_detected[source_key] = True
                    else:
                        self._congestion_detected[source_key] = False
    
    def _download_file_basic(self, task: DownloadTask, progress: DownloadProgress, resume_pos: int) -> bool:
        """Fallback download method without connection pooling."""
        try:
            import urllib.request
            
            # Create request with resume support
            req = urllib.request.Request(task.source_url)
            if resume_pos > 0:
                req.add_header('Range', f'bytes={resume_pos}-')
            
            start_time = time.time()
            
            with urllib.request.urlopen(req) as response:
                mode = 'ab' if resume_pos > 0 else 'wb'
                with open(task.local_path, mode) as f:
                    while True:
                        chunk = response.read(self.chunk_size)
                        if not chunk:
                            break
                        
                        # Apply bandwidth limiting
                        if self.bandwidth_limiter:
                            wait_time = self.bandwidth_limiter.acquire_tokens(len(chunk))
                            if wait_time > 0:
                                time.sleep(wait_time)
                        
                        f.write(chunk)
                        progress.bytes_downloaded += len(chunk)
                        
                        # Update speed calculation
                        elapsed = time.time() - start_time
                        if elapsed > 0:
                            progress.speed_bps = progress.bytes_downloaded / elapsed
                        
                        self._notify_progress(progress)
            
            return True
            
        except Exception as e:
            logger.error(f"Basic download failed: {e}")
            return False
    
    def _notify_progress(self, progress: DownloadProgress):
        """Notify progress callbacks for a model."""
        if progress.model_id in self.progress_callbacks:
            for callback in self.progress_callbacks[progress.model_id]:
                try:
                    callback(progress)
                except Exception as e:
                    logger.warning(f"Progress callback failed: {e}")
    
    def get_download_stats(self) -> Dict[str, Any]:
        """Get comprehensive download statistics."""
        with self._lock:
            # Calculate aggregate metrics
            total_active = len(self.active_downloads)
            total_queued = sum(len(mq.tasks) - len(mq.active_tasks) - len(mq.completed_tasks) - len(mq.failed_tasks) 
                             for mq in self.model_queues.values())
            total_completed = len(self.completed_downloads)
            total_failed = len(self.failed_downloads)
            
            # Get connection pool stats
            pool_stats = self.connection_pool.get_stats()
            
            # Calculate performance metrics
            active_speeds = [p.speed_bps for p in self.download_progress.values() if p.speed_bps > 0]
            avg_speed = statistics.mean(active_speeds) if active_speeds else 0.0
            peak_speed = max(active_speeds) if active_speeds else 0.0
            
            return {
                "active_downloads": total_active,
                "queued_downloads": total_queued,
                "completed_downloads": total_completed,
                "failed_downloads": total_failed,
                "active_model_queues": len(self.model_queues),
                "max_concurrent": self.max_concurrent_downloads,
                "max_concurrent_files_per_model": self.max_concurrent_files_per_model,
                "bandwidth_limited": self.bandwidth_limiter is not None,
                "adaptive_chunking_enabled": self.enable_adaptive_chunking,
                "compression_enabled": self.enable_compression,
                "current_average_speed_bps": avg_speed,
                "current_peak_speed_bps": peak_speed,
                "connection_pool": pool_stats,
                "performance_history_sources": len(self._speed_history),
                "adaptive_chunk_sizes_active": len(self._adaptive_chunk_sizes)
            }
    
    def get_model_queue_status(self, queue_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific model download queue."""
        with self._lock:
            if queue_id not in self.model_queues:
                return None
            
            model_queue = self.model_queues[queue_id]
            metrics = self.download_metrics.get(queue_id)
            
            return {
                "queue_id": queue_id,
                "model_id": model_queue.model_id,
                "priority": model_queue.priority.name,
                "total_tasks": len(model_queue.tasks),
                "active_tasks": len(model_queue.active_tasks),
                "completed_tasks": len(model_queue.completed_tasks),
                "failed_tasks": len(model_queue.failed_tasks),
                "max_concurrent_files": model_queue.max_concurrent_files,
                "created_at": model_queue.created_at,
                "metrics": {
                    "total_bytes": metrics.total_bytes if metrics else 0,
                    "bytes_downloaded": metrics.bytes_downloaded if metrics else 0,
                    "duration": metrics.duration if metrics else 0,
                    "average_speed_bps": metrics.average_speed_bps if metrics else 0,
                    "peak_speed_bps": metrics.peak_speed_bps if metrics else 0,
                    "completion_rate": metrics.completion_rate if metrics else 0,
                    "retry_count": metrics.retry_count if metrics else 0
                } if metrics else None
            }
    
    def cancel_model_download(self, queue_id: str) -> bool:
        """Cancel a model download queue."""
        with self._lock:
            if queue_id not in self.model_queues:
                return False
            
            model_queue = self.model_queues[queue_id]
            
            # Mark all active tasks for cancellation
            # Note: This is a simplified implementation - full cancellation would require
            # more sophisticated coordination with the download threads
            for task_id in model_queue.active_tasks:
                if task_id in self.active_downloads:
                    # In a full implementation, we would signal the download thread to stop
                    logger.info(f"Cancelling download task: {task_id}")
            
            # Remove from queues
            del self.model_queues[queue_id]
            if queue_id in self.download_metrics:
                del self.download_metrics[queue_id]
            
            logger.info(f"Cancelled model download queue: {queue_id}")
            return True
    
    def pause_downloads(self):
        """Pause all downloads (for maintenance or resource management)."""
        # This would require more sophisticated implementation with proper thread coordination
        logger.info("Download pause requested - not fully implemented")
    
    def resume_downloads(self):
        """Resume paused downloads."""
        # This would require more sophisticated implementation with proper thread coordination
        logger.info("Download resume requested - not fully implemented")
    
    def optimize_performance(self):
        """Trigger performance optimization based on current conditions."""
        with self._lock:
            # Clean up stale sessions
            self.connection_pool.cleanup_stale_sessions()
            
            # Adjust chunk sizes based on performance history
            for source_key, speeds in self._speed_history.items():
                if len(speeds) > 10:
                    avg_speed = statistics.mean(speeds[-10:])
                    
                    # Adjust default chunk size for this source
                    if avg_speed > 100 * 1024 * 1024:  # > 100MB/s
                        optimal_chunk = 2 * 1024 * 1024  # 2MB
                    elif avg_speed > 50 * 1024 * 1024:  # > 50MB/s
                        optimal_chunk = 1024 * 1024  # 1MB
                    elif avg_speed > 10 * 1024 * 1024:  # > 10MB/s
                        optimal_chunk = 512 * 1024  # 512KB
                    else:
                        optimal_chunk = 256 * 1024  # 256KB
                    
                    # Update chunk sizes for active downloads from this source
                    for task_id, task in self.active_downloads.items():
                        if self._get_source_key(task.source_url) == source_key:
                            self._adaptive_chunk_sizes[task_id] = optimal_chunk
            
            logger.info("Performance optimization completed")
    
    def shutdown(self):
        """Shutdown the download manager gracefully."""
        logger.info("Shutting down download manager...")
        self._shutdown = True
        
        # Wait for worker threads to finish
        if hasattr(self, '_queue_worker_thread') and self._queue_worker_thread.is_alive():
            self._queue_worker_thread.join(timeout=5.0)
        
        if hasattr(self, '_metrics_worker_thread') and self._metrics_worker_thread.is_alive():
            self._metrics_worker_thread.join(timeout=5.0)
        
        # Shutdown executors
        self.executor.shutdown(wait=True)
        self.io_executor.shutdown(wait=True)
        
        # Close connection pool
        self.connection_pool.close()
        
        # Clear all tracking data
        with self._lock:
            self.model_queues.clear()
            self.active_downloads.clear()
            self.completed_downloads.clear()
            self.failed_downloads.clear()
            self.download_progress.clear()
            self.download_metrics.clear()
            self._adaptive_chunk_sizes.clear()
            self._speed_history.clear()
            self._congestion_detected.clear()
        
        logger.info("Download manager shutdown complete")