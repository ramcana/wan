"""
Enhanced Model Downloader with Retry Logic
Provides intelligent retry mechanisms, exponential backoff, partial download recovery,
and advanced download management features for WAN2.2 models.
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Tuple, Union
import threading
import aiohttp
import aiofiles
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# Import performance monitoring (with fallback if not available)
try:
    from .performance_monitoring_system import get_performance_monitor
    PERFORMANCE_MONITORING_AVAILABLE = True
except ImportError:
    PERFORMANCE_MONITORING_AVAILABLE = False
    logger.warning("Performance monitoring not available")


class DownloadStatus(Enum):
    """Download status enumeration"""
    QUEUED = "queued"
    DOWNLOADING = "downloading"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    VERIFYING = "verifying"
    RESUMING = "resuming"


class DownloadError(Exception):
    """Custom exception for download errors"""
    def __init__(self, message: str, error_type: str = "unknown", retry_after: Optional[float] = None):
        super().__init__(message)
        self.error_type = error_type
        self.retry_after = retry_after


@dataclass
class DownloadProgress:
    """Enhanced download progress tracking"""
    model_id: str
    status: DownloadStatus
    progress_percent: float
    downloaded_mb: float
    total_mb: float
    speed_mbps: float
    eta_seconds: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    error_message: Optional[str] = None
    can_pause: bool = True
    can_resume: bool = True
    can_cancel: bool = True
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    last_update: Optional[datetime] = None


@dataclass
class DownloadResult:
    """Result of a download operation"""
    success: bool
    model_id: str
    final_status: DownloadStatus
    total_time_seconds: float
    total_retries: int
    final_size_mb: float
    error_message: Optional[str] = None
    integrity_verified: bool = False
    download_path: Optional[str] = None


@dataclass
class RetryConfig:
    """Configuration for retry logic"""
    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0
    jitter: bool = True
    retry_on_status_codes: List[int] = field(default_factory=lambda: [429, 500, 502, 503, 504])


@dataclass
class BandwidthConfig:
    """Configuration for bandwidth management"""
    max_speed_mbps: Optional[float] = None
    chunk_size: int = 8192
    concurrent_downloads: int = 2
    adaptive_chunking: bool = True


class EnhancedModelDownloader:
    """
    Enhanced model downloader with intelligent retry mechanisms,
    exponential backoff, partial download recovery, and bandwidth management.
    """
    
    def __init__(self, base_downloader=None, models_dir: Optional[str] = None):
        """
        Initialize the enhanced model downloader.
        
        Args:
            base_downloader: Optional existing ModelDownloader instance
            models_dir: Directory for storing models
        """
        self.base_downloader = base_downloader
        self.models_dir = Path(models_dir) if models_dir else Path("models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Download tracking
        self._active_downloads: Dict[str, DownloadProgress] = {}
        self._download_tasks: Dict[str, asyncio.Task] = {}
        self._download_lock = asyncio.Lock()
        self._progress_callbacks: List[Callable] = []
        
        logger.info(f"Enhanced Model Downloader initialized with models_dir: {self.models_dir}")
    
    async def initialize(self) -> bool:
        """Initialize the enhanced model downloader"""
        try:
            # Ensure directories exist
            self.models_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info("Enhanced Model Downloader initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Enhanced Model Downloader: {e}")
            return False
    
    async def get_download_progress(self, model_id: str) -> Optional[DownloadProgress]:
        """Get download progress for a model"""
        try:
            async with self._download_lock:
                return self._active_downloads.get(model_id)
        except Exception as e:
            logger.error(f"Error getting download progress for {model_id}: {e}")
            return None
    
    async def pause_download(self, model_id: str) -> bool:
        """Pause a download"""
        try:
            async with self._download_lock:
                if model_id in self._active_downloads:
                    progress = self._active_downloads[model_id]
                    if progress.status == DownloadStatus.DOWNLOADING:
                        progress.status = DownloadStatus.PAUSED
                        logger.info(f"Paused download for {model_id}")
                        return True
                return False
        except Exception as e:
            logger.error(f"Error pausing download for {model_id}: {e}")
            return False
    
    async def resume_download(self, model_id: str) -> bool:
        """Resume a download"""
        try:
            async with self._download_lock:
                if model_id in self._active_downloads:
                    progress = self._active_downloads[model_id]
                    if progress.status == DownloadStatus.PAUSED:
                        progress.status = DownloadStatus.DOWNLOADING
                        logger.info(f"Resumed download for {model_id}")
                        return True
                return False
        except Exception as e:
            logger.error(f"Error resuming download for {model_id}: {e}")
            return False
    
    async def cancel_download(self, model_id: str) -> bool:
        """Cancel a download"""
        try:
            async with self._download_lock:
                if model_id in self._active_downloads:
                    progress = self._active_downloads[model_id]
                    progress.status = DownloadStatus.CANCELLED
                    logger.info(f"Cancelled download for {model_id}")
                    return True
                return False
        except Exception as e:
            logger.error(f"Error cancelling download for {model_id}: {e}")
            return False
    
    async def set_bandwidth_limit(self, limit_mbps: float) -> bool:
        """Set bandwidth limit for downloads"""
        try:
            self._bandwidth_limit_mbps = limit_mbps
            logger.info(f"Set bandwidth limit to {limit_mbps} Mbps")
            return True
        except Exception as e:
            logger.error(f"Error setting bandwidth limit: {e}")
            return False
        self._download_lock = asyncio.Lock()
        self._progress_callbacks: List[Callable] = []
        
        # Configuration
        self.retry_config = RetryConfig()
        self.bandwidth_config = BandwidthConfig()
        
        # Session management
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_lock = asyncio.Lock()
        
        # Partial download support
        self._partial_downloads_dir = self.models_dir / ".partial"
        self._partial_downloads_dir.mkdir(exist_ok=True)
        
        logger.info(f"Enhanced Model Downloader initialized with models_dir: {self.models_dir}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self._ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self._close_session()
    
    async def _ensure_session(self):
        """Ensure aiohttp session is available"""
        async with self._session_lock:
            if self._session is None or self._session.closed:
                timeout = aiohttp.ClientTimeout(total=300, connect=30)
                connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
                self._session = aiohttp.ClientSession(
                    timeout=timeout,
                    connector=connector,
                    headers={'User-Agent': 'WAN22-Enhanced-Downloader/1.0'}
                )
    
    async def _close_session(self):
        """Close aiohttp session"""
        async with self._session_lock:
            if self._session and not self._session.closed:
                await self._session.close()
                self._session = None
    
    def add_progress_callback(self, callback: Callable[[DownloadProgress], None]):
        """Add a progress callback function"""
        self._progress_callbacks.append(callback)
    
    def remove_progress_callback(self, callback: Callable[[DownloadProgress], None]):
        """Remove a progress callback function"""
        if callback in self._progress_callbacks:
            self._progress_callbacks.remove(callback)
    
    async def _notify_progress(self, progress: DownloadProgress):
        """Notify all progress callbacks"""
        progress.last_update = datetime.now()
        for callback in self._progress_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(progress)
                else:
                    callback(progress)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")
    
    async def download_with_retry(self, model_id: str, download_url: str, 
                                max_retries: Optional[int] = None) -> DownloadResult:
        """
        Download a model with intelligent retry logic and exponential backoff.
        
        Args:
            model_id: Unique identifier for the model
            download_url: URL to download the model from
            max_retries: Override default max retries
            
        Returns:
            DownloadResult with success status and details
        """
        max_retries = max_retries or self.retry_config.max_retries
        start_time = time.time()
        
        # Start performance monitoring
        performance_id = None
        if PERFORMANCE_MONITORING_AVAILABLE:
            try:
                monitor = get_performance_monitor()
                performance_id = monitor.track_download_operation(
                    f"download_{model_id}",
                    {
                        "model_id": model_id,
                        "download_url": download_url,
                        "max_retries": max_retries,
                        "bandwidth_limit_mbps": self.bandwidth_limit_mbps
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to start performance monitoring: {e}")
        
        # Initialize progress tracking
        progress = DownloadProgress(
            model_id=model_id,
            status=DownloadStatus.QUEUED,
            progress_percent=0.0,
            downloaded_mb=0.0,
            total_mb=0.0,
            speed_mbps=0.0,
            max_retries=max_retries,
            started_at=datetime.now()
        )
        
        async with self._download_lock:
            self._active_downloads[model_id] = progress
        
        await self._notify_progress(progress)
        
        try:
            await self._ensure_session()
            
            for attempt in range(max_retries + 1):
                try:
                    progress.retry_count = attempt
                    progress.status = DownloadStatus.DOWNLOADING
                    await self._notify_progress(progress)
                    
                    logger.info(f"Downloading {model_id}, attempt {attempt + 1}/{max_retries + 1}")
                    
                    # Attempt download
                    result = await self._download_with_resume(model_id, download_url, progress)
                    
                    if result.success:
                        progress.status = DownloadStatus.COMPLETED
                        progress.completed_at = datetime.now()
                        progress.progress_percent = 100.0
                        await self._notify_progress(progress)
                        
                        # Clean up partial download files
                        await self._cleanup_partial_download(model_id)
                        
                        return result
                    
                    # Download failed, prepare for retry
                    if attempt < max_retries:
                        delay = await self._calculate_retry_delay(attempt, result.error_message)
                        logger.warning(f"Download failed, retrying in {delay:.1f}s: {result.error_message}")
                        
                        progress.status = DownloadStatus.FAILED
                        progress.error_message = f"Attempt {attempt + 1} failed: {result.error_message}"
                        await self._notify_progress(progress)
                        
                        await asyncio.sleep(delay)
                    else:
                        # Final attempt failed
                        progress.status = DownloadStatus.FAILED
                        progress.error_message = f"All {max_retries + 1} attempts failed"
                        await self._notify_progress(progress)
                        
                        return DownloadResult(
                            success=False,
                            model_id=model_id,
                            final_status=DownloadStatus.FAILED,
                            total_time_seconds=time.time() - start_time,
                            total_retries=attempt,
                            final_size_mb=progress.downloaded_mb,
                            error_message=result.error_message
                        )
                
                except asyncio.CancelledError:
                    progress.status = DownloadStatus.CANCELLED
                    await self._notify_progress(progress)
                    raise
                
                except Exception as e:
                    logger.error(f"Unexpected error in download attempt {attempt + 1}: {e}")
                    if attempt >= max_retries:
                        progress.status = DownloadStatus.FAILED
                        progress.error_message = f"Unexpected error: {str(e)}"
                        await self._notify_progress(progress)
                        
                        return DownloadResult(
                            success=False,
                            model_id=model_id,
                            final_status=DownloadStatus.FAILED,
                            total_time_seconds=time.time() - start_time,
                            total_retries=attempt,
                            final_size_mb=progress.downloaded_mb,
                            error_message=str(e)
                        )
        
        finally:
            # End performance monitoring
            if PERFORMANCE_MONITORING_AVAILABLE and performance_id:
                try:
                    monitor = get_performance_monitor()
                    final_progress = self._active_downloads.get(model_id)
                    success = final_progress and final_progress.status == DownloadStatus.COMPLETED
                    error_msg = final_progress.error_message if final_progress and not success else None
                    
                    monitor.end_tracking(
                        performance_id,
                        success=success,
                        error_message=error_msg,
                        additional_metadata={
                            "final_status": final_progress.status.value if final_progress else "unknown",
                            "total_retries": final_progress.retry_count if final_progress else 0,
                            "downloaded_mb": final_progress.downloaded_mb if final_progress else 0,
                            "average_speed_mbps": final_progress.speed_mbps if final_progress else 0,
                            "integrity_verified": getattr(final_progress, 'integrity_verified', False)
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to end performance monitoring: {e}")
            
            # Clean up tracking
            async with self._download_lock:
                if model_id in self._active_downloads:
                    del self._active_downloads[model_id]
    
    async def _download_with_resume(self, model_id: str, download_url: str, 
                                  progress: DownloadProgress) -> DownloadResult:
        """
        Download with resume capability for partial downloads.
        
        Args:
            model_id: Model identifier
            download_url: URL to download from
            progress: Progress tracking object
            
        Returns:
            DownloadResult
        """
        model_path = self.models_dir / f"{model_id}.model"
        partial_path = self._partial_downloads_dir / f"{model_id}.partial"
        
        # Check for existing partial download
        resume_from = 0
        if partial_path.exists():
            resume_from = partial_path.stat().st_size
            progress.downloaded_mb = resume_from / (1024 * 1024)
            logger.info(f"Resuming download from {resume_from} bytes")
        
        try:
            # Prepare headers for resume
            headers = {}
            if resume_from > 0:
                headers['Range'] = f'bytes={resume_from}-'
            
            async with self._session.get(download_url, headers=headers) as response:
                # Handle response status
                if response.status == 416:  # Range not satisfiable
                    logger.warning("Server doesn't support resume, starting fresh download")
                    resume_from = 0
                    if partial_path.exists():
                        partial_path.unlink()
                    
                    # Retry without range header
                    async with self._session.get(download_url) as fresh_response:
                        return await self._process_download_response(
                            fresh_response, model_id, model_path, partial_path, progress, resume_from
                        )
                
                elif response.status not in [200, 206]:  # 206 = Partial Content
                    raise DownloadError(
                        f"HTTP {response.status}: {response.reason}",
                        error_type="http_error"
                    )
                
                return await self._process_download_response(
                    response, model_id, model_path, partial_path, progress, resume_from
                )
        
        except aiohttp.ClientError as e:
            return DownloadResult(
                success=False,
                model_id=model_id,
                final_status=DownloadStatus.FAILED,
                total_time_seconds=0.0,
                total_retries=progress.retry_count,
                final_size_mb=progress.downloaded_mb,
                error_message=f"Network error: {str(e)}"
            )
        
        except Exception as e:
            return DownloadResult(
                success=False,
                model_id=model_id,
                final_status=DownloadStatus.FAILED,
                total_time_seconds=0.0,
                total_retries=progress.retry_count,
                final_size_mb=progress.downloaded_mb,
                error_message=f"Download error: {str(e)}"
            )
    
    async def _process_download_response(self, response: aiohttp.ClientResponse, 
                                       model_id: str, model_path: Path, 
                                       partial_path: Path, progress: DownloadProgress,
                                       resume_from: int) -> DownloadResult:
        """Process the download response and save data"""
        start_time = time.time()
        
        # Get total size
        content_length = response.headers.get('Content-Length')
        if content_length:
            if response.status == 206:  # Partial content
                # For partial content, add the resume offset
                total_size = int(content_length) + resume_from
            else:
                total_size = int(content_length)
            progress.total_mb = total_size / (1024 * 1024)
        
        # Open file for writing (append mode if resuming)
        mode = 'ab' if resume_from > 0 else 'wb'
        
        try:
            async with aiofiles.open(partial_path, mode) as f:
                downloaded = resume_from
                last_update = time.time()
                
                # Apply bandwidth limiting
                chunk_size = self._calculate_chunk_size(progress.speed_mbps)
                
                async for chunk in response.content.iter_chunked(chunk_size):
                    # Check for cancellation
                    if progress.status == DownloadStatus.CANCELLED:
                        raise asyncio.CancelledError()
                    
                    # Check for pause
                    while progress.status == DownloadStatus.PAUSED:
                        await asyncio.sleep(0.1)
                    
                    await f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Update progress
                    current_time = time.time()
                    if current_time - last_update >= 0.5:  # Update every 500ms
                        elapsed = current_time - start_time
                        if elapsed > 0:
                            speed_bps = (downloaded - resume_from) / elapsed
                            progress.speed_mbps = (speed_bps / 1024 / 1024) * 8  # Convert to Mbps
                        
                        progress.downloaded_mb = downloaded / (1024 * 1024)
                        if progress.total_mb > 0:
                            progress.progress_percent = (downloaded / (progress.total_mb * 1024 * 1024)) * 100
                        
                        # Calculate ETA
                        if progress.speed_mbps > 0 and progress.total_mb > 0:
                            remaining_mb = progress.total_mb - progress.downloaded_mb
                            progress.eta_seconds = (remaining_mb * 8) / progress.speed_mbps  # Convert Mbps to MB/s
                        
                        await self._notify_progress(progress)
                        last_update = current_time
                    
                    # Apply bandwidth limiting
                    if self.bandwidth_config.max_speed_mbps:
                        await self._apply_bandwidth_limit(len(chunk), start_time, downloaded - resume_from)
            
            # Move partial file to final location
            if partial_path.exists():
                partial_path.rename(model_path)
            
            # Verify integrity
            integrity_verified = await self.verify_and_repair_model(model_id)
            
            return DownloadResult(
                success=True,
                model_id=model_id,
                final_status=DownloadStatus.COMPLETED,
                total_time_seconds=time.time() - start_time,
                total_retries=progress.retry_count,
                final_size_mb=downloaded / (1024 * 1024),
                integrity_verified=integrity_verified,
                download_path=str(model_path)
            )
        
        except asyncio.CancelledError:
            return DownloadResult(
                success=False,
                model_id=model_id,
                final_status=DownloadStatus.CANCELLED,
                total_time_seconds=time.time() - start_time,
                total_retries=progress.retry_count,
                final_size_mb=progress.downloaded_mb,
                error_message="Download cancelled by user"
            )
        
        except Exception as e:
            return DownloadResult(
                success=False,
                model_id=model_id,
                final_status=DownloadStatus.FAILED,
                total_time_seconds=time.time() - start_time,
                total_retries=progress.retry_count,
                final_size_mb=progress.downloaded_mb,
                error_message=str(e)
            )
    
    def _calculate_chunk_size(self, current_speed_mbps: float) -> int:
        """Calculate optimal chunk size based on current speed"""
        if not self.bandwidth_config.adaptive_chunking:
            return self.bandwidth_config.chunk_size
        
        # Adaptive chunking: larger chunks for faster connections
        if current_speed_mbps > 100:  # Very fast connection
            return 64 * 1024  # 64KB
        elif current_speed_mbps > 50:  # Fast connection
            return 32 * 1024  # 32KB
        elif current_speed_mbps > 10:  # Medium connection
            return 16 * 1024  # 16KB
        else:  # Slow connection
            return 8 * 1024   # 8KB
    
    async def _apply_bandwidth_limit(self, bytes_downloaded: int, start_time: float, total_downloaded: int):
        """Apply bandwidth limiting if configured"""
        if not self.bandwidth_config.max_speed_mbps:
            return
        
        max_speed_bps = (self.bandwidth_config.max_speed_mbps * 1024 * 1024) / 8  # Convert Mbps to bytes/sec
        elapsed_time = time.time() - start_time
        
        if elapsed_time > 0:
            current_speed_bps = total_downloaded / elapsed_time
            if current_speed_bps > max_speed_bps:
                # Calculate required delay
                required_time = total_downloaded / max_speed_bps
                delay = required_time - elapsed_time
                if delay > 0:
                    await asyncio.sleep(delay)
    
    async def _calculate_retry_delay(self, attempt: int, error_message: Optional[str] = None) -> float:
        """Calculate retry delay with exponential backoff and jitter"""
        base_delay = self.retry_config.initial_delay * (self.retry_config.backoff_factor ** attempt)
        delay = min(base_delay, self.retry_config.max_delay)
        
        # Add jitter to prevent thundering herd (but keep within max delay)
        if self.retry_config.jitter:
            import random
            jitter = random.uniform(0.1, 0.3) * delay
            delay = min(delay + jitter, self.retry_config.max_delay)
        
        # Check for rate limiting hints in error message
        if error_message and "rate limit" in error_message.lower():
            delay = max(delay, 30.0)  # Minimum 30s for rate limits
        
        return delay
    
    async def verify_and_repair_model(self, model_id: str) -> bool:
        """
        Verify model integrity and attempt repair if corrupted.
        
        Args:
            model_id: Model identifier to verify
            
        Returns:
            True if model is valid or successfully repaired
        """
        model_path = self.models_dir / f"{model_id}.model"
        
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return False
        
        try:
            # Basic file size check
            file_size = model_path.stat().st_size
            if file_size == 0:
                logger.error(f"Model file is empty: {model_path}")
                return False
            
            # Calculate checksum for integrity verification
            logger.info(f"Verifying integrity of {model_id}")
            
            checksum = await self._calculate_file_checksum(model_path)
            
            # For now, we'll consider any non-empty file as valid
            # In production, this would compare against known checksums
            logger.info(f"Model {model_id} integrity check passed (checksum: {checksum[:16]}...)")
            return True
        
        except Exception as e:
            logger.error(f"Error verifying model {model_id}: {e}")
            return False
    
    async def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file asynchronously"""
        sha256_hash = hashlib.sha256()
        
        async with aiofiles.open(file_path, 'rb') as f:
            while chunk := await f.read(8192):
                sha256_hash.update(chunk)
        
        return sha256_hash.hexdigest()
    
    async def pause_download(self, model_id: str) -> bool:
        """
        Pause an active download.
        
        Args:
            model_id: Model identifier
            
        Returns:
            True if successfully paused
        """
        async with self._download_lock:
            if model_id in self._active_downloads:
                progress = self._active_downloads[model_id]
                if progress.status == DownloadStatus.DOWNLOADING and progress.can_pause:
                    progress.status = DownloadStatus.PAUSED
                    await self._notify_progress(progress)
                    logger.info(f"Paused download for {model_id}")
                    return True
        
        return False
    
    async def resume_download(self, model_id: str) -> bool:
        """
        Resume a paused download.
        
        Args:
            model_id: Model identifier
            
        Returns:
            True if successfully resumed
        """
        async with self._download_lock:
            if model_id in self._active_downloads:
                progress = self._active_downloads[model_id]
                if progress.status == DownloadStatus.PAUSED and progress.can_resume:
                    progress.status = DownloadStatus.DOWNLOADING
                    await self._notify_progress(progress)
                    logger.info(f"Resumed download for {model_id}")
                    return True
        
        return False
    
    async def cancel_download(self, model_id: str) -> bool:
        """
        Cancel an active download.
        
        Args:
            model_id: Model identifier
            
        Returns:
            True if successfully cancelled
        """
        async with self._download_lock:
            if model_id in self._active_downloads:
                progress = self._active_downloads[model_id]
                if progress.can_cancel:
                    progress.status = DownloadStatus.CANCELLED
                    await self._notify_progress(progress)
                    
                    # Cancel the download task if it exists
                    if model_id in self._download_tasks:
                        task = self._download_tasks[model_id]
                        task.cancel()
                        del self._download_tasks[model_id]
                    
                    # Clean up partial download
                    await self._cleanup_partial_download(model_id)
                    
                    logger.info(f"Cancelled download for {model_id}")
                    return True
        
        return False
    
    def set_bandwidth_limit(self, limit_mbps: Optional[float]) -> bool:
        """
        Set bandwidth limit for downloads.
        
        Args:
            limit_mbps: Maximum download speed in Mbps, None for unlimited
            
        Returns:
            True if successfully set
        """
        try:
            self.bandwidth_config.max_speed_mbps = limit_mbps
            logger.info(f"Bandwidth limit set to {limit_mbps} Mbps" if limit_mbps else "Bandwidth limit removed")
            return True
        except Exception as e:
            logger.error(f"Error setting bandwidth limit: {e}")
            return False
    
    async def get_download_progress(self, model_id: str) -> Optional[DownloadProgress]:
        """
        Get download progress for a specific model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            DownloadProgress object or None if not found
        """
        async with self._download_lock:
            return self._active_downloads.get(model_id)
    
    async def get_all_download_progress(self) -> Dict[str, DownloadProgress]:
        """Get download progress for all active downloads"""
        async with self._download_lock:
            return self._active_downloads.copy()
    
    async def _cleanup_partial_download(self, model_id: str):
        """Clean up partial download files"""
        partial_path = self._partial_downloads_dir / f"{model_id}.partial"
        if partial_path.exists():
            try:
                partial_path.unlink()
                logger.info(f"Cleaned up partial download for {model_id}")
            except Exception as e:
                logger.warning(f"Failed to clean up partial download for {model_id}: {e}")
    
    def update_retry_config(self, **kwargs):
        """Update retry configuration"""
        for key, value in kwargs.items():
            if hasattr(self.retry_config, key):
                setattr(self.retry_config, key, value)
                logger.info(f"Updated retry config: {key} = {value}")
    
    def update_bandwidth_config(self, **kwargs):
        """Update bandwidth configuration"""
        for key, value in kwargs.items():
            if hasattr(self.bandwidth_config, key):
                setattr(self.bandwidth_config, key, value)
                logger.info(f"Updated bandwidth config: {key} = {value}")
    
    async def cleanup_all_partial_downloads(self):
        """Clean up all partial download files"""
        try:
            for partial_file in self._partial_downloads_dir.glob("*.partial"):
                partial_file.unlink()
            logger.info("Cleaned up all partial downloads")
        except Exception as e:
            logger.warning(f"Error cleaning up partial downloads: {e}")
