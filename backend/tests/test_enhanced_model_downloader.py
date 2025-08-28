"""
Comprehensive unit tests for Enhanced Model Downloader
Tests retry logic, download management, bandwidth limiting, and error handling.
"""

import asyncio
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import aiohttp
import aiofiles
from datetime import datetime, timedelta

# Import the enhanced model downloader
import sys
sys.path.append(str(Path(__file__).parent.parent))

from backend.core.enhanced_model_downloader import (
    EnhancedModelDownloader,
    DownloadStatus,
    DownloadProgress,
    DownloadResult,
    RetryConfig,
    BandwidthConfig,
    DownloadError
)


class TestEnhancedModelDownloader:
    """Test suite for Enhanced Model Downloader"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def downloader(self, temp_dir):
        """Create downloader instance for testing"""
        return EnhancedModelDownloader(models_dir=temp_dir)
    
    @pytest.fixture
    def mock_response(self):
        """Create mock aiohttp response"""
        response = AsyncMock(spec=aiohttp.ClientResponse)
        response.status = 200
        response.headers = {'Content-Length': '1048576'}  # 1MB
        response.content.iter_chunked = AsyncMock()
        return response
    
    @pytest.mark.asyncio
    async def test_initialization(self, temp_dir):
        """Test downloader initialization"""
        downloader = EnhancedModelDownloader(models_dir=temp_dir)
        
        assert downloader.models_dir == Path(temp_dir)
        assert downloader.models_dir.exists()
        assert downloader._partial_downloads_dir.exists()
        assert isinstance(downloader.retry_config, RetryConfig)
        assert isinstance(downloader.bandwidth_config, BandwidthConfig)
        assert downloader._active_downloads == {}
    
    @pytest.mark.asyncio
    async def test_session_management(self, downloader):
        """Test aiohttp session management"""
        # Session should be created when needed
        await downloader._ensure_session()
        assert downloader._session is not None
        assert not downloader._session.closed
        
        # Session should be properly closed
        await downloader._close_session()
        assert downloader._session is None
    
    @pytest.mark.asyncio
    async def test_progress_callbacks(self, downloader):
        """Test progress callback functionality"""
        callback_calls = []
        
        def progress_callback(progress):
            callback_calls.append(progress)
        
        # Add callback
        downloader.add_progress_callback(progress_callback)
        assert progress_callback in downloader._progress_callbacks
        
        # Test notification
        progress = DownloadProgress(
            model_id="test_model",
            status=DownloadStatus.DOWNLOADING,
            progress_percent=50.0,
            downloaded_mb=5.0,
            total_mb=10.0,
            speed_mbps=10.0
        )
        
        await downloader._notify_progress(progress)
        assert len(callback_calls) == 1
        assert callback_calls[0].model_id == "test_model"
        
        # Remove callback
        downloader.remove_progress_callback(progress_callback)
        assert progress_callback not in downloader._progress_callbacks
    
    @pytest.mark.asyncio
    async def test_retry_delay_calculation(self, downloader):
        """Test exponential backoff retry delay calculation"""
        # Test basic exponential backoff
        delay1 = await downloader._calculate_retry_delay(0)
        delay2 = await downloader._calculate_retry_delay(1)
        delay3 = await downloader._calculate_retry_delay(2)
        
        assert delay1 >= downloader.retry_config.initial_delay
        assert delay2 > delay1
        assert delay3 > delay2
        
        # Test max delay cap
        large_attempt_delay = await downloader._calculate_retry_delay(10)
        assert large_attempt_delay <= downloader.retry_config.max_delay
        
        # Test rate limit handling
        rate_limit_delay = await downloader._calculate_retry_delay(0, "rate limit exceeded")
        assert rate_limit_delay >= 30.0
    
    @pytest.mark.asyncio
    async def test_chunk_size_calculation(self, downloader):
        """Test adaptive chunk size calculation"""
        # Test different speed scenarios
        slow_chunk = downloader._calculate_chunk_size(5.0)  # 5 Mbps
        medium_chunk = downloader._calculate_chunk_size(25.0)  # 25 Mbps
        fast_chunk = downloader._calculate_chunk_size(75.0)  # 75 Mbps
        very_fast_chunk = downloader._calculate_chunk_size(150.0)  # 150 Mbps
        
        assert slow_chunk == 8 * 1024  # 8KB
        assert medium_chunk == 16 * 1024  # 16KB
        assert fast_chunk == 32 * 1024  # 32KB
        assert very_fast_chunk == 64 * 1024  # 64KB
        
        # Test with adaptive chunking disabled
        downloader.bandwidth_config.adaptive_chunking = False
        fixed_chunk = downloader._calculate_chunk_size(100.0)
        assert fixed_chunk == downloader.bandwidth_config.chunk_size
    
    @pytest.mark.asyncio
    async def test_bandwidth_limiting(self, downloader):
        """Test bandwidth limiting functionality"""
        # Set bandwidth limit
        assert downloader.set_bandwidth_limit(10.0)  # 10 Mbps
        assert downloader.bandwidth_config.max_speed_mbps == 10.0
        
        # Test bandwidth limit application
        start_time = asyncio.get_event_loop().time()
        
        # Simulate downloading 1MB in 0.1 seconds (80 Mbps)
        # Should be throttled to 10 Mbps
        with patch('time.time', side_effect=[start_time, start_time + 0.1]):
            await downloader._apply_bandwidth_limit(1024*1024, start_time, 1024*1024)
        
        # Remove bandwidth limit
        assert downloader.set_bandwidth_limit(None)
        assert downloader.bandwidth_config.max_speed_mbps is None
    
    @pytest.mark.asyncio
    async def test_file_checksum_calculation(self, downloader, temp_dir):
        """Test file checksum calculation"""
        # Create test file
        test_file = Path(temp_dir) / "test_file.txt"
        test_content = b"Hello, World!" * 1000
        
        async with aiofiles.open(test_file, 'wb') as f:
            await f.write(test_content)
        
        # Calculate checksum
        checksum = await downloader._calculate_file_checksum(test_file)
        
        assert isinstance(checksum, str)
        assert len(checksum) == 64  # SHA256 hex digest length
        
        # Verify checksum is consistent
        checksum2 = await downloader._calculate_file_checksum(test_file)
        assert checksum == checksum2
    
    @pytest.mark.asyncio
    async def test_verify_and_repair_model(self, downloader, temp_dir):
        """Test model integrity verification"""
        model_id = "test_model"
        model_path = Path(temp_dir) / f"{model_id}.model"
        
        # Test with non-existent file
        result = await downloader.verify_and_repair_model(model_id)
        assert result is False
        
        # Test with empty file
        model_path.touch()
        result = await downloader.verify_and_repair_model(model_id)
        assert result is False
        
        # Test with valid file
        async with aiofiles.open(model_path, 'wb') as f:
            await f.write(b"Valid model data" * 1000)
        
        result = await downloader.verify_and_repair_model(model_id)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_download_progress_tracking(self, downloader):
        """Test download progress tracking"""
        model_id = "test_model"
        
        # Initially no progress
        progress = await downloader.get_download_progress(model_id)
        assert progress is None
        
        # Add progress
        test_progress = DownloadProgress(
            model_id=model_id,
            status=DownloadStatus.DOWNLOADING,
            progress_percent=25.0,
            downloaded_mb=2.5,
            total_mb=10.0,
            speed_mbps=5.0
        )
        
        async with downloader._download_lock:
            downloader._active_downloads[model_id] = test_progress
        
        # Retrieve progress
        retrieved_progress = await downloader.get_download_progress(model_id)
        assert retrieved_progress is not None
        assert retrieved_progress.model_id == model_id
        assert retrieved_progress.progress_percent == 25.0
        
        # Get all progress
        all_progress = await downloader.get_all_download_progress()
        assert model_id in all_progress
        assert len(all_progress) == 1
    
    @pytest.mark.asyncio
    async def test_pause_resume_cancel_download(self, downloader):
        """Test download control operations"""
        model_id = "test_model"
        
        # Create active download
        progress = DownloadProgress(
            model_id=model_id,
            status=DownloadStatus.DOWNLOADING,
            progress_percent=50.0,
            downloaded_mb=5.0,
            total_mb=10.0,
            speed_mbps=10.0,
            can_pause=True,
            can_resume=True,
            can_cancel=True
        )
        
        async with downloader._download_lock:
            downloader._active_downloads[model_id] = progress
        
        # Test pause
        result = await downloader.pause_download(model_id)
        assert result is True
        assert progress.status == DownloadStatus.PAUSED
        
        # Test resume
        result = await downloader.resume_download(model_id)
        assert result is True
        assert progress.status == DownloadStatus.DOWNLOADING
        
        # Test cancel
        result = await downloader.cancel_download(model_id)
        assert result is True
        assert progress.status == DownloadStatus.CANCELLED
        
        # Test operations on non-existent download
        result = await downloader.pause_download("non_existent")
        assert result is False
    
    @pytest.mark.asyncio
    async def test_partial_download_cleanup(self, downloader, temp_dir):
        """Test partial download cleanup"""
        model_id = "test_model"
        partial_path = downloader._partial_downloads_dir / f"{model_id}.partial"
        
        # Create partial file
        async with aiofiles.open(partial_path, 'wb') as f:
            await f.write(b"partial data")
        
        assert partial_path.exists()
        
        # Clean up specific partial download
        await downloader._cleanup_partial_download(model_id)
        assert not partial_path.exists()
        
        # Test cleanup all partial downloads
        # Create multiple partial files
        for i in range(3):
            partial_file = downloader._partial_downloads_dir / f"model_{i}.partial"
            async with aiofiles.open(partial_file, 'wb') as f:
                await f.write(b"partial data")
        
        await downloader.cleanup_all_partial_downloads()
        
        # Check all partial files are removed
        partial_files = list(downloader._partial_downloads_dir.glob("*.partial"))
        assert len(partial_files) == 0
    
    @pytest.mark.asyncio
    async def test_config_updates(self, downloader):
        """Test configuration updates"""
        # Test retry config update
        original_max_retries = downloader.retry_config.max_retries
        downloader.update_retry_config(max_retries=5, initial_delay=2.0)
        
        assert downloader.retry_config.max_retries == 5
        assert downloader.retry_config.initial_delay == 2.0
        
        # Test bandwidth config update
        downloader.update_bandwidth_config(max_speed_mbps=50.0, chunk_size=16384)
        
        assert downloader.bandwidth_config.max_speed_mbps == 50.0
        assert downloader.bandwidth_config.chunk_size == 16384
        
        # Test invalid config update (should be ignored)
        downloader.update_retry_config(invalid_param="value")
        # Should not raise error, just ignore invalid parameter
    
    @pytest.mark.asyncio
    async def test_download_with_retry_success(self, downloader, mock_response):
        """Test successful download with retry logic"""
        model_id = "test_model"
        download_url = "https://example.com/model.bin"
        
        # Mock successful response
        mock_response.content.iter_chunked.return_value = [
            b"chunk1" * 256,  # 1.5KB chunk
            b"chunk2" * 256,  # 1.5KB chunk
            b"chunk3" * 256   # 1.5KB chunk
        ].__iter__()
        
        with patch.object(downloader, '_session') as mock_session:
            mock_session.get.return_value.__aenter__.return_value = mock_response
            
            # Mock file operations
            with patch('aiofiles.open', create=True) as mock_open:
                mock_file = AsyncMock()
                mock_open.return_value.__aenter__.return_value = mock_file
                
                # Mock checksum calculation
                with patch.object(downloader, '_calculate_file_checksum', return_value="abc123"):
                    result = await downloader.download_with_retry(model_id, download_url)
        
        assert result.success is True
        assert result.model_id == model_id
        assert result.final_status == DownloadStatus.COMPLETED
        assert result.total_retries == 0
    
    @pytest.mark.asyncio
    async def test_download_with_retry_failure(self, downloader):
        """Test download failure with retry logic"""
        model_id = "test_model"
        download_url = "https://example.com/model.bin"
        
        # Mock failing response
        with patch.object(downloader, '_session') as mock_session:
            mock_session.get.side_effect = aiohttp.ClientError("Network error")
            
            result = await downloader.download_with_retry(model_id, download_url, max_retries=2)
        
        assert result.success is False
        assert result.model_id == model_id
        assert result.final_status == DownloadStatus.FAILED
        assert result.total_retries == 2
        assert "Network error" in result.error_message
    
    @pytest.mark.asyncio
    async def test_download_with_resume(self, downloader, temp_dir):
        """Test download resume functionality"""
        model_id = "test_model"
        download_url = "https://example.com/model.bin"
        partial_path = downloader._partial_downloads_dir / f"{model_id}.partial"
        
        # Create existing partial file
        existing_data = b"existing_data"
        async with aiofiles.open(partial_path, 'wb') as f:
            await f.write(existing_data)
        
        # Mock response for resume
        mock_response = AsyncMock(spec=aiohttp.ClientResponse)
        mock_response.status = 206  # Partial Content
        mock_response.headers = {'Content-Length': '1000'}
        mock_response.content.iter_chunked.return_value = [b"new_data"].__iter__()
        
        progress = DownloadProgress(
            model_id=model_id,
            status=DownloadStatus.DOWNLOADING,
            progress_percent=0.0,
            downloaded_mb=0.0,
            total_mb=0.0,
            speed_mbps=0.0
        )
        
        with patch.object(downloader, '_session') as mock_session:
            mock_session.get.return_value.__aenter__.return_value = mock_response
            
            with patch('aiofiles.open', create=True) as mock_open:
                mock_file = AsyncMock()
                mock_open.return_value.__aenter__.return_value = mock_file
                
                with patch.object(downloader, '_calculate_file_checksum', return_value="abc123"):
                    result = await downloader._download_with_resume(model_id, download_url, progress)
        
        # Verify resume was attempted (Range header should be set)
        mock_session.get.assert_called_with(download_url, headers={'Range': f'bytes={len(existing_data)}-'})
    
    @pytest.mark.asyncio
    async def test_download_cancellation(self, downloader):
        """Test download cancellation"""
        model_id = "test_model"
        
        # Create progress that will be cancelled
        progress = DownloadProgress(
            model_id=model_id,
            status=DownloadStatus.DOWNLOADING,
            progress_percent=25.0,
            downloaded_mb=2.5,
            total_mb=10.0,
            speed_mbps=5.0,
            can_cancel=True
        )
        
        async with downloader._download_lock:
            downloader._active_downloads[model_id] = progress
        
        # Mock download task
        mock_task = AsyncMock()
        downloader._download_tasks[model_id] = mock_task
        
        # Cancel download
        result = await downloader.cancel_download(model_id)
        
        assert result is True
        assert progress.status == DownloadStatus.CANCELLED
        mock_task.cancel.assert_called_once()
        assert model_id not in downloader._download_tasks
    
    @pytest.mark.asyncio
    async def test_error_handling(self, downloader):
        """Test various error handling scenarios"""
        model_id = "test_model"
        download_url = "https://example.com/model.bin"
        
        # Test HTTP error handling
        with patch.object(downloader, '_session') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 404
            mock_response.reason = "Not Found"
            mock_session.get.return_value.__aenter__.return_value = mock_response
            
            progress = DownloadProgress(
                model_id=model_id,
                status=DownloadStatus.DOWNLOADING,
                progress_percent=0.0,
                downloaded_mb=0.0,
                total_mb=0.0,
                speed_mbps=0.0
            )
            
            result = await downloader._download_with_resume(model_id, download_url, progress)
            
            assert result.success is False
            assert "HTTP 404" in result.error_message
    
    @pytest.mark.asyncio
    async def test_concurrent_downloads(self, downloader):
        """Test handling multiple concurrent downloads"""
        model_ids = ["model_1", "model_2", "model_3"]
        
        # Create progress for multiple downloads
        for model_id in model_ids:
            progress = DownloadProgress(
                model_id=model_id,
                status=DownloadStatus.DOWNLOADING,
                progress_percent=50.0,
                downloaded_mb=5.0,
                total_mb=10.0,
                speed_mbps=10.0
            )
            
            async with downloader._download_lock:
                downloader._active_downloads[model_id] = progress
        
        # Get all progress
        all_progress = await downloader.get_all_download_progress()
        assert len(all_progress) == 3
        
        for model_id in model_ids:
            assert model_id in all_progress
            assert all_progress[model_id].status == DownloadStatus.DOWNLOADING


class TestRetryConfig:
    """Test retry configuration"""
    
    def test_retry_config_defaults(self):
        """Test default retry configuration values"""
        config = RetryConfig()
        
        assert config.max_retries == 3
        assert config.initial_delay == 1.0
        assert config.max_delay == 60.0
        assert config.backoff_factor == 2.0
        assert config.jitter is True
        assert 429 in config.retry_on_status_codes
        assert 500 in config.retry_on_status_codes
    
    def test_retry_config_custom(self):
        """Test custom retry configuration"""
        config = RetryConfig(
            max_retries=5,
            initial_delay=2.0,
            max_delay=120.0,
            backoff_factor=1.5,
            jitter=False,
            retry_on_status_codes=[503, 504]
        )
        
        assert config.max_retries == 5
        assert config.initial_delay == 2.0
        assert config.max_delay == 120.0
        assert config.backoff_factor == 1.5
        assert config.jitter is False
        assert config.retry_on_status_codes == [503, 504]


class TestBandwidthConfig:
    """Test bandwidth configuration"""
    
    def test_bandwidth_config_defaults(self):
        """Test default bandwidth configuration values"""
        config = BandwidthConfig()
        
        assert config.max_speed_mbps is None
        assert config.chunk_size == 8192
        assert config.concurrent_downloads == 2
        assert config.adaptive_chunking is True
    
    def test_bandwidth_config_custom(self):
        """Test custom bandwidth configuration"""
        config = BandwidthConfig(
            max_speed_mbps=50.0,
            chunk_size=16384,
            concurrent_downloads=4,
            adaptive_chunking=False
        )
        
        assert config.max_speed_mbps == 50.0
        assert config.chunk_size == 16384
        assert config.concurrent_downloads == 4
        assert config.adaptive_chunking is False


class TestDownloadProgress:
    """Test download progress tracking"""
    
    def test_download_progress_creation(self):
        """Test download progress object creation"""
        progress = DownloadProgress(
            model_id="test_model",
            status=DownloadStatus.DOWNLOADING,
            progress_percent=75.0,
            downloaded_mb=7.5,
            total_mb=10.0,
            speed_mbps=25.0,
            eta_seconds=120.0,
            retry_count=1,
            max_retries=3
        )
        
        assert progress.model_id == "test_model"
        assert progress.status == DownloadStatus.DOWNLOADING
        assert progress.progress_percent == 75.0
        assert progress.downloaded_mb == 7.5
        assert progress.total_mb == 10.0
        assert progress.speed_mbps == 25.0
        assert progress.eta_seconds == 120.0
        assert progress.retry_count == 1
        assert progress.max_retries == 3
        assert progress.can_pause is True
        assert progress.can_resume is True
        assert progress.can_cancel is True


class TestDownloadResult:
    """Test download result object"""
    
    def test_download_result_success(self):
        """Test successful download result"""
        result = DownloadResult(
            success=True,
            model_id="test_model",
            final_status=DownloadStatus.COMPLETED,
            total_time_seconds=120.5,
            total_retries=1,
            final_size_mb=10.0,
            integrity_verified=True,
            download_path="/path/to/model"
        )
        
        assert result.success is True
        assert result.model_id == "test_model"
        assert result.final_status == DownloadStatus.COMPLETED
        assert result.total_time_seconds == 120.5
        assert result.total_retries == 1
        assert result.final_size_mb == 10.0
        assert result.integrity_verified is True
        assert result.download_path == "/path/to/model"
        assert result.error_message is None
    
    def test_download_result_failure(self):
        """Test failed download result"""
        result = DownloadResult(
            success=False,
            model_id="test_model",
            final_status=DownloadStatus.FAILED,
            total_time_seconds=60.0,
            total_retries=3,
            final_size_mb=2.5,
            error_message="Network timeout",
            integrity_verified=False
        )
        
        assert result.success is False
        assert result.final_status == DownloadStatus.FAILED
        assert result.error_message == "Network timeout"
        assert result.integrity_verified is False


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])