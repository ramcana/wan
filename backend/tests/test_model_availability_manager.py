"""
Integration Tests for Model Availability Manager
Tests the central coordination system with existing ModelManager and ModelDownloader
"""

import asyncio
import json
import pytest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock

# Import the components to test
try:
    from backend.core.model_availability_manager import (
        ModelAvailabilityManager,
        ModelAvailabilityStatus,
        ModelPriority,
        DetailedModelStatus,
        ModelRequestResult,
        CleanupResult,
        RetentionPolicy
    )
    from backend.core.enhanced_model_downloader import (
        EnhancedModelDownloader,
        DownloadStatus,
        DownloadProgress,
        DownloadResult
    )
    from backend.core.model_health_monitor import (
        ModelHealthMonitor,
        HealthStatus,
        IntegrityResult
    )
except ImportError as e:
    pytest.skip(f"Required modules not available: {e}", allow_module_level=True)


class TestModelAvailabilityManager:
    """Test suite for ModelAvailabilityManager"""
    
    @pytest.fixture
    async def temp_models_dir(self):
        """Create temporary models directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def mock_model_manager(self):
        """Create mock ModelManager"""
        mock_manager = Mock()
        mock_manager.get_model_id.return_value = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
        mock_manager.get_model_status.return_value = {
            "model_id": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            "is_cached": True,
            "is_loaded": False,
            "is_valid": True,
            "size_mb": 8500.0
        }
        return mock_manager
    
    @pytest.fixture
    async def mock_downloader(self):
        """Create mock EnhancedModelDownloader"""
        mock_downloader = AsyncMock(spec=EnhancedModelDownloader)
        mock_downloader.__aenter__ = AsyncMock(return_value=mock_downloader)
        mock_downloader.__aexit__ = AsyncMock(return_value=None)
        mock_downloader.add_progress_callback = Mock()
        mock_downloader.get_download_progress = AsyncMock(return_value=None)
        mock_downloader.download_with_retry = AsyncMock(return_value=DownloadResult(
            success=True,
            model_id="test-model",
            final_status=DownloadStatus.COMPLETED,
            total_time_seconds=120.0,
            total_retries=0,
            final_size_mb=8500.0,
            integrity_verified=True
        ))
        return mock_downloader
    
    @pytest.fixture
    def mock_health_monitor(self):
        """Create mock ModelHealthMonitor"""
        mock_monitor = Mock(spec=ModelHealthMonitor)
        mock_monitor.health_data_dir = Path("/tmp/health")
        mock_monitor.add_health_callback = Mock()
        mock_monitor.check_model_integrity = AsyncMock(return_value=IntegrityResult(
            model_id="test-model",
            is_healthy=True,
            health_status=HealthStatus.HEALTHY,
            last_checked=datetime.now()
        ))
        return mock_monitor
    
    @pytest.fixture
    async def availability_manager(self, temp_models_dir, mock_model_manager, mock_downloader, mock_health_monitor):
        """Create ModelAvailabilityManager instance for testing"""
        manager = ModelAvailabilityManager(
            model_manager=mock_model_manager,
            downloader=mock_downloader,
            health_monitor=mock_health_monitor,
            models_dir=temp_models_dir
        )
        
        # Initialize the manager
        await manager.initialize()
        
        yield manager
        
        # Cleanup
        await manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_initialization(self, temp_models_dir, mock_model_manager):
        """Test manager initialization"""
        manager = ModelAvailabilityManager(
            model_manager=mock_model_manager,
            models_dir=temp_models_dir
        )
        
        # Test initialization
        success = await manager.initialize()
        assert success is True
        
        # Check that directories are created
        assert Path(temp_models_dir).exists()
        assert (Path(temp_models_dir) / ".analytics").exists()
        
        # Cleanup
        await manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_ensure_all_models_available(self, availability_manager, mock_model_manager):
        """Test ensuring all models are available"""
        # Mock model status responses
        mock_model_manager.get_model_status.side_effect = [
            {
                "model_id": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
                "is_cached": True,
                "is_loaded": False,
                "is_valid": True,
                "size_mb": 8500.0
            },
            {
                "model_id": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
                "is_cached": False,
                "is_loaded": False,
                "is_valid": False,
                "size_mb": 0.0
            },
            {
                "model_id": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
                "is_cached": True,
                "is_loaded": True,
                "is_valid": True,
                "size_mb": 6000.0
            }
        ]
        
        # Test ensuring all models available
        result = await availability_manager.ensure_all_models_available()
        
        # Verify results
        assert isinstance(result, dict)
        assert "t2v-A14B" in result
        assert "i2v-A14B" in result
        assert "ti2v-5B" in result
        
        # Check that missing model is queued for download
        assert result["i2v-A14B"] == ModelAvailabilityStatus.MISSING
        assert result["t2v-A14B"] == ModelAvailabilityStatus.AVAILABLE
        assert result["ti2v-5B"] == ModelAvailabilityStatus.AVAILABLE
    
    @pytest.mark.asyncio
    async def test_get_comprehensive_model_status(self, availability_manager, mock_model_manager):
        """Test getting comprehensive model status"""
        # Setup mock responses
        mock_model_manager.get_model_status.return_value = {
            "model_id": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            "is_cached": True,
            "is_loaded": False,
            "is_valid": True,
            "size_mb": 8500.0
        }
        
        # Test getting comprehensive status
        result = await availability_manager.get_comprehensive_model_status()
        
        # Verify results
        assert isinstance(result, dict)
        assert len(result) > 0
        
        # Check status structure
        for model_type, status in result.items():
            assert isinstance(status, DetailedModelStatus)
            assert hasattr(status, 'model_id')
            assert hasattr(status, 'availability_status')
            assert hasattr(status, 'is_available')
            assert hasattr(status, 'size_mb')
    
    @pytest.mark.asyncio
    async def test_prioritize_model_downloads(self, availability_manager, mock_model_manager):
        """Test model download prioritization"""
        # Setup mock responses for different model states
        mock_model_manager.get_model_status.side_effect = [
            {  # t2v-A14B - available
                "model_id": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
                "is_cached": True,
                "is_loaded": False,
                "is_valid": True,
                "size_mb": 8500.0
            },
            {  # i2v-A14B - missing
                "model_id": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
                "is_cached": False,
                "is_loaded": False,
                "is_valid": False,
                "size_mb": 0.0
            },
            {  # ti2v-5B - corrupted
                "model_id": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
                "is_cached": True,
                "is_loaded": False,
                "is_valid": False,
                "size_mb": 6000.0
            }
        ]
        
        # Test prioritization
        priority_list = await availability_manager.prioritize_model_downloads()
        
        # Verify results
        assert isinstance(priority_list, list)
        # Corrupted models should come first, then missing models
        if "ti2v-5B" in priority_list and "i2v-A14B" in priority_list:
            assert priority_list.index("ti2v-5B") < priority_list.index("i2v-A14B")
    
    @pytest.mark.asyncio
    async def test_handle_model_request_available(self, availability_manager, mock_model_manager):
        """Test handling request for available model"""
        # Setup mock for available model
        mock_model_manager.get_model_status.return_value = {
            "model_id": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            "is_cached": True,
            "is_loaded": False,
            "is_valid": True,
            "size_mb": 8500.0
        }
        
        # Test request handling
        result = await availability_manager.handle_model_request("t2v-A14B")
        
        # Verify results
        assert isinstance(result, ModelRequestResult)
        assert result.success is True
        assert result.availability_status == ModelAvailabilityStatus.AVAILABLE
        assert "available and ready" in result.message.lower()
    
    @pytest.mark.asyncio
    async def test_handle_model_request_missing(self, availability_manager, mock_model_manager):
        """Test handling request for missing model"""
        # Setup mock for missing model
        mock_model_manager.get_model_status.return_value = {
            "model_id": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
            "is_cached": False,
            "is_loaded": False,
            "is_valid": False,
            "size_mb": 0.0
        }
        
        # Test request handling
        result = await availability_manager.handle_model_request("i2v-A14B")
        
        # Verify results
        assert isinstance(result, ModelRequestResult)
        assert result.success is False
        assert result.availability_status == ModelAvailabilityStatus.MISSING
        assert result.action_required == "download_required"
    
    @pytest.mark.asyncio
    async def test_handle_model_request_downloading(self, availability_manager, mock_model_manager, mock_downloader):
        """Test handling request for model currently downloading"""
        # Setup mock for missing model
        mock_model_manager.get_model_status.return_value = {
            "model_id": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
            "is_cached": False,
            "is_loaded": False,
            "is_valid": False,
            "size_mb": 0.0
        }
        
        # Setup mock download progress
        download_progress = DownloadProgress(
            model_id="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
            status=DownloadStatus.DOWNLOADING,
            progress_percent=45.0,
            downloaded_mb=3825.0,
            total_mb=8500.0,
            speed_mbps=50.0,
            eta_seconds=120.0
        )
        mock_downloader.get_download_progress.return_value = download_progress
        
        # Test request handling
        result = await availability_manager.handle_model_request("i2v-A14B")
        
        # Verify results
        assert isinstance(result, ModelRequestResult)
        assert result.success is False
        assert result.availability_status == ModelAvailabilityStatus.DOWNLOADING
        assert result.action_required == "wait_for_download"
        assert result.estimated_wait_time is not None
    
    @pytest.mark.asyncio
    async def test_cleanup_unused_models(self, availability_manager, mock_model_manager):
        """Test cleanup of unused models"""
        # Setup usage data with old usage
        old_date = datetime.now() - timedelta(days=45)
        availability_manager._usage_data = {
            "t2v-A14B": {
                "total_uses": 5,
                "last_used": old_date.isoformat(),
                "usage_frequency": 0.05,  # Very low usage
                "usage_history": []
            }
        }
        
        # Setup mock for available model
        mock_model_manager.get_model_status.return_value = {
            "model_id": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            "is_cached": True,
            "is_loaded": False,
            "is_valid": True,
            "size_mb": 8500.0
        }
        
        # Test cleanup
        policy = RetentionPolicy(max_unused_days=30, min_usage_frequency=0.1)
        result = await availability_manager.cleanup_unused_models(policy)
        
        # Verify results
        assert isinstance(result, CleanupResult)
        assert result.success is True
        assert len(result.recommendations) > 0
        
        # Check that recommendation includes the unused model
        recommendation_models = [rec.model_id for rec in result.recommendations]
        assert any("t2v" in model_id.lower() for model_id in recommendation_models)
    
    @pytest.mark.asyncio
    async def test_usage_tracking(self, availability_manager):
        """Test model usage tracking"""
        model_type = "t2v-A14B"
        
        # Track usage multiple times
        await availability_manager._track_model_usage(model_type)
        await availability_manager._track_model_usage(model_type)
        await availability_manager._track_model_usage(model_type)
        
        # Verify usage data
        assert model_type in availability_manager._usage_data
        usage_data = availability_manager._usage_data[model_type]
        
        assert usage_data["total_uses"] == 3
        assert usage_data["last_used"] is not None
        assert len(usage_data["usage_history"]) == 3
        assert usage_data["usage_frequency"] > 0
    
    @pytest.mark.asyncio
    async def test_download_progress_callback(self, availability_manager):
        """Test download progress callback handling"""
        # Create download progress
        progress = DownloadProgress(
            model_id="t2v-A14B",
            status=DownloadStatus.DOWNLOADING,
            progress_percent=75.0,
            downloaded_mb=6375.0,
            total_mb=8500.0,
            speed_mbps=25.0
        )
        
        # Initialize status cache
        availability_manager._model_status_cache["t2v-A14B"] = DetailedModelStatus(
            model_id="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            model_type="t2v-A14B",
            is_available=False,
            is_loaded=False,
            size_mb=0.0,
            availability_status=ModelAvailabilityStatus.MISSING
        )
        
        # Test callback
        await availability_manager._on_download_progress(progress)
        
        # Verify status update
        status = availability_manager._model_status_cache["t2v-A14B"]
        assert status.download_progress == 75.0
        assert status.availability_status == ModelAvailabilityStatus.DOWNLOADING
    
    @pytest.mark.asyncio
    async def test_health_check_callback(self, availability_manager):
        """Test health check callback handling"""
        # Create integrity result
        integrity_result = IntegrityResult(
            model_id="t2v-A14B",
            is_healthy=False,
            health_status=HealthStatus.CORRUPTED,
            last_checked=datetime.now()
        )
        
        # Initialize status cache
        availability_manager._model_status_cache["t2v-A14B"] = DetailedModelStatus(
            model_id="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            model_type="t2v-A14B",
            is_available=True,
            is_loaded=False,
            size_mb=8500.0,
            availability_status=ModelAvailabilityStatus.AVAILABLE
        )
        
        # Test callback
        await availability_manager._on_health_update(integrity_result)
        
        # Verify status update
        status = availability_manager._model_status_cache["t2v-A14B"]
        assert status.corruption_detected is True
        assert status.availability_status == ModelAvailabilityStatus.CORRUPTED
        assert status.integrity_score == 0.5
    
    @pytest.mark.asyncio
    async def test_system_health_report(self, availability_manager, mock_model_manager):
        """Test system health report generation"""
        # Setup mock responses for different model states
        mock_model_manager.get_model_status.side_effect = [
            {  # Healthy model
                "model_id": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
                "is_cached": True,
                "is_loaded": False,
                "is_valid": True,
                "size_mb": 8500.0
            },
            {  # Missing model
                "model_id": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
                "is_cached": False,
                "is_loaded": False,
                "is_valid": False,
                "size_mb": 0.0
            },
            {  # Available model
                "model_id": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
                "is_cached": True,
                "is_loaded": False,
                "is_valid": True,
                "size_mb": 6000.0
            }
        ]
        
        # Test health report generation
        report = await availability_manager.get_system_health_report()
        
        # Verify report structure
        assert hasattr(report, 'overall_health_score')
        assert hasattr(report, 'models_healthy')
        assert hasattr(report, 'models_missing')
        assert hasattr(report, 'recommendations')
        assert hasattr(report, 'last_updated')
        
        # Check that missing models are detected
        assert report.models_missing >= 1
        assert any("missing" in rec.lower() for rec in report.recommendations)
    
    @pytest.mark.asyncio
    async def test_analytics_persistence(self, availability_manager):
        """Test that analytics data persists across sessions"""
        model_type = "t2v-A14B"
        
        # Track some usage
        await availability_manager._track_model_usage(model_type)
        await availability_manager._track_model_usage(model_type)
        
        # Save analytics
        await availability_manager._save_usage_analytics()
        
        # Create new manager instance
        new_manager = ModelAvailabilityManager(models_dir=str(availability_manager.models_dir))
        await new_manager.initialize()
        
        # Verify data was loaded
        assert model_type in new_manager._usage_data
        assert new_manager._usage_data[model_type]["total_uses"] == 2
        
        # Cleanup
        await new_manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_download_queue_management(self, availability_manager):
        """Test download queue management"""
        # Queue models with different priorities
        await availability_manager._queue_model_download("t2v-A14B", ModelPriority.LOW)
        await availability_manager._queue_model_download("i2v-A14B", ModelPriority.HIGH)
        await availability_manager._queue_model_download("ti2v-5B", ModelPriority.CRITICAL)
        
        # Verify queue ordering (highest priority first)
        queue = availability_manager._download_queue
        assert len(queue) == 3
        
        # Check that critical priority comes first
        priorities = [item[1] for item in queue]
        assert ModelPriority.CRITICAL in priorities
        
        # Test updating priority
        await availability_manager._queue_model_download("t2v-A14B", ModelPriority.CRITICAL)
        
        # Verify no duplicates and updated priority
        model_types = [item[0] for item in availability_manager._download_queue]
        assert model_types.count("t2v-A14B") == 1
    
    @pytest.mark.asyncio
    async def test_error_handling(self, availability_manager, mock_model_manager):
        """Test error handling in various scenarios"""
        # Test with model manager that raises exceptions
        mock_model_manager.get_model_status.side_effect = Exception("Model manager error")
        
        # Test that errors are handled gracefully
        result = await availability_manager.handle_model_request("t2v-A14B")
        
        assert isinstance(result, ModelRequestResult)
        assert result.success is False
        assert result.availability_status == ModelAvailabilityStatus.UNKNOWN
        assert result.error_details is not None
    
    @pytest.mark.asyncio
    async def test_callback_registration(self, availability_manager):
        """Test callback registration and notification"""
        # Create mock callbacks
        status_callback = AsyncMock()
        download_callback = Mock()
        
        # Register callbacks
        availability_manager.add_status_callback(status_callback)
        availability_manager.add_download_callback(download_callback)
        
        # Verify callbacks are registered
        assert status_callback in availability_manager._status_callbacks
        assert download_callback in availability_manager._download_callbacks
        
        # Test callback notification
        integrity_result = IntegrityResult(
            model_id="test-model",
            is_healthy=True,
            health_status=HealthStatus.HEALTHY,
            last_checked=datetime.now()
        )
        
        await availability_manager._notify_health_callbacks(integrity_result)
        
        # Verify callback was called
        status_callback.assert_called_once_with(integrity_result)


class TestModelAvailabilityManagerIntegration:
    """Integration tests with real components"""
    
    @pytest.mark.asyncio
    async def test_integration_with_model_manager(self):
        """Test integration with actual ModelManager"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Import actual ModelManager
            from backend.core.services.model_manager import ModelManager
            
            # Create real model manager
            model_manager = ModelManager()
            
            # Create availability manager
            availability_manager = ModelAvailabilityManager(
                model_manager=model_manager,
                models_dir=temp_dir
            )
            
            try:
                # Initialize
                await availability_manager.initialize()
                
                # Test getting model status
                status = await availability_manager.get_comprehensive_model_status()
                
                # Verify we get status for supported models
                assert isinstance(status, dict)
                
                # Test model request
                result = await availability_manager.handle_model_request("t2v-A14B")
                assert isinstance(result, ModelRequestResult)
                
            finally:
                await availability_manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_integration_with_enhanced_downloader(self):
        """Test integration with EnhancedModelDownloader"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create real enhanced downloader
            downloader = EnhancedModelDownloader(models_dir=temp_dir)
            
            # Create availability manager
            availability_manager = ModelAvailabilityManager(
                downloader=downloader,
                models_dir=temp_dir
            )
            
            try:
                # Initialize
                await availability_manager.initialize()
                
                # Test that downloader callbacks are set up
                assert availability_manager._on_download_progress in downloader._progress_callbacks
                
                # Test download progress handling
                progress = DownloadProgress(
                    model_id="test-model",
                    status=DownloadStatus.DOWNLOADING,
                    progress_percent=50.0,
                    downloaded_mb=4250.0,
                    total_mb=8500.0,
                    speed_mbps=25.0
                )
                
                # Simulate progress callback
                await availability_manager._on_download_progress(progress)
                
                # Verify no errors occurred
                assert True  # If we get here, no exceptions were raised
                
            finally:
                await availability_manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_integration_with_health_monitor(self):
        """Test integration with ModelHealthMonitor"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create real health monitor
            health_monitor = ModelHealthMonitor(models_dir=temp_dir)
            
            # Create availability manager
            availability_manager = ModelAvailabilityManager(
                health_monitor=health_monitor,
                models_dir=temp_dir
            )
            
            try:
                # Initialize
                await availability_manager.initialize()
                
                # Test that health monitor callbacks are set up
                assert availability_manager._on_health_update in health_monitor._health_callbacks
                
                # Test health update handling
                integrity_result = IntegrityResult(
                    model_id="test-model",
                    is_healthy=True,
                    health_status=HealthStatus.HEALTHY,
                    last_checked=datetime.now()
                )
                
                # Simulate health callback
                await availability_manager._on_health_update(integrity_result)
                
                # Verify no errors occurred
                assert True  # If we get here, no exceptions were raised
                
            finally:
                await availability_manager.cleanup()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])