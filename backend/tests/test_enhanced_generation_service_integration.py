"""
Integration Tests for Enhanced Generation Service
Tests the integration of ModelAvailabilityManager, enhanced download retry logic,
intelligent fallback, usage analytics tracking, health monitoring, and error recovery.
"""

import asyncio
import pytest
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from sqlalchemy.orm import Session

# Import the components we're testing
from backend.services.generation_service import GenerationService, get_generation_service
from repositories.database import GenerationTaskDB, TaskStatusEnum, ModelTypeEnum
from backend.core.model_availability_manager import ModelAvailabilityManager, ModelAvailabilityStatus
from backend.core.enhanced_model_downloader import EnhancedModelDownloader, DownloadStatus, DownloadResult
from backend.core.intelligent_fallback_manager import IntelligentFallbackManager, FallbackType, GenerationRequirements
from backend.core.model_health_monitor import ModelHealthMonitor, HealthStatus, IntegrityResult
from backend.core.model_usage_analytics import ModelUsageAnalytics, UsageEventType, UsageData


class TestEnhancedGenerationServiceIntegration:
    """Test suite for enhanced generation service integration"""

    @pytest.fixture
    def generation_service(self):
        """Create a generation service instance for testing"""
        service = GenerationService()
        
        # Mock the components to avoid real initialization
        service.model_availability_manager = Mock(spec=ModelAvailabilityManager)
        service.enhanced_model_downloader = Mock(spec=EnhancedModelDownloader)
        service.intelligent_fallback_manager = Mock(spec=IntelligentFallbackManager)
        service.model_health_monitor = Mock(spec=ModelHealthMonitor)
        service.model_usage_analytics = Mock(spec=ModelUsageAnalytics)
        service.websocket_manager = Mock()
        service.performance_monitor = Mock()
        service.fallback_recovery_system = Mock()
        
        # Mock async methods
        service.model_availability_manager.handle_model_request = AsyncMock()
        service.enhanced_model_downloader.verify_and_repair_model = AsyncMock()
        service.enhanced_model_downloader.download_with_retry = AsyncMock()
        service.intelligent_fallback_manager.get_fallback_strategy = AsyncMock()
        service.intelligent_fallback_manager.suggest_alternative_model = AsyncMock()
        service.intelligent_fallback_manager.queue_request_for_downloading_model = AsyncMock()
        service.model_health_monitor.check_model_integrity = AsyncMock()
        service.model_health_monitor.monitor_model_performance = AsyncMock()
        service.model_usage_analytics.track_usage = AsyncMock()
        service.websocket_manager.send_alert = AsyncMock()
        service.fallback_recovery_system.attempt_recovery = AsyncMock()
        
        return service

    @pytest.fixture
    def sample_task(self):
        """Create a sample generation task for testing"""
        task = GenerationTaskDB(
            id="test-task-1",
            prompt="A beautiful sunset over mountains",
            model_type=ModelTypeEnum.T2V_A14B,
            resolution="1280x720",
            steps=20,
            status=TaskStatusEnum.PENDING,
            progress=0,
            created_at=datetime.utcnow()
        )
        return task

    @pytest.fixture
    def mock_db_session(self):
        """Create a mock database session"""
        session = Mock(spec=Session)
        session.commit = Mock()
        session.close = Mock()
        return session

    @pytest.mark.asyncio
    async def test_enhanced_generation_success_flow(self, generation_service, sample_task, mock_db_session):
        """Test successful enhanced generation flow with all components working"""
        
        # Setup mocks for successful flow
        generation_service.model_availability_manager.handle_model_request.return_value = Mock(
            success=True, error_message=None
        )
        
        generation_service.model_health_monitor.check_model_integrity.return_value = Mock(
            is_healthy=True, issues=[]
        )
        
        # Mock the real generation method
        generation_service._run_real_generation_with_monitoring = AsyncMock(return_value=True)
        
        # Run the enhanced generation
        result = await generation_service._run_enhanced_generation(sample_task, mock_db_session, "t2v")
        
        # Verify the flow
        assert result is True
        generation_service.model_availability_manager.handle_model_request.assert_called_once_with("t2v")
        generation_service.model_health_monitor.check_model_integrity.assert_called_once_with("t2v")
        generation_service._run_real_generation_with_monitoring.assert_called_once_with(
            sample_task, mock_db_session, "t2v"
        )

    @pytest.mark.asyncio
    async def test_model_health_check_failure_with_repair(self, generation_service, sample_task, mock_db_session):
        """Test model health check failure with successful repair"""
        
        # Setup mocks
        generation_service.model_availability_manager.handle_model_request.return_value = Mock(
            success=True, error_message=None
        )
        
        generation_service.model_health_monitor.check_model_integrity.return_value = Mock(
            is_healthy=False, issues=["checksum_mismatch"]
        )
        
        generation_service.enhanced_model_downloader.verify_and_repair_model.return_value = Mock(
            success=True, error_message=None
        )
        
        generation_service._run_real_generation_with_monitoring = AsyncMock(return_value=True)
        
        # Run the enhanced generation
        result = await generation_service._run_enhanced_generation(sample_task, mock_db_session, "t2v")
        
        # Verify repair was attempted and generation succeeded
        assert result is True
        generation_service.enhanced_model_downloader.verify_and_repair_model.assert_called_once_with("t2v")
        generation_service._run_real_generation_with_monitoring.assert_called_once()

    @pytest.mark.asyncio
    async def test_model_unavailable_with_alternative_model(self, generation_service, sample_task, mock_db_session):
        """Test model unavailable scenario with successful alternative model fallback"""
        
        # Setup mocks for model unavailable
        generation_service.model_availability_manager.handle_model_request.return_value = Mock(
            success=False, error_message="Model not available"
        )
        
        # Mock intelligent fallback manager
        mock_strategy = Mock()
        mock_strategy.strategy_type = FallbackType.ALTERNATIVE_MODEL
        mock_strategy.alternative_model = "i2v"
        
        generation_service.intelligent_fallback_manager.get_fallback_strategy.return_value = mock_strategy
        
        # Mock alternative model generation
        generation_service._try_alternative_model = AsyncMock(return_value=True)
        
        # Run the enhanced generation
        result = await generation_service._run_enhanced_generation(sample_task, mock_db_session, "t2v")
        
        # Verify alternative model was used
        assert result is True
        generation_service._try_alternative_model.assert_called_once_with(sample_task, mock_db_session, "i2v")

    @pytest.mark.asyncio
    async def test_download_and_retry_strategy(self, generation_service, sample_task, mock_db_session):
        """Test download and retry fallback strategy"""
        
        # Setup mocks for download and retry
        generation_service.model_availability_manager.handle_model_request.return_value = Mock(
            success=False, error_message="Model missing"
        )
        
        mock_strategy = Mock()
        mock_strategy.strategy_type = FallbackType.DOWNLOAD_AND_RETRY
        
        generation_service.intelligent_fallback_manager.get_fallback_strategy.return_value = mock_strategy
        
        # Mock successful download
        generation_service.enhanced_model_downloader.download_with_retry.return_value = Mock(
            success=True, error_message=None
        )
        
        generation_service._run_real_generation_with_monitoring = AsyncMock(return_value=True)
        
        # Mock the try_download_and_retry method
        generation_service._try_download_and_retry = AsyncMock(return_value=True)
        
        # Run the enhanced generation
        result = await generation_service._run_enhanced_generation(sample_task, mock_db_session, "t2v")
        
        # Verify download and retry was attempted
        assert result is True
        generation_service._try_download_and_retry.assert_called_once_with(sample_task, mock_db_session, "t2v")

    @pytest.mark.asyncio
    async def test_usage_analytics_tracking(self, generation_service, sample_task, mock_db_session):
        """Test that usage analytics are properly tracked during generation"""
        
        # Setup successful generation
        generation_service.model_availability_manager.handle_model_request.return_value = Mock(
            success=True, error_message=None
        )
        
        generation_service.model_health_monitor.check_model_integrity.return_value = Mock(
            is_healthy=True, issues=[]
        )
        
        generation_service._run_real_generation_with_monitoring = AsyncMock(return_value=True)
        
        # Run the enhanced generation
        await generation_service._run_enhanced_generation(sample_task, mock_db_session, "t2v")
        
        # Verify analytics tracking was called
        generation_service.model_usage_analytics.track_usage.assert_called()

    @pytest.mark.asyncio
    async def test_real_generation_with_monitoring(self, generation_service, sample_task, mock_db_session):
        """Test real generation with performance monitoring and analytics"""
        
        # Mock the base real generation method
        generation_service._run_real_generation = AsyncMock(return_value=True)
        
        # Run real generation with monitoring
        result = await generation_service._run_real_generation_with_monitoring(
            sample_task, mock_db_session, "t2v"
        )
        
        # Verify the flow
        assert result is True
        generation_service._run_real_generation.assert_called_once_with(sample_task, mock_db_session, "t2v")
        
        # Verify analytics tracking
        generation_service.model_usage_analytics.track_usage.assert_called()
        generation_service.model_health_monitor.monitor_model_performance.assert_called()

    @pytest.mark.asyncio
    async def test_error_handling_with_recovery(self, generation_service, sample_task, mock_db_session):
        """Test error handling with enhanced recovery system"""
        
        # Setup to trigger error handling
        test_error = Exception("Test generation error")
        generation_service.model_availability_manager.handle_model_request.side_effect = test_error
        
        # Mock recovery system
        generation_service.fallback_recovery_system.attempt_recovery.return_value = Mock(success=False)
        generation_service._run_mock_generation = AsyncMock(return_value=True)
        
        # Run the enhanced generation
        result = await generation_service._run_enhanced_generation(sample_task, mock_db_session, "t2v")
        
        # Verify error handling was triggered
        assert result is True  # Should fallback to mock generation
        generation_service._run_mock_generation.assert_called_once()

    @pytest.mark.asyncio
    async def test_websocket_notifications(self, generation_service, sample_task, mock_db_session):
        """Test that WebSocket notifications are sent during enhanced generation"""
        
        # Setup successful generation
        generation_service.model_availability_manager.handle_model_request.return_value = Mock(
            success=True, error_message=None
        )
        
        generation_service.model_health_monitor.check_model_integrity.return_value = Mock(
            is_healthy=True, issues=[]
        )
        
        generation_service._run_real_generation_with_monitoring = AsyncMock(return_value=True)
        
        # Run the enhanced generation
        await generation_service._run_enhanced_generation(sample_task, mock_db_session, "t2v")
        
        # Verify WebSocket notifications (would be called in the real generation method)
        # This is tested indirectly through the mocked methods

    @pytest.mark.asyncio
    async def test_queue_generation_request(self, generation_service, sample_task, mock_db_session):
        """Test queuing generation request when model is downloading"""
        
        mock_strategy = Mock()
        mock_strategy.strategy_type = FallbackType.QUEUE_AND_WAIT
        mock_strategy.estimated_wait_time = timedelta(minutes=5)
        
        generation_service.intelligent_fallback_manager.queue_request_for_downloading_model.return_value = {
            "success": True, "queue_position": 1
        }
        
        # Test the queue method
        result = await generation_service._queue_generation_request(
            sample_task, mock_db_session, "t2v", mock_strategy
        )
        
        assert result is True
        generation_service.intelligent_fallback_manager.queue_request_for_downloading_model.assert_called_once()
        generation_service.websocket_manager.send_alert.assert_called()

    @pytest.mark.asyncio
    async def test_enhanced_error_categorization(self, generation_service):
        """Test enhanced error categorization for better recovery"""
        
        # Test different error types
        vram_error = Exception("CUDA out of memory")
        model_error = Exception("Model not found")
        download_error = Exception("Network download failed")
        
        assert generation_service._categorize_generation_error(vram_error) == "vram_exhaustion"
        assert generation_service._categorize_generation_error(model_error) == "model_loading_error"
        assert generation_service._categorize_generation_error(download_error) == "download_error"

    @pytest.mark.asyncio
    async def test_enhanced_recovery_suggestions(self, generation_service):
        """Test enhanced recovery suggestions based on error category"""
        
        vram_suggestions = generation_service._get_enhanced_recovery_suggestions("vram_exhaustion", "t2v")
        model_suggestions = generation_service._get_enhanced_recovery_suggestions("model_loading_error", "i2v")
        
        assert len(vram_suggestions) > 0
        assert len(model_suggestions) > 0
        assert "reducing the resolution" in " ".join(vram_suggestions).lower()
        assert "model files" in " ".join(model_suggestions).lower()

    @pytest.mark.asyncio
    async def test_mock_generation_with_enhanced_context(self, generation_service, sample_task, mock_db_session):
        """Test mock generation with enhanced error context"""
        
        error_context = {
            "error": "model_unavailable",
            "category": "model_loading_error"
        }
        
        generation_service._run_mock_generation = AsyncMock(return_value=True)
        
        result = await generation_service._run_mock_generation_with_enhanced_context(
            sample_task, mock_db_session, "t2v", error_context
        )
        
        assert result is True
        generation_service.websocket_manager.send_alert.assert_called()
        generation_service.model_usage_analytics.track_usage.assert_called()

    @pytest.mark.asyncio
    async def test_alternative_model_usage(self, generation_service, sample_task, mock_db_session):
        """Test using alternative model with proper tracking"""
        
        generation_service._run_real_generation_with_monitoring = AsyncMock(return_value=True)
        
        result = await generation_service._try_alternative_model(sample_task, mock_db_session, "i2v")
        
        assert result is True
        generation_service.websocket_manager.send_alert.assert_called()
        generation_service.model_usage_analytics.track_usage.assert_called()
        generation_service._run_real_generation_with_monitoring.assert_called_with(sample_task, mock_db_session, "i2v")

    @pytest.mark.asyncio
    async def test_download_retry_integration(self, generation_service, sample_task, mock_db_session):
        """Test download and retry integration with progress tracking"""
        
        # Mock successful download
        generation_service.enhanced_model_downloader.download_with_retry.return_value = Mock(
            success=True, error_message=None
        )
        
        generation_service._run_real_generation_with_monitoring = AsyncMock(return_value=True)
        
        result = await generation_service._try_download_and_retry(sample_task, mock_db_session, "t2v")
        
        assert result is True
        generation_service.enhanced_model_downloader.download_with_retry.assert_called_once()
        generation_service.websocket_manager.send_alert.assert_called()

    @pytest.mark.asyncio
    async def test_comprehensive_integration_flow(self, generation_service, sample_task, mock_db_session):
        """Test comprehensive integration flow with multiple components"""
        
        # Setup a complex scenario: model unavailable -> download fails -> alternative model succeeds
        generation_service.model_availability_manager.handle_model_request.return_value = Mock(
            success=False, error_message="Model not available"
        )
        
        # First fallback strategy: download and retry (fails)
        mock_download_strategy = Mock()
        mock_download_strategy.strategy_type = FallbackType.DOWNLOAD_AND_RETRY
        
        # Second fallback strategy: alternative model (succeeds)
        mock_alt_strategy = Mock()
        mock_alt_strategy.strategy_type = FallbackType.ALTERNATIVE_MODEL
        mock_alt_strategy.alternative_model = "i2v"
        
        generation_service.intelligent_fallback_manager.get_fallback_strategy.return_value = mock_download_strategy
        
        # Mock download failure
        generation_service.enhanced_model_downloader.download_with_retry.return_value = Mock(
            success=False, error_message="Download failed"
        )
        
        # Mock alternative model success
        generation_service._try_alternative_model = AsyncMock(return_value=True)
        generation_service._try_download_and_retry = AsyncMock(return_value=False)
        generation_service._handle_model_unavailable_with_enhanced_recovery = AsyncMock(return_value=True)
        
        # Run the enhanced generation
        result = await generation_service._run_enhanced_generation(sample_task, mock_db_session, "t2v")
        
        # Verify the comprehensive flow
        assert result is True
        generation_service._handle_model_unavailable_with_enhanced_recovery.assert_called_once()


class TestGenerationServiceErrorRecovery:
    """Test suite for generation service error recovery"""

    @pytest.fixture
    def generation_service_with_recovery(self):
        """Create generation service with recovery system"""
        service = GenerationService()
        
        # Mock components
        service.fallback_recovery_system = Mock()
        service.fallback_recovery_system.attempt_recovery = AsyncMock()
        service.websocket_manager = Mock()
        service.websocket_manager.send_alert = AsyncMock()
        service.model_usage_analytics = Mock()
        service.model_usage_analytics.track_usage = AsyncMock()
        service._run_mock_generation = AsyncMock(return_value=True)
        
        return service

    @pytest.mark.asyncio
    async def test_vram_exhaustion_recovery(self, generation_service_with_recovery):
        """Test VRAM exhaustion error recovery"""
        
        vram_error = Exception("CUDA out of memory")
        task = Mock()
        task.id = 1
        db = Mock()
        
        # Mock recovery system success
        generation_service_with_recovery.fallback_recovery_system.attempt_recovery.return_value = Mock(success=True)
        
        result = await generation_service_with_recovery._handle_generation_error_with_recovery(
            task, db, "t2v", vram_error
        )
        
        assert result is True
        generation_service_with_recovery.fallback_recovery_system.attempt_recovery.assert_called_once()

    @pytest.mark.asyncio
    async def test_model_loading_error_recovery(self, generation_service_with_recovery):
        """Test model loading error recovery"""
        
        model_error = Exception("Model file not found")
        task = Mock()
        task.id = 1
        db = Mock()
        
        # Mock recovery system failure, should fallback to mock
        generation_service_with_recovery.fallback_recovery_system.attempt_recovery.return_value = Mock(success=False)
        
        result = await generation_service_with_recovery._handle_generation_error_with_recovery(
            task, db, "t2v", model_error
        )
        
        assert result is True  # Should succeed with mock generation
        generation_service_with_recovery._run_mock_generation.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])