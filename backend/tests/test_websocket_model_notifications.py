"""
WebSocket Model Notifications Integration Tests
Tests for real-time model status updates, download progress, health monitoring,
and fallback strategy notifications via WebSocket.
"""

import asyncio
import json
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, List, Any

from backend.websocket.manager import ConnectionManager
from backend.websocket.model_notifications import (
    ModelDownloadNotifier,
    ModelHealthNotifier,
    ModelAvailabilityNotifier,
    FallbackNotifier,
    AnalyticsNotifier,
    ModelNotificationIntegrator
)
from backend.core.enhanced_model_downloader import DownloadStatus, DownloadProgress, DownloadResult
from backend.core.model_health_monitor import HealthStatus, IntegrityResult, PerformanceHealth
from backend.core.model_availability_manager import ModelAvailabilityStatus, DetailedModelStatus
from backend.core.intelligent_fallback_manager import FallbackStrategy, ModelSuggestion, FallbackType


class MockWebSocket:
    """Mock WebSocket for testing"""
    
    def __init__(self):
        self.messages: List[str] = []
        self.closed = False
    
    async def send_text(self, message: str):
        if not self.closed:
            self.messages.append(message)
    
    async def accept(self):
        pass
    
    def close(self):
        self.closed = True


@pytest.fixture
def mock_connection_manager():
    """Create a mock connection manager for testing"""
    manager = ConnectionManager()
    manager.active_connections = {"test_conn_1": MockWebSocket(), "test_conn_2": MockWebSocket()}
    manager.subscriptions = {
        "model_status": {"test_conn_1"},
        "download_progress": {"test_conn_1", "test_conn_2"},
        "health_monitoring": {"test_conn_1"},
        "fallback_notifications": {"test_conn_2"},
        "analytics_updates": {"test_conn_1", "test_conn_2"}
    }
    return manager


@pytest.fixture
def download_notifier(mock_connection_manager):
    """Create download notifier with mock connection manager"""
    return ModelDownloadNotifier(mock_connection_manager)


@pytest.fixture
def health_notifier(mock_connection_manager):
    """Create health notifier with mock connection manager"""
    return ModelHealthNotifier(mock_connection_manager)


@pytest.fixture
def availability_notifier(mock_connection_manager):
    """Create availability notifier with mock connection manager"""
    return ModelAvailabilityNotifier(mock_connection_manager)


@pytest.fixture
def fallback_notifier(mock_connection_manager):
    """Create fallback notifier with mock connection manager"""
    return FallbackNotifier(mock_connection_manager)


@pytest.fixture
def analytics_notifier(mock_connection_manager):
    """Create analytics notifier with mock connection manager"""
    return AnalyticsNotifier(mock_connection_manager)


class TestModelDownloadNotifier:
    """Test model download WebSocket notifications"""
    
    @pytest.mark.asyncio
    async def test_download_started_notification(self, download_notifier, mock_connection_manager):
        """Test download started notification"""
        model_id = "test-model"
        download_progress = DownloadProgress(
            model_id=model_id,
            status=DownloadStatus.DOWNLOADING,
            progress_percent=0.0,
            downloaded_mb=0.0,
            total_mb=1000.0,
            speed_mbps=10.0,
            eta_seconds=100.0,
            retry_count=0,
            max_retries=3,
            can_pause=True,
            can_resume=True,
            can_cancel=True
        )
        
        await download_notifier.on_download_started(model_id, download_progress)
        
        # Check that notification was sent to download_progress subscribers
        websocket1 = mock_connection_manager.active_connections["test_conn_1"]
        websocket2 = mock_connection_manager.active_connections["test_conn_2"]
        
        assert len(websocket1.messages) == 1
        assert len(websocket2.messages) == 1
        
        message = json.loads(websocket1.messages[0])
        assert message["type"] == "download_status_change"
        assert message["data"]["model_id"] == model_id
        assert message["data"]["new_status"] == "downloading"
        assert message["data"]["progress_percent"] == 0.0
    
    @pytest.mark.asyncio
    async def test_download_progress_notification(self, download_notifier, mock_connection_manager):
        """Test download progress notification"""
        model_id = "test-model"
        download_progress = DownloadProgress(
            model_id=model_id,
            status=DownloadStatus.DOWNLOADING,
            progress_percent=45.5,
            downloaded_mb=455.0,
            total_mb=1000.0,
            speed_mbps=15.2,
            eta_seconds=36.0,
            retry_count=0,
            max_retries=3,
            can_pause=True,
            can_resume=True,
            can_cancel=True
        )
        
        await download_notifier.on_download_progress(model_id, download_progress)
        
        websocket1 = mock_connection_manager.active_connections["test_conn_1"]
        message = json.loads(websocket1.messages[0])
        
        assert message["type"] == "download_progress_update"
        assert message["data"]["model_id"] == model_id
        assert message["data"]["status"] == "downloading"
        assert message["data"]["progress_percent"] == 45.5
        assert message["data"]["speed_mbps"] == 15.2
    
    @pytest.mark.asyncio
    async def test_download_retry_notification(self, download_notifier, mock_connection_manager):
        """Test download retry notification"""
        model_id = "test-model"
        retry_count = 2
        max_retries = 3
        error_message = "Network timeout"
        
        await download_notifier.on_download_retry(model_id, retry_count, max_retries, error_message)
        
        websocket1 = mock_connection_manager.active_connections["test_conn_1"]
        message = json.loads(websocket1.messages[0])
        
        assert message["type"] == "download_retry"
        assert message["data"]["model_id"] == model_id
        assert message["data"]["retry_count"] == 2
        assert message["data"]["max_retries"] == 3
        assert message["data"]["error_message"] == error_message
        assert message["data"]["next_retry_in_seconds"] == 4  # 2^2
    
    @pytest.mark.asyncio
    async def test_download_completed_notification(self, download_notifier, mock_connection_manager):
        """Test download completed notification"""
        model_id = "test-model"
        download_result = DownloadResult(
            success=True,
            model_id=model_id,
            final_status=DownloadStatus.COMPLETED,
            total_time_seconds=120.5,
            total_retries=1,
            final_size_mb=1024.0,
            integrity_verified=True
        )
        
        await download_notifier.on_download_completed(model_id, download_result)
        
        websocket1 = mock_connection_manager.active_connections["test_conn_1"]
        message = json.loads(websocket1.messages[0])
        
        assert message["type"] == "download_status_change"
        assert message["data"]["model_id"] == model_id
        assert message["data"]["new_status"] == "completed"
        assert message["data"]["progress_percent"] == 100.0
        assert message["data"]["integrity_verified"] is True
    
    @pytest.mark.asyncio
    async def test_download_failed_notification(self, download_notifier, mock_connection_manager):
        """Test download failed notification"""
        model_id = "test-model"
        download_result = DownloadResult(
            success=False,
            model_id=model_id,
            final_status=DownloadStatus.FAILED,
            total_time_seconds=60.0,
            total_retries=3,
            final_size_mb=0.0,
            error_message="Maximum retries exceeded"
        )
        
        await download_notifier.on_download_failed(model_id, download_result)
        
        websocket1 = mock_connection_manager.active_connections["test_conn_1"]
        message = json.loads(websocket1.messages[0])
        
        assert message["type"] == "download_status_change"
        assert message["data"]["model_id"] == model_id
        assert message["data"]["new_status"] == "failed"
        assert message["data"]["error_message"] == "Maximum retries exceeded"
        assert message["data"]["total_retries"] == 3


class TestModelHealthNotifier:
    """Test model health monitoring WebSocket notifications"""
    
    @pytest.mark.asyncio
    async def test_health_check_completed_notification(self, health_notifier, mock_connection_manager):
        """Test health check completed notification"""
        model_id = "test-model"
        integrity_result = IntegrityResult(
            model_id=model_id,
            is_healthy=True,
            health_status=HealthStatus.HEALTHY,
            integrity_score=0.98,
            issues=[],
            file_count=5,
            total_size_mb=1024.0,
            last_checked=datetime.utcnow()
        )
        
        await health_notifier.on_health_check_completed(model_id, integrity_result)
        
        websocket1 = mock_connection_manager.active_connections["test_conn_1"]
        message = json.loads(websocket1.messages[0])
        
        assert message["type"] == "health_monitoring_alert"
        assert message["data"]["model_id"] == model_id
        assert message["data"]["health_status"] == "healthy"
        assert message["data"]["is_healthy"] is True
        assert message["data"]["integrity_score"] == 0.98
    
    @pytest.mark.asyncio
    async def test_corruption_detection_notification(self, health_notifier, mock_connection_manager):
        """Test corruption detection notification"""
        model_id = "test-model"
        corruption_type = "checksum_mismatch"
        severity = "high"
        repair_action = "Re-download corrupted files"
        
        await health_notifier.on_corruption_detected(model_id, corruption_type, severity, repair_action)
        
        websocket1 = mock_connection_manager.active_connections["test_conn_1"]
        message = json.loads(websocket1.messages[0])
        
        assert message["type"] == "corruption_detection"
        assert message["data"]["model_id"] == model_id
        assert message["data"]["corruption_type"] == corruption_type
        assert message["data"]["severity"] == severity
        assert message["data"]["repair_action"] == repair_action
        assert message["data"]["requires_user_action"] is True
    
    @pytest.mark.asyncio
    async def test_performance_degradation_notification(self, health_notifier, mock_connection_manager):
        """Test performance degradation notification"""
        model_id = "test-model"
        performance_health = PerformanceHealth(
            model_id=model_id,
            overall_score=0.65,
            performance_trend="degrading",
            bottlenecks=["slow_generation"],
            recommendations=["Consider model optimization"],
            last_assessment=datetime.utcnow()
        )
        
        await health_notifier.on_performance_degradation(model_id, performance_health)
        
        websocket1 = mock_connection_manager.active_connections["test_conn_1"]
        message = json.loads(websocket1.messages[0])
        
        assert message["type"] == "health_monitoring_alert"
        assert message["data"]["model_id"] == model_id
        assert message["data"]["health_status"] == "degraded"
        assert message["data"]["overall_score"] == 0.65
        assert "slow_generation" in message["data"]["bottlenecks"]
    
    @pytest.mark.asyncio
    async def test_automatic_repair_notifications(self, health_notifier, mock_connection_manager):
        """Test automatic repair start and completion notifications"""
        model_id = "test-model"
        repair_type = "file_redownload"
        
        # Test repair started
        await health_notifier.on_automatic_repair_started(model_id, repair_type)
        
        websocket1 = mock_connection_manager.active_connections["test_conn_1"]
        message = json.loads(websocket1.messages[0])
        
        assert message["type"] == "health_monitoring_alert"
        assert message["data"]["model_id"] == model_id
        assert message["data"]["health_status"] == "repairing"
        assert message["data"]["repair_type"] == repair_type
        
        # Test repair completed
        await health_notifier.on_automatic_repair_completed(model_id, True, "Successfully re-downloaded corrupted files")
        
        message = json.loads(websocket1.messages[1])
        assert message["type"] == "health_monitoring_alert"
        assert message["data"]["health_status"] == "healthy"
        assert message["data"]["repair_success"] is True


class TestModelAvailabilityNotifier:
    """Test model availability WebSocket notifications"""
    
    @pytest.mark.asyncio
    async def test_availability_change_notification(self, availability_notifier, mock_connection_manager):
        """Test model availability change notification"""
        model_id = "test-model"
        old_status = ModelAvailabilityStatus.MISSING
        new_status = ModelAvailabilityStatus.DOWNLOADING
        reason = "User requested download"
        
        await availability_notifier.on_availability_changed(model_id, old_status, new_status, reason)
        
        websocket1 = mock_connection_manager.active_connections["test_conn_1"]
        message = json.loads(websocket1.messages[0])
        
        assert message["type"] == "model_availability_change"
        assert message["data"]["model_id"] == model_id
        assert message["data"]["old_availability"] == "missing"
        assert message["data"]["new_availability"] == "downloading"
        assert message["data"]["reason"] == reason
    
    @pytest.mark.asyncio
    async def test_model_status_update_notification(self, availability_notifier, mock_connection_manager):
        """Test detailed model status update notification"""
        model_id = "test-model"
        detailed_status = DetailedModelStatus(
            model_id=model_id,
            model_type="t2v",
            is_available=True,
            is_loaded=False,
            size_mb=1024.0,
            availability_status=ModelAvailabilityStatus.AVAILABLE,
            download_progress=None,
            missing_files=[],
            integrity_score=0.95,
            last_health_check=datetime.utcnow(),
            performance_score=0.88,
            corruption_detected=False,
            usage_frequency=2.5,
            last_used=datetime.utcnow() - timedelta(hours=2),
            average_generation_time=25.3,
            can_pause_download=False,
            can_resume_download=False,
            estimated_download_time=None,
            current_version="1.0.0",
            latest_version="1.0.0",
            update_available=False
        )
        
        await availability_notifier.on_model_status_update(model_id, detailed_status)
        
        websocket1 = mock_connection_manager.active_connections["test_conn_1"]
        message = json.loads(websocket1.messages[0])
        
        assert message["type"] == "model_status_update"
        assert message["data"]["model_id"] == model_id
        assert message["data"]["availability_status"] == "available"
        assert message["data"]["integrity_score"] == 0.95
        assert message["data"]["performance_score"] == 0.88
        assert message["data"]["usage_frequency"] == 2.5
    
    @pytest.mark.asyncio
    async def test_batch_status_update_notification(self, availability_notifier, mock_connection_manager):
        """Test batch model status update notification"""
        models_status = {
            "model1": DetailedModelStatus(
                model_id="model1",
                model_type="t2v",
                is_available=True,
                is_loaded=True,
                size_mb=512.0,
                availability_status=ModelAvailabilityStatus.AVAILABLE,
                integrity_score=0.98,
                performance_score=0.92,
                usage_frequency=5.2,
                last_health_check=datetime.utcnow(),
                corruption_detected=False,
                current_version="1.0.0",
                latest_version="1.0.0",
                update_available=False
            ),
            "model2": DetailedModelStatus(
                model_id="model2",
                model_type="i2v",
                is_available=False,
                is_loaded=False,
                size_mb=0.0,
                availability_status=ModelAvailabilityStatus.MISSING,
                integrity_score=0.0,
                performance_score=0.0,
                usage_frequency=0.0,
                last_health_check=datetime.utcnow(),
                corruption_detected=False,
                current_version="",
                latest_version="1.1.0",
                update_available=False
            )
        }
        
        await availability_notifier.on_batch_status_update(models_status)
        
        websocket1 = mock_connection_manager.active_connections["test_conn_1"]
        message = json.loads(websocket1.messages[0])
        
        assert message["type"] == "batch_model_status_update"
        assert message["data"]["count"] == 2
        assert len(message["data"]["models"]) == 2
        
        model1_data = next(m for m in message["data"]["models"] if m["model_id"] == "model1")
        assert model1_data["availability_status"] == "available"
        assert model1_data["integrity_score"] == 0.98


class TestFallbackNotifier:
    """Test fallback strategy WebSocket notifications"""
    
    @pytest.mark.asyncio
    async def test_fallback_strategy_notification(self, fallback_notifier, mock_connection_manager):
        """Test fallback strategy activation notification"""
        original_model = "unavailable-model"
        fallback_strategy = FallbackStrategy(
            strategy_type=FallbackType.ALTERNATIVE_MODEL,
            recommended_action="Use alternative model",
            alternative_model="alternative-model",
            estimated_wait_time=None,
            user_message="Suggested alternative model available",
            can_queue_request=False
        )
        
        await fallback_notifier.on_fallback_strategy_activated(original_model, fallback_strategy)
        
        websocket2 = mock_connection_manager.active_connections["test_conn_2"]
        message = json.loads(websocket2.messages[0])
        
        assert message["type"] == "fallback_strategy"
        assert message["data"]["original_model"] == original_model
        assert message["data"]["strategy_type"] == "alternative_model"
        assert message["data"]["alternative_model"] == "alternative-model"
        assert message["data"]["user_interaction_required"] is False
    
    @pytest.mark.asyncio
    async def test_alternative_model_suggestion_notification(self, fallback_notifier, mock_connection_manager):
        """Test alternative model suggestion notification"""
        original_model = "requested-model"
        suggestion = ModelSuggestion(
            suggested_model="suggested-model",
            compatibility_score=0.85,
            performance_difference=-0.1,
            availability_status=ModelAvailabilityStatus.AVAILABLE,
            reason="Similar capabilities with slightly lower performance",
            estimated_quality_difference="slightly_lower"
        )
        
        await fallback_notifier.on_alternative_model_suggested(original_model, suggestion)
        
        websocket2 = mock_connection_manager.active_connections["test_conn_2"]
        message = json.loads(websocket2.messages[0])
        
        assert message["type"] == "alternative_model_suggestion"
        assert message["data"]["original_model"] == original_model
        assert message["data"]["suggested_model"] == "suggested-model"
        assert message["data"]["compatibility_score"] == 0.85
        assert message["data"]["performance_difference"] == -0.1
    
    @pytest.mark.asyncio
    async def test_model_queue_notification(self, fallback_notifier, mock_connection_manager):
        """Test model queue notification"""
        model_id = "queued-model"
        queue_position = 2
        estimated_wait_time = timedelta(minutes=5)
        
        await fallback_notifier.on_model_queued(model_id, queue_position, estimated_wait_time)
        
        websocket2 = mock_connection_manager.active_connections["test_conn_2"]
        message = json.loads(websocket2.messages[0])
        
        assert message["type"] == "model_queue_update"
        assert message["data"]["model_id"] == model_id
        assert message["data"]["queue_position"] == 2
        assert message["data"]["estimated_wait_time"] == 300.0  # 5 minutes in seconds


class TestAnalyticsNotifier:
    """Test analytics WebSocket notifications"""
    
    @pytest.mark.asyncio
    async def test_usage_statistics_update_notification(self, analytics_notifier, mock_connection_manager):
        """Test usage statistics update notification"""
        model_usage_data = {
            "total_models": 5,
            "active_models": 3,
            "most_used_model": "popular-model",
            "usage_trends": {
                "daily_usage": 25,
                "weekly_usage": 150,
                "monthly_usage": 600
            }
        }
        
        await analytics_notifier.on_usage_statistics_updated(model_usage_data)
        
        websocket1 = mock_connection_manager.active_connections["test_conn_1"]
        message = json.loads(websocket1.messages[0])
        
        assert message["type"] == "usage_statistics_update"
        assert message["data"]["total_models"] == 5
        assert message["data"]["most_used_model"] == "popular-model"
    
    @pytest.mark.asyncio
    async def test_cleanup_recommendation_notification(self, analytics_notifier, mock_connection_manager):
        """Test cleanup recommendation notification"""
        recommendation_data = {
            "recommended_cleanup": ["unused-model-1", "unused-model-2"],
            "potential_space_saved_mb": 2048.0,
            "models_not_used_days": 30,
            "cleanup_priority": "medium"
        }
        
        await analytics_notifier.on_cleanup_recommendation_generated(recommendation_data)
        
        websocket1 = mock_connection_manager.active_connections["test_conn_1"]
        message = json.loads(websocket1.messages[0])
        
        assert message["type"] == "cleanup_recommendation"
        assert message["data"]["potential_space_saved_mb"] == 2048.0
        assert len(message["data"]["recommended_cleanup"]) == 2
    
    @pytest.mark.asyncio
    async def test_performance_analytics_update_notification(self, analytics_notifier, mock_connection_manager):
        """Test performance analytics update notification"""
        analytics_data = {
            "average_download_speed": 15.5,
            "average_generation_time": 28.3,
            "success_rate": 0.95,
            "error_rate": 0.05,
            "peak_usage_hours": [14, 15, 16, 20, 21]
        }
        
        await analytics_notifier.on_performance_analytics_updated(analytics_data)
        
        websocket1 = mock_connection_manager.active_connections["test_conn_1"]
        message = json.loads(websocket1.messages[0])
        
        assert message["type"] == "analytics_update"
        assert message["data"]["analytics_type"] == "performance"
        assert message["data"]["average_download_speed"] == 15.5
        assert message["data"]["success_rate"] == 0.95


class TestModelNotificationIntegrator:
    """Test the main model notification integrator"""
    
    @pytest.mark.asyncio
    async def test_integrator_initialization(self, mock_connection_manager):
        """Test integrator initialization"""
        integrator = ModelNotificationIntegrator(mock_connection_manager)
        
        assert integrator.connection_manager == mock_connection_manager
        assert isinstance(integrator.download_notifier, ModelDownloadNotifier)
        assert isinstance(integrator.health_notifier, ModelHealthNotifier)
        assert isinstance(integrator.availability_notifier, ModelAvailabilityNotifier)
        assert isinstance(integrator.fallback_notifier, FallbackNotifier)
        assert isinstance(integrator.analytics_notifier, AnalyticsNotifier)
    
    @pytest.mark.asyncio
    async def test_component_integration_setup(self, mock_connection_manager):
        """Test component integration setup"""
        integrator = ModelNotificationIntegrator(mock_connection_manager)
        
        # Mock components
        mock_downloader = Mock()
        mock_health_monitor = Mock()
        mock_availability_manager = Mock()
        mock_fallback_manager = Mock()
        
        await integrator.setup_component_integrations(
            enhanced_downloader=mock_downloader,
            health_monitor=mock_health_monitor,
            availability_manager=mock_availability_manager,
            fallback_manager=mock_fallback_manager
        )
        
        # Verify callbacks were set
        mock_downloader.set_progress_callback.assert_called_once()
        mock_downloader.set_status_change_callback.assert_called_once()
        mock_downloader.set_retry_callback.assert_called_once()
        
        mock_health_monitor.set_health_check_callback.assert_called_once()
        mock_health_monitor.set_corruption_callback.assert_called_once()
        mock_health_monitor.set_performance_callback.assert_called_once()
        
        mock_availability_manager.set_status_change_callback.assert_called_once()
        mock_availability_manager.set_batch_update_callback.assert_called_once()
        
        mock_fallback_manager.set_fallback_callback.assert_called_once()
        mock_fallback_manager.set_suggestion_callback.assert_called_once()


class TestWebSocketManagerEnhancements:
    """Test the enhanced WebSocket manager methods"""
    
    @pytest.mark.asyncio
    async def test_enhanced_subscription_topics(self):
        """Test that enhanced subscription topics are available"""
        manager = ConnectionManager()
        
        expected_topics = [
            "system_stats", "generation_progress", "queue_updates", "alerts",
            "model_status", "download_progress", "health_monitoring", 
            "fallback_notifications", "analytics_updates"
        ]
        
        for topic in expected_topics:
            assert topic in manager.subscriptions
    
    @pytest.mark.asyncio
    async def test_model_status_update_method(self):
        """Test model status update WebSocket method"""
        manager = ConnectionManager()
        mock_websocket = MockWebSocket()
        
        # Setup connection and subscription
        await manager.connect(mock_websocket, "test_conn")
        await manager.subscribe("test_conn", "model_status")
        
        # Send model status update
        await manager.send_model_status_update(
            model_id="test-model",
            availability_status="available",
            integrity_score=0.95,
            performance_score=0.88
        )
        
        assert len(mock_websocket.messages) == 1
        message = json.loads(mock_websocket.messages[0])
        assert message["type"] == "model_status_update"
        assert message["data"]["model_id"] == "test-model"
        assert message["data"]["availability_status"] == "available"
    
    @pytest.mark.asyncio
    async def test_download_progress_update_method(self):
        """Test download progress update WebSocket method"""
        manager = ConnectionManager()
        mock_websocket = MockWebSocket()
        
        await manager.connect(mock_websocket, "test_conn")
        await manager.subscribe("test_conn", "download_progress")
        
        await manager.send_download_progress_update(
            model_id="test-model",
            status="downloading",
            progress_percent=65.5,
            speed_mbps=12.3
        )
        
        assert len(mock_websocket.messages) == 1
        message = json.loads(mock_websocket.messages[0])
        assert message["type"] == "download_progress_update"
        assert message["data"]["progress_percent"] == 65.5
    
    @pytest.mark.asyncio
    async def test_health_monitoring_alert_method(self):
        """Test health monitoring alert WebSocket method"""
        manager = ConnectionManager()
        mock_websocket = MockWebSocket()
        
        await manager.connect(mock_websocket, "test_conn")
        await manager.subscribe("test_conn", "health_monitoring")
        
        await manager.send_health_monitoring_alert(
            model_id="test-model",
            health_status="degraded",
            alert_data={
                "performance_score": 0.65,
                "issue": "Slow generation times"
            }
        )
        
        assert len(mock_websocket.messages) == 1
        message = json.loads(mock_websocket.messages[0])
        assert message["type"] == "health_monitoring_alert"
        assert message["data"]["health_status"] == "degraded"
    
    @pytest.mark.asyncio
    async def test_fallback_strategy_notification_method(self):
        """Test fallback strategy notification WebSocket method"""
        manager = ConnectionManager()
        mock_websocket = MockWebSocket()
        
        await manager.connect(mock_websocket, "test_conn")
        await manager.subscribe("test_conn", "fallback_notifications")
        
        await manager.send_fallback_strategy_notification(
            original_model="unavailable-model",
            fallback_data={
                "strategy_type": "alternative_model",
                "alternative_model": "backup-model"
            },
            user_interaction_required=True
        )
        
        assert len(mock_websocket.messages) == 1
        message = json.loads(mock_websocket.messages[0])
        assert message["type"] == "fallback_strategy"
        assert message["data"]["user_interaction_required"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])