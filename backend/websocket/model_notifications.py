"""
Model Notifications WebSocket Integration
Integrates enhanced model availability components with WebSocket real-time notifications.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta

from backend.websocket.manager import ConnectionManager, get_connection_manager
from backend.core.enhanced_model_downloader import (
    EnhancedModelDownloader, DownloadStatus, DownloadProgress, DownloadResult
)
from backend.core.model_health_monitor import (
    ModelHealthMonitor, HealthStatus, IntegrityResult, PerformanceHealth
)
from backend.core.model_availability_manager import (
    ModelAvailabilityManager, ModelAvailabilityStatus, DetailedModelStatus
)
from backend.core.intelligent_fallback_manager import (
    IntelligentFallbackManager, FallbackStrategy, ModelSuggestion, FallbackType
)

logger = logging.getLogger(__name__)


class ModelDownloadNotifier:
    """Handles WebSocket notifications for model download events"""
    
    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
        self._active_downloads: Dict[str, DownloadProgress] = {}
    
    async def on_download_started(self, model_id: str, download_progress: DownloadProgress):
        """Handle download started event"""
        self._active_downloads[model_id] = download_progress
        
        await self.connection_manager.send_download_status_change(
            model_id=model_id,
            old_status="queued",
            new_status=download_progress.status.value,
            progress_percent=download_progress.progress_percent,
            total_size_mb=download_progress.total_mb,
            download_speed_mbps=download_progress.speed_mbps
        )
        
        logger.info(f"Download started notification sent for model: {model_id}")
    
    async def on_download_progress(self, model_id: str, download_progress: DownloadProgress):
        """Handle download progress updates"""
        self._active_downloads[model_id] = download_progress
        
        await self.connection_manager.send_download_progress_update(
            model_id=model_id,
            status=download_progress.status.value,
            progress_percent=download_progress.progress_percent,
            downloaded_mb=download_progress.downloaded_mb,
            total_mb=download_progress.total_mb,
            speed_mbps=download_progress.speed_mbps,
            eta_seconds=download_progress.eta_seconds,
            can_pause=download_progress.can_pause,
            can_resume=download_progress.can_resume,
            can_cancel=download_progress.can_cancel
        )
    
    async def on_download_paused(self, model_id: str, download_progress: DownloadProgress):
        """Handle download paused event"""
        await self.connection_manager.send_download_status_change(
            model_id=model_id,
            old_status="downloading",
            new_status="paused",
            progress_percent=download_progress.progress_percent,
            reason="User requested pause"
        )
        
        logger.info(f"Download paused notification sent for model: {model_id}")
    
    async def on_download_resumed(self, model_id: str, download_progress: DownloadProgress):
        """Handle download resumed event"""
        await self.connection_manager.send_download_status_change(
            model_id=model_id,
            old_status="paused",
            new_status="downloading",
            progress_percent=download_progress.progress_percent,
            reason="User requested resume"
        )
        
        logger.info(f"Download resumed notification sent for model: {model_id}")
    
    async def on_download_retry(self, model_id: str, retry_count: int, max_retries: int, 
                              error_message: str = ""):
        """Handle download retry event"""
        await self.connection_manager.send_download_retry_notification(
            model_id=model_id,
            retry_count=retry_count,
            max_retries=max_retries,
            error_message=error_message,
            next_retry_in_seconds=min(2 ** retry_count, 60)  # Exponential backoff
        )
        
        logger.info(f"Download retry notification sent for model: {model_id} (attempt {retry_count}/{max_retries})")
    
    async def on_download_completed(self, model_id: str, download_result: DownloadResult):
        """Handle download completed event"""
        if model_id in self._active_downloads:
            del self._active_downloads[model_id]
        
        await self.connection_manager.send_download_status_change(
            model_id=model_id,
            old_status="downloading",
            new_status="completed",
            progress_percent=100.0,
            total_time_seconds=download_result.total_time_seconds,
            final_size_mb=download_result.final_size_mb,
            integrity_verified=download_result.integrity_verified
        )
        
        logger.info(f"Download completed notification sent for model: {model_id}")
    
    async def on_download_failed(self, model_id: str, download_result: DownloadResult):
        """Handle download failed event"""
        if model_id in self._active_downloads:
            del self._active_downloads[model_id]
        
        await self.connection_manager.send_download_status_change(
            model_id=model_id,
            old_status="downloading",
            new_status="failed",
            progress_percent=0.0,
            error_message=download_result.error_message,
            total_retries=download_result.total_retries
        )
        
        logger.error(f"Download failed notification sent for model: {model_id}")
    
    async def on_download_cancelled(self, model_id: str):
        """Handle download cancelled event"""
        if model_id in self._active_downloads:
            del self._active_downloads[model_id]
        
        await self.connection_manager.send_download_status_change(
            model_id=model_id,
            old_status="downloading",
            new_status="cancelled",
            progress_percent=0.0,
            reason="User requested cancellation"
        )
        
        logger.info(f"Download cancelled notification sent for model: {model_id}")


class ModelHealthNotifier:
    """Handles WebSocket notifications for model health monitoring events"""
    
    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
    
    async def on_health_check_completed(self, model_id: str, integrity_result: IntegrityResult):
        """Handle health check completion"""
        await self.connection_manager.send_health_monitoring_alert(
            model_id=model_id,
            health_status=integrity_result.health_status.value,
            is_healthy=integrity_result.is_healthy,
            integrity_score=integrity_result.integrity_score,
            issues_found=integrity_result.issues,
            file_count=integrity_result.file_count,
            total_size_mb=integrity_result.total_size_mb
        )
        
        logger.info(f"Health check notification sent for model: {model_id} - Status: {integrity_result.health_status.value}")
    
    async def on_corruption_detected(self, model_id: str, corruption_type: str, 
                                   severity: str, repair_action: str = ""):
        """Handle corruption detection"""
        await self.connection_manager.send_corruption_detection_alert(
            model_id=model_id,
            corruption_type=corruption_type,
            severity=severity,
            repair_action=repair_action,
            requires_user_action=severity in ["high", "critical"]
        )
        
        logger.warning(f"Corruption detected notification sent for model: {model_id} - Type: {corruption_type}")
    
    async def on_performance_degradation(self, model_id: str, performance_health: PerformanceHealth):
        """Handle performance degradation detection"""
        await self.connection_manager.send_health_monitoring_alert(
            model_id=model_id,
            health_status="degraded",
            overall_score=performance_health.overall_score,
            performance_trend=performance_health.performance_trend,
            bottlenecks=performance_health.bottlenecks,
            recommendations=performance_health.recommendations
        )
        
        logger.warning(f"Performance degradation notification sent for model: {model_id}")
    
    async def on_automatic_repair_started(self, model_id: str, repair_type: str):
        """Handle automatic repair start"""
        await self.connection_manager.send_health_monitoring_alert(
            model_id=model_id,
            health_status="repairing",
            repair_type=repair_type,
            status="started",
            estimated_duration_minutes=5  # Default estimate
        )
        
        logger.info(f"Automatic repair started notification sent for model: {model_id}")
    
    async def on_automatic_repair_completed(self, model_id: str, repair_success: bool, 
                                          repair_details: str = ""):
        """Handle automatic repair completion"""
        status = "healthy" if repair_success else "failed_repair"
        
        await self.connection_manager.send_health_monitoring_alert(
            model_id=model_id,
            health_status=status,
            repair_success=repair_success,
            repair_details=repair_details,
            status="completed"
        )
        
        logger.info(f"Automatic repair completed notification sent for model: {model_id} - Success: {repair_success}")


class ModelAvailabilityNotifier:
    """Handles WebSocket notifications for model availability changes"""
    
    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
        self._last_status: Dict[str, ModelAvailabilityStatus] = {}
    
    async def on_availability_changed(self, model_id: str, old_status: ModelAvailabilityStatus, 
                                    new_status: ModelAvailabilityStatus, reason: str = ""):
        """Handle model availability change"""
        self._last_status[model_id] = new_status
        
        await self.connection_manager.send_model_availability_change(
            model_id=model_id,
            old_availability=old_status.value,
            new_availability=new_status.value,
            reason=reason
        )
        
        logger.info(f"Availability change notification sent for model: {model_id} - {old_status.value} -> {new_status.value}")
    
    async def on_model_status_update(self, model_id: str, detailed_status: DetailedModelStatus):
        """Handle detailed model status update"""
        await self.connection_manager.send_model_status_update(
            model_id=model_id,
            availability_status=detailed_status.availability_status.value,
            is_available=detailed_status.is_available,
            is_loaded=detailed_status.is_loaded,
            size_mb=detailed_status.size_mb,
            download_progress=detailed_status.download_progress,
            missing_files=detailed_status.missing_files,
            integrity_score=detailed_status.integrity_score,
            performance_score=detailed_status.performance_score,
            usage_frequency=detailed_status.usage_frequency,
            last_used=detailed_status.last_used.isoformat() if detailed_status.last_used else None,
            update_available=detailed_status.update_available,
            current_version=detailed_status.current_version,
            latest_version=detailed_status.latest_version
        )
    
    async def on_batch_status_update(self, models_status: Dict[str, DetailedModelStatus]):
        """Handle batch model status update"""
        models_data = []
        for model_id, status in models_status.items():
            models_data.append({
                "model_id": model_id,
                "availability_status": status.availability_status.value,
                "is_available": status.is_available,
                "integrity_score": status.integrity_score,
                "performance_score": status.performance_score,
                "usage_frequency": status.usage_frequency
            })
        
        await self.connection_manager.send_batch_model_status_update(models_data)
        
        logger.info(f"Batch status update notification sent for {len(models_data)} models")


class FallbackNotifier:
    """Handles WebSocket notifications for fallback strategy events"""
    
    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
    
    async def on_fallback_strategy_activated(self, original_model: str, fallback_strategy: FallbackStrategy):
        """Handle fallback strategy activation"""
        user_interaction_required = fallback_strategy.strategy_type in [
            FallbackType.QUEUE_AND_WAIT, 
            FallbackType.DOWNLOAD_AND_RETRY
        ]
        
        await self.connection_manager.send_fallback_strategy_notification(
            original_model=original_model,
            strategy_type=fallback_strategy.strategy_type.value,
            recommended_action=fallback_strategy.recommended_action,
            alternative_model=fallback_strategy.alternative_model,
            estimated_wait_time=fallback_strategy.estimated_wait_time.total_seconds() if fallback_strategy.estimated_wait_time else None,
            user_message=fallback_strategy.user_message,
            can_queue_request=fallback_strategy.can_queue_request,
            user_interaction_required=user_interaction_required
        )
        
        logger.info(f"Fallback strategy notification sent for model: {original_model} - Strategy: {fallback_strategy.strategy_type.value}")
    
    async def on_alternative_model_suggested(self, original_model: str, suggestion: ModelSuggestion):
        """Handle alternative model suggestion"""
        await self.connection_manager.send_alternative_model_suggestion(
            original_model=original_model,
            suggested_model=suggestion.suggested_model,
            compatibility_score=suggestion.compatibility_score,
            reason=suggestion.reason,
            performance_difference=suggestion.performance_difference,
            quality_difference=suggestion.estimated_quality_difference,
            availability_status=suggestion.availability_status.value
        )
        
        logger.info(f"Alternative model suggestion sent: {original_model} -> {suggestion.suggested_model}")
    
    async def on_model_queued(self, model_id: str, queue_position: int, 
                            estimated_wait_time: Optional[timedelta] = None):
        """Handle model queued for download"""
        wait_time_seconds = estimated_wait_time.total_seconds() if estimated_wait_time else None
        
        await self.connection_manager.send_model_queue_notification(
            model_id=model_id,
            queue_position=queue_position,
            estimated_wait_time=wait_time_seconds
        )
        
        logger.info(f"Model queue notification sent for: {model_id} - Position: {queue_position}")


class AnalyticsNotifier:
    """Handles WebSocket notifications for analytics updates"""
    
    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
    
    async def on_usage_statistics_updated(self, model_usage_data: Dict[str, Any]):
        """Handle usage statistics update"""
        await self.connection_manager.send_usage_statistics_update(model_usage_data)
        
        logger.info("Usage statistics update notification sent")
    
    async def on_cleanup_recommendation_generated(self, recommendation_data: Dict[str, Any]):
        """Handle cleanup recommendation"""
        await self.connection_manager.send_cleanup_recommendation(recommendation_data)
        
        logger.info("Cleanup recommendation notification sent")
    
    async def on_performance_analytics_updated(self, analytics_data: Dict[str, Any]):
        """Handle performance analytics update"""
        await self.connection_manager.send_analytics_update(
            analytics_type="performance",
            analytics_data=analytics_data
        )
        
        logger.info("Performance analytics update notification sent")


class ModelNotificationIntegrator:
    """Main integration class that coordinates all model-related WebSocket notifications"""
    
    def __init__(self, connection_manager: Optional[ConnectionManager] = None):
        self.connection_manager = connection_manager or get_connection_manager()
        
        # Initialize notifiers
        self.download_notifier = ModelDownloadNotifier(self.connection_manager)
        self.health_notifier = ModelHealthNotifier(self.connection_manager)
        self.availability_notifier = ModelAvailabilityNotifier(self.connection_manager)
        self.fallback_notifier = FallbackNotifier(self.connection_manager)
        self.analytics_notifier = AnalyticsNotifier(self.connection_manager)
        
        logger.info("Model notification integrator initialized")
    
    def get_download_notifier(self) -> ModelDownloadNotifier:
        """Get the download notifier instance"""
        return self.download_notifier
    
    def get_health_notifier(self) -> ModelHealthNotifier:
        """Get the health notifier instance"""
        return self.health_notifier
    
    def get_availability_notifier(self) -> ModelAvailabilityNotifier:
        """Get the availability notifier instance"""
        return self.availability_notifier
    
    def get_fallback_notifier(self) -> FallbackNotifier:
        """Get the fallback notifier instance"""
        return self.fallback_notifier
    
    def get_analytics_notifier(self) -> AnalyticsNotifier:
        """Get the analytics notifier instance"""
        return self.analytics_notifier
    
    async def setup_component_integrations(self, 
                                         enhanced_downloader: Optional[EnhancedModelDownloader] = None,
                                         health_monitor: Optional[ModelHealthMonitor] = None,
                                         availability_manager: Optional[ModelAvailabilityManager] = None,
                                         fallback_manager: Optional[IntelligentFallbackManager] = None):
        """Setup integrations with enhanced model components"""
        
        # Setup download event callbacks
        if enhanced_downloader:
            # Note: These callback methods would need to be implemented in the actual enhanced_downloader
            if hasattr(enhanced_downloader, 'set_progress_callback'):
                enhanced_downloader.set_progress_callback(self.download_notifier.on_download_progress)
            if hasattr(enhanced_downloader, 'set_status_change_callback'):
                enhanced_downloader.set_status_change_callback(self.download_notifier.on_download_started)
            if hasattr(enhanced_downloader, 'set_retry_callback'):
                enhanced_downloader.set_retry_callback(self.download_notifier.on_download_retry)
            logger.info("Enhanced downloader callbacks configured")
        
        # Setup health monitoring callbacks
        if health_monitor:
            if hasattr(health_monitor, 'set_health_check_callback'):
                health_monitor.set_health_check_callback(self.health_notifier.on_health_check_completed)
            if hasattr(health_monitor, 'set_corruption_callback'):
                health_monitor.set_corruption_callback(self.health_notifier.on_corruption_detected)
            if hasattr(health_monitor, 'set_performance_callback'):
                health_monitor.set_performance_callback(self.health_notifier.on_performance_degradation)
            logger.info("Health monitor callbacks configured")
        
        # Setup availability manager callbacks
        if availability_manager:
            if hasattr(availability_manager, 'set_status_change_callback'):
                availability_manager.set_status_change_callback(self.availability_notifier.on_availability_changed)
            if hasattr(availability_manager, 'set_batch_update_callback'):
                availability_manager.set_batch_update_callback(self.availability_notifier.on_batch_status_update)
            logger.info("Availability manager callbacks configured")
        
        # Setup fallback manager callbacks
        if fallback_manager:
            if hasattr(fallback_manager, 'set_fallback_callback'):
                fallback_manager.set_fallback_callback(self.fallback_notifier.on_fallback_strategy_activated)
            if hasattr(fallback_manager, 'set_suggestion_callback'):
                fallback_manager.set_suggestion_callback(self.fallback_notifier.on_alternative_model_suggested)
            logger.info("Fallback manager callbacks configured")
        
        logger.info("All component integrations setup completed")


# Global integrator instance
_model_notification_integrator: Optional[ModelNotificationIntegrator] = None

def get_model_notification_integrator() -> ModelNotificationIntegrator:
    """Get the global model notification integrator instance"""
    global _model_notification_integrator
    if _model_notification_integrator is None:
        _model_notification_integrator = ModelNotificationIntegrator()
    return _model_notification_integrator