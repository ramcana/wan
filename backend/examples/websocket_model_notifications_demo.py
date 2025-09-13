from unittest.mock import Mock, patch
"""
WebSocket Model Notifications Demo
Demonstrates how to integrate and use the enhanced WebSocket model notifications system.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any

from backend.websocket.manager import get_connection_manager
from backend.websocket.model_notifications import get_model_notification_integrator
from backend.core.enhanced_model_downloader import DownloadStatus, DownloadProgress, DownloadResult
from backend.core.model_health_monitor import HealthStatus, IntegrityResult
from backend.core.model_availability_manager import ModelAvailabilityStatus, DetailedModelStatus
from backend.core.intelligent_fallback_manager import FallbackStrategy, ModelSuggestion, FallbackType

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockWebSocketConnection:
    """Mock WebSocket connection for demonstration"""
    
    def __init__(self, connection_id: str):
        self.connection_id = connection_id
        self.messages: list = []
        self.subscriptions: set = set()
    
    async def send_text(self, message: str):
        """Simulate receiving a WebSocket message"""
        data = json.loads(message)
        self.messages.append(data)
        logger.info(f"[{self.connection_id}] Received: {data['type']}")
        
        # Pretty print the message for demo purposes
        if data['type'] in ['download_progress_update', 'model_status_update']:
            self._print_progress_update(data)
        elif data['type'] in ['health_monitoring_alert', 'corruption_detection']:
            self._print_health_alert(data)
        elif data['type'] in ['fallback_strategy', 'alternative_model_suggestion']:
            self._print_fallback_notification(data)
        else:
            logger.info(f"  Data: {json.dumps(data['data'], indent=2)}")
    
    def _print_progress_update(self, data: Dict[str, Any]):
        """Print download progress in a user-friendly format"""
        d = data['data']
        if 'progress_percent' in d:
            logger.info(f"  📥 {d['model_id']}: {d['progress_percent']:.1f}% "
                       f"({d.get('speed_mbps', 0):.1f} MB/s)")
    
    def _print_health_alert(self, data: Dict[str, Any]):
        """Print health alerts in a user-friendly format"""
        d = data['data']
        status_emoji = {"healthy": "✅", "degraded": "⚠️", "corrupted": "❌"}.get(d['health_status'], "❓")
        logger.info(f"  {status_emoji} {d['model_id']}: {d['health_status']}")
    
    def _print_fallback_notification(self, data: Dict[str, Any]):
        """Print fallback notifications in a user-friendly format"""
        d = data['data']
        if data['type'] == 'alternative_model_suggestion':
            logger.info(f"  🔄 Suggest {d['suggested_model']} instead of {d['original_model']} "
                       f"(compatibility: {d['compatibility_score']:.0%})")
        else:
            logger.info(f"  🔄 Fallback strategy: {d.get('strategy_type', 'unknown')}")
    
    async def accept(self):
        """Mock WebSocket accept"""
        pass


async def demo_download_notifications():
    """Demonstrate download progress notifications"""
    logger.info("\n=== Download Notifications Demo ===")
    
    # Get components
    connection_manager = get_connection_manager()
    integrator = get_model_notification_integrator()
    download_notifier = integrator.get_download_notifier()
    
    # Setup mock connection
    mock_ws = MockWebSocketConnection("demo_user_1")
    await connection_manager.connect(mock_ws, "demo_user_1")
    await connection_manager.subscribe("demo_user_1", "download_progress")
    
    model_id = "t2v-a14b"
    
    # Simulate download started
    logger.info("Starting download simulation...")
    download_progress = DownloadProgress(
        model_id=model_id,
        status=DownloadStatus.DOWNLOADING,
        progress_percent=0.0,
        downloaded_mb=0.0,
        total_mb=1000.0,
        speed_mbps=0.0
    )
    await download_notifier.on_download_started(model_id, download_progress)
    
    # Simulate progress updates
    for progress in [15.5, 32.1, 48.7, 65.3, 82.9, 95.2]:
        download_progress.progress_percent = progress
        download_progress.downloaded_mb = progress * 10
        download_progress.speed_mbps = 12.5 + (progress * 0.1)  # Varying speed
        
        await download_notifier.on_download_progress(model_id, download_progress)
        await asyncio.sleep(0.5)  # Simulate time between updates
    
    # Simulate download completion
    download_result = DownloadResult(
        success=True,
        model_id=model_id,
        final_status=DownloadStatus.COMPLETED,
        total_time_seconds=120.5,
        total_retries=0,
        final_size_mb=1000.0,
        integrity_verified=True
    )
    await download_notifier.on_download_completed(model_id, download_result)
    
    logger.info(f"Download demo completed. Sent {len(mock_ws.messages)} notifications.")
    
    # Cleanup
    await connection_manager.disconnect("demo_user_1")


async def demo_health_monitoring():
    """Demonstrate health monitoring notifications"""
    logger.info("\n=== Health Monitoring Demo ===")
    
    # Get components
    connection_manager = get_connection_manager()
    integrator = get_model_notification_integrator()
    health_notifier = integrator.get_health_notifier()
    
    # Setup mock connection
    mock_ws = MockWebSocketConnection("demo_user_2")
    await connection_manager.connect(mock_ws, "demo_user_2")
    await connection_manager.subscribe("demo_user_2", "health_monitoring")
    
    model_id = "i2v-a14b"
    
    # Simulate healthy check
    logger.info("Simulating health check...")
    integrity_result = IntegrityResult(
        model_id=model_id,
        is_healthy=True,
        health_status=HealthStatus.HEALTHY,
        integrity_score=0.98,
        file_count=5,
        total_size_mb=1024.0
    )
    await health_notifier.on_health_check_completed(model_id, integrity_result)
    
    await asyncio.sleep(1)
    
    # Simulate corruption detection
    logger.info("Simulating corruption detection...")
    await health_notifier.on_corruption_detected(
        model_id=model_id,
        corruption_type="checksum_mismatch",
        severity="medium",
        repair_action="Re-download affected files"
    )
    
    await asyncio.sleep(1)
    
    # Simulate automatic repair
    logger.info("Simulating automatic repair...")
    await health_notifier.on_automatic_repair_started(model_id, "file_redownload")
    
    await asyncio.sleep(2)
    
    await health_notifier.on_automatic_repair_completed(
        model_id=model_id,
        repair_success=True,
        repair_details="Successfully re-downloaded 2 corrupted files"
    )
    
    logger.info(f"Health monitoring demo completed. Sent {len(mock_ws.messages)} notifications.")
    
    # Cleanup
    await connection_manager.disconnect("demo_user_2")


async def demo_fallback_notifications():
    """Demonstrate fallback strategy notifications"""
    logger.info("\n=== Fallback Notifications Demo ===")
    
    # Get components
    connection_manager = get_connection_manager()
    integrator = get_model_notification_integrator()
    fallback_notifier = integrator.get_fallback_notifier()
    
    # Setup mock connection
    mock_ws = MockWebSocketConnection("demo_user_3")
    await connection_manager.connect(mock_ws, "demo_user_3")
    await connection_manager.subscribe("demo_user_3", "fallback_notifications")
    
    original_model = "unavailable-model"
    
    # Simulate alternative model suggestion
    logger.info("Simulating alternative model suggestion...")
    suggestion = ModelSuggestion(
        suggested_model="backup-model",
        compatibility_score=0.85,
        performance_difference=-0.1,
        availability_status=ModelAvailabilityStatus.AVAILABLE,
        reason="Similar capabilities with slightly lower performance",
        estimated_quality_difference="slightly_lower"
    )
    await fallback_notifier.on_alternative_model_suggested(original_model, suggestion)
    
    await asyncio.sleep(1)
    
    # Simulate fallback strategy activation
    logger.info("Simulating fallback strategy activation...")
    fallback_strategy = FallbackStrategy(
        strategy_type=FallbackType.QUEUE_AND_WAIT,
        recommended_action="Queue request and wait for download",
        alternative_model=None,
        estimated_wait_time=timedelta(minutes=5),
        user_message="Model will be available in approximately 5 minutes",
        can_queue_request=True
    )
    await fallback_notifier.on_fallback_strategy_activated(original_model, fallback_strategy)
    
    await asyncio.sleep(1)
    
    # Simulate queue position update
    logger.info("Simulating queue position update...")
    await fallback_notifier.on_model_queued(
        model_id=original_model,
        queue_position=2,
        estimated_wait_time=timedelta(minutes=3)
    )
    
    logger.info(f"Fallback demo completed. Sent {len(mock_ws.messages)} notifications.")
    
    # Cleanup
    await connection_manager.disconnect("demo_user_3")


async def demo_model_status_updates():
    """Demonstrate model status update notifications"""
    logger.info("\n=== Model Status Updates Demo ===")
    
    # Get components
    connection_manager = get_connection_manager()
    integrator = get_model_notification_integrator()
    availability_notifier = integrator.get_availability_notifier()
    
    # Setup mock connection
    mock_ws = MockWebSocketConnection("demo_user_4")
    await connection_manager.connect(mock_ws, "demo_user_4")
    await connection_manager.subscribe("demo_user_4", "model_status")
    
    model_id = "ti2v-5b"
    
    # Simulate availability change
    logger.info("Simulating model availability change...")
    await availability_notifier.on_availability_changed(
        model_id=model_id,
        old_status=ModelAvailabilityStatus.MISSING,
        new_status=ModelAvailabilityStatus.DOWNLOADING,
        reason="User requested download"
    )
    
    await asyncio.sleep(1)
    
    # Simulate detailed status update
    logger.info("Simulating detailed status update...")
    detailed_status = DetailedModelStatus(
        model_id=model_id,
        model_type="ti2v",
        is_available=True,
        is_loaded=False,
        size_mb=2048.0,
        availability_status=ModelAvailabilityStatus.AVAILABLE,
        integrity_score=0.95,
        performance_score=0.88,
        usage_frequency=1.2,
        last_used=datetime.utcnow() - timedelta(hours=6),
        current_version="1.0.0",
        latest_version="1.1.0",
        update_available=True
    )
    await availability_notifier.on_model_status_update(model_id, detailed_status)
    
    await asyncio.sleep(1)
    
    # Simulate batch status update
    logger.info("Simulating batch status update...")
    models_status = {
        "t2v-a14b": DetailedModelStatus(
            model_id="t2v-a14b",
            model_type="t2v",
            is_available=True,
            is_loaded=True,
            size_mb=1024.0,
            availability_status=ModelAvailabilityStatus.AVAILABLE,
            integrity_score=0.98,
            performance_score=0.92,
            usage_frequency=5.2
        ),
        "i2v-a14b": DetailedModelStatus(
            model_id="i2v-a14b",
            model_type="i2v",
            is_available=False,
            is_loaded=False,
            size_mb=0.0,
            availability_status=ModelAvailabilityStatus.MISSING,
            integrity_score=0.0,
            performance_score=0.0,
            usage_frequency=0.0
        )
    }
    await availability_notifier.on_batch_status_update(models_status)
    
    logger.info(f"Model status demo completed. Sent {len(mock_ws.messages)} notifications.")
    
    # Cleanup
    await connection_manager.disconnect("demo_user_4")


async def demo_analytics_notifications():
    """Demonstrate analytics update notifications"""
    logger.info("\n=== Analytics Notifications Demo ===")
    
    # Get components
    connection_manager = get_connection_manager()
    integrator = get_model_notification_integrator()
    analytics_notifier = integrator.get_analytics_notifier()
    
    # Setup mock connection
    mock_ws = MockWebSocketConnection("demo_user_5")
    await connection_manager.connect(mock_ws, "demo_user_5")
    await connection_manager.subscribe("demo_user_5", "analytics_updates")
    
    # Simulate usage statistics update
    logger.info("Simulating usage statistics update...")
    usage_data = {
        "total_models": 5,
        "active_models": 3,
        "most_used_model": "t2v-a14b",
        "usage_trends": {
            "daily_usage": 25,
            "weekly_usage": 150,
            "monthly_usage": 600
        },
        "performance_metrics": {
            "average_generation_time": 28.3,
            "success_rate": 0.95
        }
    }
    await analytics_notifier.on_usage_statistics_updated(usage_data)
    
    await asyncio.sleep(1)
    
    # Simulate cleanup recommendation
    logger.info("Simulating cleanup recommendation...")
    cleanup_data = {
        "recommended_cleanup": ["old-model-v1", "unused-experimental"],
        "potential_space_saved_mb": 3072.0,
        "models_not_used_days": 45,
        "cleanup_priority": "medium",
        "cleanup_reasons": [
            "Not used in 45 days",
            "Superseded by newer versions"
        ]
    }
    await analytics_notifier.on_cleanup_recommendation_generated(cleanup_data)
    
    await asyncio.sleep(1)
    
    # Simulate performance analytics update
    logger.info("Simulating performance analytics update...")
    performance_data = {
        "average_download_speed": 18.5,
        "download_success_rate": 0.97,
        "model_load_times": {
            "t2v-a14b": 12.3,
            "i2v-a14b": 8.7
        },
        "peak_usage_hours": [14, 15, 16, 20, 21],
        "system_efficiency": 0.89
    }
    await analytics_notifier.on_performance_analytics_updated(performance_data)
    
    logger.info(f"Analytics demo completed. Sent {len(mock_ws.messages)} notifications.")
    
    # Cleanup
    await connection_manager.disconnect("demo_user_5")


async def main():
    """Run all WebSocket notification demos"""
    logger.info("🚀 Starting WebSocket Model Notifications Demo")
    logger.info("=" * 60)
    
    try:
        # Run all demo scenarios
        await demo_download_notifications()
        await demo_health_monitoring()
        await demo_fallback_notifications()
        await demo_model_status_updates()
        await demo_analytics_notifications()
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ All WebSocket notification demos completed successfully!")
        logger.info("\nKey Features Demonstrated:")
        logger.info("  📥 Real-time download progress with retry handling")
        logger.info("  🏥 Health monitoring with corruption detection and auto-repair")
        logger.info("  🔄 Intelligent fallback strategies with user interaction")
        logger.info("  📊 Comprehensive model status tracking")
        logger.info("  📈 Usage analytics and cleanup recommendations")
        
    except Exception as e:
        logger.error(f"❌ Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
