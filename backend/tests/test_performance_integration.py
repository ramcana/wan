"""
Integration tests for Performance Monitoring System with Enhanced Model Components

Tests the integration of performance monitoring with enhanced model downloader,
health monitor, fallback manager, and other components.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

from backend.core.performance_monitoring_system import (
    PerformanceMonitoringSystem,
    PerformanceMetricType,
    get_performance_monitor,
    initialize_performance_monitoring
)
from backend.core.enhanced_model_downloader import EnhancedModelDownloader, DownloadStatus
from backend.core.model_health_monitor import ModelHealthMonitor, HealthStatus
from backend.core.intelligent_fallback_manager import IntelligentFallbackManager, GenerationRequirements


class TestPerformanceIntegrationWithDownloader:
    """Test performance monitoring integration with enhanced model downloader"""
    
    def setup_method(self):
        self.performance_system = PerformanceMonitoringSystem()
        
        # Mock base downloader
        self.mock_base_downloader = Mock()
        self.mock_base_downloader.download_model = AsyncMock()
        
        self.downloader = EnhancedModelDownloader(
            base_downloader=self.mock_base_downloader,
            models_dir="test_models",
            max_retries=2
        )
    
    @pytest.mark.asyncio
    async def test_download_performance_tracking(self):
        """Test that download operations are properly tracked"""
        # Mock successful download
        self.mock_base_downloader.download_model.return_value = Mock(
            success=True,
            model_path="test_models/test-model.model"
        )
        
        # Patch the performance monitor to use our test instance
        with patch('backend.core.enhanced_model_downloader.get_performance_monitor', 
                  return_value=self.performance_system):
            
            # Perform download
            result = await self.downloader.download_with_retry(
                model_id="test-model",
                download_url="https://example.com/model.bin"
            )
        
        # Verify performance tracking
        download_metrics = self.performance_system.tracker.get_metrics_by_type(
            PerformanceMetricType.DOWNLOAD_OPERATION
        )
        
        assert len(download_metrics) == 1
        metric = download_metrics[0]
        assert metric.operation_name == "download_test-model"
        assert metric.success == result.success
        assert metric.duration_seconds is not None
        assert metric.metadata["model_id"] == "test-model"
        assert metric.metadata["download_url"] == "https://example.com/model.bin"
    
    @pytest.mark.asyncio
    async def test_download_retry_performance_tracking(self):
        """Test performance tracking during download retries"""
        # Mock failed then successful download
        self.mock_base_downloader.download_model.side_effect = [
            Exception("Network error"),  # First attempt fails
            Mock(success=True, model_path="test_models/test-model.model")  # Second succeeds
        ]
        
        with patch('backend.core.enhanced_model_downloader.get_performance_monitor', 
                  return_value=self.performance_system):
            
            result = await self.downloader.download_with_retry(
                model_id="retry-test-model",
                download_url="https://example.com/model.bin"
            )
        
        # Verify performance tracking includes retry information
        download_metrics = self.performance_system.tracker.get_metrics_by_type(
            PerformanceMetricType.DOWNLOAD_OPERATION
        )
        
        assert len(download_metrics) == 1
        metric = download_metrics[0]
        assert metric.success == result.success
        assert "total_retries" in metric.metadata
        assert metric.metadata["total_retries"] >= 1
    
    @pytest.mark.asyncio
    async def test_download_failure_performance_tracking(self):
        """Test performance tracking for failed downloads"""
        # Mock all download attempts failing
        self.mock_base_downloader.download_model.side_effect = Exception("Persistent network error")
        
        with patch('backend.core.enhanced_model_downloader.get_performance_monitor', 
                  return_value=self.performance_system):
            
            result = await self.downloader.download_with_retry(
                model_id="failed-model",
                download_url="https://example.com/model.bin"
            )
        
        # Verify failure is tracked
        download_metrics = self.performance_system.tracker.get_metrics_by_type(
            PerformanceMetricType.DOWNLOAD_OPERATION
        )
        
        assert len(download_metrics) == 1
        metric = download_metrics[0]
        assert metric.success == False
        assert metric.error_message is not None
        assert "failed" in metric.metadata.get("final_status", "").lower()


class TestPerformanceIntegrationWithHealthMonitor:
    """Test performance monitoring integration with model health monitor"""
    
    def setup_method(self):
        self.performance_system = PerformanceMonitoringSystem()
        self.health_monitor = ModelHealthMonitor(models_dir="test_models")
    
    @pytest.mark.asyncio
    async def test_health_check_performance_tracking(self):
        """Test that health checks are properly tracked"""
        with patch('backend.core.model_health_monitor.get_performance_monitor', 
                  return_value=self.performance_system):
            with patch.object(self.health_monitor, '_get_model_path') as mock_path:
                with patch.object(self.health_monitor, '_get_essential_files') as mock_files:
                    with patch.object(self.health_monitor, '_save_health_result') as mock_save:
                        with patch.object(self.health_monitor, '_notify_health_callbacks') as mock_notify:
                            
                            # Mock healthy model
                            mock_path.return_value = Mock(exists=Mock(return_value=True))
                            mock_files.return_value = ["config.json", "model_index.json"]
                            mock_save.return_value = None
                            mock_notify.return_value = None
                            
                            # Mock file existence checks
                            with patch('pathlib.Path.exists', return_value=True):
                                with patch.object(self.health_monitor, '_verify_file_checksums', return_value=[]):
                                    with patch.object(self.health_monitor, '_verify_file_formats', return_value=[]):
                                        with patch.object(self.health_monitor, '_calculate_model_checksum', return_value="abc123"):
                                            
                                            result = await self.health_monitor.check_model_integrity("test-model")
        
        # Verify performance tracking
        health_metrics = self.performance_system.tracker.get_metrics_by_type(
            PerformanceMetricType.HEALTH_CHECK
        )
        
        assert len(health_metrics) == 1
        metric = health_metrics[0]
        assert metric.operation_name == "integrity_check_test-model"
        assert metric.success == (result.health_status != HealthStatus.UNKNOWN)
        assert metric.metadata["model_id"] == "test-model"
        assert metric.metadata["check_type"] == "full_integrity"
    
    @pytest.mark.asyncio
    async def test_health_check_corruption_detection_tracking(self):
        """Test performance tracking when corruption is detected"""
        with patch('backend.core.model_health_monitor.get_performance_monitor', 
                  return_value=self.performance_system):
            with patch.object(self.health_monitor, '_get_model_path') as mock_path:
                with patch.object(self.health_monitor, '_get_essential_files') as mock_files:
                    with patch.object(self.health_monitor, '_save_health_result') as mock_save:
                        with patch.object(self.health_monitor, '_notify_health_callbacks') as mock_notify:
                            
                            # Mock corrupted model (missing files)
                            mock_path.return_value = Mock(exists=Mock(return_value=True))
                            mock_files.return_value = ["config.json", "model_index.json", "missing_file.bin"]
                            mock_save.return_value = None
                            mock_notify.return_value = None
                            
                            # Mock some files missing
                            def mock_file_exists(self):
                                return str(self).endswith("config.json") or str(self).endswith("model_index.json")
                            
                            with patch('pathlib.Path.exists', side_effect=mock_file_exists):
                                result = await self.health_monitor.check_model_integrity("corrupted-model")
        
        # Verify corruption is tracked in performance metrics
        health_metrics = self.performance_system.tracker.get_metrics_by_type(
            PerformanceMetricType.HEALTH_CHECK
        )
        
        assert len(health_metrics) == 1
        metric = health_metrics[0]
        assert metric.operation_name == "integrity_check_corrupted-model"
        assert "corruption_types" in metric.metadata
        assert result.health_status == HealthStatus.CORRUPTED


class TestPerformanceIntegrationWithFallbackManager:
    """Test performance monitoring integration with intelligent fallback manager"""
    
    def setup_method(self):
        self.performance_system = PerformanceMonitoringSystem()
        
        # Mock availability manager
        self.mock_availability_manager = Mock()
        self.mock_availability_manager.get_comprehensive_model_status = AsyncMock()
        self.mock_availability_manager._check_single_model_availability = AsyncMock()
        
        self.fallback_manager = IntelligentFallbackManager(
            availability_manager=self.mock_availability_manager
        )
    
    @pytest.mark.asyncio
    async def test_fallback_suggestion_performance_tracking(self):
        """Test that fallback suggestions are properly tracked"""
        # Mock available models
        self.mock_availability_manager.get_comprehensive_model_status.return_value = {
            "alternative-model": Mock(availability_status=Mock(value="available")),
            "another-model": Mock(availability_status=Mock(value="available"))
        }
        
        with patch('backend.core.intelligent_fallback_manager.get_performance_monitor', 
                  return_value=self.performance_system):
            
            requirements = GenerationRequirements(
                quality="high",
                speed="medium",
                resolution="1024x1024"
            )
            
            suggestion = await self.fallback_manager.suggest_alternative_model(
                requested_model="unavailable-model",
                requirements=requirements
            )
        
        # Verify performance tracking
        fallback_metrics = self.performance_system.tracker.get_metrics_by_type(
            PerformanceMetricType.FALLBACK_STRATEGY
        )
        
        assert len(fallback_metrics) == 1
        metric = fallback_metrics[0]
        assert metric.operation_name == "suggest_alternative_unavailable-model"
        assert metric.success == True
        assert metric.metadata["requested_model"] == "unavailable-model"
        assert metric.metadata["quality_requirement"] == "high"
        assert "suggested_model" in metric.metadata
    
    @pytest.mark.asyncio
    async def test_fallback_cache_hit_performance_tracking(self):
        """Test performance tracking for cached fallback suggestions"""
        # Pre-populate cache
        requirements = GenerationRequirements(
            quality="medium",
            speed="fast",
            resolution="512x512"
        )
        
        # First call to populate cache
        with patch('backend.core.intelligent_fallback_manager.get_performance_monitor', 
                  return_value=self.performance_system):
            
            self.mock_availability_manager.get_comprehensive_model_status.return_value = {
                "fast-model": Mock(availability_status=Mock(value="available"))
            }
            
            await self.fallback_manager.suggest_alternative_model(
                requested_model="slow-model",
                requirements=requirements
            )
            
            # Second call should hit cache
            suggestion = await self.fallback_manager.suggest_alternative_model(
                requested_model="slow-model",
                requirements=requirements
            )
        
        # Verify both calls are tracked
        fallback_metrics = self.performance_system.tracker.get_metrics_by_type(
            PerformanceMetricType.FALLBACK_STRATEGY
        )
        
        assert len(fallback_metrics) == 2
        
        # Second call should be marked as cache hit
        cache_hit_metric = fallback_metrics[1]
        assert cache_hit_metric.metadata.get("cache_hit") == True


class TestPerformanceSystemIntegration:
    """Test overall performance monitoring system integration"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_performance_monitoring(self):
        """Test complete end-to-end performance monitoring workflow"""
        # Initialize performance monitoring
        await initialize_performance_monitoring()
        monitor = get_performance_monitor()
        
        # Simulate a complete model request workflow
        
        # 1. Start with a model request that requires fallback
        fallback_id = monitor.track_fallback_strategy(
            "handle_unavailable_model",
            {"requested_model": "unavailable-model", "user_id": "test_user"}
        )
        
        await asyncio.sleep(0.01)  # Simulate processing time
        
        monitor.end_tracking(
            fallback_id,
            success=True,
            additional_metadata={
                "suggested_model": "alternative-model",
                "fallback_strategy": "alternative_model"
            }
        )
        
        # 2. Download the alternative model
        download_id = monitor.track_download_operation(
            "download_alternative_model",
            {"model_id": "alternative-model", "size_mb": 1500}
        )
        
        await asyncio.sleep(0.02)  # Simulate download time
        
        monitor.end_tracking(
            download_id,
            success=True,
            additional_metadata={
                "downloaded_mb": 1500,
                "average_speed_mbps": 75,
                "integrity_verified": True
            }
        )
        
        # 3. Perform health check on downloaded model
        health_id = monitor.track_health_check(
            "verify_downloaded_model",
            {"model_id": "alternative-model", "check_type": "post_download"}
        )
        
        await asyncio.sleep(0.005)  # Simulate health check time
        
        monitor.end_tracking(
            health_id,
            success=True,
            additional_metadata={
                "health_status": "healthy",
                "files_checked": 12,
                "integrity_score": 1.0
            }
        )
        
        # 4. Track model operation (loading/generation)
        model_op_id = monitor.track_model_operation(
            "load_and_generate",
            {"model_id": "alternative-model", "operation": "text_to_video"}
        )
        
        await asyncio.sleep(0.03)  # Simulate generation time
        
        monitor.end_tracking(
            model_op_id,
            success=True,
            additional_metadata={
                "generation_time_seconds": 45.2,
                "output_quality": "high",
                "vram_used_mb": 8192
            }
        )
        
        # Generate performance report
        report = monitor.get_performance_report(hours_back=1)
        
        # Verify comprehensive tracking
        assert report.total_operations == 4
        assert report.success_rate == 1.0
        assert len(report.operations_by_type) == 4
        
        # Verify all operation types are present
        assert PerformanceMetricType.FALLBACK_STRATEGY.value in report.operations_by_type
        assert PerformanceMetricType.DOWNLOAD_OPERATION.value in report.operations_by_type
        assert PerformanceMetricType.HEALTH_CHECK.value in report.operations_by_type
        assert PerformanceMetricType.MODEL_OPERATION.value in report.operations_by_type
        
        # Verify dashboard data includes all metrics
        dashboard_data = monitor.get_dashboard_data()
        assert dashboard_data["performance_summary"]["total_operations_24h"] == 4
        assert dashboard_data["performance_summary"]["success_rate"] == 1.0
        assert dashboard_data["recent_activity"] == 4  # All operations in last hour
    
    @pytest.mark.asyncio
    async def test_performance_monitoring_with_failures(self):
        """Test performance monitoring handles failures gracefully"""
        monitor = PerformanceMonitoringSystem()
        
        # Simulate various failure scenarios
        
        # 1. Failed download
        failed_download_id = monitor.track_download_operation(
            "failed_download",
            {"model_id": "problematic-model"}
        )
        
        await asyncio.sleep(0.01)
        
        monitor.end_tracking(
            failed_download_id,
            success=False,
            error_message="Network timeout after 3 retries"
        )
        
        # 2. Health check finds corruption
        corruption_check_id = monitor.track_health_check(
            "corruption_detection",
            {"model_id": "corrupted-model"}
        )
        
        await asyncio.sleep(0.005)
        
        monitor.end_tracking(
            corruption_check_id,
            success=False,
            error_message="Multiple files corrupted",
            additional_metadata={
                "corruption_types": ["checksum_mismatch", "file_missing"],
                "files_affected": 3
            }
        )
        
        # 3. Fallback strategy fails to find alternatives
        no_alternatives_id = monitor.track_fallback_strategy(
            "no_alternatives_available",
            {"requested_model": "unique-model"}
        )
        
        await asyncio.sleep(0.002)
        
        monitor.end_tracking(
            no_alternatives_id,
            success=False,
            error_message="No compatible alternatives found"
        )
        
        # Generate report and verify failure handling
        report = monitor.get_performance_report(hours_back=1)
        
        assert report.total_operations == 3
        assert report.success_rate == 0.0  # All operations failed
        assert len(report.bottlenecks_identified) > 0  # Should identify high failure rate
        assert len(report.optimization_recommendations) > 0
        
        # Verify failure analysis
        assert "High operation failure rate detected" in report.bottlenecks_identified
    
    def test_performance_monitoring_resource_efficiency(self):
        """Test that performance monitoring doesn't significantly impact system resources"""
        monitor = PerformanceMonitoringSystem()
        
        # Measure baseline resource usage
        import psutil
        process = psutil.Process()
        baseline_memory = process.memory_info().rss
        baseline_cpu_time = process.cpu_times().user
        
        # Perform many operations
        operation_count = 1000
        start_time = time.time()
        
        for i in range(operation_count):
            op_id = monitor.track_model_operation(f"efficiency_test_{i}")
            monitor.end_tracking(op_id, success=True)
        
        end_time = time.time()
        
        # Measure resource usage after operations
        final_memory = process.memory_info().rss
        final_cpu_time = process.cpu_times().user
        
        # Calculate overhead
        memory_overhead_mb = (final_memory - baseline_memory) / (1024 * 1024)
        cpu_overhead = final_cpu_time - baseline_cpu_time
        total_time = end_time - start_time
        
        # Verify efficiency
        assert memory_overhead_mb < 50, f"Memory overhead too high: {memory_overhead_mb:.2f}MB"
        assert cpu_overhead < 1.0, f"CPU overhead too high: {cpu_overhead:.3f}s"
        assert total_time < 15.0, f"Operations too slow: {total_time:.3f}s for {operation_count} operations"
        
        print(f"Performance monitoring efficiency:")
        print(f"  Memory overhead: {memory_overhead_mb:.2f}MB")
        print(f"  CPU overhead: {cpu_overhead:.3f}s")
        print(f"  Total time: {total_time:.3f}s ({operation_count/total_time:.1f} ops/sec)")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
