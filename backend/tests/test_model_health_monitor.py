"""
Unit tests for Model Health Monitor
Tests integrity checking, performance monitoring, corruption detection,
and automated health checks.
"""

import asyncio
import json
import pytest
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import hashlib

# Import the module under test
import sys
sys.path.append(str(Path(__file__).parent.parent))

from backend.core.model_health_monitor import (
    ModelHealthMonitor,
    HealthStatus,
    CorruptionType,
    IntegrityResult,
    PerformanceMetrics,
    PerformanceHealth,
    CorruptionReport,
    SystemHealthReport,
    HealthCheckConfig,
    get_model_health_monitor
)


class TestModelHealthMonitor:
    """Test suite for ModelHealthMonitor"""
    
    @pytest.fixture
    async def temp_models_dir(self):
        """Create temporary models directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    async def health_monitor(self, temp_models_dir):
        """Create ModelHealthMonitor instance for testing"""
        config = HealthCheckConfig(
            check_interval_hours=1,
            automatic_repair_enabled=True
        )
        monitor = ModelHealthMonitor(str(temp_models_dir), config)
        yield monitor
        await monitor.cleanup()
    
    @pytest.fixture
    async def sample_model(self, temp_models_dir):
        """Create a sample model directory structure"""
        model_id = "test-model"
        model_path = temp_models_dir / model_id
        model_path.mkdir(parents=True)
        
        # Create essential files
        config_data = {"model_type": "text-to-video", "version": "1.0"}
        (model_path / "config.json").write_text(json.dumps(config_data))
        
        model_index_data = {"_class_name": "TestPipeline"}
        (model_path / "model_index.json").write_text(json.dumps(model_index_data))
        
        # Create subdirectories with configs
        for subdir in ["unet", "text_encoder", "vae"]:
            subdir_path = model_path / subdir
            subdir_path.mkdir()
            (subdir_path / "config.json").write_text(json.dumps({"test": True}))
        
        # Create a model file
        model_file = model_path / "pytorch_model.bin"
        model_file.write_bytes(b"fake model data" * 1000)  # Create some content
        
        return model_id, model_path
    
    @pytest.fixture
    async def corrupted_model(self, temp_models_dir):
        """Create a corrupted model for testing"""
        model_id = "corrupted-model"
        model_path = temp_models_dir / model_id
        model_path.mkdir(parents=True)
        
        # Create config.json but missing model_index.json
        config_data = {"model_type": "text-to-video"}
        (model_path / "config.json").write_text(json.dumps(config_data))
        
        # Create empty file (corruption)
        (model_path / "empty_file.bin").write_bytes(b"")
        
        # Create invalid JSON
        (model_path / "invalid.json").write_text("invalid json content {")
        
        return model_id, model_path


class TestIntegrityChecking:
    """Test integrity checking functionality"""
    
    @pytest.mark.asyncio
    async def test_healthy_model_integrity(self, health_monitor, sample_model):
        """Test integrity check on healthy model"""
        model_id, model_path = sample_model
        
        result = await health_monitor.check_model_integrity(model_id)
        
        assert isinstance(result, IntegrityResult)
        assert result.model_id == model_id
        assert result.is_healthy is True
        assert result.health_status == HealthStatus.HEALTHY
        assert len(result.issues) == 0
        assert result.file_count > 0
        assert result.total_size_mb > 0
        assert result.checksum is not None
        assert result.last_checked is not None
    
    @pytest.mark.asyncio
    async def test_missing_model_integrity(self, health_monitor):
        """Test integrity check on missing model"""
        result = await health_monitor.check_model_integrity("nonexistent-model")
        
        assert result.model_id == "nonexistent-model"
        assert result.is_healthy is False
        assert result.health_status == HealthStatus.MISSING
        assert "Model directory not found" in result.issues
        assert "Re-download the model" in result.repair_suggestions
    
    @pytest.mark.asyncio
    async def test_corrupted_model_integrity(self, health_monitor, corrupted_model):
        """Test integrity check on corrupted model"""
        model_id, model_path = corrupted_model
        
        result = await health_monitor.check_model_integrity(model_id)
        
        assert result.model_id == model_id
        assert result.is_healthy is False
        assert result.health_status in [HealthStatus.CORRUPTED, HealthStatus.DEGRADED]
        assert len(result.issues) > 0
        assert CorruptionType.FILE_MISSING in result.corruption_types
        assert len(result.repair_suggestions) > 0
    
    @pytest.mark.asyncio
    async def test_checksum_calculation(self, health_monitor, sample_model):
        """Test checksum calculation consistency"""
        model_id, model_path = sample_model
        
        # Calculate checksum twice
        result1 = await health_monitor.check_model_integrity(model_id)
        result2 = await health_monitor.check_model_integrity(model_id)
        
        assert result1.checksum == result2.checksum
        assert result1.checksum != "unknown"
    
    @pytest.mark.asyncio
    async def test_essential_files_detection(self, health_monitor, temp_models_dir):
        """Test detection of missing essential files"""
        model_id = "incomplete-model"
        model_path = temp_models_dir / model_id
        model_path.mkdir(parents=True)
        
        # Only create config.json, missing model_index.json
        config_data = {"model_type": "text-to-video"}
        (model_path / "config.json").write_text(json.dumps(config_data))
        
        result = await health_monitor.check_model_integrity(model_id)
        
        assert result.is_healthy is False
        assert any("model_index.json" in issue for issue in result.issues)
        assert CorruptionType.FILE_MISSING in result.corruption_types


class TestPerformanceMonitoring:
    """Test performance monitoring functionality"""
    
    @pytest.mark.asyncio
    async def test_performance_monitoring_basic(self, health_monitor):
        """Test basic performance monitoring"""
        model_id = "test-model"
        generation_metrics = {
            "load_time_seconds": 5.0,
            "generation_time_seconds": 30.0,
            "memory_usage_mb": 2048.0,
            "vram_usage_mb": 8192.0,
            "cpu_usage_percent": 45.0,
            "throughput_fps": 2.5,
            "quality_score": 0.85,
            "error_rate": 0.0
        }
        
        health = await health_monitor.monitor_model_performance(model_id, generation_metrics)
        
        assert isinstance(health, PerformanceHealth)
        assert health.model_id == model_id
        assert 0.0 <= health.overall_score <= 1.0
        assert health.performance_trend in ["improving", "stable", "degrading", "unknown"]
        assert health.last_assessment is not None
    
    @pytest.mark.asyncio
    async def test_performance_bottleneck_detection(self, health_monitor):
        """Test detection of performance bottlenecks"""
        model_id = "slow-model"
        generation_metrics = {
            "load_time_seconds": 10.0,
            "generation_time_seconds": 400.0,  # Very slow (> 5 minutes)
            "memory_usage_mb": 4096.0,
            "vram_usage_mb": 15000.0,  # High VRAM usage
            "cpu_usage_percent": 95.0,
            "throughput_fps": 0.5,
            "error_rate": 0.15  # High error rate
        }
        
        health = await health_monitor.monitor_model_performance(model_id, generation_metrics)
        
        assert health.overall_score < 0.8  # Should be degraded
        assert len(health.bottlenecks) > 0
        assert len(health.recommendations) > 0
        
        # Check for specific bottleneck detection
        bottleneck_text = " ".join(health.bottlenecks).lower()
        assert any(keyword in bottleneck_text for keyword in ["slow", "high", "error"])
    
    @pytest.mark.asyncio
    async def test_performance_trend_analysis(self, health_monitor):
        """Test performance trend analysis over time"""
        model_id = "trending-model"
        
        # Simulate degrading performance over time
        base_time = 30.0
        for i in range(15):
            generation_metrics = {
                "load_time_seconds": 5.0,
                "generation_time_seconds": base_time + (i * 2.0),  # Increasing time
                "memory_usage_mb": 2048.0,
                "vram_usage_mb": 8192.0,
                "cpu_usage_percent": 45.0,
                "throughput_fps": 2.5 - (i * 0.1),
                "error_rate": 0.0
            }
            
            await health_monitor.monitor_model_performance(model_id, generation_metrics)
        
        # Get final assessment
        final_metrics = {
            "load_time_seconds": 5.0,
            "generation_time_seconds": base_time + 30.0,
            "memory_usage_mb": 2048.0,
            "vram_usage_mb": 8192.0,
            "cpu_usage_percent": 45.0,
            "throughput_fps": 1.0,
            "error_rate": 0.0
        }
        
        health = await health_monitor.monitor_model_performance(model_id, final_metrics)
        
        # Should detect degrading trend
        assert health.performance_trend == "degrading"
    
    @pytest.mark.asyncio
    async def test_baseline_comparison(self, health_monitor):
        """Test performance comparison with baseline"""
        model_id = "baseline-model"
        
        # Establish baseline with multiple measurements
        baseline_metrics = {
            "load_time_seconds": 5.0,
            "generation_time_seconds": 30.0,
            "memory_usage_mb": 2048.0,
            "vram_usage_mb": 8192.0,
            "cpu_usage_percent": 45.0,
            "throughput_fps": 2.5,
            "error_rate": 0.0
        }
        
        # Add baseline measurements
        for _ in range(10):
            await health_monitor.monitor_model_performance(model_id, baseline_metrics)
        
        # Test with degraded performance
        degraded_metrics = baseline_metrics.copy()
        degraded_metrics["generation_time_seconds"] = 45.0  # 50% slower
        degraded_metrics["vram_usage_mb"] = 12000.0  # 46% more memory
        
        health = await health_monitor.monitor_model_performance(model_id, degraded_metrics)
        
        assert len(health.baseline_comparison) > 0
        assert "generation_time_ratio" in health.baseline_comparison
        assert health.baseline_comparison["generation_time_ratio"] > 1.2  # 20% slower threshold


class TestCorruptionDetection:
    """Test corruption detection functionality"""
    
    @pytest.mark.asyncio
    async def test_corruption_detection_clean_model(self, health_monitor, sample_model):
        """Test corruption detection on clean model"""
        model_id, model_path = sample_model
        
        report = await health_monitor.detect_corruption(model_id)
        
        assert isinstance(report, CorruptionReport)
        assert report.model_id == model_id
        assert report.corruption_detected is False
        assert len(report.corruption_types) == 0
        assert report.detection_time is not None
    
    @pytest.mark.asyncio
    async def test_corruption_detection_corrupted_model(self, health_monitor, corrupted_model):
        """Test corruption detection on corrupted model"""
        model_id, model_path = corrupted_model
        
        report = await health_monitor.detect_corruption(model_id)
        
        assert report.model_id == model_id
        assert report.corruption_detected is True
        assert len(report.corruption_types) > 0
        assert len(report.affected_files) > 0
        assert report.severity in ["low", "medium", "high", "critical"]
        assert len(report.repair_actions) > 0
    
    @pytest.mark.asyncio
    async def test_incomplete_download_detection(self, health_monitor, temp_models_dir):
        """Test detection of incomplete downloads"""
        model_id = "incomplete-model"
        model_path = temp_models_dir / model_id
        model_path.mkdir(parents=True)
        
        # Create temporary/partial files
        (model_path / "model.tmp").write_bytes(b"temporary data")
        (model_path / "download.partial").write_bytes(b"partial data")
        
        # Create zero-byte file
        (model_path / "empty.bin").write_bytes(b"")
        
        report = await health_monitor.detect_corruption(model_id)
        
        assert report.corruption_detected is True
        assert CorruptionType.INCOMPLETE_DOWNLOAD in report.corruption_types
        assert any("tmp" in f or "partial" in f for f in report.affected_files)
    
    @pytest.mark.asyncio
    async def test_corruption_severity_assessment(self, health_monitor, temp_models_dir):
        """Test corruption severity assessment"""
        # Test critical severity (missing essential files)
        model_id = "critical-model"
        model_path = temp_models_dir / model_id
        model_path.mkdir(parents=True)
        # No essential files created
        
        report = await health_monitor.detect_corruption(model_id)
        assert report.severity == "critical"
        
        # Test medium severity (format issues)
        model_id2 = "medium-model"
        model_path2 = temp_models_dir / model_id2
        model_path2.mkdir(parents=True)
        
        # Create essential files but with format issues
        (model_path2 / "config.json").write_text("invalid json {")
        (model_path2 / "model_index.json").write_text(json.dumps({"test": True}))
        
        report2 = await health_monitor.detect_corruption(model_id2)
        assert report2.corruption_detected is True


class TestHealthChecks:
    """Test scheduled health checks functionality"""
    
    @pytest.mark.asyncio
    async def test_schedule_health_checks(self, health_monitor, sample_model):
        """Test scheduling of health checks"""
        model_id, model_path = sample_model
        
        # Start health checks with very short interval for testing
        health_monitor.config.check_interval_hours = 0.001  # ~3.6 seconds
        
        await health_monitor.schedule_health_checks()
        assert health_monitor._monitoring_active is True
        assert health_monitor._health_check_task is not None
        
        # Wait a bit to ensure at least one check runs
        await asyncio.sleep(0.1)
        
        await health_monitor.stop_health_checks()
        assert health_monitor._monitoring_active is False
    
    @pytest.mark.asyncio
    async def test_automatic_repair_trigger(self, health_monitor, temp_models_dir):
        """Test automatic repair triggering"""
        model_id = "repair-model"
        model_path = temp_models_dir / model_id
        model_path.mkdir(parents=True)
        
        # Create model with permission issues (simulated)
        (model_path / "config.json").write_text(json.dumps({"test": True}))
        (model_path / "temp.tmp").write_bytes(b"temp data")
        
        # Enable automatic repair
        health_monitor.config.automatic_repair_enabled = True
        
        # Manually trigger repair test
        corruption_report = await health_monitor.detect_corruption(model_id)
        
        if corruption_report.corruption_detected and corruption_report.repair_possible:
            await health_monitor._attempt_automatic_repair(model_id, corruption_report)
            
            # Verify repair attempt
            assert not (model_path / "temp.tmp").exists()  # Should be cleaned up
    
    @pytest.mark.asyncio
    async def test_health_callbacks(self, health_monitor, sample_model):
        """Test health check callbacks"""
        model_id, model_path = sample_model
        
        callback_called = False
        callback_result = None
        
        def health_callback(result):
            nonlocal callback_called, callback_result
            callback_called = True
            callback_result = result
        
        health_monitor.add_health_callback(health_callback)
        
        await health_monitor.check_model_integrity(model_id)
        
        assert callback_called is True
        assert callback_result is not None
        assert callback_result.model_id == model_id
    
    @pytest.mark.asyncio
    async def test_corruption_callbacks(self, health_monitor, corrupted_model):
        """Test corruption detection callbacks"""
        model_id, model_path = corrupted_model
        
        callback_called = False
        callback_report = None
        
        def corruption_callback(report):
            nonlocal callback_called, callback_report
            callback_called = True
            callback_report = report
        
        health_monitor.add_corruption_callback(corruption_callback)
        
        await health_monitor.detect_corruption(model_id)
        
        assert callback_called is True
        assert callback_report is not None
        assert callback_report.model_id == model_id


class TestSystemHealthReport:
    """Test system health reporting functionality"""
    
    @pytest.mark.asyncio
    async def test_system_health_report_empty(self, health_monitor):
        """Test system health report with no models"""
        report = await health_monitor.get_health_report()
        
        assert isinstance(report, SystemHealthReport)
        assert report.models_healthy == 0
        assert report.models_degraded == 0
        assert report.models_corrupted == 0
        assert report.models_missing == 0
        assert "No models found" in report.recommendations
        assert report.last_updated is not None
    
    @pytest.mark.asyncio
    async def test_system_health_report_mixed_models(self, health_monitor, temp_models_dir):
        """Test system health report with mixed model states"""
        # Create healthy model
        healthy_path = temp_models_dir / "healthy-model"
        healthy_path.mkdir(parents=True)
        (healthy_path / "config.json").write_text(json.dumps({"test": True}))
        (healthy_path / "model_index.json").write_text(json.dumps({"test": True}))
        
        # Create corrupted model
        corrupted_path = temp_models_dir / "corrupted-model"
        corrupted_path.mkdir(parents=True)
        (corrupted_path / "config.json").write_text(json.dumps({"test": True}))
        # Missing model_index.json
        
        report = await health_monitor.get_health_report()
        
        assert report.models_healthy >= 1
        assert report.models_corrupted >= 1
        assert len(report.detailed_reports) >= 2
        assert 0.0 <= report.overall_health_score <= 1.0
        
        # Check that recommendations are provided
        if report.models_corrupted > 0:
            assert any("corrupted" in rec.lower() for rec in report.recommendations)
    
    @pytest.mark.asyncio
    async def test_storage_usage_monitoring(self, health_monitor):
        """Test storage usage monitoring in health report"""
        with patch('psutil.disk_usage') as mock_disk_usage:
            # Mock high storage usage
            mock_disk_usage.return_value = type('DiskUsage', (), {
                'total': 1000 * 1024 * 1024 * 1024,  # 1TB
                'used': 950 * 1024 * 1024 * 1024,    # 950GB (95%)
                'free': 50 * 1024 * 1024 * 1024      # 50GB
            })()
            
            report = await health_monitor.get_health_report()
            
            assert report.storage_usage_percent > 90
            assert any("storage" in rec.lower() for rec in report.recommendations)


class TestUtilityFunctions:
    """Test utility and convenience functions"""
    
    @pytest.mark.asyncio
    async def test_global_health_monitor_instance(self, temp_models_dir):
        """Test global health monitor instance"""
        monitor1 = get_model_health_monitor(str(temp_models_dir))
        monitor2 = get_model_health_monitor(str(temp_models_dir))
        
        # Should return the same instance
        assert monitor1 is monitor2
    
    @pytest.mark.asyncio
    async def test_convenience_functions(self, temp_models_dir, sample_model):
        """Test convenience functions"""
        from backend.core.model_health_monitor import (
            check_model_integrity,
            monitor_model_performance,
            detect_model_corruption,
            get_system_health_report
        )
        
        model_id, model_path = sample_model
        
        # Test convenience functions
        integrity_result = await check_model_integrity(model_id)
        assert integrity_result.model_id == model_id
        
        performance_metrics = {
            "load_time_seconds": 5.0,
            "generation_time_seconds": 30.0,
            "memory_usage_mb": 2048.0,
            "vram_usage_mb": 8192.0,
            "cpu_usage_percent": 45.0,
            "throughput_fps": 2.5
        }
        
        performance_health = await monitor_model_performance(model_id, performance_metrics)
        assert performance_health.model_id == model_id
        
        corruption_report = await detect_model_corruption(model_id)
        assert corruption_report.model_id == model_id
        
        system_report = await get_system_health_report()
        assert isinstance(system_report, SystemHealthReport)


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    @pytest.mark.asyncio
    async def test_integrity_check_with_permission_error(self, health_monitor, temp_models_dir):
        """Test integrity check with permission errors"""
        model_id = "permission-model"
        model_path = temp_models_dir / model_id
        model_path.mkdir(parents=True)
        
        # Create a file and simulate permission error
        test_file = model_path / "config.json"
        test_file.write_text(json.dumps({"test": True}))
        
        with patch('pathlib.Path.stat', side_effect=PermissionError("Access denied")):
            result = await health_monitor.check_model_integrity(model_id)
            
            # Should handle the error gracefully
            assert result.model_id == model_id
            assert len(result.issues) > 0
    
    @pytest.mark.asyncio
    async def test_performance_monitoring_with_invalid_metrics(self, health_monitor):
        """Test performance monitoring with invalid metrics"""
        model_id = "invalid-metrics-model"
        
        # Test with missing metrics
        incomplete_metrics = {
            "load_time_seconds": 5.0
            # Missing other required metrics
        }
        
        health = await health_monitor.monitor_model_performance(model_id, incomplete_metrics)
        
        # Should handle gracefully with defaults
        assert health.model_id == model_id
        assert 0.0 <= health.overall_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_corruption_detection_with_io_error(self, health_monitor, temp_models_dir):
        """Test corruption detection with I/O errors"""
        model_id = "io-error-model"
        model_path = temp_models_dir / model_id
        model_path.mkdir(parents=True)
        
        with patch('aiofiles.open', side_effect=IOError("Disk error")):
            report = await health_monitor.detect_corruption(model_id)
            
            # Should handle the error and still provide a report
            assert report.model_id == model_id
            # May or may not detect corruption depending on what fails
    
    @pytest.mark.asyncio
    async def test_health_monitor_cleanup(self, health_monitor):
        """Test health monitor cleanup"""
        # Start monitoring
        await health_monitor.schedule_health_checks()
        assert health_monitor._monitoring_active is True
        
        # Cleanup should stop monitoring
        await health_monitor.cleanup()
        assert health_monitor._monitoring_active is False
        assert health_monitor._health_check_task is None or health_monitor._health_check_task.cancelled()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])