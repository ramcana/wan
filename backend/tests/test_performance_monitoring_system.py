"""
Comprehensive tests for the Performance Monitoring System

Tests performance tracking, resource monitoring, analysis, and optimization
validation for the enhanced model availability system.
"""

import pytest
import asyncio
import time
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
import tempfile

from backend.core.performance_monitoring_system import (
    PerformanceMonitoringSystem,
    PerformanceTracker,
    SystemResourceMonitor,
    PerformanceAnalyzer,
    PerformanceMetricType,
    PerformanceMetric,
    SystemResourceSnapshot,
    PerformanceReport
)


class TestPerformanceTracker:
    """Test performance tracking functionality"""
    
    def setup_method(self):
        self.tracker = PerformanceTracker()
    
    def test_start_operation_tracking(self):
        """Test starting operation tracking"""
        operation_id = self.tracker.start_operation(
            PerformanceMetricType.DOWNLOAD_OPERATION,
            "test_download",
            {"model_id": "test-model"}
        )
        
        assert operation_id in self.tracker.active_operations
        metric = self.tracker.active_operations[operation_id]
        assert metric.metric_type == PerformanceMetricType.DOWNLOAD_OPERATION
        assert metric.operation_name == "test_download"
        assert metric.metadata["model_id"] == "test-model"
        assert metric.start_time is not None
        assert metric.end_time is None
    
    def test_end_operation_tracking(self):
        """Test ending operation tracking"""
        operation_id = self.tracker.start_operation(
            PerformanceMetricType.HEALTH_CHECK,
            "test_health_check"
        )
        
        # Simulate some work
        time.sleep(0.1)
        
        metric = self.tracker.end_operation(
            operation_id,
            success=True,
            additional_metadata={"files_checked": 5}
        )
        
        assert metric is not None
        assert metric.success is True
        assert metric.end_time is not None
        assert metric.duration_seconds is not None
        assert metric.duration_seconds > 0.05  # At least 50ms
        assert metric.metadata["files_checked"] == 5
        assert operation_id not in self.tracker.active_operations
        assert metric in self.tracker.completed_metrics
    
    def test_end_nonexistent_operation(self):
        """Test ending operation that doesn't exist"""
        metric = self.tracker.end_operation("nonexistent_id")
        assert metric is None
    
    def test_get_metrics_by_type(self):
        """Test filtering metrics by type"""
        # Add different types of metrics
        download_id = self.tracker.start_operation(
            PerformanceMetricType.DOWNLOAD_OPERATION,
            "download_test"
        )
        health_id = self.tracker.start_operation(
            PerformanceMetricType.HEALTH_CHECK,
            "health_test"
        )
        
        self.tracker.end_operation(download_id, success=True)
        self.tracker.end_operation(health_id, success=True)
        
        download_metrics = self.tracker.get_metrics_by_type(
            PerformanceMetricType.DOWNLOAD_OPERATION
        )
        health_metrics = self.tracker.get_metrics_by_type(
            PerformanceMetricType.HEALTH_CHECK
        )
        
        assert len(download_metrics) == 1
        assert len(health_metrics) == 1
        assert download_metrics[0].operation_name == "download_test"
        assert health_metrics[0].operation_name == "health_test"
    
    def test_metrics_time_filtering(self):
        """Test filtering metrics by time window"""
        # Create old metric (simulate by modifying timestamp)
        old_id = self.tracker.start_operation(
            PerformanceMetricType.DOWNLOAD_OPERATION,
            "old_download"
        )
        old_metric = self.tracker.end_operation(old_id, success=True)
        
        # Modify timestamp to be 25 hours ago
        old_metric.start_time = datetime.now() - timedelta(hours=25)
        old_metric.end_time = old_metric.start_time + timedelta(seconds=10)
        
        # Create recent metric
        recent_id = self.tracker.start_operation(
            PerformanceMetricType.DOWNLOAD_OPERATION,
            "recent_download"
        )
        self.tracker.end_operation(recent_id, success=True)
        
        # Get metrics from last 24 hours
        recent_metrics = self.tracker.get_metrics_by_type(
            PerformanceMetricType.DOWNLOAD_OPERATION,
            hours_back=24
        )
        
        assert len(recent_metrics) == 1
        assert recent_metrics[0].operation_name == "recent_download"


class TestSystemResourceMonitor:
    """Test system resource monitoring"""
    
    def setup_method(self):
        self.monitor = SystemResourceMonitor(sample_interval=1)  # 1 second for testing
    
    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self):
        """Test starting and stopping resource monitoring"""
        assert not self.monitor._monitoring
        
        await self.monitor.start_monitoring()
        assert self.monitor._monitoring
        assert self.monitor._monitor_task is not None
        
        # Let it run briefly
        await asyncio.sleep(0.1)
        
        await self.monitor.stop_monitoring()
        assert not self.monitor._monitoring
    
    def test_capture_snapshot(self):
        """Test capturing resource snapshot"""
        snapshot = self.monitor._capture_snapshot()
        
        assert isinstance(snapshot, SystemResourceSnapshot)
        assert snapshot.timestamp is not None
        assert snapshot.cpu_percent >= 0
        assert snapshot.memory_percent >= 0
        assert snapshot.memory_used_mb >= 0
        assert snapshot.disk_usage_percent >= 0
        assert snapshot.disk_free_gb >= 0
        assert snapshot.network_bytes_sent >= 0
        assert snapshot.network_bytes_recv >= 0
    
    @pytest.mark.asyncio
    async def test_resource_history_collection(self):
        """Test resource history collection"""
        await self.monitor.start_monitoring()
        
        # Wait for a few samples
        await asyncio.sleep(2.5)
        
        await self.monitor.stop_monitoring()
        
        history = self.monitor.get_resource_history(1)  # Last hour
        assert len(history) >= 2  # Should have at least 2 samples
        
        # Check timestamps are in order
        for i in range(1, len(history)):
            assert history[i].timestamp > history[i-1].timestamp
    
    def test_get_current_usage(self):
        """Test getting current resource usage"""
        usage = self.monitor.get_current_usage()
        
        assert isinstance(usage, SystemResourceSnapshot)
        assert usage.cpu_percent >= 0
        assert usage.memory_percent >= 0


class TestPerformanceAnalyzer:
    """Test performance analysis and reporting"""
    
    def setup_method(self):
        self.tracker = PerformanceTracker()
        self.resource_monitor = SystemResourceMonitor()
        self.analyzer = PerformanceAnalyzer(self.tracker, self.resource_monitor)
    
    def _create_test_metrics(self):
        """Create test metrics for analysis"""
        # Create successful download operations
        for i in range(5):
            op_id = self.tracker.start_operation(
                PerformanceMetricType.DOWNLOAD_OPERATION,
                f"download_{i}"
            )
            time.sleep(0.01)  # Small delay
            self.tracker.end_operation(op_id, success=True)
        
        # Create failed health check
        op_id = self.tracker.start_operation(
            PerformanceMetricType.HEALTH_CHECK,
            "failed_health_check"
        )
        time.sleep(0.01)
        self.tracker.end_operation(op_id, success=False, error_message="Test failure")
        
        # Create slow fallback operation
        op_id = self.tracker.start_operation(
            PerformanceMetricType.FALLBACK_STRATEGY,
            "slow_fallback"
        )
        # Simulate slow operation by modifying duration
        metric = self.tracker.end_operation(op_id, success=True)
        metric.duration_seconds = 35.0  # 35 seconds (slow)
    
    def test_generate_performance_report_empty(self):
        """Test generating report with no metrics"""
        report = self.analyzer.generate_performance_report()
        
        assert isinstance(report, PerformanceReport)
        assert report.total_operations == 0
        assert report.success_rate == 0.0
        assert report.average_duration == 0.0
    
    def test_generate_performance_report_with_data(self):
        """Test generating report with test data"""
        self._create_test_metrics()
        
        report = self.analyzer.generate_performance_report()
        
        assert report.total_operations == 7  # 5 downloads + 1 health + 1 fallback
        assert 0.8 < report.success_rate < 1.0  # 6/7 successful
        assert report.average_duration > 0
        assert report.p95_duration > 0
        assert len(report.operations_by_type) > 0
        assert PerformanceMetricType.DOWNLOAD_OPERATION.value in report.operations_by_type
    
    def test_identify_bottlenecks(self):
        """Test bottleneck identification"""
        # Create metrics with high failure rate
        for i in range(10):
            op_id = self.tracker.start_operation(
                PerformanceMetricType.DOWNLOAD_OPERATION,
                f"failed_download_{i}"
            )
            self.tracker.end_operation(op_id, success=False, error_message="Network error")
        
        metrics = self.tracker.get_all_metrics()
        bottlenecks = self.analyzer._identify_bottlenecks(metrics, [])
        
        assert "High operation failure rate detected" in bottlenecks
    
    def test_generate_recommendations(self):
        """Test optimization recommendation generation"""
        # Create slow download operations
        for i in range(3):
            op_id = self.tracker.start_operation(
                PerformanceMetricType.DOWNLOAD_OPERATION,
                f"slow_download_{i}"
            )
            metric = self.tracker.end_operation(op_id, success=True)
            metric.duration_seconds = 400.0  # 6+ minutes (very slow)
        
        metrics = self.tracker.get_all_metrics()
        bottlenecks = []
        recommendations = self.analyzer._generate_recommendations(metrics, [], bottlenecks)
        
        assert any("parallel downloads" in rec.lower() for rec in recommendations)
        assert any("bandwidth optimization" in rec.lower() for rec in recommendations)


class TestPerformanceMonitoringSystem:
    """Test the main performance monitoring system"""
    
    def setup_method(self):
        # Create temporary config file
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_config.json"
        
        test_config = {
            "resource_sample_interval": 1,
            "metrics_retention_hours": 24,
            "dashboard_cache_ttl": 60
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(test_config, f)
        
        self.system = PerformanceMonitoringSystem(str(self.config_path))
    
    def teardown_method(self):
        # Cleanup
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @pytest.mark.asyncio
    async def test_system_start_stop(self):
        """Test starting and stopping the monitoring system"""
        await self.system.start()
        assert self.system.resource_monitor._monitoring
        
        await self.system.stop()
        assert not self.system.resource_monitor._monitoring
    
    def test_track_operations(self):
        """Test tracking different types of operations"""
        # Test download operation tracking
        download_id = self.system.track_download_operation(
            "test_download",
            {"model_id": "test-model", "size_mb": 1000}
        )
        assert download_id in self.system.tracker.active_operations
        
        # Test health check tracking
        health_id = self.system.track_health_check(
            "integrity_check",
            {"model_id": "test-model"}
        )
        assert health_id in self.system.tracker.active_operations
        
        # Test fallback strategy tracking
        fallback_id = self.system.track_fallback_strategy(
            "suggest_alternative",
            {"requested_model": "unavailable-model"}
        )
        assert fallback_id in self.system.tracker.active_operations
        
        # End operations
        self.system.end_tracking(download_id, success=True)
        self.system.end_tracking(health_id, success=True)
        self.system.end_tracking(fallback_id, success=False, error_message="No alternatives")
        
        # Verify operations completed
        assert download_id not in self.system.tracker.active_operations
        assert health_id not in self.system.tracker.active_operations
        assert fallback_id not in self.system.tracker.active_operations
    
    def test_get_performance_report(self):
        """Test getting performance report"""
        # Add some test operations
        op_id = self.system.track_model_operation("load_model", {"model_id": "test"})
        self.system.end_tracking(op_id, success=True)
        
        report = self.system.get_performance_report()
        assert isinstance(report, PerformanceReport)
        assert report.total_operations >= 1
    
    def test_get_dashboard_data(self):
        """Test getting dashboard data"""
        # Add some test data
        op_id = self.system.track_analytics_collection("usage_stats")
        self.system.end_tracking(op_id, success=True)
        
        dashboard_data = self.system.get_dashboard_data()
        
        assert "performance_summary" in dashboard_data
        assert "current_resources" in dashboard_data
        assert "operations_by_type" in dashboard_data
        assert "recent_activity" in dashboard_data
        assert "bottlenecks" in dashboard_data
        assert "recommendations" in dashboard_data
        assert "resource_trends" in dashboard_data
    
    def test_dashboard_data_caching(self):
        """Test dashboard data caching"""
        # Get data twice
        data1 = self.system.get_dashboard_data()
        data2 = self.system.get_dashboard_data()
        
        # Should be the same (cached)
        assert data1 == data2
        
        # Force refresh
        data3 = self.system.get_dashboard_data(force_refresh=True)
        
        # Structure should be the same, but timestamps might differ
        assert set(data1.keys()) == set(data3.keys())
    
    def test_config_loading(self):
        """Test configuration loading"""
        assert self.system.config["resource_sample_interval"] == 1
        assert self.system.config["metrics_retention_hours"] == 24
        assert self.system.config["dashboard_cache_ttl"] == 60
    
    def test_resource_trends_calculation(self):
        """Test resource trends calculation"""
        # Mock some resource history
        now = datetime.now()
        history = []
        
        # First half: lower usage
        for i in range(10):
            history.append(SystemResourceSnapshot(
                timestamp=now - timedelta(minutes=20-i),
                cpu_percent=30.0,
                memory_percent=40.0,
                memory_used_mb=4000.0,
                disk_usage_percent=50.0,
                disk_free_gb=100.0,
                network_bytes_sent=1000,
                network_bytes_recv=2000
            ))
        
        # Second half: higher usage
        for i in range(10):
            history.append(SystemResourceSnapshot(
                timestamp=now - timedelta(minutes=10-i),
                cpu_percent=60.0,  # Increased
                memory_percent=70.0,  # Increased
                memory_used_mb=7000.0,
                disk_usage_percent=50.0,
                disk_free_gb=80.0,  # Decreased (increased usage)
                network_bytes_sent=1000,
                network_bytes_recv=2000
            ))
        
        # Mock the resource monitor history
        self.system.resource_monitor.resource_history.extend(history)
        
        trends = self.system._calculate_resource_trends()
        
        assert trends["cpu"] == "increasing"
        assert trends["memory"] == "increasing"
        assert trends["disk"] == "increasing"


class TestPerformanceBenchmarks:
    """Performance benchmarking tests"""
    
    def setup_method(self):
        self.system = PerformanceMonitoringSystem()
    
    @pytest.mark.asyncio
    async def test_download_operation_benchmark(self):
        """Benchmark download operation tracking overhead"""
        iterations = 1000
        
        start_time = time.time()
        
        for i in range(iterations):
            op_id = self.system.track_download_operation(f"benchmark_download_{i}")
            self.system.end_tracking(op_id, success=True)
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time_per_operation = total_time / iterations
        
        # Should be reasonably fast (less than 20ms per operation)
        assert avg_time_per_operation < 0.02, f"Operation tracking too slow: {avg_time_per_operation:.6f}s per operation"
        
        print(f"Download operation tracking: {avg_time_per_operation*1000:.3f}ms per operation")
    
    @pytest.mark.asyncio
    async def test_resource_monitoring_overhead(self):
        """Test resource monitoring performance overhead"""
        monitor = SystemResourceMonitor(sample_interval=0.1)  # Very frequent sampling
        
        await monitor.start_monitoring()
        
        # Let it run and collect samples
        start_time = time.time()
        await asyncio.sleep(1.0)
        end_time = time.time()
        
        await monitor.stop_monitoring()
        
        history = monitor.get_resource_history(1)
        samples_collected = len(history)
        
        # Should collect several samples (allowing for timing variations)
        assert samples_collected >= 5, f"Too few samples collected: {samples_collected}"
        
        print(f"Resource monitoring: {samples_collected} samples in {end_time - start_time:.2f}s")
    
    def test_metrics_storage_efficiency(self):
        """Test memory efficiency of metrics storage"""
        import sys

        # Measure memory usage of storing many metrics
        initial_size = sys.getsizeof(self.system.tracker.completed_metrics)
        
        # Add many metrics
        for i in range(1000):
            op_id = self.system.track_model_operation(f"efficiency_test_{i}")
            self.system.end_tracking(op_id, success=True, additional_metadata={"test_data": i})
        
        final_size = sys.getsizeof(self.system.tracker.completed_metrics)
        size_per_metric = (final_size - initial_size) / 1000
        
        # Should be reasonably efficient (less than 1KB per metric)
        assert size_per_metric < 1024, f"Metrics storage too large: {size_per_metric:.2f} bytes per metric"
        
        print(f"Metrics storage: {size_per_metric:.2f} bytes per metric")
    
    def test_analysis_performance(self):
        """Test performance analysis speed"""
        # Create many test metrics
        for i in range(500):
            op_id = self.system.track_download_operation(f"analysis_test_{i}")
            self.system.end_tracking(op_id, success=i % 10 != 0)  # 90% success rate
        
        # Benchmark report generation
        start_time = time.time()
        report = self.system.get_performance_report()
        end_time = time.time()
        
        analysis_time = end_time - start_time
        
        # Should be fast (less than 1 second for 500 metrics)
        assert analysis_time < 1.0, f"Performance analysis too slow: {analysis_time:.3f}s"
        
        # Verify report quality
        assert report.total_operations == 500
        assert 0.85 < report.success_rate < 0.95
        
        print(f"Performance analysis: {analysis_time*1000:.2f}ms for {report.total_operations} metrics")
    
    def test_concurrent_tracking(self):
        """Test concurrent operation tracking"""
        import threading
        import queue

        results = queue.Queue()
        
        def track_operations(thread_id, count):
            try:
                for i in range(count):
                    op_id = self.system.track_model_operation(f"concurrent_{thread_id}_{i}")
                    time.sleep(0.001)  # Simulate work
                    self.system.end_tracking(op_id, success=True)
                results.put(("success", thread_id, count))
            except Exception as e:
                results.put(("error", thread_id, str(e)))
        
        # Start multiple threads
        threads = []
        thread_count = 5
        ops_per_thread = 50
        
        start_time = time.time()
        
        for i in range(thread_count):
            thread = threading.Thread(target=track_operations, args=(i, ops_per_thread))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        
        # Check results
        successful_threads = 0
        while not results.empty():
            result_type, thread_id, data = results.get()
            if result_type == "success":
                successful_threads += 1
            else:
                print(f"Thread {thread_id} failed: {data}")
        
        assert successful_threads == thread_count, "Some threads failed"
        
        # Verify all operations were tracked
        total_expected = thread_count * ops_per_thread
        completed_count = len(self.system.tracker.completed_metrics)
        
        assert completed_count >= total_expected, f"Missing operations: expected {total_expected}, got {completed_count}"
        
        total_time = end_time - start_time
        print(f"Concurrent tracking: {total_expected} operations in {total_time:.3f}s ({total_expected/total_time:.1f} ops/sec)")


@pytest.mark.asyncio
async def test_integration_with_existing_components():
    """Test integration with existing model management components"""
    system = PerformanceMonitoringSystem()
    
    # Simulate integration with enhanced model downloader
    download_id = system.track_download_operation(
        "enhanced_download_test",
        {
            "model_id": "test-model-v1",
            "size_mb": 2500,
            "retry_attempt": 1,
            "bandwidth_limit_mbps": 50
        }
    )
    
    # Simulate download progress
    await asyncio.sleep(0.1)
    
    system.end_tracking(
        download_id,
        success=True,
        additional_metadata={
            "actual_size_mb": 2487,
            "average_speed_mbps": 45.2,
            "retry_count": 0,
            "integrity_verified": True
        }
    )
    
    # Simulate health check integration
    health_id = system.track_health_check(
        "model_integrity_check",
        {
            "model_id": "test-model-v1",
            "check_type": "full_integrity",
            "files_to_check": 15
        }
    )
    
    await asyncio.sleep(0.05)
    
    system.end_tracking(
        health_id,
        success=True,
        additional_metadata={
            "files_checked": 15,
            "corruption_detected": False,
            "integrity_score": 1.0,
            "performance_score": 0.95
        }
    )
    
    # Simulate fallback strategy integration
    fallback_id = system.track_fallback_strategy(
        "suggest_alternative_model",
        {
            "requested_model": "unavailable-model-v2",
            "user_requirements": {"quality": "high", "speed": "medium"}
        }
    )
    
    await asyncio.sleep(0.02)
    
    system.end_tracking(
        fallback_id,
        success=True,
        additional_metadata={
            "suggested_model": "alternative-model-v1",
            "compatibility_score": 0.87,
            "alternatives_considered": 3,
            "fallback_strategy": "alternative_model"
        }
    )
    
    # Generate report and verify integration data
    report = system.get_performance_report()
    
    assert report.total_operations == 3
    assert report.success_rate == 1.0
    assert PerformanceMetricType.DOWNLOAD_OPERATION.value in report.operations_by_type
    assert PerformanceMetricType.HEALTH_CHECK.value in report.operations_by_type
    assert PerformanceMetricType.FALLBACK_STRATEGY.value in report.operations_by_type
    
    # Verify dashboard data includes integration metrics
    dashboard_data = system.get_dashboard_data()
    assert dashboard_data["performance_summary"]["total_operations_24h"] == 3
    assert dashboard_data["performance_summary"]["success_rate"] == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])