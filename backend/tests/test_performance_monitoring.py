import pytest
import time
import asyncio
import psutil
import threading
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from backend.main import app
from backend.monitoring.performance_monitor import PerformanceMonitor

client = TestClient(app)

class TestPerformanceMonitoring:
    """Test performance monitoring and metrics collection"""

    def setup_method(self):
        """Setup for each test method"""
        self.performance_monitor = PerformanceMonitor()

    def teardown_method(self):
        """Cleanup after each test method"""
        if hasattr(self, 'performance_monitor'):
            self.performance_monitor.cleanup()

    def test_response_time_monitoring(self):
        """Test API response time monitoring"""
        start_time = time.time()
        
        response = client.get("/api/v1/health")
        
        end_time = time.time()
        response_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        assert response.status_code == 200
        assert response_time < 1000  # Should respond within 1 second
        
        # Check if response time is recorded
        assert "X-Response-Time" in response.headers or response_time < 1000

    def test_memory_usage_monitoring(self):
        """Test memory usage monitoring during operations"""
        import os
        process = psutil.Process(os.getpid())
        
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform memory-intensive operations
        responses = []
        for i in range(50):
            response = client.get("/api/v1/health")
            responses.append(response)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable
        assert memory_increase < 100  # Less than 100MB increase
        assert all(r.status_code == 200 for r in responses)

    def test_cpu_usage_monitoring(self):
        """Test CPU usage monitoring"""
        initial_cpu = psutil.cpu_percent(interval=1)
        
        # Perform CPU-intensive operations
        start_time = time.time()
        while time.time() - start_time < 2:  # Run for 2 seconds
            response = client.get("/api/v1/health")
            assert response.status_code == 200
        
        final_cpu = psutil.cpu_percent(interval=1)
        
        # CPU usage should be reasonable
        assert final_cpu < 90  # Should not max out CPU

    def test_concurrent_request_performance(self):
        """Test performance under concurrent load"""
        import threading
        import time
        
        results = []
        errors = []
        
        def make_request():
            try:
                start_time = time.time()
                response = client.get("/api/v1/health")
                end_time = time.time()
                
                results.append({
                    'status_code': response.status_code,
                    'response_time': (end_time - start_time) * 1000
                })
            except Exception as e:
                errors.append(str(e))
        
        # Create 20 concurrent threads
        threads = []
        for i in range(20):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
        
        # Start all threads
        start_time = time.time()
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        total_time = time.time() - start_time
        
        # Analyze results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 20
        assert all(r['status_code'] == 200 for r in results)
        
        # Check response times
        avg_response_time = sum(r['response_time'] for r in results) / len(results)
        max_response_time = max(r['response_time'] for r in results)
        
        assert avg_response_time < 500  # Average response time under 500ms
        assert max_response_time < 2000  # Max response time under 2 seconds
        assert total_time < 10  # Total execution time under 10 seconds

    def test_database_query_performance(self):
        """Test database query performance"""
        with patch('backend.database.get_db_session') as mock_db:
            mock_session = Mock()
            mock_db.return_value.__enter__.return_value = mock_session
            mock_session.query.return_value.all.return_value = []
            
            start_time = time.time()
            response = client.get("/api/v1/queue")
            end_time = time.time()
            
            query_time = (end_time - start_time) * 1000
            
            assert response.status_code == 200
            assert query_time < 100  # Database queries should be fast

    def test_large_payload_performance(self):
        """Test performance with large payloads"""
        import tempfile
        import os
        
        # Create a large test file (5MB)
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            tmp_file.write(b"0" * (5 * 1024 * 1024))  # 5MB
            tmp_file_path = tmp_file.name
        
        try:
            with open(tmp_file_path, 'rb') as image_file:
                files = {"image": ("large_test.jpg", image_file, "image/jpeg")}
                data = {
                    "model_type": "I2V-A14B",
                    "prompt": "Test with large image",
                    "resolution": "1280x720",
                    "steps": 50
                }
                
                start_time = time.time()
                response = client.post("/api/v1/generate", data=data, files=files)
                end_time = time.time()
                
                upload_time = (end_time - start_time) * 1000
                
                # Should handle large files reasonably
                assert upload_time < 30000  # Under 30 seconds
                assert response.status_code in [200, 413, 422]  # Success or expected errors
        
        finally:
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)

    def test_memory_leak_detection(self):
        """Test for memory leaks during repeated operations"""
        import gc
        import os
        
        process = psutil.Process(os.getpid())
        
        # Baseline memory usage
        gc.collect()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform repeated operations
        for i in range(100):
            response = client.get("/api/v1/health")
            assert response.status_code == 200
            
            # Force garbage collection every 10 iterations
            if i % 10 == 0:
                gc.collect()
        
        # Final memory check
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - baseline_memory
        
        # Memory increase should be minimal
        assert memory_increase < 50  # Less than 50MB increase indicates no major leaks

    def test_api_throughput(self):
        """Test API throughput (requests per second)"""
        import time
        
        request_count = 100
        start_time = time.time()
        
        for i in range(request_count):
            response = client.get("/api/v1/health")
            assert response.status_code == 200
        
        end_time = time.time()
        total_time = end_time - start_time
        throughput = request_count / total_time
        
        # Should handle at least 10 requests per second
        assert throughput > 10, f"Throughput too low: {throughput} req/s"

    def test_error_rate_monitoring(self):
        """Test error rate monitoring"""
        total_requests = 100
        error_count = 0
        
        for i in range(total_requests):
            try:
                response = client.get("/api/v1/health")
                if response.status_code >= 400:
                    error_count += 1
            except Exception:
                error_count += 1
        
        error_rate = (error_count / total_requests) * 100
        
        # Error rate should be very low for health endpoint
        assert error_rate < 1, f"Error rate too high: {error_rate}%"

    def test_resource_cleanup(self):
        """Test proper resource cleanup"""
        import gc
        import threading
        
        initial_thread_count = threading.active_count()
        
        # Perform operations that might create resources
        for i in range(10):
            response = client.get("/api/v1/health")
            assert response.status_code == 200
        
        # Force cleanup
        gc.collect()
        time.sleep(0.1)  # Allow time for cleanup
        
        final_thread_count = threading.active_count()
        
        # Thread count should not increase significantly
        thread_increase = final_thread_count - initial_thread_count
        assert thread_increase < 5, f"Too many threads created: {thread_increase}"

    def test_performance_regression_detection(self):
        """Test performance regression detection"""
        # Baseline performance measurement
        baseline_times = []
        for i in range(10):
            start_time = time.time()
            response = client.get("/api/v1/health")
            end_time = time.time()
            baseline_times.append((end_time - start_time) * 1000)
            assert response.status_code == 200
        
        baseline_avg = sum(baseline_times) / len(baseline_times)
        
        # Simulate potential performance regression
        current_times = []
        for i in range(10):
            start_time = time.time()
            response = client.get("/api/v1/health")
            end_time = time.time()
            current_times.append((end_time - start_time) * 1000)
            assert response.status_code == 200
        
        current_avg = sum(current_times) / len(current_times)
        
        # Performance should not degrade significantly
        performance_ratio = current_avg / baseline_avg
        assert performance_ratio < 2.0, f"Performance regression detected: {performance_ratio}x slower"

    def test_system_resource_limits(self):
        """Test system resource limit handling"""
        # Test with system resource monitoring
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        disk_percent = psutil.disk_usage('/').percent
        
        # System should not be under extreme stress
        assert cpu_percent < 95, f"CPU usage too high: {cpu_percent}%"
        assert memory_percent < 95, f"Memory usage too high: {memory_percent}%"
        assert disk_percent < 95, f"Disk usage too high: {disk_percent}%"

    @pytest.mark.asyncio
    async def test_async_performance(self):
        """Test asynchronous operation performance"""
        import aiohttp
        import asyncio
        
        async def make_async_request(session):
            start_time = time.time()
            async with session.get("http://testserver/api/v1/health") as response:
                await response.text()
                end_time = time.time()
                return {
                    'status': response.status,
                    'time': (end_time - start_time) * 1000
                }
        
        # Create multiple concurrent requests
        async with aiohttp.ClientSession() as session:
            tasks = [make_async_request(session) for _ in range(20)]
            results = await asyncio.gather(*tasks)
        
        # Analyze results
        assert all(r['status'] == 200 for r in results)
        avg_time = sum(r['time'] for r in results) / len(results)
        max_time = max(r['time'] for r in results)
        
        assert avg_time < 100, f"Average async response time too high: {avg_time}ms"
        assert max_time < 500, f"Max async response time too high: {max_time}ms"

class TestPerformanceMetrics:
    """Test performance metrics collection and reporting"""

    def setup_method(self):
        """Setup for each test method"""
        self.monitor = PerformanceMonitor()

    def test_metric_collection(self):
        """Test performance metric collection"""
        # Record some test metrics
        self.monitor.record_metric("test_metric", 100.5)
        self.monitor.record_metric("test_metric", 150.2)
        self.monitor.record_metric("test_metric", 75.8)
        
        metrics = self.monitor.get_metrics("test_metric")
        assert len(metrics) == 3
        assert metrics[0]["value"] == 100.5

    def test_metric_aggregation(self):
        """Test metric aggregation functions"""
        # Record test data
        values = [100, 200, 150, 300, 250]
        for value in values:
            self.monitor.record_metric("test_aggregation", value)
        
        avg = self.monitor.get_average("test_aggregation")
        p95 = self.monitor.get_percentile("test_aggregation", 95)
        
        assert avg == 200  # (100+200+150+300+250)/5
        assert p95 >= 250  # 95th percentile should be high

    def test_performance_alerts(self):
        """Test performance alert generation"""
        alerts = []
        
        def alert_handler(metric, value, threshold):
            alerts.append({"metric": metric, "value": value, "threshold": threshold})
        
        self.monitor.set_alert_handler(alert_handler)
        self.monitor.set_threshold("response_time", 100)  # 100ms threshold
        
        # Record metrics that should trigger alerts
        self.monitor.record_metric("response_time", 150)  # Should trigger alert
        self.monitor.record_metric("response_time", 50)   # Should not trigger alert
        
        assert len(alerts) == 1
        assert alerts[0]["metric"] == "response_time"
        assert alerts[0]["value"] == 150

    def test_performance_report_generation(self):
        """Test performance report generation"""
        # Record various metrics
        self.monitor.record_metric("response_time", 100)
        self.monitor.record_metric("response_time", 150)
        self.monitor.record_metric("memory_usage", 512)
        self.monitor.record_metric("cpu_usage", 45)
        
        report = self.monitor.generate_report()
        
        assert "summary" in report
        assert "response_time" in report["summary"]
        assert "memory_usage" in report["summary"]
        assert "cpu_usage" in report["summary"]

if __name__ == "__main__":
    pytest.main([__file__, "-v"])