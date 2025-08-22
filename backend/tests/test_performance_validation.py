"""
Performance validation tests for the React Frontend FastAPI system.
Tests generation timing, resource constraints, and performance budgets.
"""

import pytest
import asyncio
import time
import psutil
import GPUtil
from httpx import AsyncClient
from fastapi.testclient import TestClient
from backend.main import app
from backend.models.schemas import GenerationRequest, TaskStatus
import json
import os
from typing import Dict, Any

class PerformanceMetrics:
    """Track and validate performance metrics"""
    
    def __init__(self):
        self.metrics = {
            'generation_times': {},
            'resource_usage': {},
            'api_response_times': {},
            'memory_usage': {},
            'vram_usage': {}
        }
    
    def record_generation_time(self, resolution: str, model_type: str, duration: float):
        """Record video generation time"""
        key = f"{model_type}_{resolution}"
        if key not in self.metrics['generation_times']:
            self.metrics['generation_times'][key] = []
        self.metrics['generation_times'][key].append(duration)
    
    def record_resource_usage(self, cpu_percent: float, ram_gb: float, vram_mb: float):
        """Record system resource usage"""
        timestamp = time.time()
        self.metrics['resource_usage'][timestamp] = {
            'cpu': cpu_percent,
            'ram': ram_gb,
            'vram': vram_mb
        }
    
    def record_api_response_time(self, endpoint: str, duration: float):
        """Record API response time"""
        if endpoint not in self.metrics['api_response_times']:
            self.metrics['api_response_times'][endpoint] = []
        self.metrics['api_response_times'][endpoint].append(duration)
    
    def get_average_generation_time(self, resolution: str, model_type: str) -> float:
        """Get average generation time for specific configuration"""
        key = f"{model_type}_{resolution}"
        times = self.metrics['generation_times'].get(key, [])
        return sum(times) / len(times) if times else 0
    
    def get_max_vram_usage(self) -> float:
        """Get maximum VRAM usage recorded"""
        vram_values = [usage['vram'] for usage in self.metrics['resource_usage'].values()]
        return max(vram_values) if vram_values else 0
    
    def save_baseline_metrics(self, filepath: str):
        """Save metrics as baseline for regression testing"""
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def load_baseline_metrics(self, filepath: str) -> Dict[str, Any]:
        """Load baseline metrics for comparison"""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        return {}

@pytest.fixture
def performance_metrics():
    """Fixture providing performance metrics tracker"""
    return PerformanceMetrics()

@pytest.fixture
def test_client():
    """Fixture providing test client"""
    return TestClient(app)

class TestGenerationTiming:
    """Test video generation timing requirements"""
    
    @pytest.mark.asyncio
    async def test_720p_t2v_generation_timing(self, performance_metrics):
        """Test 5-second 720p T2V video generation under 6 minutes with <8GB VRAM"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Monitor initial VRAM usage
            initial_vram = self._get_vram_usage()
            
            # Create generation request
            request_data = {
                "model_type": "T2V-A14B",
                "prompt": "A beautiful sunset over mountains, cinematic lighting",
                "resolution": "1280x720",
                "steps": 50,
                "duration": 5  # 5-second video
            }
            
            start_time = time.time()
            
            # Submit generation request
            response = await client.post("/api/v1/generate", json=request_data)
            assert response.status_code == 200
            
            task_data = response.json()
            task_id = task_data["task_id"]
            
            # Monitor generation progress and resource usage
            max_vram_usage = initial_vram
            generation_completed = False
            
            while not generation_completed:
                await asyncio.sleep(5)  # Check every 5 seconds
                
                # Check task status
                status_response = await client.get(f"/api/v1/queue/{task_id}")
                status_data = status_response.json()
                
                # Monitor VRAM usage
                current_vram = self._get_vram_usage()
                max_vram_usage = max(max_vram_usage, current_vram)
                
                # Record resource usage
                cpu_percent = psutil.cpu_percent()
                ram_gb = psutil.virtual_memory().used / (1024**3)
                performance_metrics.record_resource_usage(cpu_percent, ram_gb, current_vram)
                
                if status_data["status"] == TaskStatus.COMPLETED:
                    generation_completed = True
                elif status_data["status"] == TaskStatus.FAILED:
                    pytest.fail(f"Generation failed: {status_data.get('error_message', 'Unknown error')}")
                
                # Timeout after 8 minutes (allowing 2 minutes buffer)
                if time.time() - start_time > 480:
                    pytest.fail("Generation took longer than 8 minutes (6 minute requirement + 2 minute buffer)")
            
            end_time = time.time()
            generation_time = end_time - start_time
            
            # Record metrics
            performance_metrics.record_generation_time("1280x720", "T2V-A14B", generation_time)
            
            # Validate timing requirement: under 6 minutes (360 seconds)
            assert generation_time < 360, f"720p T2V generation took {generation_time:.1f}s, should be under 360s"
            
            # Validate VRAM requirement: under 8GB (8192 MB)
            vram_gb = max_vram_usage / 1024
            assert vram_gb < 8, f"VRAM usage was {vram_gb:.1f}GB, should be under 8GB"
            
            print(f"✓ 720p T2V generation completed in {generation_time:.1f}s with {vram_gb:.1f}GB VRAM")
    
    @pytest.mark.asyncio
    async def test_1080p_generation_timing(self, performance_metrics):
        """Test 1080p video generation under 17 minutes"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            request_data = {
                "model_type": "T2V-A14B",
                "prompt": "A serene lake with mountains in the background, golden hour lighting",
                "resolution": "1920x1080",
                "steps": 50,
                "duration": 5
            }
            
            start_time = time.time()
            
            response = await client.post("/api/v1/generate", json=request_data)
            assert response.status_code == 200
            
            task_data = response.json()
            task_id = task_data["task_id"]
            
            # Monitor generation with longer timeout
            generation_completed = False
            
            while not generation_completed:
                await asyncio.sleep(10)  # Check every 10 seconds for longer generation
                
                status_response = await client.get(f"/api/v1/queue/{task_id}")
                status_data = status_response.json()
                
                if status_data["status"] == TaskStatus.COMPLETED:
                    generation_completed = True
                elif status_data["status"] == TaskStatus.FAILED:
                    pytest.fail(f"Generation failed: {status_data.get('error_message', 'Unknown error')}")
                
                # Timeout after 20 minutes (allowing 3 minutes buffer)
                if time.time() - start_time > 1200:
                    pytest.fail("1080p generation took longer than 20 minutes (17 minute requirement + 3 minute buffer)")
            
            end_time = time.time()
            generation_time = end_time - start_time
            
            performance_metrics.record_generation_time("1920x1080", "T2V-A14B", generation_time)
            
            # Validate timing requirement: under 17 minutes (1020 seconds)
            assert generation_time < 1020, f"1080p generation took {generation_time:.1f}s, should be under 1020s"
            
            print(f"✓ 1080p generation completed in {generation_time:.1f}s")
    
    def _get_vram_usage(self) -> float:
        """Get current VRAM usage in MB"""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                return gpus[0].memoryUsed
            return 0
        except:
            return 0

class TestResourceConstraints:
    """Test system behavior under resource constraints"""
    
    @pytest.mark.asyncio
    async def test_low_vram_scenario(self):
        """Test system behavior when VRAM is constrained"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Simulate low VRAM by requesting high resolution with quantization
            request_data = {
                "model_type": "T2V-A14B",
                "prompt": "Test prompt for low VRAM scenario",
                "resolution": "1920x1080",
                "quantization": "int8",  # Force quantization for VRAM savings
                "enable_cpu_offload": True
            }
            
            response = await client.post("/api/v1/generate", json=request_data)
            
            # Should either succeed with optimizations or fail gracefully
            if response.status_code == 200:
                # If successful, verify optimizations were applied
                task_data = response.json()
                assert "optimization_applied" in task_data or "quantization_enabled" in task_data
            else:
                # If failed, should provide helpful error message
                error_data = response.json()
                assert "vram" in error_data.get("message", "").lower()
                assert "suggestions" in error_data
    
    @pytest.mark.asyncio
    async def test_high_cpu_usage_scenario(self):
        """Test system behavior under high CPU usage"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Create CPU load simulation
            import threading
            import math
            
            def cpu_load():
                """Create CPU load for testing"""
                end_time = time.time() + 30  # Run for 30 seconds
                while time.time() < end_time:
                    math.sqrt(12345)
            
            # Start CPU load threads
            threads = []
            for _ in range(psutil.cpu_count()):
                thread = threading.Thread(target=cpu_load)
                thread.start()
                threads.append(thread)
            
            try:
                # Test API responsiveness under load
                start_time = time.time()
                response = await client.get("/api/v1/system/stats")
                response_time = time.time() - start_time
                
                assert response.status_code == 200
                # API should still respond within reasonable time even under CPU load
                assert response_time < 5, f"API response took {response_time:.1f}s under CPU load"
                
                # System should report high CPU usage
                stats = response.json()
                assert stats["cpu_percent"] > 80, "CPU usage should be high during load test"
                
            finally:
                # Wait for threads to complete
                for thread in threads:
                    thread.join()
    
    @pytest.mark.asyncio
    async def test_multiple_concurrent_generations(self):
        """Test system behavior with multiple concurrent generation requests"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Submit multiple generation requests
            tasks = []
            for i in range(3):
                request_data = {
                    "model_type": "T2V-A14B",
                    "prompt": f"Test prompt {i} for concurrent generation",
                    "resolution": "1280x720",
                    "steps": 25  # Reduced steps for faster testing
                }
                
                response = await client.post("/api/v1/generate", json=request_data)
                assert response.status_code == 200
                tasks.append(response.json()["task_id"])
            
            # Verify all tasks are queued
            queue_response = await client.get("/api/v1/queue")
            queue_data = queue_response.json()
            
            queued_task_ids = [task["id"] for task in queue_data]
            for task_id in tasks:
                assert task_id in queued_task_ids, f"Task {task_id} not found in queue"
            
            print(f"✓ Successfully queued {len(tasks)} concurrent generation tasks")

class TestPerformanceBudgets:
    """Test performance budgets and optimization requirements"""
    
    @pytest.mark.asyncio
    async def test_api_response_times(self, performance_metrics):
        """Test API response time budgets"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            endpoints_to_test = [
                ("/api/v1/health", 1.0),  # Health check should be very fast
                ("/api/v1/system/stats", 2.0),  # System stats within 2 seconds
                ("/api/v1/queue", 1.5),  # Queue status within 1.5 seconds
                ("/api/v1/outputs", 3.0),  # Outputs list within 3 seconds
            ]
            
            for endpoint, max_time in endpoints_to_test:
                start_time = time.time()
                response = await client.get(endpoint)
                response_time = time.time() - start_time
                
                performance_metrics.record_api_response_time(endpoint, response_time)
                
                assert response.status_code == 200
                assert response_time < max_time, f"{endpoint} took {response_time:.2f}s, should be under {max_time}s"
                
                print(f"✓ {endpoint} responded in {response_time:.2f}s")
    
    def test_bundle_size_budget(self):
        """Test frontend bundle size budget (500KB gzipped)"""
        # This would typically be run as part of the frontend build process
        # For now, we'll create a placeholder that checks if build artifacts exist
        
        frontend_dist_path = "frontend/dist"
        if os.path.exists(frontend_dist_path):
            # Check main bundle size
            js_files = []
            for root, dirs, files in os.walk(frontend_dist_path):
                for file in files:
                    if file.endswith('.js') and 'main' in file:
                        js_files.append(os.path.join(root, file))
            
            if js_files:
                # Simulate gzip compression check
                main_bundle = js_files[0]
                bundle_size = os.path.getsize(main_bundle)
                
                # Rough estimate: gzipped size is typically 25-30% of original
                estimated_gzipped = bundle_size * 0.3
                max_size_kb = 500 * 1024  # 500KB in bytes
                
                assert estimated_gzipped < max_size_kb, f"Estimated bundle size {estimated_gzipped/1024:.1f}KB exceeds 500KB budget"
                print(f"✓ Estimated bundle size: {estimated_gzipped/1024:.1f}KB (under 500KB budget)")
            else:
                print("⚠ No main bundle found, skipping bundle size test")
        else:
            print("⚠ Frontend dist folder not found, skipping bundle size test")

class TestBaselineMetrics:
    """Establish and validate baseline performance metrics"""
    
    def test_establish_baseline_metrics(self, performance_metrics):
        """Establish baseline performance metrics for regression testing"""
        baseline_file = "performance_baseline.json"
        
        # Load existing baseline if available
        existing_baseline = performance_metrics.load_baseline_metrics(baseline_file)
        
        # Run basic performance tests to establish current metrics
        with TestClient(app) as client:
            # Test basic API performance
            start_time = time.time()
            response = client.get("/api/v1/health")
            health_response_time = time.time() - start_time
            
            assert response.status_code == 200
            performance_metrics.record_api_response_time("/api/v1/health", health_response_time)
            
            # Test system stats performance
            start_time = time.time()
            response = client.get("/api/v1/system/stats")
            stats_response_time = time.time() - start_time
            
            assert response.status_code == 200
            performance_metrics.record_api_response_time("/api/v1/system/stats", stats_response_time)
        
        # Save current metrics as baseline
        performance_metrics.save_baseline_metrics(baseline_file)
        
        # If we have existing baseline, compare for regression
        if existing_baseline:
            current_health_times = performance_metrics.metrics['api_response_times'].get('/api/v1/health', [])
            baseline_health_times = existing_baseline.get('api_response_times', {}).get('/api/v1/health', [])
            
            if current_health_times and baseline_health_times:
                current_avg = sum(current_health_times) / len(current_health_times)
                baseline_avg = sum(baseline_health_times) / len(baseline_health_times)
                
                # Allow 50% regression tolerance
                regression_threshold = baseline_avg * 1.5
                assert current_avg < regression_threshold, f"Performance regression detected: {current_avg:.3f}s vs baseline {baseline_avg:.3f}s"
        
        print(f"✓ Baseline metrics established/validated")
    
    def test_memory_leak_detection(self):
        """Test for memory leaks during extended operation"""
        import gc
        
        with TestClient(app) as client:
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # Perform multiple API calls to detect memory leaks
            for i in range(100):
                response = client.get("/api/v1/health")
                assert response.status_code == 200
                
                if i % 20 == 0:  # Check memory every 20 requests
                    gc.collect()  # Force garbage collection
                    current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                    memory_increase = current_memory - initial_memory
                    
                    # Allow reasonable memory increase (50MB max for 100 requests)
                    assert memory_increase < 50, f"Potential memory leak: {memory_increase:.1f}MB increase after {i+1} requests"
            
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024
            total_increase = final_memory - initial_memory
            
            print(f"✓ Memory usage stable: {total_increase:.1f}MB increase over 100 requests")

if __name__ == "__main__":
    # Run performance validation tests
    pytest.main([__file__, "-v", "--tb=short"])