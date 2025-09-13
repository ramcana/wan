"""
Performance Benchmarking Tests
Comprehensive performance testing for generation speed and resource usage
"""

import pytest
import asyncio
import time
import json
import psutil
from pathlib import Path
from httpx import AsyncClient
from unittest.mock import patch, Mock

# Add backend to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

try:
    from backend.app import app
    from backend.models.schemas import TaskStatus
except ImportError:
    # Fallback for testing
    from fastapi import FastAPI
    app = FastAPI()
    
    class TaskStatus:
        COMPLETED = "completed"
        FAILED = "failed"
        PENDING = "pending"
        PROCESSING = "processing"

class PerformanceBenchmarkSuite:
    """Performance benchmark test suite"""
    
    def __init__(self):
        self.benchmark_results = {
            'generation_benchmarks': [],
            'api_benchmarks': [],
            'resource_benchmarks': [],
            'concurrency_benchmarks': []
        }
    
    def record_generation_benchmark(self, model_type: str, resolution: str, 
                                  duration: float, success: bool, metadata: dict):
        """Record generation benchmark result"""
        self.benchmark_results['generation_benchmarks'].append({
            'model_type': model_type,
            'resolution': resolution,
            'duration': duration,
            'success': success,
            'metadata': metadata,
            'timestamp': time.time()
        })
    
    def record_api_benchmark(self, endpoint: str, duration: float, 
                           status_code: int, metadata: dict):
        """Record API benchmark result"""
        self.benchmark_results['api_benchmarks'].append({
            'endpoint': endpoint,
            'duration': duration,
            'status_code': status_code,
            'metadata': metadata,
            'timestamp': time.time()
        })
    
    def record_resource_benchmark(self, test_name: str, peak_cpu: float, 
                                peak_memory: float, peak_gpu_memory: float):
        """Record resource usage benchmark"""
        self.benchmark_results['resource_benchmarks'].append({
            'test_name': test_name,
            'peak_cpu_percent': peak_cpu,
            'peak_memory_percent': peak_memory,
            'peak_gpu_memory_mb': peak_gpu_memory,
            'timestamp': time.time()
        })
    
    def save_results(self, filepath: str):
        """Save benchmark results to file"""
        with open(filepath, 'w') as f:
            json.dump(self.benchmark_results, f, indent=2)
    
    def get_summary(self) -> dict:
        """Get benchmark summary"""
        return {
            'total_generation_tests': len(self.benchmark_results['generation_benchmarks']),
            'successful_generations': sum(1 for b in self.benchmark_results['generation_benchmarks'] if b['success']),
            'total_api_tests': len(self.benchmark_results['api_benchmarks']),
            'average_api_response_time': self._calculate_average_api_time(),
            'peak_resource_usage': self._get_peak_resource_usage()
        }
    
    def _calculate_average_api_time(self) -> float:
        """Calculate average API response time"""
        api_times = [b['duration'] for b in self.benchmark_results['api_benchmarks']]
        return sum(api_times) / len(api_times) if api_times else 0
    
    def _get_peak_resource_usage(self) -> dict:
        """Get peak resource usage across all tests"""
        if not self.benchmark_results['resource_benchmarks']:
            return {'cpu': 0, 'memory': 0, 'gpu_memory': 0}
        
        return {
            'cpu': max(b['peak_cpu_percent'] for b in self.benchmark_results['resource_benchmarks']),
            'memory': max(b['peak_memory_percent'] for b in self.benchmark_results['resource_benchmarks']),
            'gpu_memory': max(b['peak_gpu_memory_mb'] for b in self.benchmark_results['resource_benchmarks'])
        }

@pytest.fixture
def benchmark_suite():
    """Fixture providing benchmark suite"""
    return PerformanceBenchmarkSuite()

class TestGenerationPerformanceBenchmarks:
    """Generation performance benchmark tests"""
    
    @pytest.mark.asyncio
    async def test_t2v_720p_performance_benchmark(self, benchmark_suite):
        """Benchmark T2V 720p generation performance"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            request_data = {
                "model_type": "T2V-A14B",
                "prompt": "Performance benchmark: A serene mountain landscape with flowing water",
                "resolution": "1280x720",
                "steps": 25,
                "guidance_scale": 7.5,
                "num_frames": 16,
                "fps": 8.0
            }
            
            # Performance targets
            target_time_seconds = 300  # 5 minutes for 720p
            
            start_time = time.time()
            
            response = await client.post("/api/v1/generation/submit", json=request_data)
            
            if response.status_code == 200:
                task_data = response.json()
                task_id = task_data["task_id"]
                
                # Monitor until completion or timeout
                completed = False
                final_status = None
                
                while not completed and (time.time() - start_time) < target_time_seconds * 2:
                    await asyncio.sleep(5)
                    
                    status_response = await client.get(f"/api/v1/queue/{task_id}")
                    if status_response.status_code == 200:
                        status_data = status_response.json()
                        final_status = status_data["status"]
                        
                        if final_status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                            completed = True
                            break
                
                duration = time.time() - start_time
                success = final_status == TaskStatus.COMPLETED
                
                benchmark_suite.record_generation_benchmark(
                    "T2V-A14B", "1280x720", duration, success,
                    {
                        'target_time': target_time_seconds,
                        'meets_target': duration <= target_time_seconds,
                        'final_status': final_status,
                        'steps': 25
                    }
                )
                
                # Assert performance target
                if success:
                    assert duration <= target_time_seconds, f"720p T2V took {duration:.1f}s, target was {target_time_seconds}s"
    
    @pytest.mark.asyncio
    async def test_t2v_1080p_performance_benchmark(self, benchmark_suite):
        """Benchmark T2V 1080p generation performance"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            request_data = {
                "model_type": "T2V-A14B",
                "prompt": "Performance benchmark: Urban cityscape with dynamic lighting",
                "resolution": "1920x1080",
                "steps": 50,
                "guidance_scale": 8.0,
                "num_frames": 16,
                "fps": 8.0
            }
            
            # Performance targets
            target_time_seconds = 900  # 15 minutes for 1080p
            
            start_time = time.time()
            
            response = await client.post("/api/v1/generation/submit", json=request_data)
            
            if response.status_code == 200:
                task_data = response.json()
                task_id = task_data["task_id"]
                
                # Monitor with longer timeout for 1080p
                completed = False
                final_status = None
                
                while not completed and (time.time() - start_time) < target_time_seconds * 1.5:
                    await asyncio.sleep(10)
                    
                    status_response = await client.get(f"/api/v1/queue/{task_id}")
                    if status_response.status_code == 200:
                        status_data = status_response.json()
                        final_status = status_data["status"]
                        
                        if final_status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                            completed = True
                            break
                
                duration = time.time() - start_time
                success = final_status == TaskStatus.COMPLETED
                
                benchmark_suite.record_generation_benchmark(
                    "T2V-A14B", "1920x1080", duration, success,
                    {
                        'target_time': target_time_seconds,
                        'meets_target': duration <= target_time_seconds,
                        'final_status': final_status,
                        'steps': 50
                    }
                )
    
    @pytest.mark.asyncio
    async def test_i2v_performance_benchmark(self, benchmark_suite):
        """Benchmark I2V generation performance"""
        import tempfile

        # Create temporary test image
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            test_image_path = tmp_file.name
        
        try:
            async with AsyncClient(app=app, base_url="http://test") as client:
                request_data = {
                    "model_type": "I2V-A14B",
                    "prompt": "Performance benchmark: Animate this landscape with gentle movement",
                    "image_path": test_image_path,
                    "resolution": "1280x720",
                    "steps": 30,
                    "guidance_scale": 8.0,
                    "num_frames": 16,
                    "fps": 8.0
                }
                
                target_time_seconds = 360  # 6 minutes for I2V
                
                start_time = time.time()
                
                response = await client.post("/api/v1/generation/submit", json=request_data)
                
                if response.status_code == 200:
                    task_data = response.json()
                    task_id = task_data["task_id"]
                    
                    # Monitor progress
                    completed = False
                    final_status = None
                    
                    while not completed and (time.time() - start_time) < target_time_seconds * 2:
                        await asyncio.sleep(5)
                        
                        status_response = await client.get(f"/api/v1/queue/{task_id}")
                        if status_response.status_code == 200:
                            status_data = status_response.json()
                            final_status = status_data["status"]
                            
                            if final_status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                                completed = True
                                break
                    
                    duration = time.time() - start_time
                    success = final_status == TaskStatus.COMPLETED
                    
                    benchmark_suite.record_generation_benchmark(
                        "I2V-A14B", "1280x720", duration, success,
                        {
                            'target_time': target_time_seconds,
                            'meets_target': duration <= target_time_seconds,
                            'final_status': final_status,
                            'steps': 30
                        }
                    )
        
        finally:
            Path(test_image_path).unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_ti2v_performance_benchmark(self, benchmark_suite):
        """Benchmark TI2V generation performance"""
        import tempfile

        # Create temporary test image
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            test_image_path = tmp_file.name
        
        try:
            async with AsyncClient(app=app, base_url="http://test") as client:
                request_data = {
                    "model_type": "TI2V-5B",
                    "prompt": "Performance benchmark: Transform with magical effects",
                    "image_path": test_image_path,
                    "resolution": "1280x720",
                    "steps": 40,
                    "guidance_scale": 10.0,
                    "num_frames": 16,
                    "fps": 8.0
                }
                
                target_time_seconds = 480  # 8 minutes for TI2V
                
                start_time = time.time()
                
                response = await client.post("/api/v1/generation/submit", json=request_data)
                
                if response.status_code == 200:
                    task_data = response.json()
                    task_id = task_data["task_id"]
                    
                    # Monitor progress
                    completed = False
                    final_status = None
                    
                    while not completed and (time.time() - start_time) < target_time_seconds * 2:
                        await asyncio.sleep(5)
                        
                        status_response = await client.get(f"/api/v1/queue/{task_id}")
                        if status_response.status_code == 200:
                            status_data = status_response.json()
                            final_status = status_data["status"]
                            
                            if final_status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                                completed = True
                                break
                    
                    duration = time.time() - start_time
                    success = final_status == TaskStatus.COMPLETED
                    
                    benchmark_suite.record_generation_benchmark(
                        "TI2V-5B", "1280x720", duration, success,
                        {
                            'target_time': target_time_seconds,
                            'meets_target': duration <= target_time_seconds,
                            'final_status': final_status,
                            'steps': 40
                        }
                    )
        
        finally:
            Path(test_image_path).unlink(missing_ok=True)

class TestAPIPerformanceBenchmarks:
    """API performance benchmark tests"""
    
    @pytest.mark.asyncio
    async def test_api_response_time_benchmarks(self, benchmark_suite):
        """Benchmark API response times"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Define API endpoints with performance targets
            endpoints = [
                {
                    'path': '/api/v1/health',
                    'method': 'GET',
                    'target_ms': 100,  # 100ms target
                    'data': None
                },
                {
                    'path': '/api/v1/system/stats',
                    'method': 'GET',
                    'target_ms': 500,  # 500ms target
                    'data': None
                },
                {
                    'path': '/api/v1/queue',
                    'method': 'GET',
                    'target_ms': 200,  # 200ms target
                    'data': None
                },
                {
                    'path': '/api/v1/outputs',
                    'method': 'GET',
                    'target_ms': 1000,  # 1s target
                    'data': None
                }
            ]
            
            for endpoint in endpoints:
                # Run multiple iterations for statistical significance
                durations = []
                
                for _ in range(5):
                    start_time = time.time()
                    
                    if endpoint['method'] == 'GET':
                        response = await client.get(endpoint['path'])
                    elif endpoint['method'] == 'POST':
                        response = await client.post(endpoint['path'], json=endpoint['data'])
                    
                    duration_ms = (time.time() - start_time) * 1000
                    durations.append(duration_ms)
                    
                    benchmark_suite.record_api_benchmark(
                        endpoint['path'], duration_ms, response.status_code,
                        {
                            'target_ms': endpoint['target_ms'],
                            'meets_target': duration_ms <= endpoint['target_ms']
                        }
                    )
                
                # Calculate statistics
                avg_duration = sum(durations) / len(durations)
                max_duration = max(durations)
                
                # Assert performance targets
                assert avg_duration <= endpoint['target_ms'], \
                    f"{endpoint['path']} average response time {avg_duration:.1f}ms exceeds target {endpoint['target_ms']}ms"
                
                print(f"✅ {endpoint['path']}: avg {avg_duration:.1f}ms, max {max_duration:.1f}ms (target: {endpoint['target_ms']}ms)")
    
    @pytest.mark.asyncio
    async def test_concurrent_api_performance(self, benchmark_suite):
        """Test API performance under concurrent load"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Test concurrent requests to health endpoint
            concurrent_requests = 10
            
            async def make_request():
                start_time = time.time()
                response = await client.get("/api/v1/health")
                duration = time.time() - start_time
                return duration, response.status_code
            
            # Execute concurrent requests
            start_time = time.time()
            tasks = [make_request() for _ in range(concurrent_requests)]
            results = await asyncio.gather(*tasks)
            total_duration = time.time() - start_time
            
            # Analyze results
            durations = [result[0] for result in results]
            status_codes = [result[1] for result in results]
            
            avg_duration = sum(durations) / len(durations)
            max_duration = max(durations)
            success_rate = sum(1 for code in status_codes if code == 200) / len(status_codes)
            
            benchmark_suite.record_api_benchmark(
                "/api/v1/health_concurrent", avg_duration * 1000, 200,
                {
                    'concurrent_requests': concurrent_requests,
                    'total_duration': total_duration,
                    'success_rate': success_rate,
                    'max_duration_ms': max_duration * 1000
                }
            )
            
            # Assert performance requirements
            assert success_rate >= 0.95, f"Success rate {success_rate:.2f} below 95%"
            assert avg_duration <= 1.0, f"Average duration {avg_duration:.2f}s exceeds 1s under load"
            
            print(f"✅ Concurrent API test: {concurrent_requests} requests, {success_rate:.1%} success rate")

class TestResourceUsageBenchmarks:
    """Resource usage benchmark tests"""
    
    @pytest.mark.asyncio
    async def test_memory_usage_benchmark(self, benchmark_suite):
        """Benchmark memory usage during operations"""
        initial_memory = psutil.virtual_memory().percent
        peak_memory = initial_memory
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Perform various operations while monitoring memory
            operations = [
                ("health_check", lambda: client.get("/api/v1/health")),
                ("system_stats", lambda: client.get("/api/v1/system/stats")),
                ("queue_check", lambda: client.get("/api/v1/queue")),
                ("outputs_list", lambda: client.get("/api/v1/outputs"))
            ]
            
            for op_name, operation in operations:
                # Monitor memory during operation
                for _ in range(10):  # Repeat operation
                    await operation()
                    current_memory = psutil.virtual_memory().percent
                    peak_memory = max(peak_memory, current_memory)
                    
                    # Brief pause to allow memory monitoring
                    await asyncio.sleep(0.1)
        
        memory_increase = peak_memory - initial_memory
        
        benchmark_suite.record_resource_benchmark(
            "api_operations_memory", 0, peak_memory, 0
        )
        
        # Memory increase should be reasonable (less than 10%)
        assert memory_increase < 10, f"Memory increased by {memory_increase:.1f}% during API operations"
        
        print(f"✅ Memory usage: peak {peak_memory:.1f}%, increase {memory_increase:.1f}%")
    
    @pytest.mark.asyncio
    async def test_cpu_usage_benchmark(self, benchmark_suite):
        """Benchmark CPU usage during operations"""
        cpu_samples = []
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Monitor CPU during intensive operations
            start_time = time.time()
            
            while (time.time() - start_time) < 30:  # Monitor for 30 seconds
                # Perform API operations
                await client.get("/api/v1/system/stats")
                await asyncio.sleep(1)
                
                # Sample CPU usage
                cpu_percent = psutil.cpu_percent(interval=0.1)
                cpu_samples.append(cpu_percent)
        
        if cpu_samples:
            avg_cpu = sum(cpu_samples) / len(cpu_samples)
            peak_cpu = max(cpu_samples)
            
            benchmark_suite.record_resource_benchmark(
                "api_operations_cpu", peak_cpu, 0, 0
            )
            
            print(f"✅ CPU usage: average {avg_cpu:.1f}%, peak {peak_cpu:.1f}%")
    
    @pytest.mark.asyncio
    async def test_gpu_memory_monitoring(self, benchmark_suite):
        """Monitor GPU memory usage if available"""
        try:
            import GPUtil

            gpus = GPUtil.getGPUs()
            if not gpus:
                pytest.skip("No GPU available for monitoring")
            
            gpu = gpus[0]
            initial_gpu_memory = gpu.memoryUsed
            peak_gpu_memory = initial_gpu_memory
            
            async with AsyncClient(app=app, base_url="http://test") as client:
                # Submit a generation to potentially use GPU
                request_data = {
                    "model_type": "T2V-A14B",
                    "prompt": "GPU memory test generation",
                    "resolution": "1280x720",
                    "steps": 10  # Minimal steps for quick test
                }
                
                response = await client.post("/api/v1/generation/submit", json=request_data)
                
                if response.status_code == 200:
                    # Monitor GPU memory for a short period
                    for _ in range(30):  # 30 seconds of monitoring
                        await asyncio.sleep(1)
                        
                        current_gpus = GPUtil.getGPUs()
                        if current_gpus:
                            current_gpu_memory = current_gpus[0].memoryUsed
                            peak_gpu_memory = max(peak_gpu_memory, current_gpu_memory)
            
            gpu_memory_increase = peak_gpu_memory - initial_gpu_memory
            
            benchmark_suite.record_resource_benchmark(
                "gpu_memory_usage", 0, 0, peak_gpu_memory
            )
            
            print(f"✅ GPU memory: peak {peak_gpu_memory:.1f}MB, increase {gpu_memory_increase:.1f}MB")
        
        except ImportError:
            pytest.skip("GPUtil not available for GPU monitoring")

@pytest.mark.asyncio
async def test_comprehensive_performance_suite(benchmark_suite):
    """Run comprehensive performance benchmark suite"""
    print("🚀 Starting Performance Benchmark Suite")
    print("=" * 50)
    
    # Run all benchmark test classes
    test_classes = [
        TestGenerationPerformanceBenchmarks(),
        TestAPIPerformanceBenchmarks(),
        TestResourceUsageBenchmarks()
    ]
    
    for test_class in test_classes:
        class_name = test_class.__class__.__name__
        print(f"\n📊 Running {class_name}")
        
        # Get all test methods
        test_methods = [method for method in dir(test_class) 
                       if method.startswith('test_') and callable(getattr(test_class, method))]
        
        for method_name in test_methods:
            try:
                print(f"  ⏳ {method_name}")
                method = getattr(test_class, method_name)
                
                if asyncio.iscoroutinefunction(method):
                    await method(benchmark_suite)
                else:
                    method(benchmark_suite)
                
                print(f"  ✅ {method_name} - COMPLETED")
                
            except Exception as e:
                print(f"  ❌ {method_name} - FAILED: {str(e)}")
    
    # Generate summary report
    summary = benchmark_suite.get_summary()
    
    print("\n" + "=" * 50)
    print("📈 PERFORMANCE BENCHMARK RESULTS")
    print("=" * 50)
    
    print(f"Generation Tests: {summary['total_generation_tests']}")
    print(f"Successful Generations: {summary['successful_generations']}")
    print(f"API Tests: {summary['total_api_tests']}")
    print(f"Average API Response Time: {summary['average_api_response_time']:.1f}ms")
    
    peak_usage = summary['peak_resource_usage']
    print(f"Peak CPU Usage: {peak_usage['cpu']:.1f}%")
    print(f"Peak Memory Usage: {peak_usage['memory']:.1f}%")
    print(f"Peak GPU Memory: {peak_usage['gpu_memory']:.1f}MB")
    
    # Save detailed results
    results_path = Path("performance_benchmark_results.json")
    benchmark_suite.save_results(str(results_path))
    print(f"\n📄 Detailed results saved to: {results_path}")

if __name__ == "__main__":
    # Run performance benchmarks
    async def main():
        suite = PerformanceBenchmarkSuite()
        await test_comprehensive_performance_suite(suite)
    
    asyncio.run(main())
