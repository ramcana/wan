"""
Performance benchmark tests for the Server Startup Management System.

Tests startup time performance, resource usage, and scalability limits
to ensure the system meets performance requirements.
"""

import pytest
import time
import threading
import psutil
import gc
from pathlib import Path
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import mock StartupManager for testing
from .mock_startup_manager import StartupManager
from startup_manager.config import StartupConfig
from startup_manager.port_manager import PortManager
from startup_manager.process_manager import ProcessManager
from startup_manager.environment_validator import EnvironmentValidator
from startup_manager.recovery_engine import RecoveryEngine


class PerformanceTimer:
    """Context manager for timing operations."""
    
    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.duration = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time


class ResourceMonitor:
    """Monitor system resource usage during tests."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.initial_memory = None
        self.peak_memory = None
        self.initial_cpu = None
        self.cpu_samples = []
        self.monitoring = False
        self.monitor_thread = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.monitoring:
            self.stop_monitoring()
    
    def start_monitoring(self):
        """Start resource monitoring."""
        self.initial_memory = self.process.memory_info().rss
        self.peak_memory = self.initial_memory
        self.initial_cpu = self.process.cpu_percent()
        self.cpu_samples = []
        self.monitoring = True
        
        def monitor_loop():
            while self.monitoring:
                try:
                    memory = self.process.memory_info().rss
                    cpu = self.process.cpu_percent()
                    
                    if memory > self.peak_memory:
                        self.peak_memory = memory
                    
                    self.cpu_samples.append(cpu)
                    time.sleep(0.1)
                except:
                    break
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop resource monitoring and return results."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
        
        final_memory = self.process.memory_info().rss
        avg_cpu = sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else 0
        
        return {
            "initial_memory_mb": self.initial_memory / (1024 * 1024),
            "final_memory_mb": final_memory / (1024 * 1024),
            "peak_memory_mb": self.peak_memory / (1024 * 1024),
            "memory_growth_mb": (final_memory - self.initial_memory) / (1024 * 1024),
            "average_cpu_percent": avg_cpu,
            "peak_cpu_percent": max(self.cpu_samples) if self.cpu_samples else 0
        }


class TestStartupPerformance:
    """Test startup performance benchmarks."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = StartupConfig()
        self.startup_manager = StartupManager(self.config)
    
    def test_cold_startup_performance(self):
        """Test cold startup performance (first run)."""
        with ResourceMonitor() as monitor:
            monitor.start_monitoring()
            
            with PerformanceTimer("cold_startup") as timer:
                with patch.object(self.startup_manager.environment_validator, 'validate_all') as mock_validate:
                    mock_validate.return_value = Mock(is_valid=True, issues=[], warnings=[])
                    
                    with patch.object(self.startup_manager.port_manager, 'allocate_ports') as mock_allocate:
                        from scripts.startup_manager.port_manager import PortAllocation
                        mock_allocate.return_value = PortAllocation(
                            backend=8000, frontend=3000,
                            conflicts_resolved=[], alternative_ports_used=False
                        )
                        
                        with patch.object(self.startup_manager.process_manager, 'start_backend') as mock_start_backend:
                            with patch.object(self.startup_manager.process_manager, 'start_frontend') as mock_start_frontend:
                                from scripts.startup_manager.process_manager import ProcessResult, ProcessInfo
                                
                                # Simulate realistic startup delays
                                def delayed_backend(*args, **kwargs):
                                    time.sleep(0.2)  # 200ms backend startup
                                    return ProcessResult.success_result(
                                        ProcessInfo(name="backend", port=8000, pid=1234)
                                    )
                                
                                def delayed_frontend(*args, **kwargs):
                                    time.sleep(0.3)  # 300ms frontend startup
                                    return ProcessResult.success_result(
                                        ProcessInfo(name="frontend", port=3000, pid=5678)
                                    )
                                
                                mock_start_backend.side_effect = delayed_backend
                                mock_start_frontend.side_effect = delayed_frontend
                                
                                result = self.startup_manager.start_servers()
            
            resources = monitor.stop_monitoring()
            
            # Performance assertions
            assert result.success is True
            assert timer.duration < 5.0  # Should complete within 5 seconds
            assert resources["memory_growth_mb"] < 50  # Memory growth should be reasonable
            assert resources["peak_cpu_percent"] < 80  # CPU usage should be reasonable
            
            print(f"Cold startup time: {timer.duration:.2f}s")
            print(f"Memory usage: {resources['memory_growth_mb']:.1f}MB")
            print(f"Peak CPU: {resources['peak_cpu_percent']:.1f}%")
    
    def test_warm_startup_performance(self):
        """Test warm startup performance (subsequent runs)."""
        # First run to "warm up" the system
        with patch.object(self.startup_manager.environment_validator, 'validate_all') as mock_validate:
            mock_validate.return_value = Mock(is_valid=True, issues=[], warnings=[])
            self.startup_manager.start_servers()
        
        # Now test warm startup
        with ResourceMonitor() as monitor:
            monitor.start_monitoring()
            
            with PerformanceTimer("warm_startup") as timer:
                with patch.object(self.startup_manager.environment_validator, 'validate_all') as mock_validate:
                    mock_validate.return_value = Mock(is_valid=True, issues=[], warnings=[])
                    
                    with patch.object(self.startup_manager.port_manager, 'allocate_ports') as mock_allocate:
                        from scripts.startup_manager.port_manager import PortAllocation
                        mock_allocate.return_value = PortAllocation(
                            backend=8001, frontend=3001,
                            conflicts_resolved=[], alternative_ports_used=False
                        )
                        
                        with patch.object(self.startup_manager.process_manager, 'start_backend') as mock_start_backend:
                            with patch.object(self.startup_manager.process_manager, 'start_frontend') as mock_start_frontend:
                                from scripts.startup_manager.process_manager import ProcessResult, ProcessInfo
                                
                                # Warm startup should be faster
                                def fast_backend(*args, **kwargs):
                                    time.sleep(0.1)  # 100ms backend startup
                                    return ProcessResult.success_result(
                                        ProcessInfo(name="backend", port=8001, pid=1235)
                                    )
                                
                                def fast_frontend(*args, **kwargs):
                                    time.sleep(0.15)  # 150ms frontend startup
                                    return ProcessResult.success_result(
                                        ProcessInfo(name="frontend", port=3001, pid=5679)
                                    )
                                
                                mock_start_backend.side_effect = fast_backend
                                mock_start_frontend.side_effect = fast_frontend
                                
                                result = self.startup_manager.start_servers()
            
            resources = monitor.stop_monitoring()
            
            # Warm startup should be faster
            assert result.success is True
            assert timer.duration < 3.0  # Should be faster than cold startup
            assert resources["memory_growth_mb"] < 30  # Less memory growth
            
            print(f"Warm startup time: {timer.duration:.2f}s")
            print(f"Memory usage: {resources['memory_growth_mb']:.1f}MB")
    
    def test_environment_validation_performance(self):
        """Test environment validation performance."""
        validator = EnvironmentValidator()
        
        with PerformanceTimer("environment_validation") as timer:
            with ResourceMonitor() as monitor:
                monitor.start_monitoring()
                
                # Run validation multiple times to test consistency
                results = []
                for _ in range(10):
                    result = validator.validate_all()
                    results.append(result)
                
                resources = monitor.stop_monitoring()
        
        # Performance assertions
        assert timer.duration < 2.0  # Should complete within 2 seconds
        assert resources["memory_growth_mb"] < 10  # Minimal memory growth
        assert len(results) == 10
        
        # All results should be consistent
        for result in results:
            assert result is not None
        
        print(f"Environment validation time: {timer.duration:.2f}s for 10 runs")
        print(f"Average per validation: {timer.duration/10:.3f}s")
    
    def test_port_scanning_performance(self):
        """Test port scanning performance."""
        port_manager = PortManager()
        
        with PerformanceTimer("port_scanning") as timer:
            with ResourceMonitor() as monitor:
                monitor.start_monitoring()
                
                # Scan a range of ports
                available_ports = port_manager.find_available_ports_in_range((60000, 60100))
                
                resources = monitor.stop_monitoring()
        
        # Performance assertions
        assert timer.duration < 5.0  # Should complete within 5 seconds
        assert resources["memory_growth_mb"] < 5  # Minimal memory growth
        assert isinstance(available_ports, list)
        
        print(f"Port scanning time: {timer.duration:.2f}s for 100 ports")
        print(f"Found {len(available_ports)} available ports")
    
    def test_recovery_engine_performance(self):
        """Test recovery engine performance under load."""
        recovery_engine = RecoveryEngine()
        
        # Create various error scenarios
        from scripts.startup_manager.recovery_engine import StartupError, ErrorType
        errors = [
            StartupError(ErrorType.PORT_CONFLICT, f"Port {8000+i} in use") 
            for i in range(20)
        ]
        
        with PerformanceTimer("recovery_processing") as timer:
            with ResourceMonitor() as monitor:
                monitor.start_monitoring()
                
                results = []
                for error in errors:
                    result = recovery_engine.attempt_recovery(error, {"port": 8000 + len(results)})
                    results.append(result)
                
                resources = monitor.stop_monitoring()
        
        # Performance assertions
        assert timer.duration < 10.0  # Should handle 20 errors within 10 seconds
        assert resources["memory_growth_mb"] < 20  # Reasonable memory growth
        assert len(results) == 20
        
        print(f"Recovery processing time: {timer.duration:.2f}s for 20 errors")
        print(f"Average per error: {timer.duration/20:.3f}s")


class TestScalabilityLimits:
    """Test system scalability limits."""
    
    def test_concurrent_startup_scalability(self):
        """Test scalability with concurrent startup attempts."""
        max_concurrent = 10
        results = []
        threads = []
        
        def startup_worker(worker_id):
            config = StartupConfig()
            manager = StartupManager(config)
            
            with patch.object(manager.environment_validator, 'validate_all') as mock_validate:
                mock_validate.return_value = Mock(is_valid=True, issues=[], warnings=[])
                
                with patch.object(manager.port_manager, 'allocate_ports') as mock_allocate:
                    from scripts.startup_manager.port_manager import PortAllocation
                    mock_allocate.return_value = PortAllocation(
                        backend=8000 + worker_id, frontend=3000 + worker_id,
                        conflicts_resolved=[], alternative_ports_used=False
                    )
                    
                    with patch.object(manager.process_manager, 'start_backend') as mock_start_backend:
                        with patch.object(manager.process_manager, 'start_frontend') as mock_start_frontend:
                            from scripts.startup_manager.process_manager import ProcessResult, ProcessInfo
                            
                            mock_start_backend.return_value = ProcessResult.success_result(
                                ProcessInfo(name="backend", port=8000 + worker_id, pid=1234 + worker_id)
                            )
                            mock_start_frontend.return_value = ProcessResult.success_result(
                                ProcessInfo(name="frontend", port=3000 + worker_id, pid=5678 + worker_id)
                            )
                            
                            start_time = time.perf_counter()
                            result = manager.start_servers()
                            end_time = time.perf_counter()
                            
                            results.append({
                                "worker_id": worker_id,
                                "success": result.success,
                                "duration": end_time - start_time
                            })
        
        with ResourceMonitor() as monitor:
            monitor.start_monitoring()
            
            with PerformanceTimer("concurrent_startups") as timer:
                # Start all workers
                for i in range(max_concurrent):
                    thread = threading.Thread(target=startup_worker, args=(i,))
                    threads.append(thread)
                    thread.start()
                
                # Wait for all to complete
                for thread in threads:
                    thread.join(timeout=30)
            
            resources = monitor.stop_monitoring()
        
        # Scalability assertions
        assert len(results) == max_concurrent
        assert timer.duration < 30.0  # Should complete within 30 seconds
        assert resources["memory_growth_mb"] < 100  # Memory growth should be bounded
        
        successful_results = [r for r in results if r["success"]]
        success_rate = len(successful_results) / len(results)
        
        assert success_rate >= 0.8  # At least 80% success rate under load
        
        avg_duration = sum(r["duration"] for r in results) / len(results)
        max_duration = max(r["duration"] for r in results)
        
        print(f"Concurrent startup test: {max_concurrent} workers")
        print(f"Success rate: {success_rate:.1%}")
        print(f"Average duration: {avg_duration:.2f}s")
        print(f"Max duration: {max_duration:.2f}s")
        print(f"Total time: {timer.duration:.2f}s")
    
    def test_memory_usage_scalability(self):
        """Test memory usage scalability with multiple instances."""
        instances = []
        memory_samples = []
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        try:
            # Create multiple instances
            for i in range(20):
                config = StartupConfig()
                manager = StartupManager(config)
                instances.append(manager)
                
                # Sample memory usage
                current_memory = process.memory_info().rss
                memory_samples.append(current_memory - initial_memory)
                
                # Force garbage collection periodically
                if i % 5 == 0:
                    gc.collect()
        
        finally:
            # Clean up instances
            instances.clear()
            gc.collect()
        
        final_memory = process.memory_info().rss
        total_growth = final_memory - initial_memory
        
        # Memory scalability assertions
        assert total_growth < 200 * 1024 * 1024  # Less than 200MB total growth
        
        # Memory growth should be roughly linear
        if len(memory_samples) > 10:
            early_avg = sum(memory_samples[:5]) / 5
            late_avg = sum(memory_samples[-5:]) / 5
            growth_rate = (late_avg - early_avg) / 15  # Per instance
            
            assert growth_rate < 10 * 1024 * 1024  # Less than 10MB per instance
        
        print(f"Memory scalability test: 20 instances")
        print(f"Total memory growth: {total_growth / (1024*1024):.1f}MB")
        print(f"Average per instance: {total_growth / (20 * 1024*1024):.1f}MB")
    
    def test_port_range_scalability(self):
        """Test scalability with large port ranges."""
        port_manager = PortManager()
        
        # Test with increasingly large port ranges
        range_sizes = [100, 500, 1000, 2000]
        results = []
        
        for range_size in range_sizes:
            start_port = 60000
            end_port = start_port + range_size
            
            with PerformanceTimer(f"port_range_{range_size}") as timer:
                with ResourceMonitor() as monitor:
                    monitor.start_monitoring()
                    
                    available_ports = port_manager.find_available_ports_in_range(
                        (start_port, end_port)
                    )
                    
                    resources = monitor.stop_monitoring()
            
            results.append({
                "range_size": range_size,
                "duration": timer.duration,
                "found_ports": len(available_ports),
                "memory_growth": resources["memory_growth_mb"]
            })
            
            # Performance should scale reasonably
            assert timer.duration < range_size * 0.01  # Less than 10ms per port
            assert resources["memory_growth_mb"] < range_size * 0.001  # Less than 1KB per port
        
        # Check that performance scales reasonably
        for i in range(1, len(results)):
            prev_result = results[i-1]
            curr_result = results[i]
            
            size_ratio = curr_result["range_size"] / prev_result["range_size"]
            time_ratio = curr_result["duration"] / prev_result["duration"]
            
            # Time should scale sub-linearly (better than O(n))
            assert time_ratio < size_ratio * 1.5
        
        print("Port range scalability results:")
        for result in results:
            print(f"  Range {result['range_size']}: {result['duration']:.2f}s, "
                  f"{result['found_ports']} ports, {result['memory_growth']:.1f}MB")


class TestResourceConstraints:
    """Test behavior under resource constraints."""
    
    def test_low_memory_performance(self):
        """Test performance under simulated low memory conditions."""
        # This test simulates low memory by creating memory pressure
        memory_hogs = []
        
        try:
            # Create some memory pressure (but not too much to crash the test)
            for _ in range(10):
                memory_hogs.append(bytearray(10 * 1024 * 1024))  # 10MB each
            
            config = StartupConfig()
            manager = StartupManager(config)
            
            with PerformanceTimer("low_memory_startup") as timer:
                with patch.object(manager.environment_validator, 'validate_all') as mock_validate:
                    mock_validate.return_value = Mock(is_valid=True, issues=[], warnings=[])
                    
                    result = manager.start_servers()
            
            # Should still work under memory pressure, just slower
            assert result.success is True
            assert timer.duration < 15.0  # Allow more time under memory pressure
            
        finally:
            # Clean up memory
            memory_hogs.clear()
            gc.collect()
    
    def test_high_cpu_load_performance(self):
        """Test performance under high CPU load."""
        # Create CPU load in background threads
        cpu_workers = []
        stop_cpu_load = threading.Event()
        
        def cpu_intensive_work():
            while not stop_cpu_load.is_set():
                # Busy work
                sum(i * i for i in range(1000))
        
        try:
            # Start CPU-intensive background work
            for _ in range(psutil.cpu_count()):
                worker = threading.Thread(target=cpu_intensive_work, daemon=True)
                worker.start()
                cpu_workers.append(worker)
            
            time.sleep(0.5)  # Let CPU load build up
            
            config = StartupConfig()
            manager = StartupManager(config)
            
            with PerformanceTimer("high_cpu_startup") as timer:
                with patch.object(manager.environment_validator, 'validate_all') as mock_validate:
                    mock_validate.return_value = Mock(is_valid=True, issues=[], warnings=[])
                    
                    result = manager.start_servers()
            
            # Should still work under CPU load, just slower
            assert result.success is True
            assert timer.duration < 20.0  # Allow more time under CPU load
            
        finally:
            # Stop CPU load
            stop_cpu_load.set()
            for worker in cpu_workers:
                worker.join(timeout=1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])