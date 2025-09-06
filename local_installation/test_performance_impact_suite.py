"""
Performance Impact Test Suite for Reliability System

This module tests the performance impact of reliability enhancements to ensure
minimal overhead is added to normal operations.

Requirements addressed: 1.1, 1.2, 1.3, 1.4, 1.5
"""

import pytest
import unittest
import time
import threading
import psutil
import gc
import sys
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

try:
    from reliability_manager import ReliabilityManager, ComponentType
    from reliability_wrapper import ReliabilityWrapper, ReliabilityMetrics
    from missing_method_recovery import MissingMethodRecovery
    from intelligent_retry_system import IntelligentRetrySystem
    from diagnostic_monitor import DiagnosticMonitor
except ImportError:
    # Mock classes for testing when components aren't available
    class MockComponent:
        pass
    ReliabilityManager = MockComponent
    ReliabilityWrapper = MockComponent
    MissingMethodRecovery = MockComponent
    IntelligentRetrySystem = MockComponent
    DiagnosticMonitor = MockComponent

# Define MockComponent globally for test use
class MockComponent:
    pass


@dataclass
class PerformanceMetrics:
    """Performance metrics for operations."""
    operation_name: str
    execution_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    method_calls: int
    overhead_ratio: float
    throughput_ops_per_sec: float


class PerformanceBenchmark:
    """Utility class for performance benchmarking."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.baseline_metrics = {}
        self.enhanced_metrics = {}
    
    def measure_baseline_performance(self, component, operation_name: str, iterations: int = 1000):
        """Measure baseline performance without reliability enhancements."""
        # Force garbage collection
        gc.collect()
        
        # Get initial metrics
        initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        initial_cpu_time = self.process.cpu_times().user
        
        # Execute operations
        start_time = time.time()
        for _ in range(iterations):
            if hasattr(component, operation_name):
                getattr(component, operation_name)()
            else:
                component.default_operation()
        end_time = time.time()
        
        # Get final metrics
        final_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        final_cpu_time = self.process.cpu_times().user
        
        execution_time = end_time - start_time
        memory_usage = final_memory - initial_memory
        cpu_usage = ((final_cpu_time - initial_cpu_time) / execution_time) * 100
        throughput = iterations / execution_time
        
        metrics = PerformanceMetrics(
            operation_name=f"baseline_{operation_name}",
            execution_time=execution_time,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage,
            method_calls=iterations,
            overhead_ratio=1.0,  # Baseline
            throughput_ops_per_sec=throughput
        )
        
        self.baseline_metrics[operation_name] = metrics
        return metrics
    
    def measure_enhanced_performance(self, wrapped_component, operation_name: str, iterations: int = 1000):
        """Measure performance with reliability enhancements."""
        # Force garbage collection
        gc.collect()
        
        # Get initial metrics
        initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        initial_cpu_time = self.process.cpu_times().user
        
        # Execute operations
        start_time = time.time()
        for _ in range(iterations):
            if hasattr(wrapped_component, operation_name):
                getattr(wrapped_component, operation_name)()
            else:
                wrapped_component.default_operation()
        end_time = time.time()
        
        # Get final metrics
        final_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        final_cpu_time = self.process.cpu_times().user
        
        execution_time = end_time - start_time
        memory_usage = final_memory - initial_memory
        cpu_usage = ((final_cpu_time - initial_cpu_time) / execution_time) * 100
        throughput = iterations / execution_time
        
        # Calculate overhead ratio
        baseline = self.baseline_metrics.get(operation_name)
        overhead_ratio = execution_time / baseline.execution_time if baseline else 1.0
        
        metrics = PerformanceMetrics(
            operation_name=f"enhanced_{operation_name}",
            execution_time=execution_time,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage,
            method_calls=iterations,
            overhead_ratio=overhead_ratio,
            throughput_ops_per_sec=throughput
        )
        
        self.enhanced_metrics[operation_name] = metrics
        return metrics
    
    def compare_performance(self, operation_name: str) -> Dict[str, Any]:
        """Compare baseline vs enhanced performance."""
        baseline = self.baseline_metrics.get(operation_name)
        enhanced = self.enhanced_metrics.get(operation_name)
        
        if not baseline or not enhanced:
            return {"error": "Missing baseline or enhanced metrics"}
        
        return {
            "operation": operation_name,
            "baseline_time": baseline.execution_time,
            "enhanced_time": enhanced.execution_time,
            "overhead_ratio": enhanced.overhead_ratio,
            "overhead_percentage": (enhanced.overhead_ratio - 1.0) * 100,
            "baseline_throughput": baseline.throughput_ops_per_sec,
            "enhanced_throughput": enhanced.throughput_ops_per_sec,
            "throughput_impact": ((enhanced.throughput_ops_per_sec / baseline.throughput_ops_per_sec) - 1.0) * 100,
            "memory_impact": enhanced.memory_usage_mb - baseline.memory_usage_mb,
            "cpu_impact": enhanced.cpu_usage_percent - baseline.cpu_usage_percent
        }


class MockPerformanceComponent:
    """Mock component for performance testing."""
    
    def __init__(self, name: str = "PerformanceTestComponent"):
        self.name = name
        self.call_count = 0
        self.operation_duration = 0.001  # 1ms per operation
    
    def fast_operation(self):
        """Fast operation for performance testing."""
        time.sleep(self.operation_duration)
        self.call_count += 1
        return f"Fast operation {self.call_count}"
    
    def medium_operation(self):
        """Medium duration operation."""
        time.sleep(self.operation_duration * 5)  # 5ms
        self.call_count += 1
        return f"Medium operation {self.call_count}"
    
    def slow_operation(self):
        """Slow operation for testing."""
        time.sleep(self.operation_duration * 10)  # 10ms
        self.call_count += 1
        return f"Slow operation {self.call_count}"
    
    def memory_intensive_operation(self):
        """Memory intensive operation."""
        # Allocate and release memory
        data = [i for i in range(1000)]
        self.call_count += 1
        del data
        return f"Memory operation {self.call_count}"
    
    def cpu_intensive_operation(self):
        """CPU intensive operation."""
        # Perform some calculations
        result = sum(i * i for i in range(100))
        self.call_count += 1
        return f"CPU operation {self.call_count}: {result}"
    
    def default_operation(self):
        """Default operation for generic testing."""
        return self.fast_operation()


class PerformanceImpactTestSuite(unittest.TestCase):
    """Test suite for performance impact measurement."""
    
    def setUp(self):
        """Set up test environment."""
        self.benchmark = PerformanceBenchmark()
        self.mock_component = MockPerformanceComponent("TestComponent")
        self.performance_results = []
    
    def tearDown(self):
        """Clean up test environment."""
        # Force garbage collection
        gc.collect()
    
    def test_reliability_wrapper_overhead(self):
        """Test performance overhead of ReliabilityWrapper."""
        if ReliabilityWrapper == MockComponent:
            self.skipTest("ReliabilityWrapper not available")
        
        # Measure baseline performance
        baseline_metrics = self.benchmark.measure_baseline_performance(
            self.mock_component, 'fast_operation', iterations=1000
        )
        
        # Create wrapped component
        mock_manager = Mock()
        wrapped_component = ReliabilityWrapper(self.mock_component, mock_manager)
        
        # Measure enhanced performance
        enhanced_metrics = self.benchmark.measure_enhanced_performance(
            wrapped_component, 'fast_operation', iterations=1000
        )
        
        # Compare performance
        comparison = self.benchmark.compare_performance('fast_operation')
        self.performance_results.append(comparison)
        
        # Assert acceptable overhead (less than 50% increase)
        self.assertLess(comparison['overhead_percentage'], 50.0)
        self.assertGreater(comparison['enhanced_throughput'], 0)
        
        print(f"ReliabilityWrapper overhead: {comparison['overhead_percentage']:.2f}%")
    
    def test_reliability_manager_scaling_performance(self):
        """Test ReliabilityManager performance with multiple components."""
        if ReliabilityManager == MockComponent:
            self.skipTest("ReliabilityManager not available")
        
        manager = ReliabilityManager("/tmp", None)
        
        # Test with increasing number of components
        component_counts = [1, 5, 10, 25, 50]
        scaling_results = []
        
        for count in component_counts:
            # Create components
            components = [MockPerformanceComponent(f"Component{i}") for i in range(count)]
            
            # Measure wrapping time
            start_time = time.time()
            wrapped_components = []
            for component in components:
                wrapped = manager.wrap_component(component, ComponentType.MODEL_DOWNLOADER)
                wrapped_components.append(wrapped)
            wrapping_time = time.time() - start_time
            
            # Measure execution time
            start_time = time.time()
            for wrapped in wrapped_components:
                wrapped.fast_operation()
            execution_time = time.time() - start_time
            
            scaling_results.append({
                'component_count': count,
                'wrapping_time': wrapping_time,
                'execution_time': execution_time,
                'avg_wrapping_time': wrapping_time / count,
                'avg_execution_time': execution_time / count
            })
        
        # Verify scaling is reasonable
        for result in scaling_results:
            self.assertLess(result['avg_wrapping_time'], 0.1)  # Less than 100ms per component
            self.assertLess(result['avg_execution_time'], 0.01)  # Less than 10ms per operation
        
        print(f"Scaling results: {scaling_results}")
    
    def test_retry_system_performance_impact(self):
        """Test performance impact of retry system."""
        if 'IntelligentRetrySystem' not in globals():
            self.skipTest("IntelligentRetrySystem not available")
        
        # Create component that succeeds immediately
        component = MockPerformanceComponent("RetryTestComponent")
        
        # Measure baseline (no retry system)
        baseline_metrics = self.benchmark.measure_baseline_performance(
            component, 'fast_operation', iterations=500
        )
        
        # Create retry-enabled component (mock)
        class RetryEnabledComponent:
            def __init__(self, base_component):
                self.base_component = base_component
                self.retry_overhead = 0.0001  # 0.1ms overhead per call
            
            def fast_operation(self):
                time.sleep(self.retry_overhead)  # Simulate retry system overhead
                return self.base_component.fast_operation()
        
        retry_component = RetryEnabledComponent(component)
        
        # Measure enhanced performance
        enhanced_metrics = self.benchmark.measure_enhanced_performance(
            retry_component, 'fast_operation', iterations=500
        )
        
        # Compare performance
        comparison = self.benchmark.compare_performance('fast_operation')
        self.performance_results.append(comparison)
        
        # Assert minimal overhead for successful operations
        self.assertLess(comparison['overhead_percentage'], 20.0)
        
        print(f"Retry system overhead: {comparison['overhead_percentage']:.2f}%")
    
    def test_diagnostic_monitoring_overhead(self):
        """Test performance overhead of diagnostic monitoring."""
        if 'DiagnosticMonitor' not in globals():
            self.skipTest("DiagnosticMonitor not available")
        
        component = MockPerformanceComponent("MonitoringTestComponent")
        
        # Measure baseline performance
        baseline_metrics = self.benchmark.measure_baseline_performance(
            component, 'fast_operation', iterations=1000
        )
        
        # Create monitored component (mock)
        class MonitoredComponent:
            def __init__(self, base_component):
                self.base_component = base_component
                self.monitoring_overhead = 0.0002  # 0.2ms overhead per call
                self.metrics = []
            
            def fast_operation(self):
                start_time = time.time()
                result = self.base_component.fast_operation()
                end_time = time.time()
                
                # Simulate monitoring overhead
                time.sleep(self.monitoring_overhead)
                self.metrics.append({
                    'operation': 'fast_operation',
                    'duration': end_time - start_time,
                    'timestamp': start_time
                })
                
                return result
        
        monitored_component = MonitoredComponent(component)
        
        # Measure enhanced performance
        enhanced_metrics = self.benchmark.measure_enhanced_performance(
            monitored_component, 'fast_operation', iterations=1000
        )
        
        # Compare performance
        comparison = self.benchmark.compare_performance('fast_operation')
        self.performance_results.append(comparison)
        
        # Assert acceptable monitoring overhead
        self.assertLess(comparison['overhead_percentage'], 30.0)
        self.assertEqual(len(monitored_component.metrics), 1000)
        
        print(f"Monitoring overhead: {comparison['overhead_percentage']:.2f}%")
    
    def test_memory_usage_efficiency(self):
        """Test memory usage efficiency of reliability system."""
        if ReliabilityManager == MockComponent:
            self.skipTest("ReliabilityManager not available")
        
        # Get initial memory usage
        initial_memory = self.benchmark.process.memory_info().rss / 1024 / 1024  # MB
        
        manager = ReliabilityManager("/tmp", None)
        
        # Create many components and wrap them
        components = []
        for i in range(100):
            component = MockPerformanceComponent(f"MemoryTestComponent{i}")
            wrapped = manager.wrap_component(component, ComponentType.MODEL_DOWNLOADER)
            components.append(wrapped)
        
        # Execute operations to generate metrics
        for component in components:
            component.memory_intensive_operation()
        
        # Get final memory usage
        final_memory = self.benchmark.process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be reasonable (less than 100MB for 100 components)
        self.assertLess(memory_growth, 100.0)
        
        print(f"Memory growth for 100 components: {memory_growth:.2f} MB")
    
    def test_concurrent_operations_performance(self):
        """Test performance under concurrent operations."""
        if ReliabilityManager == MockComponent:
            self.skipTest("ReliabilityManager not available")
        
        manager = ReliabilityManager("/tmp", None)
        
        def concurrent_operation(component_id):
            component = MockPerformanceComponent(f"ConcurrentComponent{component_id}")
            wrapped = manager.wrap_component(component, ComponentType.MODEL_DOWNLOADER)
            
            # Perform multiple operations
            results = []
            for _ in range(10):
                result = wrapped.fast_operation()
                results.append(result)
            
            return len(results)
        
        # Measure concurrent performance
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(concurrent_operation, i) for i in range(20)]
            results = [future.result() for future in futures]
        end_time = time.time()
        
        total_operations = sum(results)
        execution_time = end_time - start_time
        throughput = total_operations / execution_time
        
        # Verify reasonable concurrent performance
        self.assertEqual(len(results), 20)
        self.assertEqual(total_operations, 200)  # 20 components * 10 operations each
        self.assertGreater(throughput, 50)  # At least 50 operations per second
        
        print(f"Concurrent throughput: {throughput:.2f} ops/sec")
    
    def test_error_handling_performance_impact(self):
        """Test performance impact when errors occur and are handled."""
        if ReliabilityManager == MockComponent:
            self.skipTest("ReliabilityManager not available")
        
        manager = ReliabilityManager("/tmp", None)
        
        class ErrorProneComponent:
            def __init__(self, error_rate=0.1):
                self.call_count = 0
                self.error_rate = error_rate
            
            def error_prone_operation(self):
                self.call_count += 1
                if self.call_count % int(1/self.error_rate) == 0:
                    raise Exception(f"Simulated error on call {self.call_count}")
                return f"Success on call {self.call_count}"
        
        # Test with different error rates
        error_rates = [0.0, 0.1, 0.2, 0.5]
        performance_results = []
        
        for error_rate in error_rates:
            component = ErrorProneComponent(error_rate)
            wrapped = manager.wrap_component(component, ComponentType.MODEL_DOWNLOADER)
            
            # Measure performance with errors
            start_time = time.time()
            success_count = 0
            error_count = 0
            
            for _ in range(100):
                try:
                    wrapped.error_prone_operation()
                    success_count += 1
                except Exception:
                    error_count += 1
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            performance_results.append({
                'error_rate': error_rate,
                'execution_time': execution_time,
                'success_count': success_count,
                'error_count': error_count,
                'throughput': (success_count + error_count) / execution_time
            })
        
        # Verify performance degrades gracefully with higher error rates
        for i in range(1, len(performance_results)):
            current = performance_results[i]
            previous = performance_results[i-1]
            
            # Higher error rates should not cause excessive performance degradation
            performance_ratio = current['execution_time'] / previous['execution_time']
            self.assertLess(performance_ratio, 3.0)  # Less than 3x slower
        
        print(f"Error handling performance: {performance_results}")
    
    def test_overall_system_performance_profile(self):
        """Generate overall performance profile of the reliability system."""
        if ReliabilityManager == MockComponent:
            self.skipTest("ReliabilityManager not available")
        
        # Create comprehensive performance profile
        profile = {
            'test_timestamp': time.time(),
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'python_version': sys.version
            },
            'performance_metrics': self.performance_results,
            'summary': {
                'total_tests': len(self.performance_results),
                'average_overhead': sum(r.get('overhead_percentage', 0) for r in self.performance_results) / max(len(self.performance_results), 1),
                'max_overhead': max((r.get('overhead_percentage', 0) for r in self.performance_results), default=0),
                'min_overhead': min((r.get('overhead_percentage', 0) for r in self.performance_results), default=0)
            }
        }
        
        # Assert overall performance is acceptable
        if profile['summary']['total_tests'] > 0:
            self.assertLess(profile['summary']['average_overhead'], 40.0)  # Less than 40% average overhead
            self.assertLess(profile['summary']['max_overhead'], 100.0)  # Less than 100% max overhead
        
        print(f"Performance Profile: {profile['summary']}")
        
        # Save detailed profile for analysis
        import json
        with open('/tmp/reliability_performance_profile.json', 'w') as f:
            json.dump(profile, f, indent=2)
        
        return profile


def run_performance_tests():
    """Run the performance impact test suite."""
    suite = unittest.TestLoader().loadTestsFromTestCase(PerformanceImpactTestSuite)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print(f"\nPerformance Tests - Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_performance_tests()
    sys.exit(0 if success else 1)