"""
Unit tests for ReliabilityWrapper component.
Tests transparent reliability enhancement functionality.
"""

import unittest
import tempfile
import shutil
import time
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.append('scripts')

from reliability_wrapper import ReliabilityWrapper, ReliabilityWrapperFactory, ReliabilityMetrics
from interfaces import InstallationError, ErrorCategory


class MockComponent:
    """Mock component for testing."""
    
    def __init__(self, name="MockComponent"):
        self.name = name
        self.call_count = 0
    
    def working_method(self, value=1):
        """A method that works normally."""
        self.call_count += 1
        return value * 2
    
    def failing_method(self):
        """A method that always fails."""
        self.call_count += 1
        raise ValueError("This method always fails")
    
    def slow_method(self, delay=0.1):
        """A method that takes some time."""
        time.sleep(delay)
        return "completed"


class MockComponentWithMissingMethods:
    """Mock component that's missing some expected methods."""
    
    def __init__(self):
        self.existing_method_called = False
    
    def existing_method(self):
        """A method that exists."""
        self.existing_method_called = True
        return "success"


class TestReliabilityMetrics(unittest.TestCase):
    """Test reliability metrics functionality."""
    
    def setUp(self):
        self.metrics = ReliabilityMetrics()
    
    def test_initialization(self):
        """Test metrics initialization."""
        self.assertEqual(self.metrics.success_count, 0)
        self.assertEqual(self.metrics.failure_count, 0)
        self.assertEqual(self.metrics.total_execution_time, 0.0)
        self.assertEqual(len(self.metrics.method_calls), 0)
        self.assertEqual(len(self.metrics.error_history), 0)
        self.assertEqual(len(self.metrics.recovery_attempts), 0)
    
    def test_record_successful_method_call(self):
        """Test recording successful method calls."""
        self.metrics.record_method_call("test_method", True, 0.5)
        
        self.assertEqual(self.metrics.success_count, 1)
        self.assertEqual(self.metrics.failure_count, 0)
        self.assertEqual(self.metrics.total_execution_time, 0.5)
        
        # Check method-specific stats
        self.assertIn("test_method", self.metrics.method_calls)
        stats = self.metrics.method_calls["test_method"]
        self.assertEqual(stats['total_calls'], 1)
        self.assertEqual(stats['successful_calls'], 1)
        self.assertEqual(stats['failed_calls'], 0)
        self.assertEqual(stats['total_duration'], 0.5)
        self.assertEqual(stats['average_duration'], 0.5)
    
    def test_record_failed_method_call(self):
        """Test recording failed method calls."""
        error = ValueError("Test error")
        self.metrics.record_method_call("test_method", False, 0.3, error)
        
        self.assertEqual(self.metrics.success_count, 0)
        self.assertEqual(self.metrics.failure_count, 1)
        self.assertEqual(self.metrics.total_execution_time, 0.3)
        
        # Check method-specific stats
        stats = self.metrics.method_calls["test_method"]
        self.assertEqual(stats['total_calls'], 1)
        self.assertEqual(stats['successful_calls'], 0)
        self.assertEqual(stats['failed_calls'], 1)
        self.assertEqual(len(stats['errors']), 1)
        self.assertEqual(stats['errors'][0]['error'], "Test error")
        
        # Check error history
        self.assertEqual(len(self.metrics.error_history), 1)
        self.assertEqual(self.metrics.error_history[0]['method'], "test_method")
    
    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        # Initially should be 0
        self.assertEqual(self.metrics.get_success_rate(), 0.0)
        
        # Add some successful calls
        self.metrics.record_method_call("method1", True, 0.1)
        self.metrics.record_method_call("method2", True, 0.2)
        self.assertEqual(self.metrics.get_success_rate(), 1.0)
        
        # Add a failed call
        self.metrics.record_method_call("method3", False, 0.1)
        self.assertAlmostEqual(self.metrics.get_success_rate(), 2/3, places=2)
    
    def test_method_specific_success_rate(self):
        """Test method-specific success rate calculation."""
        # Method that succeeds twice and fails once
        self.metrics.record_method_call("test_method", True, 0.1)
        self.metrics.record_method_call("test_method", True, 0.1)
        self.metrics.record_method_call("test_method", False, 0.1)
        
        self.assertAlmostEqual(self.metrics.get_method_success_rate("test_method"), 2/3, places=2)
        self.assertEqual(self.metrics.get_method_success_rate("nonexistent_method"), 0.0)
    
    def test_recovery_attempt_recording(self):
        """Test recording recovery attempts."""
        self.metrics.record_recovery_attempt("test_method", "retry", True)
        self.metrics.record_recovery_attempt("test_method", "fallback", False)
        
        self.assertEqual(len(self.metrics.recovery_attempts), 2)
        self.assertEqual(self.metrics.recovery_attempts[0]['recovery_type'], "retry")
        self.assertEqual(self.metrics.recovery_attempts[0]['success'], True)
        self.assertEqual(self.metrics.recovery_attempts[1]['recovery_type'], "fallback")
        self.assertEqual(self.metrics.recovery_attempts[1]['success'], False)


class TestReliabilityWrapper(unittest.TestCase):
    """Test ReliabilityWrapper functionality."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.mock_component = MockComponent()
        self.wrapper = ReliabilityWrapper(self.mock_component, self.temp_dir)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test wrapper initialization."""
        self.assertEqual(self.wrapper.component, self.mock_component)
        self.assertEqual(self.wrapper.component_name, "MockComponent")
        self.assertEqual(self.wrapper.installation_path, self.temp_dir)
        self.assertIsNotNone(self.wrapper.error_handler)
        self.assertIsNotNone(self.wrapper.missing_method_recovery)
        self.assertIsNotNone(self.wrapper.metrics)
    
    def test_successful_method_call(self):
        """Test successful method call through wrapper."""
        result = self.wrapper.working_method(5)
        
        self.assertEqual(result, 10)  # 5 * 2
        self.assertEqual(self.mock_component.call_count, 1)
        
        # Check metrics
        metrics = self.wrapper.get_metrics()
        self.assertEqual(metrics['metrics']['total_calls'], 1)
        self.assertEqual(metrics['metrics']['success_rate'], 1.0)
        self.assertIn('working_method', metrics['metrics']['method_stats'])
    
    def test_failed_method_call(self):
        """Test failed method call through wrapper."""
        with self.assertRaises(ValueError):
            self.wrapper.failing_method()
        
        self.assertEqual(self.mock_component.call_count, 1)
        
        # Check metrics
        metrics = self.wrapper.get_metrics()
        self.assertEqual(metrics['metrics']['total_calls'], 1)
        self.assertEqual(metrics['metrics']['success_rate'], 0.0)
        self.assertIn('failing_method', metrics['metrics']['method_stats'])
        self.assertEqual(len(metrics['metrics']['recent_errors']), 1)
    
    def test_performance_tracking(self):
        """Test performance tracking for method calls."""
        # Call a slow method
        result = self.wrapper.slow_method(0.05)  # 50ms delay
        
        self.assertEqual(result, "completed")
        
        # Check that execution time was tracked
        metrics = self.wrapper.get_metrics()
        method_stats = metrics['metrics']['method_stats']['slow_method']
        self.assertGreater(method_stats['total_duration'], 0.04)  # Should be at least 40ms
        self.assertGreater(method_stats['average_duration'], 0.04)
    
    def test_missing_method_handling(self):
        """Test handling of missing methods."""
        # This should trigger the missing method recovery
        with self.assertRaises(AttributeError):
            self.wrapper.nonexistent_method()
        
        # Check that recovery was attempted
        metrics = self.wrapper.get_metrics()
        self.assertGreater(metrics['metrics']['recovery_attempts'], 0)
    
    def test_non_callable_attribute_access(self):
        """Test access to non-callable attributes."""
        # Should return the attribute directly without wrapping
        name = self.wrapper.name
        self.assertEqual(name, "MockComponent")
    
    def test_metrics_reset(self):
        """Test metrics reset functionality."""
        # Generate some metrics
        self.wrapper.working_method(1)
        self.wrapper.working_method(2)
        
        # Verify metrics exist
        metrics_before = self.wrapper.get_metrics()
        self.assertEqual(metrics_before['metrics']['total_calls'], 2)
        
        # Reset metrics
        self.wrapper.reset_metrics()
        
        # Verify metrics are reset
        metrics_after = self.wrapper.get_metrics()
        self.assertEqual(metrics_after['metrics']['total_calls'], 0)
    
    def test_get_original_component(self):
        """Test getting the original wrapped component."""
        original = self.wrapper.get_component()
        self.assertEqual(original, self.mock_component)


class TestReliabilityWrapperFactory(unittest.TestCase):
    """Test ReliabilityWrapperFactory functionality."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.factory = ReliabilityWrapperFactory(self.temp_dir)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test factory initialization."""
        self.assertEqual(self.factory.installation_path, self.temp_dir)
        self.assertIsNotNone(self.factory.error_handler)
        self.assertIsNotNone(self.factory.missing_method_recovery)
        self.assertEqual(len(self.factory.wrapped_components), 0)
    
    def test_wrap_component(self):
        """Test basic component wrapping."""
        mock_component = MockComponent()
        wrapper = self.factory.wrap_component(mock_component)
        
        self.assertIsInstance(wrapper, ReliabilityWrapper)
        self.assertEqual(wrapper.component, mock_component)
        self.assertEqual(len(self.factory.wrapped_components), 1)
    
    def test_wrap_component_idempotent(self):
        """Test that wrapping the same component twice returns the same wrapper."""
        mock_component = MockComponent()
        wrapper1 = self.factory.wrap_component(mock_component)
        wrapper2 = self.factory.wrap_component(mock_component)
        
        self.assertEqual(wrapper1, wrapper2)
        self.assertEqual(len(self.factory.wrapped_components), 1)
    
    def test_specific_component_wrappers(self):
        """Test specific component wrapper methods."""
        mock_downloader = Mock()
        mock_downloader.__class__.__name__ = "ModelDownloader"
        
        mock_manager = Mock()
        mock_manager.__class__.__name__ = "DependencyManager"
        
        mock_engine = Mock()
        mock_engine.__class__.__name__ = "ConfigurationEngine"
        
        mock_validator = Mock()
        mock_validator.__class__.__name__ = "InstallationValidator"
        
        # Test specific wrapper methods
        downloader_wrapper = self.factory.wrap_model_downloader(mock_downloader)
        manager_wrapper = self.factory.wrap_dependency_manager(mock_manager)
        engine_wrapper = self.factory.wrap_configuration_engine(mock_engine)
        validator_wrapper = self.factory.wrap_installation_validator(mock_validator)
        
        self.assertIsInstance(downloader_wrapper, ReliabilityWrapper)
        self.assertIsInstance(manager_wrapper, ReliabilityWrapper)
        self.assertIsInstance(engine_wrapper, ReliabilityWrapper)
        self.assertIsInstance(validator_wrapper, ReliabilityWrapper)
        
        self.assertEqual(len(self.factory.wrapped_components), 4)
    
    def test_get_all_metrics(self):
        """Test getting metrics for all wrapped components."""
        mock_component1 = MockComponent("Component1")
        mock_component2 = MockComponent("Component2")
        
        wrapper1 = self.factory.wrap_component(mock_component1)
        wrapper2 = self.factory.wrap_component(mock_component2)
        
        # Generate some metrics
        wrapper1.working_method(1)
        wrapper2.working_method(2)
        
        all_metrics = self.factory.get_all_metrics()
        self.assertEqual(len(all_metrics), 2)
        
        # Each component should have metrics
        for metrics in all_metrics.values():
            self.assertEqual(metrics['metrics']['total_calls'], 1)
    
    def test_get_component_metrics(self):
        """Test getting metrics for a specific component."""
        mock_component = MockComponent()
        wrapper = self.factory.wrap_component(mock_component)
        
        # Generate some metrics
        wrapper.working_method(1)
        
        # Get metrics for the specific component
        metrics = self.factory.get_component_metrics(mock_component)
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics['metrics']['total_calls'], 1)
        
        # Test with non-wrapped component
        other_component = MockComponent()
        other_metrics = self.factory.get_component_metrics(other_component)
        self.assertIsNone(other_metrics)
    
    def test_reset_all_metrics(self):
        """Test resetting metrics for all wrapped components."""
        mock_component1 = MockComponent("Component1")
        mock_component2 = MockComponent("Component2")
        
        wrapper1 = self.factory.wrap_component(mock_component1)
        wrapper2 = self.factory.wrap_component(mock_component2)
        
        # Generate some metrics
        wrapper1.working_method(1)
        wrapper2.working_method(2)
        
        # Verify metrics exist
        all_metrics_before = self.factory.get_all_metrics()
        for metrics in all_metrics_before.values():
            self.assertEqual(metrics['metrics']['total_calls'], 1)
        
        # Reset all metrics
        self.factory.reset_all_metrics()
        
        # Verify metrics are reset
        all_metrics_after = self.factory.get_all_metrics()
        for metrics in all_metrics_after.values():
            self.assertEqual(metrics['metrics']['total_calls'], 0)
    
    def test_reliability_summary(self):
        """Test overall reliability summary."""
        mock_component1 = MockComponent("Component1")
        mock_component2 = MockComponent("Component2")
        
        wrapper1 = self.factory.wrap_component(mock_component1)
        wrapper2 = self.factory.wrap_component(mock_component2)
        
        # Generate some metrics - mix of success and failure
        wrapper1.working_method(1)  # Success
        wrapper1.working_method(2)  # Success
        
        try:
            wrapper2.failing_method()  # Failure
        except ValueError:
            pass
        
        summary = self.factory.get_reliability_summary()
        
        self.assertEqual(summary['total_components_wrapped'], 2)
        self.assertEqual(summary['total_method_calls'], 3)
        self.assertAlmostEqual(summary['overall_success_rate'], 2/3, places=2)
        self.assertEqual(summary['total_failures'], 1)


class TestReliabilityWrapperIntegration(unittest.TestCase):
    """Integration tests for ReliabilityWrapper."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.factory = ReliabilityWrapperFactory(self.temp_dir)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_real_world_scenario_with_missing_methods(self):
        """Test real-world scenario with components missing methods."""
        mock_component = MockComponentWithMissingMethods()
        wrapper = self.factory.wrap_component(mock_component)
        
        # Test existing method works
        result = wrapper.existing_method()
        self.assertEqual(result, "success")
        self.assertTrue(mock_component.existing_method_called)
        
        # Test missing method handling
        with self.assertRaises(AttributeError):
            wrapper.missing_method()
        
        # Check that recovery was attempted
        metrics = wrapper.get_metrics()
        self.assertGreater(metrics['metrics']['recovery_attempts'], 0)
    
    def test_performance_overhead_measurement(self):
        """Test that wrapper adds minimal performance overhead."""
        mock_component = MockComponent()
        
        # Measure direct call time
        start_time = time.time()
        for _ in range(100):
            mock_component.working_method(1)
        direct_time = time.time() - start_time
        
        # Reset component state
        mock_component.call_count = 0
        
        # Measure wrapped call time
        wrapper = self.factory.wrap_component(mock_component)
        start_time = time.time()
        for _ in range(100):
            wrapper.working_method(1)
        wrapped_time = time.time() - start_time
        
        # Overhead should be less than 50% (this is a generous threshold)
        overhead_ratio = (wrapped_time - direct_time) / direct_time
        self.assertLess(overhead_ratio, 0.5, f"Wrapper overhead too high: {overhead_ratio:.2%}")
        
        # Verify all calls were successful
        metrics = wrapper.get_metrics()
        self.assertEqual(metrics['metrics']['total_calls'], 100)
        self.assertEqual(metrics['metrics']['success_rate'], 1.0)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
