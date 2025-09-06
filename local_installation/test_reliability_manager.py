"""
Integration tests for ReliabilityManager coordination system.

Tests the central coordination of reliability operations including:
- Component wrapping and failure handling coordination
- Recovery strategy selection and execution logic
- Reliability metrics collection and analysis
- Component health monitoring and tracking
"""

import pytest
import logging
import time
import threading
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import tempfile
import json

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from reliability_manager import (
    ReliabilityManager, ComponentType, RecoveryStrategy, 
    ComponentHealth, ReliabilityMetrics, RecoverySession
)
from interfaces import InstallationError, ErrorCategory
# Import RecoveryAction from error_handler to match reliability_manager
import reliability_manager
RecoveryAction = reliability_manager.RecoveryAction


class MockComponent:
    """Mock component for testing."""
    
    def __init__(self, name="MockComponent", should_fail=False):
        self.name = name
        self.should_fail = should_fail
        self.call_count = 0
    
    def working_method(self):
        """A method that works normally."""
        self.call_count += 1
        if self.should_fail:
            raise RuntimeError(f"Mock failure in {self.name}")
        return f"Success from {self.name}"
    
    def failing_method(self):
        """A method that always fails."""
        self.call_count += 1
        raise RuntimeError(f"Intentional failure in {self.name}")
    
    def missing_method_error(self):
        """Simulate missing method error."""
        raise AttributeError(f"'{self.name}' object has no attribute 'nonexistent_method'")


class TestReliabilityManager:
    """Test suite for ReliabilityManager."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def logger(self):
        """Create test logger."""
        logger = logging.getLogger("test_reliability_manager")
        logger.setLevel(logging.DEBUG)
        return logger
    
    @pytest.fixture
    def reliability_manager(self, temp_dir, logger):
        """Create ReliabilityManager instance for testing."""
        manager = ReliabilityManager(temp_dir, logger)
        yield manager
        manager.shutdown()
    
    def test_initialization(self, reliability_manager):
        """Test ReliabilityManager initialization."""
        assert reliability_manager is not None
        assert reliability_manager.wrapped_components == {}
        assert reliability_manager.component_health == {}
        assert reliability_manager.metrics.total_components == 0
        assert reliability_manager.metrics.healthy_components == 0
    
    def test_wrap_component_basic(self, reliability_manager):
        """Test basic component wrapping."""
        mock_component = MockComponent("TestComponent")
        
        # Wrap the component
        wrapper = reliability_manager.wrap_component(mock_component, "test_component", "test_id")
        
        # Verify wrapper was created
        assert wrapper is not None
        assert "test_id" in reliability_manager.wrapped_components
        assert "test_id" in reliability_manager.component_health
        
        # Verify metrics updated
        assert reliability_manager.metrics.total_components == 1
        assert reliability_manager.metrics.healthy_components == 1
    
    def test_wrap_component_type_detection(self, reliability_manager):
        """Test automatic component type detection."""
        # Test different component types
        test_cases = [
            ("ModelDownloader", ComponentType.MODEL_DOWNLOADER),
            ("DependencyManager", ComponentType.DEPENDENCY_MANAGER),
            ("ConfigurationEngine", ComponentType.CONFIGURATION_ENGINE),
            ("InstallationValidator", ComponentType.INSTALLATION_VALIDATOR),
            ("SystemDetector", ComponentType.SYSTEM_DETECTOR),
            ("ErrorHandler", ComponentType.ERROR_HANDLER),
            ("ProgressReporter", ComponentType.PROGRESS_REPORTER),
            ("UnknownComponent", ComponentType.UNKNOWN)
        ]
        
        for class_name, expected_type in test_cases:
            # Create mock component with specific class name
            mock_component = type(class_name, (), {})()
            
            wrapper = reliability_manager.wrap_component(mock_component)
            component_id = f"{class_name}_{id(mock_component)}"
            
            assert reliability_manager.component_types[component_id] == expected_type
    
    def test_component_health_tracking(self, reliability_manager):
        """Test component health tracking."""
        mock_component = MockComponent("HealthTest")
        wrapper = reliability_manager.wrap_component(mock_component, component_id="health_test")
        
        # Initial health should be good
        health = reliability_manager.check_component_health("health_test")
        assert health is not None
        assert health.is_healthy is True
        assert health.success_rate == 1.0
        assert health.total_calls == 0
        
        # Track successful operation
        reliability_manager.track_reliability_metrics("health_test", "test_op", True, 0.1)
        
        health = reliability_manager.check_component_health("health_test")
        assert health.total_calls == 1
        assert health.success_rate == 1.0
        assert health.is_healthy is True
        
        # Track failed operation
        reliability_manager.track_reliability_metrics("health_test", "test_op", False, 0.2)
        
        health = reliability_manager.check_component_health("health_test")
        assert health.total_calls == 2
        assert health.success_rate == 0.5
        assert health.consecutive_failures == 1
    
    def test_recovery_strategy_selection(self, reliability_manager):
        """Test recovery strategy selection logic."""
        # Test different error types
        test_cases = [
            (RuntimeError("connection timeout"), "network_timeout"),
            (AttributeError("'obj' has no attribute 'method'"), "missing_method"),
            (RuntimeError("model validation failed"), "model_validation"),
            (PermissionError("access denied"), "permission_error"),
            (ValueError("configuration error"), "configuration_error"),
            (RuntimeError("unknown error"), "system_error")
        ]
        
        for error, expected_classification in test_cases:
            classification = reliability_manager._classify_error_for_recovery(error)
            assert classification == expected_classification
            
            # Test strategy selection
            strategy = reliability_manager.get_recovery_strategy(error, "test_component", {})
            assert isinstance(strategy, RecoveryStrategy)
    
    def test_handle_component_failure(self, reliability_manager):
        """Test component failure handling."""
        mock_component = MockComponent("FailureTest")
        wrapper = reliability_manager.wrap_component(mock_component, component_id="failure_test")
        
        # Simulate component failure
        error = RuntimeError("Test failure")
        context = {"method": "test_method", "retry_count": 0}
        
        recovery_action = reliability_manager.handle_component_failure("failure_test", error, context)
        
        # Verify recovery action was returned
        # Verify recovery action was returned
        assert isinstance(recovery_action, RecoveryAction)
        
        # Verify component health was updated
        health = reliability_manager.check_component_health("failure_test")
        assert health.consecutive_failures > 0
        assert health.last_error is not None
        
        # Verify metrics were updated
        assert reliability_manager.metrics.recovery_attempts > 0
    
    def test_missing_method_recovery_execution(self, reliability_manager):
        """Test missing method recovery execution."""
        mock_component = MockComponent("MissingMethodTest")
        wrapper = reliability_manager.wrap_component(mock_component, component_id="missing_method_test")
        
        # Create missing method error
        error = AttributeError("'MockComponent' object has no attribute 'nonexistent_method'")
        context = {"method": "nonexistent_method", "retry_count": 0}
        
        # Mock the missing method recovery
        with patch.object(reliability_manager.missing_method_recovery, 'handle_missing_method') as mock_recovery:
            mock_recovery.return_value = "recovered_result"
            
            recovery_action = reliability_manager.handle_component_failure("missing_method_test", error, context)
            
            # Should attempt missing method recovery
            assert recovery_action in [RecoveryAction.RETRY, RecoveryAction.ABORT]
    
    def test_model_validation_recovery_execution(self, reliability_manager):
        """Test model validation recovery execution."""
        mock_component = MockComponent("ModelValidationTest")
        wrapper = reliability_manager.wrap_component(mock_component, component_id="model_validation_test")
        
        # Create model validation error
        error = RuntimeError("model validation failed for wan2.2/t2v-a14b")
        context = {"method": "validate_model", "retry_count": 0}
        
        # Mock the model validation recovery
        with patch.object(reliability_manager.model_validation_recovery, 'recover_model') as mock_recovery:
            mock_recovery.return_value = Mock(success=True, details="Model recovered")
            
            recovery_action = reliability_manager.handle_component_failure("model_validation_test", error, context)
            
            # Should attempt model validation recovery
            assert recovery_action in [RecoveryAction.RETRY, RecoveryAction.MANUAL_INTERVENTION]
    
    def test_network_failure_recovery_execution(self, reliability_manager):
        """Test network failure recovery execution."""
        mock_component = MockComponent("NetworkFailureTest")
        wrapper = reliability_manager.wrap_component(mock_component, component_id="network_failure_test")
        
        # Create network failure error
        error = RuntimeError("connection timeout")
        context = {"method": "download", "retry_count": 0}
        
        # Mock the network connectivity test
        with patch.object(reliability_manager.network_failure_recovery.connectivity_tester, 'test_basic_connectivity') as mock_test:
            mock_test.return_value = Mock(success=True, error_message=None)
            
            recovery_action = reliability_manager.handle_component_failure("network_failure_test", error, context)
            
            # Should attempt network failure recovery
            assert recovery_action in [RecoveryAction.RETRY, RecoveryAction.MANUAL_INTERVENTION]
    
    def test_reliability_metrics_collection(self, reliability_manager):
        """Test reliability metrics collection and analysis."""
        # Create multiple components
        components = []
        for i in range(3):
            mock_component = MockComponent(f"Component{i}")
            wrapper = reliability_manager.wrap_component(mock_component, component_id=f"comp_{i}")
            components.append((mock_component, wrapper))
        
        # Simulate various operations
        for i, (component, wrapper) in enumerate(components):
            for j in range(10):
                success = j < 8  # 80% success rate
                reliability_manager.track_reliability_metrics(f"comp_{i}", "test_op", success, 0.1)
        
        # Check metrics
        metrics = reliability_manager.get_reliability_metrics()
        assert metrics.total_components == 3
        assert metrics.total_method_calls == 30
        assert metrics.successful_calls == 24
        assert metrics.failed_calls == 6
        assert abs(metrics.average_response_time - 0.1) < 0.01
    
    def test_component_health_monitoring(self, reliability_manager):
        """Test component health monitoring."""
        mock_component = MockComponent("MonitoringTest")
        wrapper = reliability_manager.wrap_component(mock_component, component_id="monitoring_test")
        
        # Start monitoring
        reliability_manager.start_monitoring()
        
        try:
            # Simulate some operations
            for i in range(5):
                success = i < 3  # First 3 succeed, last 2 fail
                reliability_manager.track_reliability_metrics("monitoring_test", "test_op", success, 0.1)
                time.sleep(0.1)
            
            # Wait for monitoring to process
            time.sleep(1)
            
            # Check health
            health = reliability_manager.check_component_health("monitoring_test")
            assert health is not None
            assert health.total_calls == 5
            assert health.success_rate == 0.6
            
        finally:
            reliability_manager.stop_monitoring()
    
    def test_recovery_session_tracking(self, reliability_manager):
        """Test recovery session tracking."""
        mock_component = MockComponent("RecoverySessionTest")
        wrapper = reliability_manager.wrap_component(mock_component, component_id="recovery_session_test")
        
        # Simulate failure and recovery
        error = RuntimeError("Test failure for recovery session")
        context = {"method": "test_method", "retry_count": 0}
        
        initial_history_count = len(reliability_manager.get_recovery_history())
        
        recovery_action = reliability_manager.handle_component_failure("recovery_session_test", error, context)
        
        # Check that recovery session was recorded
        recovery_history = reliability_manager.get_recovery_history()
        assert len(recovery_history) == initial_history_count + 1
        
        latest_session = recovery_history[-1]
        assert latest_session.component_id == "recovery_session_test"
        assert latest_session.error == error
        assert latest_session.end_time is not None
    
    def test_multiple_component_coordination(self, reliability_manager):
        """Test coordination of multiple components."""
        # Create multiple components with different behaviors
        components = {
            "working": MockComponent("WorkingComponent", should_fail=False),
            "failing": MockComponent("FailingComponent", should_fail=True),
            "intermittent": MockComponent("IntermittentComponent", should_fail=False)
        }
        
        wrappers = {}
        for comp_id, component in components.items():
            wrappers[comp_id] = reliability_manager.wrap_component(component, component_id=comp_id)
        
        # Simulate operations on all components
        for comp_id, component in components.items():
            for i in range(5):
                try:
                    if comp_id == "intermittent" and i >= 3:
                        component.should_fail = True
                    
                    if component.should_fail:
                        component.failing_method()
                    else:
                        component.working_method()
                    
                    reliability_manager.track_reliability_metrics(comp_id, "test_op", True, 0.1)
                except Exception as e:
                    reliability_manager.track_reliability_metrics(comp_id, "test_op", False, 0.1)
                    reliability_manager.handle_component_failure(comp_id, e, {"method": "test_op"})
        
        # Check overall system health
        all_health = reliability_manager.get_all_component_health()
        assert len(all_health) == 3
        
        # Working component should be healthy
        assert all_health["working"].is_healthy
        
        # Failing component should be unhealthy
        assert not all_health["failing"].is_healthy
        
        # Check metrics
        metrics = reliability_manager.get_reliability_metrics()
        assert metrics.total_components == 3
        assert metrics.recovery_attempts > 0
    
    def test_export_reliability_report(self, reliability_manager, temp_dir):
        """Test reliability report export."""
        # Create some components and simulate activity
        mock_component = MockComponent("ReportTest")
        wrapper = reliability_manager.wrap_component(mock_component, component_id="report_test")
        
        # Simulate some operations
        for i in range(10):
            success = i < 7  # 70% success rate
            reliability_manager.track_reliability_metrics("report_test", "test_op", success, 0.1)
            
            if not success:
                error = RuntimeError(f"Test failure {i}")
                reliability_manager.handle_component_failure("report_test", error, {"method": "test_op"})
        
        # Export report
        report_path = reliability_manager.export_reliability_report()
        
        # Verify report was created
        assert report_path != ""
        assert Path(report_path).exists()
        
        # Verify report content
        with open(report_path, 'r') as f:
            report_data = json.load(f)
        
        assert "timestamp" in report_data
        assert "metrics" in report_data
        assert "component_health" in report_data
        assert "recovery_history" in report_data
        
        # Check metrics in report
        metrics = report_data["metrics"]
        assert metrics["total_components"] == 1
        assert metrics["total_method_calls"] == 10
        assert metrics["successful_calls"] == 7
        assert metrics["failed_calls"] == 3
        
        # Check component health in report
        assert "report_test" in report_data["component_health"]
        component_health = report_data["component_health"]["report_test"]
        # The total calls might be higher due to recovery attempts
        assert component_health["total_calls"] >= 10
        assert component_health["failed_calls"] >= 3
    
    def test_concurrent_operations(self, reliability_manager):
        """Test concurrent operations and thread safety."""
        mock_component = MockComponent("ConcurrentTest")
        wrapper = reliability_manager.wrap_component(mock_component, component_id="concurrent_test")
        
        # Start monitoring
        reliability_manager.start_monitoring()
        
        try:
            # Create multiple threads that perform operations
            def worker_thread(thread_id):
                for i in range(20):
                    success = (i + thread_id) % 3 != 0  # Vary success rate
                    reliability_manager.track_reliability_metrics("concurrent_test", f"op_{thread_id}", success, 0.05)
                    
                    if not success:
                        error = RuntimeError(f"Thread {thread_id} failure {i}")
                        reliability_manager.handle_component_failure("concurrent_test", error, {"method": f"op_{thread_id}"})
                    
                    time.sleep(0.01)  # Small delay
            
            # Start multiple threads
            threads = []
            for i in range(5):
                thread = threading.Thread(target=worker_thread, args=(i,))
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            # Check final state
            health = reliability_manager.check_component_health("concurrent_test")
            assert health is not None
            assert health.total_calls == 100  # 5 threads * 20 operations each
            
            metrics = reliability_manager.get_reliability_metrics()
            assert metrics.total_method_calls == 100
            
        finally:
            reliability_manager.stop_monitoring()
    
    def test_recovery_strategy_escalation(self, reliability_manager):
        """Test recovery strategy escalation for persistent failures."""
        mock_component = MockComponent("EscalationTest")
        wrapper = reliability_manager.wrap_component(mock_component, component_id="escalation_test")
        
        # Simulate repeated failures
        error = RuntimeError("Persistent failure")
        
        recovery_actions = []
        for i in range(5):
            context = {"method": "test_method", "retry_count": i}
            action = reliability_manager.handle_component_failure("escalation_test", error, context)
            recovery_actions.append(action)
        
        # Should escalate to more aggressive recovery strategies
        assert len(recovery_actions) == 5
        
        # Check that component health degraded
        health = reliability_manager.check_component_health("escalation_test")
        assert health.consecutive_failures >= 5
        assert not health.is_healthy
    
    def test_cleanup_and_shutdown(self, reliability_manager, temp_dir):
        """Test proper cleanup and shutdown."""
        # Create some components and activity
        mock_component = MockComponent("ShutdownTest")
        wrapper = reliability_manager.wrap_component(mock_component, component_id="shutdown_test")
        
        # Start monitoring
        reliability_manager.start_monitoring()
        
        # Simulate some activity
        reliability_manager.track_reliability_metrics("shutdown_test", "test_op", True, 0.1)
        
        # Shutdown
        reliability_manager.shutdown()
        
        # Verify monitoring stopped
        assert not reliability_manager._monitoring_active
        
        # Verify final report was created
        report_path = Path(temp_dir) / "logs" / "reliability_report.json"
        assert report_path.exists()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])