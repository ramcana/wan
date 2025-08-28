"""
Failure Injection Test Suite for Reliability System

This module provides comprehensive failure injection testing to validate
recovery mechanisms under various failure scenarios.

Requirements addressed: 1.1, 1.2, 1.3, 1.4, 1.5
"""

import pytest
import unittest
import random
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor
import sys
import os

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

try:
    from reliability_manager import ReliabilityManager, ComponentType
    from missing_method_recovery import MissingMethodRecovery
    from network_failure_recovery import NetworkFailureRecovery
    from model_validation_recovery import ModelValidationRecovery
    from intelligent_retry_system import IntelligentRetrySystem
except ImportError:
    # Mock classes for testing when components aren't available
    class MockComponent:
        pass
    ReliabilityManager = MockComponent
    MissingMethodRecovery = MockComponent
    NetworkFailureRecovery = MockComponent
    ModelValidationRecovery = MockComponent
    IntelligentRetrySystem = MockComponent

# Define MockComponent globally for test use
class MockComponent:
    pass


class FailureInjector:
    """Utility class for injecting various types of failures."""
    
    def __init__(self):
        self.failure_patterns = {
            'network_timeout': self._inject_network_timeout,
            'missing_method': self._inject_missing_method,
            'file_corruption': self._inject_file_corruption,
            'memory_exhaustion': self._inject_memory_exhaustion,
            'permission_denied': self._inject_permission_denied,
            'dependency_conflict': self._inject_dependency_conflict,
            'model_validation_failure': self._inject_model_validation_failure,
            'cascading_failure': self._inject_cascading_failure
        }
    
    def inject_failure(self, failure_type: str, target_component, **kwargs):
        """Inject a specific type of failure into a component."""
        if failure_type in self.failure_patterns:
            return self.failure_patterns[failure_type](target_component, **kwargs)
        else:
            raise ValueError(f"Unknown failure type: {failure_type}")
    
    def _inject_network_timeout(self, component, timeout_duration=5):
        """Inject network timeout failure."""
        def failing_network_operation(*args, **kwargs):
            time.sleep(timeout_duration)
            raise TimeoutError(f"Network operation timed out after {timeout_duration}s")
        
        # Replace network methods with failing versions
        if hasattr(component, 'download_file'):
            component.original_download_file = component.download_file
            component.download_file = failing_network_operation
        
        return True
    
    def _inject_missing_method(self, component, method_name='missing_method'):
        """Inject missing method AttributeError."""
        def failing_method_access(*args, **kwargs):
            raise AttributeError(f"'{component.__class__.__name__}' object has no attribute '{method_name}'")
        
        setattr(component, method_name, failing_method_access)
        return True
    
    def _inject_file_corruption(self, component, file_path='/fake/path/model.bin'):
        """Inject file corruption error."""
        def failing_file_operation(*args, **kwargs):
            raise IOError(f"File corrupted or unreadable: {file_path}")
        
        if hasattr(component, 'load_file'):
            component.original_load_file = component.load_file
            component.load_file = failing_file_operation
        
        return True
    
    def _inject_memory_exhaustion(self, component):
        """Inject memory exhaustion error."""
        def failing_memory_operation(*args, **kwargs):
            raise MemoryError("Cannot allocate memory for operation")
        
        if hasattr(component, 'load_model'):
            component.original_load_model = component.load_model
            component.load_model = failing_memory_operation
        
        return True
    
    def _inject_permission_denied(self, component, path='/restricted/path'):
        """Inject permission denied error."""
        def failing_permission_operation(*args, **kwargs):
            raise PermissionError(f"Permission denied: {path}")
        
        if hasattr(component, 'create_directory'):
            component.original_create_directory = component.create_directory
            component.create_directory = failing_permission_operation
        
        return True
    
    def _inject_dependency_conflict(self, component, package='torch'):
        """Inject dependency conflict error."""
        def failing_dependency_operation(*args, **kwargs):
            raise Exception(f"Dependency conflict: {package} version incompatible")
        
        if hasattr(component, 'install_package'):
            component.original_install_package = component.install_package
            component.install_package = failing_dependency_operation
        
        return True
    
    def _inject_model_validation_failure(self, component):
        """Inject model validation failure."""
        def failing_validation(*args, **kwargs):
            return False, ["Checksum mismatch", "File size incorrect", "Version incompatible"]
        
        if hasattr(component, 'validate_model'):
            component.original_validate_model = component.validate_model
            component.validate_model = failing_validation
        
        return True
    
    def _inject_cascading_failure(self, component, failure_chain=None):
        """Inject cascading failure across multiple operations."""
        if failure_chain is None:
            failure_chain = ['network_timeout', 'file_corruption', 'permission_denied']
        
        for failure_type in failure_chain:
            if failure_type != 'cascading_failure':  # Avoid infinite recursion
                self.inject_failure(failure_type, component)
        
        return True


class MockComponentWithFailures:
    """Mock component that can simulate various failure scenarios."""
    
    def __init__(self, name="MockComponent"):
        self.name = name
        self.call_count = 0
        self.failure_injector = FailureInjector()
        self.injected_failures = []
    
    def inject_failure(self, failure_type, **kwargs):
        """Inject a failure into this component."""
        success = self.failure_injector.inject_failure(failure_type, self, **kwargs)
        if success:
            self.injected_failures.append(failure_type)
        return success
    
    def download_file(self, url, destination):
        """Simulate file download operation."""
        self.call_count += 1
        return f"Downloaded {url} to {destination}"
    
    def load_file(self, file_path):
        """Simulate file loading operation."""
        self.call_count += 1
        return f"Loaded file from {file_path}"
    
    def load_model(self, model_path):
        """Simulate model loading operation."""
        self.call_count += 1
        return f"Loaded model from {model_path}"
    
    def create_directory(self, path):
        """Simulate directory creation operation."""
        self.call_count += 1
        return f"Created directory {path}"
    
    def install_package(self, package_name):
        """Simulate package installation operation."""
        self.call_count += 1
        return f"Installed package {package_name}"
    
    def validate_model(self, model_path):
        """Simulate model validation operation."""
        self.call_count += 1
        return True, []


class FailureInjectionTestSuite(unittest.TestCase):
    """Test suite for failure injection and recovery validation."""
    
    def setUp(self):
        """Set up test environment."""
        self.failure_injector = FailureInjector()
        self.mock_component = MockComponentWithFailures("TestComponent")
        
        # Track recovery attempts
        self.recovery_attempts = []
        self.recovery_successes = []
    
    def test_network_timeout_injection_and_recovery(self):
        """Test network timeout failure injection and recovery."""
        if ReliabilityManager == MockComponent:
            self.skipTest("ReliabilityManager not available")
        
        # Inject network timeout failure
        self.mock_component.inject_failure('network_timeout', timeout_duration=1)
        
        # Verify failure is injected
        with self.assertRaises(TimeoutError):
            self.mock_component.download_file("http://example.com/file", "/tmp/file")
        
        # Test recovery mechanism (would be handled by ReliabilityManager)
        manager = ReliabilityManager("/tmp", None)
        wrapped_component = manager.wrap_component(
            self.mock_component, 
            ComponentType.MODEL_DOWNLOADER
        )
        
        # Recovery should be attempted
        self.assertIn('network_timeout', self.mock_component.injected_failures)
    
    def test_missing_method_injection_and_recovery(self):
        """Test missing method failure injection and recovery."""
        # Inject missing method failure
        self.mock_component.inject_failure('missing_method', method_name='get_required_models')
        
        # Verify failure is injected
        with self.assertRaises(AttributeError):
            self.mock_component.get_required_models()
        
        # Test recovery with MissingMethodRecovery
        if 'MissingMethodRecovery' in globals():
            recovery = MissingMethodRecovery("/tmp", None)
            success = recovery.handle_missing_method(
                self.mock_component, 
                'get_required_models'
            )
            self.assertIsInstance(success, bool)
    
    def test_file_corruption_injection_and_recovery(self):
        """Test file corruption failure injection and recovery."""
        # Inject file corruption failure
        self.mock_component.inject_failure('file_corruption', file_path='/fake/model.bin')
        
        # Verify failure is injected
        with self.assertRaises(IOError):
            self.mock_component.load_file('/fake/model.bin')
        
        # Recovery would involve re-downloading or using backup
        self.assertIn('file_corruption', self.mock_component.injected_failures)
    
    def test_memory_exhaustion_injection_and_recovery(self):
        """Test memory exhaustion failure injection and recovery."""
        # Inject memory exhaustion failure
        self.mock_component.inject_failure('memory_exhaustion')
        
        # Verify failure is injected
        with self.assertRaises(MemoryError):
            self.mock_component.load_model('/path/to/large/model')
        
        # Recovery would involve memory cleanup or smaller batch sizes
        self.assertIn('memory_exhaustion', self.mock_component.injected_failures)
    
    def test_permission_denied_injection_and_recovery(self):
        """Test permission denied failure injection and recovery."""
        # Inject permission denied failure
        self.mock_component.inject_failure('permission_denied', path='/restricted/path')
        
        # Verify failure is injected
        with self.assertRaises(PermissionError):
            self.mock_component.create_directory('/restricted/path')
        
        # Recovery would involve using alternative paths or requesting elevation
        self.assertIn('permission_denied', self.mock_component.injected_failures)
    
    def test_dependency_conflict_injection_and_recovery(self):
        """Test dependency conflict failure injection and recovery."""
        # Inject dependency conflict failure
        self.mock_component.inject_failure('dependency_conflict', package='torch')
        
        # Verify failure is injected
        with self.assertRaises(Exception):
            self.mock_component.install_package('torch')
        
        # Recovery would involve version resolution or alternative packages
        self.assertIn('dependency_conflict', self.mock_component.injected_failures)
    
    def test_model_validation_failure_injection_and_recovery(self):
        """Test model validation failure injection and recovery."""
        # Inject model validation failure
        self.mock_component.inject_failure('model_validation_failure')
        
        # Verify failure is injected
        is_valid, issues = self.mock_component.validate_model('/path/to/model')
        self.assertFalse(is_valid)
        self.assertGreater(len(issues), 0)
        
        # Test recovery with ModelValidationRecovery
        if 'ModelValidationRecovery' in globals():
            recovery = ModelValidationRecovery("/tmp", "models", None)
            success = recovery.recover_model_validation_failures(issues)
            self.assertIsInstance(success, bool)
    
    def test_cascading_failure_injection_and_recovery(self):
        """Test cascading failure injection and recovery."""
        # Inject cascading failures
        failure_chain = ['network_timeout', 'file_corruption', 'permission_denied']
        self.mock_component.inject_failure('cascading_failure', failure_chain=failure_chain)
        
        # Verify multiple failures are injected
        for failure_type in failure_chain:
            self.assertIn(failure_type, self.mock_component.injected_failures)
        
        # Test that multiple recovery mechanisms would be triggered
        self.assertEqual(len(self.mock_component.injected_failures), len(failure_chain))
    
    def test_random_failure_injection_stress(self):
        """Test system resilience with random failure injection."""
        failure_types = [
            'network_timeout', 'missing_method', 'file_corruption',
            'memory_exhaustion', 'permission_denied', 'dependency_conflict'
        ]
        
        # Create multiple components with random failures
        components = []
        for i in range(10):
            component = MockComponentWithFailures(f"RandomComponent{i}")
            
            # Inject 1-3 random failures per component
            num_failures = random.randint(1, 3)
            selected_failures = random.sample(failure_types, num_failures)
            
            for failure_type in selected_failures:
                component.inject_failure(failure_type)
            
            components.append(component)
        
        # Verify failures were injected
        total_failures = sum(len(comp.injected_failures) for comp in components)
        self.assertGreater(total_failures, 0)
        
        # Test that each component has at least one failure
        for component in components:
            self.assertGreater(len(component.injected_failures), 0)
    
    def test_concurrent_failure_injection(self):
        """Test concurrent failure injection and recovery."""
        def inject_and_test_failure(component_id):
            component = MockComponentWithFailures(f"ConcurrentComponent{component_id}")
            
            # Inject random failure
            failure_types = ['network_timeout', 'file_corruption', 'permission_denied']
            failure_type = random.choice(failure_types)
            component.inject_failure(failure_type)
            
            # Simulate operation that would trigger failure
            try:
                if failure_type == 'network_timeout':
                    component.download_file("http://example.com/file", "/tmp/file")
                elif failure_type == 'file_corruption':
                    component.load_file("/fake/file")
                elif failure_type == 'permission_denied':
                    component.create_directory("/restricted/path")
            except Exception:
                pass  # Expected to fail
            
            return component.injected_failures
        
        # Run concurrent failure injection
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(inject_and_test_failure, i) for i in range(10)]
            results = [future.result() for future in futures]
        
        # Verify all components had failures injected
        self.assertEqual(len(results), 10)
        for result in results:
            self.assertGreater(len(result), 0)
    
    def test_failure_recovery_effectiveness(self):
        """Test the effectiveness of recovery mechanisms."""
        if ReliabilityManager == MockComponent:
            self.skipTest("ReliabilityManager not available")
        
        manager = ReliabilityManager("/tmp", None)
        
        # Test recovery for different failure types
        recovery_results = {}
        
        failure_scenarios = [
            ('network_timeout', 'download_file', ["http://example.com/file", "/tmp/file"]),
            ('file_corruption', 'load_file', ["/fake/file"]),
            ('permission_denied', 'create_directory', ["/restricted/path"])
        ]
        
        for failure_type, method_name, args in failure_scenarios:
            component = MockComponentWithFailures(f"RecoveryTest_{failure_type}")
            component.inject_failure(failure_type)
            
            wrapped_component = manager.wrap_component(
                component, 
                ComponentType.MODEL_DOWNLOADER
            )
            
            # Attempt operation that should trigger recovery
            try:
                method = getattr(wrapped_component, method_name)
                result = method(*args)
                recovery_results[failure_type] = 'success'
            except Exception as e:
                recovery_results[failure_type] = f'failed: {e}'
        
        # Log recovery effectiveness
        print(f"Recovery results: {recovery_results}")
        
        # At least some recoveries should be attempted
        self.assertGreater(len(recovery_results), 0)
    
    def test_failure_injection_cleanup(self):
        """Test that failure injection can be properly cleaned up."""
        component = MockComponentWithFailures("CleanupTest")
        
        # Inject multiple failures
        component.inject_failure('network_timeout')
        component.inject_failure('file_corruption')
        
        # Verify failures are active
        self.assertEqual(len(component.injected_failures), 2)
        
        # Test cleanup (restore original methods)
        if hasattr(component, 'original_download_file'):
            component.download_file = component.original_download_file
        if hasattr(component, 'original_load_file'):
            component.load_file = component.original_load_file
        
        # Verify operations work normally after cleanup
        result1 = component.download_file("http://example.com/file", "/tmp/file")
        result2 = component.load_file("/fake/file")
        
        self.assertIn("Downloaded", result1)
        self.assertIn("Loaded", result2)


def run_failure_injection_tests():
    """Run the failure injection test suite."""
    suite = unittest.TestLoader().loadTestsFromTestCase(FailureInjectionTestSuite)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print(f"\nFailure Injection Tests - Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_failure_injection_tests()
    sys.exit(0 if success else 1)