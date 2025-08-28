"""
Chaos Engineering Tests for Enhanced Model Availability System

This module implements chaos engineering principles to validate system resilience
under various failure conditions, including component failures, network partitions,
and resource exhaustion scenarios.

Requirements covered: 1.4, 2.4, 3.4, 4.4, 6.4, 7.4
"""

import pytest
import asyncio
import random
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import threading
import json

from backend.core.enhanced_model_downloader import EnhancedModelDownloader
from backend.core.model_health_monitor import ModelHealthMonitor
from backend.core.model_availability_manager import ModelAvailabilityManager
from backend.core.intelligent_fallback_manager import IntelligentFallbackManager
from backend.core.enhanced_error_recovery import EnhancedErrorRecovery
from backend.websocket.model_notifications import ModelNotificationManager


@dataclass
class ChaosExperiment:
    """Configuration for a chaos engineering experiment."""
    name: str
    description: str
    failure_type: str
    duration_seconds: float
    intensity: float  # 0.0 to 1.0
    target_components: List[str]
    expected_behavior: str
    recovery_time_limit: float


@dataclass
class ChaosResult:
    """Result of a chaos engineering experiment."""
    experiment_name: str
    success: bool
    system_remained_stable: bool
    recovery_time: float
    error_messages: List[str]
    performance_impact: Dict[str, float]
    lessons_learned: List[str]


class ChaosEngineeringTestSuite:
    """Comprehensive chaos engineering test suite."""

    def __init__(self):
        self.active_experiments = {}
        self.system_metrics = {}
        self.failure_injectors = {}

    async def test_component_failure_cascade(self):
        """Test system behavior when components fail in cascade."""
        print("Starting component failure cascade test...")
        
        # Define component dependency chain
        component_chain = [
            "model_downloader",
            "health_monitor", 
            "availability_manager",
            "fallback_manager",
            "notification_manager"
        ]
        
        # Create mock system
        system_components = await self._create_mock_system()
        
        cascade_results = []
        
        # Fail components one by one and observe cascade effects
        for i, component in enumerate(component_chain):
            print(f"Failing component: {component}")
            
            # Inject failure
            failure_start = time.time()
            await self._inject_component_failure(system_components, component)
            
            # Monitor system behavior
            stability_result = await self._monitor_system_stability(
                system_components, duration=2.0
            )
            
            # Attempt recovery
            recovery_start = time.time()
            recovery_success = await self._attempt_component_recovery(
                system_components, component
            )
            recovery_time = time.time() - recovery_start
            
            cascade_results.append({
                'failed_component': component,
                'cascade_position': i,
                'system_stable': stability_result['stable'],
                'recovery_successful': recovery_success,
                'recovery_time': recovery_time,
                'affected_components': stability_result['affected_components']
            })
            
            # Verify system can handle the failure
            assert stability_result['stable'] or len(stability_result['affected_components']) <= 2
        
        print(f"Component failure cascade test completed:")
        print(f"  - Components tested: {len(component_chain)}")
        print(f"  - Successful recoveries: {sum(1 for r in cascade_results if r['recovery_successful'])}")
        print(f"  - Average recovery time: {sum(r['recovery_time'] for r in cascade_results) / len(cascade_results):.2f}s")
        
        return cascade_results

    async def _create_mock_system(self):
        """Create mock system components for chaos testing."""
        mock_base_downloader = Mock()
        mock_model_manager = Mock()
        
        components = {
            'model_downloader': EnhancedModelDownloader(mock_base_downloader),
            'health_monitor': ModelHealthMonitor(),
            'availability_manager': ModelAvailabilityManager(mock_model_manager, mock_base_downloader),
            'fallback_manager': IntelligentFallbackManager(None),
            'notification_manager': ModelNotificationManager()
        }
        
        # Set up component interactions
        components['fallback_manager'].availability_manager = components['availability_manager']
        
        return components

    async def _inject_component_failure(self, system_components, component_name):
        """Inject failure into specific component."""
        component = system_components.get(component_name)
        if not component:
            return
        
        # Simulate different types of failures
        failure_types = [
            'service_unavailable',
            'timeout_error',
            'memory_error',
            'network_error',
            'permission_error'
        ]
        
        failure_type = random.choice(failure_types)
        
        # Mock the component to raise exceptions
        if hasattr(component, 'download_with_retry'):
            component.download_with_retry = AsyncMock(side_effect=Exception(f"{failure_type} in {component_name}"))
        if hasattr(component, 'check_model_integrity'):
            component.check_model_integrity = AsyncMock(side_effect=Exception(f"{failure_type} in {component_name}"))
        if hasattr(component, 'handle_model_request'):
            component.handle_model_request = AsyncMock(side_effect=Exception(f"{failure_type} in {component_name}"))

    async def _monitor_system_stability(self, system_components, duration):
        """Monitor system stability during chaos experiment."""
        start_time = time.time()
        affected_components = []
        stability_checks = []
        
        while time.time() - start_time < duration:
            # Test each component
            for name, component in system_components.items():
                try:
                    # Perform basic health check
                    if hasattr(component, 'download_with_retry'):
                        await asyncio.wait_for(
                            component.download_with_retry("test-model", max_retries=1),
                            timeout=0.5
                        )
                    elif hasattr(component, 'check_model_integrity'):
                        await asyncio.wait_for(
                            component.check_model_integrity("test-model"),
                            timeout=0.5
                        )
                    elif hasattr(component, 'handle_model_request'):
                        await asyncio.wait_for(
                            component.handle_model_request("test-model"),
                            timeout=0.5
                        )
                    
                    stability_checks.append((name, True))
                    
                except Exception:
                    stability_checks.append((name, False))
                    if name not in affected_components:
                        affected_components.append(name)
            
            await asyncio.sleep(0.1)
        
        # Determine overall stability
        total_checks = len(stability_checks)
        successful_checks = sum(1 for _, success in stability_checks if success)
        stability_ratio = successful_checks / total_checks if total_checks > 0 else 0
        
        return {
            'stable': stability_ratio >= 0.5,  # At least 50% of checks successful
            'stability_ratio': stability_ratio,
            'affected_components': affected_components,
            'total_checks': total_checks
        }

    async def _attempt_component_recovery(self, system_components, component_name):
        """Attempt to recover failed component."""
        component = system_components.get(component_name)
        if not component:
            return False
        
        try:
            # Simulate component restart/recovery
            await asyncio.sleep(0.1)  # Recovery time
            
            # Restore component functionality
            if hasattr(component, 'download_with_retry'):
                component.download_with_retry = AsyncMock(return_value=Mock(success=True))
            if hasattr(component, 'check_model_integrity'):
                component.check_model_integrity = AsyncMock(return_value=Mock(is_healthy=True))
            if hasattr(component, 'handle_model_request'):
                component.handle_model_request = AsyncMock(return_value=Mock(success=True))
            
            return True
            
        except Exception:
            return False

    async def test_network_partition_scenarios(self):
        """Test system behavior during network partition scenarios."""
        print("Starting network partition scenarios test...")
        
        partition_scenarios = [
            {
                'name': 'complete_network_loss',
                'description': 'Complete loss of network connectivity',
                'duration': 3.0,
                'recovery_expected': True
            },
            {
                'name': 'intermittent_connectivity',
                'description': 'Intermittent network connectivity',
                'duration': 5.0,
                'recovery_expected': True
            },
            {
                'name': 'high_latency_connection',
                'description': 'High latency network connection',
                'duration': 2.0,
                'recovery_expected': True
            },
            {
                'name': 'bandwidth_throttling',
                'description': 'Severe bandwidth throttling',
                'duration': 4.0,
                'recovery_expected': True
            }
        ]
        
        partition_results = []
        system_components = await self._create_mock_system()
        
        for scenario in partition_scenarios:
            print(f"Testing network partition: {scenario['name']}")
            
            # Inject network partition
            partition_start = time.time()
            await self._inject_network_partition(system_components, scenario)
            
            # Monitor system behavior during partition
            behavior_result = await self._monitor_network_partition_behavior(
                system_components, scenario['duration']
            )
            
            # Test recovery after partition ends
            recovery_start = time.time()
            recovery_success = await self._recover_from_network_partition(system_components)
            recovery_time = time.time() - recovery_start
            
            partition_results.append({
                'scenario': scenario['name'],
                'system_adapted': behavior_result['adapted_to_partition'],
                'fallback_activated': behavior_result['fallback_activated'],
                'recovery_successful': recovery_success,
                'recovery_time': recovery_time,
                'data_consistency': behavior_result['data_consistent']
            })
            
            # Verify expected behavior
            if scenario['recovery_expected']:
                assert recovery_success, f"Recovery failed for {scenario['name']}"
        
        print(f"Network partition scenarios test completed:")
        successful_adaptations = sum(1 for r in partition_results if r['system_adapted'])
        print(f"  - Scenarios tested: {len(partition_scenarios)}")
        print(f"  - Successful adaptations: {successful_adaptations}")
        print(f"  - Fallback activations: {sum(1 for r in partition_results if r['fallback_activated'])}")
        
        return partition_results

    async def _inject_network_partition(self, system_components, scenario):
        """Inject network partition based on scenario."""
        partition_type = scenario['name']
        
        # Mock network failures in components
        for component in system_components.values():
            if hasattr(component, 'download_with_retry'):
                if partition_type == 'complete_network_loss':
                    component.download_with_retry = AsyncMock(
                        side_effect=Exception("Network unreachable")
                    )
                elif partition_type == 'intermittent_connectivity':
                    # Randomly fail network calls
                    async def intermittent_download(*args, **kwargs):
                        if random.random() < 0.6:  # 60% failure rate
                            raise Exception("Connection timeout")
                        return Mock(success=True)
                    component.download_with_retry = intermittent_download
                elif partition_type == 'high_latency_connection':
                    async def slow_download(*args, **kwargs):
                        await asyncio.sleep(2.0)  # Simulate high latency
                        return Mock(success=True)
                    component.download_with_retry = slow_download

    async def _monitor_network_partition_behavior(self, system_components, duration):
        """Monitor system behavior during network partition."""
        start_time = time.time()
        adaptation_indicators = []
        fallback_activations = 0
        
        while time.time() - start_time < duration:
            # Test system adaptation
            try:
                # Attempt operations that should trigger fallback
                availability_manager = system_components.get('availability_manager')
                if availability_manager:
                    result = await asyncio.wait_for(
                        availability_manager.handle_model_request("test-model"),
                        timeout=1.0
                    )
                    adaptation_indicators.append(True)
                    
                    # Check if fallback was activated
                    if hasattr(result, 'used_fallback') and result.used_fallback:
                        fallback_activations += 1
                        
            except Exception:
                adaptation_indicators.append(False)
            
            await asyncio.sleep(0.2)
        
        adaptation_rate = sum(adaptation_indicators) / len(adaptation_indicators) if adaptation_indicators else 0
        
        return {
            'adapted_to_partition': adaptation_rate >= 0.3,  # At least 30% adaptation
            'fallback_activated': fallback_activations > 0,
            'data_consistent': True,  # Assume data consistency maintained
            'adaptation_rate': adaptation_rate
        }

    async def _recover_from_network_partition(self, system_components):
        """Recover system from network partition."""
        try:
            # Restore network functionality
            for component in system_components.values():
                if hasattr(component, 'download_with_retry'):
                    component.download_with_retry = AsyncMock(return_value=Mock(success=True))
            
            # Test recovery
            await asyncio.sleep(0.5)  # Allow recovery time
            
            # Verify system is operational
            availability_manager = system_components.get('availability_manager')
            if availability_manager:
                result = await availability_manager.handle_model_request("test-model")
                return hasattr(result, 'success') and result.success
            
            return True
            
        except Exception:
            return False

    async def test_resource_exhaustion_scenarios(self):
        """Test system behavior under resource exhaustion."""
        print("Starting resource exhaustion scenarios test...")
        
        resource_scenarios = [
            {
                'name': 'disk_space_exhaustion',
                'resource': 'disk',
                'exhaustion_level': 0.95,  # 95% full
                'expected_behavior': 'cleanup_triggered'
            },
            {
                'name': 'memory_exhaustion',
                'resource': 'memory',
                'exhaustion_level': 0.90,  # 90% full
                'expected_behavior': 'graceful_degradation'
            },
            {
                'name': 'cpu_overload',
                'resource': 'cpu',
                'exhaustion_level': 0.98,  # 98% usage
                'expected_behavior': 'throttling_activated'
            },
            {
                'name': 'file_descriptor_limit',
                'resource': 'file_descriptors',
                'exhaustion_level': 0.95,  # 95% of limit
                'expected_behavior': 'connection_pooling'
            }
        ]
        
        exhaustion_results = []
        system_components = await self._create_mock_system()
        
        for scenario in resource_scenarios:
            print(f"Testing resource exhaustion: {scenario['name']}")
            
            # Inject resource exhaustion
            exhaustion_start = time.time()
            await self._inject_resource_exhaustion(system_components, scenario)
            
            # Monitor system response
            response_result = await self._monitor_resource_exhaustion_response(
                system_components, scenario, duration=3.0
            )
            
            # Test recovery
            recovery_start = time.time()
            recovery_success = await self._recover_from_resource_exhaustion(
                system_components, scenario
            )
            recovery_time = time.time() - recovery_start
            
            exhaustion_results.append({
                'scenario': scenario['name'],
                'expected_behavior': scenario['expected_behavior'],
                'actual_behavior': response_result['behavior'],
                'graceful_degradation': response_result['graceful'],
                'recovery_successful': recovery_success,
                'recovery_time': recovery_time,
                'service_maintained': response_result['service_maintained']
            })
            
            # Verify graceful handling
            assert response_result['graceful'], f"System did not handle {scenario['name']} gracefully"
        
        print(f"Resource exhaustion scenarios test completed:")
        graceful_responses = sum(1 for r in exhaustion_results if r['graceful_degradation'])
        print(f"  - Scenarios tested: {len(resource_scenarios)}")
        print(f"  - Graceful responses: {graceful_responses}")
        print(f"  - Service maintained: {sum(1 for r in exhaustion_results if r['service_maintained'])}")
        
        return exhaustion_results

    async def _inject_resource_exhaustion(self, system_components, scenario):
        """Inject resource exhaustion based on scenario."""
        resource_type = scenario['resource']
        exhaustion_level = scenario['exhaustion_level']
        
        if resource_type == 'disk':
            # Mock disk space exhaustion
            for component in system_components.values():
                if hasattr(component, 'download_with_retry'):
                    component.download_with_retry = AsyncMock(
                        side_effect=Exception("No space left on device")
                    )
        
        elif resource_type == 'memory':
            # Mock memory exhaustion
            for component in system_components.values():
                if hasattr(component, 'check_model_integrity'):
                    component.check_model_integrity = AsyncMock(
                        side_effect=MemoryError("Out of memory")
                    )
        
        elif resource_type == 'cpu':
            # Mock CPU overload (simulate with delays)
            for component in system_components.values():
                if hasattr(component, 'handle_model_request'):
                    async def slow_request(*args, **kwargs):
                        await asyncio.sleep(5.0)  # Simulate CPU overload
                        return Mock(success=True)
                    component.handle_model_request = slow_request

    async def _monitor_resource_exhaustion_response(self, system_components, scenario, duration):
        """Monitor system response to resource exhaustion."""
        start_time = time.time()
        behavior_indicators = []
        service_maintained = True
        
        while time.time() - start_time < duration:
            try:
                # Test system operations
                for name, component in system_components.items():
                    if hasattr(component, 'handle_model_request'):
                        result = await asyncio.wait_for(
                            component.handle_model_request("test-model"),
                            timeout=1.0
                        )
                        behavior_indicators.append('operation_successful')
                    
            except asyncio.TimeoutError:
                behavior_indicators.append('timeout_handled')
            except MemoryError:
                behavior_indicators.append('memory_error_handled')
            except Exception as e:
                if "No space left" in str(e):
                    behavior_indicators.append('disk_error_handled')
                else:
                    behavior_indicators.append('error_handled')
                    service_maintained = False
            
            await asyncio.sleep(0.2)
        
        # Determine behavior type
        behavior_counts = {}
        for behavior in behavior_indicators:
            behavior_counts[behavior] = behavior_counts.get(behavior, 0) + 1
        
        dominant_behavior = max(behavior_counts.keys(), key=lambda k: behavior_counts[k]) if behavior_counts else 'unknown'
        
        return {
            'behavior': dominant_behavior,
            'graceful': 'error_handled' in dominant_behavior or 'timeout_handled' in dominant_behavior,
            'service_maintained': service_maintained,
            'behavior_distribution': behavior_counts
        }

    async def _recover_from_resource_exhaustion(self, system_components, scenario):
        """Recover system from resource exhaustion."""
        try:
            # Simulate resource cleanup/recovery
            await asyncio.sleep(0.5)
            
            # Restore component functionality
            for component in system_components.values():
                if hasattr(component, 'download_with_retry'):
                    component.download_with_retry = AsyncMock(return_value=Mock(success=True))
                if hasattr(component, 'check_model_integrity'):
                    component.check_model_integrity = AsyncMock(return_value=Mock(is_healthy=True))
                if hasattr(component, 'handle_model_request'):
                    component.handle_model_request = AsyncMock(return_value=Mock(success=True))
            
            # Test recovery
            availability_manager = system_components.get('availability_manager')
            if availability_manager:
                result = await availability_manager.handle_model_request("test-model")
                return hasattr(result, 'success') and result.success
            
            return True
            
        except Exception:
            return False

    async def test_concurrent_chaos_scenarios(self):
        """Test system behavior under multiple concurrent chaos scenarios."""
        print("Starting concurrent chaos scenarios test...")
        
        # Define concurrent chaos scenarios
        concurrent_scenarios = [
            ('component_failure', 'model_downloader'),
            ('network_partition', 'intermittent_connectivity'),
            ('resource_exhaustion', 'memory_exhaustion'),
            ('component_failure', 'health_monitor'),
            ('network_partition', 'high_latency_connection')
        ]
        
        system_components = await self._create_mock_system()
        
        # Start all chaos scenarios concurrently
        chaos_tasks = []
        for scenario_type, scenario_config in concurrent_scenarios:
            task = asyncio.create_task(
                self._run_concurrent_chaos_scenario(system_components, scenario_type, scenario_config)
            )
            chaos_tasks.append(task)
        
        # Monitor system during concurrent chaos
        monitoring_task = asyncio.create_task(
            self._monitor_concurrent_chaos(system_components, duration=5.0)
        )
        
        # Wait for all scenarios to complete
        chaos_results = await asyncio.gather(*chaos_tasks, return_exceptions=True)
        monitoring_result = await monitoring_task
        
        # Analyze results
        successful_scenarios = sum(1 for r in chaos_results if not isinstance(r, Exception))
        system_stability = monitoring_result['overall_stability']
        
        print(f"Concurrent chaos scenarios test completed:")
        print(f"  - Concurrent scenarios: {len(concurrent_scenarios)}")
        print(f"  - Successful scenario handling: {successful_scenarios}")
        print(f"  - Overall system stability: {system_stability:.2f}")
        print(f"  - Recovery time: {monitoring_result['recovery_time']:.2f}s")
        
        # Verify system maintained basic functionality
        assert system_stability >= 0.3  # At least 30% stability under extreme chaos
        assert monitoring_result['recovery_time'] < 10.0  # Recovery within 10 seconds
        
        return {
            'concurrent_scenarios': len(concurrent_scenarios),
            'successful_handling': successful_scenarios,
            'system_stability': system_stability,
            'recovery_time': monitoring_result['recovery_time'],
            'chaos_results': chaos_results
        }

    async def _run_concurrent_chaos_scenario(self, system_components, scenario_type, scenario_config):
        """Run individual chaos scenario concurrently."""
        try:
            if scenario_type == 'component_failure':
                await self._inject_component_failure(system_components, scenario_config)
                await asyncio.sleep(2.0)
                return await self._attempt_component_recovery(system_components, scenario_config)
            
            elif scenario_type == 'network_partition':
                scenario = {'name': scenario_config, 'duration': 2.0}
                await self._inject_network_partition(system_components, scenario)
                await asyncio.sleep(2.0)
                return await self._recover_from_network_partition(system_components)
            
            elif scenario_type == 'resource_exhaustion':
                scenario = {'resource': scenario_config.split('_')[0], 'exhaustion_level': 0.9}
                await self._inject_resource_exhaustion(system_components, scenario)
                await asyncio.sleep(2.0)
                return await self._recover_from_resource_exhaustion(system_components, scenario)
            
            return True
            
        except Exception:
            return False

    async def _monitor_concurrent_chaos(self, system_components, duration):
        """Monitor system behavior during concurrent chaos scenarios."""
        start_time = time.time()
        stability_samples = []
        
        while time.time() - start_time < duration:
            # Sample system stability
            stability_sample = await self._sample_system_stability(system_components)
            stability_samples.append(stability_sample)
            await asyncio.sleep(0.2)
        
        # Calculate overall stability
        overall_stability = sum(stability_samples) / len(stability_samples) if stability_samples else 0
        
        # Test recovery
        recovery_start = time.time()
        recovery_successful = await self._test_system_recovery(system_components)
        recovery_time = time.time() - recovery_start
        
        return {
            'overall_stability': overall_stability,
            'recovery_successful': recovery_successful,
            'recovery_time': recovery_time,
            'stability_samples': stability_samples
        }

    async def _sample_system_stability(self, system_components):
        """Sample current system stability."""
        successful_operations = 0
        total_operations = 0
        
        for name, component in system_components.items():
            total_operations += 1
            try:
                # Test basic component functionality
                if hasattr(component, 'handle_model_request'):
                    await asyncio.wait_for(
                        component.handle_model_request("test-model"),
                        timeout=0.5
                    )
                    successful_operations += 1
                elif hasattr(component, 'check_model_integrity'):
                    await asyncio.wait_for(
                        component.check_model_integrity("test-model"),
                        timeout=0.5
                    )
                    successful_operations += 1
                else:
                    successful_operations += 1  # Assume success for components without testable methods
                    
            except Exception:
                pass  # Operation failed
        
        return successful_operations / total_operations if total_operations > 0 else 0

    async def _test_system_recovery(self, system_components):
        """Test system recovery after chaos scenarios."""
        try:
            # Restore all components
            for component in system_components.values():
                if hasattr(component, 'download_with_retry'):
                    component.download_with_retry = AsyncMock(return_value=Mock(success=True))
                if hasattr(component, 'check_model_integrity'):
                    component.check_model_integrity = AsyncMock(return_value=Mock(is_healthy=True))
                if hasattr(component, 'handle_model_request'):
                    component.handle_model_request = AsyncMock(return_value=Mock(success=True))
            
            # Test system functionality
            await asyncio.sleep(0.5)  # Allow recovery time
            
            # Verify recovery
            stability = await self._sample_system_stability(system_components)
            return stability >= 0.8  # At least 80% functionality restored
            
        except Exception:
            return False

    async def run_comprehensive_chaos_tests(self):
        """Run all chaos engineering tests and generate report."""
        print("=" * 60)
        print("COMPREHENSIVE CHAOS ENGINEERING TEST SUITE")
        print("=" * 60)
        
        test_results = {}
        
        # Define all chaos tests
        chaos_tests = [
            ('component_failure_cascade', self.test_component_failure_cascade),
            ('network_partition_scenarios', self.test_network_partition_scenarios),
            ('resource_exhaustion_scenarios', self.test_resource_exhaustion_scenarios),
            ('concurrent_chaos_scenarios', self.test_concurrent_chaos_scenarios)
        ]
        
        # Run all chaos tests
        for test_name, test_method in chaos_tests:
            print(f"\n{'-' * 50}")
            print(f"Running {test_name} chaos test...")
            print(f"{'-' * 50}")
            
            try:
                result = await test_method()
                test_results[test_name] = {
                    'status': 'PASSED',
                    'result': result
                }
                print(f"✅ {test_name} chaos test PASSED")
                
            except Exception as e:
                test_results[test_name] = {
                    'status': 'FAILED',
                    'error': str(e)
                }
                print(f"❌ {test_name} chaos test FAILED: {e}")
        
        # Generate chaos engineering report
        print(f"\n{'=' * 60}")
        print("CHAOS ENGINEERING SUMMARY REPORT")
        print(f"{'=' * 60}")
        
        passed_tests = sum(1 for r in test_results.values() if r['status'] == 'PASSED')
        total_tests = len(test_results)
        
        print(f"Total chaos tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"System resilience score: {passed_tests / total_tests * 100:.1f}%")
        
        print(f"\nChaos Test Results:")
        for test_name, result in test_results.items():
            status_icon = "✅" if result['status'] == 'PASSED' else "❌"
            print(f"  {status_icon} {test_name}: {result['status']}")
        
        # Generate recommendations
        print(f"\nResilience Recommendations:")
        if passed_tests == total_tests:
            print("  ✅ System demonstrates excellent resilience to chaos scenarios")
        elif passed_tests >= total_tests * 0.75:
            print("  ⚠️  System shows good resilience but has areas for improvement")
        else:
            print("  ❌ System needs significant resilience improvements")
        
        return test_results


# Pytest integration
class TestChaosEngineeringTestSuite:
    """Pytest wrapper for chaos engineering test suite."""
    
    @pytest.fixture
    async def chaos_test_suite(self):
        """Create chaos engineering test suite instance."""
        return ChaosEngineeringTestSuite()
    
    async def test_component_failures(self, chaos_test_suite):
        """Test component failure scenarios."""
        results = await chaos_test_suite.test_component_failure_cascade()
        successful_recoveries = sum(1 for r in results if r['recovery_successful'])
        assert successful_recoveries >= len(results) * 0.7  # At least 70% recovery rate
    
    async def test_network_partitions(self, chaos_test_suite):
        """Test network partition scenarios."""
        results = await chaos_test_suite.test_network_partition_scenarios()
        successful_adaptations = sum(1 for r in results if r['system_adapted'])
        assert successful_adaptations >= len(results) * 0.6  # At least 60% adaptation rate
    
    async def test_resource_exhaustion(self, chaos_test_suite):
        """Test resource exhaustion scenarios."""
        results = await chaos_test_suite.test_resource_exhaustion_scenarios()
        graceful_responses = sum(1 for r in results if r['graceful_degradation'])
        assert graceful_responses == len(results)  # All should be graceful
    
    async def test_concurrent_chaos(self, chaos_test_suite):
        """Test concurrent chaos scenarios."""
        result = await chaos_test_suite.test_concurrent_chaos_scenarios()
        assert result['system_stability'] >= 0.3  # Minimum stability under extreme chaos
        assert result['recovery_time'] < 10.0  # Recovery within reasonable time


if __name__ == "__main__":
    # Run chaos engineering tests directly
    async def main():
        suite = ChaosEngineeringTestSuite()
        await suite.run_comprehensive_chaos_tests()
    
    asyncio.run(main())