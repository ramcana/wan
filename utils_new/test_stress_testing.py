#!/usr/bin/env python3
"""
Stress testing for WAN22 System Optimization
Tests system behavior under high load conditions and edge cases
"""

import unittest
import tempfile
import threading
import time
import random
from unittest.mock import patch, MagicMock
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import optimization components
from vram_manager import VRAMManager, GPUInfo, VRAMUsage
from health_monitor import HealthMonitor, SystemMetrics, SafetyThresholds
from error_recovery_system import ErrorRecoverySystem, SystemState
from model_loading_manager import ModelLoadingManager
from quantization_controller import QuantizationController, ModelInfo


class TestStressTesting(unittest.TestCase):
    """Stress tests for system optimization components"""
    
    def setUp(self):
        """Set up stress testing environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.results = []
        self.errors = []
    
    def tearDown(self):
        """Clean up stress testing environment"""
        import shutil
shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_concurrent_vram_monitoring(self):
        """Test VRAM monitoring under concurrent access"""
        vram_manager = VRAMManager()
        
        # Mock GPU detection
        with patch.object(vram_manager, '_detect_via_nvml') as mock_nvml:
            mock_gpu = GPUInfo(0, "RTX 4080", 16384, "531.79")
            mock_nvml.return_value = [mock_gpu]
            vram_manager.detect_vram_capacity()
        
        # Mock memory usage with varying values
        def mock_memory_usage(gpu_index):
            usage_percent = random.uniform(50.0, 95.0)
            used_mb = int(16384 * usage_percent / 100)
            free_mb = 16384 - used_mb
            return VRAMUsage(
                gpu_index=gpu_index,
                used_mb=used_mb,
                free_mb=free_mb,
                total_mb=16384,
                usage_percent=usage_percent,
                timestamp=datetime.now()
            )
        
        # Concurrent VRAM monitoring
        def monitor_vram(thread_id):
            try:
                for i in range(50):  # 50 iterations per thread
                    with patch.object(vram_manager, '_get_gpu_memory_usage', side_effect=mock_memory_usage):
                        usage = vram_manager.get_current_vram_usage()
                        if usage:
                            self.results.append(f"Thread-{thread_id}: {usage[0].usage_percent:.1f}%")
                    time.sleep(0.01)  # Small delay
                return True
            except Exception as e:
                self.errors.append(f"Thread-{thread_id}: {e}")
                return False
        
        # Run 10 concurrent threads
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(monitor_vram, i) for i in range(10)]
            results = [future.result() for future in as_completed(futures)]
        
        # Verify all threads completed successfully
        self.assertEqual(len(results), 10)
        self.assertTrue(all(results))
        self.assertEqual(len(self.errors), 0)
        self.assertGreater(len(self.results), 400)  # 10 threads * 50 iterations
    
    def test_health_monitoring_stress(self):
        """Test health monitoring under stress conditions"""
        health_monitor = HealthMonitor(
            monitoring_interval=0.01,  # Very fast monitoring
            thresholds=SafetyThresholds(
                gpu_temperature_critical=85.0,
                vram_usage_critical=90.0
            )
        )
        
        alerts_generated = []
        workload_reductions = []
        
        def alert_callback(alert):
            alerts_generated.append(alert)
        
        def workload_callback(reason, value):
            workload_reductions.append((reason, value))
        
        health_monitor.add_alert_callback(alert_callback)
        health_monitor.add_workload_reduction_callback(workload_callback)
        
        # Generate stress metrics
        def generate_stress_metrics():
            metrics_list = []
            for i in range(100):
                # Gradually increase stress
                stress_factor = i / 100.0
                
                metrics = SystemMetrics(
                    timestamp=datetime.now(),
                    gpu_temperature=70.0 + (stress_factor * 25.0),  # 70-95°C
                    gpu_utilization=50.0 + (stress_factor * 45.0),  # 50-95%
                    vram_usage_mb=int(8192 + (stress_factor * 8192)),  # 8-16GB
                    vram_total_mb=16384,
                    vram_usage_percent=50.0 + (stress_factor * 45.0),  # 50-95%
                    cpu_usage_percent=40.0 + (stress_factor * 50.0),  # 40-90%
                    memory_usage_gb=16.0 + (stress_factor * 16.0),  # 16-32GB
                    memory_total_gb=32.0,
                    memory_usage_percent=50.0 + (stress_factor * 40.0),  # 50-90%
                    disk_usage_percent=60.0 + (stress_factor * 30.0)  # 60-90%
                )
                metrics_list.append(metrics)
            return metrics_list
        
        stress_metrics = generate_stress_metrics()
        
        # Process metrics rapidly
        for metrics in stress_metrics:
            health_monitor._check_safety_thresholds(metrics)
            health_monitor.metrics_history.append(metrics)
        
        # Verify stress handling
        self.assertGreater(len(alerts_generated), 0)
        self.assertGreater(len(workload_reductions), 0)
        
        # Verify critical alerts were generated
        critical_alerts = [alert for alert in alerts_generated if alert.severity == 'critical']
        self.assertGreater(len(critical_alerts), 0)
        
        # Verify workload reduction was triggered
        gpu_temp_reductions = [wr for wr in workload_reductions if wr[0] == 'gpu_temperature']
        self.assertGreater(len(gpu_temp_reductions), 0)
    
    def test_error_recovery_cascade(self):
        """Test error recovery under cascading failure conditions"""
        error_recovery = ErrorRecoverySystem(
            state_dir=str(Path(self.temp_dir) / "recovery_states"),
            log_dir=str(Path(self.temp_dir) / "logs"),
            max_recovery_attempts=3
        )
        
        recovery_attempts = []
        
        # Register cascading error handler
        def cascading_handler(error, context):
            from error_recovery_system import RecoveryResult, RecoveryStrategy
            recovery_attempts.append((error.__class__.__name__, context.recovery_attempts))
            
            # Fail first two attempts, succeed on third
            if context.recovery_attempts < 3:
                return RecoveryResult(
                    success=False,
                    strategy_used=RecoveryStrategy.IMMEDIATE_RETRY,
                    actions_taken=[f"Attempt {context.recovery_attempts} failed"],
                    time_taken=0.1,
                    error_resolved=False,
                    fallback_applied=False,
                    user_intervention_required=False,
                    recovery_message="Retry failed",
                    warnings=[]
                )
            else:
                return RecoveryResult(
                    success=True,
                    strategy_used=RecoveryStrategy.FALLBACK_CONFIG,
                    actions_taken=["Applied fallback after multiple attempts"],
                    time_taken=0.5,
                    error_resolved=True,
                    fallback_applied=True,
                    user_intervention_required=False,
                    recovery_message="Recovery successful",
                    warnings=[]
                )
        
        error_recovery.register_error_handler(RuntimeError, cascading_handler)
        
        # Simulate cascading failures
        cascade_errors = [
            RuntimeError("Primary system failure"),
            RuntimeError("Secondary system failure"),
            RuntimeError("Tertiary system failure")
        ]
        
        final_results = []
        for i, error in enumerate(cascade_errors):
            result = error_recovery.attempt_recovery(error, component=f"system_{i}")
            final_results.append(result.success)
        
        # Verify recovery behavior
        self.assertEqual(len(recovery_attempts), 9)  # 3 errors * 3 attempts each
        
        # Last attempt should succeed
        self.assertTrue(final_results[-1])
    
    @patch('model_loading_manager.DEPENDENCIES_AVAILABLE', True)
    def test_concurrent_model_loading(self):
        """Test concurrent model loading operations"""
        model_manager = ModelLoadingManager(
            cache_dir=str(Path(self.temp_dir) / "model_cache"),
            enable_logging=False
        )
        
        loading_results = []
        loading_times = []
        
        def load_model_concurrent(model_id):
            start_time = time.time()
            try:
                with patch.object(model_manager, '_validate_model_path', return_value=True):
                    with patch.object(model_manager, '_get_memory_usage', side_effect=[1024.0, 4096.0]):
                        with patch('model_loading_manager.DiffusionPipeline') as mock_pipeline:
                            mock_pipeline.from_pretrained.return_value = MagicMock()
                            
                            result = model_manager.load_model(f"test/model-{model_id}")
                            loading_results.append(result.success)
                            loading_times.append(time.time() - start_time)
                            return result.success
            except Exception as e:
                self.errors.append(f"Model-{model_id}: {e}")
                return False
        
        # Load 5 models concurrently
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(load_model_concurrent, i) for i in range(5)]
            concurrent_results = [future.result() for future in as_completed(futures)]
        
        # Verify concurrent loading
        self.assertEqual(len(concurrent_results), 5)
        self.assertTrue(all(concurrent_results))
        self.assertEqual(len(self.errors), 0)
        
        # Verify reasonable loading times (should be concurrent, not sequential)
        max_loading_time = max(loading_times)
        total_sequential_time = sum(loading_times)
        
        # Concurrent loading should be much faster than sequential
        self.assertLess(max_loading_time * 2, total_sequential_time)
    
    def test_memory_pressure_simulation(self):
        """Test system behavior under memory pressure"""
        # Simulate increasing memory pressure
        memory_pressure_levels = [50, 70, 85, 90, 95, 98]  # Percentage
        
        health_monitor = HealthMonitor(
            thresholds=SafetyThresholds(
                memory_usage_warning=80.0,
                memory_usage_critical=90.0,
                vram_usage_warning=85.0,
                vram_usage_critical=95.0
            )
        )
        
        pressure_alerts = []
        
        def pressure_alert_callback(alert):
            pressure_alerts.append((alert.component, alert.metric, alert.current_value))
        
        health_monitor.add_alert_callback(pressure_alert_callback)
        
        # Simulate memory pressure scenarios
        for pressure in memory_pressure_levels:
            metrics = SystemMetrics(
                timestamp=datetime.now(),
                gpu_temperature=75.0,
                gpu_utilization=80.0,
                vram_usage_mb=int(16384 * pressure / 100),
                vram_total_mb=16384,
                vram_usage_percent=pressure,
                cpu_usage_percent=60.0,
                memory_usage_gb=32.0 * pressure / 100,
                memory_total_gb=32.0,
                memory_usage_percent=pressure,
                disk_usage_percent=70.0
            )
            
            health_monitor._check_safety_thresholds(metrics)
        
        # Verify pressure handling
        self.assertGreater(len(pressure_alerts), 0)
        
        # Should have both memory and VRAM alerts at high pressure
        memory_alerts = [alert for alert in pressure_alerts if alert[0] == 'memory']
        vram_alerts = [alert for alert in pressure_alerts if alert[0] == 'gpu' and alert[1] == 'vram_usage']
        
        self.assertGreater(len(memory_alerts), 0)
        self.assertGreater(len(vram_alerts), 0)
    
    def test_quantization_strategy_stress(self):
        """Test quantization strategy determination under various conditions"""
        quantization_controller = QuantizationController(
            config_path=str(Path(self.temp_dir) / "config.json"),
            preferences_path=str(Path(self.temp_dir) / "prefs.json")
        )
        
        # Create various model scenarios
        model_scenarios = [
            ModelInfo("tiny-model", 1.0, "simple", ["unet"], 1024.0),
            ModelInfo("small-model", 3.0, "stable-diffusion", ["unet", "text_encoder"], 3072.0),
            ModelInfo("medium-model", 7.0, "stable-diffusion", ["unet", "text_encoder", "vae"], 7168.0),
            ModelInfo("large-model", 15.0, "stable-diffusion-xl", ["unet", "text_encoder", "text_encoder_2", "vae"], 15360.0),
            ModelInfo("huge-model", 30.0, "video-diffusion", ["transformer", "text_encoder", "vae"], 30720.0)
        ]
        
        strategy_results = []
        compatibility_results = []
        
        # Test all quantization methods with all models
        from quantization_controller import QuantizationMethod
        methods = [QuantizationMethod.NONE, QuantizationMethod.FP16, QuantizationMethod.BF16, QuantizationMethod.INT8]
        
        for model in model_scenarios:
            for method in methods:
                try:
                    # Test strategy determination
                    strategy = quantization_controller.determine_optimal_strategy(model)
                    strategy_results.append((model.name, method.value, strategy.method.value))
                    
                    # Test compatibility
                    compatibility = quantization_controller.validate_quantization_compatibility(model, method)
                    compatibility_results.append((model.name, method.value, compatibility["compatible"]))
                    
                except Exception as e:
                    self.errors.append(f"Strategy error for {model.name} with {method.value}: {e}")
        
        # Verify stress testing results
        self.assertEqual(len(self.errors), 0)
        self.assertEqual(len(strategy_results), len(model_scenarios))  # One strategy per model
        self.assertEqual(len(compatibility_results), len(model_scenarios) * len(methods))
        
        # Verify larger models tend toward more aggressive quantization
        large_model_strategies = [result for result in strategy_results if "large" in result[0] or "huge" in result[0]]
        aggressive_strategies = [result for result in large_model_strategies if result[2] in ["bf16", "int8"]]
        
        # At least some large models should use aggressive quantization
        self.assertGreater(len(aggressive_strategies), 0)
    
    def test_system_state_management_stress(self):
        """Test system state management under rapid state changes"""
        error_recovery = ErrorRecoverySystem(
            state_dir=str(Path(self.temp_dir) / "states"),
            log_dir=str(Path(self.temp_dir) / "logs")
        )
        
        # Generate rapid state changes
        state_operations = []
        
        def rapid_state_operations(thread_id):
            try:
                for i in range(20):  # 20 operations per thread
                    # Create varying system states
                    state = SystemState(
                        timestamp=datetime.now(),
                        active_model=f"model-{thread_id}-{i}",
                        configuration={"thread": thread_id, "iteration": i},
                        memory_usage={"gpu": random.randint(4096, 16384)},
                        gpu_state={"temp": random.uniform(60.0, 90.0)},
                        pipeline_state={"loaded": random.choice([True, False])},
                        user_preferences={"quantization": random.choice(["fp16", "bf16", "int8"])}
                    )
                    
                    # Save state
                    state_path = error_recovery.save_system_state(state, f"thread_{thread_id}_state_{i}")
                    state_operations.append(f"saved:{state_path}")
                    
                    # Occasionally restore a state
                    if i % 5 == 0 and i > 0:
                        with patch.object(error_recovery, '_apply_state_restoration', return_value=True):
                            restore_result = error_recovery.restore_system_state(state_path)
                            state_operations.append(f"restored:{restore_result.success}")
                
                return True
            except Exception as e:
                self.errors.append(f"State thread-{thread_id}: {e}")
                return False
        
        # Run concurrent state operations
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(rapid_state_operations, i) for i in range(5)]
            state_results = [future.result() for future in as_completed(futures)]
        
        # Verify state management stress test
        self.assertEqual(len(state_results), 5)
        self.assertTrue(all(state_results))
        self.assertEqual(len(self.errors), 0)
        self.assertGreater(len(state_operations), 80)  # 5 threads * 20 operations (some restores)
        
        # Verify state files were created
        state_files = list(Path(self.temp_dir).glob("states/*.json"))
        self.assertGreater(len(state_files), 50)  # Should have many state files


class TestEdgeCaseHandling(unittest.TestCase):
    """Test handling of edge cases and boundary conditions"""
    
    def setUp(self):
        """Set up edge case testing environment"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up edge case testing environment"""
        import shutil
shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_zero_vram_handling(self):
        """Test handling of zero VRAM scenarios"""
        vram_manager = VRAMManager()
        
        # Mock zero VRAM detection
        with patch.object(vram_manager, '_detect_via_nvml') as mock_nvml:
            mock_gpu = GPUInfo(0, "Integrated GPU", 0, "Unknown")  # 0 VRAM
            mock_nvml.return_value = [mock_gpu]
            
            gpus = vram_manager.detect_vram_capacity()
            self.assertEqual(len(gpus), 1)
            self.assertEqual(gpus[0].total_memory_mb, 0)
        
        # Test memory usage with zero VRAM
        with patch.object(vram_manager, '_get_gpu_memory_usage') as mock_usage:
            mock_usage.return_value = VRAMUsage(0, 0, 0, 0, 0.0, datetime.now())
            
            usage = vram_manager.get_current_vram_usage()
            self.assertEqual(len(usage), 1)
            self.assertEqual(usage[0].usage_percent, 0.0)
    
    def test_extreme_temperature_handling(self):
        """Test handling of extreme temperature values"""
        health_monitor = HealthMonitor(
            thresholds=SafetyThresholds(
                gpu_temperature_warning=80.0,
                gpu_temperature_critical=90.0
            )
        )
        
        extreme_alerts = []
        
        def extreme_alert_callback(alert):
            extreme_alerts.append(alert)
        
        health_monitor.add_alert_callback(extreme_alert_callback)
        
        # Test extreme temperature values
        extreme_temps = [-10.0, 0.0, 150.0, 200.0, 999.0]
        
        for temp in extreme_temps:
            metrics = SystemMetrics(
                timestamp=datetime.now(),
                gpu_temperature=temp,
                gpu_utilization=50.0,
                vram_usage_mb=8192,
                vram_total_mb=16384,
                vram_usage_percent=50.0,
                cpu_usage_percent=50.0,
                memory_usage_gb=16.0,
                memory_total_gb=32.0,
                memory_usage_percent=50.0,
                disk_usage_percent=70.0
            )
            
            # Should not crash on extreme values
            health_monitor._check_safety_thresholds(metrics)
        
        # Should generate alerts for extreme high temperatures
        high_temp_alerts = [alert for alert in extreme_alerts if alert.current_value > 100.0]
        self.assertGreater(len(high_temp_alerts), 0)
    
    def test_invalid_model_paths_handling(self):
        """Test handling of various invalid model paths"""
        model_manager = ModelLoadingManager(enable_logging=False)
        
        invalid_paths = [
            "",  # Empty path
            "/nonexistent/path/to/model",  # Non-existent absolute path
            "relative/nonexistent/path",  # Non-existent relative path
            "invalid://protocol/path",  # Invalid protocol
            "model with spaces and special chars!@#$%",  # Special characters
            "a" * 1000,  # Extremely long path
            None,  # None value
        ]
        
        for path in invalid_paths:
            if path is not None:
                is_valid = model_manager._validate_model_path(path)
                # Most should be invalid, but shouldn't crash
                self.assertIsInstance(is_valid, bool)
    
    def test_corrupted_cache_handling(self):
        """Test handling of corrupted cache files"""
        model_manager = ModelLoadingManager(
            cache_dir=str(Path(self.temp_dir) / "corrupted_cache"),
            enable_logging=False
        )
        
        # Create corrupted cache file
        cache_file = Path(model_manager.cache_dir) / "parameter_cache.json"
        cache_file.parent.mkdir(exist_ok=True)
        
        # Write invalid JSON
        with open(cache_file, 'w') as f:
            f.write('{"invalid": json, "missing": quotes}')
        
        # Should handle corrupted cache gracefully
        model_manager._load_parameter_cache()
        
        # Cache should be empty after failed load
        self.assertEqual(len(model_manager._parameter_cache), 0)
    
    def test_boundary_value_quantization(self):
        """Test quantization with boundary values"""
        quantization_controller = QuantizationController(
            config_path=str(Path(self.temp_dir) / "config.json"),
            preferences_path=str(Path(self.temp_dir) / "prefs.json")
        )
        
        # Test boundary model sizes
        boundary_models = [
            ModelInfo("zero-model", 0.0, "empty", [], 0.0),  # Zero size
            ModelInfo("tiny-model", 0.001, "minimal", ["unet"], 1.0),  # Minimal size
            ModelInfo("huge-model", 1000.0, "massive", ["transformer"] * 100, 1000000.0),  # Massive size
        ]
        
        for model in boundary_models:
            try:
                strategy = quantization_controller.determine_optimal_strategy(model)
                self.assertIsNotNone(strategy)
                self.assertIsNotNone(strategy.method)
                
                # Test compatibility
                from quantization_controller import QuantizationMethod
                compatibility = quantization_controller.validate_quantization_compatibility(
                    model, QuantizationMethod.BF16
                )
                self.assertIsInstance(compatibility, dict)
                self.assertIn("compatible", compatibility)
                
            except Exception as e:
                self.fail(f"Boundary model {model.name} caused exception: {e}")


if __name__ == '__main__':
    unittest.main()