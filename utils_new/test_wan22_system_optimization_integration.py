#!/usr/bin/env python3
"""
Integration tests for WAN22 System Optimization
Tests end-to-end optimization workflows and cross-component interactions
"""

import unittest
import tempfile
import json
import time
import threading
from unittest.mock import patch, MagicMock, PropertyMock
from pathlib import Path
from datetime import datetime

# Import all optimization components
from syntax_validator import SyntaxValidator
from vram_manager import VRAMManager, GPUInfo
from quantization_controller import QuantizationController, QuantizationMethod
from config_validator import ConfigValidator
from hardware_optimizer import HardwareOptimizer, HardwareProfile
from error_recovery_system import ErrorRecoverySystem, RecoveryStrategy
from health_monitor import HealthMonitor, SafetyThresholds
from model_loading_manager import ModelLoadingManager


class TestWAN22SystemOptimizationIntegration(unittest.TestCase):
    """Integration tests for the complete WAN22 system optimization workflow"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_config = {
            "system": {
                "default_quantization": "bf16",
                "enable_offload": True,
                "vae_tile_size": 256
            },
            "directories": {
                "output_directory": str(Path(self.temp_dir) / "output"),
                "models_directory": str(Path(self.temp_dir) / "models")
            },
            "generation": {
                "default_resolution": "512x512",
                "default_steps": 20
            }
        }
        
        # Create test config file
        self.config_path = Path(self.temp_dir) / "config.json"
        with open(self.config_path, 'w') as f:
            json.dump(self.test_config, f)
        
        # Initialize components
        self.syntax_validator = SyntaxValidator(backup_dir=str(Path(self.temp_dir) / "syntax_backups"))
        self.vram_manager = VRAMManager(config_path=str(Path(self.temp_dir) / "vram_config.json"))
        self.quantization_controller = QuantizationController(
            config_path=str(self.config_path),
            preferences_path=str(Path(self.temp_dir) / "quant_prefs.json")
        )
        self.config_validator = ConfigValidator(backup_dir=Path(self.temp_dir) / "config_backups")
        self.hardware_optimizer = HardwareOptimizer(config_path=str(self.config_path))
        self.error_recovery = ErrorRecoverySystem(
            state_dir=str(Path(self.temp_dir) / "recovery_states"),
            log_dir=str(Path(self.temp_dir) / "logs")
        )
        self.health_monitor = HealthMonitor(
            monitoring_interval=0.1,
            thresholds=SafetyThresholds(gpu_temperature_critical=85.0)
        )
        self.model_loading_manager = ModelLoadingManager(
            cache_dir=str(Path(self.temp_dir) / "model_cache"),
            enable_logging=False
        )
    
    def tearDown(self):
        """Clean up integration test environment"""
        import shutil
        if hasattr(self.health_monitor, 'stop_monitoring'):
            self.health_monitor.stop_monitoring()
        shutil.rmtree(self.temp_dir, ignore_errors=True)    d
ef test_end_to_end_optimization_workflow(self):
        """Test complete end-to-end optimization workflow"""
        # Step 1: Validate and clean configuration
        config_result = self.config_validator.validate_config_file(self.config_path)
        self.assertTrue(config_result.is_valid or len(config_result.cleaned_attributes) > 0)
        
        # Step 2: Detect hardware profile
        with patch('hardware_optimizer.torch.cuda.is_available', return_value=True):
            with patch('hardware_optimizer.torch.cuda.get_device_name', return_value="NVIDIA GeForce RTX 4080"):
                with patch('hardware_optimizer.psutil.cpu_count', return_value=16):
                    profile = self.hardware_optimizer.detect_hardware_profile()
        
        self.assertIsInstance(profile, HardwareProfile)
        
        # Step 3: Detect VRAM capacity
        with patch.object(self.vram_manager, '_detect_via_nvml') as mock_nvml:
            mock_gpu = GPUInfo(0, "RTX 4080", 16384, "531.79")
            mock_nvml.return_value = [mock_gpu]
            
            gpus = self.vram_manager.detect_vram_capacity()
            self.assertEqual(len(gpus), 1)
            self.assertEqual(gpus[0].total_memory_mb, 16384)
        
        # Step 4: Apply hardware optimizations
        with patch('hardware_optimizer.torch.cuda.is_available', return_value=True):
            optimization_result = self.hardware_optimizer.apply_rtx_4080_optimizations(profile)
            self.assertTrue(optimization_result.success)
            self.assertGreater(len(optimization_result.optimizations_applied), 0)
        
        # Step 5: Configure quantization strategy
        from quantization_controller import ModelInfo
        model_info = ModelInfo(
            name="test-model",
            size_gb=5.0,
            architecture="stable-diffusion",
            components=["unet", "text_encoder", "vae"],
            estimated_vram_usage=8192.0
        )
        
        strategy = self.quantization_controller.determine_optimal_strategy(model_info)
        self.assertIn(strategy.method, [QuantizationMethod.BF16, QuantizationMethod.FP16])
        
        # Step 6: Start health monitoring
        with patch.object(self.health_monitor, '_collect_metrics', return_value=None):
            self.health_monitor.start_monitoring()
            self.assertTrue(self.health_monitor.is_monitoring)
            self.health_monitor.stop_monitoring()
    
    def test_error_recovery_integration(self):
        """Test error recovery integration with other components"""
        # Register error handlers for different components
        def vram_error_handler(error, context):
            from error_recovery_system import RecoveryResult, RecoveryStrategy
            return RecoveryResult(
                success=True,
                strategy_used=RecoveryStrategy.FALLBACK_CONFIG,
                actions_taken=["Applied VRAM fallback configuration"],
                time_taken=1.0,
                error_resolved=True,
                fallback_applied=True,
                user_intervention_required=False,
                recovery_message="VRAM error recovered",
                warnings=[]
            )
        
        self.error_recovery.register_error_handler(RuntimeError, vram_error_handler)
        
        # Simulate VRAM detection error
        vram_error = RuntimeError("VRAM detection failed")
        recovery_result = self.error_recovery.attempt_recovery(vram_error, component="vram_manager")
        
        self.assertTrue(recovery_result.success)
        self.assertTrue(recovery_result.error_resolved)
        self.assertEqual(recovery_result.strategy_used, RecoveryStrategy.FALLBACK_CONFIG)
    
    def test_health_monitoring_with_workload_reduction(self):
        """Test health monitoring integration with workload reduction"""
        workload_reductions = []
        
        def workload_reduction_callback(reason, value):
            workload_reductions.append((reason, value))
        
        self.health_monitor.add_workload_reduction_callback(workload_reduction_callback)
        
        # Simulate high GPU temperature
        from health_monitor import SystemMetrics
        critical_metrics = SystemMetrics(
            timestamp=datetime.now(),
            gpu_temperature=90.0,  # Above critical threshold
            gpu_utilization=95.0,
            vram_usage_mb=15360,  # 93.75% of 16GB
            vram_total_mb=16384,
            vram_usage_percent=93.75,
            cpu_usage_percent=85.0,
            memory_usage_gb=28.8,
            memory_total_gb=32.0,
            memory_usage_percent=90.0,
            disk_usage_percent=75.0
        )
        
        self.health_monitor._check_safety_thresholds(critical_metrics)
        
        # Should trigger multiple workload reductions
        self.assertGreater(len(workload_reductions), 0)
        self.assertTrue(any(reason == 'gpu_temperature' for reason, value in workload_reductions))
    
    @patch('model_loading_manager.DEPENDENCIES_AVAILABLE', True)
    @patch('model_loading_manager.DiffusionPipeline')
    def test_model_loading_with_optimization_integration(self, mock_pipeline_class):
        """Test model loading integration with optimization components"""
        # Mock successful model loading
        mock_pipeline = MagicMock()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline
        
        # Set up progress tracking
        progress_updates = []
        def progress_callback(progress):
            progress_updates.append(progress.phase)
        
        self.model_loading_manager.add_progress_callback(progress_callback)
        
        # Load model with optimization settings
        with patch.object(self.model_loading_manager, '_validate_model_path', return_value=True):
            with patch.object(self.model_loading_manager, '_get_memory_usage', side_effect=[1024.0, 8192.0]):
                result = self.model_loading_manager.load_model(
                    "test/optimized-model",
                    torch_dtype="float16",
                    device_map="auto"
                )
        
        self.assertTrue(result.success)
        self.assertGreater(len(progress_updates), 0)
        self.assertEqual(result.memory_usage_mb, 7168.0)  # 8192 - 1024
    
    def test_quantization_with_vram_monitoring_integration(self):
        """Test quantization integration with VRAM monitoring"""
        # Set up VRAM monitoring
        with patch.object(self.vram_manager, '_detect_via_nvml') as mock_nvml:
            mock_gpu = GPUInfo(0, "RTX 4080", 16384, "531.79")
            mock_nvml.return_value = [mock_gpu]
            
            gpus = self.vram_manager.detect_vram_capacity()
            optimal_gpu = self.vram_manager.select_optimal_gpu()
            
            self.assertIsNotNone(optimal_gpu)
            self.assertEqual(optimal_gpu.total_memory_mb, 16384)
        
        # Test quantization compatibility with detected VRAM
        from quantization_controller import ModelInfo
        model_info = ModelInfo(
            name="large-model",
            size_gb=12.0,  # Large model
            architecture="stable-diffusion",
            components=["unet", "text_encoder", "vae"],
            estimated_vram_usage=15360.0  # Close to VRAM limit
        )
        
        compatibility = self.quantization_controller.validate_quantization_compatibility(
            model_info, QuantizationMethod.NONE
        )
        
        # Should warn about high VRAM usage
        self.assertTrue(any("exceeds 90%" in warning for warning in compatibility["warnings"]))
        
        # Should recommend more aggressive quantization
        self.assertTrue(any("more aggressive" in rec for rec in compatibility["recommendations"]))
    
    def test_syntax_validation_with_error_recovery_integration(self):
        """Test syntax validation integration with error recovery"""
        # Create a Python file with syntax error
        test_file = Path(self.temp_dir) / "test_syntax.py"
        invalid_code = """
def test_function():
    x = [item if condition]  # Missing else clause
    return x
"""
        test_file.write_text(invalid_code)
        
        # Validate file
        validation_result = self.syntax_validator.validate_file(str(test_file))
        self.assertFalse(validation_result.is_valid)
        
        # Simulate error recovery for syntax errors
        syntax_error = SyntaxError("expected 'else'")
        recovery_result = self.error_recovery.attempt_recovery(
            syntax_error, 
            component="syntax_validator"
        )
        
        # Should attempt recovery (even if no specific handler registered)
        self.assertIsNotNone(recovery_result)
    
    def test_hardware_optimization_with_health_monitoring_integration(self):
        """Test hardware optimization integration with health monitoring"""
        # Create RTX 4080 profile
        rtx_profile = HardwareProfile(
            cpu_model="AMD Ryzen 9 7950X",
            cpu_cores=16,
            total_memory_gb=64,
            gpu_model="NVIDIA GeForce RTX 4080",
            vram_gb=16,
            cuda_version="12.1",
            driver_version="531.79",
            is_rtx_4080=True,
            is_threadripper_pro=False
        )
        
        # Apply optimizations
        with patch('hardware_optimizer.torch.cuda.is_available', return_value=True):
            optimization_result = self.hardware_optimizer.apply_rtx_4080_optimizations(rtx_profile)
            self.assertTrue(optimization_result.success)
        
        # Start health monitoring with optimized settings
        optimized_thresholds = SafetyThresholds(
            gpu_temperature_warning=75.0,  # More aggressive for RTX 4080
            gpu_temperature_critical=80.0,
            vram_usage_warning=85.0,
            vram_usage_critical=95.0
        )
        
        self.health_monitor.update_thresholds(optimized_thresholds)
        
        # Verify thresholds were updated
        self.assertEqual(self.health_monitor.thresholds.gpu_temperature_warning, 75.0)
        self.assertEqual(self.health_monitor.thresholds.gpu_temperature_critical, 80.0)
    
    def test_config_validation_with_component_integration(self):
        """Test configuration validation integration with other components"""
        # Create config with issues that affect multiple components
        problematic_config = {
            "system": {
                "default_quantization": "invalid_method",  # Invalid quantization
                "enable_offload": "true",  # Wrong type
                "vae_tile_size": 2048  # Too high
            },
            "directories": {
                "output_directory": "/nonexistent/path",
                "models_directory": "/another/nonexistent/path"
            },
            "clip_output": True,  # Should be cleaned
            "unknown_section": {
                "unknown_property": "value"
            }
        }
        
        problem_config_path = Path(self.temp_dir) / "problem_config.json"
        with open(problem_config_path, 'w') as f:
            json.dump(problematic_config, f)
        
        # Validate and clean
        validation_result = self.config_validator.validate_config_file(problem_config_path)
        
        # Should have cleaned attributes
        self.assertGreater(len(validation_result.cleaned_attributes), 0)
        
        # Should have validation errors
        self.assertTrue(validation_result.has_errors())
        
        # Should have created backup
        self.assertIsNotNone(validation_result.backup_path)
        self.assertTrue(Path(validation_result.backup_path).exists())
    
    def test_cross_component_state_management(self):
        """Test state management across multiple components"""
        # Create system state with data from multiple components
        from error_recovery_system import SystemState
        
        # Simulate state from various components
        system_state = SystemState(
            timestamp=datetime.now(),
            active_model="stable-diffusion-2-1",
            configuration=self.test_config,
            memory_usage={
                "gpu_vram_mb": 12288,
                "system_memory_gb": 24.0,
                "model_cache_mb": 2048
            },
            gpu_state={
                "temperature": 72.0,
                "utilization": 85.0,
                "power_usage": 280.0
            },
            pipeline_state={
                "loaded": True,
                "quantization_method": "bf16",
                "offload_enabled": True,
                "components": ["unet", "text_encoder", "vae"]
            },
            user_preferences={
                "preferred_quantization": "bf16",
                "enable_monitoring": True,
                "auto_optimization": True
            }
        )
        
        # Save state
        state_path = self.error_recovery.save_system_state(system_state, "integration_test")
        self.assertTrue(Path(state_path).exists())
        
        # Restore state
        with patch.object(self.error_recovery, '_apply_state_restoration', return_value=True):
            restore_result = self.error_recovery.restore_system_state(state_path)
            self.assertTrue(restore_result.success)
    
    def test_performance_optimization_workflow(self):
        """Test complete performance optimization workflow"""
        # Step 1: Hardware detection and optimization
        with patch('hardware_optimizer.torch.cuda.is_available', return_value=True):
            with patch('hardware_optimizer.torch.cuda.get_device_name', return_value="NVIDIA GeForce RTX 4080"):
                profile = self.hardware_optimizer.detect_hardware_profile()
                optimization_result = self.hardware_optimizer.apply_hardware_optimizations(profile)
        
        self.assertTrue(optimization_result.success)
        
        # Step 2: VRAM optimization
        with patch.object(self.vram_manager, '_detect_via_nvml') as mock_nvml:
            mock_gpu = GPUInfo(0, "RTX 4080", 16384, "531.79")
            mock_nvml.return_value = [mock_gpu]
            
            gpus = self.vram_manager.detect_vram_capacity()
            memory_settings = self.hardware_optimizer.get_memory_optimization_settings(16)  # 16GB
        
        # Should use RTX 4080 optimized settings
        self.assertEqual(memory_settings['batch_size'], 2)
        self.assertEqual(memory_settings['vae_tile_size'], (256, 256))
        self.assertFalse(memory_settings['enable_attention_slicing'])  # Not needed for 16GB
        
        # Step 3: Quantization optimization
        from quantization_controller import ModelInfo
        model_info = ModelInfo(
            name="optimized-model",
            size_gb=5.0,
            architecture="stable-diffusion",
            components=["unet", "text_encoder", "vae"],
            estimated_vram_usage=6144.0
        )
        
        strategy = self.quantization_controller.determine_optimal_strategy(model_info)
        compatibility = self.quantization_controller.validate_quantization_compatibility(
            model_info, strategy.method
        )
        
        self.assertTrue(compatibility["compatible"])
        self.assertLess(compatibility["estimated_memory_usage"], 8192)  # Should be optimized
    
    def test_stress_testing_integration(self):
        """Test system behavior under stress conditions"""
        # Simulate high load conditions
        stress_metrics = []
        
        # Create multiple concurrent operations
        def simulate_model_loading():
            with patch('model_loading_manager.DEPENDENCIES_AVAILABLE', True):
                with patch.object(self.model_loading_manager, '_validate_model_path', return_value=True):
                    with patch.object(self.model_loading_manager, '_get_memory_usage', side_effect=[1024.0, 8192.0]):
                        result = self.model_loading_manager.load_model("stress/test/model")
                        stress_metrics.append(("model_loading", result.success))
        
        def simulate_vram_monitoring():
            with patch.object(self.vram_manager, '_get_gpu_memory_usage') as mock_usage:
                from vram_manager import VRAMUsage
                mock_usage.return_value = VRAMUsage(0, 14336, 2048, 16384, 87.5, datetime.now())
                usage = self.vram_manager.get_current_vram_usage()
                stress_metrics.append(("vram_monitoring", len(usage) > 0))
        
        def simulate_health_monitoring():
            from health_monitor import SystemMetrics
            metrics = SystemMetrics(
                timestamp=datetime.now(),
                gpu_temperature=82.0,
                gpu_utilization=95.0,
                vram_usage_mb=14336,
                vram_total_mb=16384,
                vram_usage_percent=87.5,
                cpu_usage_percent=88.0,
                memory_usage_gb=28.0,
                memory_total_gb=32.0,
                memory_usage_percent=87.5,
                disk_usage_percent=80.0
            )
            self.health_monitor._check_safety_thresholds(metrics)
            stress_metrics.append(("health_monitoring", len(self.health_monitor.active_alerts) > 0))
        
        # Run concurrent operations
        threads = [
            threading.Thread(target=simulate_model_loading),
            threading.Thread(target=simulate_vram_monitoring),
            threading.Thread(target=simulate_health_monitoring)
        ]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join(timeout=5.0)
        
        # Verify all operations completed
        self.assertEqual(len(stress_metrics), 3)
        self.assertTrue(all(success for operation, success in stress_metrics))


class TestHardwareSimulationIntegration(unittest.TestCase):
    """Integration tests with different hardware configurations"""
    
    def setUp(self):
        """Set up hardware simulation test environment"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up hardware simulation test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_rtx_4080_configuration_integration(self):
        """Test complete integration for RTX 4080 configuration"""
        # Create RTX 4080 system
        hardware_optimizer = HardwareOptimizer()
        
        rtx_profile = HardwareProfile(
            cpu_model="AMD Ryzen 9 7950X",
            cpu_cores=16,
            total_memory_gb=64,
            gpu_model="NVIDIA GeForce RTX 4080",
            vram_gb=16,
            cuda_version="12.1",
            driver_version="531.79",
            is_rtx_4080=True,
            is_threadripper_pro=False
        )
        
        # Generate optimized settings
        settings = hardware_optimizer.generate_rtx_4080_settings(rtx_profile)
        
        # Verify RTX 4080 specific optimizations
        self.assertEqual(settings.vae_tile_size, (256, 256))  # As specified in requirements
        self.assertEqual(settings.batch_size, 2)  # For 16GB VRAM
        self.assertTrue(settings.enable_tensor_cores)
        self.assertTrue(settings.use_bf16)  # RTX 4080 supports BF16
        self.assertEqual(settings.memory_fraction, 0.9)
        
        # Test VRAM management for RTX 4080
        vram_manager = VRAMManager()
        with patch.object(vram_manager, '_detect_via_nvml') as mock_nvml:
            mock_gpu = GPUInfo(0, "NVIDIA GeForce RTX 4080", 16384, "531.79")
            mock_nvml.return_value = [mock_gpu]
            
            gpus = vram_manager.detect_vram_capacity()
            optimal_gpu = vram_manager.select_optimal_gpu()
            
            self.assertEqual(optimal_gpu.total_memory_mb, 16384)
            self.assertEqual(optimal_gpu.name, "NVIDIA GeForce RTX 4080")
    
    def test_threadripper_pro_configuration_integration(self):
        """Test complete integration for Threadripper PRO configuration"""
        hardware_optimizer = HardwareOptimizer()
        
        threadripper_profile = HardwareProfile(
            cpu_model="AMD Ryzen Threadripper PRO 5995WX",
            cpu_cores=64,
            total_memory_gb=128,
            gpu_model="NVIDIA GeForce RTX 4080",
            vram_gb=16,
            cuda_version="12.1",
            driver_version="531.79",
            is_rtx_4080=True,
            is_threadripper_pro=True
        )
        
        # Generate optimized settings
        with patch.object(hardware_optimizer, '_detect_numa_nodes', return_value=[0, 1]):
            with patch.object(hardware_optimizer, '_generate_cpu_affinity', return_value=list(range(32))):
                settings = hardware_optimizer.generate_threadripper_pro_settings(threadripper_profile)
        
        # Verify Threadripper PRO specific optimizations
        self.assertEqual(settings.vae_tile_size, (384, 384))  # Larger for powerful CPU
        self.assertEqual(settings.batch_size, 4)  # Higher with CPU support
        self.assertFalse(settings.text_encoder_offload)  # Keep on GPU with powerful CPU
        self.assertEqual(settings.memory_fraction, 0.95)  # Higher with CPU support
        self.assertFalse(settings.gradient_checkpointing)  # Disable with abundant resources
        self.assertEqual(settings.numa_nodes, [0, 1])
        self.assertTrue(settings.enable_numa_optimization)
        self.assertEqual(settings.parallel_workers, min(8, 64 // 8))
    
    def test_low_end_hardware_configuration_integration(self):
        """Test integration for low-end hardware configuration"""
        hardware_optimizer = HardwareOptimizer()
        
        low_end_profile = HardwareProfile(
            cpu_model="Intel Core i5-10400",
            cpu_cores=6,
            total_memory_gb=16,
            gpu_model="NVIDIA GeForce GTX 1660",
            vram_gb=6,
            cuda_version="11.8",
            driver_version="516.94",
            is_rtx_4080=False,
            is_threadripper_pro=False
        )
        
        # Generate conservative settings
        settings = hardware_optimizer.generate_optimal_settings(low_end_profile)
        
        # Verify conservative optimizations
        self.assertEqual(settings.batch_size, 1)  # Conservative for low VRAM
        self.assertTrue(settings.enable_cpu_offload)
        self.assertFalse(settings.enable_tensor_cores)  # Not available on GTX 1660
        self.assertEqual(settings.memory_fraction, 0.8)  # Conservative
        self.assertTrue(settings.gradient_checkpointing)  # Memory saving
        self.assertFalse(settings.use_bf16)  # Not supported on older hardware
        
        # Test memory optimization settings
        memory_settings = hardware_optimizer.get_memory_optimization_settings(6)  # 6GB VRAM
        
        self.assertTrue(memory_settings['enable_attention_slicing'])
        self.assertTrue(memory_settings['enable_vae_slicing'])
        self.assertEqual(memory_settings['batch_size'], 1)
        self.assertEqual(memory_settings['vae_tile_size'], (128, 128))  # Small tiles


if __name__ == '__main__':
    unittest.main()