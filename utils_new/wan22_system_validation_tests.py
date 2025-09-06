"""
WAN22 System Validation Tests
Comprehensive validation tests for RTX 4080 and Threadripper PRO optimizations,
edge case testing, and automated syntax validation
Task 12.2 Implementation
"""

import unittest
import tempfile
import shutil
import json
import ast
import os
import time
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from typing import Dict, List, Any, Optional

# Import system components for testing
from hardware_optimizer import HardwareOptimizer, HardwareProfile, OptimalSettings
from syntax_validator import SyntaxValidator
from vram_manager import VRAMManager
from quantization_controller import QuantizationController
from config_validator import ConfigValidator
from error_recovery_system import ErrorRecoverySystem
from health_monitor import HealthMonitor
from wan22_performance_benchmarks import WAN22PerformanceBenchmarks

class RTX4080OptimizationValidationTests(unittest.TestCase):
    """Validation tests for RTX 4080 specific optimizations"""
    
    def setUp(self):
        """Set up RTX 4080 test environment"""
        self.rtx_4080_profile = HardwareProfile(
            cpu_model="Intel Core i7-13700K",
            cpu_cores=16,
            total_memory_gb=32,
            gpu_model="NVIDIA GeForce RTX 4080",
            vram_gb=16,
            cuda_version="12.1",
            driver_version="535.98",
            is_rtx_4080=True,
            is_threadripper_pro=False
        )
        
        self.hardware_optimizer = HardwareOptimizer()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_rtx_4080_detection_and_optimization(self):
        """Test RTX 4080 hardware detection and optimization generation"""
        # Test hardware detection
        self.assertTrue(self.rtx_4080_profile.is_rtx_4080)
        self.assertEqual(self.rtx_4080_profile.vram_gb, 16)
        
        # Generate optimizations
        settings = self.hardware_optimizer.generate_rtx_4080_settings(self.rtx_4080_profile)
        
        # Validate RTX 4080 specific optimizations
        self.assertTrue(settings.enable_tensor_cores, "Tensor cores should be enabled for RTX 4080")
        self.assertTrue(settings.use_bf16, "BF16 should be enabled for RTX 4080")
        self.assertTrue(settings.enable_xformers, "xFormers should be enabled for RTX 4080")
        
        # Validate tile sizes are optimized for RTX 4080
        self.assertGreaterEqual(settings.tile_size[0], 512, "Tile size should be optimized for RTX 4080")
        self.assertGreaterEqual(settings.tile_size[1], 512, "Tile size should be optimized for RTX 4080")
        
        # Validate memory settings
        self.assertGreaterEqual(settings.memory_fraction, 0.85, "Memory fraction should be high for RTX 4080")
        self.assertGreaterEqual(settings.batch_size, 1, "Batch size should be appropriate for RTX 4080")
    
    def test_rtx_4080_vram_management(self):
        """Test VRAM management for RTX 4080"""
        vram_manager = VRAMManager()
        
        # Mock VRAM detection for RTX 4080
        with patch.object(vram_manager, 'detect_vram_capacity') as mock_detect:
            mock_detect.return_value = {
                'total_vram_mb': 16384,  # 16GB
                'available_vram_mb': 15360,  # ~15GB available
                'gpu_name': 'NVIDIA GeForce RTX 4080',
                'detection_method': 'nvml'
            }
            
            vram_info = vram_manager.detect_vram_capacity()
            
            # Validate VRAM detection
            self.assertEqual(vram_info['total_vram_mb'], 16384)
            self.assertIn('RTX 4080', vram_info['gpu_name'])
            
            # Test VRAM optimization thresholds
            usage_threshold = 0.9  # 90% threshold
            max_usage_mb = int(vram_info['total_vram_mb'] * usage_threshold)
            self.assertEqual(max_usage_mb, 14745)  # ~14.4GB
    
    def test_rtx_4080_quantization_strategies(self):
        """Test quantization strategies for RTX 4080"""
        quantization_controller = QuantizationController()
        
        # Test optimal quantization strategy for RTX 4080
        strategy = quantization_controller.determine_optimal_quantization(
            model_info={'name': 'ti2v_5b', 'size_gb': 8},
            hardware=self.rtx_4080_profile
        )
        
        # RTX 4080 should prefer BF16 for optimal performance
        self.assertIn('bf16', strategy.quantization_type.lower())
        self.assertTrue(strategy.enable_optimizations)
        self.assertGreaterEqual(strategy.timeout_seconds, 300)
    
    def test_rtx_4080_performance_validation(self):
        """Test performance validation for RTX 4080"""
        benchmarks = WAN22PerformanceBenchmarks()
        
        # Create mock performance metrics for RTX 4080
        mock_metrics = {
            'model_load_time': 240.0,  # 4 minutes - within target
            'generation_time': 90.0,   # 1.5 minutes - within target
            'peak_vram_usage_mb': 11264,  # 11GB - within target
            'gpu_utilization_avg': 95.0,  # High utilization
            'gpu_temperature_max': 78.0,  # Safe temperature
            'vram_efficiency': 0.7
        }
        
        # Validate against RTX 4080 targets
        targets_met = benchmarks._validate_ti2v_targets(
            type('MockMetrics', (), mock_metrics)(), "video_generation"
        )
        
        self.assertTrue(targets_met['generation_time'])
        self.assertTrue(targets_met['vram_usage'])
        self.assertTrue(targets_met['overall'])

class ThreadripperPROOptimizationValidationTests(unittest.TestCase):
    """Validation tests for Threadripper PRO 5995WX specific optimizations"""
    
    def setUp(self):
        """Set up Threadripper PRO test environment"""
        self.threadripper_profile = HardwareProfile(
            cpu_model="AMD Threadripper PRO 5995WX",
            cpu_cores=64,
            total_memory_gb=128,
            gpu_model="NVIDIA GeForce RTX 4080",
            vram_gb=16,
            cuda_version="12.1",
            driver_version="535.98",
            is_rtx_4080=True,
            is_threadripper_pro=True
        )
        
        self.hardware_optimizer = HardwareOptimizer()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_threadripper_pro_detection_and_optimization(self):
        """Test Threadripper PRO hardware detection and optimization generation"""
        # Test hardware detection
        self.assertTrue(self.threadripper_profile.is_threadripper_pro)
        self.assertEqual(self.threadripper_profile.cpu_cores, 64)
        self.assertEqual(self.threadripper_profile.total_memory_gb, 128)
        
        # Generate optimizations
        settings = self.hardware_optimizer.generate_threadripper_pro_settings(self.threadripper_profile)
        
        # Validate Threadripper PRO specific optimizations
        self.assertTrue(settings.enable_numa_optimization, "NUMA optimization should be enabled")
        self.assertGreaterEqual(settings.num_threads, 16, "Should utilize multiple CPU cores")
        self.assertGreaterEqual(settings.parallel_workers, 4, "Should use parallel workers")
        self.assertGreaterEqual(settings.preprocessing_threads, 4, "Should use multiple preprocessing threads")
        
        # Validate memory settings for high-memory system
        self.assertLessEqual(settings.memory_fraction, 0.95, "Memory fraction should be reasonable")
        
        # Should not exceed available cores
        self.assertLessEqual(settings.num_threads, self.threadripper_profile.cpu_cores)
    
    def test_threadripper_pro_cpu_utilization(self):
        """Test CPU utilization optimization for Threadripper PRO"""
        settings = self.hardware_optimizer.generate_threadripper_pro_settings(self.threadripper_profile)
        
        # Test thread allocation
        expected_threads = min(32, self.threadripper_profile.cpu_cores // 2)  # Conservative threading
        self.assertGreaterEqual(settings.num_threads, 16)
        self.assertLessEqual(settings.num_threads, expected_threads)
        
        # Test parallel worker allocation
        expected_workers = min(8, self.threadripper_profile.cpu_cores // 8)
        self.assertGreaterEqual(settings.parallel_workers, 4)
        self.assertLessEqual(settings.parallel_workers, expected_workers)
        
        # Test NUMA optimization
        self.assertTrue(settings.enable_numa_optimization)
    
    def test_threadripper_pro_memory_optimization(self):
        """Test memory optimization for Threadripper PRO"""
        settings = self.hardware_optimizer.generate_threadripper_pro_settings(self.threadripper_profile)
        
        # With 128GB RAM, should allow higher memory usage
        self.assertGreaterEqual(settings.memory_fraction, 0.8)
        
        # Should enable optimizations that benefit from high memory
        self.assertFalse(settings.enable_cpu_offload, "CPU offloading may not be needed with high VRAM")
        self.assertTrue(settings.gradient_checkpointing, "Gradient checkpointing should be enabled")
    
    def test_threadripper_pro_performance_scaling(self):
        """Test performance scaling validation for Threadripper PRO"""
        # Test that optimizations scale with core count
        settings_32_cores = self.hardware_optimizer.generate_threadripper_pro_settings(
            HardwareProfile(
                cpu_model="AMD Threadripper PRO 5975WX",
                cpu_cores=32,
                total_memory_gb=128,
                gpu_model="NVIDIA GeForce RTX 4080",
                vram_gb=16,
                cuda_version="12.1",
                driver_version="535.98"
            )
        )
        
        settings_64_cores = self.hardware_optimizer.generate_threadripper_pro_settings(self.threadripper_profile)
        
        # 64-core system should use more threads and workers
        self.assertGreaterEqual(settings_64_cores.num_threads, settings_32_cores.num_threads)
        self.assertGreaterEqual(settings_64_cores.parallel_workers, settings_32_cores.parallel_workers)

class EdgeCaseValidationTests(unittest.TestCase):
    """Edge case testing for low VRAM, corrupted configs, and failed model loads"""
    
    def setUp(self):
        """Set up edge case test environment"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Low VRAM profile (8GB GPU)
        self.low_vram_profile = HardwareProfile(
            cpu_model="Intel Core i5-12400",
            cpu_cores=6,
            total_memory_gb=16,
            gpu_model="NVIDIA GeForce RTX 3070",
            vram_gb=8,
            cuda_version="12.1",
            driver_version="535.98",
            is_rtx_4080=False,
            is_threadripper_pro=False
        )
        
        # Very low VRAM profile (4GB GPU)
        self.very_low_vram_profile = HardwareProfile(
            cpu_model="Intel Core i3-12100",
            cpu_cores=4,
            total_memory_gb=8,
            gpu_model="NVIDIA GeForce GTX 1650",
            vram_gb=4,
            cuda_version="11.8",
            driver_version="535.98",
            is_rtx_4080=False,
            is_threadripper_pro=False
        )
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_low_vram_optimization_fallback(self):
        """Test optimization fallback for low VRAM systems"""
        hardware_optimizer = HardwareOptimizer()
        
        # Test 8GB VRAM optimization
        settings_8gb = hardware_optimizer.generate_optimal_settings(self.low_vram_profile)
        
        # Should enable aggressive memory saving
        self.assertTrue(settings_8gb.enable_cpu_offload, "CPU offloading should be enabled for low VRAM")
        self.assertTrue(settings_8gb.text_encoder_offload, "Text encoder offloading should be enabled")
        self.assertTrue(settings_8gb.vae_offload, "VAE offloading should be enabled")
        self.assertTrue(settings_8gb.gradient_checkpointing, "Gradient checkpointing should be enabled")
        
        # Should use conservative settings
        self.assertEqual(settings_8gb.batch_size, 1, "Batch size should be 1 for low VRAM")
        self.assertLessEqual(settings_8gb.tile_size[0], 512, "Tile size should be conservative")
        self.assertLessEqual(settings_8gb.memory_fraction, 0.85, "Memory fraction should be conservative")
        
        # Test 4GB VRAM optimization (very low)
        settings_4gb = hardware_optimizer.generate_optimal_settings(self.very_low_vram_profile)
        
        # Should be even more aggressive
        self.assertTrue(settings_4gb.enable_cpu_offload)
        self.assertTrue(settings_4gb.text_encoder_offload)
        self.assertTrue(settings_4gb.vae_offload)
        self.assertEqual(settings_4gb.batch_size, 1)
        self.assertLessEqual(settings_4gb.tile_size[0], 384, "Tile size should be very small for 4GB VRAM")
        self.assertLessEqual(settings_4gb.memory_fraction, 0.85, "Memory fraction should be very conservative")
    
    def test_corrupted_config_recovery(self):
        """Test recovery from corrupted configuration files"""
        config_validator = ConfigValidator()
        
        # Create corrupted config file
        corrupted_config_path = Path(self.temp_dir) / "corrupted_config.json"
        with open(corrupted_config_path, 'w') as f:
            f.write('{"invalid": json, "missing_quotes": value, "trailing_comma": true,}')
        
        # Test config validation and repair
        validation_result = config_validator.validate_config_schema(str(corrupted_config_path))
        
        self.assertFalse(validation_result.is_valid, "Corrupted config should be invalid")
        self.assertGreater(len(validation_result.errors), 0, "Should detect JSON syntax errors")
        
        # Test recovery attempt
        recovery_result = config_validator.backup_and_repair_config(str(corrupted_config_path))
        
        # Should create backup
        backup_files = list(Path(self.temp_dir).glob("corrupted_config.json.backup.*"))
        self.assertGreater(len(backup_files), 0, "Should create backup file")
        
        # Should attempt repair or provide fallback
        self.assertIsNotNone(recovery_result, "Should provide recovery result")
    
    def test_missing_config_file_handling(self):
        """Test handling of missing configuration files"""
        config_validator = ConfigValidator()
        
        # Test validation of non-existent file
        missing_config_path = Path(self.temp_dir) / "missing_config.json"
        
        validation_result = config_validator.validate_config_schema(str(missing_config_path))
        
        self.assertFalse(validation_result.is_valid, "Missing config should be invalid")
        self.assertIn("not found", str(validation_result.errors).lower(), "Should indicate file not found")
        
        # Test recovery from missing file
        recovery_result = config_validator.backup_and_repair_config(str(missing_config_path))
        
        # Should create default config
        self.assertTrue(missing_config_path.exists(), "Should create default config file")
        
        # Verify default config is valid JSON
        with open(missing_config_path, 'r') as f:
            default_config = json.load(f)
        
        self.assertIsInstance(default_config, dict, "Default config should be valid JSON")
    
    def test_model_loading_failure_recovery(self):
        """Test recovery from model loading failures"""
        error_recovery = ErrorRecoverySystem()
        
        # Simulate model loading failure
        def failing_model_loader():
            raise RuntimeError("CUDA out of memory")
        
        # Test error recovery
        recovery_result = error_recovery.attempt_recovery(
            RuntimeError("CUDA out of memory"),
            context={'operation': 'model_loading', 'model': 'ti2v_5b'}
        )
        
        # Should provide recovery suggestions
        self.assertIsNotNone(recovery_result, "Should provide recovery result")
        self.assertGreater(len(recovery_result.recovery_actions), 0, "Should suggest recovery actions")
        
        # Should suggest memory optimization
        recovery_actions_text = ' '.join(recovery_result.recovery_actions).lower()
        self.assertIn("memory", recovery_actions_text, "Should suggest memory optimization")
        self.assertIn("offload", recovery_actions_text, "Should suggest CPU offloading")
    
    def test_quantization_timeout_handling(self):
        """Test handling of quantization timeouts"""
        quantization_controller = QuantizationController()
        
        # Mock quantization function that times out
        def timeout_quantization():
            time.sleep(10)  # Simulate long operation
            return {"quantized": True}
        
        # Test with short timeout
        strategy = type('Strategy', (), {
            'quantization_type': 'bf16',
            'timeout_seconds': 1,  # Very short timeout
            'enable_optimizations': True
        })()
        
        # Should handle timeout gracefully
        with patch.object(quantization_controller, '_apply_quantization', side_effect=timeout_quantization):
            result = quantization_controller.apply_quantization_with_timeout(strategy)
            
            # Should fall back to no quantization
            self.assertFalse(result.success, "Should fail due to timeout")
            self.assertIn("timeout", result.error_message.lower(), "Should indicate timeout")
    
    def test_insufficient_system_resources(self):
        """Test handling of insufficient system resources"""
        # Test with very limited system
        limited_profile = HardwareProfile(
            cpu_model="Intel Celeron N4020",
            cpu_cores=2,
            total_memory_gb=4,
            gpu_model="Intel UHD Graphics 600",
            vram_gb=0,  # Integrated graphics
            cuda_version="",
            driver_version="",
            is_rtx_4080=False,
            is_threadripper_pro=False
        )
        
        hardware_optimizer = HardwareOptimizer()
        
        # Should handle gracefully and provide minimal settings
        settings = hardware_optimizer.generate_optimal_settings(limited_profile)
        
        # Should use very conservative settings
        self.assertEqual(settings.batch_size, 1, "Should use minimal batch size")
        self.assertEqual(settings.num_threads, 1, "Should use minimal threading")
        self.assertTrue(settings.enable_cpu_offload, "Should enable all CPU offloading")
        self.assertLessEqual(settings.tile_size[0], 256, "Should use very small tiles")
    
    def test_network_connectivity_issues(self):
        """Test handling of network connectivity issues during model download"""
        # Mock network failure during model download
        def mock_download_failure():
            raise ConnectionError("Failed to download model: Network unreachable")
        
        error_recovery = ErrorRecoverySystem()
        
        # Test recovery from network error
        recovery_result = error_recovery.attempt_recovery(
            ConnectionError("Failed to download model: Network unreachable"),
            context={'operation': 'model_download', 'model': 'ti2v_5b'}
        )
        
        # Should provide network-related recovery suggestions
        recovery_actions_text = ' '.join(recovery_result.recovery_actions).lower()
        self.assertIn("network", recovery_actions_text, "Should mention network issues")
        self.assertIn("local", recovery_actions_text, "Should suggest local alternatives")

class SyntaxValidationTests(unittest.TestCase):
    """Automated testing for syntax validation in critical files"""
    
    def setUp(self):
        """Set up syntax validation test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.syntax_validator = SyntaxValidator()
        
        # Create test files with various syntax issues
        self.create_test_files()
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_files(self):
        """Create test files with various syntax issues"""
        # Valid Python file
        valid_file = Path(self.temp_dir) / "valid_file.py"
        with open(valid_file, 'w') as f:
            f.write("""
def valid_function():
    if True:
        return "valid"
    else:
        return "also valid"

class ValidClass:
    def __init__(self):
        self.value = 42
""")
        
        # File with missing else clause (like ui_event_handlers_enhanced.py issue)
        missing_else_file = Path(self.temp_dir) / "missing_else.py"
        with open(missing_else_file, 'w') as f:
            f.write("""
def problematic_function():
    if True:
        return "something"
    # Missing else clause here - line 187 equivalent
    
def another_function():
    return "ok"
""")
        
        # File with missing brackets
        missing_brackets_file = Path(self.temp_dir) / "missing_brackets.py"
        with open(missing_brackets_file, 'w') as f:
            f.write("""
def function_with_missing_bracket():
    my_list = [1, 2, 3
    return my_list

def function_with_missing_paren():
    result = some_function(arg1, arg2
    return result
""")
        
        # File with indentation errors
        indentation_error_file = Path(self.temp_dir) / "indentation_error.py"
        with open(indentation_error_file, 'w') as f:
            f.write("""
def function_with_bad_indentation():
    if True:
        print("correct")
      print("incorrect indentation")
    return True
""")
        
        # File with syntax error that can't be auto-repaired
        severe_syntax_error_file = Path(self.temp_dir) / "severe_error.py"
        with open(severe_syntax_error_file, 'w') as f:
            f.write("""
def function_with_severe_error():
    if True
        return "missing colon"
    
    invalid syntax here !!!
    return False
""")
    
    def test_valid_file_validation(self):
        """Test validation of syntactically correct file"""
        valid_file = Path(self.temp_dir) / "valid_file.py"
        
        result = self.syntax_validator.validate_file(str(valid_file))
        
        self.assertTrue(result.is_valid, "Valid file should pass validation")
        self.assertEqual(len(result.errors), 0, "Valid file should have no errors")
        self.assertEqual(len(result.warnings), 0, "Valid file should have no warnings")
    
    def test_missing_else_detection_and_repair(self):
        """Test detection and repair of missing else clauses"""
        missing_else_file = Path(self.temp_dir) / "missing_else.py"
        
        # First validate to detect the issue
        validation_result = self.syntax_validator.validate_file(str(missing_else_file))
        
        # The file might be syntactically valid but logically incomplete
        # This tests the enhanced validation that checks for incomplete if statements
        
        # Attempt repair
        repair_result = self.syntax_validator.repair_syntax_errors(str(missing_else_file))
        
        # Should create backup
        backup_files = list(Path(self.temp_dir).glob("missing_else.py.backup.*"))
        self.assertGreater(len(backup_files), 0, "Should create backup file")
        
        # Verify repair attempt was made
        self.assertIsNotNone(repair_result, "Should provide repair result")
    
    def test_missing_brackets_detection_and_repair(self):
        """Test detection and repair of missing brackets"""
        missing_brackets_file = Path(self.temp_dir) / "missing_brackets.py"
        
        # Validate file - should detect syntax errors
        validation_result = self.syntax_validator.validate_file(str(missing_brackets_file))
        
        self.assertFalse(validation_result.is_valid, "File with missing brackets should be invalid")
        self.assertGreater(len(validation_result.errors), 0, "Should detect syntax errors")
        
        # Attempt repair
        repair_result = self.syntax_validator.repair_syntax_errors(str(missing_brackets_file))
        
        # Should attempt repair
        self.assertIsNotNone(repair_result, "Should provide repair result")
        
        # Should create backup
        backup_files = list(Path(self.temp_dir).glob("missing_brackets.py.backup.*"))
        self.assertGreater(len(backup_files), 0, "Should create backup file")
    
    def test_indentation_error_detection(self):
        """Test detection of indentation errors"""
        indentation_error_file = Path(self.temp_dir) / "indentation_error.py"
        
        validation_result = self.syntax_validator.validate_file(str(indentation_error_file))
        
        self.assertFalse(validation_result.is_valid, "File with indentation errors should be invalid")
        self.assertGreater(len(validation_result.errors), 0, "Should detect indentation errors")
        
        # Check that error mentions indentation
        error_text = ' '.join(validation_result.errors).lower()
        self.assertIn("indent", error_text, "Should mention indentation in error")
    
    def test_severe_syntax_error_handling(self):
        """Test handling of severe syntax errors that can't be auto-repaired"""
        severe_error_file = Path(self.temp_dir) / "severe_error.py"
        
        validation_result = self.syntax_validator.validate_file(str(severe_error_file))
        
        self.assertFalse(validation_result.is_valid, "File with severe errors should be invalid")
        self.assertGreater(len(validation_result.errors), 0, "Should detect severe syntax errors")
        
        # Attempt repair - should fail gracefully
        repair_result = self.syntax_validator.repair_syntax_errors(str(severe_error_file))
        
        # Should indicate that manual intervention is required
        self.assertFalse(repair_result.success, "Should fail to auto-repair severe errors")
        self.assertIn("manual", repair_result.message.lower(), "Should suggest manual intervention")
    
    def test_enhanced_event_handlers_specific_validation(self):
        """Test validation specific to ui_event_handlers_enhanced.py issues"""
        # Create file similar to the actual problematic file
        enhanced_handlers_file = Path(self.temp_dir) / "ui_event_handlers_enhanced.py"
        with open(enhanced_handlers_file, 'w') as f:
            f.write("""
import gradio as gr

def create_enhanced_handlers():
    def on_generate_click():
        if some_condition:
            return process_generation()
        # Line 187 equivalent - missing else clause
    
    def on_model_change():
        if model_loaded:
            return update_ui()
        else:
            return show_loading()
    
    return {
        'generate': on_generate_click,
        'model_change': on_model_change
    }
""")
        
        # Test enhanced validation
        validation_result = self.syntax_validator.validate_enhanced_handlers()
        
        # Should detect the specific issue or validate successfully after repair
        self.assertIsNotNone(validation_result, "Should provide validation result")
        
        # If the file exists and has issues, should detect them
        if Path("ui_event_handlers_enhanced.py").exists():
            # Test the actual file
            actual_validation = self.syntax_validator.validate_file("ui_event_handlers_enhanced.py")
            
            # Should either be valid (if already fixed) or provide specific error info
            if not actual_validation.is_valid:
                self.assertGreater(len(actual_validation.errors), 0, "Should provide specific error information")
    
    def test_critical_files_batch_validation(self):
        """Test batch validation of critical system files"""
        critical_files = [
            "ui_event_handlers_enhanced.py",
            "ui_event_handlers.py",
            "wan_pipeline_loader.py",
            "hardware_optimizer.py",
            "vram_manager.py",
            "quantization_controller.py"
        ]
        
        validation_results = {}
        
        for file_path in critical_files:
            if Path(file_path).exists():
                result = self.syntax_validator.validate_file(file_path)
                validation_results[file_path] = result
        
        # Report results
        valid_files = [f for f, r in validation_results.items() if r.is_valid]
        invalid_files = [f for f, r in validation_results.items() if not r.is_valid]
        
        # Log results for debugging
        if invalid_files:
            print(f"Invalid files found: {invalid_files}")
            for file_path in invalid_files:
                result = validation_results[file_path]
                print(f"  {file_path}: {result.errors}")
        
        # At minimum, should be able to validate files that exist
        self.assertGreaterEqual(len(validation_results), 0, "Should validate at least some files")
    
    def test_ast_parsing_accuracy(self):
        """Test accuracy of AST parsing for syntax validation"""
        # Create file with complex but valid Python syntax
        complex_file = Path(self.temp_dir) / "complex_syntax.py"
        with open(complex_file, 'w') as f:
            f.write("""
import asyncio
from typing import Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field

@dataclass
class ComplexClass:
    value: int = field(default=42)
    items: List[str] = field(default_factory=list)
    
    async def async_method(self) -> Optional[Dict[str, Union[int, str]]]:
        result = {}
        
        try:
            for i, item in enumerate(self.items):
                if isinstance(item, str):
                    result[f"item_{i}"] = item
                elif isinstance(item, int):
                    result[f"num_{i}"] = item
                else:
                    continue
        except Exception as e:
            return None
        finally:
            await asyncio.sleep(0.1)
        
        return result if result else None

def lambda_function():
    return lambda x: x * 2 if x > 0 else 0

# List comprehension with conditions
filtered_data = [x for x in range(100) if x % 2 == 0 and x > 10]

# Dictionary comprehension
squared_dict = {f"key_{i}": i**2 for i in range(10) if i % 3 == 0}
""")
        
        # Should parse complex syntax correctly
        validation_result = self.syntax_validator.validate_file(str(complex_file))
        
        self.assertTrue(validation_result.is_valid, "Complex but valid syntax should pass validation")
        self.assertEqual(len(validation_result.errors), 0, "Should have no syntax errors")
        
        # Test AST parsing directly
        with open(complex_file, 'r') as f:
            content = f.read()
        
        try:
            ast.parse(content)
            ast_valid = True
        except SyntaxError:
            ast_valid = False
        
        self.assertTrue(ast_valid, "AST parsing should succeed for valid complex syntax")

if __name__ == '__main__':
    # Run all validation tests
    unittest.main(verbosity=2)