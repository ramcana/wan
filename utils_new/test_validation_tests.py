#!/usr/bin/env python3
"""
Validation tests for WAN22 System Optimization
Tests syntax validation accuracy, VRAM detection reliability, and quantization quality
"""

import unittest
import tempfile
import ast
import random
from unittest.mock import patch, MagicMock
from pathlib import Path
from datetime import datetime

# Import components for validation testing
from syntax_validator import SyntaxValidator, ValidationResult, RepairResult
from vram_manager import VRAMManager, GPUInfo, VRAMDetectionError
from quantization_controller import QuantizationController, QuantizationMethod, ModelInfo


class TestSyntaxValidationAccuracy(unittest.TestCase):
    """Test syntax validation accuracy and reliability"""
    
    def setUp(self):
        """Set up syntax validation test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.validator = SyntaxValidator(backup_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up syntax validation test environment"""
        import shutil
shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_syntax_validation_accuracy_valid_code(self):
        """Test syntax validation accuracy on valid Python code"""
        valid_code_samples = [
            # Basic function
            """
def hello_world():
    print("Hello, World!")
    return True
""",
            # Class definition
            """
class TestClass:
    def __init__(self, value):
        self.value = value
    
    def get_value(self):
        return self.value
""",
            # List comprehension
            """
numbers = [1, 2, 3, 4, 5]
squares = [x**2 for x in numbers if x % 2 == 0]
result = sum(squares)
""",
            # Complex control flow
            """
def process_data(data):
    if not data:
        return None
    
    processed = []
    for item in data:
        try:
            value = int(item)
            if value > 0:
                processed.append(value * 2)
        except ValueError:
            continue
    
    return processed if processed else None
""",
            # Async/await
            """
import asyncio

async def fetch_data():
    await asyncio.sleep(1)
    return {"status": "success"}

async def main():
    result = await fetch_data()
    print(result)
"""
        ]
        
        validation_results = []
        for i, code in enumerate(valid_code_samples):
            test_file = Path(self.temp_dir) / f"valid_test_{i}.py"
            test_file.write_text(code)
            
            result = self.validator.validate_file(str(test_file))
            validation_results.append(result.is_valid)
        
        # All valid code should pass validation
        self.assertTrue(all(validation_results))
        self.assertEqual(len(validation_results), len(valid_code_samples))
    
    def test_syntax_validation_accuracy_invalid_code(self):
        """Test syntax validation accuracy on invalid Python code"""
        invalid_code_samples = [
            # Missing colon
            """
def broken_function()
    return True
""",
            # Unmatched parentheses
            """
def test():
    result = some_function(arg1, arg2
    return result
""",
            # Missing else in conditional expression
            """
def test():
    x = [item if condition]
    return x
""",
            # Invalid indentation
            """
def test():
print("This is wrong")
    return True
""",
            # Unclosed string
            """
def test():
    message = "This string is not closed
    return message
""",
            # Invalid syntax in list comprehension
            """
def test():
    result = [x for x in range(10) if x > 5 else 0]
    return result
"""
        ]
        
        validation_results = []
        error_types = []
        
        for i, code in enumerate(invalid_code_samples):
            test_file = Path(self.temp_dir) / f"invalid_test_{i}.py"
            test_file.write_text(code)
            
            result = self.validator.validate_file(str(test_file))
            validation_results.append(result.is_valid)
            
            if result.errors:
                error_types.append(result.errors[0].error_type)
        
        # All invalid code should fail validation
        self.assertFalse(any(validation_results))
        self.assertEqual(len(validation_results), len(invalid_code_samples))
        self.assertGreater(len(error_types), 0)
    
    def test_syntax_repair_accuracy(self):
        """Test syntax repair accuracy and effectiveness"""
        repairable_code_samples = [
            # Missing else clause - should be repairable
            ("""
def test():
    x = [item if condition]
    return x
""", "missing_else"),
            
            # Missing closing bracket - should be repairable
            ("""
def test():
    data = [1, 2, 3
    return data
""", "missing_bracket"),
        ]
        
        repair_success_count = 0
        
        for i, (code, error_type) in enumerate(repairable_code_samples):
            test_file = Path(self.temp_dir) / f"repair_test_{i}.py"
            test_file.write_text(code)
            
            # Validate original (should fail)
            original_result = self.validator.validate_file(str(test_file))
            self.assertFalse(original_result.is_valid)
            
            # Attempt repair
            repair_result = self.validator.repair_syntax_errors(str(test_file))
            
            if repair_result.success and len(repair_result.repairs_made) > 0:
                # Validate repaired code
                repaired_result = self.validator.validate_file(str(test_file))
                if repaired_result.is_valid:
                    repair_success_count += 1
        
        # At least some repairs should be successful
        self.assertGreater(repair_success_count, 0)
    
    def test_syntax_validation_edge_cases(self):
        """Test syntax validation on edge cases"""
        edge_case_samples = [
            # Empty file
            "",
            
            # Only comments
            """
# This is a comment
# Another comment
""",
            
            # Only whitespace
            "   \n\t\n   ",
            
            # Single line
            "print('hello')",
            
            # Unicode characters
            """
def test_unicode():
    message = "Hello 世界 🌍"
    return message
""",
            
            # Very long line
            f"x = {'a' * 1000}",
            
            # Nested structures
            """
def complex_nested():
    return {
        'level1': {
            'level2': [
                {'level3': [1, 2, 3]},
                {'level3': [4, 5, 6]}
            ]
        }
    }
"""
        ]
        
        edge_case_results = []
        
        for i, code in enumerate(edge_case_samples):
            test_file = Path(self.temp_dir) / f"edge_case_{i}.py"
            test_file.write_text(code)
            
            try:
                result = self.validator.validate_file(str(test_file))
                edge_case_results.append(True)  # No exception raised
                
                # Verify result structure
                self.assertIsInstance(result, ValidationResult)
                self.assertIsInstance(result.is_valid, bool)
                self.assertIsInstance(result.errors, list)
                
            except Exception as e:
                edge_case_results.append(False)
                self.fail(f"Edge case {i} caused exception: {e}")
        
        # All edge cases should be handled without exceptions
        self.assertTrue(all(edge_case_results))


class TestVRAMDetectionReliability(unittest.TestCase):
    """Test VRAM detection reliability across different scenarios"""
    
    def setUp(self):
        """Set up VRAM detection test environment"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up VRAM detection test environment"""
        import shutil
shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_vram_detection_method_fallback_reliability(self):
        """Test reliability of VRAM detection method fallback chain"""
        manager = VRAMManager()
        
        # Test NVML detection success
        with patch('vram_manager.pynvml') as mock_pynvml:
            with patch('vram_manager.NVML_AVAILABLE', True):
                mock_pynvml.nvmlDeviceGetCount.return_value = 1
                mock_handle = MagicMock()
                mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
                mock_pynvml.nvmlDeviceGetName.return_value = b"RTX 4080"
                
                mock_memory_info = MagicMock()
                mock_memory_info.total = 16 * 1024**3
                mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_memory_info
                mock_pynvml.nvmlSystemGetDriverVersion.return_value = b"531.79"
                
                manager.nvml_initialized = True
                gpus = manager._detect_via_nvml()
                
                self.assertEqual(len(gpus), 1)
                self.assertEqual(gpus[0].total_memory_mb, 16 * 1024)
        
        # Test PyTorch fallback
        with patch('vram_manager.torch') as mock_torch:
            with patch('vram_manager.TORCH_AVAILABLE', True):
                mock_torch.cuda.is_available.return_value = True
                mock_torch.cuda.device_count.return_value = 1
                
                mock_props = MagicMock()
                mock_props.name = "RTX 4080"
                mock_props.total_memory = 16 * 1024**3
                mock_torch.cuda.get_device_properties.return_value = mock_props
                mock_torch.version.cuda = "12.1"
                
                gpus = manager._detect_via_pytorch()
                
                self.assertEqual(len(gpus), 1)
                self.assertEqual(gpus[0].total_memory_mb, 16 * 1024)
        
        # Test nvidia-smi fallback
        with patch('subprocess.run') as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "0, NVIDIA GeForce RTX 4080, 16384, 531.79\n"
            mock_run.return_value = mock_result
            
            gpus = manager._detect_via_nvidia_smi()
            
            self.assertEqual(len(gpus), 1)
            self.assertEqual(gpus[0].total_memory_mb, 16384)
        
        # Test manual config fallback
        manager.config.manual_vram_gb = {0: 16}
        gpus = manager._detect_via_manual_config()
        
        self.assertEqual(len(gpus), 1)
        self.assertEqual(gpus[0].total_memory_mb, 16 * 1024)
    
    def test_vram_detection_error_scenarios(self):
        """Test VRAM detection reliability under error conditions"""
        manager = VRAMManager()
        
        # Test all methods failing
        with patch.object(manager, '_detect_via_nvml', side_effect=Exception("NVML failed")):
            with patch.object(manager, '_detect_via_pytorch', side_effect=Exception("PyTorch failed")):
                with patch.object(manager, '_detect_via_nvidia_smi', side_effect=Exception("nvidia-smi failed")):
                    with patch.object(manager, '_detect_via_manual_config', side_effect=Exception("Manual failed")):
                        
                        with self.assertRaises(VRAMDetectionError):
                            manager.detect_vram_capacity()
        
        # Test partial failures with successful fallback
        with patch.object(manager, '_detect_via_nvml', side_effect=Exception("NVML failed")):
            with patch.object(manager, '_detect_via_pytorch', side_effect=Exception("PyTorch failed")):
                with patch.object(manager, '_detect_via_nvidia_smi') as mock_smi:
                    mock_smi.return_value = [GPUInfo(0, "RTX 4080", 16384, "531.79")]
                    
                    gpus = manager.detect_vram_capacity()
                    self.assertEqual(len(gpus), 1)
    
    def test_vram_usage_monitoring_accuracy(self):
        """Test accuracy of VRAM usage monitoring"""
        manager = VRAMManager()
        
        # Test NVML usage monitoring
        with patch('vram_manager.pynvml') as mock_pynvml:
            with patch('vram_manager.NVML_AVAILABLE', True):
                manager.nvml_initialized = True
                
                mock_handle = MagicMock()
                mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
                
                # Test various usage scenarios
                usage_scenarios = [
                    (8 * 1024**3, 8 * 1024**3, 16 * 1024**3),  # 50% usage
                    (12 * 1024**3, 4 * 1024**3, 16 * 1024**3),  # 75% usage
                    (15 * 1024**3, 1 * 1024**3, 16 * 1024**3),  # 93.75% usage
                ]
                
                for used_bytes, free_bytes, total_bytes in usage_scenarios:
                    mock_memory_info = MagicMock()
                    mock_memory_info.used = used_bytes
                    mock_memory_info.free = free_bytes
                    mock_memory_info.total = total_bytes
                    mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_memory_info
                    
                    usage = manager._get_gpu_memory_usage(0)
                    
                    self.assertIsNotNone(usage)
                    self.assertEqual(usage.used_mb, used_bytes // (1024 * 1024))
                    self.assertEqual(usage.free_mb, free_bytes // (1024 * 1024))
                    self.assertEqual(usage.total_mb, total_bytes // (1024 * 1024))
                    
                    expected_percent = (used_bytes / total_bytes) * 100
                    self.assertAlmostEqual(usage.usage_percent, expected_percent, places=1)
    
    def test_vram_detection_consistency(self):
        """Test consistency of VRAM detection across multiple calls"""
        manager = VRAMManager()
        
        # Mock consistent GPU detection
        with patch.object(manager, '_detect_via_nvml') as mock_nvml:
            mock_gpu = GPUInfo(0, "RTX 4080", 16384, "531.79")
            mock_nvml.return_value = [mock_gpu]
            
            # Multiple detection calls should return consistent results
            results = []
            for _ in range(10):
                gpus = manager.detect_vram_capacity()
                results.append((len(gpus), gpus[0].total_memory_mb, gpus[0].name))
            
            # All results should be identical
            first_result = results[0]
            for result in results[1:]:
                self.assertEqual(result, first_result)
    
    def test_multi_gpu_detection_accuracy(self):
        """Test accuracy of multi-GPU detection"""
        manager = VRAMManager()
        
        # Mock multi-GPU setup
        with patch.object(manager, '_detect_via_nvml') as mock_nvml:
            mock_gpus = [
                GPUInfo(0, "RTX 4080", 16384, "531.79"),
                GPUInfo(1, "RTX 3080", 10240, "531.79"),
                GPUInfo(2, "RTX 4090", 24576, "531.79")
            ]
            mock_nvml.return_value = mock_gpus
            
            gpus = manager.detect_vram_capacity()
            
            self.assertEqual(len(gpus), 3)
            
            # Verify each GPU was detected correctly
            expected_vram = [16384, 10240, 24576]
            expected_names = ["RTX 4080", "RTX 3080", "RTX 4090"]
            
            for i, gpu in enumerate(gpus):
                self.assertEqual(gpu.index, i)
                self.assertEqual(gpu.total_memory_mb, expected_vram[i])
                self.assertIn(expected_names[i], gpu.name)
        
        # Test optimal GPU selection
        optimal_gpu = manager.select_optimal_gpu()
        self.assertIsNotNone(optimal_gpu)
        self.assertEqual(optimal_gpu.total_memory_mb, 24576)  # Should select RTX 4090


class TestQuantizationQualityValidation(unittest.TestCase):
    """Test quantization quality validation and effectiveness"""
    
    def setUp(self):
        """Set up quantization quality test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.controller = QuantizationController(
            config_path=str(Path(self.temp_dir) / "config.json"),
            preferences_path=str(Path(self.temp_dir) / "prefs.json")
        )
    
    def tearDown(self):
        """Clean up quantization quality test environment"""
        import shutil
shutil.rmtree(self.temp_dir, ignore_errors=True) 
   def test_quantization_strategy_quality_validation(self):
        """Test quality validation of quantization strategies"""
        # Test different model types and sizes
        model_scenarios = [
            ModelInfo("small-sd", 3.0, "stable-diffusion", ["unet", "text_encoder"], 3072.0),
            ModelInfo("large-sd", 7.0, "stable-diffusion", ["unet", "text_encoder", "vae"], 7168.0),
            ModelInfo("xl-model", 12.0, "stable-diffusion-xl", ["unet", "text_encoder", "text_encoder_2", "vae"], 12288.0),
            ModelInfo("video-model", 20.0, "video-diffusion", ["transformer", "text_encoder", "vae"], 20480.0)
        ]
        
        quantization_methods = [
            QuantizationMethod.NONE,
            QuantizationMethod.FP16,
            QuantizationMethod.BF16,
            QuantizationMethod.INT8
        ]
        
        quality_results = []
        
        for model in model_scenarios:
            for method in quantization_methods:
                # Test strategy determination
                strategy = self.controller.determine_optimal_strategy(model)
                
                # Validate compatibility
                compatibility = self.controller.validate_quantization_compatibility(model, method)
                
                # Calculate expected quality impact
                expected_quality = self.controller._get_quality_threshold(method)
                
                quality_results.append({
                    'model': model.name,
                    'method': method.value,
                    'compatible': compatibility['compatible'],
                    'estimated_quality_impact': compatibility['estimated_quality_impact'],
                    'expected_threshold': expected_quality,
                    'memory_savings': compatibility['estimated_memory_usage'] < model.estimated_vram_usage
                })
        
        # Validate quality expectations
        for result in quality_results:
            # More aggressive quantization should have higher memory savings
            if result['method'] in ['int8', 'fp8']:
                self.assertTrue(result['memory_savings'], 
                    f"Aggressive quantization {result['method']} should save memory for {result['model']}")
            
            # Quality thresholds should be reasonable
            if result['method'] == 'none':
                self.assertEqual(result['expected_threshold'], 1.0)
            elif result['method'] in ['fp16', 'bf16']:
                self.assertGreaterEqual(result['expected_threshold'], 0.9)
            elif result['method'] == 'int8':
                self.assertGreaterEqual(result['expected_threshold'], 0.8)
    
    def test_quantization_compatibility_validation(self):
        """Test quantization compatibility validation accuracy"""
        # Mock different hardware capabilities
        hardware_scenarios = [
            # RTX 4080 - supports all methods
            {'supports_bf16': True, 'supports_int8': True, 'supports_fp8': True, 'vram_gb': 16},
            # RTX 3080 - limited FP8 support
            {'supports_bf16': True, 'supports_int8': True, 'supports_fp8': False, 'vram_gb': 10},
            # GTX 1660 - basic support only
            {'supports_bf16': False, 'supports_int8': False, 'supports_fp8': False, 'vram_gb': 6},
            # High-end workstation
            {'supports_bf16': True, 'supports_int8': True, 'supports_fp8': True, 'vram_gb': 48}
        ]
        
        test_model = ModelInfo("test-model", 5.0, "stable-diffusion", ["unet", "text_encoder", "vae"], 6144.0)
        
        compatibility_results = []
        
        for i, hw_config in enumerate(hardware_scenarios):
            # Update hardware profile
            self.controller.hardware_profile.supports_bf16 = hw_config['supports_bf16']
            self.controller.hardware_profile.supports_int8 = hw_config['supports_int8']
            self.controller.hardware_profile.supports_fp8 = hw_config['supports_fp8']
            self.controller.hardware_profile.vram_gb = hw_config['vram_gb']
            
            for method in [QuantizationMethod.BF16, QuantizationMethod.INT8, QuantizationMethod.FP8]:
                compatibility = self.controller.validate_quantization_compatibility(test_model, method)
                
                compatibility_results.append({
                    'hardware': f"hw_{i}",
                    'method': method.value,
                    'compatible': compatibility['compatible'],
                    'warnings': len(compatibility['warnings']),
                    'recommendations': len(compatibility['recommendations'])
                })
        
        # Validate compatibility logic
        # High-end hardware should support more methods
        high_end_results = [r for r in compatibility_results if r['hardware'] == 'hw_0']  # RTX 4080
        low_end_results = [r for r in compatibility_results if r['hardware'] == 'hw_2']   # GTX 1660
        
        high_end_compatible = sum(1 for r in high_end_results if r['compatible'])
        low_end_compatible = sum(1 for r in low_end_results if r['compatible'])
        
        self.assertGreater(high_end_compatible, low_end_compatible)
    
    def test_quantization_memory_estimation_accuracy(self):
        """Test accuracy of quantization memory usage estimation"""
        test_models = [
            ModelInfo("tiny", 1.0, "simple", ["unet"], 1024.0),
            ModelInfo("small", 3.0, "stable-diffusion", ["unet", "text_encoder"], 3072.0),
            ModelInfo("medium", 7.0, "stable-diffusion", ["unet", "text_encoder", "vae"], 7168.0),
            ModelInfo("large", 15.0, "stable-diffusion-xl", ["unet", "text_encoder", "text_encoder_2", "vae"], 15360.0)
        ]
        
        memory_estimations = []
        
        for model in test_models:
            base_usage = model.estimated_vram_usage
            
            for method in [QuantizationMethod.NONE, QuantizationMethod.FP16, QuantizationMethod.BF16, QuantizationMethod.INT8]:
                compatibility = self.controller.validate_quantization_compatibility(model, method)
                estimated_usage = compatibility['estimated_memory_usage']
                
                memory_estimations.append({
                    'model': model.name,
                    'method': method.value,
                    'base_usage': base_usage,
                    'estimated_usage': estimated_usage,
                    'reduction_ratio': estimated_usage / base_usage if base_usage > 0 else 1.0
                })
        
        # Validate memory estimation logic
        for estimation in memory_estimations:
            # No quantization should not reduce memory
            if estimation['method'] == 'none':
                self.assertEqual(estimation['reduction_ratio'], 1.0)
            
            # Quantization should reduce memory usage
            elif estimation['method'] in ['fp16', 'bf16']:
                self.assertLess(estimation['reduction_ratio'], 1.0)
                self.assertGreater(estimation['reduction_ratio'], 0.5)  # Reasonable reduction
            
            elif estimation['method'] == 'int8':
                self.assertLess(estimation['reduction_ratio'], 0.8)  # More aggressive reduction
                self.assertGreater(estimation['reduction_ratio'], 0.3)  # But not too extreme
    
    def test_quantization_fallback_chain_validation(self):
        """Test validation of quantization fallback chain logic"""
        fallback_chains = [
            (QuantizationMethod.FP8, QuantizationMethod.BF16),
            (QuantizationMethod.INT8, QuantizationMethod.BF16),
            (QuantizationMethod.BF16, QuantizationMethod.FP16),
            (QuantizationMethod.FP16, QuantizationMethod.NONE),
            (QuantizationMethod.NONE, None)
        ]
        
        for primary, expected_fallback in fallback_chains:
            actual_fallback = self.controller._get_fallback_method(primary)
            self.assertEqual(actual_fallback, expected_fallback,
                f"Fallback for {primary.value} should be {expected_fallback}")
    
    def test_quantization_component_priority_validation(self):
        """Test validation of component quantization priorities"""
        priorities = self.controller._get_component_priorities(QuantizationMethod.BF16)
        
        # Validate priority ordering
        self.assertGreater(priorities.get('unet', 0), priorities.get('vae', 0),
            "UNet should have higher priority than VAE")
        self.assertGreater(priorities.get('text_encoder', 0), priorities.get('vae', 0),
            "Text encoder should have higher priority than VAE")
        
        # Test INT8 priorities (more conservative with VAE)
        int8_priorities = self.controller._get_component_priorities(QuantizationMethod.INT8)
        bf16_priorities = self.controller._get_component_priorities(QuantizationMethod.BF16)
        
        self.assertLessEqual(int8_priorities.get('vae', 0), bf16_priorities.get('vae', 0),
            "INT8 should be more conservative with VAE quantization")
    
    def test_quantization_timeout_calculation_validation(self):
        """Test validation of quantization timeout calculations"""
        test_scenarios = [
            # (model_size_gb, method, expected_min_timeout, expected_max_timeout)
            (1.0, QuantizationMethod.FP16, 60, 300),      # Small model, simple method
            (5.0, QuantizationMethod.BF16, 150, 600),     # Medium model, moderate method
            (15.0, QuantizationMethod.INT8, 600, 1800),   # Large model, complex method
            (30.0, QuantizationMethod.FP8, 1200, 3600),   # Huge model, experimental method
        ]
        
        for model_size, method, min_timeout, max_timeout in test_scenarios:
            model = ModelInfo(f"test-{model_size}gb", model_size, "test", ["unet"], model_size * 1024)
            timeout = self.controller._calculate_timeout(method, model)
            
            self.assertGreaterEqual(timeout, min_timeout,
                f"Timeout for {model_size}GB model with {method.value} should be at least {min_timeout}s")
            self.assertLessEqual(timeout, max_timeout,
                f"Timeout for {model_size}GB model with {method.value} should be at most {max_timeout}s")
    
    def test_quantization_quality_threshold_validation(self):
        """Test validation of quality thresholds for different methods"""
        expected_thresholds = {
            QuantizationMethod.NONE: 1.0,
            QuantizationMethod.FP16: 0.95,
            QuantizationMethod.BF16: 0.95,
            QuantizationMethod.INT8: 0.85,
            QuantizationMethod.FP8: 0.80
        }
        
        for method, expected_threshold in expected_thresholds.items():
            actual_threshold = self.controller._get_quality_threshold(method)
            self.assertEqual(actual_threshold, expected_threshold,
                f"Quality threshold for {method.value} should be {expected_threshold}")
        
        # Validate threshold ordering (more aggressive = lower threshold)
        thresholds = [(method, self.controller._get_quality_threshold(method)) 
                     for method in expected_thresholds.keys()]
        
        # Sort by aggressiveness (NONE least aggressive, FP8 most aggressive)
        aggressiveness_order = [
            QuantizationMethod.NONE,
            QuantizationMethod.FP16,
            QuantizationMethod.BF16,
            QuantizationMethod.INT8,
            QuantizationMethod.FP8
        ]
        
        for i in range(len(aggressiveness_order) - 1):
            current_threshold = self.controller._get_quality_threshold(aggressiveness_order[i])
            next_threshold = self.controller._get_quality_threshold(aggressiveness_order[i + 1])
            
            self.assertGreaterEqual(current_threshold, next_threshold,
                f"{aggressiveness_order[i].value} should have higher quality threshold than {aggressiveness_order[i + 1].value}")


class TestValidationIntegration(unittest.TestCase):
    """Integration tests for validation across components"""
    
    def setUp(self):
        """Set up validation integration test environment"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up validation integration test environment"""
        import shutil
shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_cross_component_validation_consistency(self):
        """Test validation consistency across different components"""
        # Initialize components
        syntax_validator = SyntaxValidator(backup_dir=self.temp_dir)
        vram_manager = VRAMManager()
        quantization_controller = QuantizationController(
            config_path=str(Path(self.temp_dir) / "config.json"),
            preferences_path=str(Path(self.temp_dir) / "prefs.json")
        )
        
        # Test consistent validation behavior
        validation_results = {
            'syntax_validation_passed': 0,
            'vram_detection_passed': 0,
            'quantization_validation_passed': 0
        }
        
        # Test syntax validation consistency
        valid_code = """
def test_function():
    return True
"""
        test_file = Path(self.temp_dir) / "consistency_test.py"
        test_file.write_text(valid_code)
        
        for _ in range(5):  # Multiple runs
            result = syntax_validator.validate_file(str(test_file))
            if result.is_valid:
                validation_results['syntax_validation_passed'] += 1
        
        # Test VRAM detection consistency
        with patch.object(vram_manager, '_detect_via_nvml') as mock_nvml:
            mock_gpu = GPUInfo(0, "RTX 4080", 16384, "531.79")
            mock_nvml.return_value = [mock_gpu]
            
            for _ in range(5):  # Multiple runs
                try:
                    gpus = vram_manager.detect_vram_capacity()
                    if len(gpus) > 0 and gpus[0].total_memory_mb == 16384:
                        validation_results['vram_detection_passed'] += 1
                except Exception:
                    pass
        
        # Test quantization validation consistency
        test_model = ModelInfo("consistency-test", 5.0, "stable-diffusion", ["unet"], 5120.0)
        
        for _ in range(5):  # Multiple runs
            try:
                strategy = quantization_controller.determine_optimal_strategy(test_model)
                compatibility = quantization_controller.validate_quantization_compatibility(
                    test_model, strategy.method
                )
                if compatibility['compatible']:
                    validation_results['quantization_validation_passed'] += 1
            except Exception:
                pass
        
        # All validation should be consistent across runs
        for component, passed_count in validation_results.items():
            self.assertEqual(passed_count, 5, f"{component} should be consistent across multiple runs")
    
    def test_validation_error_handling_robustness(self):
        """Test robustness of validation error handling"""
        # Test syntax validator error handling
        syntax_validator = SyntaxValidator(backup_dir=self.temp_dir)
        
        # Test with non-existent file
        result = syntax_validator.validate_file("/nonexistent/file.py")
        self.assertFalse(result.is_valid)
        self.assertGreater(len(result.errors), 0)
        
        # Test VRAM manager error handling
        vram_manager = VRAMManager()
        
        # Test with all detection methods failing
        with patch.object(vram_manager, '_detect_via_nvml', side_effect=Exception("NVML failed")):
            with patch.object(vram_manager, '_detect_via_pytorch', side_effect=Exception("PyTorch failed")):
                with patch.object(vram_manager, '_detect_via_nvidia_smi', side_effect=Exception("nvidia-smi failed")):
                    with patch.object(vram_manager, '_detect_via_manual_config', side_effect=Exception("Manual failed")):
                        
                        with self.assertRaises(VRAMDetectionError):
                            vram_manager.detect_vram_capacity()
        
        # Test quantization controller error handling
        quantization_controller = QuantizationController(
            config_path=str(Path(self.temp_dir) / "config.json"),
            preferences_path=str(Path(self.temp_dir) / "prefs.json")
        )
        
        # Test with invalid model
        invalid_model = ModelInfo("", 0.0, "", [], 0.0)  # Empty model
        
        try:
            strategy = quantization_controller.determine_optimal_strategy(invalid_model)
            # Should handle gracefully without crashing
            self.assertIsNotNone(strategy)
        except Exception as e:
            # If it does raise an exception, it should be a reasonable one
            self.assertIsInstance(e, (ValueError, TypeError))
    
    def test_validation_performance_benchmarks(self):
        """Test validation performance meets benchmarks"""
        import time

        # Syntax validation performance
        syntax_validator = SyntaxValidator(backup_dir=self.temp_dir)
        
        # Create test file
        test_code = """
def performance_test():
    data = [i for i in range(1000)]
    result = sum(x * 2 for x in data if x % 2 == 0)
    return result
"""
        test_file = Path(self.temp_dir) / "performance_test.py"
        test_file.write_text(test_code)
        
        # Measure syntax validation time
        start_time = time.time()
        for _ in range(10):
            syntax_validator.validate_file(str(test_file))
        syntax_time = time.time() - start_time
        
        # Should validate 10 files in under 1 second
        self.assertLess(syntax_time, 1.0, "Syntax validation should be fast")
        
        # VRAM detection performance
        vram_manager = VRAMManager()
        
        with patch.object(vram_manager, '_detect_via_nvml') as mock_nvml:
            mock_gpu = GPUInfo(0, "RTX 4080", 16384, "531.79")
            mock_nvml.return_value = [mock_gpu]
            
            # Measure VRAM detection time
            start_time = time.time()
            for _ in range(10):
                vram_manager.detect_vram_capacity()
            vram_time = time.time() - start_time
            
            # Should detect VRAM 10 times in under 0.5 seconds
            self.assertLess(vram_time, 0.5, "VRAM detection should be fast")
        
        # Quantization strategy performance
        quantization_controller = QuantizationController(
            config_path=str(Path(self.temp_dir) / "config.json"),
            preferences_path=str(Path(self.temp_dir) / "prefs.json")
        )
        
        test_model = ModelInfo("perf-test", 5.0, "stable-diffusion", ["unet", "text_encoder"], 5120.0)
        
        # Measure quantization strategy time
        start_time = time.time()
        for _ in range(10):
            quantization_controller.determine_optimal_strategy(test_model)
        quant_time = time.time() - start_time
        
        # Should determine strategy 10 times in under 0.1 seconds
        self.assertLess(quant_time, 0.1, "Quantization strategy determination should be fast")


if __name__ == '__main__':
    unittest.main()