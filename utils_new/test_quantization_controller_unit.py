#!/usr/bin/env python3
"""
Unit tests for QuantizationController component
Tests quantization strategy determination, timeout handling, and progress monitoring
"""

import unittest
import tempfile
import json
import threading
import time
from unittest.mock import patch, MagicMock, PropertyMock
from pathlib import Path
from datetime import datetime

from quantization_controller import (
    QuantizationController, QuantizationMethod, QuantizationStatus,
    HardwareProfile, ModelInfo, QuantizationStrategy, QuantizationProgress,
    QuantizationResult, UserPreferences
)


class TestQuantizationController(unittest.TestCase):
    """Test cases for QuantizationController class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "config.json"
        self.preferences_path = Path(self.temp_dir) / "quantization_preferences.json"
        
        # Create minimal config file
        with open(self.config_path, 'w') as f:
            json.dump({}, f)
        
        self.controller = QuantizationController(
            config_path=str(self.config_path),
            preferences_path=str(self.preferences_path)
        )
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init(self):
        """Test QuantizationController initialization"""
        self.assertIsInstance(self.controller, QuantizationController)
        self.assertEqual(str(self.controller.config_path), str(self.config_path))
        self.assertEqual(str(self.controller.preferences_path), str(self.preferences_path))
        self.assertIsInstance(self.controller.preferences, UserPreferences)
        self.assertIsInstance(self.controller.hardware_profile, HardwareProfile)

        assert True  # TODO: Add proper assertion
    
    @patch('quantization_controller.torch')
    def test_detect_hardware_profile_with_cuda(self, mock_torch):
        """Test hardware profile detection with CUDA available"""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "NVIDIA GeForce RTX 4080"
        
        mock_props = MagicMock()
        mock_props.total_memory = 16 * 1024**3  # 16GB
        mock_torch.cuda.get_device_properties.return_value = mock_props
        mock_torch.cuda.get_device_capability.return_value = (8, 6)  # RTX 4080 capability
        mock_torch.cuda.is_bf16_supported.return_value = True
        mock_torch.version.cuda = "12.1"
        
        with patch.object(self.controller, '_check_int8_support', return_value=True):
            with patch.object(self.controller, '_check_fp8_support', return_value=True):
                profile = self.controller._detect_hardware_profile()
        
        self.assertEqual(profile.gpu_model, "NVIDIA GeForce RTX 4080")
        self.assertEqual(profile.vram_gb, 16)
        self.assertEqual(profile.cuda_version, "12.1")
        self.assertEqual(profile.compute_capability, (8, 6))
        self.assertTrue(profile.supports_bf16)
        self.assertTrue(profile.supports_int8)
        self.assertTrue(profile.supports_fp8)

        assert True  # TODO: Add proper assertion
    
    @patch('quantization_controller.torch')
    def test_detect_hardware_profile_cpu_only(self, mock_torch):
        """Test hardware profile detection with CPU only"""
        mock_torch.cuda.is_available.return_value = False
        
        profile = self.controller._detect_hardware_profile()
        
        self.assertEqual(profile.gpu_model, "CPU")
        self.assertEqual(profile.vram_gb, 0)
        self.assertEqual(profile.cuda_version, "N/A")
        self.assertEqual(profile.compute_capability, (0, 0))

        assert True  # TODO: Add proper assertion
    
    def test_determine_optimal_strategy_user_preference(self):
        """Test strategy determination with user preference"""
        model_info = ModelInfo(
            name="test-model",
            size_gb=5.0,
            architecture="transformer",
            components=["unet", "text_encoder"],
            estimated_vram_usage=8192.0
        )
        
        self.controller.preferences.preferred_method = QuantizationMethod.BF16
        self.controller.hardware_profile.supports_bf16 = True
        
        strategy = self.controller.determine_optimal_strategy(model_info)
        
        self.assertEqual(strategy.method, QuantizationMethod.BF16)
        self.assertIsInstance(strategy.timeout_seconds, int)
        self.assertGreater(strategy.timeout_seconds, 0)
        self.assertIsInstance(strategy.component_priorities, dict)

        assert True  # TODO: Add proper assertion
    
    def test_determine_optimal_strategy_model_specific(self):
        """Test strategy determination with model-specific preference"""
        model_info = ModelInfo(
            name="specific-model",
            size_gb=3.0,
            architecture="diffusion",
            components=["unet"],
            estimated_vram_usage=4096.0
        )
        
        self.controller.preferences.model_specific_preferences["specific-model"] = QuantizationMethod.INT8
        self.controller.hardware_profile.supports_int8 = True
        
        strategy = self.controller.determine_optimal_strategy(model_info)
        
        self.assertEqual(strategy.method, QuantizationMethod.INT8)

        assert True  # TODO: Add proper assertion
    
    def test_validate_hardware_compatibility_bf16_unsupported(self):
        """Test hardware compatibility validation when BF16 is unsupported"""
        self.controller.hardware_profile.supports_bf16 = False
        
        result = self.controller._validate_hardware_compatibility(
            QuantizationMethod.BF16, 
            MagicMock()
        )
        
        self.assertEqual(result, QuantizationMethod.FP16)

        assert True  # TODO: Add proper assertion
    
    def test_validate_hardware_compatibility_int8_unsupported(self):
        """Test hardware compatibility validation when INT8 is unsupported"""
        self.controller.hardware_profile.supports_int8 = False
        self.controller.hardware_profile.supports_bf16 = True
        
        result = self.controller._validate_hardware_compatibility(
            QuantizationMethod.INT8,
            MagicMock()
        )
        
        self.assertEqual(result, QuantizationMethod.BF16)

        assert True  # TODO: Add proper assertion
    
    def test_validate_hardware_compatibility_fp8_unsupported(self):
        """Test hardware compatibility validation when FP8 is unsupported"""
        self.controller.hardware_profile.supports_fp8 = False
        self.controller.hardware_profile.supports_bf16 = True
        
        result = self.controller._validate_hardware_compatibility(
            QuantizationMethod.FP8,
            MagicMock()
        )
        
        self.assertEqual(result, QuantizationMethod.BF16)

        assert True  # TODO: Add proper assertion
    
    def test_calculate_timeout_large_model(self):
        """Test timeout calculation for large model"""
        model_info = ModelInfo(
            name="large-model",
            size_gb=15.0,  # Large model
            architecture="transformer",
            components=["unet", "text_encoder"],
            estimated_vram_usage=20480.0
        )
        
        self.controller.preferences.timeout_seconds = 300  # 5 minutes base
        
        timeout = self.controller._calculate_timeout(QuantizationMethod.INT8, model_info)
        
        # Should be larger than base timeout due to model size and INT8 complexity
        self.assertGreater(timeout, 300)

        assert True  # TODO: Add proper assertion
    
    def test_calculate_timeout_small_model(self):
        """Test timeout calculation for small model"""
        model_info = ModelInfo(
            name="small-model",
            size_gb=2.0,  # Small model
            architecture="simple",
            components=["unet"],
            estimated_vram_usage=2048.0
        )
        
        self.controller.preferences.timeout_seconds = 300
        
        timeout = self.controller._calculate_timeout(QuantizationMethod.FP16, model_info)
        
        # Should be close to base timeout
        self.assertLessEqual(timeout, 450)  # Some multiplier but not too much

        assert True  # TODO: Add proper assertion
    
    def test_get_component_priorities(self):
        """Test component priority determination"""
        priorities = self.controller._get_component_priorities(QuantizationMethod.BF16)
        
        self.assertIsInstance(priorities, dict)
        self.assertIn("unet", priorities)
        self.assertIn("text_encoder", priorities)
        self.assertGreater(priorities["unet"], priorities["vae"])  # UNet should have higher priority

        assert True  # TODO: Add proper assertion
    
    def test_get_component_priorities_int8_conservative(self):
        """Test component priorities for INT8 (more conservative with VAE)"""
        priorities = self.controller._get_component_priorities(QuantizationMethod.INT8)
        
        self.assertEqual(priorities["vae"], 1)  # Should be very low priority for INT8

        assert True  # TODO: Add proper assertion
    
    def test_get_fallback_method(self):
        """Test fallback method determination"""
        fallback = self.controller._get_fallback_method(QuantizationMethod.FP8)
        self.assertEqual(fallback, QuantizationMethod.BF16)
        
        fallback = self.controller._get_fallback_method(QuantizationMethod.INT8)
        self.assertEqual(fallback, QuantizationMethod.BF16)
        
        fallback = self.controller._get_fallback_method(QuantizationMethod.BF16)
        self.assertEqual(fallback, QuantizationMethod.FP16)
        
        fallback = self.controller._get_fallback_method(QuantizationMethod.FP16)
        self.assertEqual(fallback, QuantizationMethod.NONE)
        
        fallback = self.controller._get_fallback_method(QuantizationMethod.NONE)
        self.assertIsNone(fallback)

        assert True  # TODO: Add proper assertion
    
    def test_get_quality_threshold(self):
        """Test quality threshold determination"""
        threshold = self.controller._get_quality_threshold(QuantizationMethod.NONE)
        self.assertEqual(threshold, 1.0)
        
        threshold = self.controller._get_quality_threshold(QuantizationMethod.BF16)
        self.assertEqual(threshold, 0.95)
        
        threshold = self.controller._get_quality_threshold(QuantizationMethod.INT8)
        self.assertEqual(threshold, 0.85)
        
        threshold = self.controller._get_quality_threshold(QuantizationMethod.FP8)
        self.assertEqual(threshold, 0.80)

        assert True  # TODO: Add proper assertion
    
    def test_get_model_components(self):
        """Test model component detection"""
        mock_model = MagicMock()
        mock_model.unet = MagicMock()
        mock_model.text_encoder = MagicMock()
        mock_model.vae = MagicMock()
        mock_model.scheduler = MagicMock()
        
        # Mock hasattr to return True for existing components
        with patch('builtins.hasattr') as mock_hasattr:
            def hasattr_side_effect(obj, name):
                return name in ['unet', 'text_encoder', 'vae', 'scheduler']
            mock_hasattr.side_effect = hasattr_side_effect
            
            components = self.controller._get_model_components(mock_model)
        
        self.assertIn('unet', components)
        self.assertIn('text_encoder', components)
        self.assertIn('vae', components)
        self.assertIn('scheduler', components)

        assert True  # TODO: Add proper assertion
    
    def test_quantize_component_fp16(self):
        """Test quantizing component to FP16"""
        mock_component = MagicMock()
        
        self.controller._quantize_component(mock_component, QuantizationMethod.FP16)
        
        # Should call half() method
        mock_component.half.assert_called_once()

        assert True  # TODO: Add proper assertion
    
    @patch('quantization_controller.torch')
    def test_quantize_component_bf16(self, mock_torch):
        """Test quantizing component to BF16"""
        mock_component = MagicMock()
        mock_torch.bfloat16 = MagicMock()
        
        self.controller._quantize_component(mock_component, QuantizationMethod.BF16)
        
        # Should call to() method with bfloat16
        mock_component.to.assert_called_once_with(dtype=mock_torch.bfloat16)

        assert True  # TODO: Add proper assertion
    
    def test_quantize_component_none(self):
        """Test quantizing component with NONE method (no-op)"""
        mock_component = MagicMock()
        
        self.controller._quantize_component(mock_component, QuantizationMethod.NONE)
        
        # Should not call any methods
        mock_component.half.assert_not_called()
        mock_component.to.assert_not_called()

        assert True  # TODO: Add proper assertion
    
    def test_is_large_component(self):
        """Test large component detection"""
        # Mock component with many parameters
        mock_large_component = MagicMock()
        mock_param = MagicMock()
        mock_param.numel.return_value = 2e9  # 2B parameters
        mock_large_component.parameters.return_value = [mock_param]
        
        is_large = self.controller._is_large_component(mock_large_component)
        self.assertTrue(is_large)
        
        # Mock component with few parameters
        mock_small_component = MagicMock()
        mock_param.numel.return_value = 1e8  # 100M parameters
        mock_small_component.parameters.return_value = [mock_param]
        
        is_large = self.controller._is_large_component(mock_small_component)
        self.assertFalse(is_large)

        assert True  # TODO: Add proper assertion
    
    @patch('quantization_controller.torch')
    def test_get_current_vram_usage(self, mock_torch):
        """Test VRAM usage measurement"""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 8 * 1024 * 1024 * 1024  # 8GB
        
        usage = self.controller._get_current_vram_usage()
        
        self.assertEqual(usage, 8 * 1024)  # Should return MB

        assert True  # TODO: Add proper assertion
    
    def test_set_progress_callback(self):
        """Test setting progress callback"""
        callback = MagicMock()
        
        self.controller.set_progress_callback(callback)
        
        self.assertIn(callback, self.controller._progress_callback)

        assert True  # TODO: Add proper assertion
    
    def test_cancel_quantization(self):
        """Test quantization cancellation"""
        # Start a mock operation
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = True
        self.controller._current_operation = mock_thread
        
        result = self.controller.cancel_quantization()
        
        self.assertTrue(self.controller._cancellation_event.is_set())
        mock_thread.join.assert_called_once()

        assert True  # TODO: Add proper assertion
    
    def test_validate_quantization_compatibility_compatible(self):
        """Test quantization compatibility validation for compatible setup"""
        model_info = ModelInfo(
            name="test-model",
            size_gb=5.0,
            architecture="transformer",
            components=["unet"],
            estimated_vram_usage=6144.0
        )
        
        self.controller.hardware_profile.supports_bf16 = True
        self.controller.hardware_profile.vram_gb = 16
        
        compatibility = self.controller.validate_quantization_compatibility(
            model_info, QuantizationMethod.BF16
        )
        
        self.assertTrue(compatibility["compatible"])
        self.assertEqual(len(compatibility["warnings"]), 0)
        self.assertLess(compatibility["estimated_memory_usage"], 8192)  # Should be reduced

        assert True  # TODO: Add proper assertion
    
    def test_validate_quantization_compatibility_incompatible(self):
        """Test quantization compatibility validation for incompatible setup"""
        model_info = ModelInfo(
            name="test-model",
            size_gb=5.0,
            architecture="transformer",
            components=["unet"],
            estimated_vram_usage=6144.0
        )
        
        self.controller.hardware_profile.supports_bf16 = False
        
        compatibility = self.controller.validate_quantization_compatibility(
            model_info, QuantizationMethod.BF16
        )
        
        self.assertFalse(compatibility["compatible"])
        self.assertGreater(len(compatibility["warnings"]), 0)
        self.assertIn("does not support BF16", compatibility["warnings"][0])

        assert True  # TODO: Add proper assertion
    
    def test_validate_quantization_compatibility_vram_warning(self):
        """Test quantization compatibility with VRAM warning"""
        model_info = ModelInfo(
            name="large-model",
            size_gb=20.0,
            architecture="transformer",
            components=["unet"],
            estimated_vram_usage=20480.0  # 20GB
        )
        
        self.controller.hardware_profile.vram_gb = 16  # Only 16GB available
        
        compatibility = self.controller.validate_quantization_compatibility(
            model_info, QuantizationMethod.NONE
        )
        
        self.assertTrue(any("exceeds 90%" in warning for warning in compatibility["warnings"]))

        assert True  # TODO: Add proper assertion
    
    @patch('quantization_controller.torch')
    def test_get_supported_methods_with_cuda(self, mock_torch):
        """Test getting supported methods with CUDA"""
        mock_torch.cuda.is_available.return_value = True
        self.controller.hardware_profile.supports_bf16 = True
        self.controller.hardware_profile.supports_int8 = True
        self.controller.hardware_profile.supports_fp8 = False
        
        methods = self.controller.get_supported_methods()
        
        self.assertIn(QuantizationMethod.NONE, methods)
        self.assertIn(QuantizationMethod.FP16, methods)
        self.assertIn(QuantizationMethod.BF16, methods)
        self.assertIn(QuantizationMethod.INT8, methods)
        self.assertNotIn(QuantizationMethod.FP8, methods)

        assert True  # TODO: Add proper assertion
    
    @patch('quantization_controller.torch')
    def test_get_supported_methods_cpu_only(self, mock_torch):
        """Test getting supported methods with CPU only"""
        mock_torch.cuda.is_available.return_value = False
        
        methods = self.controller.get_supported_methods()
        
        self.assertEqual(methods, [QuantizationMethod.NONE])

        assert True  # TODO: Add proper assertion
    
    def test_update_preferences(self):
        """Test updating user preferences"""
        new_preferences = UserPreferences(
            preferred_method=QuantizationMethod.INT8,
            auto_fallback_enabled=False,
            timeout_seconds=600,
            skip_quality_check=True,
            remember_model_settings=False,
            model_specific_preferences={"test": QuantizationMethod.BF16}
        )
        
        self.controller.update_preferences(new_preferences)
        
        self.assertEqual(self.controller.preferences.preferred_method, QuantizationMethod.INT8)
        self.assertFalse(self.controller.preferences.auto_fallback_enabled)
        self.assertEqual(self.controller.preferences.timeout_seconds, 600)
        self.assertTrue(self.controller.preferences.skip_quality_check)
        self.assertFalse(self.controller.preferences.remember_model_settings)

        assert True  # TODO: Add proper assertion
    
    def test_get_preferences(self):
        """Test getting current preferences"""
        preferences = self.controller.get_preferences()
        
        self.assertIsInstance(preferences, UserPreferences)
        self.assertEqual(preferences, self.controller.preferences)


        assert True  # TODO: Add proper assertion

class TestQuantizationMethod(unittest.TestCase):
    """Test cases for QuantizationMethod enum"""
    
    def test_quantization_method_values(self):
        """Test QuantizationMethod enum values"""
        self.assertEqual(QuantizationMethod.NONE.value, "none")
        self.assertEqual(QuantizationMethod.FP16.value, "fp16")
        self.assertEqual(QuantizationMethod.BF16.value, "bf16")
        self.assertEqual(QuantizationMethod.INT8.value, "int8")
        self.assertEqual(QuantizationMethod.FP8.value, "fp8")


        assert True  # TODO: Add proper assertion

class TestQuantizationStatus(unittest.TestCase):
    """Test cases for QuantizationStatus enum"""
    
    def test_quantization_status_values(self):
        """Test QuantizationStatus enum values"""
        self.assertEqual(QuantizationStatus.NOT_STARTED.value, "not_started")
        self.assertEqual(QuantizationStatus.IN_PROGRESS.value, "in_progress")
        self.assertEqual(QuantizationStatus.COMPLETED.value, "completed")
        self.assertEqual(QuantizationStatus.TIMEOUT.value, "timeout")
        self.assertEqual(QuantizationStatus.CANCELLED.value, "cancelled")
        self.assertEqual(QuantizationStatus.FAILED.value, "failed")


        assert True  # TODO: Add proper assertion

class TestHardwareProfile(unittest.TestCase):
    """Test cases for HardwareProfile dataclass"""
    
    def test_hardware_profile_creation(self):
        """Test HardwareProfile creation"""
        profile = HardwareProfile(
            gpu_model="NVIDIA GeForce RTX 4080",
            vram_gb=16,
            cuda_version="12.1",
            driver_version="531.79",
            compute_capability=(8, 6),
            supports_bf16=True,
            supports_fp8=True,
            supports_int8=True
        )
        
        self.assertEqual(profile.gpu_model, "NVIDIA GeForce RTX 4080")
        self.assertEqual(profile.vram_gb, 16)
        self.assertEqual(profile.cuda_version, "12.1")
        self.assertEqual(profile.driver_version, "531.79")
        self.assertEqual(profile.compute_capability, (8, 6))
        self.assertTrue(profile.supports_bf16)
        self.assertTrue(profile.supports_fp8)
        self.assertTrue(profile.supports_int8)


        assert True  # TODO: Add proper assertion

class TestModelInfo(unittest.TestCase):
    """Test cases for ModelInfo dataclass"""
    
    def test_model_info_creation(self):
        """Test ModelInfo creation"""
        model_info = ModelInfo(
            name="test-model",
            size_gb=5.2,
            architecture="transformer",
            components=["unet", "text_encoder", "vae"],
            estimated_vram_usage=6144.0
        )
        
        self.assertEqual(model_info.name, "test-model")
        self.assertEqual(model_info.size_gb, 5.2)
        self.assertEqual(model_info.architecture, "transformer")
        self.assertEqual(model_info.components, ["unet", "text_encoder", "vae"])
        self.assertEqual(model_info.estimated_vram_usage, 6144.0)


        assert True  # TODO: Add proper assertion

class TestQuantizationStrategy(unittest.TestCase):
    """Test cases for QuantizationStrategy dataclass"""
    
    def test_quantization_strategy_creation(self):
        """Test QuantizationStrategy creation"""
        strategy = QuantizationStrategy(
            method=QuantizationMethod.BF16,
            timeout_seconds=300,
            skip_large_components=False,
            component_priorities={"unet": 10, "vae": 3},
            fallback_method=QuantizationMethod.FP16,
            quality_threshold=0.95
        )
        
        self.assertEqual(strategy.method, QuantizationMethod.BF16)
        self.assertEqual(strategy.timeout_seconds, 300)
        self.assertFalse(strategy.skip_large_components)
        self.assertEqual(strategy.component_priorities, {"unet": 10, "vae": 3})
        self.assertEqual(strategy.fallback_method, QuantizationMethod.FP16)
        self.assertEqual(strategy.quality_threshold, 0.95)


        assert True  # TODO: Add proper assertion

class TestQuantizationProgress(unittest.TestCase):
    """Test cases for QuantizationProgress dataclass"""
    
    def test_quantization_progress_creation(self):
        """Test QuantizationProgress creation"""
        progress = QuantizationProgress(
            current_component="unet",
            components_completed=2,
            total_components=5,
            elapsed_seconds=45.5,
            estimated_remaining_seconds=67.8,
            memory_usage_mb=8192.0,
            status=QuantizationStatus.IN_PROGRESS
        )
        
        self.assertEqual(progress.current_component, "unet")
        self.assertEqual(progress.components_completed, 2)
        self.assertEqual(progress.total_components, 5)
        self.assertEqual(progress.elapsed_seconds, 45.5)
        self.assertEqual(progress.estimated_remaining_seconds, 67.8)
        self.assertEqual(progress.memory_usage_mb, 8192.0)
        self.assertEqual(progress.status, QuantizationStatus.IN_PROGRESS)


        assert True  # TODO: Add proper assertion

class TestQuantizationResult(unittest.TestCase):
    """Test cases for QuantizationResult dataclass"""
    
    def test_quantization_result_success(self):
        """Test QuantizationResult for successful quantization"""
        result = QuantizationResult(
            success=True,
            method_used=QuantizationMethod.BF16,
            components_quantized=["unet", "text_encoder"],
            memory_saved_mb=2048.0,
            time_taken_seconds=120.5,
            quality_score=0.96,
            warnings=["Minor quality impact on VAE"],
            errors=[],
            status=QuantizationStatus.COMPLETED
        )
        
        self.assertTrue(result.success)
        self.assertEqual(result.method_used, QuantizationMethod.BF16)
        self.assertEqual(result.components_quantized, ["unet", "text_encoder"])
        self.assertEqual(result.memory_saved_mb, 2048.0)
        self.assertEqual(result.time_taken_seconds, 120.5)
        self.assertEqual(result.quality_score, 0.96)
        self.assertEqual(len(result.warnings), 1)
        self.assertEqual(len(result.errors), 0)
        self.assertEqual(result.status, QuantizationStatus.COMPLETED)

        assert True  # TODO: Add proper assertion
    
    def test_quantization_result_failure(self):
        """Test QuantizationResult for failed quantization"""
        result = QuantizationResult(
            success=False,
            method_used=QuantizationMethod.INT8,
            components_quantized=[],
            memory_saved_mb=0.0,
            time_taken_seconds=300.0,
            quality_score=None,
            warnings=[],
            errors=["Quantization timeout", "Hardware incompatibility"],
            status=QuantizationStatus.TIMEOUT
        )
        
        self.assertFalse(result.success)
        self.assertEqual(result.method_used, QuantizationMethod.INT8)
        self.assertEqual(len(result.components_quantized), 0)
        self.assertEqual(result.memory_saved_mb, 0.0)
        self.assertIsNone(result.quality_score)
        self.assertEqual(len(result.errors), 2)
        self.assertEqual(result.status, QuantizationStatus.TIMEOUT)


        assert True  # TODO: Add proper assertion

class TestUserPreferences(unittest.TestCase):
    """Test cases for UserPreferences dataclass"""
    
    def test_user_preferences_defaults(self):
        """Test UserPreferences default values"""
        preferences = UserPreferences(
            preferred_method=QuantizationMethod.BF16,
            auto_fallback_enabled=True,
            timeout_seconds=300,
            skip_quality_check=False,
            remember_model_settings=True,
            model_specific_preferences={}
        )
        
        self.assertEqual(preferences.preferred_method, QuantizationMethod.BF16)
        self.assertTrue(preferences.auto_fallback_enabled)
        self.assertEqual(preferences.timeout_seconds, 300)
        self.assertFalse(preferences.skip_quality_check)
        self.assertTrue(preferences.remember_model_settings)
        self.assertEqual(preferences.model_specific_preferences, {})


        assert True  # TODO: Add proper assertion

class TestQuantizationControllerIntegration(unittest.TestCase):
    """Integration tests for QuantizationController"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "config.json"
        self.preferences_path = Path(self.temp_dir) / "preferences.json"
        
        # Create config files
        with open(self.config_path, 'w') as f:
            json.dump({}, f)
        
        self.controller = QuantizationController(
            config_path=str(self.config_path),
            preferences_path=str(self.preferences_path)
        )
    
    def tearDown(self):
        """Clean up integration test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_end_to_end_strategy_determination(self):
        """Test complete strategy determination workflow"""
        # Create a realistic model info
        model_info = ModelInfo(
            name="stabilityai/stable-diffusion-2-1",
            size_gb=5.2,
            architecture="stable-diffusion",
            components=["unet", "text_encoder", "vae", "scheduler"],
            estimated_vram_usage=6144.0
        )
        
        # Set hardware profile
        self.controller.hardware_profile.supports_bf16 = True
        self.controller.hardware_profile.supports_int8 = True
        self.controller.hardware_profile.vram_gb = 16
        
        # Determine strategy
        strategy = self.controller.determine_optimal_strategy(model_info)
        
        # Verify strategy is reasonable
        self.assertIsInstance(strategy, QuantizationStrategy)
        self.assertIn(strategy.method, [QuantizationMethod.BF16, QuantizationMethod.FP16, QuantizationMethod.INT8])
        self.assertGreater(strategy.timeout_seconds, 0)
        self.assertIsInstance(strategy.component_priorities, dict)
        self.assertIn("unet", strategy.component_priorities)

        assert True  # TODO: Add proper assertion
    
    def test_preferences_persistence(self):
        """Test that preferences are saved and loaded correctly"""
        # Update preferences
        new_preferences = UserPreferences(
            preferred_method=QuantizationMethod.INT8,
            auto_fallback_enabled=False,
            timeout_seconds=600,
            skip_quality_check=True,
            remember_model_settings=False,
            model_specific_preferences={"test-model": QuantizationMethod.BF16}
        )
        
        self.controller.update_preferences(new_preferences)
        
        # Create new controller instance to test loading
        new_controller = QuantizationController(
            config_path=str(self.config_path),
            preferences_path=str(self.preferences_path)
        )
        
        # Verify preferences were loaded correctly
        loaded_prefs = new_controller.get_preferences()
        self.assertEqual(loaded_prefs.preferred_method, QuantizationMethod.INT8)
        self.assertFalse(loaded_prefs.auto_fallback_enabled)
        self.assertEqual(loaded_prefs.timeout_seconds, 600)
        self.assertTrue(loaded_prefs.skip_quality_check)
        self.assertFalse(loaded_prefs.remember_model_settings)
        self.assertEqual(loaded_prefs.model_specific_preferences["test-model"], QuantizationMethod.BF16)


        assert True  # TODO: Add proper assertion

if __name__ == '__main__':
    unittest.main()