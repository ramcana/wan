#!/usr/bin/env python3
"""
Simple unit tests for Enhanced Model Management System
Tests core functionality without requiring full torch installation
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
import json
import os
from pathlib import Path
from datetime import datetime, timedelta
import sys

# Mock torch before importing enhanced_model_manager
sys.modules['torch'] = Mock()
sys.modules['torch.nn'] = Mock()
sys.modules['torch.cuda'] = Mock()
sys.modules['transformers'] = Mock()
sys.modules['diffusers'] = Mock()
sys.modules['huggingface_hub'] = Mock()
sys.modules['huggingface_hub.utils'] = Mock()
sys.modules['psutil'] = Mock()
sys.modules['PIL'] = Mock()

# Mock error handler
sys.modules['error_handler'] = Mock()

# Import after mocking
from enhanced_model_manager import (
    EnhancedModelManager,
    ModelStatus,
    GenerationMode,
    ModelCompatibility,
    ModelMetadata,
    ModelLoadingResult,
    CompatibilityCheck
)

class TestEnhancedModelManagerCore(unittest.TestCase):
    """Test core functionality of EnhancedModelManager"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            "directories": {
                "models_directory": self.temp_dir,
                "outputs_directory": "outputs",
                "loras_directory": "loras"
            },
            "optimization": {"max_vram_usage_gb": 12},
            "model_validation": {
                "validate_on_startup": False,
                "validation_interval_hours": 24,
                "auto_repair_corrupted": True
            }
        }
        
        # Create config file
        self.config_path = os.path.join(self.temp_dir, "test_config.json")
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f)
        
        # Initialize manager
        self.manager = EnhancedModelManager(self.config_path)
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test manager initialization"""
        self.assertIsInstance(self.manager, EnhancedModelManager)
        self.assertEqual(str(self.manager.cache_dir), self.temp_dir)
        
        # Check model registry initialization
        self.assertIn("Wan-AI/Wan2.2-T2V-A14B-Diffusers", self.manager.model_registry)
        self.assertIn("Wan-AI/Wan2.2-I2V-A14B-Diffusers", self.manager.model_registry)
        self.assertIn("Wan-AI/Wan2.2-TI2V-5B-Diffusers", self.manager.model_registry)
        
        # Check model status initialization
        for model_id in self.manager.model_registry.keys():
            self.assertEqual(self.manager.model_status[model_id], ModelStatus.UNKNOWN)
    
    def test_config_loading_fallback(self):
        """Test config loading with fallback on error"""
        # Test with non-existent config file
        manager = EnhancedModelManager("non_existent_config.json")
        self.assertIsInstance(manager.config, dict)
        self.assertIn("directories", manager.config)
        self.assertIn("optimization", manager.config)
        self.assertIn("model_validation", manager.config)
    
    def test_get_model_id(self):
        """Test model ID resolution"""
        test_cases = [
            ("t2v-A14B", "Wan-AI/Wan2.2-T2V-A14B-Diffusers"),
            ("i2v-A14B", "Wan-AI/Wan2.2-I2V-A14B-Diffusers"),
            ("ti2v-5B", "Wan-AI/Wan2.2-TI2V-5B-Diffusers"),
            ("custom/model", "custom/model")
        ]
        
        for input_id, expected_id in test_cases:
            with self.subTest(input_id=input_id):
                result = self.manager.get_model_id(input_id)
                self.assertEqual(result, expected_id)
    
    def test_model_metadata_structure(self):
        """Test model metadata structure"""
        model_id = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
        metadata = self.manager.model_registry[model_id]
        
        self.assertIsInstance(metadata, ModelMetadata)
        self.assertEqual(metadata.model_id, model_id)
        self.assertEqual(metadata.model_type, "text-to-video")
        self.assertIn(GenerationMode.TEXT_TO_VIDEO, metadata.generation_modes)
        self.assertIn("1280x720", metadata.supported_resolutions)
        self.assertGreater(metadata.min_vram_mb, 0)
        self.assertGreater(metadata.recommended_vram_mb, metadata.min_vram_mb)
        self.assertIn("bf16", metadata.quantization_support)
        self.assertTrue(metadata.cpu_offload_support)
        self.assertTrue(metadata.vae_tiling_support)
    
    def test_check_local_model_missing(self):
        """Test local model check when model is missing"""
        model_id = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
        status = self.manager._check_local_model(model_id)
        self.assertEqual(status, ModelStatus.MISSING)
    
    def test_check_local_model_corrupted_missing_config(self):
        """Test local model check when config.json is missing"""
        model_id = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
        model_path = self.manager.cache_dir / model_id.replace("/", "_")
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Create model without config.json
        (model_path / "some_file.bin").touch()
        
        status = self.manager._check_local_model(model_id)
        self.assertEqual(status, ModelStatus.CORRUPTED)
    
    def test_check_local_model_corrupted_missing_weights(self):
        """Test local model check when model weights are missing"""
        model_id = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
        model_path = self.manager.cache_dir / model_id.replace("/", "_")
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Create config but no weights
        config = {"model_type": "diffusion"}
        with open(model_path / "config.json", 'w') as f:
            json.dump(config, f)
        
        status = self.manager._check_local_model(model_id)
        self.assertEqual(status, ModelStatus.CORRUPTED)
    
    def test_check_local_model_available(self):
        """Test local model check when model is available"""
        model_id = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
        model_path = self.manager.cache_dir / model_id.replace("/", "_")
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Create valid model files
        config = {"model_type": "diffusion"}
        with open(model_path / "config.json", 'w') as f:
            json.dump(config, f)
        (model_path / "pytorch_model.bin").touch()
        
        status = self.manager._check_local_model(model_id)
        self.assertEqual(status, ModelStatus.AVAILABLE)
    
    def test_check_local_model_loaded(self):
        """Test local model check when model is loaded"""
        model_id = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
        
        # Mock loaded model
        self.manager.loaded_models[model_id] = Mock()
        
        status = self.manager._check_local_model(model_id)
        self.assertEqual(status, ModelStatus.LOADED)
    
    def test_check_local_model_config_validation(self):
        """Test local model config validation with hash checking"""
        model_id = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
        model_path = self.manager.cache_dir / model_id.replace("/", "_")
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Create valid model files
        config = {"model_type": "diffusion", "version": "1.0"}
        with open(model_path / "config.json", 'w') as f:
            json.dump(config, f)
        (model_path / "pytorch_model.bin").touch()
        
        # First check should set the hash
        status1 = self.manager._check_local_model(model_id)
        self.assertEqual(status1, ModelStatus.AVAILABLE)
        self.assertIsNotNone(self.manager.model_registry[model_id].config_hash)
        
        # Second check should validate against stored hash
        status2 = self.manager._check_local_model(model_id)
        self.assertEqual(status2, ModelStatus.AVAILABLE)
    
    def test_check_local_model_config_hash_mismatch(self):
        """Test local model config validation with hash mismatch"""
        model_id = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
        model_path = self.manager.cache_dir / model_id.replace("/", "_")
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Set a fake hash in registry
        self.manager.model_registry[model_id].config_hash = "fake_hash"
        
        # Create model with different config
        config = {"model_type": "diffusion", "version": "2.0"}
        with open(model_path / "config.json", 'w') as f:
            json.dump(config, f)
        (model_path / "pytorch_model.bin").touch()
        
        status = self.manager._check_local_model(model_id)
        self.assertEqual(status, ModelStatus.CORRUPTED)
    
    def test_check_local_model_invalid_json(self):
        """Test local model check with invalid JSON config"""
        model_id = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
        model_path = self.manager.cache_dir / model_id.replace("/", "_")
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Create invalid JSON config
        with open(model_path / "config.json", 'w') as f:
            f.write("invalid json content {")
        (model_path / "pytorch_model.bin").touch()
        
        status = self.manager._check_local_model(model_id)
        self.assertEqual(status, ModelStatus.CORRUPTED)
    
    def test_compatibility_check_fully_compatible(self):
        """Test model compatibility check for fully compatible scenario"""
        model_id = "t2v-A14B"
        generation_mode = GenerationMode.TEXT_TO_VIDEO
        resolution = "1280x720"
        
        # Mock sufficient VRAM
        with patch.object(self.manager, '_get_vram_info') as mock_vram:
            mock_vram.return_value = {"total_mb": 10000, "used_mb": 2000, "free_mb": 8000}
            
            # Mock sufficient disk space
            with patch('enhanced_model_manager.psutil') as mock_psutil:
                mock_disk_usage = Mock()
                mock_disk_usage.free = 50 * 1024**3  # 50GB free
                mock_psutil.disk_usage.return_value = mock_disk_usage
                
                compat = self.manager.check_model_compatibility(model_id, generation_mode, resolution)
                
                self.assertEqual(compat.model_id, "Wan-AI/Wan2.2-T2V-A14B-Diffusers")
                self.assertEqual(compat.generation_mode, generation_mode)
                self.assertEqual(compat.resolution, resolution)
                self.assertEqual(compat.compatibility, ModelCompatibility.FULLY_COMPATIBLE)
                self.assertEqual(len(compat.issues), 0)
    
    def test_compatibility_check_incompatible_mode(self):
        """Test model compatibility check for incompatible generation mode"""
        model_id = "t2v-A14B"  # Only supports TEXT_TO_VIDEO
        generation_mode = GenerationMode.IMAGE_TO_VIDEO  # Incompatible
        resolution = "1280x720"
        
        compat = self.manager.check_model_compatibility(model_id, generation_mode, resolution)
        
        self.assertEqual(compat.compatibility, ModelCompatibility.INCOMPATIBLE)
        self.assertTrue(any("does not support i2v generation" in issue for issue in compat.issues))
    
    def test_compatibility_check_insufficient_vram(self):
        """Test model compatibility check with insufficient VRAM"""
        model_id = "t2v-A14B"
        generation_mode = GenerationMode.TEXT_TO_VIDEO
        resolution = "1280x720"
        
        # Mock insufficient VRAM
        with patch.object(self.manager, '_get_vram_info') as mock_vram:
            mock_vram.return_value = {"total_mb": 4000, "used_mb": 1000, "free_mb": 3000}  # Less than required 6000MB
            
            compat = self.manager.check_model_compatibility(model_id, generation_mode, resolution)
            
            self.assertEqual(compat.compatibility, ModelCompatibility.INCOMPATIBLE)
            self.assertTrue(any("Insufficient VRAM" in issue for issue in compat.issues))
            self.assertTrue(any("quantization or CPU offload" in rec for rec in compat.recommendations))
    
    def test_compatibility_check_unsupported_resolution(self):
        """Test model compatibility check with unsupported resolution"""
        model_id = "t2v-A14B"
        generation_mode = GenerationMode.TEXT_TO_VIDEO
        resolution = "4096x2160"  # Not in supported resolutions
        
        # Mock sufficient VRAM
        with patch.object(self.manager, '_get_vram_info') as mock_vram:
            mock_vram.return_value = {"total_mb": 10000, "used_mb": 2000, "free_mb": 8000}
            
            # Mock sufficient disk space
            with patch('enhanced_model_manager.psutil') as mock_psutil:
                mock_disk_usage = Mock()
                mock_disk_usage.free = 50 * 1024**3
                mock_psutil.disk_usage.return_value = mock_disk_usage
                
                compat = self.manager.check_model_compatibility(model_id, generation_mode, resolution)
                
                self.assertEqual(compat.compatibility, ModelCompatibility.PARTIALLY_COMPATIBLE)
                self.assertTrue(any("not officially supported" in issue for issue in compat.issues))
                self.assertTrue(any("Consider using supported resolutions" in rec for rec in compat.recommendations))
    
    def test_compatibility_check_insufficient_disk_space(self):
        """Test model compatibility check with insufficient disk space"""
        model_id = "t2v-A14B"
        generation_mode = GenerationMode.TEXT_TO_VIDEO
        resolution = "1280x720"
        
        # Mock sufficient VRAM
        with patch.object(self.manager, '_get_vram_info') as mock_vram:
            mock_vram.return_value = {"total_mb": 10000, "used_mb": 2000, "free_mb": 8000}
            
            # Mock insufficient disk space
            with patch('enhanced_model_manager.psutil') as mock_psutil:
                mock_disk_usage = Mock()
                mock_disk_usage.free = 1 * 1024**3  # Only 1GB free
                mock_psutil.disk_usage.return_value = mock_disk_usage
                
                compat = self.manager.check_model_compatibility(model_id, generation_mode, resolution)
                
                self.assertEqual(compat.compatibility, ModelCompatibility.INCOMPATIBLE)
                self.assertTrue(any("Insufficient disk space" in issue for issue in compat.issues))
    
    def test_compatibility_check_below_recommended_vram(self):
        """Test model compatibility check with VRAM below recommended but above minimum"""
        model_id = "t2v-A14B"
        generation_mode = GenerationMode.TEXT_TO_VIDEO
        resolution = "1280x720"
        
        # Mock VRAM between minimum (6000MB) and recommended (8000MB)
        with patch.object(self.manager, '_get_vram_info') as mock_vram:
            mock_vram.return_value = {"total_mb": 7000, "used_mb": 1000, "free_mb": 6000}
            
            # Mock sufficient disk space
            with patch('enhanced_model_manager.psutil') as mock_psutil:
                mock_disk_usage = Mock()
                mock_disk_usage.free = 50 * 1024**3
                mock_psutil.disk_usage.return_value = mock_disk_usage
                
                compat = self.manager.check_model_compatibility(model_id, generation_mode, resolution)
                
                self.assertEqual(compat.compatibility, ModelCompatibility.PARTIALLY_COMPATIBLE)
                self.assertTrue(any("Below recommended VRAM" in issue for issue in compat.issues))
                self.assertTrue(any("Performance may be reduced" in issue for issue in compat.issues))
    
    def test_get_directory_size_mb(self):
        """Test directory size calculation"""
        # Create test directory with files
        test_dir = Path(self.temp_dir) / "test_size"
        test_dir.mkdir()
        
        # Create files of known sizes
        (test_dir / "file1.txt").write_bytes(b"x" * 1024)  # 1KB
        (test_dir / "file2.txt").write_bytes(b"x" * 2048)  # 2KB
        
        size_mb = self.manager._get_directory_size_mb(test_dir)
        expected_mb = (1024 + 2048) / (1024 * 1024)  # Convert to MB
        
        self.assertAlmostEqual(size_mb, expected_mb, places=6)
    
    def test_get_directory_size_mb_nonexistent(self):
        """Test directory size calculation for non-existent directory"""
        nonexistent_dir = Path(self.temp_dir) / "nonexistent"
        size_mb = self.manager._get_directory_size_mb(nonexistent_dir)
        self.assertEqual(size_mb, 0.0)
    
    def test_unload_model(self):
        """Test model unloading"""
        model_id = "t2v-A14B"
        full_model_id = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
        mock_model = Mock()
        
        # Pre-load model
        self.manager.loaded_models[full_model_id] = mock_model
        self.manager.model_status[full_model_id] = ModelStatus.LOADED
        
        # Mock torch.cuda
        with patch('enhanced_model_manager.torch') as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.empty_cache = Mock()
            
            self.manager.unload_model(model_id)
            
            # Model should be removed from loaded models
            self.assertNotIn(full_model_id, self.manager.loaded_models)
            self.assertEqual(self.manager.model_status[full_model_id], ModelStatus.AVAILABLE)
            
            # GPU cache should be cleared
            mock_torch.cuda.empty_cache.assert_called_once()
    
    def test_unload_model_not_loaded(self):
        """Test unloading model that is not loaded"""
        model_id = "t2v-A14B"
        full_model_id = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
        
        # Ensure model is not loaded
        self.assertNotIn(full_model_id, self.manager.loaded_models)
        
        # Should not raise exception
        self.manager.unload_model(model_id)
    
    def test_model_loading_result_structure(self):
        """Test ModelLoadingResult data structure"""
        result = ModelLoadingResult(success=True)
        
        self.assertTrue(result.success)
        self.assertIsNone(result.model)
        self.assertIsNone(result.metadata)
        self.assertIsNone(result.error_message)
        self.assertFalse(result.fallback_applied)
        self.assertEqual(result.optimization_applied, {})
        self.assertEqual(result.loading_time_seconds, 0.0)
        self.assertEqual(result.memory_usage_mb, 0.0)
    
    def test_compatibility_check_structure(self):
        """Test CompatibilityCheck data structure"""
        check = CompatibilityCheck(
            model_id="test-model",
            generation_mode=GenerationMode.TEXT_TO_VIDEO,
            resolution="1280x720",
            compatibility=ModelCompatibility.FULLY_COMPATIBLE
        )
        
        self.assertEqual(check.model_id, "test-model")
        self.assertEqual(check.generation_mode, GenerationMode.TEXT_TO_VIDEO)
        self.assertEqual(check.resolution, "1280x720")
        self.assertEqual(check.compatibility, ModelCompatibility.FULLY_COMPATIBLE)
        self.assertEqual(check.issues, [])
        self.assertEqual(check.recommendations, [])
        self.assertEqual(check.estimated_vram_mb, 0.0)


class TestModelStatusValidation(unittest.TestCase):
    """Test model status validation functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            "directories": {"models_directory": self.temp_dir},
            "optimization": {"max_vram_usage_gb": 12},
            "model_validation": {"validate_on_startup": False}
        }
        
        config_path = os.path.join(self.temp_dir, "test_config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config, f)
        
        self.manager = EnhancedModelManager(config_path)
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_validate_model_availability_cached_result(self):
        """Test model availability validation with cached result"""
        model_id = "t2v-A14B"
        full_model_id = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
        
        # Set recent validation
        self.manager.model_status[full_model_id] = ModelStatus.AVAILABLE
        self.manager.model_registry[full_model_id].last_validated = datetime.now()
        
        # Should return cached result without checking
        status = self.manager.validate_model_availability(model_id, force_check=False)
        self.assertEqual(status, ModelStatus.AVAILABLE)
    
    def test_validate_model_availability_force_check(self):
        """Test model availability validation with force check"""
        model_id = "t2v-A14B"
        full_model_id = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
        
        # Set old validation
        self.manager.model_status[full_model_id] = ModelStatus.AVAILABLE
        self.manager.model_registry[full_model_id].last_validated = datetime.now() - timedelta(hours=2)
        
        # Mock local and remote checks
        with patch.object(self.manager, '_check_local_model') as mock_local:
            mock_local.return_value = ModelStatus.MISSING
            
            with patch.object(self.manager, '_check_remote_model') as mock_remote:
                mock_remote.return_value = ModelStatus.AVAILABLE
                
                status = self.manager.validate_model_availability(model_id, force_check=True)
                
                self.assertEqual(status, ModelStatus.AVAILABLE)
                mock_local.assert_called_once()
                mock_remote.assert_called_once()
    
    def test_validate_model_availability_local_available(self):
        """Test model availability validation when model is locally available"""
        model_id = "t2v-A14B"
        full_model_id = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
        
        # Mock local check to return available
        with patch.object(self.manager, '_check_local_model') as mock_local:
            mock_local.return_value = ModelStatus.AVAILABLE
            
            status = self.manager.validate_model_availability(model_id, force_check=True)
            
            self.assertEqual(status, ModelStatus.AVAILABLE)
            self.assertEqual(self.manager.model_status[full_model_id], ModelStatus.AVAILABLE)
            mock_local.assert_called_once()
    
    def test_validate_model_availability_error_handling(self):
        """Test model availability validation error handling"""
        model_id = "t2v-A14B"
        full_model_id = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
        
        # Mock local check to raise exception
        with patch.object(self.manager, '_check_local_model') as mock_local:
            mock_local.side_effect = Exception("Test error")
            
            status = self.manager.validate_model_availability(model_id, force_check=True)
            
            self.assertEqual(status, ModelStatus.ERROR)
            self.assertEqual(self.manager.model_status[full_model_id], ModelStatus.ERROR)


def run_tests():
    """Run all tests"""
    test_classes = [
        TestEnhancedModelManagerCore,
        TestModelStatusValidation
    ]
    
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            error_msg = traceback.split('AssertionError: ')[-1].split('\n')[0] if 'AssertionError: ' in traceback else 'Unknown failure'
            print(f"- {test}: {error_msg}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            error_msg = traceback.split('\n')[-2] if traceback else 'Unknown error'
            print(f"- {test}: {error_msg}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("Enhanced Model Manager - Simple Unit Tests")
    print("=" * 50)
    success = run_tests()
    
    if success:
        print("\n🎉 All tests passed! Enhanced model management system is working correctly.")
        print("\nKey features tested:")
        print("- Model registry initialization and metadata management")
        print("- Model ID resolution and mapping")
        print("- Local model validation (missing, corrupted, available)")
        print("- Model compatibility checking for different scenarios")
        print("- Configuration loading with fallback")
        print("- Error handling and status management")
        print("- Model unloading and cleanup")
    else:
        print("\n⚠️  Some tests failed. Please check the implementation.")
    
    exit(0 if success else 1)