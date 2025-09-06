"""
Unit tests for LoRAManager UI integration methods
Tests the new methods added for UI integration: upload_lora_file, delete_lora_file, 
rename_lora_file, get_ui_display_data, and estimate_memory_impact
"""

import unittest
import tempfile
import shutil
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import torch

# Import the LoRAManager and related classes
from utils import LoRAManager, handle_error_with_recovery


class TestLoRAManagerUIIntegration(unittest.TestCase):
    """Test cases for LoRAManager UI integration methods"""
    
    def setUp(self):
        """Set up test environment before each test"""
        # Create temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.loras_dir = Path(self.temp_dir) / "loras"
        self.loras_dir.mkdir(exist_ok=True)
        
        # Create test config
        self.test_config = {
            "directories": {
                "loras_directory": str(self.loras_dir),
                "models_directory": str(Path(self.temp_dir) / "models"),
                "outputs_directory": str(Path(self.temp_dir) / "outputs")
            },
            "lora_max_file_size_mb": 100,
            "optimization": {
                "max_vram_usage_gb": 8
            }
        }
        
        # Initialize LoRAManager with test config
        self.lora_manager = LoRAManager(self.test_config)
        
        # Create test LoRA files
        self.create_test_lora_files()
    
    def tearDown(self):
        """Clean up after each test"""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_lora_files(self):
        """Create mock LoRA files for testing"""
        # Create a mock safetensors file
        test_lora_1 = self.loras_dir / "test_anime_style.safetensors"
        test_lora_1.write_bytes(b"mock_safetensors_data" * 1000)  # ~21KB
        
        # Create a mock .pt file
        test_lora_2 = self.loras_dir / "test_realistic.pt"
        test_lora_2.write_bytes(b"mock_pytorch_data" * 2000)  # ~34KB
        
        # Create a mock .ckpt file
        test_lora_3 = self.loras_dir / "test_cartoon.ckpt"
        test_lora_3.write_bytes(b"mock_checkpoint_data" * 1500)  # ~30KB
    
    def create_temp_upload_file(self, filename: str, size_kb: int = 50) -> str:
        """Create a temporary file for upload testing"""
        temp_file = Path(self.temp_dir) / filename
        temp_file.write_bytes(b"mock_lora_data" * (size_kb * 1024 // 14))  # Approximate size
        return str(temp_file)

    def test_upload_lora_file_success(self):
        """Test successful LoRA file upload"""
        # Create a temporary file to upload
        upload_file = self.create_temp_upload_file("new_style.safetensors", 30)
        
        # Mock the validation methods
        with patch.object(self.lora_manager, '_validate_lora_weights', return_value=True):
            with patch('safetensors.torch.load_file', return_value={"mock": "weights"}):
                result = self.lora_manager.upload_lora_file(upload_file, "new_style.safetensors")
        
        # Verify successful upload
        self.assertTrue(result["success"])
        self.assertEqual(result["filename"], "new_style.safetensors")
        self.assertIn("Successfully uploaded", result["message"])
        
        # Verify file was copied to loras directory
        uploaded_file = self.loras_dir / "new_style.safetensors"
        self.assertTrue(uploaded_file.exists())

        assert True  # TODO: Add proper assertion

    def test_upload_lora_file_invalid_extension(self):
        """Test upload with invalid file extension"""
        upload_file = self.create_temp_upload_file("invalid.txt", 30)
        
        result = self.lora_manager.upload_lora_file(upload_file, "invalid.txt")
        
        # Verify upload failed
        self.assertFalse(result["success"])
        self.assertIn("Invalid file format", result["error"])

        assert True  # TODO: Add proper assertion

    def test_upload_lora_file_too_large(self):
        """Test upload with file too large"""
        # Create a large file (larger than max_file_size_mb)
        upload_file = self.create_temp_upload_file("large_file.safetensors", 150 * 1024)  # 150MB
        
        result = self.lora_manager.upload_lora_file(upload_file, "large_file.safetensors")
        
        # Verify upload failed due to size
        self.assertFalse(result["success"])
        self.assertIn("File too large", result["error"])

        assert True  # TODO: Add proper assertion

    def test_upload_lora_file_duplicate_name(self):
        """Test upload with duplicate filename"""
        # Upload first file
        upload_file1 = self.create_temp_upload_file("duplicate.safetensors", 30)
        
        with patch.object(self.lora_manager, '_validate_lora_weights', return_value=True):
            with patch('safetensors.torch.load_file', return_value={"mock": "weights"}):
                result1 = self.lora_manager.upload_lora_file(upload_file1, "duplicate.safetensors")
        
        self.assertTrue(result1["success"])
        
        # Upload second file with same name
        upload_file2 = self.create_temp_upload_file("duplicate2.safetensors", 30)
        
        with patch.object(self.lora_manager, '_validate_lora_weights', return_value=True):
            with patch('safetensors.torch.load_file', return_value={"mock": "weights"}):
                result2 = self.lora_manager.upload_lora_file(upload_file2, "duplicate.safetensors")
        
        # Verify second file was renamed
        self.assertTrue(result2["success"])
        self.assertEqual(result2["filename"], "duplicate_1.safetensors")

        assert True  # TODO: Add proper assertion

    def test_upload_lora_file_invalid_structure(self):
        """Test upload with invalid LoRA structure"""
        upload_file = self.create_temp_upload_file("invalid_structure.safetensors", 30)
        
        # Mock validation to return False (invalid structure)
        with patch.object(self.lora_manager, '_validate_lora_weights', return_value=False):
            with patch('safetensors.torch.load_file', return_value={"invalid": "structure"}):
                result = self.lora_manager.upload_lora_file(upload_file, "invalid_structure.safetensors")
        
        # Verify upload failed
        self.assertFalse(result["success"])
        self.assertIn("Invalid LoRA file structure", result["error"])
        
        # Verify file was not kept in loras directory
        uploaded_file = self.loras_dir / "invalid_structure.safetensors"
        self.assertFalse(uploaded_file.exists())

        assert True  # TODO: Add proper assertion

    def test_delete_lora_file_success(self):
        """Test successful LoRA file deletion"""
        # Verify file exists before deletion
        test_file = self.loras_dir / "test_anime_style.safetensors"
        self.assertTrue(test_file.exists())
        
        # Delete the LoRA
        result = self.lora_manager.delete_lora_file("test_anime_style")
        
        # Verify deletion was successful
        self.assertTrue(result)
        self.assertFalse(test_file.exists())

        assert True  # TODO: Add proper assertion

    def test_delete_lora_file_not_found(self):
        """Test deletion of non-existent LoRA file"""
        result = self.lora_manager.delete_lora_file("non_existent_lora")
        
        # Verify deletion failed gracefully
        self.assertFalse(result)

        assert True  # TODO: Add proper assertion

    def test_delete_lora_file_with_loaded_lora(self):
        """Test deletion of LoRA that is currently loaded"""
        # Mock a loaded LoRA
        self.lora_manager.loaded_loras["test_anime_style"] = {
            "name": "test_anime_style",
            "weights": {"mock": "weights"}
        }
        self.lora_manager.applied_loras["test_anime_style"] = 0.8
        
        # Delete the LoRA
        result = self.lora_manager.delete_lora_file("test_anime_style")
        
        # Verify deletion was successful and tracking was cleaned up
        self.assertTrue(result)
        self.assertNotIn("test_anime_style", self.lora_manager.loaded_loras)
        self.assertNotIn("test_anime_style", self.lora_manager.applied_loras)

        assert True  # TODO: Add proper assertion

    def test_rename_lora_file_success(self):
        """Test successful LoRA file renaming"""
        # Verify original file exists
        original_file = self.loras_dir / "test_anime_style.safetensors"
        self.assertTrue(original_file.exists())
        
        # Rename the LoRA
        result = self.lora_manager.rename_lora_file("test_anime_style", "anime_style_v2")
        
        # Verify rename was successful
        self.assertTrue(result)
        self.assertFalse(original_file.exists())
        
        new_file = self.loras_dir / "anime_style_v2.safetensors"
        self.assertTrue(new_file.exists())

        assert True  # TODO: Add proper assertion

    def test_rename_lora_file_not_found(self):
        """Test renaming of non-existent LoRA file"""
        result = self.lora_manager.rename_lora_file("non_existent", "new_name")
        
        # Verify rename failed gracefully
        self.assertFalse(result)

        assert True  # TODO: Add proper assertion

    def test_rename_lora_file_duplicate_name(self):
        """Test renaming to an existing name"""
        # Try to rename to an existing LoRA name
        result = self.lora_manager.rename_lora_file("test_anime_style", "test_realistic")
        
        # Verify rename failed due to duplicate name
        self.assertFalse(result)

        assert True  # TODO: Add proper assertion

    def test_rename_lora_file_invalid_name(self):
        """Test renaming with invalid characters"""
        result = self.lora_manager.rename_lora_file("test_anime_style", "")
        
        # Verify rename failed due to empty name
        self.assertFalse(result)

        assert True  # TODO: Add proper assertion

    def test_rename_lora_file_with_loaded_lora(self):
        """Test renaming of LoRA that is currently loaded"""
        # Mock a loaded LoRA
        self.lora_manager.loaded_loras["test_anime_style"] = {
            "name": "test_anime_style",
            "path": str(self.loras_dir / "test_anime_style.safetensors"),
            "weights": {"mock": "weights"}
        }
        self.lora_manager.applied_loras["test_anime_style"] = 0.8
        
        # Rename the LoRA
        result = self.lora_manager.rename_lora_file("test_anime_style", "anime_v2")
        
        # Verify rename was successful and tracking was updated
        self.assertTrue(result)
        self.assertNotIn("test_anime_style", self.lora_manager.loaded_loras)
        self.assertIn("anime_v2", self.lora_manager.loaded_loras)
        self.assertEqual(self.lora_manager.loaded_loras["anime_v2"]["name"], "anime_v2")
        self.assertNotIn("test_anime_style", self.lora_manager.applied_loras)
        self.assertIn("anime_v2", self.lora_manager.applied_loras)

        assert True  # TODO: Add proper assertion

    def test_get_ui_display_data(self):
        """Test UI display data formatting"""
        # Mock some loaded and applied LoRAs
        self.lora_manager.loaded_loras["test_anime_style"] = {"mock": "data"}
        self.lora_manager.applied_loras["test_realistic"] = 0.7
        
        result = self.lora_manager.get_ui_display_data()
        
        # Verify structure of returned data
        self.assertIn("loras", result)
        self.assertIn("summary", result)
        self.assertIn("recent_loras", result)
        self.assertIn("supported_formats", result)
        self.assertIn("max_file_size_mb", result)
        self.assertIn("max_concurrent_loras", result)
        
        # Verify summary data
        summary = result["summary"]
        self.assertEqual(summary["total_count"], 3)  # 3 test files created
        self.assertEqual(summary["loaded_count"], 1)  # 1 loaded LoRA
        self.assertEqual(summary["applied_count"], 1)  # 1 applied LoRA
        self.assertGreater(summary["total_size_mb"], 0)
        
        # Verify supported formats
        expected_formats = ['.safetensors', '.ckpt', '.pt', '.pth', '.bin']
        self.assertEqual(result["supported_formats"], expected_formats)

        assert True  # TODO: Add proper assertion

    def test_get_ui_display_data_error_handling(self):
        """Test UI display data with error conditions"""
        # Mock an error in list_available_loras
        with patch.object(self.lora_manager, 'list_available_loras', side_effect=Exception("Mock error")):
            result = self.lora_manager.get_ui_display_data()
        
        # Verify error handling
        self.assertIn("error", result)
        self.assertEqual(result["loras"], {})
        self.assertEqual(result["summary"]["total_count"], 0)

        assert True  # TODO: Add proper assertion

    def test_estimate_memory_impact_empty_list(self):
        """Test memory estimation with empty LoRA list"""
        result = self.lora_manager.estimate_memory_impact([])
        
        # Verify empty result
        self.assertEqual(result["total_memory_mb"], 0.0)
        self.assertEqual(result["individual_memory_mb"], {})
        self.assertEqual(result["estimated_load_time_seconds"], 0.0)
        self.assertEqual(result["vram_impact_mb"], 0.0)
        self.assertEqual(result["recommendations"], [])

        assert True  # TODO: Add proper assertion

    def test_estimate_memory_impact_single_lora(self):
        """Test memory estimation with single LoRA"""
        result = self.lora_manager.estimate_memory_impact(["test_anime_style"])
        
        # Verify calculation
        self.assertGreater(result["total_memory_mb"], 0)
        self.assertIn("test_anime_style", result["individual_memory_mb"])
        self.assertGreater(result["estimated_load_time_seconds"], 0)
        self.assertGreater(result["vram_impact_mb"], 0)

        assert True  # TODO: Add proper assertion

    def test_estimate_memory_impact_multiple_loras(self):
        """Test memory estimation with multiple LoRAs"""
        lora_names = ["test_anime_style", "test_realistic", "test_cartoon"]
        result = self.lora_manager.estimate_memory_impact(lora_names)
        
        # Verify calculations
        self.assertGreater(result["total_memory_mb"], 0)
        self.assertEqual(len(result["individual_memory_mb"]), 3)
        
        # Verify total is sum of individual
        individual_total = sum(result["individual_memory_mb"].values())
        self.assertAlmostEqual(result["total_memory_mb"], individual_total, places=1)

        assert True  # TODO: Add proper assertion

    def test_estimate_memory_impact_high_usage_recommendations(self):
        """Test memory estimation recommendations for high usage"""
        # Create larger mock files to trigger recommendations
        large_lora = self.loras_dir / "large_lora.safetensors"
        large_lora.write_bytes(b"large_mock_data" * 100000)  # ~1.4MB
        
        # Test with many LoRAs
        many_loras = ["test_anime_style", "test_realistic", "test_cartoon", "large_lora"]
        result = self.lora_manager.estimate_memory_impact(many_loras)
        
        # Verify recommendations are generated
        self.assertGreater(len(result["recommendations"]), 0)
        
        # Check for specific recommendation types
        recommendations_text = " ".join(result["recommendations"])
        if len(many_loras) > 3:
            self.assertIn("many LoRAs", recommendations_text.lower())

        assert True  # TODO: Add proper assertion

    def test_estimate_memory_impact_nonexistent_lora(self):
        """Test memory estimation with non-existent LoRA"""
        result = self.lora_manager.estimate_memory_impact(["non_existent_lora"])
        
        # Verify graceful handling
        self.assertEqual(result["total_memory_mb"], 0.0)
        self.assertEqual(result["individual_memory_mb"], {})

        assert True  # TODO: Add proper assertion

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.get_device_properties')
    @patch('torch.cuda.memory_allocated')
    def test_estimate_memory_impact_with_vram_check(self, mock_memory_allocated, mock_device_props, mock_cuda_available):
        """Test memory estimation with VRAM availability check"""
        # Mock CUDA device properties
        mock_device = Mock()
        mock_device.total_memory = 8 * 1024 * 1024 * 1024  # 8GB
        mock_device_props.return_value = mock_device
        mock_memory_allocated.return_value = 6 * 1024 * 1024 * 1024  # 6GB used
        
        result = self.lora_manager.estimate_memory_impact(["test_anime_style"])
        
        # Verify VRAM info was considered
        self.assertGreater(result["vram_impact_mb"], 0)

        assert True  # TODO: Add proper assertion

    def test_estimate_memory_impact_error_handling(self):
        """Test memory estimation error handling"""
        # Mock an error in list_available_loras
        with patch.object(self.lora_manager, 'list_available_loras', side_effect=Exception("Mock error")):
            result = self.lora_manager.estimate_memory_impact(["test_lora"])
        
        # Verify error handling
        self.assertIn("error", result)
        self.assertEqual(result["total_memory_mb"], 0.0)
        self.assertIn("Error estimating memory impact", result["recommendations"][0])


        assert True  # TODO: Add proper assertion

class TestLoRAManagerUIIntegrationEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions for LoRAManager UI integration"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.loras_dir = Path(self.temp_dir) / "loras"
        self.loras_dir.mkdir(exist_ok=True)
        
        self.test_config = {
            "directories": {
                "loras_directory": str(self.loras_dir)
            },
            "lora_max_file_size_mb": 100
        }
        
        self.lora_manager = LoRAManager(self.test_config)
    
    def tearDown(self):
        """Clean up after tests"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_upload_with_permission_error(self):
        """Test upload when file permissions prevent copying"""
        upload_file = Path(self.temp_dir) / "test.safetensors"
        upload_file.write_bytes(b"test_data")
        
        # Mock shutil.copy2 to raise permission error
        with patch('shutil.copy2', side_effect=PermissionError("Permission denied")):
            result = self.lora_manager.upload_lora_file(str(upload_file), "test.safetensors")
        
        self.assertFalse(result["success"])
        self.assertIn("Permission denied", result["error"])

        assert True  # TODO: Add proper assertion

    def test_delete_with_permission_error(self):
        """Test delete when file permissions prevent deletion"""
        # Create test file
        test_file = self.loras_dir / "test.safetensors"
        test_file.write_bytes(b"test_data")
        
        # Mock unlink to raise permission error
        with patch.object(Path, 'unlink', side_effect=PermissionError("Permission denied")):
            result = self.lora_manager.delete_lora_file("test")
        
        self.assertFalse(result)

        assert True  # TODO: Add proper assertion

    def test_rename_with_permission_error(self):
        """Test rename when file permissions prevent renaming"""
        # Create test file
        test_file = self.loras_dir / "test.safetensors"
        test_file.write_bytes(b"test_data")
        
        # Mock rename to raise permission error
        with patch.object(Path, 'rename', side_effect=PermissionError("Permission denied")):
            result = self.lora_manager.rename_lora_file("test", "new_name")
        
        self.assertFalse(result)

        assert True  # TODO: Add proper assertion

    def test_special_characters_in_rename(self):
        """Test renaming with special characters"""
        # Create test file
        test_file = self.loras_dir / "test.safetensors"
        test_file.write_bytes(b"test_data")
        
        # Try to rename with special characters
        result = self.lora_manager.rename_lora_file("test", "test<>:\"/\\|?*name")
        
        # Verify special characters were sanitized
        self.assertTrue(result)
        sanitized_file = self.loras_dir / "test___________name.safetensors"
        self.assertTrue(sanitized_file.exists())


        assert True  # TODO: Add proper assertion

if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)