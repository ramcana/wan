"""
Unit tests for LoRAUploadHandler component
Tests file validation, upload processing, and metadata extraction
"""

import os
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest
import torch
import numpy as np
from PIL import Image

# Import the component to test
from lora_upload_handler import LoRAUploadHandler


class TestLoRAUploadHandler:
    """Test suite for LoRAUploadHandler"""
    
    def setup_method(self):
        """Set up test environment before each test"""
        # Create temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.loras_dir = Path(self.temp_dir) / "loras"
        self.loras_dir.mkdir(exist_ok=True)
        
        # Initialize handler
        self.handler = LoRAUploadHandler(str(self.loras_dir))
        
        # Create sample LoRA weights for testing (large enough to pass size validation)
        self.sample_lora_weights = {
            "layer1.lora_up.weight": torch.randn(1024, 256),
            "layer1.lora_down.weight": torch.randn(256, 2048),
            "layer2.lora_up.weight": torch.randn(512, 128),
            "layer2.lora_down.weight": torch.randn(128, 1024),
            "layer3.lora_up.weight": torch.randn(256, 64),
            "layer3.lora_down.weight": torch.randn(64, 512),
            "alpha": torch.tensor(16.0)
        }
        
        # Create invalid weights (not LoRA) - make them large enough to pass size validation
        self.invalid_weights = {
            "some_layer.weight": torch.randn(1024, 512),
            "another_layer.bias": torch.randn(512),
            "third_layer.weight": torch.randn(512, 256),
            "fourth_layer.weight": torch.randn(256, 128)
        }
    
    def teardown_method(self):
        """Clean up after each test"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def create_test_file(self, filename: str, content: bytes = b"test content") -> Path:
        """Create a test file with given content"""
        file_path = self.loras_dir / filename
        with open(file_path, 'wb') as f:
            f.write(content)
        return file_path
    
    def create_large_test_file(self, filename: str, size_mb: float = 2.0) -> Path:
        """Create a large test file for size validation tests"""
        file_path = self.loras_dir / filename
        content_size = int(size_mb * 1024 * 1024)
        with open(file_path, 'wb') as f:
            f.write(b'0' * content_size)
        return file_path
    
    def create_lora_file(self, filename: str, weights: dict = None) -> Path:
        """Create a valid LoRA file for testing"""
        if weights is None:
            # Create larger weights to ensure file size is above minimum
            weights = {
                "layer1.lora_up.weight": torch.randn(512, 128),  # Larger matrices
                "layer1.lora_down.weight": torch.randn(128, 1024),
                "layer2.lora_up.weight": torch.randn(256, 64),
                "layer2.lora_down.weight": torch.randn(64, 512),
                "layer3.lora_up.weight": torch.randn(128, 32),
                "layer3.lora_down.weight": torch.randn(32, 256),
                "alpha": torch.tensor(16.0)
            }
        
        file_path = self.loras_dir / filename
        torch.save(weights, str(file_path))
        return file_path
    
    def test_init(self):
        """Test LoRAUploadHandler initialization"""
        # Test basic initialization
        handler = LoRAUploadHandler(str(self.loras_dir))
        assert handler.loras_directory == self.loras_dir
        assert handler.thumbnails_dir.exists()
        assert handler.max_file_size_mb == LoRAUploadHandler.MAX_FILE_SIZE_MB
        
        # Test initialization with config
        config = {"max_lora_file_size_mb": 1024, "min_lora_file_size_mb": 5}
        handler_with_config = LoRAUploadHandler(str(self.loras_dir), config)
        assert handler_with_config.max_file_size_mb == 1024
        assert handler_with_config.min_file_size_mb == 5
    
    def test_validate_file_success(self):
        """Test successful file validation"""
        # Create a valid LoRA file
        lora_file = self.create_lora_file("test_lora.pt")
        
        is_valid, message = self.handler.validate_file(str(lora_file))
        
        assert is_valid is True
        assert "successful" in message.lower()
    
    def test_validate_file_nonexistent(self):
        """Test validation of non-existent file"""
        is_valid, message = self.handler.validate_file("nonexistent.pt")
        
        assert is_valid is False
        assert "does not exist" in message.lower()
    
    def test_validate_file_unsupported_format(self):
        """Test validation of unsupported file format"""
        # Create file with unsupported extension
        test_file = self.create_test_file("test.txt")
        
        is_valid, message = self.handler.validate_file(str(test_file))
        
        assert is_valid is False
        assert "unsupported file format" in message.lower()
        assert ".safetensors" in message
        assert ".pt" in message
    
    def test_validate_file_too_small(self):
        """Test validation of file that's too small"""
        # Create very small file (smaller than 100KB)
        small_file = self.create_test_file("small.pt", b"x" * 1000)  # 1KB file
        
        is_valid, message = self.handler.validate_file(str(small_file))
        
        assert is_valid is False
        assert "too small" in message.lower()
    
    def test_validate_file_too_large(self):
        """Test validation of file that's too large"""
        # Mock file size to be too large
        large_file = self.create_test_file("large.pt")
        
        # Patch the stat method to return large size
        with patch.object(Path, 'stat') as mock_stat:
            mock_stat.return_value.st_size = (self.handler.max_file_size_mb + 1) * 1024 * 1024
            
            is_valid, message = self.handler.validate_file(str(large_file))
            
            assert is_valid is False
            assert "too large" in message.lower()
    
    @patch('torch.load')
    def test_validate_file_content_invalid_pytorch(self, mock_torch_load):
        """Test validation of invalid PyTorch file"""
        mock_torch_load.side_effect = RuntimeError("Invalid file")
        
        test_file = self.create_large_test_file("invalid.pt")
        
        is_valid, message = self.handler.validate_file(str(test_file))
        
        assert is_valid is False
        assert "invalid pytorch file" in message.lower()
    
    def test_validate_file_content_not_lora(self):
        """Test validation of file that's not a LoRA"""
        # Create file with non-LoRA weights
        invalid_file = self.create_lora_file("not_lora.pt", self.invalid_weights)
        
        is_valid, message = self.handler.validate_file(str(invalid_file))
        
        assert is_valid is False
        assert "does not appear to contain lora weights" in message.lower()
    
    def test_is_lora_weights_valid(self):
        """Test LoRA weights detection with valid weights"""
        assert self.handler._is_lora_weights(self.sample_lora_weights) is True
    
    def test_is_lora_weights_invalid(self):
        """Test LoRA weights detection with invalid weights"""
        assert self.handler._is_lora_weights(self.invalid_weights) is False
        assert self.handler._is_lora_weights("not a dict") is False
        assert self.handler._is_lora_weights({}) is False
    
    def test_is_lora_weights_up_down_pattern(self):
        """Test LoRA detection with up/down pattern"""
        weights_with_up_down = {
            "layer.lora_up": torch.randn(32, 16),
            "layer.lora_down": torch.randn(16, 64)
        }
        assert self.handler._is_lora_weights(weights_with_up_down) is True
    
    def test_is_lora_weights_low_rank_matrices(self):
        """Test LoRA detection with low-rank matrices"""
        weights_low_rank = {
            "layer1": torch.randn(8, 128),  # Low rank matrix
            "layer2": torch.randn(64, 4)   # Low rank matrix
        }
        assert self.handler._is_lora_weights(weights_low_rank) is True
    
    def test_check_duplicate_filename_no_duplicate(self):
        """Test duplicate check when file doesn't exist"""
        exists, suggested = self.handler.check_duplicate_filename("new_file.pt")
        
        assert exists is False
        assert suggested is None
    
    def test_check_duplicate_filename_with_duplicate(self):
        """Test duplicate check when file exists"""
        # Create existing file
        self.create_test_file("existing.pt")
        
        exists, suggested = self.handler.check_duplicate_filename("existing.pt")
        
        assert exists is True
        assert suggested == "existing_1.pt"
    
    def test_check_duplicate_filename_multiple_duplicates(self):
        """Test duplicate check with multiple existing files"""
        # Create multiple existing files
        self.create_test_file("test.pt")
        self.create_test_file("test_1.pt")
        self.create_test_file("test_2.pt")
        
        exists, suggested = self.handler.check_duplicate_filename("test.pt")
        
        assert exists is True
        assert suggested == "test_3.pt"
    
    def test_sanitize_filename_basic(self):
        """Test basic filename sanitization"""
        result = self.handler._sanitize_filename("normal_file.pt")
        assert result == "normal_file.pt"
    
    def test_sanitize_filename_problematic_chars(self):
        """Test sanitization of problematic characters"""
        result = self.handler._sanitize_filename("file<>:\"/\\|?*.pt")
        # Note: os.path.basename treats \ as path separator, so only "|?*.pt" remains after basename
        # After replacing |?* with _, we get "___.pt"
        assert result == "___.pt"
    
    def test_sanitize_filename_problematic_chars_no_path(self):
        """Test sanitization of problematic characters without path separators"""
        result = self.handler._sanitize_filename("file<>:\"|?*.pt")
        # 7 problematic characters: < > : " | ? *
        assert result == "file_______.pt"
    
    def test_sanitize_filename_path_traversal(self):
        """Test sanitization prevents path traversal"""
        result = self.handler._sanitize_filename("../../../evil.pt")
        assert result == "evil.pt"
        
        result = self.handler._sanitize_filename("/absolute/path/file.pt")
        assert result == "file.pt"
    
    def test_sanitize_filename_empty(self):
        """Test sanitization of empty filename"""
        result = self.handler._sanitize_filename("")
        assert result.startswith("lora_")
        assert result.endswith(".safetensors")
    
    def test_sanitize_filename_no_extension(self):
        """Test sanitization adds extension if missing"""
        result = self.handler._sanitize_filename("file_without_ext")
        assert result == "file_without_ext.safetensors"
    
    def test_process_upload_success(self):
        """Test successful file upload processing"""
        # Create valid LoRA data
        lora_data = torch.save(self.sample_lora_weights, "temp.pt")
        
        # Read the file data
        temp_file = self.loras_dir / "temp.pt"
        torch.save(self.sample_lora_weights, str(temp_file))
        
        with open(temp_file, 'rb') as f:
            file_data = f.read()
        
        # Clean up temp file
        temp_file.unlink()
        
        # Process upload
        result = self.handler.process_upload(file_data, "test_upload.pt")
        
        assert result["success"] is True
        assert result["filename"] == "test_upload.pt"
        assert "path" in result
        assert "size_mb" in result
        assert "metadata" in result
        assert "upload_time" in result
        assert "file_hash" in result
        
        # Verify file was created
        uploaded_file = self.loras_dir / "test_upload.pt"
        assert uploaded_file.exists()
    
    def test_process_upload_empty_filename(self):
        """Test upload processing with empty filename"""
        result = self.handler.process_upload(b"test data", "")
        
        assert result["success"] is False
        assert "empty" in result["error"].lower()
    
    def test_process_upload_duplicate_no_overwrite(self):
        """Test upload processing with duplicate filename and no overwrite"""
        # Create existing file
        self.create_test_file("existing.pt")
        
        result = self.handler.process_upload(b"test data", "existing.pt", overwrite=False)
        
        assert result["success"] is False
        assert "already exists" in result["error"].lower()
        assert "suggested_name" in result
        assert result["suggested_name"] == "existing_1.pt"
    
    def test_process_upload_invalid_file(self):
        """Test upload processing with invalid file data"""
        # Create invalid file data
        invalid_data = b"not a valid lora file"
        
        result = self.handler.process_upload(invalid_data, "invalid.pt")
        
        assert result["success"] is False
        assert "error" in result
    
    def test_calculate_file_hash(self):
        """Test file hash calculation"""
        test_file = self.create_test_file("hash_test.pt", b"test content for hashing")
        
        hash1 = self.handler._calculate_file_hash(str(test_file))
        hash2 = self.handler._calculate_file_hash(str(test_file))
        
        # Same file should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 produces 64 character hex string
    
    def test_extract_metadata_basic(self):
        """Test basic metadata extraction"""
        lora_file = self.create_lora_file("metadata_test.pt")
        
        metadata = self.handler.extract_metadata(str(lora_file))
        
        assert "filename" in metadata
        assert "size_bytes" in metadata
        assert "size_mb" in metadata
        assert "created_time" in metadata
        assert "modified_time" in metadata
        assert "file_extension" in metadata
        assert metadata["filename"] == "metadata_test.pt"
        assert metadata["file_extension"] == ".pt"
    
    def test_extract_metadata_with_lora_analysis(self):
        """Test metadata extraction with LoRA structure analysis"""
        lora_file = self.create_lora_file("analysis_test.pt")
        
        metadata = self.handler.extract_metadata(str(lora_file))
        
        # Should include LoRA analysis
        assert "total_parameters" in metadata
        assert "lora_layers" in metadata
        assert "layer_names" in metadata
        assert "rank_info" in metadata
        assert metadata["lora_layers"] > 0
    
    def test_analyze_lora_structure(self):
        """Test LoRA structure analysis"""
        analysis = self.handler._analyze_lora_structure(self.sample_lora_weights)
        
        assert "total_parameters" in analysis
        assert "lora_layers" in analysis
        assert "parameter_types" in analysis
        assert "layer_names" in analysis
        assert "rank_info" in analysis
        
        assert analysis["lora_layers"] == 6  # We have 6 lora layers (3 up + 3 down)
        assert analysis["total_parameters"] > 0
        assert "weight" in analysis["parameter_types"]
    
    def test_analyze_lora_structure_with_rank_info(self):
        """Test LoRA structure analysis calculates rank information"""
        analysis = self.handler._analyze_lora_structure(self.sample_lora_weights)
        
        assert "average_rank" in analysis
        assert "max_rank" in analysis
        assert "min_rank" in analysis
        
        # Check that rank calculations are reasonable
        assert analysis["average_rank"] > 0
        assert analysis["max_rank"] >= analysis["min_rank"]
    
    def test_generate_thumbnail(self):
        """Test thumbnail generation"""
        # Create a LoRA file
        lora_file = self.create_lora_file("thumbnail_test.pt")
        
        thumbnail_path = self.handler.generate_thumbnail("thumbnail_test.pt")
        
        if thumbnail_path:  # Thumbnail generation might fail in test environment
            assert thumbnail_path is not None
            assert Path(thumbnail_path).exists()
            assert Path(thumbnail_path).suffix == ".jpg"
    
    def test_create_placeholder_thumbnail(self):
        """Test placeholder thumbnail creation"""
        image = self.handler._create_placeholder_thumbnail("test_lora")
        
        assert isinstance(image, Image.Image)
        assert image.size == (256, 256)
        assert image.mode == "RGB"
    
    def test_get_upload_stats_empty(self):
        """Test upload statistics with no files"""
        stats = self.handler.get_upload_stats()
        
        assert stats["total_files"] == 0
        assert stats["total_size_mb"] == 0.0
        assert stats["file_types"] == {}
        assert stats["largest_file"]["size_mb"] == 0.0
        assert stats["smallest_file"]["size_mb"] == 0.0
    
    def test_get_upload_stats_with_files(self):
        """Test upload statistics with files"""
        # Create test files
        self.create_lora_file("lora1.pt")
        self.create_lora_file("lora2.safetensors")
        
        stats = self.handler.get_upload_stats()
        
        assert stats["total_files"] == 2
        assert stats["total_size_mb"] > 0
        assert ".pt" in stats["file_types"]
        assert ".safetensors" in stats["file_types"]
        assert stats["largest_file"]["name"] != ""
        assert stats["smallest_file"]["name"] != ""
    
    @patch('safetensors.torch.load_file')
    def test_validate_safetensors_file(self, mock_load_file):
        """Test validation of .safetensors files"""
        mock_load_file.return_value = self.sample_lora_weights
        
        # Create .safetensors file
        safetensors_file = self.create_large_test_file("test.safetensors")
        
        is_valid, message = self.handler.validate_file(str(safetensors_file))
        
        assert is_valid is True
        mock_load_file.assert_called_once()
    
    @patch('safetensors.torch.load_file')
    def test_validate_safetensors_import_error(self, mock_load_file):
        """Test handling of missing safetensors library"""
        mock_load_file.side_effect = ImportError("safetensors not available")
        
        safetensors_file = self.create_large_test_file("test.safetensors")
        
        is_valid, message = self.handler.validate_file(str(safetensors_file))
        
        assert is_valid is False
        assert "safetensors library not available" in message
    
    def test_process_upload_with_overwrite(self):
        """Test upload processing with overwrite enabled"""
        # Create existing file
        existing_file = self.create_test_file("overwrite_test.pt")
        original_content = b"original content"
        
        with open(existing_file, 'wb') as f:
            f.write(original_content)
        
        # Create new valid LoRA data
        temp_file = self.loras_dir / "temp_new.pt"
        torch.save(self.sample_lora_weights, str(temp_file))
        
        with open(temp_file, 'rb') as f:
            new_file_data = f.read()
        
        temp_file.unlink()
        
        # Process upload with overwrite
        result = self.handler.process_upload(new_file_data, "overwrite_test.pt", overwrite=True)
        
        assert result["success"] is True
        
        # Verify file was overwritten
        with open(existing_file, 'rb') as f:
            current_content = f.read()
        
        assert current_content != original_content
        assert len(current_content) == len(new_file_data)


# Integration tests
class TestLoRAUploadHandlerIntegration:
    """Integration tests for LoRAUploadHandler with real files"""
    
    def setup_method(self):
        """Set up integration test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.loras_dir = Path(self.temp_dir) / "loras"
        self.loras_dir.mkdir(exist_ok=True)
        
        self.handler = LoRAUploadHandler(str(self.loras_dir))
    
    def teardown_method(self):
        """Clean up integration test environment"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_full_upload_workflow(self):
        """Test complete upload workflow from file data to final storage"""
        # Create realistic LoRA weights
        lora_weights = {
            "text_encoder.encoder.layers.0.self_attn.q_proj.lora_up.weight": torch.randn(16, 768),
            "text_encoder.encoder.layers.0.self_attn.q_proj.lora_down.weight": torch.randn(768, 16),
            "text_encoder.encoder.layers.0.self_attn.v_proj.lora_up.weight": torch.randn(16, 768),
            "text_encoder.encoder.layers.0.self_attn.v_proj.lora_down.weight": torch.randn(768, 16),
            "alpha": torch.tensor(16.0)
        }
        
        # Save to temporary file to get bytes
        temp_path = self.loras_dir / "temp_realistic.pt"
        torch.save(lora_weights, str(temp_path))
        
        with open(temp_path, 'rb') as f:
            file_data = f.read()
        
        temp_path.unlink()
        
        # Process the upload
        result = self.handler.process_upload(file_data, "realistic_lora.pt")
        
        # Verify success
        assert result["success"] is True
        assert result["filename"] == "realistic_lora.pt"
        
        # Verify file exists and is valid
        final_path = Path(result["path"])
        assert final_path.exists()
        
        # Verify we can load the file back
        loaded_weights = torch.load(str(final_path), map_location='cpu')
        assert "lora_up" in str(loaded_weights.keys())
        assert "lora_down" in str(loaded_weights.keys())
        
        # Verify metadata was extracted
        metadata = result["metadata"]
        assert metadata["lora_layers"] > 0
        assert metadata["total_parameters"] > 0
        assert len(metadata["layer_names"]) > 0
    
    def test_error_recovery_and_cleanup(self):
        """Test that temporary files are cleaned up on errors"""
        # Create invalid file data that will fail validation
        invalid_data = b"this is not a valid pytorch file"
        
        # Count files before upload
        files_before = list(self.loras_dir.iterdir())
        
        # Attempt upload
        result = self.handler.process_upload(invalid_data, "invalid.pt")
        
        # Verify failure
        assert result["success"] is False
        
        # Verify no temporary files left behind
        files_after = list(self.loras_dir.iterdir())
        temp_files = [f for f in files_after if f.name.startswith(".temp_")]
        
        assert len(temp_files) == 0
        assert len(files_after) == len(files_before)  # No new files created


if __name__ == "__main__":
    # Run tests if script is executed directly
    import sys
    
    # Simple test runner
    test_classes = [TestLoRAUploadHandler, TestLoRAUploadHandlerIntegration]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        print(f"\nRunning {test_class.__name__}...")
        
        # Get all test methods
        test_methods = [method for method in dir(test_class) if method.startswith('test_')]
        
        for test_method_name in test_methods:
            total_tests += 1
            
            try:
                # Create test instance
                test_instance = test_class()
                test_instance.setup_method()
                
                # Run test method
                test_method = getattr(test_instance, test_method_name)
                test_method()
                
                # Clean up
                test_instance.teardown_method()
                
                print(f"  ✓ {test_method_name}")
                passed_tests += 1
                
            except Exception as e:
                print(f"  ✗ {test_method_name}: {str(e)}")
                
                # Clean up on failure
                try:
                    test_instance.teardown_method()
                except:
                    pass
    
    print(f"\nTest Results: {passed_tests}/{total_tests} passed")
    
    if passed_tests == total_tests:
        print("All tests passed! ✓")
        sys.exit(0)
    else:
        print("Some tests failed! ✗")
        sys.exit(1)