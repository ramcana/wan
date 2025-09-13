"""
Tests for Frame Tensor Handler

This module tests the frame tensor processing functionality including:
- Tensor format detection and conversion
- Frame validation and normalization
- Batch processing capabilities
- Different tensor formats and dimensions
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch
from frame_tensor_handler import (
    FrameTensorHandler, 
    ProcessedFrames, 
    TensorFormat, 
    ValidationResult
)


class TestProcessedFrames:
    """Test ProcessedFrames data model"""
    
    def test_processed_frames_creation(self):
        """Test basic ProcessedFrames creation"""
        frames = np.random.rand(10, 256, 256, 3).astype(np.float32)
        processed = ProcessedFrames(
            frames=frames,
            fps=24.0,
            duration=10/24.0,
            resolution=(256, 256),
            format=TensorFormat.NUMPY_FLOAT32
        )
        
        assert processed.num_frames == 10
        assert processed.height == 256
        assert processed.width == 256
        assert processed.channels == 3
        assert processed.fps == 24.0
        assert processed.resolution == (256, 256)
    
    def test_processed_frames_properties(self):
        """Test ProcessedFrames property calculations"""
        frames = np.random.rand(5, 128, 64, 3).astype(np.float32)
        processed = ProcessedFrames(
            frames=frames,
            fps=30.0,
            duration=5/30.0,
            resolution=(64, 128),
            format=TensorFormat.NUMPY_FLOAT32
        )
        
        assert processed.num_frames == 5
        assert processed.height == 128
        assert processed.width == 64
        assert processed.channels == 3
    
    def test_processed_frames_validation_valid(self):
        """Test validation of valid ProcessedFrames"""
        frames = np.random.rand(8, 512, 512, 3).astype(np.float32)
        processed = ProcessedFrames(
            frames=frames,
            fps=24.0,
            duration=8/24.0,
            resolution=(512, 512),
            format=TensorFormat.NUMPY_FLOAT32
        )
        
        result = processed.validate_shape()
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_processed_frames_validation_invalid_shape(self):
        """Test validation with invalid shape"""
        frames = np.random.rand(8, 512, 512).astype(np.float32)  # Missing channel dim
        processed = ProcessedFrames(
            frames=frames,
            fps=24.0,
            duration=8/24.0,
            resolution=(512, 512),
            format=TensorFormat.NUMPY_FLOAT32
        )
        
        result = processed.validate_shape()
        assert not result.is_valid
        assert len(result.errors) > 0
    
    def test_processed_frames_validation_warnings(self):
        """Test validation warnings for unusual configurations"""
        frames = np.random.rand(8, 512, 512, 5).astype(np.float32)  # 5 channels
        processed = ProcessedFrames(
            frames=frames,
            fps=24.0,
            duration=8/24.0,
            resolution=(512, 512),
            format=TensorFormat.NUMPY_FLOAT32
        )
        
        result = processed.validate_shape()
        assert result.is_valid  # Still valid but with warnings
        assert len(result.warnings) > 0


class TestFrameTensorHandler:
    """Test FrameTensorHandler functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.handler = FrameTensorHandler(default_fps=24.0)
    
    def test_handler_initialization(self):
        """Test handler initialization"""
        handler = FrameTensorHandler(default_fps=30.0, target_dtype=np.float16)
        assert handler.default_fps == 30.0
        assert handler.target_dtype == np.float16
    
    def test_process_numpy_tensor(self):
        """Test processing numpy tensor"""
        # Create test tensor (10 frames, 128x128, RGB)
        frames = np.random.rand(10, 128, 128, 3).astype(np.float32)
        
        result = self.handler.process_output_tensors(frames, fps=30.0)
        
        assert isinstance(result, ProcessedFrames)
        assert result.num_frames == 10
        assert result.height == 128
        assert result.width == 128
        assert result.channels == 3
        assert result.fps == 30.0
        assert result.format == TensorFormat.NUMPY_FLOAT32
    
    def test_process_torch_tensor(self):
        """Test processing torch tensor"""
        # Create test tensor (5 frames, 256x256, RGB)
        frames = torch.rand(5, 256, 256, 3, dtype=torch.float32)
        
        result = self.handler.process_output_tensors(frames, fps=24.0)
        
        assert isinstance(result, ProcessedFrames)
        assert result.num_frames == 5
        assert result.height == 256
        assert result.width == 256
        assert result.channels == 3
        assert result.fps == 24.0
        assert result.format == TensorFormat.TORCH_FLOAT32
    
    def test_process_dict_output(self):
        """Test processing dictionary output"""
        frames = np.random.rand(8, 64, 64, 3).astype(np.float32)
        output_dict = {"frames": frames}
        
        result = self.handler.process_output_tensors(output_dict, fps=25.0)
        
        assert isinstance(result, ProcessedFrames)
        assert result.num_frames == 8
        assert result.fps == 25.0
    
    def test_process_single_frame(self):
        """Test processing single frame"""
        frame = np.random.rand(128, 128, 3).astype(np.float32)
        
        result = self.handler.process_output_tensors(frame)
        
        assert isinstance(result, ProcessedFrames)
        assert result.num_frames == 1
        assert result.height == 128
        assert result.width == 128
        assert result.channels == 3
    
    def test_tensor_format_detection(self):
        """Test tensor format detection"""
        # Test torch tensors
        torch_f32 = torch.rand(5, 64, 64, 3, dtype=torch.float32)
        torch_f16 = torch.rand(5, 64, 64, 3, dtype=torch.float16)
        
        format_f32 = self.handler._detect_tensor_format(torch_f32)
        format_f16 = self.handler._detect_tensor_format(torch_f16)
        
        assert format_f32 == TensorFormat.TORCH_FLOAT32
        assert format_f16 == TensorFormat.TORCH_FLOAT16
        
        # Test numpy arrays
        numpy_f32 = np.random.rand(5, 64, 64, 3).astype(np.float32)
        numpy_u8 = np.random.randint(0, 256, (5, 64, 64, 3), dtype=np.uint8)
        
        format_nf32 = self.handler._detect_tensor_format(numpy_f32)
        format_nu8 = self.handler._detect_tensor_format(numpy_u8)
        
        assert format_nf32 == TensorFormat.NUMPY_FLOAT32
        assert format_nu8 == TensorFormat.NUMPY_UINT8
    
    def test_dimension_normalization_nchw_to_nhwc(self):
        """Test NCHW to NHWC conversion"""
        # Create NCHW tensor (N=5, C=3, H=64, W=64)
        frames_nchw = np.random.rand(5, 3, 64, 64).astype(np.float32)
        
        normalized = self.handler._normalize_tensor_dimensions(frames_nchw)
        
        # Should be converted to NHWC (5, 64, 64, 3)
        assert normalized.shape == (5, 64, 64, 3)
    
    def test_dimension_normalization_single_frame(self):
        """Test single frame dimension expansion"""
        # Create single frame (H=128, W=128, C=3)
        frame = np.random.rand(128, 128, 3).astype(np.float32)
        
        normalized = self.handler._normalize_tensor_dimensions(frame)
        
        # Should be expanded to (1, 128, 128, 3)
        assert normalized.shape == (1, 128, 128, 3)
    
    def test_dimension_normalization_5d_tensor(self):
        """Test 5D tensor handling (batch dimension)"""
        # Create 5D tensor (B=1, T=10, C=3, H=64, W=64)
        frames_5d = np.random.rand(1, 10, 3, 64, 64).astype(np.float32)
        
        normalized = self.handler._normalize_tensor_dimensions(frames_5d)
        
        # Should be converted to (10, 64, 64, 3)
        assert normalized.shape == (10, 64, 64, 3)
    
    def test_data_range_normalization_diffusion_range(self):
        """Test normalization from diffusion model range [-1, 1]"""
        # Create frames in [-1, 1] range
        frames = np.random.uniform(-1.0, 1.0, (5, 64, 64, 3)).astype(np.float32)
        
        normalized = self.handler.normalize_frame_data(frames, TensorFormat.TORCH_FLOAT32)
        
        # Should be normalized to [0, 1]
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0
        assert normalized.dtype == np.float32
    
    def test_data_range_normalization_uint8(self):
        """Test normalization from uint8 [0, 255]"""
        # Create uint8 frames
        frames = np.random.randint(0, 256, (5, 64, 64, 3), dtype=np.uint8)
        
        normalized = self.handler.normalize_frame_data(frames, TensorFormat.NUMPY_UINT8)
        
        # Should be normalized to [0, 1] as float32
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0
        assert normalized.dtype == np.float32
    
    def test_frame_validation_valid_frames(self):
        """Test validation of valid frame arrays"""
        frames = np.random.rand(10, 256, 256, 3).astype(np.float32)
        
        result = self.handler.validate_frame_dimensions(frames)
        
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_frame_validation_invalid_dimensions(self):
        """Test validation of invalid dimensions"""
        frames = np.random.rand(256, 256).astype(np.float32)  # 2D array
        
        result = self.handler.validate_frame_dimensions(frames)
        
        assert not result.is_valid
        assert len(result.errors) > 0
    
    def test_frame_validation_warnings(self):
        """Test validation warnings"""
        # Very high resolution
        frames = np.random.rand(5, 4096, 4096, 3).astype(np.float32)
        
        result = self.handler.validate_frame_dimensions(frames)
        
        assert result.is_valid
        assert len(result.warnings) > 0
    
    def test_batch_processing_list_input(self):
        """Test batch processing with list input"""
        batch_outputs = [
            np.random.rand(5, 64, 64, 3).astype(np.float32),
            np.random.rand(8, 64, 64, 3).astype(np.float32),
            np.random.rand(3, 64, 64, 3).astype(np.float32)
        ]
        
        results = self.handler.handle_batch_outputs(batch_outputs)
        
        assert len(results) == 3
        assert all(isinstance(r, ProcessedFrames) for r in results)
        assert results[0].num_frames == 5
        assert results[1].num_frames == 8
        assert results[2].num_frames == 3
        
        # Check batch indices
        for i, result in enumerate(results):
            assert result.metadata["batch_index"] == i
    
    def test_batch_processing_dict_input(self):
        """Test batch processing with dictionary input"""
        batch_tensor = np.random.rand(3, 10, 128, 128, 3).astype(np.float32)
        batch_output = {"frames": batch_tensor}
        
        results = self.handler.handle_batch_outputs(batch_output)
        
        assert len(results) == 3
        assert all(isinstance(r, ProcessedFrames) for r in results)
        assert all(r.num_frames == 10 for r in results)
        
        # Check batch indices
        for i, result in enumerate(results):
            assert result.metadata["batch_index"] == i
    
    def test_batch_processing_tensor_input(self):
        """Test batch processing with tensor input"""
        batch_tensor = torch.rand(2, 6, 64, 64, 3, dtype=torch.float32)
        
        results = self.handler.handle_batch_outputs(batch_tensor)
        
        assert len(results) == 2
        assert all(isinstance(r, ProcessedFrames) for r in results)
        assert all(r.num_frames == 6 for r in results)
    
    def test_batch_processing_single_output(self):
        """Test batch processing with single output"""
        single_output = np.random.rand(5, 128, 128, 3).astype(np.float32)
        
        results = self.handler.handle_batch_outputs(single_output)
        
        assert len(results) == 1
        assert isinstance(results[0], ProcessedFrames)
        assert results[0].num_frames == 5
        assert results[0].metadata["batch_index"] == 0
    
    def test_extract_tensor_from_dict(self):
        """Test tensor extraction from dictionary"""
        frames = np.random.rand(5, 64, 64, 3).astype(np.float32)
        
        # Test different key names
        test_cases = [
            {"frames": frames},
            {"videos": frames},
            {"images": frames},
            {"samples": frames}
        ]
        
        for output_dict in test_cases:
            tensor = self.handler._extract_tensor_from_output(output_dict)
            assert np.array_equal(tensor, frames)
    
    def test_extract_tensor_from_object(self):
        """Test tensor extraction from object with attributes"""
        frames = np.random.rand(5, 64, 64, 3).astype(np.float32)
        
        # Mock object with frames attribute
        mock_output = Mock()
        mock_output.frames = frames
        
        tensor = self.handler._extract_tensor_from_output(mock_output)
        assert np.array_equal(tensor, frames)
    
    def test_error_handling_invalid_output(self):
        """Test error handling for invalid output types"""
        with pytest.raises(RuntimeError, match="Tensor processing failed"):
            self.handler.process_output_tensors("invalid_output")

        assert True  # TODO: Add proper assertion
    
    def test_error_handling_empty_dict(self):
        """Test error handling for empty dictionary"""
        with pytest.raises(RuntimeError, match="Tensor processing failed"):
            self.handler.process_output_tensors({})

        assert True  # TODO: Add proper assertion
    
    def test_error_handling_invalid_tensor_dimensions(self):
        """Test error handling for invalid tensor dimensions"""
        invalid_tensor = np.random.rand(10, 10).astype(np.float32)  # 2D
        
        with pytest.raises(RuntimeError, match="Tensor processing failed"):
            self.handler.process_output_tensors(invalid_tensor)

        assert True  # TODO: Add proper assertion
    
    def test_metadata_generation(self):
        """Test metadata generation in processed frames"""
        frames = np.random.rand(5, 128, 128, 3).astype(np.float32)
        
        result = self.handler.process_output_tensors(frames, fps=30.0)
        
        assert "original_shape" in result.metadata
        assert "original_dtype" in result.metadata
        assert "processing_timestamp" in result.metadata
        assert "normalization_applied" in result.metadata
        assert result.metadata["normalization_applied"] is True
    
    def test_resolution_inference(self):
        """Test automatic resolution inference"""
        frames = np.random.rand(5, 480, 640, 3).astype(np.float32)
        
        result = self.handler.process_output_tensors(frames)
        
        assert result.resolution == (640, 480)  # (width, height)
    
    def test_duration_calculation(self):
        """Test duration calculation"""
        frames = np.random.rand(60, 128, 128, 3).astype(np.float32)
        fps = 30.0
        
        result = self.handler.process_output_tensors(frames, fps=fps)
        
        expected_duration = 60 / 30.0  # 2.0 seconds
        assert abs(result.duration - expected_duration) < 0.001
    
    def test_different_channel_counts(self):
        """Test processing different channel counts"""
        test_cases = [
            (1, "grayscale"),
            (3, "RGB"),
            (4, "RGBA")
        ]
        
        for channels, description in test_cases:
            frames = np.random.rand(5, 64, 64, channels).astype(np.float32)
            
            result = self.handler.process_output_tensors(frames)
            
            assert result.channels == channels, f"Failed for {description}"
            assert result.frames.shape[-1] == channels
    
    def test_torch_to_numpy_conversion(self):
        """Test torch tensor to numpy conversion"""
        torch_frames = torch.rand(5, 64, 64, 3, dtype=torch.float32)
        
        numpy_frames = self.handler._convert_to_numpy(torch_frames)
        
        assert isinstance(numpy_frames, np.ndarray)
        assert numpy_frames.shape == torch_frames.shape
        assert numpy_frames.dtype == np.float32
    
    def test_mixed_precision_handling(self):
        """Test handling of different precision types"""
        # Test float16 input
        frames_f16 = torch.rand(5, 64, 64, 3, dtype=torch.float16)
        
        result = self.handler.process_output_tensors(frames_f16)
        
        assert result.format == TensorFormat.TORCH_FLOAT16
        assert result.frames.dtype == np.float32  # Should be converted to target dtype


class TestValidationResult:
    """Test ValidationResult utility class"""
    
    def test_validation_result_creation(self):
        """Test ValidationResult creation"""
        result = ValidationResult(True)
        assert result.is_valid
        assert len(result.errors) == 0
        assert len(result.warnings) == 0
    
    def test_validation_result_with_errors(self):
        """Test ValidationResult with initial errors"""
        errors = ["Error 1", "Error 2"]
        result = ValidationResult(False, errors=errors)
        
        assert not result.is_valid
        assert result.errors == errors
    
    def test_add_error(self):
        """Test adding errors to ValidationResult"""
        result = ValidationResult(True)
        result.add_error("Test error")
        
        assert not result.is_valid
        assert "Test error" in result.errors
    
    def test_add_warning(self):
        """Test adding warnings to ValidationResult"""
        result = ValidationResult(True)
        result.add_warning("Test warning")
        
        assert result.is_valid  # Warnings don't affect validity
        assert "Test warning" in result.warnings


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
