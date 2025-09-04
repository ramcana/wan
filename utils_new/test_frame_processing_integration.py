"""
Integration tests for frame processing pipeline

This module tests the complete frame processing workflow with different
tensor formats and demonstrates the functionality working end-to-end.
"""

import pytest
import numpy as np
import torch
from frame_tensor_handler import FrameTensorHandler, ProcessedFrames, TensorFormat


class TestFrameProcessingIntegration:
    """Integration tests for complete frame processing workflow"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.handler = FrameTensorHandler(default_fps=24.0)
    
    def test_complete_workflow_torch_tensor(self):
        """Test complete workflow with torch tensor input"""
        # Simulate Wan model output: 10 frames, 256x256, RGB, in [-1, 1] range
        frames = torch.rand(10, 256, 256, 3, dtype=torch.float32) * 2.0 - 1.0
        
        # Process the frames
        result = self.handler.process_output_tensors(frames, fps=30.0, resolution=(256, 256))
        
        # Verify processing results
        assert isinstance(result, ProcessedFrames)
        assert result.num_frames == 10
        assert result.height == 256
        assert result.width == 256
        assert result.channels == 3
        assert result.fps == 30.0
        assert result.resolution == (256, 256)
        assert result.format == TensorFormat.TORCH_FLOAT32
        
        # Verify normalization to [0, 1] range
        assert result.frames.min() >= 0.0
        assert result.frames.max() <= 1.0
        assert result.frames.dtype == np.float32
        
        # Verify metadata
        assert "original_shape" in result.metadata
        assert "processing_timestamp" in result.metadata
        assert result.metadata["normalization_applied"] is True
        
        # Verify validation passes
        validation = result.validate_shape()
        assert validation.is_valid
    
    def test_complete_workflow_numpy_uint8(self):
        """Test complete workflow with numpy uint8 input"""
        # Simulate standard video frames: 5 frames, 128x128, RGB, uint8 [0, 255]
        frames = np.random.randint(0, 256, (5, 128, 128, 3), dtype=np.uint8)
        
        # Process the frames
        result = self.handler.process_output_tensors(frames, fps=25.0)
        
        # Verify processing results
        assert isinstance(result, ProcessedFrames)
        assert result.num_frames == 5
        assert result.height == 128
        assert result.width == 128
        assert result.channels == 3
        assert result.fps == 25.0
        assert result.format == TensorFormat.NUMPY_UINT8
        
        # Verify normalization from uint8 to float32 [0, 1]
        assert result.frames.min() >= 0.0
        assert result.frames.max() <= 1.0
        assert result.frames.dtype == np.float32
    
    def test_complete_workflow_dict_output(self):
        """Test complete workflow with dictionary output format"""
        # Simulate model output as dictionary
        frames = np.random.rand(8, 64, 64, 3).astype(np.float32)
        model_output = {
            "frames": frames,
            "metadata": {"model": "wan_t2v", "version": "2.2"}
        }
        
        # Process the output
        result = self.handler.process_output_tensors(model_output, fps=24.0)
        
        # Verify processing
        assert isinstance(result, ProcessedFrames)
        assert result.num_frames == 8
        assert result.fps == 24.0
        # Verify frames are in valid range and have correct shape
        assert result.frames.shape == frames.shape
        assert result.frames.min() >= 0.0
        assert result.frames.max() <= 1.0
    
    def test_complete_workflow_nchw_conversion(self):
        """Test complete workflow with NCHW tensor conversion"""
        # Create NCHW format tensor (common in PyTorch models)
        frames_nchw = torch.rand(6, 3, 128, 128, dtype=torch.float32)
        
        # Process the frames
        result = self.handler.process_output_tensors(frames_nchw, fps=30.0)
        
        # Verify conversion to NHWC
        assert result.frames.shape == (6, 128, 128, 3)
        assert result.num_frames == 6
        assert result.height == 128
        assert result.width == 128
        assert result.channels == 3
    
    def test_complete_workflow_single_frame(self):
        """Test complete workflow with single frame input"""
        # Single frame input
        frame = np.random.rand(512, 512, 3).astype(np.float32)
        
        # Process the frame
        result = self.handler.process_output_tensors(frame, fps=1.0)
        
        # Verify single frame handling
        assert result.num_frames == 1
        assert result.height == 512
        assert result.width == 512
        assert result.channels == 3
        assert result.duration == 1.0  # 1 frame at 1 fps = 1 second
    
    def test_batch_processing_workflow(self):
        """Test complete batch processing workflow"""
        # Create batch of different sized videos
        batch_outputs = [
            np.random.rand(5, 64, 64, 3).astype(np.float32),   # 5 frames
            np.random.rand(10, 64, 64, 3).astype(np.float32),  # 10 frames
            np.random.rand(3, 64, 64, 3).astype(np.float32)    # 3 frames
        ]
        
        # Process the batch
        results = self.handler.handle_batch_outputs(batch_outputs)
        
        # Verify batch processing
        assert len(results) == 3
        assert all(isinstance(r, ProcessedFrames) for r in results)
        
        # Verify individual results
        assert results[0].num_frames == 5
        assert results[1].num_frames == 10
        assert results[2].num_frames == 3
        
        # Verify batch indices
        for i, result in enumerate(results):
            assert result.metadata["batch_index"] == i
        
        # Verify all have consistent properties
        for result in results:
            assert result.height == 64
            assert result.width == 64
            assert result.channels == 3
            assert result.fps == 24.0  # Default fps
    
    def test_validation_workflow(self):
        """Test validation workflow with various frame configurations"""
        test_cases = [
            # (shape, should_be_valid, description)
            ((10, 256, 256, 3), True, "Standard RGB video"),
            ((5, 128, 128, 1), True, "Grayscale video"),
            ((1, 64, 64, 4), True, "Single RGBA frame"),
            ((100, 32, 32, 3), True, "Long low-res video"),
        ]
        
        for shape, should_be_valid, description in test_cases:
            frames = np.random.rand(*shape).astype(np.float32)
            
            # Process frames
            result = self.handler.process_output_tensors(frames)
            
            # Validate
            validation = self.handler.validate_frame_dimensions(result.frames)
            
            if should_be_valid:
                assert validation.is_valid, f"Failed for {description}: {validation.errors}"
            else:
                assert not validation.is_valid, f"Should have failed for {description}"
    
    def test_error_recovery_workflow(self):
        """Test error handling and recovery in processing workflow"""
        # Test various error conditions
        error_cases = [
            ("string_input", "invalid string input"),
            ({}, "empty dictionary"),
            ({"invalid_key": "value"}, "dict without tensor"),
        ]
        
        for invalid_input, description in error_cases:
            with pytest.raises(RuntimeError, match="Tensor processing failed"):
                self.handler.process_output_tensors(invalid_input)

        assert True  # TODO: Add proper assertion
    
    def test_memory_efficiency_workflow(self):
        """Test memory efficiency with large tensors"""
        # Create a moderately large tensor to test memory handling
        large_frames = np.random.rand(20, 512, 512, 3).astype(np.float32)
        
        # Process the large tensor
        result = self.handler.process_output_tensors(large_frames, fps=24.0)
        
        # Verify processing succeeded
        assert isinstance(result, ProcessedFrames)
        assert result.num_frames == 20
        assert result.height == 512
        assert result.width == 512
        
        # Verify memory usage is reasonable (frames should be same size as input)
        expected_size = large_frames.nbytes
        actual_size = result.frames.nbytes
        assert actual_size == expected_size, "Memory usage should be consistent"
    
    def test_mixed_precision_workflow(self):
        """Test workflow with mixed precision inputs"""
        # Test different precision types
        precision_cases = [
            (torch.float32, TensorFormat.TORCH_FLOAT32),
            (torch.float16, TensorFormat.TORCH_FLOAT16),
            (np.float32, TensorFormat.NUMPY_FLOAT32),
            (np.uint8, TensorFormat.NUMPY_UINT8),
        ]
        
        for dtype, expected_format in precision_cases:
            if isinstance(dtype, torch.dtype):
                frames = torch.rand(5, 64, 64, 3, dtype=dtype)
            else:
                if dtype == np.uint8:
                    frames = np.random.randint(0, 256, (5, 64, 64, 3), dtype=dtype)
                else:
                    frames = np.random.rand(5, 64, 64, 3).astype(dtype)
            
            # Process frames
            result = self.handler.process_output_tensors(frames)
            
            # Verify format detection
            assert result.format == expected_format
            
            # Verify output is always float32 in [0, 1] range
            assert result.frames.dtype == np.float32
            assert result.frames.min() >= 0.0
            assert result.frames.max() <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])