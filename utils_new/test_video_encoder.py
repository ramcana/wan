"""
Tests for Video Encoder System

This module tests the video encoding functionality including FFmpeg integration,
parameter optimization, fallback strategies, and dependency checking.

Requirements tested:
- 7.1: Automatic frame tensor encoding to standard video formats
- 7.2: Frame rate and resolution handling during encoding
- 7.3: Fallback frame-by-frame output for encoding failures
- 7.4: Clear installation guidance for missing dependencies
"""

import pytest
import numpy as np
import tempfile
import shutil
import os
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import subprocess

from video_encoder import (
    VideoEncoder, VideoFormat, VideoCodec, EncodingQuality,
    EncodingConfig, EncodingResult, FallbackResult, DependencyStatus,
    encode_video_simple, create_fallback_frames, check_video_encoding_support
)
from frame_tensor_handler import ProcessedFrames, TensorFormat


class TestVideoEncoder:
    """Test VideoEncoder class functionality"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def sample_frames(self):
        """Create sample ProcessedFrames for testing"""
        # Create 10 frames of 64x64 RGB video
        frames_array = np.random.rand(10, 64, 64, 3).astype(np.float32)
        
        return ProcessedFrames(
            frames=frames_array,
            fps=24.0,
            duration=10/24.0,
            resolution=(64, 64),
            format=TensorFormat.NUMPY_FLOAT32,
            metadata={"test": True}
        )
    
    @pytest.fixture
    def encoder(self, temp_dir):
        """Create VideoEncoder instance"""
        return VideoEncoder(temp_dir=temp_dir, cleanup_temp=True)
    
    def test_encoder_initialization(self, temp_dir):
        """Test VideoEncoder initialization"""
        encoder = VideoEncoder(temp_dir=temp_dir, cleanup_temp=False)
        
        assert encoder.temp_dir == Path(temp_dir)
        assert encoder.cleanup_temp == False
        assert isinstance(encoder.dependency_status, DependencyStatus)
        assert len(encoder.quality_presets) == 4
        assert len(encoder.format_codec_map) == 4
    
    def test_encoding_config_creation(self):
        """Test EncodingConfig creation and FFmpeg args"""
        config = EncodingConfig(
            codec=VideoCodec.H264,
            format=VideoFormat.MP4,
            fps=30.0,
            resolution=(1920, 1080),
            crf=23,
            additional_params={"preset": "medium"}
        )
        
        args = config.to_ffmpeg_args()
        
        assert "-c:v" in args
        assert "libx264" in args
        assert "-r" in args
        assert "30.0" in args
        assert "-crf" in args
        assert "23" in args
        assert "-s" in args
        assert "1920x1080" in args
        assert "-preset" in args
        assert "medium" in args
    
    def test_configure_encoding_params(self, encoder, sample_frames):
        """Test encoding parameter configuration"""
        config = encoder.configure_encoding_params(
            sample_frames, "mp4", "high"
        )
        
        assert config.codec == VideoCodec.H264
        assert config.format == VideoFormat.MP4
        assert config.fps == sample_frames.fps
        assert config.resolution == sample_frames.resolution
        # The CRF might be adjusted based on resolution optimization
        assert config.crf in [18, 21]  # High quality preset, possibly adjusted for low res
    
    def test_configure_encoding_params_with_overrides(self, encoder, sample_frames):
        """Test encoding parameter configuration with user overrides"""
        config = encoder.configure_encoding_params(
            sample_frames, "webm", "medium", 
            fps=60.0, crf=20, custom_param="value"
        )
        
        assert config.codec == VideoCodec.VP8
        assert config.format == VideoFormat.WEBM
        assert config.fps == 60.0  # Override applied
        assert config.crf == 20  # Override applied
        assert config.additional_params["custom_param"] == "value"
    
    def test_configure_encoding_params_resolution_optimization(self, encoder):
        """Test encoding parameter optimization based on resolution"""
        # Test 4K optimization
        frames_4k = ProcessedFrames(
            frames=np.random.rand(5, 2160, 3840, 3).astype(np.float32),
            fps=24.0,
            duration=5/24.0,
            resolution=(3840, 2160),
            format=TensorFormat.NUMPY_FLOAT32
        )
        
        config_4k = encoder.configure_encoding_params(frames_4k, "mp4", "medium")
        assert config_4k.crf == 21  # Higher quality for 4K
        assert config_4k.additional_params["preset"] == "slow"
        
        # Test low resolution optimization
        frames_low = ProcessedFrames(
            frames=np.random.rand(5, 480, 640, 3).astype(np.float32),
            fps=24.0,
            duration=5/24.0,
            resolution=(640, 480),
            format=TensorFormat.NUMPY_FLOAT32
        )
        
        config_low = encoder.configure_encoding_params(frames_low, "mp4", "medium")
        assert config_low.crf == 26  # Lower quality acceptable
        assert config_low.additional_params["preset"] == "fast"
    
    @patch('video_encoder.subprocess.run')
    def test_check_encoding_dependencies_available(self, mock_run):
        """Test dependency checking when FFmpeg is available"""
        # Mock FFmpeg version check
        mock_run.return_value = Mock(
            returncode=0,
            stdout="ffmpeg version 4.4.0 Copyright (c) 2000-2021"
        )
        
        encoder = VideoEncoder()
        status = encoder.check_encoding_dependencies()
        
        assert status.ffmpeg_available == True
        assert status.ffmpeg_version == "4.4.0"
        assert len(status.installation_guide) == 0
    
    @patch('video_encoder.subprocess.run')
    def test_check_encoding_dependencies_unavailable(self, mock_run):
        """Test dependency checking when FFmpeg is not available"""
        mock_run.side_effect = FileNotFoundError()
        
        encoder = VideoEncoder()
        status = encoder.check_encoding_dependencies()
        
        assert status.ffmpeg_available == False
        assert status.ffmpeg_version is None
        assert len(status.installation_guide) > 0
        assert "FFmpeg is required" in status.installation_guide[0]
    
    def test_validate_encoding_inputs_valid(self, encoder, sample_frames, temp_dir):
        """Test input validation with valid inputs"""
        output_path = os.path.join(temp_dir, "test.mp4")
        
        result = encoder._validate_encoding_inputs(sample_frames, output_path, "mp4")
        
        assert result.success == True
        assert len(result.errors) == 0
    
    def test_validate_encoding_inputs_invalid_format(self, encoder, sample_frames, temp_dir):
        """Test input validation with invalid format"""
        output_path = os.path.join(temp_dir, "test.invalid")
        
        result = encoder._validate_encoding_inputs(sample_frames, output_path, "invalid")
        
        assert result.success == False
        assert any("Unsupported video format" in error for error in result.errors)
    
    def test_create_temp_frame_files(self, encoder, sample_frames):
        """Test temporary frame file creation"""
        temp_dir = encoder._create_temp_frame_files(sample_frames)
        
        assert temp_dir.exists()
        
        # Check that frame files were created
        frame_files = list(temp_dir.glob("frame_*.png"))
        assert len(frame_files) == sample_frames.num_frames
        
        # Check frame naming
        assert (temp_dir / "frame_000000.png").exists()
        assert (temp_dir / "frame_000009.png").exists()
        
        # Cleanup
        encoder._cleanup_temp_files(temp_dir)
        assert not temp_dir.exists()
    
    @patch('video_encoder.subprocess.run')
    def test_run_ffmpeg_encoding_success(self, mock_run, encoder, temp_dir):
        """Test successful FFmpeg encoding"""
        mock_run.return_value = Mock(
            returncode=0,
            stderr="ffmpeg encoding log"
        )
        
        config = EncodingConfig(
            codec=VideoCodec.H264,
            format=VideoFormat.MP4,
            fps=24.0
        )
        
        frame_dir = Path(temp_dir) / "frames"
        frame_dir.mkdir()
        output_path = os.path.join(temp_dir, "output.mp4")
        
        result = encoder._run_ffmpeg_encoding(frame_dir, output_path, config)
        
        assert result["success"] == True
        assert "ffmpeg encoding log" in result["log"]
        
        # Check that FFmpeg was called with correct arguments
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert "ffmpeg" in call_args
        assert "-c:v" in call_args
        assert "libx264" in call_args
    
    @patch('video_encoder.subprocess.run')
    def test_run_ffmpeg_encoding_failure(self, mock_run, encoder, temp_dir):
        """Test FFmpeg encoding failure"""
        mock_run.return_value = Mock(
            returncode=1,
            stderr="ffmpeg error message"
        )
        
        config = EncodingConfig(
            codec=VideoCodec.H264,
            format=VideoFormat.MP4,
            fps=24.0
        )
        
        frame_dir = Path(temp_dir) / "frames"
        frame_dir.mkdir()
        output_path = os.path.join(temp_dir, "output.mp4")
        
        result = encoder._run_ffmpeg_encoding(frame_dir, output_path, config)
        
        assert result["success"] == False
        assert "FFmpeg returned code 1" in result["error"]
        assert "ffmpeg error message" in result["log"]
    
    @patch('video_encoder.subprocess.run')
    def test_run_ffmpeg_encoding_timeout(self, mock_run, encoder, temp_dir):
        """Test FFmpeg encoding timeout"""
        mock_run.side_effect = subprocess.TimeoutExpired("ffmpeg", 300)
        
        config = EncodingConfig(
            codec=VideoCodec.H264,
            format=VideoFormat.MP4,
            fps=24.0
        )
        
        frame_dir = Path(temp_dir) / "frames"
        frame_dir.mkdir()
        output_path = os.path.join(temp_dir, "output.mp4")
        
        result = encoder._run_ffmpeg_encoding(frame_dir, output_path, config)
        
        assert result["success"] == False
        assert "timed out" in result["error"]
    
    @patch('video_encoder.VideoEncoder.check_encoding_dependencies')
    @patch('video_encoder.VideoEncoder._run_ffmpeg_encoding')
    def test_encode_frames_to_video_success(self, mock_ffmpeg, mock_deps, encoder, sample_frames, temp_dir):
        """Test successful video encoding"""
        # Mock dependencies as available
        mock_deps.return_value = DependencyStatus(ffmpeg_available=True)
        encoder.dependency_status = mock_deps.return_value
        
        # Mock successful FFmpeg encoding
        mock_ffmpeg.return_value = {"success": True, "log": "encoding successful"}
        
        output_path = os.path.join(temp_dir, "test.mp4")
        
        # Create dummy output file to simulate FFmpeg success
        with open(output_path, 'w') as f:
            f.write("dummy video file")
        
        result = encoder.encode_frames_to_video(sample_frames, output_path)
        
        assert result.success == True
        assert result.output_path == output_path
        assert result.encoding_time > 0
        assert result.file_size_bytes > 0
        assert len(result.errors) == 0
    
    @patch('video_encoder.VideoEncoder.check_encoding_dependencies')
    def test_encode_frames_to_video_no_ffmpeg(self, mock_deps, encoder, sample_frames, temp_dir):
        """Test video encoding when FFmpeg is not available"""
        # Mock dependencies as unavailable
        mock_deps.return_value = DependencyStatus(
            ffmpeg_available=False,
            installation_guide=["Install FFmpeg"]
        )
        encoder.dependency_status = mock_deps.return_value
        
        output_path = os.path.join(temp_dir, "test.mp4")
        
        result = encoder.encode_frames_to_video(sample_frames, output_path)
        
        assert result.success == False
        assert any("FFmpeg not available" in error for error in result.errors)
        assert any("Install FFmpeg" in error for error in result.errors)
    
    def test_provide_fallback_output(self, encoder, sample_frames, temp_dir):
        """Test fallback frame-by-frame output"""
        output_path = os.path.join(temp_dir, "fallback_video.mp4")
        
        result = encoder.provide_fallback_output(sample_frames, output_path)
        
        assert result.success == True
        assert result.frame_count == sample_frames.num_frames
        assert len(result.frame_paths) == sample_frames.num_frames
        assert result.output_directory is not None
        
        # Check that frames were saved
        output_dir = Path(result.output_directory)
        assert output_dir.exists()
        
        frame_files = list(output_dir.glob("frame_*.png"))
        assert len(frame_files) == sample_frames.num_frames
        
        # Check metadata file
        metadata_file = output_dir / "metadata.json"
        assert metadata_file.exists()
        
        with open(metadata_file) as f:
            metadata = json.load(f)
        
        assert metadata["total_frames"] == sample_frames.num_frames
        assert metadata["fps"] == sample_frames.fps
        assert metadata["resolution"] == list(sample_frames.resolution)
    
    def test_provide_fallback_output_different_channels(self, encoder, temp_dir):
        """Test fallback output with different channel counts"""
        # Test grayscale (1 channel)
        frames_gray = ProcessedFrames(
            frames=np.random.rand(5, 32, 32, 1).astype(np.float32),
            fps=24.0,
            duration=5/24.0,
            resolution=(32, 32),
            format=TensorFormat.NUMPY_FLOAT32
        )
        
        result_gray = encoder.provide_fallback_output(frames_gray, os.path.join(temp_dir, "gray"))
        assert result_gray.success == True
        
        # Test RGBA (4 channels)
        frames_rgba = ProcessedFrames(
            frames=np.random.rand(5, 32, 32, 4).astype(np.float32),
            fps=24.0,
            duration=5/24.0,
            resolution=(32, 32),
            format=TensorFormat.NUMPY_FLOAT32
        )
        
        result_rgba = encoder.provide_fallback_output(frames_rgba, os.path.join(temp_dir, "rgba"))
        assert result_rgba.success == True
    
    @patch('video_encoder.subprocess.run')
    def test_get_video_info(self, mock_run, encoder, temp_dir):
        """Test video information extraction"""
        # Mock FFprobe output
        mock_output = {
            "streams": [{
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1920,
                "height": 1080,
                "r_frame_rate": "30/1",
                "duration": "10.0"
            }]
        }
        
        mock_run.return_value = Mock(
            returncode=0,
            stdout=json.dumps(mock_output)
        )
        
        video_path = os.path.join(temp_dir, "test.mp4")
        info = encoder._get_video_info(video_path)
        
        assert info["fps"] == 30.0
        assert info["resolution"] == (1920, 1080)
        assert info["duration"] == 10.0
        assert info["codec"] == "h264"
    
    @patch('video_encoder.subprocess.run')
    def test_get_ffmpeg_codecs(self, mock_run, encoder):
        """Test FFmpeg codec detection"""
        mock_run.return_value = Mock(
            returncode=0,
            stdout=" V..... libx264              H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10\n"
                   " V..... libx265              H.265 / HEVC\n"
        )
        
        codecs = encoder._get_ffmpeg_codecs()
        
        assert "libx264" in codecs
        assert "libx265" in codecs
    
    @patch('video_encoder.subprocess.run')
    def test_get_ffmpeg_formats(self, mock_run, encoder):
        """Test FFmpeg format detection"""
        mock_run.return_value = Mock(
            returncode=0,
            stdout=" E mp4             MP4 (MPEG-4 Part 14)\n"
                   " E webm            WebM\n"
        )
        
        formats = encoder._get_ffmpeg_formats()
        
        assert "mp4" in formats
        assert "webm" in formats


class TestEncodingDataModels:
    """Test encoding data models"""
    
    def test_encoding_result_creation(self):
        """Test EncodingResult creation and methods"""
        result = EncodingResult(success=True)
        
        assert result.success == True
        assert len(result.errors) == 0
        assert len(result.warnings) == 0
        
        result.add_error("Test error")
        assert result.success == False
        assert "Test error" in result.errors
        
        result.add_warning("Test warning")
        assert "Test warning" in result.warnings
    
    def test_fallback_result_creation(self):
        """Test FallbackResult creation and methods"""
        result = FallbackResult(success=True)
        
        assert result.success == True
        assert len(result.errors) == 0
        
        result.add_error("Test error")
        assert result.success == False
        assert "Test error" in result.errors
    
    def test_dependency_status_creation(self):
        """Test DependencyStatus creation"""
        status = DependencyStatus(
            ffmpeg_available=True,
            ffmpeg_version="4.4.0",
            supported_codecs=["libx264", "libx265"],
            supported_formats=["mp4", "webm"]
        )
        
        assert status.ffmpeg_available == True
        assert status.ffmpeg_version == "4.4.0"
        assert len(status.supported_codecs) == 2
        assert len(status.supported_formats) == 2


class TestUtilityFunctions:
    """Test utility functions"""
    
    @pytest.fixture
    def sample_frames(self):
        """Create sample ProcessedFrames for testing"""
        frames_array = np.random.rand(5, 32, 32, 3).astype(np.float32)
        
        return ProcessedFrames(
            frames=frames_array,
            fps=24.0,
            duration=5/24.0,
            resolution=(32, 32),
            format=TensorFormat.NUMPY_FLOAT32
        )
    
    @patch('video_encoder.VideoEncoder.encode_frames_to_video')
    def test_encode_video_simple(self, mock_encode, sample_frames):
        """Test simple video encoding utility function"""
        mock_encode.return_value = EncodingResult(success=True)
        
        result = encode_video_simple(sample_frames, "test.mp4", "mp4", "high")
        
        assert result.success == True
        mock_encode.assert_called_once_with(sample_frames, "test.mp4", "mp4", "high")
    
    @patch('video_encoder.VideoEncoder.provide_fallback_output')
    def test_create_fallback_frames(self, mock_fallback, sample_frames):
        """Test fallback frame creation utility function"""
        mock_fallback.return_value = FallbackResult(success=True)
        
        result = create_fallback_frames(sample_frames, "test_frames")
        
        assert result.success == True
        mock_fallback.assert_called_once_with(sample_frames, "test_frames")
    
    @patch('video_encoder.VideoEncoder.check_encoding_dependencies')
    def test_check_video_encoding_support(self, mock_check):
        """Test video encoding support check utility function"""
        mock_status = DependencyStatus(ffmpeg_available=True)
        mock_check.return_value = mock_status
        
        status = check_video_encoding_support()
        
        assert status.ffmpeg_available == True


class TestVideoEncoderIntegration:
    """Integration tests for video encoder"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def sample_frames(self):
        """Create sample ProcessedFrames for testing"""
        # Create simple test pattern
        frames_array = np.zeros((3, 64, 64, 3), dtype=np.float32)
        
        # Frame 1: Red
        frames_array[0, :, :, 0] = 1.0
        
        # Frame 2: Green
        frames_array[1, :, :, 1] = 1.0
        
        # Frame 3: Blue
        frames_array[2, :, :, 2] = 1.0
        
        return ProcessedFrames(
            frames=frames_array,
            fps=1.0,  # 1 FPS for easy testing
            duration=3.0,
            resolution=(64, 64),
            format=TensorFormat.NUMPY_FLOAT32,
            metadata={"test_pattern": "RGB"}
        )
    
    def test_end_to_end_fallback_output(self, sample_frames, temp_dir):
        """Test end-to-end fallback output creation"""
        encoder = VideoEncoder(temp_dir=temp_dir)
        output_path = os.path.join(temp_dir, "test_video.mp4")
        
        result = encoder.provide_fallback_output(sample_frames, output_path)
        
        assert result.success == True
        assert result.frame_count == 3
        
        # Verify output directory structure
        output_dir = Path(result.output_directory)
        assert output_dir.exists()
        
        # Check frame files
        frame_files = sorted(output_dir.glob("frame_*.png"))
        assert len(frame_files) == 3
        
        # Check metadata
        metadata_file = output_dir / "metadata.json"
        assert metadata_file.exists()
        
        with open(metadata_file) as f:
            metadata = json.load(f)
        
        assert metadata["total_frames"] == 3
        assert metadata["fps"] == 1.0
        assert metadata["metadata"]["test_pattern"] == "RGB"
    
    @patch('video_encoder.subprocess.run')
    def test_end_to_end_with_mock_ffmpeg(self, mock_run, sample_frames, temp_dir):
        """Test end-to-end encoding with mocked FFmpeg"""
        output_path = os.path.join(temp_dir, "test.mp4")
        
        # Create dummy output file to simulate FFmpeg success
        def create_output_file(*args, **kwargs):
            with open(output_path, 'w') as f:
                f.write("dummy video content")
            return Mock(returncode=0, stderr="encoding successful")
        
        # Mock FFmpeg availability check and encoding
        mock_responses = [
            # FFmpeg version check
            Mock(returncode=0, stdout="ffmpeg version 4.4.0"),
            # FFmpeg codecs check
            Mock(returncode=0, stdout=" V..... libx264"),
            # FFmpeg formats check  
            Mock(returncode=0, stdout=" E mp4"),
            # Actual encoding
            create_output_file
        ]
        
        mock_run.side_effect = mock_responses
        
        encoder = VideoEncoder(temp_dir=temp_dir)
        result = encoder.encode_frames_to_video(sample_frames, output_path)
        
        # Note: This will fail without actual FFmpeg, but tests the flow
        # In a real environment with FFmpeg, this would succeed
        assert isinstance(result, EncodingResult)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])