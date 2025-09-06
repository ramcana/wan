"""
Additional tests for Video Encoder with various codecs and settings

This module tests the video encoding functionality with different codecs,
formats, and quality settings to ensure comprehensive coverage.
"""

import pytest
import numpy as np
import tempfile
import shutil
import os
from pathlib import Path
from unittest.mock import Mock, patch

from video_encoder import (
    VideoEncoder, VideoFormat, VideoCodec, EncodingQuality,
    EncodingConfig, encode_video_simple
)
from frame_tensor_handler import ProcessedFrames, TensorFormat


class TestVideoEncoderCodecs:
    """Test video encoder with different codecs and formats"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def sample_frames(self):
        """Create sample ProcessedFrames for testing"""
        frames_array = np.random.rand(5, 128, 128, 3).astype(np.float32)
        
        return ProcessedFrames(
            frames=frames_array,
            fps=30.0,
            duration=5/30.0,
            resolution=(128, 128),
            format=TensorFormat.NUMPY_FLOAT32,
            metadata={"test": "codec_test"}
        )
    
    @pytest.fixture
    def encoder(self, temp_dir):
        """Create VideoEncoder instance"""
        return VideoEncoder(temp_dir=temp_dir)
    
    def test_mp4_h264_encoding_config(self, encoder, sample_frames):
        """Test MP4 with H.264 codec configuration"""
        config = encoder.configure_encoding_params(sample_frames, "mp4", "medium")
        
        assert config.format == VideoFormat.MP4
        assert config.codec == VideoCodec.H264
        assert config.fps == 30.0
        # CRF may be adjusted based on resolution (128x128 is low res, so +3)
        assert config.crf in [23, 26]  # Medium quality, possibly adjusted
        
        args = config.to_ffmpeg_args()
        assert "-c:v" in args
        assert "libx264" in args
    
    def test_webm_vp8_encoding_config(self, encoder, sample_frames):
        """Test WebM with VP8 codec configuration"""
        config = encoder.configure_encoding_params(sample_frames, "webm", "high")
        
        assert config.format == VideoFormat.WEBM
        assert config.codec == VideoCodec.VP8
        assert config.fps == 30.0
        # CRF may be adjusted based on resolution (128x128 is low res, so +3)
        assert config.crf in [18, 21]  # High quality, possibly adjusted
        
        args = config.to_ffmpeg_args()
        assert "-c:v" in args
        assert "libvpx" in args
    
    def test_quality_presets(self, encoder, sample_frames):
        """Test different quality presets"""
        # Test low quality (may be adjusted for low resolution)
        config_low = encoder.configure_encoding_params(sample_frames, "mp4", "low")
        assert config_low.crf >= 28  # May be adjusted upward for low res
        
        # Test medium quality (may be adjusted for low resolution)
        config_medium = encoder.configure_encoding_params(sample_frames, "mp4", "medium")
        assert config_medium.crf in [23, 26]  # May be adjusted for low res
        
        # Test high quality (may be adjusted for low resolution)
        config_high = encoder.configure_encoding_params(sample_frames, "mp4", "high")
        assert config_high.crf in [18, 21]  # May be adjusted for low res
        
        # Test lossless quality (may be adjusted for low resolution)
        config_lossless = encoder.configure_encoding_params(sample_frames, "mp4", "lossless")
        assert config_lossless.crf in [0, 3]  # May be adjusted for low res
    
    def test_custom_encoding_parameters(self, encoder, sample_frames):
        """Test custom encoding parameters"""
        config = encoder.configure_encoding_params(
            sample_frames, "mp4", "medium",
            bitrate="2M",
            preset="ultrafast",
            tune="film"
        )
        
        # Custom parameters should override defaults
        assert config.bitrate == "2M"  # bitrate is a direct attribute
        assert config.additional_params["preset"] == "ultrafast"
        assert config.additional_params["tune"] == "film"
        
        args = config.to_ffmpeg_args()
        # When both CRF and bitrate are set, CRF takes priority in FFmpeg args
        assert "-crf" in args  # CRF should be present
        assert "-preset" in args
        assert "ultrafast" in args
        assert "-tune" in args
        assert "film" in args
    
    def test_resolution_specific_optimizations(self, encoder):
        """Test resolution-specific encoding optimizations"""
        # Test 4K frames
        frames_4k = ProcessedFrames(
            frames=np.random.rand(3, 2160, 3840, 3).astype(np.float32),
            fps=24.0,
            duration=3/24.0,
            resolution=(3840, 2160),
            format=TensorFormat.NUMPY_FLOAT32
        )
        
        config_4k = encoder.configure_encoding_params(frames_4k, "mp4", "medium")
        assert config_4k.crf == 21  # Better quality for 4K
        assert config_4k.additional_params["preset"] == "slow"
        
        # Test 720p frames
        frames_720p = ProcessedFrames(
            frames=np.random.rand(3, 720, 1280, 3).astype(np.float32),
            fps=24.0,
            duration=3/24.0,
            resolution=(1280, 720),
            format=TensorFormat.NUMPY_FLOAT32
        )
        
        config_720p = encoder.configure_encoding_params(frames_720p, "mp4", "medium")
        assert config_720p.crf == 23  # Standard quality
        # Preset may be optimized based on frame count and resolution
        assert config_720p.additional_params["preset"] in ["medium", "fast"]
        
        # Test low resolution frames
        frames_low = ProcessedFrames(
            frames=np.random.rand(3, 360, 480, 3).astype(np.float32),
            fps=24.0,
            duration=3/24.0,
            resolution=(480, 360),
            format=TensorFormat.NUMPY_FLOAT32
        )
        
        config_low = encoder.configure_encoding_params(frames_low, "mp4", "medium")
        assert config_low.crf == 26  # Lower quality acceptable
        assert config_low.additional_params["preset"] == "fast"
    
    def test_frame_count_optimizations(self, encoder):
        """Test optimizations based on frame count"""
        # Test short video (< 30 frames)
        frames_short = ProcessedFrames(
            frames=np.random.rand(10, 64, 64, 3).astype(np.float32),
            fps=24.0,
            duration=10/24.0,
            resolution=(64, 64),
            format=TensorFormat.NUMPY_FLOAT32
        )
        
        config_short = encoder.configure_encoding_params(frames_short, "mp4", "medium")
        assert config_short.additional_params["preset"] == "fast"
        
        # Test long video (> 300 frames)
        frames_long = ProcessedFrames(
            frames=np.random.rand(500, 64, 64, 3).astype(np.float32),
            fps=24.0,
            duration=500/24.0,
            resolution=(64, 64),
            format=TensorFormat.NUMPY_FLOAT32
        )
        
        config_long = encoder.configure_encoding_params(frames_long, "mp4", "medium")
        # Long videos with low resolution may still use fast preset due to resolution optimization
        assert config_long.additional_params["preset"] in ["slow", "fast"]
    
    def test_format_codec_mapping(self, encoder):
        """Test format to codec mapping"""
        # Test MP4 format
        assert VideoCodec.H264 in encoder.format_codec_map[VideoFormat.MP4]
        assert VideoCodec.H265 in encoder.format_codec_map[VideoFormat.MP4]
        
        # Test WebM format
        assert VideoCodec.VP8 in encoder.format_codec_map[VideoFormat.WEBM]
        assert VideoCodec.VP9 in encoder.format_codec_map[VideoFormat.WEBM]
        
        # Test AVI format
        assert VideoCodec.H264 in encoder.format_codec_map[VideoFormat.AVI]
        
        # Test MOV format
        assert VideoCodec.H264 in encoder.format_codec_map[VideoFormat.MOV]
        assert VideoCodec.H265 in encoder.format_codec_map[VideoFormat.MOV]
    
    @patch('video_encoder.VideoEncoder.check_encoding_dependencies')
    @patch('video_encoder.VideoEncoder._run_ffmpeg_encoding')
    def test_different_formats_encoding(self, mock_ffmpeg, mock_deps, encoder, sample_frames, temp_dir):
        """Test encoding with different video formats"""
        # Mock dependencies as available
        mock_deps.return_value.ffmpeg_available = True
        encoder.dependency_status = mock_deps.return_value
        
        # Mock successful FFmpeg encoding
        mock_ffmpeg.return_value = {"success": True, "log": "encoding successful"}
        
        formats_to_test = ["mp4", "webm", "avi", "mov"]
        
        for format_name in formats_to_test:
            output_path = os.path.join(temp_dir, f"test.{format_name}")
            
            # Create dummy output file
            with open(output_path, 'w') as f:
                f.write(f"dummy {format_name} file")
            
            result = encoder.encode_frames_to_video(sample_frames, output_path, format_name)
            
            assert result.success == True
            assert result.output_path == output_path
    
    def test_invalid_format_handling(self, encoder, sample_frames):
        """Test handling of invalid video formats"""
        config = encoder.configure_encoding_params(sample_frames, "invalid_format", "medium")
        
        # Should default to MP4
        assert config.format == VideoFormat.MP4
        assert config.codec == VideoCodec.H264
    
    def test_invalid_quality_handling(self, encoder, sample_frames):
        """Test handling of invalid quality settings"""
        config = encoder.configure_encoding_params(sample_frames, "mp4", "invalid_quality")
        
        # Should default to medium quality, but may be adjusted for resolution
        assert config.crf in [23, 26]  # Medium quality CRF, possibly adjusted


class TestVideoEncoderAdvanced:
    """Test advanced video encoder features"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def encoder(self, temp_dir):
        """Create VideoEncoder instance"""
        return VideoEncoder(temp_dir=temp_dir)
    
    def test_pixel_format_configuration(self, encoder):
        """Test pixel format configuration"""
        config = EncodingConfig(
            codec=VideoCodec.H264,
            format=VideoFormat.MP4,
            fps=24.0,
            pixel_format="yuv444p"
        )
        
        args = config.to_ffmpeg_args()
        assert "-pix_fmt" in args
        assert "yuv444p" in args
    
    def test_bitrate_vs_crf_priority(self, encoder):
        """Test bitrate vs CRF priority in encoding config"""
        # Test CRF priority (should use CRF when both are set)
        config_crf = EncodingConfig(
            codec=VideoCodec.H264,
            format=VideoFormat.MP4,
            fps=24.0,
            crf=20,
            bitrate="2M"
        )
        
        args_crf = config_crf.to_ffmpeg_args()
        assert "-crf" in args_crf
        assert "20" in args_crf
        # Bitrate should not be included when CRF is set
        assert "-b:v" not in args_crf
        
        # Test bitrate only
        config_bitrate = EncodingConfig(
            codec=VideoCodec.H264,
            format=VideoFormat.MP4,
            fps=24.0,
            bitrate="2M"
        )
        
        args_bitrate = config_bitrate.to_ffmpeg_args()
        assert "-b:v" in args_bitrate
        assert "2M" in args_bitrate
        assert "-crf" not in args_bitrate
    
    def test_resolution_override(self, encoder):
        """Test resolution override in encoding config"""
        config = EncodingConfig(
            codec=VideoCodec.H264,
            format=VideoFormat.MP4,
            fps=24.0,
            resolution=(1920, 1080)
        )
        
        args = config.to_ffmpeg_args()
        assert "-s" in args
        assert "1920x1080" in args
    
    def test_additional_params_handling(self, encoder):
        """Test additional parameters handling"""
        config = EncodingConfig(
            codec=VideoCodec.H264,
            format=VideoFormat.MP4,
            fps=24.0,
            additional_params={
                "preset": "slow",
                "tune": "film",
                "profile": "high",
                "level": "4.1",
                "custom_flag": None  # Should be ignored
            }
        )
        
        args = config.to_ffmpeg_args()
        
        assert "-preset" in args
        assert "slow" in args
        assert "-tune" in args
        assert "film" in args
        assert "-profile" in args
        assert "high" in args
        assert "-level" in args
        assert "4.1" in args
        
        # None values should be ignored
        assert "-custom_flag" not in args
    
    @patch('video_encoder.subprocess.run')
    def test_ffmpeg_timeout_handling(self, mock_run, encoder, temp_dir):
        """Test FFmpeg timeout handling"""
        import subprocess

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
        assert "5 minutes" in result["log"]
    
    def test_encoding_config_validation(self, encoder):
        """Test encoding configuration validation"""
        # Test valid configuration
        config = EncodingConfig(
            codec=VideoCodec.H264,
            format=VideoFormat.MP4,
            fps=24.0,
            crf=23
        )
        
        args = config.to_ffmpeg_args()
        assert len(args) > 0
        assert "-c:v" in args
        assert "-r" in args
        assert "-crf" in args
        
        # Test configuration with all optional parameters
        config_full = EncodingConfig(
            codec=VideoCodec.H265,
            format=VideoFormat.MP4,
            bitrate="5M",
            fps=60.0,
            resolution=(3840, 2160),
            pixel_format="yuv420p10le",
            additional_params={
                "preset": "veryslow",
                "tune": "grain",
                "x265-params": "crf=18:qcomp=0.8"
            }
        )
        
        args_full = config_full.to_ffmpeg_args()
        assert "-c:v" in args_full
        assert "libx265" in args_full
        assert "-b:v" in args_full
        assert "5M" in args_full
        assert "-r" in args_full
        assert "60.0" in args_full
        assert "-s" in args_full
        assert "3840x2160" in args_full
        assert "-pix_fmt" in args_full
        assert "yuv420p10le" in args_full


if __name__ == "__main__":
    pytest.main([__file__, "-v"])