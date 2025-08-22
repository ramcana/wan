"""
Video Encoder for Wan Model Video Processing Pipeline

This module handles encoding of processed frame arrays into standard video formats
(MP4, WebM) with FFmpeg integration, parameter optimization, and fallback strategies.

Requirements addressed:
- 7.1: Automatic frame tensor encoding to standard video formats
- 7.2: Frame rate and resolution handling during encoding
- 7.3: Fallback frame-by-frame output for encoding failures
- 7.4: Clear installation guidance for missing dependencies
"""

import subprocess
import shutil
import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
import numpy as np
import logging
from PIL import Image
import tempfile

from frame_tensor_handler import ProcessedFrames

logger = logging.getLogger(__name__)


class VideoFormat(Enum):
    """Supported video output formats"""
    MP4 = "mp4"
    WEBM = "webm"
    AVI = "avi"
    MOV = "mov"


class VideoCodec(Enum):
    """Supported video codecs"""
    H264 = "libx264"
    H265 = "libx265"
    VP8 = "libvpx"
    VP9 = "libvpx-vp9"
    AV1 = "libaom-av1"


class EncodingQuality(Enum):
    """Encoding quality presets"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    LOSSLESS = "lossless"


@dataclass
class EncodingConfig:
    """Configuration for video encoding parameters"""
    codec: VideoCodec
    format: VideoFormat
    bitrate: Optional[str] = None
    crf: Optional[int] = None  # Constant Rate Factor for quality-based encoding
    fps: float = 24.0
    resolution: Optional[Tuple[int, int]] = None
    pixel_format: str = "yuv420p"
    additional_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_ffmpeg_args(self) -> List[str]:
        """Convert config to FFmpeg command line arguments"""
        args = []
        
        # Video codec
        args.extend(["-c:v", self.codec.value])
        
        # Frame rate
        args.extend(["-r", str(self.fps)])
        
        # Pixel format
        args.extend(["-pix_fmt", self.pixel_format])
        
        # Quality/bitrate settings
        if self.crf is not None:
            args.extend(["-crf", str(self.crf)])
        elif self.bitrate is not None:
            args.extend(["-b:v", self.bitrate])
        
        # Resolution
        if self.resolution is not None:
            args.extend(["-s", f"{self.resolution[0]}x{self.resolution[1]}"])
        
        # Additional parameters
        for key, value in self.additional_params.items():
            if value is not None:
                args.extend([f"-{key}", str(value)])
        
        return args


@dataclass
class EncodingResult:
    """Result of video encoding operation"""
    success: bool
    output_path: Optional[str] = None
    encoding_time: float = 0.0
    file_size_bytes: int = 0
    actual_fps: Optional[float] = None
    actual_resolution: Optional[Tuple[int, int]] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    ffmpeg_log: Optional[str] = None
    
    def add_error(self, error: str):
        """Add encoding error"""
        self.errors.append(error)
        self.success = False
    
    def add_warning(self, warning: str):
        """Add encoding warning"""
        self.warnings.append(warning)


@dataclass
class FallbackResult:
    """Result of fallback frame-by-frame output"""
    success: bool
    output_directory: Optional[str] = None
    frame_count: int = 0
    frame_paths: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    def add_error(self, error: str):
        """Add fallback error"""
        self.errors.append(error)
        self.success = False


@dataclass
class DependencyStatus:
    """Status of encoding dependencies"""
    ffmpeg_available: bool = False
    ffmpeg_version: Optional[str] = None
    supported_codecs: List[str] = field(default_factory=list)
    supported_formats: List[str] = field(default_factory=list)
    installation_guide: List[str] = field(default_factory=list)


class VideoEncoder:
    """
    Video encoder for converting processed frame arrays to standard video formats
    
    This class handles video encoding with FFmpeg integration, automatic parameter
    optimization, and fallback strategies for encoding failures.
    """
    
    def __init__(self, temp_dir: Optional[str] = None, cleanup_temp: bool = True):
        """
        Initialize video encoder
        
        Args:
            temp_dir: Directory for temporary files (uses system temp if None)
            cleanup_temp: Whether to cleanup temporary files after encoding
        """
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir())
        self.cleanup_temp = cleanup_temp
        self.dependency_status = self.check_encoding_dependencies()
        
        # Default encoding configurations
        self.quality_presets = {
            EncodingQuality.LOW: {"crf": 28, "preset": "fast"},
            EncodingQuality.MEDIUM: {"crf": 23, "preset": "medium"},
            EncodingQuality.HIGH: {"crf": 18, "preset": "slow"},
            EncodingQuality.LOSSLESS: {"crf": 0, "preset": "veryslow"}
        }
        
        # Format-codec mappings
        self.format_codec_map = {
            VideoFormat.MP4: [VideoCodec.H264, VideoCodec.H265],
            VideoFormat.WEBM: [VideoCodec.VP8, VideoCodec.VP9],
            VideoFormat.AVI: [VideoCodec.H264],
            VideoFormat.MOV: [VideoCodec.H264, VideoCodec.H265]
        }
    
    def encode_frames_to_video(self, frames: ProcessedFrames, output_path: str, 
                              format: str = "mp4", quality: str = "medium",
                              **kwargs) -> EncodingResult:
        """
        Encode frame arrays to video file
        
        Args:
            frames: ProcessedFrames object with frame data
            output_path: Path for output video file
            format: Video format ("mp4", "webm", "avi", "mov")
            quality: Encoding quality ("low", "medium", "high", "lossless")
            **kwargs: Additional encoding parameters
            
        Returns:
            EncodingResult with encoding status and metadata
        """
        import time
        start_time = time.time()
        
        logger.info(f"Encoding {frames.num_frames} frames to {format} at {output_path}")
        
        result = EncodingResult(success=False)
        
        try:
            # Validate inputs
            validation_result = self._validate_encoding_inputs(frames, output_path, format)
            if not validation_result.success:
                result.errors.extend(validation_result.errors)
                return result
            
            # Check dependencies
            if not self.dependency_status.ffmpeg_available:
                result.add_error("FFmpeg not available. Please install FFmpeg to encode videos.")
                result.errors.extend(self.dependency_status.installation_guide)
                return result
            
            # Configure encoding parameters
            config = self.configure_encoding_params(frames, format, quality, **kwargs)
            
            # Create temporary frame files
            temp_frame_dir = self._create_temp_frame_files(frames)
            
            try:
                # Run FFmpeg encoding
                ffmpeg_result = self._run_ffmpeg_encoding(temp_frame_dir, output_path, config)
                
                if ffmpeg_result["success"]:
                    result.success = True
                    result.output_path = output_path
                    result.encoding_time = time.time() - start_time
                    result.ffmpeg_log = ffmpeg_result.get("log", "")
                    
                    # Get file info
                    if os.path.exists(output_path):
                        result.file_size_bytes = os.path.getsize(output_path)
                        video_info = self._get_video_info(output_path)
                        result.actual_fps = video_info.get("fps")
                        result.actual_resolution = video_info.get("resolution")
                    
                    logger.info(f"Successfully encoded video: {output_path} "
                               f"({result.file_size_bytes} bytes, {result.encoding_time:.2f}s)")
                else:
                    result.add_error(f"FFmpeg encoding failed: {ffmpeg_result.get('error', 'Unknown error')}")
                    result.ffmpeg_log = ffmpeg_result.get("log", "")
            
            finally:
                # Cleanup temporary files
                if self.cleanup_temp:
                    self._cleanup_temp_files(temp_frame_dir)
        
        except Exception as e:
            logger.error(f"Video encoding failed: {e}")
            result.add_error(f"Encoding error: {str(e)}")
        
        return result
    
    def configure_encoding_params(self, frames: ProcessedFrames, target_format: str,
                                 quality: str = "medium", **kwargs) -> EncodingConfig:
        """
        Configure optimal encoding parameters based on frame properties
        
        Args:
            frames: ProcessedFrames object
            target_format: Target video format
            quality: Encoding quality preset
            **kwargs: Override parameters
            
        Returns:
            EncodingConfig with optimized parameters
        """
        logger.debug(f"Configuring encoding for {target_format} at {quality} quality")
        
        # Parse format and quality
        try:
            video_format = VideoFormat(target_format.lower())
            encoding_quality = EncodingQuality(quality.lower())
        except ValueError as e:
            logger.warning(f"Invalid format/quality, using defaults: {e}")
            video_format = VideoFormat.MP4
            encoding_quality = EncodingQuality.MEDIUM
        
        # Select optimal codec for format
        available_codecs = self.format_codec_map.get(video_format, [VideoCodec.H264])
        codec = available_codecs[0]  # Use first available codec
        
        # Get quality preset
        quality_params = self.quality_presets.get(encoding_quality, self.quality_presets[EncodingQuality.MEDIUM])
        
        # Configure based on frame properties
        config = EncodingConfig(
            codec=codec,
            format=video_format,
            fps=frames.fps,
            resolution=frames.resolution,
            crf=quality_params.get("crf", 23)
        )
        
        # Optimize for frame properties
        if frames.num_frames < 30:  # Short video
            config.additional_params["preset"] = "fast"
        elif frames.num_frames > 300:  # Long video
            config.additional_params["preset"] = "slow"
        else:
            config.additional_params["preset"] = quality_params.get("preset", "medium")
        
        # Optimize for resolution
        if frames.width * frames.height > 1920 * 1080:  # 4K+
            if config.crf is not None:
                config.crf = max(config.crf - 2, 15)  # Higher quality for 4K
            config.additional_params["preset"] = "slow"
        elif frames.width * frames.height < 720 * 480:  # Low res
            if config.crf is not None:
                config.crf = min(config.crf + 3, 28)  # Lower quality acceptable
            config.additional_params["preset"] = "fast"
        
        # Apply user overrides
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                config.additional_params[key] = value
        
        logger.debug(f"Configured encoding: {codec.value} {video_format.value} "
                    f"CRF={config.crf} FPS={config.fps}")
        
        return config
    
    def check_encoding_dependencies(self) -> DependencyStatus:
        """
        Check if FFmpeg and other encoding dependencies are available
        
        Returns:
            DependencyStatus with availability and installation guidance
        """
        status = DependencyStatus()
        
        # Check FFmpeg availability
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                status.ffmpeg_available = True
                # Extract version from output
                version_line = result.stdout.split('\n')[0]
                if "ffmpeg version" in version_line:
                    status.ffmpeg_version = version_line.split()[2]
                
                # Get supported codecs and formats
                status.supported_codecs = self._get_ffmpeg_codecs()
                status.supported_formats = self._get_ffmpeg_formats()
                
                logger.info(f"FFmpeg available: {status.ffmpeg_version}")
            else:
                status.ffmpeg_available = False
                logger.warning("FFmpeg found but returned error")
        
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            status.ffmpeg_available = False
            logger.warning("FFmpeg not found or not working")
        
        # Generate installation guide
        if not status.ffmpeg_available:
            status.installation_guide = [
                "FFmpeg is required for video encoding. Install options:",
                "",
                "Windows:",
                "1. Download from https://ffmpeg.org/download.html",
                "2. Extract and add to PATH",
                "3. Or use: winget install FFmpeg",
                "",
                "macOS:",
                "1. brew install ffmpeg",
                "",
                "Linux (Ubuntu/Debian):",
                "1. sudo apt update && sudo apt install ffmpeg",
                "",
                "Linux (CentOS/RHEL):",
                "1. sudo yum install ffmpeg",
                "",
                "After installation, restart your terminal and try again."
            ]
        
        return status
    
    def provide_fallback_output(self, frames: ProcessedFrames, output_path: str) -> FallbackResult:
        """
        Provide frame-by-frame output when video encoding fails
        
        Args:
            frames: ProcessedFrames object
            output_path: Base path for output (will create directory)
            
        Returns:
            FallbackResult with frame output status
        """
        logger.info(f"Creating fallback frame-by-frame output at {output_path}")
        
        result = FallbackResult(success=False)
        
        try:
            # Create output directory
            output_dir = Path(output_path).with_suffix("")  # Remove extension
            output_dir.mkdir(parents=True, exist_ok=True)
            result.output_directory = str(output_dir)
            
            # Save individual frames
            frame_paths = []
            for i in range(frames.num_frames):
                frame = frames.frames[i]
                
                # Convert to PIL Image
                if frame.dtype != np.uint8:
                    frame_uint8 = (frame * 255).astype(np.uint8)
                else:
                    frame_uint8 = frame
                
                # Handle different channel counts
                if frame.shape[-1] == 1:
                    image = Image.fromarray(frame_uint8.squeeze())
                elif frame.shape[-1] == 3:
                    image = Image.fromarray(frame_uint8)
                elif frame.shape[-1] == 4:
                    image = Image.fromarray(frame_uint8)
                else:
                    result.add_error(f"Unsupported channel count: {frame.shape[-1]}")
                    continue
                
                # Save frame
                frame_filename = f"frame_{i:06d}.png"
                frame_path = output_dir / frame_filename
                image.save(frame_path)
                frame_paths.append(str(frame_path))
            
            result.success = True
            result.frame_count = len(frame_paths)
            result.frame_paths = frame_paths
            
            # Create metadata file
            metadata = {
                "total_frames": frames.num_frames,
                "fps": frames.fps,
                "duration": frames.duration,
                "resolution": frames.resolution,
                "format": frames.format.value,
                "frame_files": [Path(p).name for p in frame_paths],
                "metadata": frames.metadata
            }
            
            metadata_path = output_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Successfully saved {result.frame_count} frames to {output_dir}")
            
        except Exception as e:
            logger.error(f"Fallback output failed: {e}")
            result.add_error(f"Fallback error: {str(e)}")
        
        return result
    
    def _validate_encoding_inputs(self, frames: ProcessedFrames, output_path: str, 
                                 format: str) -> EncodingResult:
        """Validate encoding inputs"""
        result = EncodingResult(success=True)
        
        # Validate frames
        frame_validation = frames.validate_shape()
        if not frame_validation.is_valid:
            result.errors.extend(frame_validation.errors)
            result.success = False
        
        # Validate output path
        output_dir = Path(output_path).parent
        if not output_dir.exists():
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                result.add_error(f"Cannot create output directory: {e}")
        
        # Validate format
        try:
            VideoFormat(format.lower())
        except ValueError:
            result.add_error(f"Unsupported video format: {format}")
        
        return result
    
    def _create_temp_frame_files(self, frames: ProcessedFrames) -> Path:
        """Create temporary frame files for FFmpeg input"""
        temp_dir = self.temp_dir / f"frames_{os.getpid()}_{id(frames)}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        logger.debug(f"Creating temporary frames in {temp_dir}")
        
        for i in range(frames.num_frames):
            frame = frames.frames[i]
            
            # Convert to uint8 if needed
            if frame.dtype != np.uint8:
                frame_uint8 = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
            else:
                frame_uint8 = frame
            
            # Create PIL Image
            if frame.shape[-1] == 1:
                image = Image.fromarray(frame_uint8.squeeze())
            elif frame.shape[-1] == 3:
                image = Image.fromarray(frame_uint8)
            elif frame.shape[-1] == 4:
                image = Image.fromarray(frame_uint8)
            else:
                raise ValueError(f"Unsupported channel count: {frame.shape[-1]}")
            
            # Save as PNG for lossless intermediate format
            frame_path = temp_dir / f"frame_{i:06d}.png"
            image.save(frame_path)
        
        return temp_dir
    
    def _run_ffmpeg_encoding(self, frame_dir: Path, output_path: str, 
                           config: EncodingConfig) -> Dict[str, Any]:
        """Run FFmpeg encoding process"""
        
        # Build FFmpeg command
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output file
            "-framerate", str(config.fps),
            "-i", str(frame_dir / "frame_%06d.png"),
        ]
        
        # Add encoding parameters
        cmd.extend(config.to_ffmpeg_args())
        
        # Output file
        cmd.append(output_path)
        
        logger.debug(f"Running FFmpeg: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                return {
                    "success": True,
                    "log": result.stderr  # FFmpeg outputs to stderr
                }
            else:
                return {
                    "success": False,
                    "error": f"FFmpeg returned code {result.returncode}",
                    "log": result.stderr
                }
        
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "FFmpeg encoding timed out",
                "log": "Process timed out after 5 minutes"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"FFmpeg execution failed: {str(e)}",
                "log": ""
            }
    
    def _get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Get video file information using FFprobe"""
        try:
            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                
                # Extract video stream info
                for stream in data.get("streams", []):
                    if stream.get("codec_type") == "video":
                        fps_str = stream.get("r_frame_rate", "0/1")
                        if "/" in fps_str:
                            num, den = fps_str.split("/")
                            fps = float(num) / float(den) if float(den) != 0 else 0
                        else:
                            fps = float(fps_str)
                        
                        return {
                            "fps": fps,
                            "resolution": (stream.get("width"), stream.get("height")),
                            "duration": float(stream.get("duration", 0)),
                            "codec": stream.get("codec_name")
                        }
        
        except Exception as e:
            logger.warning(f"Could not get video info: {e}")
        
        return {}
    
    def _get_ffmpeg_codecs(self) -> List[str]:
        """Get list of supported codecs from FFmpeg"""
        try:
            result = subprocess.run(
                ["ffmpeg", "-codecs"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                codecs = []
                for line in result.stdout.split('\n'):
                    if 'V' in line and 'E' in line:  # Video encoder
                        parts = line.split()
                        if len(parts) > 1:
                            codecs.append(parts[1])
                return codecs
        
        except Exception:
            pass
        
        return []
    
    def _get_ffmpeg_formats(self) -> List[str]:
        """Get list of supported formats from FFmpeg"""
        try:
            result = subprocess.run(
                ["ffmpeg", "-formats"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                formats = []
                for line in result.stdout.split('\n'):
                    if 'E' in line:  # Encoder format
                        parts = line.split()
                        if len(parts) > 1:
                            formats.append(parts[1])
                return formats
        
        except Exception:
            pass
        
        return []
    
    def _cleanup_temp_files(self, temp_dir: Path):
        """Clean up temporary frame files"""
        try:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                logger.debug(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Could not cleanup temp directory {temp_dir}: {e}")


# Utility functions for common encoding tasks

def encode_video_simple(frames: ProcessedFrames, output_path: str, 
                       format: str = "mp4", quality: str = "medium") -> EncodingResult:
    """
    Simple video encoding function for common use cases
    
    Args:
        frames: ProcessedFrames object
        output_path: Output video file path
        format: Video format ("mp4", "webm")
        quality: Encoding quality ("low", "medium", "high")
        
    Returns:
        EncodingResult with encoding status
    """
    encoder = VideoEncoder()
    return encoder.encode_frames_to_video(frames, output_path, format, quality)


def create_fallback_frames(frames: ProcessedFrames, output_path: str) -> FallbackResult:
    """
    Create fallback frame-by-frame output
    
    Args:
        frames: ProcessedFrames object
        output_path: Base output path
        
    Returns:
        FallbackResult with frame output status
    """
    encoder = VideoEncoder()
    return encoder.provide_fallback_output(frames, output_path)


def check_video_encoding_support() -> DependencyStatus:
    """
    Check if video encoding is supported on this system
    
    Returns:
        DependencyStatus with availability information
    """
    encoder = VideoEncoder()
    return encoder.dependency_status