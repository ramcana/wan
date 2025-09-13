"""
Frame Tensor Handler for Wan Model Video Processing Pipeline

This module handles the processing of raw model output tensors into standardized
frame arrays suitable for video encoding. It supports batch processing, frame
validation, and normalization for different tensor formats.

Requirements addressed:
- 7.1: Process frame tensors for video encoding
- 7.2: Handle frame rate and resolution metadata
- 7.3: Support fallback frame-by-frame output
"""

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class TensorFormat(Enum):
    """Supported tensor formats for frame processing"""
    TORCH_FLOAT32 = "torch_float32"
    TORCH_FLOAT16 = "torch_float16"
    NUMPY_FLOAT32 = "numpy_float32"
    NUMPY_UINT8 = "numpy_uint8"
    UNKNOWN = "unknown"


class ValidationResult:
    """Result of frame validation operations"""
    def __init__(self, is_valid: bool, errors: List[str] = None, warnings: List[str] = None):
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []

    def add_error(self, error: str):
        """Add validation error"""
        self.errors.append(error)
        self.is_valid = False

    def add_warning(self, warning: str):
        """Add validation warning"""
        self.warnings.append(warning)


@dataclass
class ProcessedFrames:
    """
    Data model for processed video frames with metadata support
    
    Attributes:
        frames: Normalized frame array (num_frames, height, width, channels)
        fps: Frame rate for video encoding
        duration: Total video duration in seconds
        resolution: (width, height) tuple
        format: Original tensor format
        metadata: Additional processing metadata
    """
    frames: np.ndarray
    fps: float
    duration: float
    resolution: Tuple[int, int]
    format: TensorFormat
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def num_frames(self) -> int:
        """Get number of frames"""
        return self.frames.shape[0] if len(self.frames.shape) >= 4 else 0
    
    @property
    def height(self) -> int:
        """Get frame height"""
        return self.frames.shape[1] if len(self.frames.shape) >= 4 else 0
    
    @property
    def width(self) -> int:
        """Get frame width"""
        return self.frames.shape[2] if len(self.frames.shape) >= 4 else 0
    
    @property
    def channels(self) -> int:
        """Get number of channels"""
        return self.frames.shape[3] if len(self.frames.shape) >= 4 else 0
    
    def validate_shape(self) -> ValidationResult:
        """Validate frame array shape"""
        result = ValidationResult(True)
        
        if len(self.frames.shape) != 4:
            result.add_error(f"Expected 4D array (N,H,W,C), got {len(self.frames.shape)}D")
        
        if self.num_frames == 0:
            result.add_error("No frames found in array")
        
        if self.height == 0 or self.width == 0:
            result.add_error(f"Invalid frame dimensions: {self.width}x{self.height}")
        
        if self.channels not in [1, 3, 4]:
            result.add_warning(f"Unusual channel count: {self.channels}")
        
        return result


class FrameTensorHandler:
    """
    Handler for processing raw model output tensors into standardized frame arrays
    
    This class processes various tensor formats from video generation models,
    validates frame dimensions, normalizes data ranges, and handles batch outputs.
    """
    
    def __init__(self, default_fps: float = 24.0, target_dtype: np.dtype = np.float32):
        """
        Initialize frame tensor handler
        
        Args:
            default_fps: Default frame rate when not specified
            target_dtype: Target numpy dtype for normalized frames
        """
        self.default_fps = default_fps
        self.target_dtype = target_dtype
        self.supported_formats = {
            torch.float32: TensorFormat.TORCH_FLOAT32,
            torch.float16: TensorFormat.TORCH_FLOAT16,
            np.float32: TensorFormat.NUMPY_FLOAT32,
            np.uint8: TensorFormat.NUMPY_UINT8
        }
    
    def process_output_tensors(self, output: Any, fps: Optional[float] = None, 
                             resolution: Optional[Tuple[int, int]] = None) -> ProcessedFrames:
        """
        Process raw model output tensors into standardized frame arrays
        
        Args:
            output: Raw model output (torch.Tensor, np.ndarray, or dict)
            fps: Target frame rate (uses default if None)
            resolution: Target resolution (inferred if None)
            
        Returns:
            ProcessedFrames object with normalized frame data
            
        Raises:
            ValueError: If output format is unsupported
            RuntimeError: If processing fails
        """
        try:
            logger.info(f"Processing output tensors: {type(output)}")
            
            # Extract tensor from various output formats
            tensor = self._extract_tensor_from_output(output)
            
            # Detect tensor format
            tensor_format = self._detect_tensor_format(tensor)
            
            # Convert to numpy array
            frames_array = self._convert_to_numpy(tensor)
            
            # Validate and normalize dimensions
            frames_array = self._normalize_tensor_dimensions(frames_array)
            
            # Normalize data range
            frames_array = self._normalize_data_range(frames_array, tensor_format)
            
            # Infer metadata
            inferred_fps = fps or self.default_fps
            inferred_resolution = resolution or (frames_array.shape[2], frames_array.shape[1])
            duration = frames_array.shape[0] / inferred_fps
            
            # Create metadata
            metadata = {
                "original_shape": tensor.shape if hasattr(tensor, 'shape') else None,
                "original_dtype": str(tensor.dtype) if hasattr(tensor, 'dtype') else None,
                "processing_timestamp": np.datetime64('now').astype(str),
                "normalization_applied": True
            }
            
            processed_frames = ProcessedFrames(
                frames=frames_array,
                fps=inferred_fps,
                duration=duration,
                resolution=inferred_resolution,
                format=tensor_format,
                metadata=metadata
            )
            
            logger.info(f"Successfully processed {processed_frames.num_frames} frames "
                       f"at {processed_frames.fps} fps ({processed_frames.duration:.2f}s)")
            
            return processed_frames
            
        except Exception as e:
            logger.error(f"Failed to process output tensors: {e}")
            raise RuntimeError(f"Tensor processing failed: {e}") from e
    
    def validate_frame_dimensions(self, frames: np.ndarray) -> ValidationResult:
        """
        Validate frame tensor dimensions and format
        
        Args:
            frames: Frame array to validate
            
        Returns:
            ValidationResult with validation status and messages
        """
        result = ValidationResult(True)
        
        # Check basic shape requirements
        if not isinstance(frames, np.ndarray):
            result.add_error(f"Expected numpy array, got {type(frames)}")
            return result
        
        if len(frames.shape) < 3:
            result.add_error(f"Expected at least 3D array, got {len(frames.shape)}D")
            return result
        
        if len(frames.shape) == 3:
            result.add_warning("3D array detected, assuming single frame (H,W,C)")
        elif len(frames.shape) == 4:
            # Standard format (N,H,W,C)
            num_frames, height, width, channels = frames.shape
            
            if num_frames == 0:
                result.add_error("No frames found")
            elif num_frames > 1000:
                result.add_warning(f"Very long video: {num_frames} frames")
            
            if height < 64 or width < 64:
                result.add_warning(f"Low resolution: {width}x{height}")
            elif height > 2160 or width > 3840:
                result.add_warning(f"Very high resolution: {width}x{height}")
            
            if channels not in [1, 3, 4]:
                result.add_error(f"Unsupported channel count: {channels}")
        else:
            result.add_error(f"Unsupported array dimensions: {frames.shape}")
        
        # Check data type
        if frames.dtype not in [np.float32, np.float16, np.uint8]:
            result.add_warning(f"Unusual data type: {frames.dtype}")
        
        # Check data range
        if frames.dtype in [np.float32, np.float16]:
            if frames.min() < -2.0 or frames.max() > 2.0:
                result.add_warning(f"Unusual float range: [{frames.min():.3f}, {frames.max():.3f}]")
        elif frames.dtype == np.uint8:
            if frames.min() < 0 or frames.max() > 255:
                result.add_error(f"Invalid uint8 range: [{frames.min()}, {frames.max()}]")
        
        return result
    
    def normalize_frame_data(self, frames: np.ndarray, source_format: TensorFormat) -> np.ndarray:
        """
        Normalize frame data to standard video format
        
        Args:
            frames: Input frame array
            source_format: Original tensor format
            
        Returns:
            Normalized frame array in range [0, 1] as float32
        """
        logger.debug(f"Normalizing frames from {source_format} to float32 [0,1]")
        
        frames = frames.astype(self.target_dtype)
        
        if source_format in [TensorFormat.TORCH_FLOAT32, TensorFormat.TORCH_FLOAT16]:
            # Typical range for diffusion models: [-1, 1] -> [0, 1]
            if frames.min() >= -1.1 and frames.max() <= 1.1:
                frames = (frames + 1.0) / 2.0
            # Already in [0, 1] range
            elif frames.min() >= -0.1 and frames.max() <= 1.1:
                frames = np.clip(frames, 0.0, 1.0)
            else:
                # Unknown range, normalize to [0, 1]
                frames = (frames - frames.min()) / (frames.max() - frames.min())
                logger.warning(f"Unknown float range, normalized from [{frames.min():.3f}, {frames.max():.3f}]")
        
        elif source_format == TensorFormat.NUMPY_UINT8:
            # Convert uint8 [0, 255] -> float32 [0, 1]
            frames = frames / 255.0
        
        elif source_format == TensorFormat.NUMPY_FLOAT32:
            # Assume already normalized or apply same logic as torch
            if frames.min() >= -1.1 and frames.max() <= 1.1:
                frames = (frames + 1.0) / 2.0
            elif frames.min() >= -0.1 and frames.max() <= 1.1:
                frames = np.clip(frames, 0.0, 1.0)
            else:
                frames = (frames - frames.min()) / (frames.max() - frames.min())
        
        # Final clipping to ensure [0, 1] range
        frames = np.clip(frames, 0.0, 1.0)
        
        return frames
    
    def handle_batch_outputs(self, batch_output: Any) -> List[ProcessedFrames]:
        """
        Handle batched video generation outputs
        
        Args:
            batch_output: Batch output from model (list, dict, or tensor)
            
        Returns:
            List of ProcessedFrames objects, one per batch item
        """
        logger.info(f"Processing batch output: {type(batch_output)}")
        
        processed_batch = []
        
        try:
            if isinstance(batch_output, (list, tuple)):
                # List of individual outputs
                for i, output in enumerate(batch_output):
                    logger.debug(f"Processing batch item {i+1}/{len(batch_output)}")
                    processed = self.process_output_tensors(output)
                    processed.metadata["batch_index"] = i
                    processed_batch.append(processed)
            
            elif isinstance(batch_output, dict):
                # Dictionary with batch dimension
                if "frames" in batch_output or "videos" in batch_output:
                    key = "frames" if "frames" in batch_output else "videos"
                    batch_tensor = batch_output[key]
                    
                    if hasattr(batch_tensor, 'shape') and len(batch_tensor.shape) == 5:
                        # 5D tensor with batch dimension (B, T, H, W, C)
                        for i in range(batch_tensor.shape[0]):
                            single_output = {key: batch_tensor[i]}
                            processed = self.process_output_tensors(single_output)
                            processed.metadata["batch_index"] = i
                            processed_batch.append(processed)
                    else:
                        # Single item in dict format
                        processed = self.process_output_tensors(batch_output)
                        processed.metadata["batch_index"] = 0
                        processed_batch.append(processed)
                else:
                    # Unknown dict format, try as single output
                    processed = self.process_output_tensors(batch_output)
                    processed.metadata["batch_index"] = 0
                    processed_batch.append(processed)
            
            elif hasattr(batch_output, 'shape') and len(batch_output.shape) == 5:
                # 5D tensor with batch dimension (B, T, H, W, C) or (B, T, C, H, W)
                batch_size = batch_output.shape[0]
                for i in range(batch_size):
                    single_tensor = batch_output[i]
                    processed = self.process_output_tensors(single_tensor)
                    processed.metadata["batch_index"] = i
                    processed_batch.append(processed)
            
            elif hasattr(batch_output, 'shape') and len(batch_output.shape) == 4:
                # For 4D tensors, we need to determine if it's (B, T, H, W) or (T, H, W, C)
                # Heuristic: if last dimension is 1, 3, or 4 (typical channel counts), treat as single video
                # Otherwise, if first dimension is small compared to second, treat as batch
                if batch_output.shape[-1] in [1, 3, 4]:
                    # Likely (T, H, W, C) - single video
                    processed = self.process_output_tensors(batch_output)
                    processed.metadata["batch_index"] = 0
                    processed_batch.append(processed)
                elif batch_output.shape[0] <= 8 and batch_output.shape[1] > batch_output.shape[0] * 2:
                    # Likely (B, T, H, W) - batch of videos without explicit channel dimension
                    batch_size = batch_output.shape[0]
                    for i in range(batch_size):
                        single_tensor = batch_output[i]
                        processed = self.process_output_tensors(single_tensor)
                        processed.metadata["batch_index"] = i
                        processed_batch.append(processed)
                else:
                    # Default to single video
                    processed = self.process_output_tensors(batch_output)
                    processed.metadata["batch_index"] = 0
                    processed_batch.append(processed)
            
            else:
                # Single output, treat as batch of 1
                processed = self.process_output_tensors(batch_output)
                processed.metadata["batch_index"] = 0
                processed_batch.append(processed)
            
            logger.info(f"Successfully processed batch of {len(processed_batch)} videos")
            return processed_batch
            
        except Exception as e:
            logger.error(f"Failed to process batch output: {e}")
            raise RuntimeError(f"Batch processing failed: {e}") from e
    
    def _extract_tensor_from_output(self, output: Any) -> Union[torch.Tensor, np.ndarray]:
        """Extract tensor from various output formats"""
        if isinstance(output, (torch.Tensor, np.ndarray)):
            return output
        
        elif isinstance(output, dict):
            # Common keys for video outputs
            for key in ["frames", "videos", "images", "samples"]:
                if key in output:
                    return output[key]
            
            # If no standard key found, try first tensor-like value
            for value in output.values():
                if isinstance(value, (torch.Tensor, np.ndarray)):
                    return value
            
            raise ValueError(f"No tensor found in output dict with keys: {list(output.keys())}")
        
        elif hasattr(output, 'frames'):
            return output.frames
        elif hasattr(output, 'videos'):
            return output.videos
        elif hasattr(output, 'images'):
            return output.images
        
        else:
            raise ValueError(f"Unsupported output type: {type(output)}")
    
    def _detect_tensor_format(self, tensor: Union[torch.Tensor, np.ndarray]) -> TensorFormat:
        """Detect the format of input tensor"""
        if isinstance(tensor, torch.Tensor):
            if tensor.dtype == torch.float32:
                return TensorFormat.TORCH_FLOAT32
            elif tensor.dtype == torch.float16:
                return TensorFormat.TORCH_FLOAT16
            else:
                logger.warning(f"Unknown torch dtype: {tensor.dtype}")
                return TensorFormat.UNKNOWN
        
        elif isinstance(tensor, np.ndarray):
            if tensor.dtype == np.float32:
                return TensorFormat.NUMPY_FLOAT32
            elif tensor.dtype == np.uint8:
                return TensorFormat.NUMPY_UINT8
            else:
                logger.warning(f"Unknown numpy dtype: {tensor.dtype}")
                return TensorFormat.UNKNOWN
        
        else:
            return TensorFormat.UNKNOWN
    
    def _convert_to_numpy(self, tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Convert tensor to numpy array"""
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        elif isinstance(tensor, np.ndarray):
            return tensor
        else:
            raise ValueError(f"Cannot convert {type(tensor)} to numpy array")
    
    def _normalize_tensor_dimensions(self, frames: np.ndarray) -> np.ndarray:
        """Normalize tensor dimensions to (N, H, W, C) format"""
        original_shape = frames.shape
        
        if len(frames.shape) == 3:
            # Single frame (H, W, C) -> (1, H, W, C)
            frames = frames[np.newaxis, ...]
            logger.debug(f"Expanded single frame: {original_shape} -> {frames.shape}")
        
        elif len(frames.shape) == 4:
            # Check if it's (N, C, H, W) and convert to (N, H, W, C)
            if frames.shape[1] in [1, 3, 4] and frames.shape[1] < frames.shape[2]:
                frames = np.transpose(frames, (0, 2, 3, 1))
                logger.debug(f"Transposed NCHW to NHWC: {original_shape} -> {frames.shape}")
        
        elif len(frames.shape) == 5:
            # Video tensor (B, T, C, H, W) -> (T, H, W, C) for first batch
            if frames.shape[0] == 1:
                frames = frames[0]  # Remove batch dimension
                if frames.shape[1] in [1, 3, 4]:  # (T, C, H, W) -> (T, H, W, C)
                    frames = np.transpose(frames, (0, 2, 3, 1))
                logger.debug(f"Converted 5D to 4D: {original_shape} -> {frames.shape}")
            else:
                raise ValueError(f"Cannot handle batch size > 1 in 5D tensor: {frames.shape}")
        
        else:
            raise ValueError(f"Unsupported tensor dimensions: {frames.shape}")
        
        return frames
    
    def _normalize_data_range(self, frames: np.ndarray, tensor_format: TensorFormat) -> np.ndarray:
        """Normalize data range using the normalize_frame_data method"""
        return self.normalize_frame_data(frames, tensor_format)
