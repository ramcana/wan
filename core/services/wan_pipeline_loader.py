"""
WanPipeline wrapper and loader for resource-managed generation.

This module implements the WanPipelineLoader and WanPipelineWrapper classes
for automatic optimization application and resource-managed video generation.

Requirements addressed: 3.1, 3.2, 3.3, 5.1, 5.2
"""

import torch
import logging
import gc
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union, Callable, Tuple
from pathlib import Path
import json

from infrastructure.hardware.architecture_detector import ArchitectureDetector, ModelArchitecture, ArchitectureType
from pipeline_manager import PipelineManager, PipelineLoadResult, PipelineLoadStatus
from core.services.optimization_manager import (
    OptimizationManager, OptimizationPlan, OptimizationResult, 
    SystemResources, ModelRequirements, ChunkedProcessor
)

# Import WAN22 System Optimization components
from vram_manager import VRAMManager, VRAMUsage, GPUInfo
from quantization_controller import (
    QuantizationController, QuantizationStrategy, QuantizationMethod,
    QuantizationResult, HardwareProfile as QuantHardwareProfile,
    ModelInfo as QuantModelInfo
)

logger = logging.getLogger(__name__)


@dataclass
class MemoryEstimate:
    """Memory usage estimation for generation parameters."""
    base_model_mb: int
    generation_overhead_mb: int
    output_tensors_mb: int
    total_estimated_mb: int
    peak_usage_mb: int
    confidence: float  # 0.0 to 1.0
    warnings: List[str] = field(default_factory=list)


@dataclass
class VideoGenerationResult:
    """Result of video generation with metadata."""
    success: bool
    frames: Optional[List[torch.Tensor]] = None
    output_path: Optional[str] = None
    generation_time: float = 0.0
    memory_used_mb: int = 0
    peak_memory_mb: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    applied_optimizations: List[str] = field(default_factory=list)


@dataclass
class GenerationConfig:
    """Configuration for video generation."""
    prompt: str
    negative_prompt: Optional[str] = None
    num_frames: int = 16
    width: int = 512
    height: int = 512
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    seed: Optional[int] = None
    batch_size: int = 1
    fps: float = 8.0
    
    # Optimization settings
    enable_optimizations: bool = True
    force_chunked_processing: bool = False
    max_chunk_size: Optional[int] = None
    precision_override: Optional[str] = None  # "fp16", "bf16", "fp32"
    
    # Callbacks
    progress_callback: Optional[Callable[[int, int, torch.Tensor], None]] = None
    chunk_callback: Optional[Callable[[int, int], None]] = None


class WanPipelineWrapper:
    """
    Wrapper for Wan pipelines with resource management and optimization.
    
    Provides memory usage estimation, monitoring, and automatic optimization
    application for efficient video generation.
    """
    
    def __init__(self, 
                 pipeline: Any,
                 model_architecture: ModelArchitecture,
                 optimization_result: OptimizationResult,
                 chunked_processor: Optional[ChunkedProcessor] = None):
        """
        Initialize WanPipelineWrapper.
        
        Args:
            pipeline: The loaded Wan pipeline
            model_architecture: Model architecture information
            optimization_result: Applied optimization results
            chunked_processor: Optional chunked processor for memory-constrained systems
        """
        self.pipeline = pipeline
        self.model_architecture = model_architecture
        self.optimization_result = optimization_result
        self.chunked_processor = chunked_processor
        self.logger = logging.getLogger(__name__ + ".WanPipelineWrapper")
        
        # Initialize monitoring
        self._generation_count = 0
        self._total_generation_time = 0.0
        self._peak_memory_usage = 0
        
        # Store pipeline capabilities
        self._supports_chunked_processing = chunked_processor is not None
        self._supports_progress_callback = hasattr(pipeline, 'callback_on_step_end')
        
        self.logger.info(f"WanPipelineWrapper initialized with optimizations: {optimization_result.applied_optimizations}")
    
    def generate(self, config: GenerationConfig) -> VideoGenerationResult:
        """
        Generate video with automatic resource management.
        
        Args:
            config: Generation configuration
            
        Returns:
            VideoGenerationResult with generated frames and metadata
        """
        start_time = time.time()
        initial_memory = self._get_current_memory_usage()
        
        self.logger.info(f"Starting video generation: {config.num_frames} frames at {config.width}x{config.height}")
        
        try:
            # Validate generation parameters
            validation_result = self._validate_generation_config(config)
            if not validation_result["valid"]:
                return VideoGenerationResult(
                    success=False,
                    errors=validation_result["errors"],
                    warnings=validation_result["warnings"]
                )
            
            # Estimate memory usage
            memory_estimate = self.estimate_memory_usage(config)
            
            # Check if chunked processing is needed
            use_chunked = self._should_use_chunked_processing(config, memory_estimate)
            
            # Generate frames
            if use_chunked and self.chunked_processor:
                frames = self._generate_chunked(config)
            else:
                frames = self._generate_standard(config)
            
            # Calculate final metrics
            end_time = time.time()
            generation_time = end_time - start_time
            peak_memory = self._get_peak_memory_usage()
            final_memory = self._get_current_memory_usage()
            memory_used = final_memory - initial_memory
            
            # Update statistics
            self._generation_count += 1
            self._total_generation_time += generation_time
            self._peak_memory_usage = max(self._peak_memory_usage, peak_memory)
            
            # Create result
            result = VideoGenerationResult(
                success=True,
                frames=frames,
                generation_time=generation_time,
                memory_used_mb=memory_used,
                peak_memory_mb=peak_memory,
                applied_optimizations=self.optimization_result.applied_optimizations.copy(),
                metadata={
                    "model_architecture": self.model_architecture.architecture_type.value,
                    "generation_config": config.__dict__.copy(),
                    "memory_estimate": memory_estimate.__dict__.copy(),
                    "used_chunked_processing": use_chunked,
                    "generation_count": self._generation_count,
                    "average_generation_time": self._total_generation_time / self._generation_count
                }
            )
            
            # Add warnings from memory estimate
            result.warnings.extend(memory_estimate.warnings)
            
            # Add performance warnings
            if generation_time > 300:  # 5 minutes
                result.warnings.append("Generation took longer than expected - consider using optimizations")
            
            if memory_used > 1024:  # 1GB
                result.warnings.append("High memory usage detected - consider chunked processing")
            
            self.logger.info(f"Generation completed in {generation_time:.2f}s, peak memory: {peak_memory}MB")
            return result
            
        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            return VideoGenerationResult(
                success=False,
                generation_time=time.time() - start_time,
                errors=[f"Generation failed: {str(e)}"],
                applied_optimizations=self.optimization_result.applied_optimizations.copy()
            )
        finally:
            # Cleanup memory
            self._cleanup_memory()
    
    def estimate_memory_usage(self, config: GenerationConfig) -> MemoryEstimate:
        """
        Estimate memory usage for generation parameters.
        
        Args:
            config: Generation configuration
            
        Returns:
            MemoryEstimate with detailed memory breakdown
        """
        # Base model memory (from optimization result or architecture)
        base_model_mb = getattr(self.model_architecture.requirements, 'model_size_mb', 8192)
        
        # Calculate generation overhead based on parameters
        pixel_count = config.width * config.height * config.num_frames * config.batch_size
        
        # Estimate intermediate tensor memory (rough calculation)
        # This varies significantly based on model architecture and optimization
        bytes_per_pixel = 4  # float32
        if "fp16" in self.optimization_result.applied_optimizations:
            bytes_per_pixel = 2
        elif "bf16" in self.optimization_result.applied_optimizations:
            bytes_per_pixel = 2
        
        # Intermediate tensors (activations, attention maps, etc.)
        # Wan models typically have multiple transformer layers
        num_layers = 24  # Typical for Wan models
        intermediate_mb = (pixel_count * bytes_per_pixel * num_layers) // (1024 * 1024)
        
        # Output tensor memory
        output_mb = (pixel_count * bytes_per_pixel * 3) // (1024 * 1024)  # RGB channels
        
        # Apply optimization reductions
        if "CPU offloading" in str(self.optimization_result.applied_optimizations):
            # CPU offloading reduces peak VRAM usage
            intermediate_mb = int(intermediate_mb * 0.6)
            base_model_mb = int(base_model_mb * 0.7)
        
        # Total and peak estimates
        total_estimated = base_model_mb + intermediate_mb + output_mb
        peak_usage = int(total_estimated * 1.2)  # 20% overhead for peak usage
        
        # Confidence calculation based on known factors
        confidence = 0.7  # Base confidence
        if self.model_architecture.architecture_type == ArchitectureType.WAN_T2V:
            confidence += 0.1  # More confident with known architecture
        if len(self.optimization_result.applied_optimizations) > 0:
            confidence += 0.1  # More confident with applied optimizations
        confidence = min(1.0, confidence)
        
        # Generate warnings
        warnings = []
        if total_estimated > 12288:  # 12GB
            warnings.append("High memory usage estimated - consider chunked processing")
        if config.num_frames > 32:
            warnings.append("Large number of frames may require significant memory")
        if config.width * config.height > 1024 * 1024:
            warnings.append("High resolution may require additional memory")
        
        # Add warnings for large generations even if under thresholds
        if config.num_frames >= 32 or (config.width * config.height >= 1024 * 1024):
            warnings.append("Large generation parameters detected")
        
        return MemoryEstimate(
            base_model_mb=base_model_mb,
            generation_overhead_mb=intermediate_mb,
            output_tensors_mb=output_mb,
            total_estimated_mb=total_estimated,
            peak_usage_mb=peak_usage,
            confidence=confidence,
            warnings=warnings
        )
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """
        Get generation statistics and performance metrics.
        
        Returns:
            Dictionary with generation statistics
        """
        return {
            "generation_count": self._generation_count,
            "total_generation_time": self._total_generation_time,
            "average_generation_time": self._total_generation_time / max(1, self._generation_count),
            "peak_memory_usage_mb": self._peak_memory_usage,
            "applied_optimizations": self.optimization_result.applied_optimizations,
            "supports_chunked_processing": self._supports_chunked_processing,
            "supports_progress_callback": self._supports_progress_callback,
            "model_architecture": self.model_architecture.architecture_type.value,
            "optimization_success": self.optimization_result.success
        }
    
    def _validate_generation_config(self, config: GenerationConfig) -> Dict[str, Any]:
        """Validate generation configuration parameters."""
        errors = []
        warnings = []
        
        # Basic parameter validation
        if config.num_frames < 1:
            errors.append("num_frames must be at least 1")
        if config.width < 64 or config.height < 64:
            errors.append("width and height must be at least 64")
        if config.num_inference_steps < 1:
            errors.append("num_inference_steps must be at least 1")
        if config.guidance_scale < 0:
            errors.append("guidance_scale must be non-negative")
        if config.batch_size < 1:
            errors.append("batch_size must be at least 1")
        
        # Architecture-specific validation
        if self.model_architecture.architecture_type == ArchitectureType.WAN_T2V:
            if config.num_frames > 64:
                warnings.append("Large number of frames may cause memory issues")
            if config.width % 8 != 0 or config.height % 8 != 0:
                warnings.append("Width and height should be multiples of 8 for optimal performance")
        
        # Optimization-specific warnings
        if "CPU offloading" in str(self.optimization_result.applied_optimizations):
            if config.batch_size > 1:
                warnings.append("Batch size > 1 with CPU offloading may be slow")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    def _should_use_chunked_processing(self, config: GenerationConfig, 
                                     memory_estimate: MemoryEstimate) -> bool:
        """Determine if chunked processing should be used."""
        if config.force_chunked_processing:
            return True
        
        if not self._supports_chunked_processing:
            return False
        
        # Check if estimated memory exceeds available VRAM
        if torch.cuda.is_available():
            available_vram = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
            available_vram -= torch.cuda.memory_allocated() // (1024 * 1024)
            
            if memory_estimate.peak_usage_mb > available_vram * 0.8:  # 80% threshold
                return True
        
        # Use chunked processing for very large generations
        if config.num_frames > 32 or (config.width * config.height > 1024 * 1024):
            return True
        
        return False
    
    def _generate_standard(self, config: GenerationConfig) -> List[torch.Tensor]:
        """Generate video using standard (non-chunked) processing."""
        self.logger.info("Using standard generation")
        
        # Prepare generation arguments
        generation_args = {
            "prompt": config.prompt,
            "num_frames": config.num_frames,
            "width": config.width,
            "height": config.height,
            "num_inference_steps": config.num_inference_steps,
            "guidance_scale": config.guidance_scale,
        }
        
        # Add optional parameters
        if config.negative_prompt:
            generation_args["negative_prompt"] = config.negative_prompt
        if config.seed is not None:
            generation_args["generator"] = torch.Generator().manual_seed(config.seed)
        
        # Add progress callback if supported
        if self._supports_progress_callback and config.progress_callback:
            generation_args["callback_on_step_end"] = config.progress_callback
        
        # Generate
        with torch.no_grad():
            result = self.pipeline(**generation_args)
        
        # Extract frames from result
        frames = self._extract_frames_from_result(result)
        
        # Ensure we return the correct number of frames
        if len(frames) != config.num_frames:
            self.logger.warning(f"Expected {config.num_frames} frames, got {len(frames)}")
            # Truncate or pad as needed
            if len(frames) > config.num_frames:
                frames = frames[:config.num_frames]
        
        return frames
    
    def _generate_chunked(self, config: GenerationConfig) -> List[torch.Tensor]:
        """Generate video using chunked processing."""
        self.logger.info(f"Using chunked generation with processor: {self.chunked_processor}")
        
        # Determine chunk size
        chunk_size = config.max_chunk_size or self.chunked_processor.chunk_size
        
        # Create chunk callback wrapper
        def chunk_progress_callback(chunk_idx: int, total_chunks: int):
            if config.chunk_callback:
                config.chunk_callback(chunk_idx, total_chunks)
            self.logger.info(f"Processing chunk {chunk_idx + 1}/{total_chunks}")
        
        # Generate using chunked processor
        frames = self.chunked_processor.process_chunked_generation(
            pipeline=self.pipeline,
            prompt=config.prompt,
            num_frames=config.num_frames,
            negative_prompt=config.negative_prompt,
            width=config.width,
            height=config.height,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
            generator=torch.Generator().manual_seed(config.seed) if config.seed else None,
            chunk_callback=chunk_progress_callback
        )
        
        return frames
    
    def _extract_frames_from_result(self, result: Any) -> List[torch.Tensor]:
        """Extract frame tensors from pipeline result."""
        if hasattr(result, 'frames'):
            frames = result.frames
        elif hasattr(result, 'images'):
            frames = result.images
        elif hasattr(result, 'videos'):
            frames = result.videos
        elif isinstance(result, torch.Tensor):
            frames = result
        elif isinstance(result, list):
            frames = result
        else:
            # Try to find frames in result attributes
            for attr in ['frames', 'images', 'videos', 'outputs']:
                if hasattr(result, attr):
                    frames = getattr(result, attr)
                    break
            else:
                raise ValueError(f"Could not extract frames from result type: {type(result)}")
        
        # Ensure frames is a list of tensors
        if isinstance(frames, torch.Tensor):
            if frames.dim() == 4:  # (batch, channels, height, width)
                frames = [frames[i] for i in range(frames.shape[0])]
            elif frames.dim() == 5:  # (batch, frames, channels, height, width)
                frames = [frames[0, i] for i in range(frames.shape[1])]
            else:
                frames = [frames]
        
        return frames
    
    def _get_current_memory_usage(self) -> int:
        """Get current GPU memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() // (1024 * 1024)
        return 0
    
    def _get_peak_memory_usage(self) -> int:
        """Get peak GPU memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() // (1024 * 1024)
        return 0
    
    def _cleanup_memory(self):
        """Clean up GPU and system memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()


class WanPipelineLoader:
    """
    Loader for Wan pipelines with automatic optimization application.
    
    Handles model detection, pipeline loading, optimization application,
    and wrapper creation for seamless video generation.
    """
    
    def __init__(self, 
                 optimization_config_path: Optional[str] = None,
                 enable_caching: bool = True,
                 vram_manager: Optional[VRAMManager] = None,
                 quantization_controller: Optional[QuantizationController] = None):
        """
        Initialize WanPipelineLoader.
        
        Args:
            optimization_config_path: Path to optimization configuration
            enable_caching: Whether to cache loaded pipelines
            vram_manager: Optional VRAM manager instance
            quantization_controller: Optional quantization controller instance
        """
        self.architecture_detector = ArchitectureDetector()
        self.pipeline_manager = PipelineManager()
        self.optimization_manager = OptimizationManager(optimization_config_path)
        self.enable_caching = enable_caching
        self.logger = logging.getLogger(__name__ + ".WanPipelineLoader")
        
        # Initialize WAN22 optimization components
        self.vram_manager = vram_manager or VRAMManager()
        self.quantization_controller = quantization_controller or QuantizationController()
        
        # Pipeline cache
        self._pipeline_cache = {} if enable_caching else None
        
        # System resources (cached after first analysis)
        self._system_resources = None
        
        self.logger.info("WanPipelineLoader initialized with WAN22 optimization components")
    
    def load_wan_pipeline(self, 
                         model_path: str,
                         trust_remote_code: bool = True,
                         apply_optimizations: bool = True,
                         optimization_config: Optional[Dict[str, Any]] = None,
                         **pipeline_kwargs) -> WanPipelineWrapper:
        """
        Load Wan pipeline with automatic optimization.
        
        Args:
            model_path: Path to the model
            trust_remote_code: Whether to trust remote code
            apply_optimizations: Whether to apply automatic optimizations
            optimization_config: Optional optimization configuration override
            **pipeline_kwargs: Additional pipeline loading arguments
            
        Returns:
            WanPipelineWrapper with loaded and optimized pipeline
            
        Raises:
            ValueError: If model cannot be loaded or is incompatible
            RuntimeError: If optimization fails critically
        """
        self.logger.info(f"Loading Wan pipeline from: {model_path}")
        
        # Check cache first
        config_hash = hash(str(sorted(optimization_config.items()))) if optimization_config else 0
        cache_key = f"{model_path}_{trust_remote_code}_{apply_optimizations}_{config_hash}"
        if self._pipeline_cache is not None and cache_key in self._pipeline_cache:
            self.logger.info("Returning cached pipeline")
            return self._pipeline_cache[cache_key]
        
        try:
            # Step 1: Detect model architecture
            self.logger.info("Detecting model architecture...")
            model_architecture = self.architecture_detector.detect_model_architecture(model_path)
            
            # Validate that this is a Wan model
            if model_architecture.architecture_type not in [
                ArchitectureType.WAN_T2V, ArchitectureType.WAN_T2I, ArchitectureType.WAN_I2V
            ]:
                raise ValueError(f"Model is not a Wan architecture: {model_architecture.architecture_type.value}")
            
            # Step 2: Select and load pipeline
            self.logger.info("Loading pipeline...")
            pipeline_class = self.pipeline_manager.select_pipeline_class(model_architecture.signature)
            
            # For WAN models, always ensure trust_remote_code is True
            if pipeline_class == "WanPipeline":
                trust_remote_code = True
                self.logger.info("WAN model detected - enabling trust_remote_code")
            
            # Prepare pipeline loading arguments - filter out WAN-specific args that cause conflicts
            wan_specific_args = {'model_architecture', 'progress_callback', 'boundary_ratio'}
            filtered_kwargs = {k: v for k, v in pipeline_kwargs.items() if k not in wan_specific_args}
            
            load_args = {
                "trust_remote_code": trust_remote_code,
                **filtered_kwargs
            }
            
            # Apply precision settings if specified
            if optimization_config and "precision" in optimization_config:
                precision = optimization_config["precision"]
                if precision == "fp16":
                    load_args["torch_dtype"] = torch.float16
                elif precision == "bf16":
                    load_args["torch_dtype"] = torch.bfloat16
                elif precision == "fp32":
                    load_args["torch_dtype"] = torch.float32
            
            # Load pipeline
            load_result = self.pipeline_manager.load_custom_pipeline(
                model_path, pipeline_class, **load_args
            )
            
            if load_result.status != PipelineLoadStatus.SUCCESS:
                # For WAN models, try direct loading with DiffusionPipeline as fallback
                if pipeline_class == "WanPipeline":
                    self.logger.warning(f"WAN pipeline loading failed, trying direct DiffusionPipeline loading: {load_result.error_message}")
                    try:
                        from diffusers import DiffusionPipeline
                        # Remove trust_remote_code from load_args to avoid duplication
                        fallback_args = {k: v for k, v in load_args.items() if k != 'trust_remote_code'}
                        pipeline = DiffusionPipeline.from_pretrained(model_path, trust_remote_code=True, **fallback_args)
                        self.logger.info(f"Successfully loaded WAN model with DiffusionPipeline: {type(pipeline).__name__}")
                        
                        # Create a successful load result
                        load_result = PipelineLoadResult(
                            status=PipelineLoadStatus.SUCCESS,
                            pipeline=pipeline,
                            pipeline_class=type(pipeline).__name__,
                            warnings=[f"Used fallback loading method for WAN model"]
                        )
                    except Exception as fallback_error:
                        raise ValueError(f"Failed to load WAN pipeline with both methods. Original error: {load_result.error_message}. Fallback error: {str(fallback_error)}")
                else:
                    raise ValueError(f"Failed to load pipeline: {load_result.error_message}")
            
            pipeline = load_result.pipeline
            
            # Step 3: Apply optimizations
            optimization_result = OptimizationResult(
                success=True,
                applied_optimizations=[],
                final_vram_usage_mb=0,
                performance_impact=0.0,
                errors=[],
                warnings=[]
            )
            
            if apply_optimizations:
                self.logger.info("Applying optimizations...")
                optimization_result = self._apply_optimizations(
                    pipeline, model_architecture, optimization_config
                )
                
                if not optimization_result.success:
                    self.logger.warning(f"Optimization failed: {optimization_result.errors}")
                    # Continue with unoptimized pipeline
            
            # Step 4: Create chunked processor if needed
            chunked_processor = None
            if model_architecture.requirements.supports_cpu_offload:
                # Create chunked processor for memory-constrained systems
                chunk_size = 8
                if optimization_config and "chunk_size" in optimization_config:
                    chunk_size = optimization_config["chunk_size"]
                
                chunked_processor = ChunkedProcessor(chunk_size=chunk_size)
                self.logger.info(f"Created chunked processor with chunk size: {chunk_size}")
            
            # Step 5: Create wrapper
            wrapper = WanPipelineWrapper(
                pipeline=pipeline,
                model_architecture=model_architecture,
                optimization_result=optimization_result,
                chunked_processor=chunked_processor
            )
            
            # Cache the wrapper
            if self._pipeline_cache is not None:
                self._pipeline_cache[cache_key] = wrapper

            
            self.logger.info("Pipeline loaded successfully")
            return wrapper
            
        except Exception as e:
            self.logger.error(f"Failed to load pipeline: {e}")
            raise
    
    def _apply_optimizations(self, 
                           pipeline: Any,
                           model_architecture: ModelArchitecture,
                           optimization_config: Optional[Dict[str, Any]]) -> OptimizationResult:
        """Apply optimizations to the loaded pipeline with WAN22 components."""
        try:
            # Get system resources
            if self._system_resources is None:
                self._system_resources = self.optimization_manager.analyze_system_resources()
            
            # Initialize optimization result
            optimization_result = OptimizationResult(
                success=True,
                applied_optimizations=[],
                final_vram_usage_mb=0,
                performance_impact=0.0,
                errors=[],
                warnings=[]
            )
            
            # Step 1: VRAM Management Integration
            self.logger.info("Applying VRAM management optimizations...")
            try:
                # Detect available GPUs
                detected_gpus = self.vram_manager.detect_gpus()
                if detected_gpus:
                    primary_gpu = detected_gpus[0]
                    self.logger.info(f"Primary GPU detected: {primary_gpu.name} ({primary_gpu.total_memory_mb}MB)")
                    
                    # Get current VRAM usage
                    vram_usage = self.vram_manager.get_vram_usage(primary_gpu.index)
                    if vram_usage:
                        optimization_result.final_vram_usage_mb = vram_usage.used_mb
                        
                        # Apply VRAM optimizations if usage is high
                        if vram_usage.usage_percent > 0.8:  # 80% threshold
                            self.vram_manager.apply_memory_optimizations(primary_gpu.index)
                            optimization_result.applied_optimizations.append("VRAM memory optimization")
                            self.logger.info("Applied VRAM memory optimizations due to high usage")
                    
                    optimization_result.applied_optimizations.append("VRAM management integration")
                else:
                    optimization_result.warnings.append("No GPUs detected for VRAM management")
                    
            except Exception as e:
                self.logger.warning(f"VRAM management integration failed: {e}")
                optimization_result.warnings.append(f"VRAM management failed: {str(e)}")
            
            # Step 2: Quantization Integration
            self.logger.info("Applying quantization optimizations...")
            try:
                # Create hardware profile for quantization
                if detected_gpus:
                    gpu = detected_gpus[0]
                    hardware_profile = QuantHardwareProfile(
                        gpu_model=gpu.name,
                        vram_gb=gpu.total_memory_mb // 1024,
                        cuda_version=gpu.cuda_version or "unknown",
                        driver_version=gpu.driver_version,
                        compute_capability=(7, 5),  # Default, could be detected
                        supports_bf16=True,  # Could be detected based on GPU
                        supports_fp8=False,  # Experimental
                        supports_int8=True
                    )
                    
                    # Create model info
                    model_info = QuantModelInfo(
                        name=model_architecture.model_name,
                        size_gb=getattr(model_architecture.requirements, 'model_size_mb', 8192) / 1024,
                        architecture=model_architecture.architecture_type.value,
                        components=["unet", "vae", "text_encoder"],
                        estimated_vram_usage=model_architecture.requirements.recommended_vram_mb / 1024
                    )
                    
                    # Determine quantization strategy
                    strategy = self.quantization_controller.determine_optimal_quantization(
                        model_info, hardware_profile
                    )
                    
                    # Apply quantization if strategy suggests it
                    if strategy.method != QuantizationMethod.NONE:
                        # Check user preferences
                        user_prefs = optimization_config.get("quantization", {}) if optimization_config else {}
                        if user_prefs.get("enabled", True):  # Default to enabled
                            quant_result = self.quantization_controller.apply_quantization_with_timeout(
                                pipeline, strategy
                            )
                            
                            if quant_result.success:
                                optimization_result.applied_optimizations.append(f"Quantization ({quant_result.method_used.value})")
                                optimization_result.final_vram_usage_mb -= quant_result.memory_saved_mb
                                self.logger.info(f"Applied {quant_result.method_used.value} quantization, saved {quant_result.memory_saved_mb}MB")
                            else:
                                optimization_result.warnings.extend(quant_result.warnings)
                                optimization_result.errors.extend(quant_result.errors)
                        else:
                            optimization_result.applied_optimizations.append("Quantization (disabled by user)")
                    else:
                        optimization_result.applied_optimizations.append("Quantization (not recommended)")
                        
            except Exception as e:
                self.logger.warning(f"Quantization integration failed: {e}")
                optimization_result.warnings.append(f"Quantization failed: {str(e)}")
            
            # Step 3: Traditional Optimizations
            self.logger.info("Applying traditional optimizations...")
            try:
                # Create model requirements from architecture
                model_requirements = ModelRequirements(
                    min_vram_mb=model_architecture.requirements.min_vram_mb,
                    recommended_vram_mb=model_architecture.requirements.recommended_vram_mb,
                    model_size_mb=getattr(model_architecture.requirements, 'model_size_mb', 8192),
                    supports_mixed_precision=model_architecture.requirements.supports_mixed_precision,
                    supports_cpu_offload=model_architecture.requirements.supports_cpu_offload,
                    supports_chunked_processing=True,  # Wan models support chunked processing
                    component_sizes={}  # Could be populated from component analysis
                )
                
                # Override with user configuration
                if optimization_config:
                    if "min_vram_mb" in optimization_config:
                        model_requirements.min_vram_mb = optimization_config["min_vram_mb"]
                    if "recommended_vram_mb" in optimization_config:
                        model_requirements.recommended_vram_mb = optimization_config["recommended_vram_mb"]
                
                # Get optimization plan
                optimization_plan = self.optimization_manager.recommend_optimizations(
                    model_requirements, self._system_resources
                )
                
                # Apply traditional optimizations
                traditional_result = self.optimization_manager.apply_memory_optimizations(
                    pipeline, optimization_plan
                )
                
                # Merge results
                optimization_result.applied_optimizations.extend(traditional_result.applied_optimizations)
                optimization_result.warnings.extend(traditional_result.warnings)
                optimization_result.errors.extend(traditional_result.errors)
                optimization_result.performance_impact += traditional_result.performance_impact
                
                if not traditional_result.success:
                    optimization_result.success = False
                    
            except Exception as e:
                self.logger.warning(f"Traditional optimization failed: {e}")
                optimization_result.warnings.append(f"Traditional optimization failed: {str(e)}")
            
            # Final validation
            if optimization_result.errors:
                optimization_result.success = False
            
            self.logger.info(f"Optimization completed: {len(optimization_result.applied_optimizations)} optimizations applied")
            return optimization_result
            
        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            return OptimizationResult(
                success=False,
                applied_optimizations=[],
                final_vram_usage_mb=0,
                performance_impact=0.0,
                errors=[f"Optimization failed: {str(e)}"],
                warnings=[]
            )
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information and capabilities."""
        if self._system_resources is None:
            self._system_resources = self.optimization_manager.analyze_system_resources()
        
        return {
            "system_resources": self._system_resources.__dict__,
            "optimization_config": self.optimization_manager.config,
            "cache_enabled": self.enable_caching,
            "cached_pipelines": len(self._pipeline_cache) if self._pipeline_cache else 0
        }
    
    def clear_cache(self):
        """Clear the pipeline cache."""
        if self._pipeline_cache:
            self._pipeline_cache.clear()
            self.logger.info("Pipeline cache cleared")
    
    def preload_pipeline(self, 
                        model_path: str,
                        **kwargs) -> bool:
        """
        Preload a pipeline into cache.
        
        Args:
            model_path: Path to the model
            **kwargs: Pipeline loading arguments
            
        Returns:
            True if preloading succeeded, False otherwise
        """
        try:
            self.load_wan_pipeline(model_path, **kwargs)
            return True
        except Exception as e:
            self.logger.error(f"Failed to preload pipeline: {e}")
            return False
    
    def get_vram_status(self) -> Dict[str, Any]:
        """Get current VRAM status from the VRAM manager."""
        try:
            detected_gpus = self.vram_manager.detect_gpus()
            if not detected_gpus:
                return {"error": "No GPUs detected"}
            
            gpu_status = []
            for gpu in detected_gpus:
                usage = self.vram_manager.get_vram_usage(gpu.index)
                gpu_status.append({
                    "index": gpu.index,
                    "name": gpu.name,
                    "total_mb": gpu.total_memory_mb,
                    "used_mb": usage.used_mb if usage else 0,
                    "free_mb": usage.free_mb if usage else gpu.total_memory_mb,
                    "usage_percent": usage.usage_percent if usage else 0.0,
                    "temperature": gpu.temperature,
                    "utilization": gpu.utilization
                })
            
            return {
                "gpus": gpu_status,
                "monitoring_active": self.vram_manager.monitoring_active
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get VRAM status: {e}")
            return {"error": str(e)}
    
    def get_quantization_status(self) -> Dict[str, Any]:
        """Get current quantization controller status."""
        try:
            return {
                "available_methods": [method.value for method in QuantizationMethod],
                "user_preferences": self.quantization_controller.get_user_preferences(),
                "active_operations": len(self.quantization_controller._active_operations),
                "history": self.quantization_controller.get_quantization_history()[-5:]  # Last 5 operations
            }
        except Exception as e:
            self.logger.error(f"Failed to get quantization status: {e}")
            return {"error": str(e)}
    
    def optimize_for_model(self, model_path: str, target_vram_gb: Optional[float] = None) -> Dict[str, Any]:
        """
        Get optimization recommendations for a specific model.
        
        Args:
            model_path: Path to the model
            target_vram_gb: Target VRAM usage in GB
            
        Returns:
            Dictionary with optimization recommendations
        """
        try:
            # Detect model architecture
            model_architecture = self.architecture_detector.detect_model_architecture(model_path)
            
            # Get VRAM status
            vram_status = self.get_vram_status()
            if "error" in vram_status:
                return {"error": f"VRAM detection failed: {vram_status['error']}"}
            
            primary_gpu = vram_status["gpus"][0] if vram_status["gpus"] else None
            if not primary_gpu:
                return {"error": "No primary GPU available"}
            
            # Calculate recommendations
            available_vram_gb = primary_gpu["free_mb"] / 1024
            required_vram_gb = model_architecture.requirements.recommended_vram_mb / 1024
            
            recommendations = {
                "model_info": {
                    "name": model_architecture.model_name,
                    "architecture": model_architecture.architecture_type.value,
                    "required_vram_gb": required_vram_gb
                },
                "system_info": {
                    "available_vram_gb": available_vram_gb,
                    "total_vram_gb": primary_gpu["total_mb"] / 1024,
                    "gpu_name": primary_gpu["name"]
                },
                "recommendations": []
            }
            
            # Generate recommendations based on VRAM availability
            if target_vram_gb:
                target_available = target_vram_gb
            else:
                target_available = available_vram_gb
            
            if required_vram_gb > target_available:
                # Need optimizations
                vram_deficit = required_vram_gb - target_available
                
                recommendations["recommendations"].append({
                    "type": "quantization",
                    "reason": f"Model requires {required_vram_gb:.1f}GB but only {target_available:.1f}GB available",
                    "suggested_method": "bf16" if vram_deficit < 2 else "int8",
                    "estimated_savings_gb": vram_deficit * 0.5  # Rough estimate
                })
                
                if vram_deficit > 4:
                    recommendations["recommendations"].append({
                        "type": "cpu_offload",
                        "reason": "Large VRAM deficit detected",
                        "components": ["text_encoder", "vae"],
                        "estimated_savings_gb": 2.0
                    })
                
                recommendations["recommendations"].append({
                    "type": "chunked_processing",
                    "reason": "Reduce peak memory usage during generation",
                    "suggested_chunk_size": 4 if vram_deficit > 2 else 8
                })
            else:
                recommendations["recommendations"].append({
                    "type": "no_optimization",
                    "reason": f"Sufficient VRAM available ({available_vram_gb:.1f}GB >= {required_vram_gb:.1f}GB)"
                })
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Failed to generate optimization recommendations: {e}")
            return {"error": str(e)}