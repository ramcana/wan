"""
Optimization and Resource Management System for Wan Model Compatibility

This module implements comprehensive optimization strategies for memory-constrained systems,
including mixed precision, CPU offloading, and chunked processing capabilities.

Requirements addressed: 5.1, 5.2, 5.3, 5.4, 5.5
"""

import torch
import psutil
import gc
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
import json
import time

logger = logging.getLogger(__name__)


@dataclass
class SystemResources:
    """System resource information"""
    total_vram_mb: int
    available_vram_mb: int
    total_ram_mb: int
    available_ram_mb: int
    gpu_name: str
    gpu_compute_capability: Optional[Tuple[int, int]]
    cpu_cores: int
    supports_mixed_precision: bool
    supports_cpu_offload: bool


@dataclass
class ModelRequirements:
    """Model resource requirements"""
    min_vram_mb: int
    recommended_vram_mb: int
    model_size_mb: int
    supports_mixed_precision: bool
    supports_cpu_offload: bool
    supports_chunked_processing: bool
    component_sizes: Dict[str, int]  # Component name -> size in MB


@dataclass
class OptimizationPlan:
    """Optimization strategy plan"""
    use_mixed_precision: bool
    precision_type: str  # "fp16", "bf16", "fp32"
    enable_cpu_offload: bool
    offload_strategy: str  # "sequential", "model", "full"
    chunk_frames: bool
    max_chunk_size: int
    estimated_vram_reduction: float
    estimated_performance_impact: float
    optimization_steps: List[str]
    warnings: List[str]


@dataclass
class OptimizationResult:
    """Result of applying optimizations"""
    success: bool
    applied_optimizations: List[str]
    final_vram_usage_mb: int
    performance_impact: float
    errors: List[str]
    warnings: List[str]


class OptimizationManager:
    """
    Manages system resource analysis and optimization strategies for Wan models.
    
    Provides automatic optimization recommendations based on available VRAM,
    implements memory optimization strategies, and handles chunked processing
    for memory-constrained systems.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize OptimizationManager.
        
        Args:
            config_path: Optional path to optimization configuration file
        """
        self.config = self._load_config(config_path)
        self.system_resources = None
        self._optimization_cache = {}
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load optimization configuration"""
        default_config = {
            "vram_safety_margin_mb": 1024,  # Reserve 1GB for system
            "max_chunk_size": 8,  # Maximum frames per chunk
            "mixed_precision_threshold_mb": 8192,  # Use mixed precision below 8GB
            "cpu_offload_threshold_mb": 6144,  # Use CPU offload below 6GB
            "performance_priority": "balanced",  # "speed", "memory", "balanced"
            "enable_aggressive_optimization": False
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
                
        return default_config
    
    def analyze_system_resources(self) -> SystemResources:
        """
        Analyze available system resources.
        
        Returns:
            SystemResources object with current system capabilities
        """
        try:
            # GPU information
            if torch.cuda.is_available():
                gpu_props = torch.cuda.get_device_properties(0)
                total_vram = gpu_props.total_memory // (1024 * 1024)  # Convert to MB
                
                # Get available VRAM
                torch.cuda.empty_cache()
                available_vram = (gpu_props.total_memory - torch.cuda.memory_allocated()) // (1024 * 1024)
                
                gpu_name = gpu_props.name
                compute_capability = (gpu_props.major, gpu_props.minor)
                
                # Check mixed precision support
                supports_mixed_precision = compute_capability >= (7, 0)  # Volta and newer
            else:
                total_vram = 0
                available_vram = 0
                gpu_name = "No GPU"
                compute_capability = None
                supports_mixed_precision = False
            
            # RAM information
            memory = psutil.virtual_memory()
            total_ram = memory.total // (1024 * 1024)
            available_ram = memory.available // (1024 * 1024)
            
            # CPU information
            cpu_cores = psutil.cpu_count(logical=False)
            
            # CPU offload support (always available)
            supports_cpu_offload = True
            
            self.system_resources = SystemResources(
                total_vram_mb=total_vram,
                available_vram_mb=available_vram,
                total_ram_mb=total_ram,
                available_ram_mb=available_ram,
                gpu_name=gpu_name,
                gpu_compute_capability=compute_capability,
                cpu_cores=cpu_cores,
                supports_mixed_precision=supports_mixed_precision,
                supports_cpu_offload=supports_cpu_offload
            )
            
            logger.info(f"System resources analyzed: {total_vram}MB VRAM, {total_ram}MB RAM, {cpu_cores} CPU cores")
            return self.system_resources
            
        except Exception as e:
            logger.error(f"Failed to analyze system resources: {e}")
            # Return minimal fallback resources
            return SystemResources(
                total_vram_mb=0,
                available_vram_mb=0,
                total_ram_mb=8192,
                available_ram_mb=4096,
                gpu_name="Unknown",
                gpu_compute_capability=None,
                cpu_cores=4,
                supports_mixed_precision=False,
                supports_cpu_offload=True
            )
    
    def recommend_optimizations(self, 
                               model_requirements: ModelRequirements,
                               system_resources: Optional[SystemResources] = None) -> OptimizationPlan:
        """
        Recommend optimization strategies based on model requirements and system resources.
        
        Args:
            model_requirements: Model resource requirements
            system_resources: System resources (will analyze if not provided)
            
        Returns:
            OptimizationPlan with recommended strategies
        """
        if system_resources is None:
            system_resources = self.analyze_system_resources()
        
        # Calculate available VRAM with safety margin
        safety_margin = self.config["vram_safety_margin_mb"]
        effective_vram = max(0, system_resources.available_vram_mb - safety_margin)
        
        # Initialize optimization plan
        plan = OptimizationPlan(
            use_mixed_precision=False,
            precision_type="fp32",
            enable_cpu_offload=False,
            offload_strategy="none",
            chunk_frames=False,
            max_chunk_size=1,
            estimated_vram_reduction=0.0,
            estimated_performance_impact=0.0,
            optimization_steps=[],
            warnings=[]
        )
        
        # Check if model fits without optimization
        if effective_vram >= model_requirements.recommended_vram_mb:
            plan.optimization_steps.append("No optimization needed - sufficient VRAM available")
            return plan
        
        vram_multiplier = 1.0  # Start with no reduction
        performance_impact = 0.0
        
        # Mixed precision optimization
        if (system_resources.supports_mixed_precision and 
            model_requirements.supports_mixed_precision and
            effective_vram < self.config["mixed_precision_threshold_mb"]):
            
            plan.use_mixed_precision = True
            
            # Choose precision type based on GPU capability
            if system_resources.gpu_compute_capability and system_resources.gpu_compute_capability >= (8, 0):
                plan.precision_type = "bf16"  # Ampere and newer support bfloat16
                vram_multiplier *= 0.6  # ~40% VRAM reduction (multiply by 0.6)
                performance_impact += 0.1  # ~10% performance improvement
            else:
                plan.precision_type = "fp16"
                vram_multiplier *= 0.65  # ~35% VRAM reduction (multiply by 0.65)
                performance_impact += 0.05  # ~5% performance improvement
                
            plan.optimization_steps.append(f"Enable {plan.precision_type} mixed precision")
        
        # CPU offload optimization
        current_vram_need = model_requirements.recommended_vram_mb * vram_multiplier
        if (system_resources.supports_cpu_offload and 
            model_requirements.supports_cpu_offload and
            effective_vram < current_vram_need):
            
            plan.enable_cpu_offload = True
            
            # Determine offload strategy based on available resources
            if effective_vram < self.config["cpu_offload_threshold_mb"]:
                plan.offload_strategy = "full"
                vram_multiplier *= 0.4  # ~60% additional VRAM reduction
                performance_impact -= 0.3  # ~30% performance penalty
                plan.optimization_steps.append("Enable full CPU offloading")
            elif effective_vram < model_requirements.min_vram_mb:
                plan.offload_strategy = "model"
                vram_multiplier *= 0.6  # ~40% additional VRAM reduction
                performance_impact -= 0.2  # ~20% performance penalty
                plan.optimization_steps.append("Enable model CPU offloading")
            else:
                plan.offload_strategy = "sequential"
                vram_multiplier *= 0.8  # ~20% additional VRAM reduction
                performance_impact -= 0.1  # ~10% performance penalty
                plan.optimization_steps.append("Enable sequential CPU offloading")
        
        # Chunked processing optimization
        current_vram_need = model_requirements.recommended_vram_mb * vram_multiplier
        if (model_requirements.supports_chunked_processing and
            effective_vram < current_vram_need):
            
            plan.chunk_frames = True
            
            # Calculate optimal chunk size
            available_for_chunks = effective_vram
            estimated_per_frame_mb = current_vram_need / 16  # Assume 16 frames baseline
            
            if estimated_per_frame_mb > 0:
                optimal_chunk_size = max(1, min(
                    self.config["max_chunk_size"],
                    int(available_for_chunks / estimated_per_frame_mb)
                ))
            else:
                optimal_chunk_size = 1
            
            plan.max_chunk_size = optimal_chunk_size
            
            # Chunked processing reduces peak VRAM but increases total time
            chunk_vram_reduction = 1 - (optimal_chunk_size / 16)
            vram_multiplier *= (1 - chunk_vram_reduction * 0.3)  # Partial reduction due to overhead
            performance_impact -= chunk_vram_reduction * 0.5  # Significant time increase
            
            plan.optimization_steps.append(f"Enable chunked processing ({optimal_chunk_size} frames per chunk)")
        
        # Final calculations
        plan.estimated_vram_reduction = 1 - vram_multiplier  # Convert multiplier back to reduction percentage
        plan.estimated_performance_impact = performance_impact
        
        # Add warnings for significant performance impacts
        if performance_impact < -0.2:
            plan.warnings.append("Significant performance reduction expected due to aggressive optimization")
        
        if plan.chunk_frames and plan.max_chunk_size == 1:
            plan.warnings.append("Frame-by-frame processing will be very slow")
        
        # Check if optimizations are sufficient
        final_vram_need = model_requirements.recommended_vram_mb * (1 - plan.estimated_vram_reduction)
        if final_vram_need > effective_vram:
            plan.warnings.append(f"Even with optimizations, model may require {final_vram_need:.0f}MB but only {effective_vram:.0f}MB available")
        
        return plan
    
    def apply_memory_optimizations(self, 
                                  pipeline: Any, 
                                  plan: OptimizationPlan) -> OptimizationResult:
        """
        Apply memory optimization strategies to a pipeline.
        
        Args:
            pipeline: The pipeline to optimize
            plan: Optimization plan to apply
            
        Returns:
            OptimizationResult with applied optimizations and results
        """
        applied_optimizations = []
        errors = []
        warnings = []
        
        try:
            # Apply mixed precision
            if plan.use_mixed_precision:
                try:
                    if hasattr(pipeline, 'to'):
                        if plan.precision_type == "fp16":
                            pipeline = pipeline.to(torch.float16)
                        elif plan.precision_type == "bf16":
                            pipeline = pipeline.to(torch.bfloat16)
                        applied_optimizations.append(f"Mixed precision ({plan.precision_type})")
                    else:
                        warnings.append("Pipeline does not support precision conversion")
                except Exception as e:
                    errors.append(f"Failed to apply mixed precision: {e}")
            
            # Apply CPU offloading
            if plan.enable_cpu_offload:
                try:
                    if hasattr(pipeline, 'enable_model_cpu_offload'):
                        if plan.offload_strategy == "full":
                            pipeline.enable_model_cpu_offload()
                            applied_optimizations.append("Full CPU offloading")
                        elif plan.offload_strategy == "model":
                            pipeline.enable_model_cpu_offload()
                            applied_optimizations.append("Model CPU offloading")
                        elif plan.offload_strategy == "sequential":
                            if hasattr(pipeline, 'enable_sequential_cpu_offload'):
                                pipeline.enable_sequential_cpu_offload()
                                applied_optimizations.append("Sequential CPU offloading")
                            else:
                                pipeline.enable_model_cpu_offload()
                                applied_optimizations.append("Model CPU offloading (fallback)")
                    else:
                        warnings.append("Pipeline does not support CPU offloading")
                except Exception as e:
                    errors.append(f"Failed to apply CPU offloading: {e}")
            
            # Memory cleanup
            torch.cuda.empty_cache()
            gc.collect()
            
            # Measure final VRAM usage
            final_vram_usage = 0
            if torch.cuda.is_available():
                final_vram_usage = torch.cuda.memory_allocated() // (1024 * 1024)
            
            # Calculate actual performance impact (simplified estimation)
            performance_impact = plan.estimated_performance_impact
            
            return OptimizationResult(
                success=len(errors) == 0,
                applied_optimizations=applied_optimizations,
                final_vram_usage_mb=final_vram_usage,
                performance_impact=performance_impact,
                errors=errors,
                warnings=warnings
            )
            
        except Exception as e:
            logger.error(f"Failed to apply optimizations: {e}")
            return OptimizationResult(
                success=False,
                applied_optimizations=applied_optimizations,
                final_vram_usage_mb=0,
                performance_impact=0.0,
                errors=[f"Optimization failed: {e}"],
                warnings=warnings
            )
    
    def enable_chunked_processing(self, 
                                 pipeline: Any, 
                                 chunk_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Enable chunked processing for memory-constrained systems.
        
        Args:
            pipeline: Pipeline to configure for chunked processing
            chunk_size: Number of frames per chunk (auto-calculated if None)
            
        Returns:
            Dictionary with chunked processing configuration
        """
        if chunk_size is None:
            chunk_size = self.config["max_chunk_size"]
        
        # Create chunked processing configuration
        chunked_config = {
            "enabled": True,
            "chunk_size": chunk_size,
            "overlap_frames": 1,  # Overlap for smooth transitions
            "memory_cleanup": True,  # Clean memory between chunks
            "progress_callback": None  # Can be set by caller
        }
        
        # Store configuration in pipeline if possible
        if hasattr(pipeline, '_chunked_config'):
            pipeline._chunked_config = chunked_config
        
        logger.info(f"Chunked processing enabled: {chunk_size} frames per chunk")
        return chunked_config
    
    def estimate_memory_usage(self, 
                             model_requirements: ModelRequirements,
                             generation_params: Dict[str, Any]) -> Dict[str, int]:
        """
        Estimate memory usage for given model and generation parameters.
        
        Args:
            model_requirements: Model resource requirements
            generation_params: Generation parameters (resolution, frames, etc.)
            
        Returns:
            Dictionary with memory usage estimates in MB
        """
        # Base model memory
        base_memory = model_requirements.model_size_mb
        
        # Extract generation parameters
        width = generation_params.get('width', 512)
        height = generation_params.get('height', 512)
        num_frames = generation_params.get('num_frames', 16)
        batch_size = generation_params.get('batch_size', 1)
        
        # Calculate additional memory for generation
        # Rough estimation based on tensor sizes
        pixel_count = width * height * num_frames * batch_size
        
        # Memory for intermediate tensors (rough estimation)
        # This is a simplified calculation - real usage depends on model architecture
        intermediate_memory = (pixel_count * 4 * 3) // (1024 * 1024)  # 4 bytes per float, 3 channels
        
        # Memory for output tensors
        output_memory = (pixel_count * 4 * 3) // (1024 * 1024)
        
        # Additional overhead (activations, gradients, etc.)
        overhead_memory = int(base_memory * 0.3)  # 30% overhead
        
        total_memory = base_memory + intermediate_memory + output_memory + overhead_memory
        
        return {
            "base_model_mb": base_memory,
            "intermediate_tensors_mb": intermediate_memory,
            "output_tensors_mb": output_memory,
            "overhead_mb": overhead_memory,
            "total_estimated_mb": total_memory
        }
    
    def monitor_memory_usage(self) -> Dict[str, float]:
        """
        Monitor current memory usage.
        
        Returns:
            Dictionary with current memory usage statistics
        """
        stats = {}
        
        # GPU memory
        if torch.cuda.is_available():
            stats["gpu_allocated_mb"] = torch.cuda.memory_allocated() // (1024 * 1024)
            stats["gpu_reserved_mb"] = torch.cuda.memory_reserved() // (1024 * 1024)
            stats["gpu_max_allocated_mb"] = torch.cuda.max_memory_allocated() // (1024 * 1024)
            
            # Calculate utilization
            total_vram = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
            stats["gpu_utilization_percent"] = (stats["gpu_allocated_mb"] / total_vram) * 100
        else:
            stats["gpu_allocated_mb"] = 0
            stats["gpu_reserved_mb"] = 0
            stats["gpu_max_allocated_mb"] = 0
            stats["gpu_utilization_percent"] = 0
        
        # System memory
        memory = psutil.virtual_memory()
        stats["system_used_mb"] = (memory.total - memory.available) // (1024 * 1024)
        stats["system_available_mb"] = memory.available // (1024 * 1024)
        stats["system_utilization_percent"] = memory.percent
        
        return stats
    
    def cleanup_memory(self):
        """Clean up GPU and system memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        
    def get_optimization_recommendations(self, 
                                       model_path: str,
                                       generation_params: Dict[str, Any]) -> List[str]:
        """
        Get optimization recommendations for specific model and parameters.
        
        Args:
            model_path: Path to the model
            generation_params: Generation parameters
            
        Returns:
            List of optimization recommendations
        """
        recommendations = []
        
        # Analyze current system state
        system_resources = self.analyze_system_resources()
        
        # Create mock model requirements (in real implementation, this would be detected)
        model_requirements = ModelRequirements(
            min_vram_mb=6144,
            recommended_vram_mb=12288,
            model_size_mb=8192,
            supports_mixed_precision=True,
            supports_cpu_offload=True,
            supports_chunked_processing=True,
            component_sizes={"transformer": 4096, "vae": 2048, "text_encoder": 1024}
        )
        
        # Get optimization plan
        plan = self.recommend_optimizations(model_requirements, system_resources)
        
        # Convert plan to recommendations
        if plan.use_mixed_precision:
            recommendations.append(f"Enable {plan.precision_type} mixed precision to reduce VRAM usage by ~{plan.estimated_vram_reduction*35:.0f}%")
        
        if plan.enable_cpu_offload:
            recommendations.append(f"Enable {plan.offload_strategy} CPU offloading to reduce VRAM usage")
        
        if plan.chunk_frames:
            recommendations.append(f"Use chunked processing with {plan.max_chunk_size} frames per chunk")
        
        # Add parameter-specific recommendations
        width = generation_params.get('width', 512)
        height = generation_params.get('height', 512)
        num_frames = generation_params.get('num_frames', 16)
        
        if width * height > 512 * 512:
            recommendations.append("Consider reducing resolution to save VRAM")
        
        if num_frames > 16:
            recommendations.append("Consider reducing frame count for faster generation")
        
        # Add warnings
        for warning in plan.warnings:
            recommendations.append(f"⚠️ {warning}")
        
        return recommendations


class ChunkedProcessor:
    """
    Handles chunked processing for memory-constrained video generation.
    """
    
    def __init__(self, chunk_size: int = 8, overlap_frames: int = 1):
        """
        Initialize ChunkedProcessor.
        
        Args:
            chunk_size: Number of frames per chunk
            overlap_frames: Number of overlapping frames between chunks
        """
        self.chunk_size = chunk_size
        self.overlap_frames = overlap_frames
        self.logger = logging.getLogger(__name__ + ".ChunkedProcessor")
    
    def process_chunked_generation(self, 
                                  pipeline: Any,
                                  prompt: str,
                                  num_frames: int,
                                  **generation_kwargs) -> List[torch.Tensor]:
        """
        Process video generation in chunks.
        
        Args:
            pipeline: The generation pipeline
            prompt: Text prompt
            num_frames: Total number of frames to generate
            **generation_kwargs: Additional generation parameters
            
        Returns:
            List of generated frame tensors
        """
        if num_frames <= self.chunk_size:
            # No chunking needed
            return self._generate_single_chunk(pipeline, prompt, num_frames, **generation_kwargs)
        
        # Calculate chunks
        chunks = self._calculate_chunks(num_frames)
        all_frames = []
        
        self.logger.info(f"Processing {len(chunks)} chunks for {num_frames} frames")
        
        for i, (start_frame, end_frame) in enumerate(chunks):
            chunk_frames = end_frame - start_frame
            
            self.logger.info(f"Processing chunk {i+1}/{len(chunks)}: frames {start_frame}-{end_frame}")
            
            # Generate chunk
            chunk_result = self._generate_single_chunk(
                pipeline, prompt, chunk_frames, **generation_kwargs
            )
            
            # Handle overlap removal (except for first chunk)
            if i > 0 and self.overlap_frames > 0:
                chunk_result = chunk_result[self.overlap_frames:]
            
            all_frames.extend(chunk_result)
            
            # Memory cleanup between chunks
            torch.cuda.empty_cache()
            gc.collect()
        
        return all_frames
    
    def _calculate_chunks(self, num_frames: int) -> List[Tuple[int, int]]:
        """Calculate chunk boundaries with overlap"""
        chunks = []
        current_frame = 0
        
        while current_frame < num_frames:
            end_frame = min(current_frame + self.chunk_size, num_frames)
            
            # Add overlap for smooth transitions (except last chunk)
            if end_frame < num_frames:
                end_frame = min(end_frame + self.overlap_frames, num_frames)
            
            chunks.append((current_frame, end_frame))
            
            # Move to next chunk, ensuring we make progress
            next_frame = end_frame - self.overlap_frames
            
            # If we've reached the end, break
            if end_frame >= num_frames:
                break
                
            # Ensure we always make progress to prevent infinite loop
            if next_frame <= current_frame:
                next_frame = current_frame + max(1, self.chunk_size - self.overlap_frames)
            
            current_frame = next_frame
            
            # Safety check to prevent infinite loop
            if len(chunks) > num_frames:
                break
        
        return chunks
    
    def _generate_single_chunk(self, 
                              pipeline: Any,
                              prompt: str,
                              num_frames: int,
                              **generation_kwargs) -> List[torch.Tensor]:
        """Generate a single chunk of frames"""
        try:
            # Update generation parameters for chunk
            chunk_kwargs = generation_kwargs.copy()
            chunk_kwargs['num_frames'] = num_frames
            
            # Generate frames
            with torch.no_grad():
                result = pipeline(prompt, **chunk_kwargs)
            
            # Extract frames from result
            if hasattr(result, 'frames'):
                frames = result.frames
            elif isinstance(result, torch.Tensor):
                frames = result
            elif isinstance(result, list):
                frames = result
            else:
                # Try to extract frames from various result formats
                frames = getattr(result, 'images', getattr(result, 'videos', result))
            
            # Ensure frames is a list of tensors
            if isinstance(frames, torch.Tensor):
                frames = [frames[i] for i in range(frames.shape[0])]
            
            return frames
            
        except Exception as e:
            self.logger.error(f"Failed to generate chunk: {e}")
            raise


# Utility functions for optimization
def get_gpu_memory_info() -> Dict[str, int]:
    """Get current GPU memory information"""
    if not torch.cuda.is_available():
        return {"total": 0, "allocated": 0, "reserved": 0, "free": 0}
    
    total = torch.cuda.get_device_properties(0).total_memory
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    free = total - allocated
    
    return {
        "total": total // (1024 * 1024),
        "allocated": allocated // (1024 * 1024),
        "reserved": reserved // (1024 * 1024),
        "free": free // (1024 * 1024)
    }


def estimate_model_vram_usage(model_size_mb: int, 
                             precision: str = "fp32",
                             batch_size: int = 1) -> int:
    """
    Estimate VRAM usage for a model.
    
    Args:
        model_size_mb: Model size in MB
        precision: Precision type ("fp32", "fp16", "bf16")
        batch_size: Batch size
        
    Returns:
        Estimated VRAM usage in MB
    """
    # Precision multipliers
    precision_multipliers = {
        "fp32": 1.0,
        "fp16": 0.5,
        "bf16": 0.5
    }
    
    multiplier = precision_multipliers.get(precision, 1.0)
    
    # Base model memory
    base_memory = model_size_mb * multiplier
    
    # Additional memory for activations, gradients, etc.
    # This is a rough estimation
    overhead = base_memory * 0.5 * batch_size
    
    return int(base_memory + overhead)
