"""
Generation Orchestrator Components for Wan2.2 Video Generation
Provides pre-flight validation, resource management, and pipeline routing
"""

import logging
import torch
import psutil
import gc
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple, Union
from enum import Enum
from datetime import datetime
from pathlib import Path
import json
import os

# Import existing validation components
from input_validation import ValidationResult, PromptValidator, ImageValidator, ConfigValidator
from backend.core.services.optimization_service import VRAMOptimizer, get_vram_optimizer
from resource_manager import ResourceStatus, get_resource_manager

logger = logging.getLogger(__name__)

class GenerationMode(Enum):
    """Supported generation modes"""
    TEXT_TO_VIDEO = "t2v-A14B"
    IMAGE_TO_VIDEO = "i2v-A14B"
    TEXT_IMAGE_TO_VIDEO = "ti2v-5B"

class ResourceStatus(Enum):
    """Resource availability status"""
    AVAILABLE = "available"
    LIMITED = "limited"
    INSUFFICIENT = "insufficient"

@dataclass
class ResourceEstimate:
    """Estimated resource requirements for generation"""
    vram_mb: float
    system_ram_mb: float
    estimated_time_seconds: float
    gpu_utilization_percent: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "vram_mb": self.vram_mb,
            "system_ram_mb": self.system_ram_mb,
            "estimated_time_seconds": self.estimated_time_seconds,
            "gpu_utilization_percent": self.gpu_utilization_percent
        }

@dataclass
class GenerationRequest:
    """Structured request for video generation with enhanced image support"""
    model_type: str
    prompt: str
    image: Optional[Any] = None  # Start frame image for I2V/TI2V
    end_image: Optional[Any] = None  # End frame image for I2V/TI2V
    resolution: str = "720p"
    steps: int = 50
    guidance_scale: float = 7.5
    strength: float = 0.8
    seed: int = -1
    fps: int = 24
    duration: int = 4
    lora_config: Dict[str, float] = field(default_factory=dict)
    optimization_settings: Dict[str, Any] = field(default_factory=dict)
    
    # Image metadata for better tracking and validation
    image_metadata: Optional[Dict[str, Any]] = None
    end_image_metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate and normalize request parameters"""
        # Normalize model type
        if self.model_type in ["t2v", "text-to-video"]:
            self.model_type = GenerationMode.TEXT_TO_VIDEO.value
        elif self.model_type in ["i2v", "image-to-video"]:
            self.model_type = GenerationMode.IMAGE_TO_VIDEO.value
        elif self.model_type in ["ti2v", "text-image-to-video"]:
            self.model_type = GenerationMode.TEXT_IMAGE_TO_VIDEO.value
        
        # Validate resolution
        valid_resolutions = ["720p", "1080p", "480p"]
        if self.resolution not in valid_resolutions:
            self.resolution = "720p"
        
        # Ensure positive values
        self.steps = max(1, self.steps)
        self.guidance_scale = max(0.1, self.guidance_scale)
        self.strength = max(0.1, min(1.0, self.strength))
        self.fps = max(1, self.fps)
        self.duration = max(1, self.duration)

@dataclass
class ModelStatus:
    """Status of model availability and loading"""
    is_available: bool
    is_loaded: bool
    model_path: Optional[str] = None
    loading_error: Optional[str] = None
    memory_usage_mb: float = 0.0
    compatibility_issues: List[str] = field(default_factory=list)

@dataclass
class PreflightResult:
    """Result of pre-flight checks before generation"""
    can_proceed: bool
    model_status: ModelStatus
    resource_estimate: ResourceEstimate
    optimization_recommendations: List[str] = field(default_factory=list)
    blocking_issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

@dataclass
class GenerationPipeline:
    """Configuration for a specific generation pipeline"""
    pipeline_type: str
    model_path: str
    config: Dict[str, Any]
    memory_requirements: ResourceEstimate
    supported_resolutions: List[str]
    optimization_flags: Dict[str, bool] = field(default_factory=dict)

@dataclass
class GenerationResult:
    """Result of a generation operation"""
    success: bool
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    generation_time: Optional[float] = None
    resource_usage: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class PreflightChecker:
    """Handles pre-generation validation and readiness checks"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.prompt_validator = PromptValidator()
        self.image_validator = ImageValidator()
        self.config_validator = ConfigValidator()
        
    def run_preflight_checks(self, generation_request: GenerationRequest) -> PreflightResult:
        """Run all pre-generation validation checks"""
        try:
            # Check model availability
            model_status = self.check_model_availability(generation_request.model_type)
            
            # Estimate resource requirements
            resource_estimate = self.estimate_resource_requirements(generation_request)
            
            # Collect optimization recommendations
            optimization_recommendations = self._get_optimization_recommendations(
                generation_request, resource_estimate
            )
            
            # Identify blocking issues
            blocking_issues = []
            warnings = []
            
            # Validate inputs
            validation_result = self._validate_inputs(generation_request)
            if not validation_result.is_valid:
                blocking_issues.append(f"Input validation failed: {validation_result.message}")
            
            # Check model status
            if not model_status.is_available:
                blocking_issues.append(f"Model not available: {model_status.loading_error}")
            
            # Check resource availability
            resource_manager = ResourceManager(self.config)
            if not resource_manager.check_vram_availability(int(resource_estimate.vram_mb)):
                blocking_issues.append("Insufficient VRAM for generation")
            
            can_proceed = len(blocking_issues) == 0
            
            return PreflightResult(
                can_proceed=can_proceed,
                model_status=model_status,
                resource_estimate=resource_estimate,
                optimization_recommendations=optimization_recommendations,
                blocking_issues=blocking_issues,
                warnings=warnings
            )
            
        except Exception as e:
            logger.error(f"Preflight checks failed: {e}")
            return PreflightResult(
                can_proceed=False,
                model_status=ModelStatus(is_available=False, is_loaded=False, loading_error=str(e)),
                resource_estimate=ResourceEstimate(0, 0, 0, 0),
                blocking_issues=[f"Preflight check error: {str(e)}"]
            )
    
    def check_model_availability(self, model_type: str) -> ModelStatus:
        """Verify model is available and loadable"""
        try:
            # Define model paths based on type
            model_paths = {
                GenerationMode.TEXT_TO_VIDEO.value: "models/Wan-AI_Wan2.2-T2V-A14B-Diffusers",
                GenerationMode.IMAGE_TO_VIDEO.value: "models/Wan-AI_Wan2.2-I2V-A14B-Diffusers", 
                GenerationMode.TEXT_IMAGE_TO_VIDEO.value: "models/Wan-AI_Wan2.2-TI2V-5B-Diffusers"
            }
            
            model_path = model_paths.get(model_type)
            if not model_path:
                return ModelStatus(
                    is_available=False,
                    is_loaded=False,
                    loading_error=f"Unknown model type: {model_type}"
                )
            
            # Check if model directory exists
            full_path = Path(model_path)
            if not full_path.exists():
                return ModelStatus(
                    is_available=False,
                    is_loaded=False,
                    model_path=str(full_path),
                    loading_error=f"Model directory not found: {full_path}"
                )
            
            # Check for required model files
            required_files = ["model_index.json", "scheduler", "unet"]
            missing_files = []
            for file in required_files:
                if not (full_path / file).exists():
                    missing_files.append(file)
            
            if missing_files:
                return ModelStatus(
                    is_available=False,
                    is_loaded=False,
                    model_path=str(full_path),
                    loading_error=f"Missing model files: {', '.join(missing_files)}"
                )
            
            return ModelStatus(
                is_available=True,
                is_loaded=False,  # Not loaded yet, just available
                model_path=str(full_path)
            )
            
        except Exception as e:
            logger.error(f"Model availability check failed: {e}")
            return ModelStatus(
                is_available=False,
                is_loaded=False,
                loading_error=str(e)
            )
    
    def estimate_resource_requirements(self, request: GenerationRequest) -> ResourceEstimate:
        """Estimate VRAM and processing requirements using enhanced resource manager"""
        try:
            # Use VRAMOptimizer for more accurate estimation
            vram_optimizer = get_vram_optimizer(self.config)
            
            requirement = vram_optimizer.estimate_resource_requirements(
                model_type=request.model_type,
                resolution=request.resolution,
                steps=request.steps,
                duration=request.duration,
                lora_count=len(request.lora_config) if request.lora_config else 0
            )
            
            # Convert to ResourceEstimate format for backward compatibility
            return ResourceEstimate(
                vram_mb=requirement.vram_mb,
                system_ram_mb=requirement.ram_mb,
                estimated_time_seconds=requirement.estimated_time_seconds,
                gpu_utilization_percent=min(95.0, 70.0 + ((request.steps / 50.0) * 20))
            )
            
        except Exception as e:
            logger.error(f"Resource estimation failed: {e}")
            return ResourceEstimate(8000, 4000, 120, 90.0)
    
    def _validate_inputs(self, request: GenerationRequest) -> ValidationResult:
        """Validate all inputs in the generation request"""
        # Validate prompt
        prompt_result = self.prompt_validator.validate(request.prompt)
        if not prompt_result.is_valid:
            return prompt_result
        
        # Validate image if provided
        if request.image is not None:
            image_result = self.image_validator.validate(request.image)
            if not image_result.is_valid:
                return image_result
        
        # Validate configuration
        config_data = {
            "resolution": request.resolution,
            "steps": request.steps,
            "guidance_scale": request.guidance_scale,
            "strength": request.strength,
            "fps": request.fps,
            "duration": request.duration
        }
        return self.config_validator.validate(config_data)
    
    def _get_optimization_recommendations(self, request: GenerationRequest, 
                                        estimate: ResourceEstimate) -> List[str]:
        """Generate optimization recommendations based on request and estimates"""
        recommendations = []
        
        # High VRAM usage recommendations
        if estimate.vram_mb > 8000:
            recommendations.append("Consider reducing steps or resolution to lower VRAM usage")
            
        if estimate.vram_mb > 12000:
            recommendations.append("Enable gradient checkpointing to reduce memory usage")
            
        # Long generation time recommendations
        if estimate.estimated_time_seconds > 180:
            recommendations.append("Consider reducing steps or duration for faster generation")
            
        # Resolution-specific recommendations
        if request.resolution == "1080p" and estimate.vram_mb > 10000:
            recommendations.append("Try 720p resolution for better performance")
            
        return recommendations


class ResourceManager:
    """Manages VRAM and hardware resources for generation - Legacy wrapper for optimization service"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.vram_optimizer = get_resource_manager(config)
        self._initialize_hardware_info()
        
    def _initialize_hardware_info(self):
        """Initialize hardware information"""
        try:
            self.gpu_available = self.vram_optimizer.gpu_available
            self.gpu_count = self.vram_optimizer.gpu_count
            self.total_vram = self.vram_optimizer.total_vram
            self.gpu_name = self.vram_optimizer.gpu_name
            self.total_ram = psutil.virtual_memory().total
            
            logger.info(f"Hardware initialized: {self.gpu_name}, "
                       f"VRAM: {self.total_vram/1024**3:.1f}GB, "
                       f"RAM: {self.total_ram/1024**3:.1f}GB")
        except Exception as e:
            logger.error(f"Hardware initialization failed: {e}")
            self.gpu_available = False
            self.gpu_count = 0
            self.total_vram = 0
            self.total_ram = 0
    
    def check_vram_availability(self, required_mb: int) -> bool:
        """Check if sufficient VRAM is available"""
        available, _ = self.vram_optimizer.check_vram_availability(required_mb)
        return available
    
    def optimize_for_available_resources(self, params: GenerationRequest) -> GenerationRequest:
        """Adjust parameters based on available resources"""
        try:
            # Convert GenerationRequest to dict for optimization
            params_dict = {
                "model_type": params.model_type,
                "resolution": params.resolution,
                "steps": params.steps,
                "duration": params.duration,
                "guidance_scale": params.guidance_scale,
                "lora_config": params.lora_config
            }
            
            # Get optimized parameters
            optimized_dict, suggestions = self.vram_optimizer.optimize_parameters_for_resources(params_dict)
            
            # Log optimization suggestions
            if suggestions:
                logger.info(f"Applied {len(suggestions)} resource optimizations:")
                for suggestion in suggestions:
                    logger.info(f"  - {suggestion.parameter}: {suggestion.current_value} -> {suggestion.suggested_value} ({suggestion.reason})")
            
            # Create optimized GenerationRequest
            optimized = GenerationRequest(
                model_type=optimized_dict.get("model_type", params.model_type),
                prompt=params.prompt,
                image=params.image,
                resolution=optimized_dict.get("resolution", params.resolution),
                steps=optimized_dict.get("steps", params.steps),
                guidance_scale=optimized_dict.get("guidance_scale", params.guidance_scale),
                strength=params.strength,
                seed=params.seed,
                fps=params.fps,
                duration=optimized_dict.get("duration", params.duration),
                lora_config=optimized_dict.get("lora_config", params.lora_config),
                optimization_settings=optimized_dict.get("optimization_settings", params.optimization_settings)
            )
            
            return optimized
            
        except Exception as e:
            logger.error(f"Resource optimization failed: {e}")
            return params
    
    def prepare_generation_environment(self, model_type: str) -> None:
        """Prepare optimal environment for generation"""
        try:
            # Use optimization service's cleanup functionality
            cleanup_result = self.vram_optimizer.cleanup_memory(aggressive=False)
            logger.info(f"Generation environment prepared for {model_type}: freed {cleanup_result.get('vram_freed_mb', 0):.1f}MB VRAM")
            
        except Exception as e:
            logger.error(f"Environment preparation failed: {e}")
    
    def get_resource_status(self) -> Dict[str, Any]:
        """Get current resource status"""
        try:
            resource_info = self.vram_optimizer.get_system_resource_info()
            
            status = {
                "gpu_available": self.gpu_available,
                "gpu_name": self.gpu_name,
                "timestamp": datetime.now().isoformat(),
                "total_vram_gb": resource_info.vram.total_mb / 1024,
                "allocated_vram_gb": resource_info.vram.allocated_mb / 1024,
                "cached_vram_gb": resource_info.vram.cached_mb / 1024,
                "available_vram_gb": resource_info.vram.free_mb / 1024,
                "total_ram_gb": resource_info.ram_total_gb,
                "available_ram_gb": resource_info.ram_available_gb,
                "ram_usage_percent": resource_info.ram_usage_percent,
                "cpu_usage_percent": resource_info.cpu_usage_percent,
                "disk_free_gb": resource_info.disk_free_gb
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Resource status check failed: {e}")
            return {"error": str(e)}


class PipelineRouter:
    """Routes generation requests to optimal pipeline configurations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._initialize_pipelines()
        
    def _initialize_pipelines(self):
        """Initialize available pipeline configurations"""
        self.pipelines = {
            GenerationMode.TEXT_TO_VIDEO.value: GenerationPipeline(
                pipeline_type="text_to_video",
                model_path="models/Wan-AI_Wan2.2-T2V-A14B-Diffusers",
                config={
                    "scheduler": "DPMSolverMultistepScheduler",
                    "guidance_scale_range": (1.0, 20.0),
                    "steps_range": (10, 100)
                },
                memory_requirements=ResourceEstimate(6000, 3000, 45, 85),
                supported_resolutions=["480p", "720p", "1080p"],
                optimization_flags={
                    "supports_memory_efficient_attention": True,
                    "supports_cpu_offload": True,
                    "supports_lora": True
                }
            ),
            GenerationMode.IMAGE_TO_VIDEO.value: GenerationPipeline(
                pipeline_type="image_to_video",
                model_path="models/Wan-AI_Wan2.2-I2V-A14B-Diffusers",
                config={
                    "scheduler": "DPMSolverMultistepScheduler",
                    "guidance_scale_range": (1.0, 15.0),
                    "strength_range": (0.1, 1.0),
                    "steps_range": (10, 80)
                },
                memory_requirements=ResourceEstimate(5500, 2500, 35, 80),
                supported_resolutions=["480p", "720p", "1080p"],
                optimization_flags={
                    "supports_memory_efficient_attention": True,
                    "supports_cpu_offload": True,
                    "supports_lora": True
                }
            ),
            GenerationMode.TEXT_IMAGE_TO_VIDEO.value: GenerationPipeline(
                pipeline_type="text_image_to_video",
                model_path="models/Wan-AI_Wan2.2-TI2V-5B-Diffusers",
                config={
                    "scheduler": "DPMSolverMultistepScheduler",
                    "guidance_scale_range": (1.0, 12.0),
                    "strength_range": (0.2, 0.9),
                    "steps_range": (15, 60)
                },
                memory_requirements=ResourceEstimate(4000, 2000, 25, 75),
                supported_resolutions=["480p", "720p"],
                optimization_flags={
                    "supports_memory_efficient_attention": True,
                    "supports_cpu_offload": True,
                    "supports_lora": False
                }
            )
        }
    
    def route_generation_request(self, request: GenerationRequest) -> GenerationPipeline:
        """Route request to appropriate generation pipeline"""
        try:
            pipeline = self.pipelines.get(request.model_type)
            if not pipeline:
                raise ValueError(f"No pipeline available for model type: {request.model_type}")
            
            # Validate request against pipeline capabilities
            self._validate_request_compatibility(request, pipeline)
            
            return pipeline
            
        except Exception as e:
            logger.error(f"Pipeline routing failed: {e}")
            # Return default pipeline as fallback
            return self.pipelines[GenerationMode.TEXT_TO_VIDEO.value]
    
    def select_optimal_pipeline(self, model_type: str, params: GenerationRequest) -> str:
        """Select best pipeline configuration for request"""
        try:
            pipeline = self.pipelines.get(model_type)
            if not pipeline:
                return "default"
            
            # Analyze request requirements
            resource_manager = ResourceManager(self.config)
            available_vram_mb = 0
            
            if resource_manager.gpu_available:
                allocated = torch.cuda.memory_allocated(0)
                available_vram_mb = (resource_manager.total_vram - allocated) / (1024 * 1024)
            
            # Select configuration based on available resources
            if available_vram_mb < 6000:
                return "memory_optimized"
            elif available_vram_mb < 10000:
                return "balanced"
            else:
                return "performance"
                
        except Exception as e:
            logger.error(f"Pipeline selection failed: {e}")
            return "default"
    
    def _validate_request_compatibility(self, request: GenerationRequest, 
                                      pipeline: GenerationPipeline) -> None:
        """Validate request compatibility with pipeline"""
        # Check resolution support
        if request.resolution not in pipeline.supported_resolutions:
            raise ValueError(f"Resolution {request.resolution} not supported by pipeline")
        
        # Check parameter ranges
        config = pipeline.config
        if "guidance_scale_range" in config:
            min_gs, max_gs = config["guidance_scale_range"]
            if not (min_gs <= request.guidance_scale <= max_gs):
                raise ValueError(f"Guidance scale {request.guidance_scale} outside valid range")
        
        if "steps_range" in config:
            min_steps, max_steps = config["steps_range"]
            if not (min_steps <= request.steps <= max_steps):
                raise ValueError(f"Steps {request.steps} outside valid range")
        
        # Check LoRA compatibility
        if request.lora_config and not pipeline.optimization_flags.get("supports_lora", False):
            raise ValueError("LoRA not supported by this pipeline")
    
    def get_pipeline_info(self, model_type: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific pipeline"""
        pipeline = self.pipelines.get(model_type)
        if not pipeline:
            return None
        
        return {
            "pipeline_type": pipeline.pipeline_type,
            "model_path": pipeline.model_path,
            "supported_resolutions": pipeline.supported_resolutions,
            "memory_requirements": pipeline.memory_requirements.to_dict(),
            "optimization_flags": pipeline.optimization_flags,
            "config": pipeline.config
        }
    
    def list_available_pipelines(self) -> Dict[str, Dict[str, Any]]:
        """List all available pipelines with their information"""
        return {
            model_type: self.get_pipeline_info(model_type)
            for model_type in self.pipelines.keys()
        }


class GenerationOrchestrator:
    """Main orchestrator for video generation pipeline"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.preflight_checker = PreflightChecker(config)
        self.resource_manager = ResourceManager(config)
        self.pipeline_router = PipelineRouter(config)
        
        # Legacy compatibility
        self.prompt_validator = self.preflight_checker.prompt_validator
        self.image_validator = self.preflight_checker.image_validator
        self.config_validator = self.preflight_checker.config_validator
        
        # Initialize resource monitoring for backward compatibility
        self._initialize_resource_monitoring()
        
    def _initialize_resource_monitoring(self):
        """Initialize resource monitoring capabilities"""
        try:
            self.gpu_available = torch.cuda.is_available()
            if self.gpu_available:
                self.gpu_count = torch.cuda.device_count()
                self.gpu_memory = torch.cuda.get_device_properties(0).total_memory
            else:
                self.gpu_count = 0
                self.gpu_memory = 0
                
            self.system_memory = psutil.virtual_memory().total
            logger.info(f"Resource monitoring initialized: GPU={self.gpu_available}, "
                       f"GPU Memory={self.gpu_memory/1024**3:.1f}GB, "
                       f"System Memory={self.system_memory/1024**3:.1f}GB")
        except Exception as e:
            logger.warning(f"Failed to initialize resource monitoring: {e}")
            self.gpu_available = False
            self.gpu_count = 0
            self.gpu_memory = 0
            self.system_memory = 0
    
    def validate_request(self, request: GenerationRequest) -> ValidationResult:
        """Comprehensive validation of generation request (legacy method)"""
        return self.preflight_checker._validate_inputs(request)
    
    def estimate_resources(self, request: GenerationRequest) -> ResourceEstimate:
        """Estimate resource requirements for the generation request (legacy method)"""
        return self.preflight_checker.estimate_resource_requirements(request)
    
    def check_resource_availability(self, estimate: ResourceEstimate) -> ResourceStatus:
        """Check if system resources are sufficient for the request (legacy method)"""
        vram_available = self.resource_manager.check_vram_availability(int(estimate.vram_mb))
        
        if not vram_available:
            return ResourceStatus.INSUFFICIENT
        
        # Check system memory
        memory_info = psutil.virtual_memory()
        if memory_info.available < estimate.system_ram_mb * 1024 * 1024:
            return ResourceStatus.LIMITED
        
        # Check margins
        if estimate.vram_mb > 10000 or memory_info.percent > 80:
            return ResourceStatus.LIMITED
        
        return ResourceStatus.AVAILABLE
    
    def prepare_generation(self, request: GenerationRequest) -> Tuple[bool, str]:
        """Prepare the system for generation using new orchestrator components"""
        try:
            # Run preflight checks
            preflight_result = self.preflight_checker.run_preflight_checks(request)
            
            if not preflight_result.can_proceed:
                issues = "; ".join(preflight_result.blocking_issues)
                return False, f"Preflight checks failed: {issues}"
            
            # Optimize request for available resources
            optimized_request = self.resource_manager.optimize_for_available_resources(request)
            
            # Prepare generation environment
            self.resource_manager.prepare_generation_environment(request.model_type)
            
            # Route to appropriate pipeline
            pipeline = self.pipeline_router.route_generation_request(optimized_request)
            
            # Log preparation details
            logger.info(f"Generation prepared: {request.model_type}, "
                       f"Pipeline: {pipeline.pipeline_type}, "
                       f"Resolution: {request.resolution}, "
                       f"Estimated VRAM: {preflight_result.resource_estimate.vram_mb:.0f}MB")
            
            if preflight_result.optimization_recommendations:
                logger.info(f"Recommendations: {'; '.join(preflight_result.optimization_recommendations)}")
            
            return True, "Generation preparation successful"
            
        except Exception as e:
            logger.error(f"Generation preparation failed: {e}")
            return False, f"Preparation error: {str(e)}"
    
    def get_generation_status(self) -> Dict[str, Any]:
        """Get current system status for generation"""
        return self.resource_manager.get_resource_status()
    
    def run_preflight_checks(self, request: GenerationRequest) -> PreflightResult:
        """Run comprehensive preflight checks"""
        return self.preflight_checker.run_preflight_checks(request)
    
    def get_pipeline_info(self, model_type: str) -> Optional[Dict[str, Any]]:
        """Get information about available pipelines"""
        return self.pipeline_router.get_pipeline_info(model_type)
    
    def list_available_pipelines(self) -> Dict[str, Dict[str, Any]]:
        """List all available generation pipelines"""
        return self.pipeline_router.list_available_pipelines()

def create_generation_request(
    model_type: str,
    prompt: str,
    image: Optional[Any] = None,
    **kwargs
) -> GenerationRequest:
    """Factory function to create a generation request"""
    return GenerationRequest(
        model_type=model_type,
        prompt=prompt,
        image=image,
        **kwargs
    )

# Example usage and testing functions
if __name__ == "__main__":
    # Basic test configuration
    test_config = {
        "model_path": "./models",
        "output_path": "./outputs",
        "cache_dir": "./cache"
    }
    
    # Create orchestrator
    orchestrator = GenerationOrchestrator(test_config)
    
    # Test request
    test_request = create_generation_request(
        model_type="t2v-A14B",
        prompt="A beautiful sunset over the ocean",
        resolution="720p",
        steps=50
    )
    
    # Test the new components
    print("Testing Generation Orchestrator Components...")
    print(f"System Status: {orchestrator.get_generation_status()}")
    
    # Test preflight checks
    preflight_result = orchestrator.run_preflight_checks(test_request)
    print(f"Preflight Checks: {'PASS' if preflight_result.can_proceed else 'FAIL'}")
    if preflight_result.blocking_issues:
        print(f"Blocking Issues: {'; '.join(preflight_result.blocking_issues)}")
    if preflight_result.optimization_recommendations:
        print(f"Recommendations: {'; '.join(preflight_result.optimization_recommendations)}")
    
    # Test pipeline routing
    try:
        pipeline_info = orchestrator.get_pipeline_info(test_request.model_type)
        print(f"Pipeline Info: {pipeline_info}")
    except Exception as e:
        print(f"Pipeline routing error: {e}")
    
    # Test resource optimization
    optimized_request = orchestrator.resource_manager.optimize_for_available_resources(test_request)
    print(f"Original steps: {test_request.steps}, Optimized steps: {optimized_request.steps}")
    
    # Test generation preparation
    success, message = orchestrator.prepare_generation(test_request)
    print(f"Preparation: {'Success' if success else 'Failed'} - {message}")
    
    # List available pipelines
    pipelines = orchestrator.list_available_pipelines()
    print(f"Available Pipelines: {list(pipelines.keys())}")
