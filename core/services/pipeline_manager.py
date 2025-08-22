"""
Pipeline management system for Wan model compatibility.

This module provides classes for selecting, loading, and validating custom pipelines
based on model architecture signatures and requirements.
"""

import logging
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Type, Callable
from enum import Enum

from infrastructure.hardware.architecture_detector import (
    ArchitectureSignature, ArchitectureType, ModelArchitecture, 
    ModelRequirements, ComponentInfo
)

logger = logging.getLogger(__name__)


class PipelineLoadStatus(Enum):
    """Status of pipeline loading attempt."""
    SUCCESS = "success"
    FAILED_MISSING_CLASS = "failed_missing_class"
    FAILED_INVALID_ARGS = "failed_invalid_args"
    FAILED_DEPENDENCIES = "failed_dependencies"
    FAILED_REMOTE_CODE = "failed_remote_code"
    FAILED_UNKNOWN = "failed_unknown"


@dataclass
class PipelineRequirements:
    """Requirements for a specific pipeline class."""
    required_args: List[str] = field(default_factory=list)
    optional_args: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    min_vram_mb: int = 4096
    supports_cpu_offload: bool = True
    supports_mixed_precision: bool = True
    requires_trust_remote_code: bool = False
    pipeline_source: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of pipeline argument validation."""
    is_valid: bool
    missing_required: List[str] = field(default_factory=list)
    invalid_args: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


@dataclass
class PipelineLoadResult:
    """Result of pipeline loading attempt."""
    status: PipelineLoadStatus
    pipeline: Optional[Any] = None
    pipeline_class: Optional[str] = None
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    applied_optimizations: List[str] = field(default_factory=list)


class PipelineManager:
    """Manages custom pipeline selection, loading, and validation."""
    
    # Pipeline class mappings based on architecture signatures
    PIPELINE_MAPPINGS = {
        ArchitectureType.WAN_T2V: "WanPipeline",
        ArchitectureType.WAN_T2I: "WanPipeline", 
        ArchitectureType.WAN_I2V: "WanPipeline",
        ArchitectureType.STABLE_DIFFUSION: "StableDiffusionPipeline",
    }
    
    # Known pipeline requirements
    PIPELINE_REQUIREMENTS = {
        "WanPipeline": PipelineRequirements(
            required_args=["transformer", "scheduler", "vae"],
            optional_args=["transformer_2", "text_encoder", "tokenizer", "safety_checker", "feature_extractor"],
            dependencies=["transformers>=4.25.0", "torch>=2.0.0"],
            min_vram_mb=8192,
            supports_cpu_offload=True,
            supports_mixed_precision=True,
            requires_trust_remote_code=True,
            pipeline_source="huggingface"
        ),
        "StableDiffusionPipeline": PipelineRequirements(
            required_args=["unet", "scheduler", "vae", "text_encoder", "tokenizer"],
            optional_args=["safety_checker", "feature_extractor"],
            dependencies=["transformers>=4.21.0", "torch>=1.11.0"],
            min_vram_mb=4096,
            supports_cpu_offload=True,
            supports_mixed_precision=True,
            requires_trust_remote_code=False
        ),
        "StableDiffusionXLPipeline": PipelineRequirements(
            required_args=["unet", "scheduler", "vae", "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2"],
            optional_args=["safety_checker", "feature_extractor"],
            dependencies=["transformers>=4.25.0", "torch>=2.0.0"],
            min_vram_mb=6144,
            supports_cpu_offload=True,
            supports_mixed_precision=True,
            requires_trust_remote_code=False
        )
    }
    
    def __init__(self):
        """Initialize the pipeline manager."""
        self.logger = logging.getLogger(__name__ + ".PipelineManager")
        self._pipeline_cache = {}
        
    def select_pipeline_class(self, architecture: ArchitectureSignature) -> str:
        """
        Select appropriate pipeline class based on architecture signature.
        
        Args:
            architecture: Architecture signature from model analysis
            
        Returns:
            Pipeline class name
        """
        self.logger.info(f"Selecting pipeline for architecture: {architecture}")
        
        # First check if pipeline class is explicitly specified
        if architecture.pipeline_class:
            pipeline_class = architecture.pipeline_class
            self.logger.info(f"Using explicit pipeline class: {pipeline_class}")
            return pipeline_class
        
        # Determine architecture type and map to pipeline
        arch_type = architecture.get_architecture_type()
        pipeline_class = self.PIPELINE_MAPPINGS.get(arch_type, "DiffusionPipeline")
        
        self.logger.info(f"Selected pipeline class: {pipeline_class} for architecture type: {arch_type.value}")
        
        # Additional logic for specific Wan variants
        if arch_type in [ArchitectureType.WAN_T2V, ArchitectureType.WAN_T2I, ArchitectureType.WAN_I2V]:
            # Check for specific Wan pipeline variants
            if architecture.has_transformer_2 and architecture.vae_dimensions == 3:
                pipeline_class = "WanPipeline"  # Full T2V pipeline
            elif architecture.has_transformer and not architecture.has_transformer_2:
                pipeline_class = "WanPipeline"  # T2I or simplified pipeline
            
            self.logger.info(f"Refined Wan pipeline selection: {pipeline_class}")
        
        return pipeline_class
    
    def load_custom_pipeline(self, model_path: str, pipeline_class: str, 
                           trust_remote_code: bool = False, **kwargs) -> PipelineLoadResult:
        """
        Load custom pipeline with proper error handling.
        
        Args:
            model_path: Path to the model
            pipeline_class: Pipeline class name to load
            trust_remote_code: Whether to trust remote code
            **kwargs: Additional arguments for pipeline loading
            
        Returns:
            PipelineLoadResult with loading status and pipeline
        """
        self.logger.info(f"Loading pipeline {pipeline_class} from {model_path}")
        
        try:
            # Import diffusers here to avoid circular imports
            from diffusers import DiffusionPipeline
            
            # Prepare loading arguments - filter out model_index.json contents for WAN models
            if pipeline_class == "WanPipeline":
                # For WAN models, only pass essential loading arguments
                filtered_kwargs = {k: v for k, v in kwargs.items() 
                                 if k in ['torch_dtype', 'device_map', 'low_cpu_mem_usage', 'variant']}
                load_args = {
                    "pretrained_model_name_or_path": model_path,
                    "trust_remote_code": trust_remote_code,
                    **filtered_kwargs
                }
            else:
                load_args = {
                    "pretrained_model_name_or_path": model_path,
                    "trust_remote_code": trust_remote_code,
                    **kwargs
                }
            
            # Remove None values
            load_args = {k: v for k, v in load_args.items() if v is not None}
            
            self.logger.debug(f"Loading with args: {load_args}")
            
            # Attempt to load the pipeline
            if pipeline_class == "DiffusionPipeline":
                # Use auto-detection
                pipeline = DiffusionPipeline.from_pretrained(**load_args)
            elif pipeline_class == "WanPipeline":
                # Special handling for WAN models
                self.logger.info("Loading WAN model with trust_remote_code=True")
                load_args["trust_remote_code"] = True
                
                try:
                    # Temporarily remove our compatibility layer to allow remote code loading
                    try:
                        from wan22_compatibility_clean import remove_wan22_compatibility, apply_wan22_compatibility
                        remove_wan22_compatibility()
                        compatibility_removed = True
                    except ImportError:
                        compatibility_removed = False
                    
                    try:
                        # Try to load with auto-detection first (will use remote code)
                        pipeline = DiffusionPipeline.from_pretrained(**load_args)
                        self.logger.info(f"Successfully loaded WAN model with pipeline: {type(pipeline).__name__}")
                    finally:
                        # Restore compatibility layer
                        if compatibility_removed:
                            apply_wan22_compatibility()
                            
                except Exception as e:
                    self.logger.error(f"Failed to load WAN model: {e}")
                    return PipelineLoadResult(
                        status=PipelineLoadStatus.FAILED_REMOTE_CODE,
                        error_message=f"Failed to load WAN model: {str(e)}. Ensure trust_remote_code=True and model has proper pipeline code."
                    )
            else:
                # Try to get specific pipeline class
                try:
                    # First try to import from diffusers
                    pipeline_module = __import__("diffusers", fromlist=[pipeline_class])
                    pipeline_cls = getattr(pipeline_module, pipeline_class)
                    pipeline = pipeline_cls.from_pretrained(**load_args)
                except (ImportError, AttributeError):
                    # Fallback to DiffusionPipeline with custom_pipeline
                    if trust_remote_code:
                        load_args["custom_pipeline"] = pipeline_class.lower()
                        pipeline = DiffusionPipeline.from_pretrained(**load_args)
                    else:
                        return PipelineLoadResult(
                            status=PipelineLoadStatus.FAILED_MISSING_CLASS,
                            pipeline_class=pipeline_class,
                            error_message=f"Pipeline class {pipeline_class} not found and trust_remote_code=False"
                        )
            
            self.logger.info(f"Successfully loaded pipeline: {type(pipeline).__name__}")
            
            return PipelineLoadResult(
                status=PipelineLoadStatus.SUCCESS,
                pipeline=pipeline,
                pipeline_class=type(pipeline).__name__
            )
            
        except ImportError as e:
            self.logger.error(f"Missing dependencies for {pipeline_class}: {e}")
            return PipelineLoadResult(
                status=PipelineLoadStatus.FAILED_DEPENDENCIES,
                pipeline_class=pipeline_class,
                error_message=f"Missing dependencies: {str(e)}"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to load pipeline {pipeline_class}: {e}")
            return PipelineLoadResult(
                status=PipelineLoadStatus.FAILED_UNKNOWN,
                pipeline_class=pipeline_class,
                error_message=str(e)
            )
    
    def validate_pipeline_args(self, pipeline_class: str, provided_args: Dict[str, Any]) -> ValidationResult:
        """
        Validate that all required pipeline arguments are provided.
        
        Args:
            pipeline_class: Pipeline class name
            provided_args: Arguments provided for pipeline loading
            
        Returns:
            ValidationResult with validation status and suggestions
        """
        self.logger.debug(f"Validating args for {pipeline_class}: {list(provided_args.keys())}")
        
        requirements = self.get_pipeline_requirements(pipeline_class)
        
        # Check for missing required arguments
        missing_required = []
        for required_arg in requirements.required_args:
            if required_arg not in provided_args:
                missing_required.append(required_arg)
        
        # Check for invalid arguments (basic validation)
        invalid_args = []
        warnings = []
        suggestions = []
        
        # Validate specific argument types if possible
        for arg_name, arg_value in provided_args.items():
            if arg_value is None:
                warnings.append(f"Argument '{arg_name}' is None - may cause issues")
            elif arg_name in ["torch_dtype"] and not str(arg_value).startswith("torch."):
                suggestions.append(f"Consider using torch.float16 or torch.float32 for {arg_name}")
        
        # Add suggestions based on pipeline type
        if pipeline_class == "WanPipeline":
            if "trust_remote_code" not in provided_args or not provided_args["trust_remote_code"]:
                suggestions.append("WanPipeline typically requires trust_remote_code=True")
            if "torch_dtype" not in provided_args:
                suggestions.append("Consider adding torch_dtype=torch.float16 for memory efficiency")
        
        is_valid = len(missing_required) == 0 and len(invalid_args) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            missing_required=missing_required,
            invalid_args=invalid_args,
            warnings=warnings,
            suggestions=suggestions
        )
    
    def get_pipeline_requirements(self, pipeline_class: str) -> PipelineRequirements:
        """
        Get required arguments and dependencies for pipeline class.
        
        Args:
            pipeline_class: Pipeline class name
            
        Returns:
            PipelineRequirements with detailed requirements
        """
        # Return known requirements or defaults
        if pipeline_class in self.PIPELINE_REQUIREMENTS:
            return self.PIPELINE_REQUIREMENTS[pipeline_class]
        
        # Try to infer requirements for unknown pipelines
        self.logger.warning(f"Unknown pipeline class {pipeline_class}, using defaults")
        
        # Default requirements
        requirements = PipelineRequirements()
        
        # Infer some requirements based on naming patterns
        if "Wan" in pipeline_class:
            requirements.requires_trust_remote_code = True
            requirements.min_vram_mb = 8192
            requirements.dependencies = ["transformers>=4.25.0"]
            requirements.required_args = ["transformer", "scheduler", "vae"]
        elif "StableDiffusion" in pipeline_class:
            requirements.required_args = ["unet", "scheduler", "vae", "text_encoder", "tokenizer"]
            requirements.min_vram_mb = 4096
        elif "XL" in pipeline_class:
            requirements.min_vram_mb = 6144
            requirements.required_args = ["unet", "scheduler", "vae", "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2"]
        
        return requirements
    
    def create_pipeline_mapping(self, architecture: ModelArchitecture) -> Dict[str, str]:
        """
        Create component-to-pipeline argument mapping.
        
        Args:
            architecture: Model architecture information
            
        Returns:
            Dictionary mapping component names to pipeline arguments
        """
        mapping = {}
        
        # Standard component mappings
        standard_mappings = {
            "scheduler": "scheduler",
            "vae": "vae", 
            "text_encoder": "text_encoder",
            "tokenizer": "tokenizer",
            "safety_checker": "safety_checker",
            "feature_extractor": "feature_extractor"
        }
        
        # Architecture-specific mappings
        if architecture.architecture_type in [ArchitectureType.WAN_T2V, ArchitectureType.WAN_T2I, ArchitectureType.WAN_I2V]:
            # Wan models use transformers instead of unet
            wan_mappings = {
                "transformer": "transformer",
                "transformer_2": "transformer_2"
            }
            mapping.update(wan_mappings)
        else:
            # Standard diffusion models
            mapping["unet"] = "unet"
        
        # Add standard mappings
        mapping.update(standard_mappings)
        
        # Filter to only include components that exist in the model
        available_components = set(architecture.components.keys())
        filtered_mapping = {k: v for k, v in mapping.items() if k in available_components}
        
        self.logger.debug(f"Created pipeline mapping: {filtered_mapping}")
        return filtered_mapping
    
    def suggest_pipeline_alternatives(self, failed_pipeline: str, 
                                    architecture: ArchitectureSignature) -> List[str]:
        """
        Suggest alternative pipeline classes if the primary choice fails.
        
        Args:
            failed_pipeline: Pipeline class that failed to load
            architecture: Architecture signature
            
        Returns:
            List of alternative pipeline class names
        """
        alternatives = []
        
        if failed_pipeline == "WanPipeline":
            # Fallback options for Wan models
            alternatives.extend([
                "DiffusionPipeline",  # Auto-detection
                "StableDiffusionPipeline"  # May work with some components
            ])
        elif failed_pipeline == "StableDiffusionPipeline":
            alternatives.extend([
                "DiffusionPipeline",
                "StableDiffusionXLPipeline"
            ])
        elif failed_pipeline == "DiffusionPipeline":
            # Last resort - try specific pipelines
            arch_type = architecture.get_architecture_type()
            if arch_type != ArchitectureType.UNKNOWN:
                alternatives.append(self.PIPELINE_MAPPINGS.get(arch_type, "StableDiffusionPipeline"))
        
        # Remove the failed pipeline from alternatives
        alternatives = [alt for alt in alternatives if alt != failed_pipeline]
        
        self.logger.info(f"Suggested alternatives for {failed_pipeline}: {alternatives}")
        return alternatives
    
    def validate_pipeline_compatibility(self, pipeline_class: str, 
                                      architecture: ModelArchitecture) -> ValidationResult:
        """
        Validate compatibility between pipeline class and model architecture.
        
        Args:
            pipeline_class: Pipeline class name
            architecture: Model architecture
            
        Returns:
            ValidationResult with compatibility assessment
        """
        warnings = []
        suggestions = []
        invalid_args = []
        
        # Get pipeline requirements
        requirements = self.get_pipeline_requirements(pipeline_class)
        
        # Check component compatibility
        available_components = set(architecture.components.keys())
        required_components = set(requirements.required_args)
        
        missing_components = required_components - available_components
        
        # Architecture-specific validation
        if pipeline_class == "WanPipeline":
            if architecture.architecture_type not in [ArchitectureType.WAN_T2V, ArchitectureType.WAN_T2I, ArchitectureType.WAN_I2V]:
                warnings.append("Using WanPipeline with non-Wan model may not work")
            
            if not architecture.signature or not architecture.signature.is_wan_architecture():
                warnings.append("Model doesn't appear to be a Wan architecture")
                
        elif pipeline_class == "StableDiffusionPipeline":
            if architecture.architecture_type in [ArchitectureType.WAN_T2V, ArchitectureType.WAN_T2I, ArchitectureType.WAN_I2V]:
                warnings.append("Using StableDiffusionPipeline with Wan model will likely fail")
                suggestions.append("Consider using WanPipeline instead")
        
        # VRAM compatibility check
        if architecture.requirements.min_vram_mb > requirements.min_vram_mb:
            warnings.append(f"Model requires {architecture.requirements.min_vram_mb}MB VRAM, pipeline supports {requirements.min_vram_mb}MB")
        
        is_valid = len(missing_components) == 0 and len(invalid_args) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            missing_required=list(missing_components),
            invalid_args=invalid_args,
            warnings=warnings,
            suggestions=suggestions
        )