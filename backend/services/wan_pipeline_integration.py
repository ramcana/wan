"""
WAN Pipeline Integration with Model Orchestrator.

This module provides the integration layer between the Model Orchestrator
and WAN pipeline loading, including model-specific handling and component validation.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Type
from dataclasses import dataclass
from enum import Enum

from backend.core.model_orchestrator.model_ensurer import ModelEnsurer
from backend.core.model_orchestrator.model_registry import ModelRegistry
from backend.core.model_orchestrator.exceptions import ModelNotFoundError, ModelOrchestratorError

logger = logging.getLogger(__name__)


class WanModelType(Enum):
    """WAN model types with their characteristics."""
    T2V = "t2v"      # Text-to-Video
    I2V = "i2v"      # Image-to-Video  
    TI2V = "ti2v"    # Text+Image-to-Video


@dataclass
class WanModelSpec:
    """Specification for a WAN model type."""
    model_type: WanModelType
    pipeline_class: str
    required_components: List[str]
    optional_components: List[str]
    vram_estimation_gb: float
    max_frames: int
    max_resolution: Tuple[int, int]
    supports_image_input: bool
    supports_text_input: bool


# Pipeline class mappings for different WAN model types
WAN_PIPELINE_MAPPINGS = {
    "t2v-A14B": WanModelSpec(
        model_type=WanModelType.T2V,
        pipeline_class="WanT2VPipeline",
        required_components=["text_encoder", "unet", "vae", "scheduler"],
        optional_components=[],
        vram_estimation_gb=12.0,
        max_frames=64,
        max_resolution=(1920, 1080),
        supports_image_input=False,
        supports_text_input=True
    ),
    "i2v-A14B": WanModelSpec(
        model_type=WanModelType.I2V,
        pipeline_class="WanI2VPipeline", 
        required_components=["image_encoder", "unet", "vae", "scheduler"],
        optional_components=["text_encoder"],
        vram_estimation_gb=12.0,
        max_frames=64,
        max_resolution=(1920, 1080),
        supports_image_input=True,
        supports_text_input=False
    ),
    "ti2v-5b": WanModelSpec(
        model_type=WanModelType.TI2V,
        pipeline_class="WanTI2VPipeline",
        required_components=["text_encoder", "image_encoder", "unet", "vae", "scheduler"],
        optional_components=[],
        vram_estimation_gb=8.0,
        max_frames=64,
        max_resolution=(2560, 1440),
        supports_image_input=True,
        supports_text_input=True
    )
}


@dataclass
class ComponentValidationResult:
    """Result of component validation."""
    is_valid: bool
    missing_components: List[str]
    invalid_components: List[str]
    warnings: List[str]


class WanPipelineIntegration:
    """Integration layer between Model Orchestrator and WAN pipeline loading."""
    
    def __init__(self, model_ensurer: ModelEnsurer, model_registry: ModelRegistry):
        """
        Initialize the WAN pipeline integration.
        
        Args:
            model_ensurer: Model ensurer for downloading and managing models
            model_registry: Model registry for model specifications
        """
        self.model_ensurer = model_ensurer
        self.model_registry = model_registry
        self.logger = logging.getLogger(__name__ + ".WanPipelineIntegration")
    
    def get_wan_paths(self, model_id: str, variant: Optional[str] = None) -> str:
        """
        Get the local path for a WAN model, ensuring it's downloaded.
        
        This is the main integration point that replaces hardcoded paths
        in the pipeline loader.
        
        Args:
            model_id: Model identifier (e.g., "t2v-A14B@2.2.0")
            variant: Optional variant (e.g., "fp16", "bf16")
            
        Returns:
            Absolute path to the ready-to-use model directory
            
        Raises:
            ModelNotFoundError: If model is not found in registry
            ModelOrchestratorError: If model cannot be ensured
        """
        try:
            # Ensure the model is available locally
            model_path = self.model_ensurer.ensure(model_id, variant)
            self.logger.info(f"Model {model_id} available at: {model_path}")
            return model_path
            
        except Exception as e:
            self.logger.error(f"Failed to get WAN paths for {model_id}: {e}")
            raise
    
    def get_pipeline_class(self, model_id: str) -> str:
        """
        Get the appropriate pipeline class for a model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Pipeline class name
            
        Raises:
            ModelNotFoundError: If model type is not recognized
        """
        model_base = self._extract_model_base(model_id)
        
        if model_base in WAN_PIPELINE_MAPPINGS:
            return WAN_PIPELINE_MAPPINGS[model_base].pipeline_class
        
        raise ModelNotFoundError(
            f"Unknown WAN model type: {model_base}",
            list(WAN_PIPELINE_MAPPINGS.keys())
        )
    
    def validate_components(self, model_id: str, model_path: str) -> ComponentValidationResult:
        """
        Validate that all required components are present before GPU initialization.
        
        Args:
            model_id: Model identifier
            model_path: Path to the model directory
            
        Returns:
            ComponentValidationResult with validation details
        """
        model_base = self._extract_model_base(model_id)
        
        if model_base not in WAN_PIPELINE_MAPPINGS:
            return ComponentValidationResult(
                is_valid=False,
                missing_components=[],
                invalid_components=[],
                warnings=[f"Unknown model type: {model_base}"]
            )
        
        spec = WAN_PIPELINE_MAPPINGS[model_base]
        model_dir = Path(model_path)
        
        missing_components = []
        invalid_components = []
        warnings = []
        
        # Check for model_index.json
        model_index_path = model_dir / "model_index.json"
        if not model_index_path.exists():
            missing_components.append("model_index.json")
        else:
            try:
                with open(model_index_path, 'r') as f:
                    model_index = json.load(f)
                
                # Validate pipeline class
                expected_class = spec.pipeline_class
                actual_class = model_index.get("_class_name", "")
                if actual_class != expected_class:
                    warnings.append(
                        f"Pipeline class mismatch: expected {expected_class}, got {actual_class}"
                    )
                
                # Check required components
                for component in spec.required_components:
                    if component not in model_index:
                        missing_components.append(component)
                
                # Check for invalid components (e.g., image_encoder in T2V model)
                if spec.model_type == WanModelType.T2V and "image_encoder" in model_index:
                    invalid_components.append("image_encoder")
                    warnings.append("T2V model should not have image_encoder component")
                
            except (json.JSONDecodeError, IOError) as e:
                warnings.append(f"Failed to read model_index.json: {e}")
        
        # Check for component directories/files
        for component in spec.required_components:
            component_paths = [
                model_dir / component,
                model_dir / f"{component}.pth",
                model_dir / f"{component}.safetensors"
            ]
            
            if not any(p.exists() for p in component_paths):
                missing_components.append(f"{component} (files)")
        
        is_valid = len(missing_components) == 0 and len(invalid_components) == 0
        
        return ComponentValidationResult(
            is_valid=is_valid,
            missing_components=missing_components,
            invalid_components=invalid_components,
            warnings=warnings
        )
    
    def estimate_vram_usage(self, model_id: str, **generation_params) -> float:
        """
        Estimate VRAM usage for a model with given generation parameters.
        
        Args:
            model_id: Model identifier
            **generation_params: Generation parameters (num_frames, width, height, etc.)
            
        Returns:
            Estimated VRAM usage in GB
        """
        model_base = self._extract_model_base(model_id)
        
        if model_base not in WAN_PIPELINE_MAPPINGS:
            # Fallback estimation
            return 8.0
        
        spec = WAN_PIPELINE_MAPPINGS[model_base]
        base_vram = spec.vram_estimation_gb
        
        # Adjust based on generation parameters
        num_frames = generation_params.get('num_frames', 16)
        width = generation_params.get('width', 512)
        height = generation_params.get('height', 512)
        batch_size = generation_params.get('batch_size', 1)
        
        # Calculate additional VRAM for generation
        pixel_count = width * height * num_frames * batch_size
        
        # Rough estimation: 4 bytes per pixel for intermediate tensors
        intermediate_gb = (pixel_count * 4) / (1024 ** 3)
        
        # Model-specific adjustments
        if spec.model_type == WanModelType.TI2V:
            # TI2V has dual conditioning, needs more memory
            intermediate_gb *= 1.3
            base_vram *= 1.1  # Also increase base VRAM for dual conditioning
        elif spec.model_type == WanModelType.I2V:
            # I2V processes image input, needs additional memory
            intermediate_gb *= 1.1
        
        total_vram = base_vram + intermediate_gb
        
        # Add 20% safety margin
        return total_vram * 1.2
    
    def get_model_capabilities(self, model_id: str) -> Dict[str, Any]:
        """
        Get model capabilities and constraints.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Dictionary with model capabilities
        """
        model_base = self._extract_model_base(model_id)
        
        if model_base not in WAN_PIPELINE_MAPPINGS:
            return {}
        
        spec = WAN_PIPELINE_MAPPINGS[model_base]
        
        return {
            "model_type": spec.model_type.value,
            "pipeline_class": spec.pipeline_class,
            "max_frames": spec.max_frames,
            "max_resolution": spec.max_resolution,
            "supports_image_input": spec.supports_image_input,
            "supports_text_input": spec.supports_text_input,
            "estimated_vram_gb": spec.vram_estimation_gb,
            "required_components": spec.required_components,
            "optional_components": spec.optional_components
        }
    
    def _extract_model_base(self, model_id: str) -> str:
        """Extract the base model name from a full model ID."""
        # Handle both "t2v-A14B@2.2.0" and "t2v-A14B" formats
        if "@" in model_id:
            return model_id.split("@")[0]
        return model_id


# Global instance for easy access
_wan_integration: Optional[WanPipelineIntegration] = None


def get_wan_integration() -> WanPipelineIntegration:
    """Get the global WAN pipeline integration instance."""
    global _wan_integration
    if _wan_integration is None:
        raise RuntimeError("WAN pipeline integration not initialized. Call initialize_wan_integration() first.")
    return _wan_integration


def initialize_wan_integration(model_ensurer: ModelEnsurer, model_registry: ModelRegistry) -> None:
    """Initialize the global WAN pipeline integration instance."""
    global _wan_integration
    _wan_integration = WanPipelineIntegration(model_ensurer, model_registry)


def get_wan_paths(model_id: str, variant: Optional[str] = None) -> str:
    """
    Global function to get WAN model paths.
    
    This is the main function that replaces hardcoded paths in the pipeline loader.
    
    Args:
        model_id: Model identifier
        variant: Optional variant
        
    Returns:
        Absolute path to the model directory
    """
    return get_wan_integration().get_wan_paths(model_id, variant)