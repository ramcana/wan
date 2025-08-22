"""
Core architecture detection system for Wan model compatibility.

This module provides classes for detecting model architectures, validating components,
and determining compatibility requirements for different model types.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class ArchitectureType(Enum):
    """Supported model architecture types."""
    WAN_T2V = "wan_t2v"
    WAN_T2I = "wan_t2i" 
    WAN_I2V = "wan_i2v"
    STABLE_DIFFUSION = "stable_diffusion"
    UNKNOWN = "unknown"


class VAEType(Enum):
    """VAE architecture types."""
    VAE_2D = "2d"  # Standard SD VAE
    VAE_3D = "3d"  # Wan video VAE
    UNKNOWN = "unknown"


@dataclass
class ComponentInfo:
    """Information about a model component."""
    class_name: str
    config_path: str
    weight_path: str
    is_custom: bool = False
    dependencies: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate component info after initialization."""
        if not self.class_name:
            raise ValueError("Component class_name cannot be empty")
        if not self.config_path and not self.weight_path:
            raise ValueError("Component must have either config_path or weight_path")


@dataclass
class ArchitectureSignature:
    """Signature characteristics that identify model architecture."""
    has_transformer: bool = False
    has_transformer_2: bool = False
    has_boundary_ratio: bool = False
    vae_dimensions: int = 2
    component_classes: Dict[str, str] = field(default_factory=dict)
    pipeline_class: Optional[str] = None
    custom_attributes: Dict[str, Any] = field(default_factory=dict)
    
    def is_wan_architecture(self) -> bool:
        """Determine if this signature indicates a Wan model architecture."""
        wan_indicators = [
            self.has_transformer_2,
            self.has_boundary_ratio,
            self.vae_dimensions == 3,
            self.pipeline_class and "Wan" in self.pipeline_class,
            any("Wan" in class_name for class_name in self.component_classes.values()),
            # Check for transformer with Wan-specific attributes
            self.has_transformer and (self.has_boundary_ratio or 
                                    self.pipeline_class and "Wan" in self.pipeline_class or
                                    any("Wan" in class_name for class_name in self.component_classes.values()))
        ]
        return any(wan_indicators)
    
    def get_architecture_type(self) -> ArchitectureType:
        """Determine the specific architecture type."""
        if not self.is_wan_architecture():
            # Check for Stable Diffusion indicators
            if (any("StableDiffusion" in class_name for class_name in self.component_classes.values()) or
                self.pipeline_class and "StableDiffusion" in self.pipeline_class or
                any(comp in self.component_classes for comp in ["unet", "text_encoder", "tokenizer"])):
                return ArchitectureType.STABLE_DIFFUSION
            return ArchitectureType.UNKNOWN
        
        # Determine Wan variant based on components and attributes
        if self.has_transformer_2 and self.vae_dimensions == 3:
            return ArchitectureType.WAN_T2V
        elif self.has_transformer and not self.has_transformer_2:
            return ArchitectureType.WAN_T2I
        elif self.pipeline_class and "i2v" in str(self.pipeline_class).lower():
            return ArchitectureType.WAN_I2V
        
        return ArchitectureType.WAN_T2V  # Default for Wan models


@dataclass
class ModelRequirements:
    """Requirements for loading and running a model."""
    min_vram_mb: int = 4096
    recommended_vram_mb: int = 8192
    requires_trust_remote_code: bool = False
    required_dependencies: List[str] = field(default_factory=list)
    supports_cpu_offload: bool = True
    supports_mixed_precision: bool = True


@dataclass
class ModelArchitecture:
    """Complete model architecture information."""
    architecture_type: ArchitectureType
    version: Optional[str] = None
    components: Dict[str, ComponentInfo] = field(default_factory=dict)
    requirements: ModelRequirements = field(default_factory=ModelRequirements)
    capabilities: List[str] = field(default_factory=list)
    signature: Optional[ArchitectureSignature] = None
    
    def __post_init__(self):
        """Set default capabilities based on architecture type."""
        if not self.capabilities:
            if self.architecture_type == ArchitectureType.WAN_T2V:
                self.capabilities = ["text_to_video"]
            elif self.architecture_type == ArchitectureType.WAN_T2I:
                self.capabilities = ["text_to_image"]
            elif self.architecture_type == ArchitectureType.WAN_I2V:
                self.capabilities = ["image_to_video"]
            elif self.architecture_type == ArchitectureType.STABLE_DIFFUSION:
                self.capabilities = ["text_to_image"]


@dataclass
class CompatibilityReport:
    """Report on model component compatibility."""
    is_compatible: bool
    compatible_components: List[str] = field(default_factory=list)
    incompatible_components: List[str] = field(default_factory=list)
    missing_components: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class ArchitectureDetector:
    """Detects model architecture and validates component compatibility."""
    
    def __init__(self):
        """Initialize the architecture detector."""
        self.logger = logging.getLogger(__name__ + ".ArchitectureDetector")
    
    def detect_model_architecture(self, model_path: Union[str, Path]) -> ModelArchitecture:
        """
        Detect model architecture from model files and configuration.
        
        Args:
            model_path: Path to the model directory or file
            
        Returns:
            ModelArchitecture object with detected information
            
        Raises:
            FileNotFoundError: If model path doesn't exist
            ValueError: If model configuration is invalid
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        
        self.logger.info(f"Detecting architecture for model at: {model_path}")
        
        # Analyze model_index.json if available
        signature = self.analyze_model_index(model_path)
        
        # Analyze individual component configs
        components = self._analyze_components(model_path)
        
        # Determine architecture type
        architecture_type = signature.get_architecture_type()
        
        # Set requirements based on architecture
        requirements = self._determine_requirements(signature, architecture_type)
        
        architecture = ModelArchitecture(
            architecture_type=architecture_type,
            components=components,
            requirements=requirements,
            signature=signature
        )
        
        self.logger.info(f"Detected architecture: {architecture_type.value}")
        return architecture
    
    def analyze_model_index(self, model_path: Union[str, Path]) -> ArchitectureSignature:
        """
        Analyze model_index.json for architecture patterns.
        
        Args:
            model_path: Path to the model directory
            
        Returns:
            ArchitectureSignature with detected patterns
        """
        model_path = Path(model_path)
        model_index_path = model_path / "model_index.json"
        
        signature = ArchitectureSignature()
        
        if not model_index_path.exists():
            self.logger.warning(f"No model_index.json found at {model_index_path}")
            return signature
        
        try:
            with open(model_index_path, 'r', encoding='utf-8') as f:
                model_index = json.load(f)
            
            self.logger.debug(f"Loaded model_index.json: {model_index}")
            
            # Extract pipeline class
            signature.pipeline_class = model_index.get("_class_name")
            
            # Check for Wan-specific components
            signature.has_transformer = "transformer" in model_index
            signature.has_transformer_2 = "transformer_2" in model_index
            signature.has_boundary_ratio = "boundary_ratio" in model_index
            
            # Extract component classes
            for key, value in model_index.items():
                if isinstance(value, list) and len(value) >= 2:
                    # Format: ["component_type", "ClassName"]
                    signature.component_classes[key] = value[1] if len(value) > 1 else value[0]
                elif isinstance(value, str) and not key.startswith("_"):
                    signature.component_classes[key] = value
            
            # Store custom attributes
            for key, value in model_index.items():
                if key not in ["_class_name", "_diffusers_version"] and not isinstance(value, list):
                    signature.custom_attributes[key] = value
            
            # Check VAE dimensions
            if "vae" in model_index:
                vae_type = self.check_vae_dimensions(model_path / "vae")
                signature.vae_dimensions = 3 if vae_type == VAEType.VAE_3D else 2
            
        except (json.JSONDecodeError, IOError) as e:
            self.logger.error(f"Error reading model_index.json: {e}")
            raise ValueError(f"Invalid model_index.json: {e}")
        
        return signature
    
    def check_vae_dimensions(self, vae_path: Union[str, Path]) -> VAEType:
        """
        Determine if VAE is 2D (SD) or 3D (Wan) based on configuration.
        
        Args:
            vae_path: Path to VAE component directory
            
        Returns:
            VAEType indicating 2D or 3D architecture
        """
        vae_path = Path(vae_path)
        config_path = vae_path / "config.json"
        
        if not config_path.exists():
            self.logger.warning(f"No VAE config found at {config_path}")
            return VAEType.UNKNOWN
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                vae_config = json.load(f)
            
            # Check for 3D VAE indicators
            # Wan VAEs typically have different latent dimensions
            latent_channels = vae_config.get("latent_channels", 4)
            out_channels = vae_config.get("out_channels", 3)
            
            # Check for 3D-specific parameters
            has_3d_params = any(key in vae_config for key in [
                "temporal_compression_ratio",
                "temporal_downsample_factor", 
                "time_compression",
                "num_frames"
            ])
            
            # Check latent dimensions - Wan VAEs often have different channel counts
            if has_3d_params or latent_channels > 4 or out_channels > 3:
                self.logger.info("Detected 3D VAE architecture")
                return VAEType.VAE_3D
            
            # Check for specific class names indicating 3D
            vae_class = vae_config.get("_class_name", "")
            if "3D" in vae_class or "Video" in vae_class or "Temporal" in vae_class:
                return VAEType.VAE_3D
            
            return VAEType.VAE_2D
            
        except (json.JSONDecodeError, IOError) as e:
            self.logger.error(f"Error reading VAE config: {e}")
            return VAEType.UNKNOWN
    
    def validate_component_compatibility(self, components: Dict[str, ComponentInfo]) -> CompatibilityReport:
        """
        Check if all components are compatible with detected architecture.
        
        Args:
            components: Dictionary of component information
            
        Returns:
            CompatibilityReport with validation results
        """
        report = CompatibilityReport(is_compatible=True)
        
        required_components = {"scheduler", "vae"}  # Minimum required
        wan_components = {"transformer", "transformer_2"}
        sd_components = {"unet", "text_encoder", "tokenizer"}
        
        found_components = set(components.keys())
        
        # Check for missing required components
        missing_required = required_components - found_components
        if missing_required:
            report.missing_components.extend(missing_required)
            report.is_compatible = False
            report.recommendations.append(
                f"Missing required components: {', '.join(missing_required)}"
            )
        
        # Determine if this is a Wan or SD model
        has_wan_components = bool(wan_components & found_components)
        has_sd_components = bool(sd_components & found_components)
        
        if has_wan_components and has_sd_components:
            report.warnings.append(
                "Model contains both Wan and SD components - may indicate hybrid model"
            )
        elif has_wan_components:
            # Validate Wan model components
            if "transformer_2" in found_components:
                report.compatible_components.append("transformer_2")
                report.recommendations.append("Use WanPipeline for optimal performance")
            else:
                report.warnings.append("Wan model without transformer_2 may have limited functionality")
        elif has_sd_components:
            # Validate SD model components
            report.compatible_components.extend(found_components & sd_components)
            if len(found_components & sd_components) == len(sd_components):
                report.recommendations.append("Standard StableDiffusionPipeline compatible")
        else:
            report.is_compatible = False
            report.recommendations.append("Unknown model type - manual pipeline selection required")
        
        # Check for custom components
        for name, component in components.items():
            if component.is_custom:
                report.warnings.append(f"Custom component detected: {name}")
                report.recommendations.append("May require trust_remote_code=True")
        
        return report
    
    def _analyze_components(self, model_path: Path) -> Dict[str, ComponentInfo]:
        """Analyze individual model components."""
        components = {}
        
        # Common component directories
        component_dirs = [
            "scheduler", "vae", "unet", "text_encoder", "tokenizer",
            "transformer", "transformer_2", "feature_extractor"
        ]
        
        for component_name in component_dirs:
            component_path = model_path / component_name
            if component_path.exists() and component_path.is_dir():
                config_path = component_path / "config.json"
                
                # Determine component class and properties
                class_name = "Unknown"
                is_custom = False
                
                if config_path.exists():
                    try:
                        with open(config_path, 'r', encoding='utf-8') as f:
                            config = json.load(f)
                        class_name = config.get("_class_name", component_name.title())
                        
                        # Check if this is a custom component
                        is_custom = "Wan" in class_name or class_name not in [
                            "DDIMScheduler", "PNDMScheduler", "LMSDiscreteScheduler",
                            "AutoencoderKL", "UNet2DConditionModel", 
                            "CLIPTextModel", "CLIPTokenizer"
                        ]
                        
                    except (json.JSONDecodeError, IOError):
                        self.logger.warning(f"Could not read config for {component_name}")
                
                components[component_name] = ComponentInfo(
                    class_name=class_name,
                    config_path=str(config_path) if config_path.exists() else "",
                    weight_path=str(component_path),
                    is_custom=is_custom
                )
        
        return components
    
    def _determine_requirements(self, signature: ArchitectureSignature, 
                              architecture_type: ArchitectureType) -> ModelRequirements:
        """Determine model requirements based on architecture."""
        requirements = ModelRequirements()
        
        if architecture_type in [ArchitectureType.WAN_T2V, ArchitectureType.WAN_I2V]:
            # Wan video models require more resources
            requirements.min_vram_mb = 8192
            requirements.recommended_vram_mb = 12288
            requirements.requires_trust_remote_code = True
            requirements.required_dependencies = ["transformers>=4.25.0"]
        elif architecture_type == ArchitectureType.WAN_T2I:
            # Wan image models
            requirements.min_vram_mb = 6144
            requirements.recommended_vram_mb = 8192
            requirements.requires_trust_remote_code = True
            requirements.required_dependencies = ["transformers>=4.25.0"]
        else:
            # Standard SD models
            requirements.min_vram_mb = 4096
            requirements.recommended_vram_mb = 6144
            requirements.requires_trust_remote_code = False
        
        return requirements