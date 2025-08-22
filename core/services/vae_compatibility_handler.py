"""
VAE Compatibility Handler for Wan 2.2 Models

This module handles VAE shape detection, validation, and loading for 3D architectures
to prevent random initialization fallback and handle dimensional mismatches.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VAEDimensions:
    """VAE dimensional information"""
    channels: int
    height: int
    width: int
    depth: Optional[int] = None  # For 3D VAEs
    is_3d: bool = False
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Get the full shape tuple"""
        if self.is_3d and self.depth is not None:
            return (self.channels, self.depth, self.height, self.width)
        return (self.channels, self.height, self.width)


@dataclass
class VAECompatibilityResult:
    """Result of VAE compatibility check"""
    is_compatible: bool
    detected_dimensions: VAEDimensions
    expected_dimensions: Optional[VAEDimensions] = None
    compatibility_issues: List[str] = None
    loading_strategy: str = "standard"  # "standard", "reshape", "custom"
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.compatibility_issues is None:
            self.compatibility_issues = []


@dataclass
class VAELoadingResult:
    """Result of VAE loading attempt"""
    success: bool
    vae_model: Optional[Any] = None
    loading_strategy_used: str = "standard"
    warnings: List[str] = None
    errors: List[str] = None
    fallback_used: bool = False
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.errors is None:
            self.errors = []


class VAECompatibilityHandler:
    """
    Handles VAE compatibility detection and loading for Wan models.
    
    Key responsibilities:
    - Detect VAE architecture (2D vs 3D)
    - Handle shape mismatches gracefully
    - Prevent random initialization fallback
    - Provide specific error messages for VAE issues
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".VAECompatibilityHandler")
        
        # Known VAE configurations for different model types
        self.known_vae_configs = {
            "wan_t2v": VAEDimensions(channels=4, height=64, width=64, depth=16, is_3d=True),
            "wan_t2i": VAEDimensions(channels=4, height=64, width=64, is_3d=False),
            "stable_diffusion": VAEDimensions(channels=4, height=64, width=64, is_3d=False),
        }
        
        # Shape patterns that indicate 3D VAE
        self.shape_3d_indicators = [
            (4, 16, 64, 64),  # Standard Wan T2V VAE shape
            (4, 8, 64, 64),   # Compressed Wan T2V VAE
            (4, 32, 64, 64),  # Extended Wan T2V VAE
        ]
        
        # Shape patterns for dimension mismatches
        self.dimension_mappings = {
            (4, 384, 384): (4, 64, 64),  # Common mismatch pattern
            (4, 512, 512): (4, 64, 64),  # Another common pattern
        }
    
    def detect_vae_architecture(self, vae_config_path: Union[str, Path]) -> VAECompatibilityResult:
        """
        Detect VAE architecture from configuration file.
        
        Args:
            vae_config_path: Path to VAE config.json file
            
        Returns:
            VAECompatibilityResult with detection results
        """
        try:
            config_path = Path(vae_config_path)
            if not config_path.exists():
                return VAECompatibilityResult(
                    is_compatible=False,
                    detected_dimensions=VAEDimensions(0, 0, 0),
                    error_message=f"VAE config file not found: {config_path}"
                )
            
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            return self._analyze_vae_config(config)
            
        except Exception as e:
            self.logger.error(f"Error detecting VAE architecture: {e}")
            return VAECompatibilityResult(
                is_compatible=False,
                detected_dimensions=VAEDimensions(0, 0, 0),
                error_message=f"Failed to analyze VAE config: {str(e)}"
            )
    
    def _analyze_vae_config(self, config: Dict[str, Any]) -> VAECompatibilityResult:
        """Analyze VAE configuration to determine architecture"""
        try:
            # Extract key configuration parameters
            in_channels = config.get('in_channels', 3)
            out_channels = config.get('out_channels', 3)
            latent_channels = config.get('latent_channels', 4)
            
            # Check for 3D-specific parameters
            has_temporal_layers = 'temporal_layers' in config
            has_3d_conv = any('3d' in str(key).lower() for key in config.keys())
            has_depth_param = 'depth' in config or 'temporal_depth' in config
            
            # Determine if this is a 3D VAE
            is_3d = has_temporal_layers or has_3d_conv or has_depth_param
            
            # Extract dimensions
            sample_size = config.get('sample_size', 64)
            if isinstance(sample_size, list):
                if len(sample_size) == 2:
                    height, width = sample_size
                    depth = None
                elif len(sample_size) == 3:
                    depth, height, width = sample_size
                    is_3d = True
                else:
                    height = width = sample_size[0] if sample_size else 64
                    depth = None
            else:
                height = width = sample_size
                depth = config.get('temporal_depth', 16 if is_3d else None)
            
            detected_dims = VAEDimensions(
                channels=latent_channels,
                height=height,
                width=width,
                depth=depth,
                is_3d=is_3d
            )
            
            # Check compatibility
            compatibility_issues = []
            loading_strategy = "standard"
            
            # Check for known problematic patterns
            if detected_dims.shape in [(4, 384, 384), (4, 512, 512)]:
                compatibility_issues.append(
                    f"Detected problematic VAE shape {detected_dims.shape}, may need reshaping"
                )
                loading_strategy = "reshape"
            
            # Check for 3D architecture requirements
            if is_3d and depth is None:
                compatibility_issues.append("3D VAE detected but temporal depth not specified")
                loading_strategy = "custom"
            
            is_compatible = len(compatibility_issues) == 0
            
            return VAECompatibilityResult(
                is_compatible=is_compatible,
                detected_dimensions=detected_dims,
                compatibility_issues=compatibility_issues,
                loading_strategy=loading_strategy
            )
            
        except Exception as e:
            return VAECompatibilityResult(
                is_compatible=False,
                detected_dimensions=VAEDimensions(0, 0, 0),
                error_message=f"Error analyzing VAE config: {str(e)}"
            )
    
    def validate_vae_weights(self, weights_path: Union[str, Path], 
                           expected_dims: VAEDimensions) -> VAECompatibilityResult:
        """
        Validate VAE weights against expected dimensions.
        
        Args:
            weights_path: Path to VAE weights file
            expected_dims: Expected VAE dimensions
            
        Returns:
            VAECompatibilityResult with validation results
        """
        try:
            weights_path = Path(weights_path)
            if not weights_path.exists():
                return VAECompatibilityResult(
                    is_compatible=False,
                    detected_dimensions=VAEDimensions(0, 0, 0),
                    expected_dimensions=expected_dims,
                    error_message=f"VAE weights file not found: {weights_path}"
                )
            
            # Load weights to check shapes
            if weights_path.suffix == '.safetensors':
                try:
                    from safetensors import safe_open
                    with safe_open(weights_path, framework="pt") as f:
                        weight_shapes = {key: f.get_tensor(key).shape for key in f.keys()}
                except ImportError:
                    self.logger.warning("safetensors not available, falling back to torch.load")
                    state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
                    weight_shapes = {key: tensor.shape for key, tensor in state_dict.items()}
            else:
                state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
                weight_shapes = {key: tensor.shape for key, tensor in state_dict.items()}
            
            return self._validate_weight_shapes(weight_shapes, expected_dims)
            
        except Exception as e:
            self.logger.error(f"Error validating VAE weights: {e}")
            return VAECompatibilityResult(
                is_compatible=False,
                detected_dimensions=VAEDimensions(0, 0, 0),
                expected_dimensions=expected_dims,
                error_message=f"Failed to validate VAE weights: {str(e)}"
            )
    
    def _validate_weight_shapes(self, weight_shapes: Dict[str, Tuple], 
                              expected_dims: VAEDimensions) -> VAECompatibilityResult:
        """Validate weight shapes against expected dimensions"""
        compatibility_issues = []
        loading_strategy = "standard"
        
        # Look for encoder/decoder layer shapes to infer architecture
        encoder_shapes = {k: v for k, v in weight_shapes.items() if 'encoder' in k.lower()}
        decoder_shapes = {k: v for k, v in weight_shapes.items() if 'decoder' in k.lower()}
        
        # Check for 3D convolution patterns
        has_3d_weights = any(
            len(shape) == 5 for shape in weight_shapes.values()  # 3D conv weights have 5 dimensions
        )
        
        # Detect actual dimensions from weights
        detected_is_3d = has_3d_weights
        detected_channels = expected_dims.channels  # Default fallback
        
        # Try to infer channels from first layer
        for key, shape in weight_shapes.items():
            if 'conv_in' in key or 'first' in key:
                if len(shape) >= 2:
                    detected_channels = shape[1]  # Input channels
                break
        
        detected_dims = VAEDimensions(
            channels=detected_channels,
            height=expected_dims.height,
            width=expected_dims.width,
            depth=expected_dims.depth if detected_is_3d else None,
            is_3d=detected_is_3d
        )
        
        # Check for dimension mismatches
        if expected_dims.is_3d != detected_is_3d:
            compatibility_issues.append(
                f"Architecture mismatch: expected {'3D' if expected_dims.is_3d else '2D'} "
                f"but detected {'3D' if detected_is_3d else '2D'} VAE"
            )
            loading_strategy = "custom"
        
        # Check for shape mismatches that need special handling
        problematic_shapes = [(4, 384, 384), (4, 512, 512)]
        if detected_dims.shape[:3] in problematic_shapes:
            compatibility_issues.append(
                f"Detected problematic shape {detected_dims.shape[:3]}, will attempt reshape to (4, 64, 64)"
            )
            loading_strategy = "reshape"
        
        is_compatible = len(compatibility_issues) == 0 or loading_strategy in ["reshape", "custom"]
        
        return VAECompatibilityResult(
            is_compatible=is_compatible,
            detected_dimensions=detected_dims,
            expected_dimensions=expected_dims,
            compatibility_issues=compatibility_issues,
            loading_strategy=loading_strategy
        )
    
    def load_vae_with_compatibility(self, vae_path: Union[str, Path], 
                                  compatibility_result: VAECompatibilityResult) -> VAELoadingResult:
        """
        Load VAE with compatibility handling based on detection results.
        
        Args:
            vae_path: Path to VAE model directory
            compatibility_result: Result from compatibility detection
            
        Returns:
            VAELoadingResult with loading outcome
        """
        try:
            vae_path = Path(vae_path)
            
            if compatibility_result.loading_strategy == "standard":
                return self._load_vae_standard(vae_path)
            elif compatibility_result.loading_strategy == "reshape":
                return self._load_vae_with_reshape(vae_path, compatibility_result)
            elif compatibility_result.loading_strategy == "custom":
                return self._load_vae_custom(vae_path, compatibility_result)
            else:
                return VAELoadingResult(
                    success=False,
                    errors=[f"Unknown loading strategy: {compatibility_result.loading_strategy}"]
                )
                
        except Exception as e:
            self.logger.error(f"Error loading VAE: {e}")
            return VAELoadingResult(
                success=False,
                errors=[f"Failed to load VAE: {str(e)}"]
            )
    
    def _load_vae_standard(self, vae_path: Path) -> VAELoadingResult:
        """Load VAE using standard Diffusers loading"""
        try:
            from diffusers import AutoencoderKL
            
            vae = AutoencoderKL.from_pretrained(str(vae_path))
            
            return VAELoadingResult(
                success=True,
                vae_model=vae,
                loading_strategy_used="standard"
            )
            
        except Exception as e:
            return VAELoadingResult(
                success=False,
                errors=[f"Standard VAE loading failed: {str(e)}"]
            )
    
    def _load_vae_with_reshape(self, vae_path: Path, 
                             compatibility_result: VAECompatibilityResult) -> VAELoadingResult:
        """Load VAE with shape reshaping to handle dimension mismatches"""
        try:
            from diffusers import AutoencoderKL
            
            # First try standard loading
            try:
                vae = AutoencoderKL.from_pretrained(str(vae_path))
                
                # Check if reshaping is needed
                if hasattr(vae.config, 'sample_size'):
                    current_size = vae.config.sample_size
                    if isinstance(current_size, (list, tuple)) and len(current_size) >= 2:
                        if current_size[0] == 384 or current_size[1] == 384:
                            # Apply reshape logic
                            self.logger.info(f"Reshaping VAE from {current_size} to (64, 64)")
                            vae.config.sample_size = [64, 64]
                
                return VAELoadingResult(
                    success=True,
                    vae_model=vae,
                    loading_strategy_used="reshape",
                    warnings=[f"Applied reshape handling for VAE dimensions"]
                )
                
            except Exception as e:
                return VAELoadingResult(
                    success=False,
                    errors=[f"VAE reshape loading failed: {str(e)}"]
                )
                
        except Exception as e:
            return VAELoadingResult(
                success=False,
                errors=[f"VAE reshape strategy failed: {str(e)}"]
            )
    
    def _load_vae_custom(self, vae_path: Path, 
                        compatibility_result: VAECompatibilityResult) -> VAELoadingResult:
        """Load VAE with custom handling for 3D architectures"""
        try:
            # For 3D VAEs, we need special handling
            if compatibility_result.detected_dimensions.is_3d:
                return self._load_3d_vae(vae_path, compatibility_result)
            else:
                # Fall back to standard loading with warnings
                result = self._load_vae_standard(vae_path)
                if result.success:
                    result.loading_strategy_used = "custom"
                    result.warnings.append("Used custom loading strategy")
                return result
                
        except Exception as e:
            return VAELoadingResult(
                success=False,
                errors=[f"Custom VAE loading failed: {str(e)}"]
            )
    
    def _load_3d_vae(self, vae_path: Path, 
                    compatibility_result: VAECompatibilityResult) -> VAELoadingResult:
        """Load 3D VAE with special handling"""
        try:
            # Try to load with trust_remote_code for custom 3D VAE classes
            from diffusers import AutoencoderKL
            
            try:
                # First attempt: try with trust_remote_code
                vae = AutoencoderKL.from_pretrained(
                    str(vae_path), 
                    trust_remote_code=True
                )
                
                return VAELoadingResult(
                    success=True,
                    vae_model=vae,
                    loading_strategy_used="custom_3d",
                    warnings=["Loaded 3D VAE with trust_remote_code=True"]
                )
                
            except Exception as e:
                # Second attempt: try standard loading
                self.logger.warning(f"3D VAE loading with trust_remote_code failed: {e}")
                
                vae = AutoencoderKL.from_pretrained(str(vae_path))
                
                return VAELoadingResult(
                    success=True,
                    vae_model=vae,
                    loading_strategy_used="custom_fallback",
                    warnings=[
                        "3D VAE loaded with standard method, may have compatibility issues",
                        f"Original error: {str(e)}"
                    ],
                    fallback_used=True
                )
                
        except Exception as e:
            return VAELoadingResult(
                success=False,
                errors=[f"3D VAE loading failed: {str(e)}"]
            )
    
    def get_vae_error_guidance(self, compatibility_result: VAECompatibilityResult, 
                             loading_result: Optional[VAELoadingResult] = None) -> List[str]:
        """
        Generate user-friendly error guidance for VAE compatibility issues.
        
        Args:
            compatibility_result: Result from compatibility detection
            loading_result: Optional result from loading attempt
            
        Returns:
            List of guidance messages
        """
        guidance = []
        
        if not compatibility_result.is_compatible:
            guidance.append("VAE Compatibility Issues Detected:")
            
            for issue in compatibility_result.compatibility_issues:
                guidance.append(f"  • {issue}")
            
            if compatibility_result.error_message:
                guidance.append(f"  • Error: {compatibility_result.error_message}")
        
        if loading_result and not loading_result.success:
            guidance.append("VAE Loading Failed:")
            
            for error in loading_result.errors:
                guidance.append(f"  • {error}")
        
        # Provide specific guidance based on detected issues
        if any("384" in issue for issue in compatibility_result.compatibility_issues):
            guidance.extend([
                "",
                "Recommended Solutions for Shape Mismatch:",
                "  1. The VAE has non-standard dimensions (384x384 instead of 64x64)",
                "  2. This is common with Wan models and should be handled automatically",
                "  3. If loading fails, try using trust_remote_code=True",
                "  4. Ensure you have the latest version of diffusers installed"
            ])
        
        if compatibility_result.detected_dimensions.is_3d:
            guidance.extend([
                "",
                "3D VAE Detected:",
                "  1. This VAE is designed for video generation (3D architecture)",
                "  2. Requires custom pipeline code (WanPipeline)",
                "  3. May need trust_remote_code=True for proper loading",
                "  4. Ensure sufficient VRAM (8GB+ recommended)"
            ])
        
        if loading_result and loading_result.fallback_used:
            guidance.extend([
                "",
                "Fallback Loading Used:",
                "  1. VAE loaded with reduced functionality",
                "  2. Some features may not work as expected",
                "  3. Consider updating dependencies or model files",
                "  4. Check for custom pipeline requirements"
            ])
        
        return guidance


def create_vae_compatibility_handler() -> VAECompatibilityHandler:
    """Factory function to create VAE compatibility handler"""
    return VAECompatibilityHandler()