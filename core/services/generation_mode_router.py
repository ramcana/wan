"""
Generation Mode Router for Wan2.2 Video Generation
Handles routing between T2V, I2V, and TI2V modes with proper validation
"""

import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
from pathlib import Path

from input_validation import ValidationResult, PromptValidator, ImageValidator, ConfigValidator
from generation_orchestrator import GenerationRequest, GenerationMode
from infrastructure.hardware.error_handler import ErrorCategory, ErrorSeverity, UserFriendlyError

logger = logging.getLogger(__name__)

class GenerationModeType(Enum):
    """Generation mode types with validation requirements"""
    TEXT_TO_VIDEO = "t2v-A14B"
    IMAGE_TO_VIDEO = "i2v-A14B" 
    TEXT_IMAGE_TO_VIDEO = "ti2v-5B"

@dataclass
class ModeRequirements:
    """Requirements for a specific generation mode"""
    requires_prompt: bool
    requires_image: bool
    supports_lora: bool
    supported_resolutions: List[str]
    min_steps: int
    max_steps: int
    default_steps: int
    guidance_scale_range: Tuple[float, float]
    default_guidance_scale: float

@dataclass
class ModeValidationResult:
    """Result of mode-specific validation"""
    is_valid: bool
    mode: GenerationModeType
    validation_issues: List[str]
    warnings: List[str]
    optimized_request: Optional[GenerationRequest] = None

class GenerationModeRouter:
    """Routes generation requests to appropriate modes with validation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize validators
        self.prompt_validator = PromptValidator(config)
        self.image_validator = ImageValidator(config)
        self.config_validator = ConfigValidator(config)
        
        # Define mode requirements
        self.mode_requirements = {
            GenerationModeType.TEXT_TO_VIDEO: ModeRequirements(
                requires_prompt=True,
                requires_image=False,
                supports_lora=True,
                supported_resolutions=["720p", "1080p", "480p"],
                min_steps=10,
                max_steps=100,
                default_steps=50,
                guidance_scale_range=(1.0, 20.0),
                default_guidance_scale=7.5
            ),
            GenerationModeType.IMAGE_TO_VIDEO: ModeRequirements(
                requires_prompt=False,  # Optional for I2V
                requires_image=True,
                supports_lora=True,
                supported_resolutions=["720p", "1080p", "480p"],
                min_steps=10,
                max_steps=80,
                default_steps=40,
                guidance_scale_range=(1.0, 15.0),
                default_guidance_scale=7.5
            ),
            GenerationModeType.TEXT_IMAGE_TO_VIDEO: ModeRequirements(
                requires_prompt=True,
                requires_image=True,
                supports_lora=False,  # TI2V model doesn't support LoRA
                supported_resolutions=["720p", "480p"],  # Limited resolution support
                min_steps=15,
                max_steps=60,
                default_steps=30,
                guidance_scale_range=(1.0, 12.0),
                default_guidance_scale=7.5
            )
        }
        
        logger.info("Generation mode router initialized")
    
    def route_request(self, request: GenerationRequest) -> ModeValidationResult:
        """Route and validate a generation request based on its mode"""
        try:
            # Determine the generation mode
            mode = self._determine_mode(request)
            
            # Get mode requirements
            requirements = self.mode_requirements[mode]
            
            # Validate request against mode requirements
            validation_result = self._validate_mode_requirements(request, mode, requirements)
            
            if not validation_result.is_valid:
                return validation_result
            
            # Optimize request for the specific mode
            optimized_request = self._optimize_for_mode(request, mode, requirements)
            
            return ModeValidationResult(
                is_valid=True,
                mode=mode,
                validation_issues=[],
                warnings=validation_result.warnings,
                optimized_request=optimized_request
            )
            
        except Exception as e:
            logger.error(f"Mode routing failed: {e}")
            return ModeValidationResult(
                is_valid=False,
                mode=GenerationModeType.TEXT_TO_VIDEO,  # Default fallback
                validation_issues=[f"Mode routing error: {str(e)}"],
                warnings=[]
            )
    
    def _determine_mode(self, request: GenerationRequest) -> GenerationModeType:
        """Determine the appropriate generation mode based on request"""
        # Check if model type is explicitly specified
        if request.model_type in [mode.value for mode in GenerationModeType]:
            for mode in GenerationModeType:
                if mode.value == request.model_type:
                    return mode
        
        # Auto-detect mode based on inputs
        has_prompt = bool(request.prompt and request.prompt.strip())
        has_image = request.image is not None
        
        if has_prompt and has_image:
            return GenerationModeType.TEXT_IMAGE_TO_VIDEO
        elif has_image and not has_prompt:
            return GenerationModeType.IMAGE_TO_VIDEO
        elif has_prompt and not has_image:
            return GenerationModeType.TEXT_TO_VIDEO
        else:
            # Default to T2V if unclear
            logger.warning("Could not determine mode, defaulting to T2V")
            return GenerationModeType.TEXT_TO_VIDEO
    
    def _validate_mode_requirements(self, request: GenerationRequest, 
                                  mode: GenerationModeType, 
                                  requirements: ModeRequirements) -> ModeValidationResult:
        """Validate request against mode-specific requirements"""
        validation_issues = []
        warnings = []
        
        # Check prompt requirements
        if requirements.requires_prompt:
            if not request.prompt or not request.prompt.strip():
                validation_issues.append(f"{mode.value} mode requires a text prompt")
            else:
                # Validate prompt content
                prompt_result = self.prompt_validator.validate(request.prompt)
                if not prompt_result.is_valid:
                    for issue in prompt_result.get_errors():
                        validation_issues.append(f"Prompt validation: {issue.message}")
                
                for issue in prompt_result.get_warnings():
                    warnings.append(f"Prompt warning: {issue.message}")
        
        # Check image requirements
        if requirements.requires_image:
            if request.image is None:
                validation_issues.append(f"{mode.value} mode requires an input image")
            else:
                # Validate image
                image_result = self.image_validator.validate(request.image)
                if not image_result.is_valid:
                    for issue in image_result.get_errors():
                        validation_issues.append(f"Image validation: {issue.message}")
                
                for issue in image_result.get_warnings():
                    warnings.append(f"Image warning: {issue.message}")
        
        # Check resolution support
        if request.resolution not in requirements.supported_resolutions:
            validation_issues.append(
                f"Resolution {request.resolution} not supported by {mode.value}. "
                f"Supported: {', '.join(requirements.supported_resolutions)}"
            )
        
        # Check steps range
        if not (requirements.min_steps <= request.steps <= requirements.max_steps):
            validation_issues.append(
                f"Steps {request.steps} outside valid range for {mode.value} "
                f"({requirements.min_steps}-{requirements.max_steps})"
            )
        
        # Check guidance scale range
        min_gs, max_gs = requirements.guidance_scale_range
        if not (min_gs <= request.guidance_scale <= max_gs):
            validation_issues.append(
                f"Guidance scale {request.guidance_scale} outside valid range for {mode.value} "
                f"({min_gs}-{max_gs})"
            )
        
        # Check LoRA support
        if request.lora_config and not requirements.supports_lora:
            validation_issues.append(f"{mode.value} mode does not support LoRA")
        
        return ModeValidationResult(
            is_valid=len(validation_issues) == 0,
            mode=mode,
            validation_issues=validation_issues,
            warnings=warnings
        )
    
    def _optimize_for_mode(self, request: GenerationRequest, 
                          mode: GenerationModeType, 
                          requirements: ModeRequirements) -> GenerationRequest:
        """Optimize request parameters for the specific mode"""
        optimized = GenerationRequest(
            model_type=mode.value,
            prompt=request.prompt,
            image=request.image,
            resolution=request.resolution,
            steps=request.steps,
            guidance_scale=request.guidance_scale,
            strength=request.strength,
            seed=request.seed,
            fps=request.fps,
            duration=request.duration,
            lora_config=request.lora_config.copy() if request.lora_config else {},
            optimization_settings=request.optimization_settings.copy() if request.optimization_settings else {}
        )
        
        # Apply mode-specific optimizations
        if mode == GenerationModeType.TEXT_TO_VIDEO:
            # T2V optimizations
            if optimized.steps < 30:
                optimized.steps = max(30, optimized.steps)  # Minimum for good quality
            
            # Adjust guidance scale for better prompt adherence
            if optimized.guidance_scale < 5.0:
                optimized.guidance_scale = 7.5
        
        elif mode == GenerationModeType.IMAGE_TO_VIDEO:
            # I2V optimizations
            if optimized.steps > 60:
                optimized.steps = 50  # I2V doesn't need as many steps
            
            # Adjust strength for image conditioning
            if optimized.strength > 0.9:
                optimized.strength = 0.8  # Prevent over-conditioning
        
        elif mode == GenerationModeType.TEXT_IMAGE_TO_VIDEO:
            # TI2V optimizations
            if optimized.steps > 40:
                optimized.steps = 35  # TI2V is more efficient
            
            # Remove LoRA config as it's not supported
            optimized.lora_config = {}
            
            # Adjust guidance scale for dual conditioning
            if optimized.guidance_scale > 10.0:
                optimized.guidance_scale = 8.0
        
        # Apply resolution-specific optimizations
        if optimized.resolution == "1080p":
            # Higher resolution needs more steps for quality
            optimized.steps = max(optimized.steps, 40)
        elif optimized.resolution == "480p":
            # Lower resolution can use fewer steps
            optimized.steps = min(optimized.steps, 40)
        
        return optimized
    
    def get_mode_info(self, mode: GenerationModeType) -> Dict[str, Any]:
        """Get information about a specific generation mode"""
        requirements = self.mode_requirements[mode]
        
        return {
            "mode": mode.value,
            "name": self._get_mode_display_name(mode),
            "description": self._get_mode_description(mode),
            "requirements": {
                "requires_prompt": requirements.requires_prompt,
                "requires_image": requirements.requires_image,
                "supports_lora": requirements.supports_lora
            },
            "parameters": {
                "supported_resolutions": requirements.supported_resolutions,
                "steps_range": [requirements.min_steps, requirements.max_steps],
                "default_steps": requirements.default_steps,
                "guidance_scale_range": list(requirements.guidance_scale_range),
                "default_guidance_scale": requirements.default_guidance_scale
            }
        }
    
    def _get_mode_display_name(self, mode: GenerationModeType) -> str:
        """Get display name for a mode"""
        names = {
            GenerationModeType.TEXT_TO_VIDEO: "Text-to-Video",
            GenerationModeType.IMAGE_TO_VIDEO: "Image-to-Video",
            GenerationModeType.TEXT_IMAGE_TO_VIDEO: "Text+Image-to-Video"
        }
        return names[mode]
    
    def _get_mode_description(self, mode: GenerationModeType) -> str:
        """Get description for a mode"""
        descriptions = {
            GenerationModeType.TEXT_TO_VIDEO: "Generate videos from text prompts only",
            GenerationModeType.IMAGE_TO_VIDEO: "Generate videos from input images with optional text guidance",
            GenerationModeType.TEXT_IMAGE_TO_VIDEO: "Generate videos using both text prompts and input images"
        }
        return descriptions[mode]
    
    def list_available_modes(self) -> List[Dict[str, Any]]:
        """List all available generation modes with their information"""
        return [self.get_mode_info(mode) for mode in GenerationModeType]
    
    def validate_mode_compatibility(self, request: GenerationRequest) -> Tuple[bool, List[str]]:
        """Quick validation of mode compatibility"""
        try:
            result = self.route_request(request)
            return result.is_valid, result.validation_issues
        except Exception as e:
            return False, [f"Validation error: {str(e)}"]
    
    def suggest_mode_for_inputs(self, has_prompt: bool, has_image: bool) -> GenerationModeType:
        """Suggest the best mode based on available inputs"""
        if has_prompt and has_image:
            return GenerationModeType.TEXT_IMAGE_TO_VIDEO
        elif has_image:
            return GenerationModeType.IMAGE_TO_VIDEO
        elif has_prompt:
            return GenerationModeType.TEXT_TO_VIDEO
        else:
            return GenerationModeType.TEXT_TO_VIDEO  # Default
    
    def get_mode_specific_validation_rules(self, mode: GenerationModeType) -> Dict[str, Any]:
        """Get validation rules specific to a mode"""
        requirements = self.mode_requirements[mode]
        
        return {
            "prompt": {
                "required": requirements.requires_prompt,
                "max_length": self.config.get("generation", {}).get("max_prompt_length", 512)
            },
            "image": {
                "required": requirements.requires_image,
                "supported_formats": ["PNG", "JPG", "JPEG", "WEBP"],
                "max_size_mb": 10
            },
            "resolution": {
                "supported": requirements.supported_resolutions,
                "default": "720p"
            },
            "steps": {
                "min": requirements.min_steps,
                "max": requirements.max_steps,
                "default": requirements.default_steps
            },
            "guidance_scale": {
                "min": requirements.guidance_scale_range[0],
                "max": requirements.guidance_scale_range[1],
                "default": requirements.default_guidance_scale
            },
            "lora": {
                "supported": requirements.supports_lora,
                "max_count": 5 if requirements.supports_lora else 0
            }
        }


# Global router instance
_router_instance = None

def get_generation_mode_router(config: Dict[str, Any]) -> GenerationModeRouter:
    """Get the global generation mode router instance"""
    global _router_instance
    if _router_instance is None:
        _router_instance = GenerationModeRouter(config)
    return _router_instance