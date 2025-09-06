"""
Input Validation Framework for Wan2.2 Video Generation
Provides comprehensive validation for prompts, images, and configuration parameters
"""

print("Starting validation_framework import...")

import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from enum import Enum
from pathlib import Path
import json

print("Basic imports successful")

logger = logging.getLogger(__name__)

print("Logger created")

class ValidationSeverity(Enum):
    """Severity levels for validation issues"""
    ERROR = "error"      # Blocking issues that prevent generation
    WARNING = "warning"  # Issues that may affect quality but don't block
    INFO = "info"       # Informational messages and suggestions

@dataclass
class ValidationIssue:
    """Represents a single validation issue"""
    severity: ValidationSeverity
    message: str
    field: str
    suggestion: Optional[str] = None
    code: Optional[str] = None

@dataclass
class ValidationResult:
    """Result of validation containing all issues and overall status"""
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    
    def add_error(self, message: str, field: str, suggestion: Optional[str] = None, code: Optional[str] = None):
        """Add an error issue"""
        self.issues.append(ValidationIssue(ValidationSeverity.ERROR, message, field, suggestion, code))
        self.is_valid = False
    
    def add_warning(self, message: str, field: str, suggestion: Optional[str] = None, code: Optional[str] = None):
        """Add a warning issue"""
        self.issues.append(ValidationIssue(ValidationSeverity.WARNING, message, field, suggestion, code))
    
    def add_info(self, message: str, field: str, suggestion: Optional[str] = None, code: Optional[str] = None):
        """Add an info issue"""
        self.issues.append(ValidationIssue(ValidationSeverity.INFO, message, field, suggestion, code))
    
    def has_errors(self) -> bool:
        """Check if there are any error-level issues"""
        return any(issue.severity == ValidationSeverity.ERROR for issue in self.issues)
    
    def has_warnings(self) -> bool:
        """Check if there are any warning-level issues"""
        return any(issue.severity == ValidationSeverity.WARNING for issue in self.issues)
    
    def get_errors(self) -> List[ValidationIssue]:
        """Get all error-level issues"""
        return [issue for issue in self.issues if issue.severity == ValidationSeverity.ERROR]
    
    def get_warnings(self) -> List[ValidationIssue]:
        """Get all warning-level issues"""
        return [issue for issue in self.issues if issue.severity == ValidationSeverity.WARNING]
    
    def get_info(self) -> List[ValidationIssue]:
        """Get all info-level issues"""
        return [issue for issue in self.issues if issue.severity == ValidationSeverity.INFO]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "is_valid": self.is_valid,
            "issues": [
                {
                    "severity": issue.severity.value,
                    "message": issue.message,
                    "field": issue.field,
                    "suggestion": issue.suggestion,
                    "code": issue.code
                }
                for issue in self.issues
            ]
        }

class PromptValidator:
    """Validates text prompts for video generation"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.max_length = self.config.get("max_prompt_length", 512)
        self.min_length = self.config.get("min_prompt_length", 3)
        
        # Problematic patterns that may cause generation issues
        self.problematic_patterns = [
            (r'\b(nude|naked|nsfw|explicit|sexual)\b', "NSFW_CONTENT", "Contains potentially inappropriate content"),
            (r'[<>{}[\]\\]', "SPECIAL_CHARS", "Contains special characters that may cause parsing issues"),
            (r'\b(error|fail|crash|bug)\b', "NEGATIVE_TERMS", "Contains negative terms that may affect generation"),
            (r'(.)\1{10,}', "REPETITIVE", "Contains excessive character repetition"),
            (r'\d{10,}', "LONG_NUMBERS", "Contains very long numbers that may cause issues"),
        ]
        
        # Model-specific requirements
        self.model_requirements = {
            "t2v-A14B": {
                "max_length": 512,
                "preferred_style": "descriptive, cinematic",
                "avoid_patterns": ["static", "still image"]
            },
            "i2v-A14B": {
                "max_length": 256,
                "preferred_style": "motion description",
                "avoid_patterns": ["generate image", "create picture"]
            },
            "ti2v-5B": {
                "max_length": 384,
                "preferred_style": "combined text-image description",
                "avoid_patterns": ["ignore image", "text only"]
            }
        }
    
    def validate_prompt(self, prompt: str, model_type: str = "t2v-A14B") -> ValidationResult:
        """Validate a text prompt for the specified model type"""
        result = ValidationResult(is_valid=True)
        
        if not prompt:
            result.add_error("Prompt cannot be empty", "prompt", "Please provide a descriptive text prompt")
            return result
        
        if not isinstance(prompt, str):
            result.add_error("Prompt must be a string", "prompt", "Convert prompt to string format")
            return result
        
        # Check length constraints
        self._validate_length(prompt, model_type, result)
        
        # Check for problematic content
        self._check_problematic_content(prompt, result)
        
        # Model-specific validation
        self._validate_model_specific(prompt, model_type, result)
        
        # Check encoding compatibility
        self._validate_encoding(prompt, result)
        
        # Provide optimization suggestions
        self._suggest_optimizations(prompt, model_type, result)
        
        return result
    
    def _validate_length(self, prompt: str, model_type: str, result: ValidationResult):
        """Validate prompt length constraints"""
        length = len(prompt)
        model_config = self.model_requirements.get(model_type, {})
        max_length = model_config.get("max_length", self.max_length)
        
        if length < self.min_length:
            result.add_error(
                f"Prompt too short ({length} characters, minimum {self.min_length})",
                "prompt",
                "Add more descriptive details to improve generation quality"
            )
        elif length > max_length:
            result.add_error(
                f"Prompt too long ({length} characters, maximum {max_length} for {model_type})",
                "prompt",
                f"Shorten prompt to under {max_length} characters"
            )
        elif length > max_length * 0.8:
            result.add_warning(
                f"Prompt is quite long ({length} characters)",
                "prompt",
                "Consider shortening for better performance"
            )
    
    def _check_problematic_content(self, prompt: str, result: ValidationResult):
        """Check for patterns that may cause generation issues"""
        prompt_lower = prompt.lower()
        
        for pattern, code, description in self.problematic_patterns:
            if re.search(pattern, prompt_lower, re.IGNORECASE):
                result.add_warning(
                    f"Potentially problematic content detected: {description}",
                    "prompt",
                    "Consider rephrasing to avoid generation issues",
                    code
                )
    
    def _validate_model_specific(self, prompt: str, model_type: str, result: ValidationResult):
        """Validate prompt against model-specific requirements"""
        if model_type not in self.model_requirements:
            result.add_warning(
                f"Unknown model type: {model_type}",
                "model_type",
                "Using default validation rules"
            )
            return
        
        model_config = self.model_requirements[model_type]
        prompt_lower = prompt.lower()
        
        # Check for patterns to avoid
        for avoid_pattern in model_config.get("avoid_patterns", []):
            if avoid_pattern.lower() in prompt_lower:
                result.add_warning(
                    f"Prompt contains '{avoid_pattern}' which may not work well with {model_type}",
                    "prompt",
                    f"Consider rephrasing for better results with {model_type}"
                )
        
        # Suggest preferred style
        preferred_style = model_config.get("preferred_style")
        if preferred_style:
            result.add_info(
                f"For best results with {model_type}, use {preferred_style} prompts",
                "prompt",
                f"Consider incorporating {preferred_style} elements"
            )
    
    def _validate_encoding(self, prompt: str, result: ValidationResult):
        """Validate that prompt can be properly encoded"""
        try:
            # Test UTF-8 encoding
            prompt.encode('utf-8')
            
            # Check for unusual Unicode characters that might cause issues
            if any(ord(char) > 65535 for char in prompt):
                result.add_warning(
                    "Prompt contains unusual Unicode characters",
                    "prompt",
                    "Consider using standard ASCII characters for better compatibility"
                )
        except UnicodeEncodeError:
            result.add_error(
                "Prompt contains characters that cannot be encoded",
                "prompt",
                "Remove or replace problematic characters"
            )
    
    def _suggest_optimizations(self, prompt: str, model_type: str, result: ValidationResult):
        """Suggest optimizations for better generation results"""
        # Check for video-specific terms
        video_terms = ["video", "motion", "movement", "animation", "sequence", "frames"]
        has_video_terms = any(term in prompt.lower() for term in video_terms)
        
        if not has_video_terms and model_type.startswith("t2v"):
            result.add_info(
                "Consider adding motion-related terms for better video generation",
                "prompt",
                "Add words like 'moving', 'flowing', 'dynamic' to enhance video quality"
            )
        
        # Check for descriptive adjectives
        if len(prompt.split()) < 5:
            result.add_info(
                "Short prompt detected",
                "prompt",
                "Add more descriptive details for richer video content"
            )

class ImageValidator:
    """Validates images for I2V and TI2V generation"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Supported formats and constraints
        self.supported_formats = self.config.get("supported_formats", ["JPEG", "PNG", "WEBP", "BMP"])
        self.max_file_size_mb = self.config.get("max_file_size_mb", 50)
        self.min_resolution = self.config.get("min_resolution", (256, 256))
        self.max_resolution = self.config.get("max_resolution", (2048, 2048))
        
        # Resolution presets for different models
        self.resolution_presets = {
            "720p": (1280, 720),
            "1080p": (1920, 1080),
            "480p": (854, 480),
            "square": (1024, 1024)
        }
    
    def validate_image(self, image: Union[str, Path, Any], model_type: str = "i2v-A14B") -> ValidationResult:
        """Validate an image for video generation"""
        result = ValidationResult(is_valid=True)
        
        if image is None:
            result.add_error("Image cannot be None", "image", "Provide a valid image")
            return result
        
        # Try to import PIL for full validation
        try:
            from PIL import Image as PILImage
            
            # Load image if path provided
            if isinstance(image, (str, Path)):
                try:
                    image_path = Path(image)
                    if not image_path.exists():
                        result.add_error(f"Image file not found: {image_path}", "image", "Check file path and permissions")
                        return result
                    
                    # Check file size
                    file_size_mb = image_path.stat().st_size / (1024 * 1024)
                    if file_size_mb > self.max_file_size_mb:
                        result.add_error(
                            f"Image file too large ({file_size_mb:.1f}MB, max {self.max_file_size_mb}MB)",
                            "image",
                            "Compress image or use a smaller file"
                        )
                    
                    image = PILImage.open(image_path)
                except Exception as e:
                    result.add_error(f"Failed to load image: {str(e)}", "image", "Check image file format and integrity")
                    return result
            
            if hasattr(image, 'size'):  # PIL Image object
                # Validate dimensions
                width, height = image.size
                min_w, min_h = self.min_resolution
                max_w, max_h = self.max_resolution
                
                # Check minimum resolution
                if width < min_w or height < min_h:
                    result.add_error(
                        f"Image resolution too low ({width}x{height}, minimum {min_w}x{min_h})",
                        "image",
                        "Use a higher resolution image"
                    )
                
                # Check maximum resolution
                if width > max_w or height > max_h:
                    result.add_error(
                        f"Image resolution too high ({width}x{height}, maximum {max_w}x{max_h})",
                        "image",
                        f"Resize image to under {max_w}x{max_h}"
                    )
                
                # Check aspect ratio
                aspect_ratio = width / height
                if aspect_ratio <= 0.5 or aspect_ratio >= 2.0:
                    result.add_warning(
                        f"Unusual aspect ratio ({aspect_ratio:.2f})",
                        "image",
                        "Consider using images with more standard aspect ratios (16:9, 4:3, 1:1)"
                    )
                
                # Validate color mode
                if hasattr(image, 'mode'):
                    if image.mode not in ["RGB", "RGBA"]:
                        if image.mode == "L":
                            result.add_warning(
                                "Grayscale image detected",
                                "image",
                                "Consider using color images for better video generation"
                            )
                        elif image.mode == "P":
                            result.add_info(
                                "Palette mode image detected",
                                "image",
                                "Image will be converted to RGB automatically"
                            )
                        else:
                            result.add_warning(
                                f"Unusual color mode: {image.mode}",
                                "image",
                                "Convert to RGB mode for best compatibility"
                            )
                
                # Model-specific recommendations
                if model_type == "i2v-A14B":
                    result.add_info(
                        "For I2V generation, use clear images with distinct subjects",
                        "image",
                        "Images with clear foreground/background separation work best"
                    )
                elif model_type == "ti2v-5B":
                    result.add_info(
                        "For TI2V generation, ensure image complements the text prompt",
                        "image",
                        "Image should relate to or enhance the text description"
                    )
            else:
                result.add_warning(
                    "Unknown image format, limited validation performed",
                    "image",
                    "Use PIL Image objects for full validation"
                )
        
        except ImportError:
            result.add_warning(
                "PIL not available, limited image validation",
                "image",
                "Install Pillow for full image validation"
            )
        except Exception as e:
            result.add_error(f"Image validation failed: {str(e)}", "image", "Check image format and integrity")
        
        return result

class ConfigValidator:
    """Validates generation configuration parameters"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Parameter constraints
        self.constraints = {
            "steps": {"min": 1, "max": 100, "default": 50},
            "guidance_scale": {"min": 1.0, "max": 20.0, "default": 7.5},
            "strength": {"min": 0.0, "max": 1.0, "default": 0.8},
            "seed": {"min": 0, "max": 2**32 - 1, "default": -1},
            "fps": {"min": 8, "max": 60, "default": 24},
            "duration": {"min": 1, "max": 30, "default": 4}
        }
        
        # Supported resolutions
        self.supported_resolutions = [
            "480p", "720p", "1080p", "square",
            "854x480", "480x854", "1280x720", "1280x704", "1920x1080", "1024x1024"
        ]
        
        # Model-specific constraints
        self.model_constraints = {
            "t2v-A14B": {
                "max_steps": 80,
                "recommended_guidance": 7.5,
                "supported_resolutions": ["720p", "1080p"]
            },
            "i2v-A14B": {
                "max_steps": 60,
                "recommended_guidance": 5.0,
                "supported_resolutions": ["720p", "1080p", "square"]
            },
            "ti2v-5B": {
                "max_steps": 70,
                "recommended_guidance": 6.0,
                "supported_resolutions": ["720p", "1080p"]
            }
        }
    
    def validate_generation_params(self, params: Dict[str, Any], model_type: str = "t2v-A14B") -> ValidationResult:
        """Validate all generation parameters"""
        result = ValidationResult(is_valid=True)
        
        # Validate individual parameters
        for param_name, value in params.items():
            if param_name in self.constraints:
                self._validate_numeric_param(param_name, value, result)
            elif param_name == "resolution":
                self._validate_resolution(value, model_type, result)
            elif param_name == "lora_config":
                self._validate_lora_config(value, result)
            elif param_name == "model_type":
                self._validate_model_type(value, result)
        
        # Check for required parameters
        self._check_required_params(params, model_type, result)
        
        # Validate parameter combinations
        self._validate_param_combinations(params, model_type, result)
        
        # Model-specific validation
        self._validate_model_specific_params(params, model_type, result)
        
        return result
    
    def _validate_numeric_param(self, param_name: str, value: Any, result: ValidationResult):
        """Validate numeric parameters"""
        constraints = self.constraints[param_name]
        
        # Type check
        if not isinstance(value, (int, float)):
            try:
                value = float(value)
            except (ValueError, TypeError):
                result.add_error(
                    f"{param_name} must be a number, got {type(value).__name__}",
                    param_name,
                    f"Use a numeric value between {constraints['min']} and {constraints['max']}"
                )
                return
        
        # Range check
        if value < constraints["min"]:
            result.add_error(
                f"{param_name} too low ({value}, minimum {constraints['min']})",
                param_name,
                f"Use a value >= {constraints['min']}"
            )
        elif value > constraints["max"]:
            result.add_error(
                f"{param_name} too high ({value}, maximum {constraints['max']})",
                param_name,
                f"Use a value <= {constraints['max']}"
            )
        
        # Optimization suggestions
        if param_name == "steps":
            if value > 60:
                result.add_info(
                    f"High step count ({value}) will increase generation time",
                    param_name,
                    "Consider using 30-50 steps for good quality/speed balance"
                )
        elif param_name == "guidance_scale":
            if value > 10:
                result.add_warning(
                    f"High guidance scale ({value}) may cause over-saturation",
                    param_name,
                    "Consider using values between 5-8 for natural results"
                )
    
    def _validate_resolution(self, resolution: str, model_type: str, result: ValidationResult):
        """Validate resolution parameter"""
        if not isinstance(resolution, str):
            result.add_error(
                f"Resolution must be a string, got {type(resolution).__name__}",
                "resolution",
                f"Use one of: {', '.join(self.supported_resolutions)}"
            )
            return
        
        if resolution not in self.supported_resolutions:
            # Check if it's a valid WxH format
            if re.match(r'^\d+x\d+$', resolution):
                try:
                    width, height = map(int, resolution.split('x'))
                    if width < 256 or height < 256:
                        result.add_error(
                            f"Resolution too low: {resolution}",
                            "resolution",
                            "Use minimum 256x256 resolution"
                        )
                    elif width > 2048 or height > 2048:
                        result.add_error(
                            f"Resolution too high: {resolution}",
                            "resolution",
                            "Use maximum 2048x2048 resolution"
                        )
                    else:
                        result.add_info(
                            f"Custom resolution: {resolution}",
                            "resolution",
                            "Consider using standard presets for better performance"
                        )
                except ValueError:
                    result.add_error(
                        f"Invalid resolution format: {resolution}",
                        "resolution",
                        "Use format like '1280x720' or preset names like '720p'"
                    )
            else:
                result.add_error(
                    f"Unsupported resolution: {resolution}",
                    "resolution",
                    f"Use one of: {', '.join(self.supported_resolutions)}"
                )
        
        # Model-specific resolution validation
        if model_type in self.model_constraints:
            supported = self.model_constraints[model_type].get("supported_resolutions", [])
            if supported and resolution not in supported:
                result.add_warning(
                    f"Resolution '{resolution}' may not be optimal for {model_type}",
                    "resolution",
                    f"Recommended resolutions for {model_type}: {', '.join(supported)}"
                )
    
    def _validate_lora_config(self, lora_config: Any, result: ValidationResult):
        """Validate LoRA configuration"""
        if not isinstance(lora_config, dict):
            result.add_error(
                f"LoRA config must be a dictionary, got {type(lora_config).__name__}",
                "lora_config",
                "Use format: {'lora_name': strength_value}"
            )
            return
        
        for lora_name, strength in lora_config.items():
            if not isinstance(lora_name, str):
                result.add_error(
                    f"LoRA name must be string, got {type(lora_name).__name__}",
                    "lora_config",
                    "Use string names for LoRA identifiers"
                )
            
            if not isinstance(strength, (int, float)):
                result.add_error(
                    f"LoRA strength must be numeric, got {type(strength).__name__} for {lora_name}",
                    "lora_config",
                    "Use numeric values between 0.0 and 2.0"
                )
            elif strength < 0.0 or strength > 2.0:
                result.add_warning(
                    f"LoRA strength {strength} for '{lora_name}' is outside typical range (0.0-2.0)",
                    "lora_config",
                    "Consider using strengths between 0.5-1.5 for best results"
                )
    
    def _validate_model_type(self, model_type: str, result: ValidationResult):
        """Validate model type parameter"""
        valid_models = ["t2v-A14B", "i2v-A14B", "ti2v-5B"]
        
        if not isinstance(model_type, str):
            result.add_error(
                f"Model type must be string, got {type(model_type).__name__}",
                "model_type",
                f"Use one of: {', '.join(valid_models)}"
            )
        elif model_type not in valid_models:
            result.add_error(
                f"Unknown model type: {model_type}",
                "model_type",
                f"Use one of: {', '.join(valid_models)}"
            )
    
    def _check_required_params(self, params: Dict[str, Any], model_type: str, result: ValidationResult):
        """Check for required parameters"""
        required_params = ["resolution", "steps"]
        
        # Model-specific required params
        if model_type in ["i2v-A14B", "ti2v-5B"]:
            required_params.append("strength")
        
        for param in required_params:
            if param not in params:
                result.add_error(
                    f"Missing required parameter: {param}",
                    param,
                    f"Provide {param} parameter for {model_type}"
                )
    
    def _validate_param_combinations(self, params: Dict[str, Any], model_type: str, result: ValidationResult):
        """Validate parameter combinations"""
        # High steps + high resolution warning
        steps = params.get("steps", 50)
        resolution = params.get("resolution", "720p")
        
        # Only check combinations if parameters are valid types
        if isinstance(steps, (int, float)) and isinstance(resolution, str):
            if steps > 60 and resolution in ["1080p", "1920x1080"]:
                result.add_warning(
                    "High steps with high resolution will significantly increase generation time",
                    "performance",
                    "Consider reducing steps to 30-50 for 1080p generation"
                )
        
        # LoRA + high guidance warning
        guidance = params.get("guidance_scale", 7.5)
        lora_config = params.get("lora_config", {})
        
        if isinstance(guidance, (int, float)) and lora_config and guidance > 8:
            result.add_warning(
                "High guidance scale with LoRA may cause over-stylization",
                "guidance_scale",
                "Consider reducing guidance to 5-7 when using LoRA"
            )
    
    def _validate_model_specific_params(self, params: Dict[str, Any], model_type: str, result: ValidationResult):
        """Validate parameters against model-specific constraints"""
        if model_type not in self.model_constraints:
            return
        
        constraints = self.model_constraints[model_type]
        
        # Check max steps
        steps = params.get("steps", 50)
        max_steps = constraints.get("max_steps")
        if max_steps and steps > max_steps:
            result.add_warning(
                f"Steps ({steps}) exceed recommended maximum for {model_type} ({max_steps})",
                "steps",
                f"Consider using <= {max_steps} steps for {model_type}"
            )
        
        # Suggest recommended guidance
        guidance = params.get("guidance_scale")
        recommended_guidance = constraints.get("recommended_guidance")
        if guidance and recommended_guidance and abs(guidance - recommended_guidance) > 2:
            result.add_info(
                f"Consider using guidance scale around {recommended_guidance} for {model_type}",
                "guidance_scale",
                f"Current: {guidance}, recommended: {recommended_guidance}"
            )

# Convenience function for comprehensive validation
def validate_generation_request(
    prompt: str,
    image: Optional[Union[str, Path, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
    model_type: str = "t2v-A14B"
) -> ValidationResult:
    """Validate a complete generation request"""
    combined_result = ValidationResult(is_valid=True)
    params = params or {}
    
    # Validate prompt
    prompt_validator = PromptValidator()
    prompt_result = prompt_validator.validate_prompt(prompt, model_type)
    combined_result.issues.extend(prompt_result.issues)
    if not prompt_result.is_valid:
        combined_result.is_valid = False
    
    # Validate image if provided
    if image is not None:
        image_validator = ImageValidator()
        image_result = image_validator.validate_image(image, model_type)
        combined_result.issues.extend(image_result.issues)
        if not image_result.is_valid:
            combined_result.is_valid = False
    
    # Validate configuration
    config_validator = ConfigValidator()
    config_result = config_validator.validate_generation_params(params, model_type)
    combined_result.issues.extend(config_result.issues)
    if not config_result.is_valid:
        combined_result.is_valid = False
    
    return combined_result