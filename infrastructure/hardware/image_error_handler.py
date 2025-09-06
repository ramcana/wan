"""
Comprehensive Error Handling System for Image Operations
Provides specific error handling, recovery suggestions, and user-friendly error messages
"""

import logging
import traceback
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, Union
from enum import Enum
import os
import sys

logger = logging.getLogger(__name__)

class ImageErrorType(Enum):
    """Enumeration of specific image error types"""
    # Format-related errors
    UNSUPPORTED_FORMAT = "unsupported_format"
    CORRUPTED_FILE = "corrupted_file"
    INVALID_FILE_STRUCTURE = "invalid_file_structure"
    
    # Dimension-related errors
    TOO_SMALL = "too_small"
    TOO_LARGE = "too_large"
    INVALID_DIMENSIONS = "invalid_dimensions"
    ASPECT_RATIO_MISMATCH = "aspect_ratio_mismatch"
    
    # Size-related errors
    FILE_TOO_LARGE = "file_too_large"
    MEMORY_INSUFFICIENT = "memory_insufficient"
    
    # Quality-related errors
    TOO_DARK = "too_dark"
    TOO_BRIGHT = "too_bright"
    LOW_CONTRAST = "low_contrast"
    BLURRY_IMAGE = "blurry_image"
    
    # Compatibility errors
    INCOMPATIBLE_IMAGES = "incompatible_images"
    COLOR_MODE_MISMATCH = "color_mode_mismatch"
    
    # System errors
    PIL_NOT_AVAILABLE = "pil_not_available"
    NUMPY_NOT_AVAILABLE = "numpy_not_available"
    PERMISSION_DENIED = "permission_denied"
    DISK_SPACE_INSUFFICIENT = "disk_space_insufficient"
    
    # Processing errors
    THUMBNAIL_GENERATION_FAILED = "thumbnail_generation_failed"
    VALIDATION_FAILED = "validation_failed"
    METADATA_EXTRACTION_FAILED = "metadata_extraction_failed"
    
    # Unknown errors
    UNKNOWN_ERROR = "unknown_error"

@dataclass
class ErrorContext:
    """Context information for error handling"""
    image_type: str = "unknown"  # "start", "end", "unknown"
    model_type: str = "unknown"  # "t2v-A14B", "i2v-A14B", "ti2v-5B"
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    dimensions: Optional[Tuple[int, int]] = None
    format: Optional[str] = None
    operation: str = "validation"  # "validation", "upload", "processing"
    user_action: str = "upload"  # "upload", "replace", "clear"

@dataclass
class RecoveryAction:
    """Specific recovery action with instructions"""
    action_type: str  # "retry", "convert", "resize", "replace", "install"
    title: str
    description: str
    instructions: List[str] = field(default_factory=list)
    tools_needed: List[str] = field(default_factory=list)
    estimated_time: str = "1-2 minutes"
    difficulty: str = "easy"  # "easy", "medium", "hard"

@dataclass
class ImageError:
    """Comprehensive image error with recovery suggestions"""
    error_type: ImageErrorType
    severity: str  # "error", "warning", "info"
    title: str
    message: str
    technical_details: str = ""
    context: Optional[ErrorContext] = None
    recovery_actions: List[RecoveryAction] = field(default_factory=list)
    prevention_tips: List[str] = field(default_factory=list)
    related_errors: List[ImageErrorType] = field(default_factory=list)
    
    def to_user_friendly_dict(self) -> Dict[str, Any]:
        """Convert to user-friendly dictionary"""
        return {
            "error_type": self.error_type.value,
            "severity": self.severity,
            "title": self.title,
            "message": self.message,
            "recovery_actions": [
                {
                    "type": action.action_type,
                    "title": action.title,
                    "description": action.description,
                    "instructions": action.instructions,
                    "tools_needed": action.tools_needed,
                    "estimated_time": action.estimated_time,
                    "difficulty": action.difficulty
                }
                for action in self.recovery_actions
            ],
            "prevention_tips": self.prevention_tips,
            "context": {
                "image_type": self.context.image_type if self.context else "unknown",
                "model_type": self.context.model_type if self.context else "unknown",
                "operation": self.context.operation if self.context else "validation"
            }
        }

class ImageErrorHandler:
    """Comprehensive error handler for image operations"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.error_registry = self._build_error_registry()
        self.recovery_registry = self._build_recovery_registry()
    
    def handle_format_error(self, format_name: str, context: ErrorContext) -> ImageError:
        """Handle image format-related errors"""
        if format_name.upper() not in ["JPEG", "PNG", "WEBP", "BMP", "JPG"]:
            return self._create_error(
                ImageErrorType.UNSUPPORTED_FORMAT,
                context,
                format_name=format_name
            )
        return self._create_error(ImageErrorType.CORRUPTED_FILE, context)
    
    def handle_dimension_error(self, dimensions: Tuple[int, int], 
                             min_dims: Tuple[int, int], max_dims: Tuple[int, int],
                             context: ErrorContext) -> ImageError:
        """Handle dimension-related errors"""
        width, height = dimensions
        min_w, min_h = min_dims
        max_w, max_h = max_dims
        
        if width < min_w or height < min_h:
            return self._create_error(
                ImageErrorType.TOO_SMALL,
                context,
                current_dims=dimensions,
                required_dims=min_dims
            )
        elif width > max_w or height > max_h:
            return self._create_error(
                ImageErrorType.TOO_LARGE,
                context,
                current_dims=dimensions,
                max_dims=max_dims
            )
        else:
            return self._create_error(ImageErrorType.INVALID_DIMENSIONS, context)
    
    def handle_file_size_error(self, file_size_mb: float, max_size_mb: float,
                              context: ErrorContext) -> ImageError:
        """Handle file size errors"""
        return self._create_error(
            ImageErrorType.FILE_TOO_LARGE,
            context,
            current_size=file_size_mb,
            max_size=max_size_mb
        )
    
    def handle_quality_error(self, quality_issue: str, value: float,
                           context: ErrorContext) -> ImageError:
        """Handle image quality errors"""
        quality_map = {
            "brightness_low": ImageErrorType.TOO_DARK,
            "brightness_high": ImageErrorType.TOO_BRIGHT,
            "contrast_low": ImageErrorType.LOW_CONTRAST,
            "blur_high": ImageErrorType.BLURRY_IMAGE
        }
        
        error_type = quality_map.get(quality_issue, ImageErrorType.UNKNOWN_ERROR)
        return self._create_error(error_type, context, quality_value=value)
    
    def handle_compatibility_error(self, start_dims: Tuple[int, int], 
                                 end_dims: Tuple[int, int],
                                 context: ErrorContext) -> ImageError:
        """Handle image compatibility errors"""
        return self._create_error(
            ImageErrorType.INCOMPATIBLE_IMAGES,
            context,
            start_dims=start_dims,
            end_dims=end_dims
        )
    
    def handle_system_error(self, error_type: ImageErrorType, 
                          exception: Optional[Exception] = None,
                          context: Optional[ErrorContext] = None) -> ImageError:
        """Handle system-level errors"""
        if context is None:
            context = ErrorContext()
        
        return self._create_error(
            error_type,
            context,
            exception=exception
        )
    
    def handle_processing_error(self, operation: str, exception: Exception,
                              context: ErrorContext) -> ImageError:
        """Handle processing errors with exception details"""
        # Determine error type based on exception
        error_type = self._classify_exception(exception)
        
        return self._create_error(
            error_type,
            context,
            operation=operation,
            exception=exception
        )
    
    def create_validation_summary(self, errors: List[ImageError]) -> Dict[str, Any]:
        """Create a comprehensive validation summary"""
        if not errors:
            return {
                "status": "success",
                "message": "All validations passed",
                "errors": [],
                "warnings": [],
                "suggestions": []
            }
        
        error_list = [e for e in errors if e.severity == "error"]
        warning_list = [e for e in errors if e.severity == "warning"]
        
        # Collect all recovery actions
        all_actions = []
        for error in errors:
            all_actions.extend(error.recovery_actions)
        
        # Deduplicate actions by type and title
        unique_actions = []
        seen = set()
        for action in all_actions:
            key = (action.action_type, action.title)
            if key not in seen:
                unique_actions.append(action)
                seen.add(key)
        
        return {
            "status": "error" if error_list else "warning",
            "message": self._generate_summary_message(error_list, warning_list),
            "errors": [e.to_user_friendly_dict() for e in error_list],
            "warnings": [e.to_user_friendly_dict() for e in warning_list],
            "recovery_actions": [
                {
                    "type": action.action_type,
                    "title": action.title,
                    "description": action.description,
                    "instructions": action.instructions,
                    "difficulty": action.difficulty,
                    "estimated_time": action.estimated_time
                }
                for action in unique_actions
            ],
            "prevention_tips": self._collect_prevention_tips(errors)
        }
    
    def _create_error(self, error_type: ImageErrorType, context: ErrorContext,
                     **kwargs) -> ImageError:
        """Create an error with appropriate details and recovery actions"""
        error_template = self.error_registry.get(error_type, {})
        
        # Format message with context
        title = error_template.get("title", "Image Error")
        message = error_template.get("message", "An error occurred")
        severity = error_template.get("severity", "error")
        
        # Replace placeholders in message
        try:
            # Merge context vars and kwargs, with kwargs taking precedence
            format_vars = self._get_context_vars(context)
            format_vars.update(kwargs)
            message = message.format(**format_vars)
            title = title.format(**format_vars)
        except KeyError as e:
            logger.warning(f"Missing template variable: {e}")
        except (ValueError, TypeError) as e:
            logger.warning(f"Template formatting error: {e}")
        
        # Get recovery actions
        recovery_actions = self._get_recovery_actions(error_type, context, **kwargs)
        
        # Get prevention tips
        prevention_tips = error_template.get("prevention_tips", [])
        
        # Add technical details if exception provided
        technical_details = ""
        if "exception" in kwargs and kwargs["exception"]:
            technical_details = f"{type(kwargs['exception']).__name__}: {str(kwargs['exception'])}"
        
        return ImageError(
            error_type=error_type,
            severity=severity,
            title=title,
            message=message,
            technical_details=technical_details,
            context=context,
            recovery_actions=recovery_actions,
            prevention_tips=prevention_tips,
            related_errors=error_template.get("related_errors", [])
        )
    
    def _get_context_vars(self, context: ErrorContext) -> Dict[str, Any]:
        """Get context variables for message formatting"""
        return {
            "image_type": context.image_type,
            "model_type": context.model_type,
            "operation": context.operation,
            "user_action": context.user_action
        }
    
    def _get_recovery_actions(self, error_type: ImageErrorType, 
                            context: ErrorContext, **kwargs) -> List[RecoveryAction]:
        """Get appropriate recovery actions for error type"""
        actions = []
        
        # Get base actions for error type
        base_actions = self.recovery_registry.get(error_type, [])
        
        for action_template in base_actions:
            action = RecoveryAction(
                action_type=action_template["type"],
                title=action_template["title"],
                description=action_template["description"],
                instructions=action_template.get("instructions", []),
                tools_needed=action_template.get("tools_needed", []),
                estimated_time=action_template.get("estimated_time", "1-2 minutes"),
                difficulty=action_template.get("difficulty", "easy")
            )
            
            # Customize action based on context and kwargs
            action = self._customize_recovery_action(action, context, **kwargs)
            actions.append(action)
        
        return actions
    
    def _customize_recovery_action(self, action: RecoveryAction, 
                                 context: ErrorContext, **kwargs) -> RecoveryAction:
        """Customize recovery action based on context"""
        # Add specific dimensions, formats, etc. to instructions
        if "current_dims" in kwargs and "required_dims" in kwargs:
            current = kwargs["current_dims"]
            required = kwargs["required_dims"]
            action.instructions = [
                instr.replace("{current_dims}", f"{current[0]}×{current[1]}")
                     .replace("{required_dims}", f"{required[0]}×{required[1]}")
                for instr in action.instructions
            ]
        
        if "format_name" in kwargs:
            format_name = kwargs["format_name"]
            action.instructions = [
                instr.replace("{format_name}", format_name)
                for instr in action.instructions
            ]
        
        return action
    
    def _classify_exception(self, exception: Exception) -> ImageErrorType:
        """Classify exception into appropriate error type"""
        exception_name = type(exception).__name__
        exception_msg = str(exception).lower()
        
        if "pil" in exception_msg or "pillow" in exception_msg:
            return ImageErrorType.PIL_NOT_AVAILABLE
        elif "numpy" in exception_msg:
            return ImageErrorType.NUMPY_NOT_AVAILABLE
        elif "permission" in exception_msg or "access" in exception_msg:
            return ImageErrorType.PERMISSION_DENIED
        elif "space" in exception_msg or "disk" in exception_msg:
            return ImageErrorType.DISK_SPACE_INSUFFICIENT
        elif "memory" in exception_msg:
            return ImageErrorType.MEMORY_INSUFFICIENT
        elif "corrupt" in exception_msg or "invalid" in exception_msg:
            return ImageErrorType.CORRUPTED_FILE
        else:
            return ImageErrorType.UNKNOWN_ERROR
    
    def _generate_summary_message(self, errors: List[ImageError], 
                                warnings: List[ImageError]) -> str:
        """Generate summary message for validation results"""
        if errors and warnings:
            return f"Found {len(errors)} error(s) and {len(warnings)} warning(s) that need attention"
        elif errors:
            return f"Found {len(errors)} error(s) that must be fixed before proceeding"
        elif warnings:
            return f"Found {len(warnings)} warning(s) - image can be used but may not be optimal"
        else:
            return "All validations passed successfully"
    
    def _collect_prevention_tips(self, errors: List[ImageError]) -> List[str]:
        """Collect unique prevention tips from all errors"""
        tips = set()
        for error in errors:
            tips.update(error.prevention_tips)
        return list(tips)
    
    def _build_error_registry(self) -> Dict[ImageErrorType, Dict[str, Any]]:
        """Build registry of error templates"""
        return {
            ImageErrorType.UNSUPPORTED_FORMAT: {
                "title": "Unsupported Image Format",
                "message": "The {format_name} format is not supported. Please use PNG, JPEG, WebP, or BMP format.",
                "severity": "error",
                "prevention_tips": [
                    "Always use common image formats (PNG, JPEG, WebP, BMP)",
                    "Check file extension matches actual format",
                    "Avoid proprietary or uncommon formats"
                ]
            },
            ImageErrorType.CORRUPTED_FILE: {
                "title": "Corrupted Image File",
                "message": "The image file appears to be corrupted or incomplete.",
                "severity": "error",
                "prevention_tips": [
                    "Ensure complete file download before upload",
                    "Avoid interrupting file transfers",
                    "Use reliable image editing software"
                ]
            },
            ImageErrorType.TOO_SMALL: {
                "title": "Image Too Small",
                "message": "Image dimensions {current_dims[0]}×{current_dims[1]} are below minimum requirement of {required_dims[0]}×{required_dims[1]}.",
                "severity": "error",
                "prevention_tips": [
                    "Use high-resolution source images",
                    "Avoid excessive downscaling",
                    "Check image requirements before upload"
                ]
            },
            ImageErrorType.TOO_LARGE: {
                "title": "Image Too Large",
                "message": "Image dimensions {current_dims[0]}×{current_dims[1]} exceed maximum limit of {max_dims[0]}×{max_dims[1]}.",
                "severity": "error",
                "prevention_tips": [
                    "Resize images before upload",
                    "Use appropriate resolution for your needs",
                    "Consider file size and processing time"
                ]
            },
            ImageErrorType.FILE_TOO_LARGE: {
                "title": "File Size Too Large",
                "message": "File size {current_size:.1f}MB exceeds maximum limit of {max_size:.1f}MB.",
                "severity": "error",
                "prevention_tips": [
                    "Compress images before upload",
                    "Use appropriate quality settings",
                    "Consider using more efficient formats like WebP"
                ]
            },
            ImageErrorType.TOO_DARK: {
                "title": "Image Too Dark",
                "message": "Image appears very dark (brightness: {quality_value:.1f}/255). This may affect video generation quality.",
                "severity": "warning",
                "prevention_tips": [
                    "Adjust brightness/exposure before upload",
                    "Use well-lit source images",
                    "Avoid underexposed photos"
                ]
            },
            ImageErrorType.TOO_BRIGHT: {
                "title": "Image Too Bright",
                "message": "Image appears overexposed (brightness: {quality_value:.1f}/255). This may affect video generation quality.",
                "severity": "warning",
                "prevention_tips": [
                    "Reduce exposure/brightness before upload",
                    "Avoid overexposed photos",
                    "Use proper lighting when capturing images"
                ]
            },
            ImageErrorType.LOW_CONTRAST: {
                "title": "Low Contrast Image",
                "message": "Image has low contrast (contrast: {quality_value:.1f}). This may result in flat-looking videos.",
                "severity": "warning",
                "prevention_tips": [
                    "Increase contrast before upload",
                    "Use images with good dynamic range",
                    "Avoid washed-out or flat images"
                ]
            },
            ImageErrorType.INCOMPATIBLE_IMAGES: {
                "title": "Incompatible Start and End Images",
                "message": "Start image ({start_dims[0]}×{start_dims[1]}) and end image ({end_dims[0]}×{end_dims[1]}) have different dimensions.",
                "severity": "warning",
                "prevention_tips": [
                    "Use images with matching dimensions",
                    "Crop or resize images to match",
                    "Maintain consistent aspect ratios"
                ]
            },
            ImageErrorType.PIL_NOT_AVAILABLE: {
                "title": "Image Processing Library Missing",
                "message": "PIL/Pillow library is required for image validation but is not installed.",
                "severity": "error",
                "prevention_tips": [
                    "Ensure all required dependencies are installed",
                    "Use proper installation procedures"
                ]
            },
            ImageErrorType.MEMORY_INSUFFICIENT: {
                "title": "Insufficient Memory",
                "message": "Not enough memory available to process the image.",
                "severity": "error",
                "prevention_tips": [
                    "Close other applications to free memory",
                    "Use smaller images",
                    "Restart the application if needed"
                ]
            },
            ImageErrorType.UNKNOWN_ERROR: {
                "title": "Unknown Error",
                "message": "An unexpected error occurred during {operation}.",
                "severity": "error",
                "prevention_tips": [
                    "Try uploading a different image",
                    "Restart the application",
                    "Check system resources"
                ]
            }
        }
    
    def _build_recovery_registry(self) -> Dict[ImageErrorType, List[Dict[str, Any]]]:
        """Build registry of recovery actions"""
        return {
            ImageErrorType.UNSUPPORTED_FORMAT: [
                {
                    "type": "convert",
                    "title": "Convert Image Format",
                    "description": "Convert your image to a supported format",
                    "instructions": [
                        "Open your image in an image editor (Paint, GIMP, Photoshop, etc.)",
                        "Go to File → Export/Save As",
                        "Choose PNG or JPEG format",
                        "Save the converted image",
                        "Upload the new file"
                    ],
                    "tools_needed": ["Image editor (Paint, GIMP, Photoshop, etc.)"],
                    "estimated_time": "2-3 minutes",
                    "difficulty": "easy"
                },
                {
                    "type": "replace",
                    "title": "Use Different Image",
                    "description": "Select a different image that's already in a supported format",
                    "instructions": [
                        "Choose a different image file",
                        "Ensure it's in PNG, JPEG, WebP, or BMP format",
                        "Upload the new image"
                    ],
                    "estimated_time": "1 minute",
                    "difficulty": "easy"
                }
            ],
            ImageErrorType.TOO_SMALL: [
                {
                    "type": "resize",
                    "title": "Upscale Image",
                    "description": "Increase the image size to meet minimum requirements",
                    "instructions": [
                        "Open image in an editor that supports upscaling",
                        "Go to Image → Resize/Scale",
                        "Set width to at least {required_dims[0]} and height to at least {required_dims[1]}",
                        "Use 'Preserve aspect ratio' to avoid distortion",
                        "Choose a good resampling method (Lanczos/Bicubic)",
                        "Save and upload the resized image"
                    ],
                    "tools_needed": ["Image editor with upscaling (GIMP, Photoshop, online tools)"],
                    "estimated_time": "3-5 minutes",
                    "difficulty": "medium"
                },
                {
                    "type": "replace",
                    "title": "Use Higher Resolution Image",
                    "description": "Find or capture a higher resolution version of the image",
                    "instructions": [
                        "Look for a higher resolution version of the same image",
                        "If it's your own photo, re-export at full resolution",
                        "Consider using AI upscaling tools for better quality",
                        "Upload the higher resolution image"
                    ],
                    "tools_needed": ["Higher resolution source", "AI upscaling tools (optional)"],
                    "estimated_time": "5-10 minutes",
                    "difficulty": "easy"
                }
            ],
            ImageErrorType.TOO_LARGE: [
                {
                    "type": "resize",
                    "title": "Reduce Image Size",
                    "description": "Resize the image to fit within size limits",
                    "instructions": [
                        "Open image in an image editor",
                        "Go to Image → Resize/Scale",
                        "Set maximum width to {max_dims[0]} and height to {max_dims[1]}",
                        "Keep 'Preserve aspect ratio' enabled",
                        "Save and upload the resized image"
                    ],
                    "tools_needed": ["Image editor"],
                    "estimated_time": "2-3 minutes",
                    "difficulty": "easy"
                }
            ],
            ImageErrorType.FILE_TOO_LARGE: [
                {
                    "type": "convert",
                    "title": "Compress Image",
                    "description": "Reduce file size while maintaining acceptable quality",
                    "instructions": [
                        "Open image in an editor",
                        "Go to File → Export/Save As",
                        "Choose JPEG format for photos or PNG for graphics",
                        "Adjust quality slider to reduce file size (try 80-90% for JPEG)",
                        "Save and check file size",
                        "Repeat with lower quality if still too large"
                    ],
                    "tools_needed": ["Image editor"],
                    "estimated_time": "2-4 minutes",
                    "difficulty": "easy"
                },
                {
                    "type": "resize",
                    "title": "Reduce Resolution",
                    "description": "Lower the image resolution to reduce file size",
                    "instructions": [
                        "Open image in an editor",
                        "Note current dimensions",
                        "Go to Image → Resize",
                        "Reduce dimensions by 25-50%",
                        "Save and check file size"
                    ],
                    "tools_needed": ["Image editor"],
                    "estimated_time": "2-3 minutes",
                    "difficulty": "easy"
                }
            ],
            ImageErrorType.PIL_NOT_AVAILABLE: [
                {
                    "type": "install",
                    "title": "Install PIL/Pillow",
                    "description": "Install the required image processing library",
                    "instructions": [
                        "Open command prompt/terminal",
                        "Run: pip install Pillow",
                        "Wait for installation to complete",
                        "Restart the application",
                        "Try uploading the image again"
                    ],
                    "tools_needed": ["Command prompt/terminal", "Internet connection"],
                    "estimated_time": "2-5 minutes",
                    "difficulty": "medium"
                }
            ],
            ImageErrorType.MEMORY_INSUFFICIENT: [
                {
                    "type": "retry",
                    "title": "Free Up Memory",
                    "description": "Close other applications and try again",
                    "instructions": [
                        "Close unnecessary applications",
                        "Clear browser cache if using web interface",
                        "Restart the application",
                        "Try uploading a smaller image first",
                        "Consider restarting your computer if issues persist"
                    ],
                    "tools_needed": ["System access"],
                    "estimated_time": "2-5 minutes",
                    "difficulty": "easy"
                }
            ]
        }

# Convenience functions for common error scenarios
def create_format_error(format_name: str, context: ErrorContext) -> ImageError:
    """Create format-related error"""
    handler = ImageErrorHandler()
    return handler.handle_format_error(format_name, context)

def create_dimension_error(dimensions: Tuple[int, int], min_dims: Tuple[int, int], 
                         max_dims: Tuple[int, int], context: ErrorContext) -> ImageError:
    """Create dimension-related error"""
    handler = ImageErrorHandler()
    return handler.handle_dimension_error(dimensions, min_dims, max_dims, context)

def create_file_size_error(file_size_mb: float, max_size_mb: float, 
                         context: ErrorContext) -> ImageError:
    """Create file size error"""
    handler = ImageErrorHandler()
    return handler.handle_file_size_error(file_size_mb, max_size_mb, context)

def create_system_error(error_type: ImageErrorType, exception: Optional[Exception] = None,
                       context: Optional[ErrorContext] = None) -> ImageError:
    """Create system error"""
    handler = ImageErrorHandler()
    return handler.handle_system_error(error_type, exception, context)

def get_error_handler(config: Optional[Dict[str, Any]] = None) -> ImageErrorHandler:
    """Get configured error handler instance"""
    return ImageErrorHandler(config)