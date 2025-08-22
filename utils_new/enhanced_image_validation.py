"""
Enhanced Image Upload Validation and Feedback System for Wan2.2
Provides comprehensive validation, thumbnail generation, and detailed feedback for image uploads
"""

import logging
import base64
import io
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, Union
from datetime import datetime
from pathlib import Path
import json

# Import comprehensive error handling
from image_error_handler import (
    ImageErrorHandler, ImageError, ImageErrorType, ErrorContext,
    create_format_error, create_dimension_error, create_file_size_error,
    create_system_error
)

logger = logging.getLogger(__name__)

try:
    from PIL import Image, ImageOps, ImageStat
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL not available - image validation will be limited")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.warning("NumPy not available - advanced image analysis disabled")

@dataclass
class ImageMetadata:
    """Comprehensive image metadata"""
    filename: str
    format: str
    dimensions: Tuple[int, int]
    file_size_bytes: int
    aspect_ratio: float
    color_mode: str
    has_transparency: bool
    upload_timestamp: datetime
    thumbnail_data: Optional[str] = None  # Base64 encoded thumbnail
    
    @property
    def file_size_mb(self) -> float:
        """File size in megabytes"""
        return self.file_size_bytes / (1024 * 1024)
    
    @property
    def aspect_ratio_string(self) -> str:
        """Human-readable aspect ratio"""
        width, height = self.dimensions
        # Common aspect ratios
        ratio_map = {
            16/9: "16:9 (Widescreen)",
            4/3: "4:3 (Standard)",
            1/1: "1:1 (Square)",
            3/2: "3:2 (Photo)",
            21/9: "21:9 (Ultrawide)",
            9/16: "9:16 (Portrait)"
        }
        
        for ratio, name in ratio_map.items():
            if abs(self.aspect_ratio - ratio) < 0.05:
                return name
        
        return f"{width}:{height} (Custom)"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "filename": self.filename,
            "format": self.format,
            "dimensions": self.dimensions,
            "file_size_bytes": self.file_size_bytes,
            "file_size_mb": round(self.file_size_mb, 2),
            "aspect_ratio": round(self.aspect_ratio, 3),
            "aspect_ratio_string": self.aspect_ratio_string,
            "color_mode": self.color_mode,
            "has_transparency": self.has_transparency,
            "upload_timestamp": self.upload_timestamp.isoformat(),
            "thumbnail_data": self.thumbnail_data
        }

@dataclass
class ValidationFeedback:
    """Detailed validation feedback with suggestions"""
    is_valid: bool
    severity: str  # "success", "warning", "error"
    title: str
    message: str
    details: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    metadata: Optional[ImageMetadata] = None
    
    def to_html(self) -> str:
        """Generate HTML representation of feedback"""
        severity_colors = {
            "success": "#28a745",
            "warning": "#ffc107", 
            "error": "#dc3545"
        }
        
        severity_icons = {
            "success": "✅",
            "warning": "⚠️",
            "error": "❌"
        }
        
        color = severity_colors.get(self.severity, "#6c757d")
        icon = severity_icons.get(self.severity, "ℹ️")
        
        html = f"""
        <div style="
            border: 2px solid {color};
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
            background: linear-gradient(135deg, {color}15, {color}05);
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        ">
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <span style="font-size: 1.2em; margin-right: 8px;">{icon}</span>
                <strong style="color: {color}; font-size: 1.1em;">{self.title}</strong>
            </div>
            
            <div style="margin-bottom: 10px; color: #333;">
                {self.message}
            </div>
        """
        
        # Add details if available
        if self.details:
            html += """
            <div style="margin: 10px 0;">
                <strong style="color: #555;">Details:</strong>
                <ul style="margin: 5px 0; padding-left: 20px; color: #666;">
            """
            for detail in self.details:
                html += f"<li>{detail}</li>"
            html += "</ul></div>"
        
        # Add suggestions if available
        if self.suggestions:
            html += """
            <div style="margin: 10px 0;">
                <strong style="color: #007bff;">💡 Suggestions:</strong>
                <ul style="margin: 5px 0; padding-left: 20px; color: #0056b3;">
            """
            for suggestion in self.suggestions:
                html += f"<li>{suggestion}</li>"
            html += "</ul></div>"
        
        # Add metadata display if available
        if self.metadata:
            html += f"""
            <div style="
                margin-top: 15px;
                padding: 10px;
                background: rgba(0,0,0,0.05);
                border-radius: 4px;
                font-size: 0.9em;
            ">
                <strong>📊 Image Information:</strong><br>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 8px; margin-top: 8px;">
                    <div><strong>Size:</strong> {self.metadata.dimensions[0]}×{self.metadata.dimensions[1]}</div>
                    <div><strong>Format:</strong> {self.metadata.format}</div>
                    <div><strong>File Size:</strong> {self.metadata.file_size_mb:.2f} MB</div>
                    <div><strong>Aspect Ratio:</strong> {self.metadata.aspect_ratio_string}</div>
                    <div><strong>Color Mode:</strong> {self.metadata.color_mode}</div>
                    <div><strong>Transparency:</strong> {"Yes" if self.metadata.has_transparency else "No"}</div>
                </div>
            </div>
            """
        
        html += "</div>"
        return html

class EnhancedImageValidator:
    """Enhanced image validator with comprehensive feedback and thumbnail generation"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Configuration parameters
        self.supported_formats = self.config.get("supported_formats", ["JPEG", "PNG", "WEBP", "BMP"])
        self.max_file_size_mb = self.config.get("max_file_size_mb", 50)
        self.min_dimensions = self.config.get("min_dimensions", (256, 256))
        self.max_dimensions = self.config.get("max_dimensions", (4096, 4096))
        self.thumbnail_size = self.config.get("thumbnail_size", (150, 150))
        
        # Quality thresholds
        self.quality_thresholds = {
            "min_brightness": 20,
            "max_brightness": 235,
            "min_contrast": 15,
            "max_blur_threshold": 100  # Lower values indicate more blur
        }
        
        # Model-specific requirements
        self.model_requirements = {
            "t2v-A14B": {
                "preferred_aspect_ratios": [16/9, 4/3],
                "recommended_min_size": (512, 512),
                "notes": "Text-to-Video generation - images not required but can be used for style reference"
            },
            "i2v-A14B": {
                "preferred_aspect_ratios": [16/9, 4/3, 1/1],
                "recommended_min_size": (512, 512),
                "notes": "Image-to-Video generation - start image defines the first frame"
            },
            "ti2v-5B": {
                "preferred_aspect_ratios": [16/9, 4/3],
                "recommended_min_size": (512, 512),
                "notes": "Text+Image-to-Video generation - image should complement text prompt"
            }
        }
        
        # Initialize comprehensive error handler
        self.error_handler = ImageErrorHandler(config)
    
    def validate_image_upload(self, image: Any, image_type: str = "start", 
                            model_type: str = "i2v-A14B") -> ValidationFeedback:
        """
        Comprehensive image validation with detailed feedback and error handling
        
        Args:
            image: PIL Image object or None
            image_type: "start" or "end" 
            model_type: Target model type for generation
            
        Returns:
            ValidationFeedback object with detailed results
        """
        # Create error context
        context = ErrorContext(
            image_type=image_type,
            model_type=model_type,
            operation="validation",
            user_action="upload"
        )
        
        if not PIL_AVAILABLE:
            error = self.error_handler.handle_system_error(
                ImageErrorType.PIL_NOT_AVAILABLE, 
                context=context
            )
            return self._convert_error_to_feedback(error)
        
        if image is None:
            return ValidationFeedback(
                is_valid=True,
                severity="success",
                title="No Image Uploaded",
                message=f"No {image_type} image provided" + (" (optional)" if image_type == "end" else " (required for I2V/TI2V)"),
                details=["Upload an image to enable image-based generation modes"]
            )
        
        try:
            # Extract metadata
            metadata = self._extract_metadata(image)
            context.file_path = metadata.filename
            context.file_size = metadata.file_size_bytes
            context.dimensions = metadata.dimensions
            context.format = metadata.format
            
            # Collect all validation errors
            validation_errors = []
            
            # Format validation with comprehensive error handling
            try:
                format_valid, format_error = self._validate_format_comprehensive(image, metadata, context)
                if format_error:
                    validation_errors.append(format_error)
            except Exception as e:
                error = self.error_handler.handle_processing_error("format_validation", e, context)
                validation_errors.append(error)
            
            # Dimension validation with comprehensive error handling
            try:
                dim_valid, dim_error = self._validate_dimensions_comprehensive(metadata, model_type, context)
                if dim_error:
                    validation_errors.append(dim_error)
            except Exception as e:
                error = self.error_handler.handle_processing_error("dimension_validation", e, context)
                validation_errors.append(error)
            
            # File size validation with comprehensive error handling
            try:
                size_valid, size_error = self._validate_file_size_comprehensive(metadata, context)
                if size_error:
                    validation_errors.append(size_error)
            except Exception as e:
                error = self.error_handler.handle_processing_error("size_validation", e, context)
                validation_errors.append(size_error)
            
            # Quality analysis with error handling
            try:
                quality_errors = self._analyze_image_quality_comprehensive(image, context)
                validation_errors.extend(quality_errors)
            except Exception as e:
                error = self.error_handler.handle_processing_error("quality_analysis", e, context)
                validation_errors.append(error)
            
            # Model-specific validation with error handling
            try:
                model_errors = self._validate_for_model_comprehensive(metadata, model_type, image_type, context)
                validation_errors.extend(model_errors)
            except Exception as e:
                error = self.error_handler.handle_processing_error("model_validation", e, context)
                validation_errors.append(error)
            
            # Generate thumbnail with error handling
            try:
                metadata.thumbnail_data = self._generate_thumbnail(image)
            except Exception as e:
                logger.warning(f"Thumbnail generation failed: {e}")
                # Don't fail validation for thumbnail issues
                metadata.thumbnail_data = None
            
            # Convert errors to validation feedback
            return self._create_comprehensive_feedback(validation_errors, metadata, image_type, model_type)
        
        except Exception as e:
            logger.error(f"Image validation failed: {str(e)}")
            error = self.error_handler.handle_processing_error("validation", e, context)
            return self._convert_error_to_feedback(error)
    
    def validate_image_compatibility(self, start_image: Any, end_image: Any) -> ValidationFeedback:
        """
        Validate compatibility between start and end images with comprehensive error handling
        
        Args:
            start_image: PIL Image object for start frame
            end_image: PIL Image object for end frame
            
        Returns:
            ValidationFeedback object with compatibility results
        """
        context = ErrorContext(
            image_type="compatibility",
            operation="compatibility_validation",
            user_action="validate"
        )
        
        if not PIL_AVAILABLE:
            error = self.error_handler.handle_system_error(
                ImageErrorType.PIL_NOT_AVAILABLE,
                context=context
            )
            return self._convert_error_to_feedback(error)
        
        if start_image is None or end_image is None:
            return ValidationFeedback(
                is_valid=True,
                severity="success",
                title="Compatibility Check Skipped",
                message="Both start and end images required for compatibility validation"
            )
        
        try:
            start_meta = self._extract_metadata(start_image)
            end_meta = self._extract_metadata(end_image)
            
            compatibility_errors = []
            
            # Check dimensions with comprehensive error handling
            if start_meta.dimensions != end_meta.dimensions:
                error = self.error_handler.handle_compatibility_error(
                    start_meta.dimensions,
                    end_meta.dimensions,
                    context
                )
                compatibility_errors.append(error)
            
            # Check aspect ratios
            if abs(start_meta.aspect_ratio - end_meta.aspect_ratio) > 0.05:
                error = ImageError(
                    error_type=ImageErrorType.ASPECT_RATIO_MISMATCH,
                    severity="warning",
                    title="Aspect Ratio Mismatch",
                    message=f"Start image aspect ratio {start_meta.aspect_ratio:.3f} differs from end image {end_meta.aspect_ratio:.3f}",
                    context=context,
                    recovery_actions=[],
                    prevention_tips=["Use images with matching aspect ratios", "Crop images to same aspect ratio"]
                )
                compatibility_errors.append(error)
            
            # Check color modes
            if start_meta.color_mode != end_meta.color_mode:
                error = ImageError(
                    error_type=ImageErrorType.COLOR_MODE_MISMATCH,
                    severity="warning",
                    title="Color Mode Mismatch",
                    message=f"Start image color mode {start_meta.color_mode} differs from end image {end_meta.color_mode}",
                    context=context,
                    recovery_actions=[],
                    prevention_tips=["Convert images to the same color mode (preferably RGB)"]
                )
                compatibility_errors.append(error)
            
            # Create comprehensive feedback
            if not compatibility_errors:
                return ValidationFeedback(
                    is_valid=True,
                    severity="success",
                    title="Images Are Compatible",
                    message="Start and end images are well-matched for video generation",
                    details=[
                        f"Both images: {start_meta.dimensions[0]}×{start_meta.dimensions[1]}",
                        f"Matching aspect ratio: {start_meta.aspect_ratio_string}",
                        f"Same color mode: {start_meta.color_mode}"
                    ]
                )
            else:
                # Collect all suggestions
                all_suggestions = []
                for error in compatibility_errors:
                    all_suggestions.extend(error.prevention_tips)
                
                return ValidationFeedback(
                    is_valid=True,
                    severity="warning",
                    title="Image Compatibility Issues",
                    message=f"Start and end images have {len(compatibility_errors)} compatibility concern(s)",
                    details=[error.message for error in compatibility_errors],
                    suggestions=list(set(all_suggestions))  # Remove duplicates
                )
        
        except Exception as e:
            logger.error(f"Image compatibility validation failed: {str(e)}")
            error = self.error_handler.handle_processing_error("compatibility_validation", e, context)
            return self._convert_error_to_feedback(error)
    
    def _extract_metadata(self, image: Image.Image) -> ImageMetadata:
        """Extract comprehensive metadata from PIL Image"""
        # Get basic properties
        width, height = image.size
        format_name = getattr(image, 'format', None)
        
        # If no format is set (e.g., created in memory), assume PNG for validation
        if format_name is None:
            format_name = 'PNG'
        
        # Estimate file size (rough approximation)
        # For more accurate size, we'd need the original file
        estimated_size = width * height * 3  # RGB assumption
        if hasattr(image, 'mode'):
            if image.mode == 'RGBA':
                estimated_size = width * height * 4
            elif image.mode == 'L':
                estimated_size = width * height
        
        return ImageMetadata(
            filename=getattr(image, 'filename', 'uploaded_image'),
            format=format_name,
            dimensions=(width, height),
            file_size_bytes=estimated_size,
            aspect_ratio=width / height,
            color_mode=getattr(image, 'mode', 'RGB'),
            has_transparency=getattr(image, 'mode', '') == 'RGBA',
            upload_timestamp=datetime.now()
        )
    
    def _validate_format_comprehensive(self, image: Any, metadata: ImageMetadata, 
                                     context: ErrorContext) -> Tuple[bool, Optional[ImageError]]:
        """Comprehensive format validation with detailed error handling"""
        if metadata.format not in self.supported_formats:
            error = self.error_handler.handle_format_error(metadata.format, context)
            return False, error
        
        # Additional format-specific checks
        try:
            # Try to verify the image can be processed
            if hasattr(image, 'verify'):
                image_copy = image.copy()  # Don't modify original
                image_copy.verify()
        except Exception as e:
            # Image is corrupted or has invalid structure
            error = self.error_handler.handle_system_error(
                ImageErrorType.CORRUPTED_FILE, 
                exception=e, 
                context=context
            )
            return False, error
        
        return True, None
    
    def _validate_format(self, image: Image.Image, metadata: ImageMetadata) -> Tuple[bool, str, List[str]]:
        """Validate image format (legacy method for backward compatibility)"""
        if metadata.format not in self.supported_formats:
            return False, f"Unsupported format: {metadata.format}", [
                f"Convert to one of: {', '.join(self.supported_formats)}",
                "Most image editors can export to PNG or JPEG"
            ]
        return True, "", []
    
    def _validate_dimensions_comprehensive(self, metadata: ImageMetadata, model_type: str,
                                         context: ErrorContext) -> Tuple[bool, Optional[ImageError]]:
        """Comprehensive dimension validation with detailed error handling"""
        width, height = metadata.dimensions
        min_w, min_h = self.min_dimensions
        max_w, max_h = self.max_dimensions
        
        # Check for invalid dimensions
        if width <= 0 or height <= 0:
            error = self.error_handler.handle_system_error(
                ImageErrorType.INVALID_DIMENSIONS,
                context=context
            )
            return False, error
        
        # Check minimum dimensions
        if width < min_w or height < min_h:
            error = self.error_handler.handle_dimension_error(
                metadata.dimensions, 
                self.min_dimensions, 
                self.max_dimensions,
                context
            )
            return False, error
        
        # Check maximum dimensions
        if width > max_w or height > max_h:
            error = self.error_handler.handle_dimension_error(
                metadata.dimensions,
                self.min_dimensions,
                self.max_dimensions, 
                context
            )
            return False, error
        
        return True, None
    
    def _validate_dimensions(self, metadata: ImageMetadata, model_type: str) -> Tuple[bool, str, List[str]]:
        """Validate image dimensions (legacy method for backward compatibility)"""
        width, height = metadata.dimensions
        min_w, min_h = self.min_dimensions
        max_w, max_h = self.max_dimensions
        
        if width < min_w or height < min_h:
            return False, f"Image too small: {width}×{height} (minimum: {min_w}×{min_h})", [
                f"Resize image to at least {min_w}×{min_h}",
                "Use image editing software to upscale the image"
            ]
        
        if width > max_w or height > max_h:
            return False, f"Image too large: {width}×{height} (maximum: {max_w}×{max_h})", [
                f"Resize image to under {max_w}×{max_h}",
                "Large images may cause memory issues during generation"
            ]
        
        return True, "", []
    
    def _validate_file_size_comprehensive(self, metadata: ImageMetadata, 
                                        context: ErrorContext) -> Tuple[bool, Optional[ImageError]]:
        """Comprehensive file size validation with detailed error handling"""
        if metadata.file_size_mb > self.max_file_size_mb:
            error = self.error_handler.handle_file_size_error(
                metadata.file_size_mb,
                self.max_file_size_mb,
                context
            )
            return False, error
        
        return True, None
    
    def _validate_file_size(self, metadata: ImageMetadata) -> Tuple[bool, str, List[str]]:
        """Validate file size (legacy method for backward compatibility)"""
        if metadata.file_size_mb > self.max_file_size_mb:
            return False, f"File too large: {metadata.file_size_mb:.2f} MB (maximum: {self.max_file_size_mb} MB)", [
                "Compress the image using image editing software",
                "Reduce image quality/resolution to decrease file size"
            ]
        return True, "", []
    
    def _analyze_image_quality_comprehensive(self, image: Any, 
                                           context: ErrorContext) -> List[ImageError]:
        """Comprehensive image quality analysis with detailed error handling"""
        quality_errors = []
        
        try:
            if NUMPY_AVAILABLE:
                # Convert to numpy for analysis
                img_array = np.array(image)
                
                # Brightness analysis
                mean_brightness = np.mean(img_array)
                if mean_brightness < self.quality_thresholds["min_brightness"]:
                    error = self.error_handler.handle_quality_error(
                        "brightness_low", mean_brightness, context
                    )
                    quality_errors.append(error)
                elif mean_brightness > self.quality_thresholds["max_brightness"]:
                    error = self.error_handler.handle_quality_error(
                        "brightness_high", mean_brightness, context
                    )
                    quality_errors.append(error)
                
                # Contrast analysis
                if hasattr(img_array, 'shape') and len(img_array.shape) >= 3:  # Color image
                    contrast = np.std(img_array)
                    if contrast < self.quality_thresholds["min_contrast"]:
                        error = self.error_handler.handle_quality_error(
                            "contrast_low", contrast, context
                        )
                        quality_errors.append(error)
            
            # PIL-based analysis for color mode issues
            if hasattr(image, 'mode'):
                if image.mode == 'L':
                    # Create a warning for grayscale images
                    error = ImageError(
                        error_type=ImageErrorType.LOW_CONTRAST,
                        severity="warning",
                        title="Grayscale Image Detected",
                        message="Image is in grayscale mode. Color images typically produce better video results.",
                        context=context,
                        recovery_actions=[],
                        prevention_tips=["Use color images when possible", "Convert grayscale to RGB if needed"]
                    )
                    quality_errors.append(error)
                elif image.mode == 'P':
                    # Create a warning for palette mode
                    error = ImageError(
                        error_type=ImageErrorType.COLOR_MODE_MISMATCH,
                        severity="warning", 
                        title="Palette Mode Image",
                        message="Image is in palette mode and will be converted to RGB during processing.",
                        context=context,
                        recovery_actions=[],
                        prevention_tips=["Convert to RGB mode before upload for better control"]
                    )
                    quality_errors.append(error)
        
        except Exception as e:
            logger.debug(f"Quality analysis failed: {e}")
            # Don't fail validation for quality analysis issues
        
        return quality_errors
    
    def _analyze_image_quality(self, image: Image.Image) -> Tuple[List[str], List[str]]:
        """Analyze image quality and provide feedback (legacy method for backward compatibility)"""
        issues = []
        suggestions = []
        
        try:
            if NUMPY_AVAILABLE:
                # Convert to numpy for analysis
                img_array = np.array(image)
                
                # Brightness analysis
                mean_brightness = np.mean(img_array)
                if mean_brightness < self.quality_thresholds["min_brightness"]:
                    issues.append(f"Image appears very dark (brightness: {mean_brightness:.1f}/255)")
                    suggestions.append("Consider brightening the image for better visibility")
                elif mean_brightness > self.quality_thresholds["max_brightness"]:
                    issues.append(f"Image appears overexposed (brightness: {mean_brightness:.1f}/255)")
                    suggestions.append("Consider reducing exposure or brightness")
                
                # Contrast analysis
                if hasattr(img_array, 'shape') and len(img_array.shape) >= 3:  # Color image
                    contrast = np.std(img_array)
                    if contrast < self.quality_thresholds["min_contrast"]:
                        issues.append(f"Low contrast detected (contrast: {contrast:.1f})")
                        suggestions.append("Increase contrast for more dynamic video generation")
            
            # PIL-based analysis
            if hasattr(image, 'mode'):
                if image.mode == 'L':
                    issues.append("Grayscale image detected")
                    suggestions.append("Color images typically produce better video results")
                elif image.mode == 'P':
                    issues.append("Palette mode image - will be converted to RGB")
                    suggestions.append("Convert to RGB mode before upload for better control")
        
        except Exception as e:
            logger.debug(f"Quality analysis failed: {e}")
        
        return issues, suggestions
    
    def _validate_for_model_comprehensive(self, metadata: ImageMetadata, model_type: str,
                                        image_type: str, context: ErrorContext) -> List[ImageError]:
        """Comprehensive model-specific validation with detailed error handling"""
        model_errors = []
        
        if model_type not in self.model_requirements:
            return model_errors
        
        requirements = self.model_requirements[model_type]
        
        # Check recommended minimum size
        rec_min = requirements.get("recommended_min_size", (512, 512))
        if metadata.dimensions[0] < rec_min[0] or metadata.dimensions[1] < rec_min[1]:
            error = ImageError(
                error_type=ImageErrorType.TOO_SMALL,
                severity="warning",
                title=f"Below Recommended Size for {model_type}",
                message=f"Image size {metadata.dimensions[0]}×{metadata.dimensions[1]} is below recommended minimum of {rec_min[0]}×{rec_min[1]} for optimal results with {model_type}.",
                context=context,
                recovery_actions=[],
                prevention_tips=[f"Use images at least {rec_min[0]}×{rec_min[1]} for best results"]
            )
            model_errors.append(error)
        
        # Check preferred aspect ratios
        preferred_ratios = requirements.get("preferred_aspect_ratios", [])
        if preferred_ratios:
            ratio_match = any(abs(metadata.aspect_ratio - ratio) < 0.1 for ratio in preferred_ratios)
            if not ratio_match:
                ratio_names = []
                for ratio in preferred_ratios:
                    if ratio == 16/9:
                        ratio_names.append("16:9")
                    elif ratio == 4/3:
                        ratio_names.append("4:3")
                    elif ratio == 1/1:
                        ratio_names.append("1:1")
                    else:
                        ratio_names.append(f"{ratio:.2f}")
                
                error = ImageError(
                    error_type=ImageErrorType.ASPECT_RATIO_MISMATCH,
                    severity="warning",
                    title=f"Non-Optimal Aspect Ratio for {model_type}",
                    message=f"Aspect ratio {metadata.aspect_ratio:.2f} is not optimal for {model_type}. Preferred ratios: {', '.join(ratio_names)}",
                    context=context,
                    recovery_actions=[],
                    prevention_tips=[f"Consider using aspect ratios: {', '.join(ratio_names)}"]
                )
                model_errors.append(error)
        
        return model_errors
    
    def _validate_for_model(self, metadata: ImageMetadata, model_type: str, 
                          image_type: str) -> Tuple[List[str], List[str]]:
        """Model-specific validation (legacy method for backward compatibility)"""
        issues = []
        suggestions = []
        
        if model_type not in self.model_requirements:
            return issues, suggestions
        
        requirements = self.model_requirements[model_type]
        
        # Check recommended minimum size
        rec_min = requirements.get("recommended_min_size", (512, 512))
        if metadata.dimensions[0] < rec_min[0] or metadata.dimensions[1] < rec_min[1]:
            issues.append(f"Below recommended size for {model_type}: {metadata.dimensions[0]}×{metadata.dimensions[1]} < {rec_min[0]}×{rec_min[1]}")
            suggestions.append(f"Use images at least {rec_min[0]}×{rec_min[1]} for best results with {model_type}")
        
        # Check preferred aspect ratios
        preferred_ratios = requirements.get("preferred_aspect_ratios", [])
        if preferred_ratios:
            ratio_match = any(abs(metadata.aspect_ratio - ratio) < 0.1 for ratio in preferred_ratios)
            if not ratio_match:
                ratio_names = []
                for ratio in preferred_ratios:
                    if ratio == 16/9:
                        ratio_names.append("16:9")
                    elif ratio == 4/3:
                        ratio_names.append("4:3")
                    elif ratio == 1/1:
                        ratio_names.append("1:1")
                    else:
                        ratio_names.append(f"{ratio:.2f}")
                
                issues.append(f"Aspect ratio {metadata.aspect_ratio:.2f} not optimal for {model_type}")
                suggestions.append(f"Consider using aspect ratios: {', '.join(ratio_names)}")
        
        return issues, suggestions
    
    def _get_optimization_suggestions(self, metadata: ImageMetadata, model_type: str) -> List[str]:
        """Get optimization suggestions for successful uploads"""
        suggestions = []
        
        # General optimization tips
        if metadata.file_size_mb > 10:
            suggestions.append("Large file size - consider compressing for faster upload")
        
        if metadata.dimensions[0] > 1920 or metadata.dimensions[1] > 1920:
            suggestions.append("High resolution detected - may increase generation time")
        
        # Model-specific tips
        if model_type in self.model_requirements:
            notes = self.model_requirements[model_type].get("notes", "")
            if notes:
                suggestions.append(f"💡 {notes}")
        
        return suggestions
    
    def _convert_error_to_feedback(self, error: ImageError) -> ValidationFeedback:
        """Convert ImageError to ValidationFeedback for backward compatibility"""
        # Extract recovery suggestions
        suggestions = []
        for action in error.recovery_actions:
            suggestions.append(f"{action.title}: {action.description}")
            suggestions.extend(action.instructions[:2])  # Limit to first 2 instructions
        
        suggestions.extend(error.prevention_tips)
        
        return ValidationFeedback(
            is_valid=error.severity != "error",
            severity=error.severity,
            title=error.title,
            message=error.message,
            details=[error.technical_details] if error.technical_details else [],
            suggestions=suggestions
        )
    
    def _create_comprehensive_feedback(self, validation_errors: List[ImageError], 
                                     metadata: ImageMetadata, image_type: str, 
                                     model_type: str) -> ValidationFeedback:
        """Create comprehensive validation feedback from errors"""
        if not validation_errors:
            # No errors - success case
            return ValidationFeedback(
                is_valid=True,
                severity="success",
                title=f"{image_type.title()} Image Validated Successfully",
                message=f"Image meets all requirements for {model_type} generation",
                details=[
                    f"Resolution: {metadata.dimensions[0]}×{metadata.dimensions[1]}",
                    f"Format: {metadata.format}",
                    f"File size: {metadata.file_size_mb:.2f} MB",
                    f"Aspect ratio: {metadata.aspect_ratio_string}"
                ],
                suggestions=self._get_optimization_suggestions(metadata, model_type),
                metadata=metadata
            )
        
        # Separate errors by severity
        blocking_errors = [e for e in validation_errors if e.severity == "error"]
        warnings = [e for e in validation_errors if e.severity == "warning"]
        
        # Collect all details and suggestions
        all_details = []
        all_suggestions = []
        
        for error in validation_errors:
            if error.technical_details:
                all_details.append(error.technical_details)
            
            # Add recovery action suggestions
            for action in error.recovery_actions:
                all_suggestions.append(f"{action.title}: {action.description}")
            
            all_suggestions.extend(error.prevention_tips)
        
        if blocking_errors:
            # Has blocking errors
            return ValidationFeedback(
                is_valid=False,
                severity="error",
                title=f"Invalid {image_type.title()} Image",
                message=f"Found {len(blocking_errors)} error(s) that must be fixed before proceeding",
                details=[error.message for error in blocking_errors],
                suggestions=list(set(all_suggestions)),  # Remove duplicates
                metadata=metadata
            )
        else:
            # Only warnings
            return ValidationFeedback(
                is_valid=True,
                severity="warning",
                title=f"{image_type.title()} Image Uploaded with Warnings",
                message=f"Image uploaded successfully but has {len(warnings)} quality concern(s)",
                details=[warning.message for warning in warnings],
                suggestions=list(set(all_suggestions)),  # Remove duplicates
                metadata=metadata
            )
    
    def _generate_thumbnail(self, image: Image.Image) -> Optional[str]:
        """Generate base64-encoded thumbnail for preview"""
        try:
            # Create thumbnail
            thumbnail = image.copy()
            thumbnail.thumbnail(self.thumbnail_size, Image.Resampling.LANCZOS)
            
            # Convert to base64
            buffer = io.BytesIO()
            thumbnail.save(buffer, format='PNG')
            thumbnail_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return f"data:image/png;base64,{thumbnail_data}"
        
        except Exception as e:
            logger.error(f"Thumbnail generation failed: {e}")
            return None

# Convenience functions for UI integration
def validate_start_image(image: Any, model_type: str = "i2v-A14B") -> ValidationFeedback:
    """Validate start image upload"""
    validator = EnhancedImageValidator()
    return validator.validate_image_upload(image, "start", model_type)

def validate_end_image(image: Any, model_type: str = "i2v-A14B") -> ValidationFeedback:
    """Validate end image upload"""
    validator = EnhancedImageValidator()
    return validator.validate_image_upload(image, "end", model_type)

def validate_image_pair(start_image: Any, end_image: Any) -> ValidationFeedback:
    """Validate compatibility between start and end images"""
    validator = EnhancedImageValidator()
    return validator.validate_image_compatibility(start_image, end_image)

def get_image_validator(config: Optional[Dict[str, Any]] = None) -> EnhancedImageValidator:
    """Get configured image validator instance"""
    return EnhancedImageValidator(config)