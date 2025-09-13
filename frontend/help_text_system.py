"""
Comprehensive Help Text and Guidance System for Wan2.2 UI
Provides context-sensitive help, tooltips, and responsive guidance for image upload functionality
"""

import json
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class HelpContent:
    """Structure for help content with responsive design support"""
    title: str
    content: str
    tooltip: str
    mobile_content: Optional[str] = None
    requirements: Optional[List[str]] = None
    examples: Optional[List[str]] = None


class HelpTextSystem:
    """Comprehensive help text and guidance system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self._initialize_help_content()
    
    def _initialize_help_content(self):
        """Initialize all help content for different contexts"""
        
        # Model-specific help content
        self.model_help = {
            "t2v-A14B": HelpContent(
                title="Text-to-Video Generation",
                content="""
**📝 Text-to-Video (T2V-A14B)**

**What it does:** Creates videos entirely from text descriptions
**Best for:** Original video content from imagination
**Input required:** Detailed text prompt only
**No images needed** - This mode generates videos from text alone

**Supported resolutions:** 1280x720, 1280x704, 1920x1080
**Estimated generation time:** 9-15 minutes depending on resolution
**VRAM usage:** ~14GB base model size

**💡 Pro Tips:**
- Use descriptive prompts with camera movements (pan, zoom, tilt)
- Include lighting details (golden hour, dramatic shadows)
- Specify style (cinematic, documentary, artistic)
- Mention specific subjects and actions clearly
                """,
                tooltip="T2V creates videos from text descriptions only. No images required.",
                mobile_content="""
**📝 Text-to-Video**
Creates videos from text only.
No images needed.
Best for original content.
~9-15 min generation time.
                """,
                requirements=[
                    "Detailed text prompt (max 500 characters)",
                    "~14GB VRAM available",
                    "Sufficient disk space for output"
                ],
                examples=[
                    "A majestic eagle soaring over snow-capped mountains, cinematic shot",
                    "Time-lapse of a flower blooming in spring garden, macro lens",
                    "Futuristic city at night with neon lights, cyberpunk style"
                ]
            ),
            
            "i2v-A14B": HelpContent(
                title="Image-to-Video Generation",
                content="""
**🖼️ Image-to-Video (I2V-A14B)**

**What it does:** Animates static images into videos
**Best for:** Bringing photos to life, creating motion from stills
**Input required:** Start image (required) + optional text prompt

**📸 Image Requirements:**
- **Formats:** PNG, JPG, JPEG, WebP
- **Minimum size:** 256x256 pixels
- **Recommended:** High resolution for best results
- **Aspect ratio:** Will be maintained in output

**Supported resolutions:** 1280x720, 1280x704, 1920x1080
**Estimated generation time:** 9-15 minutes depending on resolution
**VRAM usage:** ~14GB base model size

**💡 Pro Tips:**
- Upload high-quality, well-lit images
- Use text prompts to guide animation style
- Images with clear subjects work best
- Avoid heavily compressed or blurry images
                """,
                tooltip="I2V animates static images. Requires start image upload.",
                mobile_content="""
**🖼️ Image-to-Video**
Animates static images.
Requires: Start image upload
Optional: Text prompt
~9-15 min generation time.
                """,
                requirements=[
                    "Start image (PNG/JPG/JPEG/WebP)",
                    "Minimum 256x256 pixels",
                    "Optional text prompt for guidance",
                    "~14GB VRAM available"
                ],
                examples=[
                    "Portrait photo → animated with subtle breathing",
                    "Landscape → animated with moving clouds/water",
                    "Still life → animated with gentle movement"
                ]
            ),
            
            "ti2v-5B": HelpContent(
                title="Text-Image-to-Video Generation",
                content="""
**🎬 Text-Image-to-Video (TI2V-5B)**

**What it does:** Combines text and images for precise video generation
**Best for:** Controlled video creation with visual and textual guidance
**Input required:** Text prompt + start image + optional end image

**📸 Image Requirements:**
- **Start image:** Required - defines video beginning
- **End image:** Optional - defines video ending/target
- **Formats:** PNG, JPG, JPEG, WebP
- **Minimum size:** 256x256 pixels
- **Compatibility:** Start and end images should have similar aspect ratios

**Supported resolutions:** 1280x720, 1280x704, 1920x1080, 1024x1024
**Estimated generation time:** 12-20 minutes depending on resolution
**VRAM usage:** ~5GB base model size (most efficient)

**💡 Pro Tips:**
- Use start image for composition/style reference
- Use text prompt for motion and detail guidance
- End image creates smooth transitions to target state
- Works well with consistent lighting between images
                """,
                tooltip="TI2V combines text and images for precise control. Most versatile mode.",
                mobile_content="""
**🎬 Text-Image-to-Video**
Combines text + images.
Required: Text + start image
Optional: End image
Most precise control.
                """,
                requirements=[
                    "Text prompt (descriptive)",
                    "Start image (PNG/JPG/JPEG/WebP)",
                    "Optional end image",
                    "Images should have similar aspect ratios",
                    "~5GB VRAM available"
                ],
                examples=[
                    "Start: Person standing → End: Person walking + 'walking forward'",
                    "Start: Closed flower → End: Open flower + 'blooming time-lapse'",
                    "Start: Day scene → End: Night scene + 'transition to evening'"
                ]
            )
        }
        
        # Image upload help content
        self.image_help = {
            "start_image": HelpContent(
                title="Start Frame Image",
                content="""
**📸 Start Frame Image Upload**

**Purpose:** Defines the first frame of your generated video
**Required for:** I2V and TI2V generation modes
**Status:** Required when using image-based generation

**📋 Technical Requirements:**
- **File formats:** PNG, JPG, JPEG, WebP
- **Minimum dimensions:** 256x256 pixels
- **Maximum file size:** 50MB recommended
- **Aspect ratio:** Any (will be preserved in output)
- **Color space:** RGB preferred

**✅ Quality Guidelines:**
- Use high-resolution images for best results
- Ensure good lighting and contrast
- Avoid heavily compressed or artifacts
- Clear, well-defined subjects work best

**🔧 Upload Process:**
1. Click the upload area or drag & drop
2. Wait for validation (automatic)
3. Preview thumbnail will appear
4. Check validation status below
                """,
                tooltip="Start image defines the first frame. Required for I2V/TI2V modes. PNG/JPG, min 256x256px.",
                mobile_content="""
**📸 Start Image**
First frame of video.
Required for I2V/TI2V.
PNG/JPG, min 256x256px.
Drag & drop or click to upload.
                """,
                requirements=[
                    "PNG, JPG, JPEG, or WebP format",
                    "Minimum 256x256 pixels",
                    "Maximum 50MB file size",
                    "Good lighting and contrast"
                ]
            ),
            
            "end_image": HelpContent(
                title="End Frame Image",
                content="""
**🎯 End Frame Image Upload**

**Purpose:** Defines the target final frame of your generated video
**Required for:** Optional for all modes (enhances control)
**Status:** Optional but recommended for precise control

**📋 Technical Requirements:**
- **File formats:** PNG, JPG, JPEG, WebP
- **Minimum dimensions:** 256x256 pixels
- **Maximum file size:** 50MB recommended
- **Aspect ratio:** Should match start image for best results
- **Color space:** RGB preferred

**🎨 Creative Guidelines:**
- Should relate to start image (same subject/scene)
- Can show progression (closed → open, day → night)
- Maintains visual continuity
- Smooth transitions work best

**⚠️ Compatibility Notes:**
- Aspect ratio should match start image
- Similar lighting conditions recommended
- Consistent subject positioning preferred
- Dramatic changes may cause artifacts
                """,
                tooltip="End image defines the final frame. Optional but provides better control. Should match start image aspect ratio.",
                mobile_content="""
**🎯 End Image**
Final frame target.
Optional for all modes.
Should match start image.
Better control & transitions.
                """,
                requirements=[
                    "PNG, JPG, JPEG, or WebP format",
                    "Minimum 256x256 pixels",
                    "Similar aspect ratio to start image",
                    "Related to start image content"
                ]
            )
        }
        
        # Tooltip content for various UI elements
        self.tooltips = {
            "model_type": "Choose generation mode: T2V (text only), I2V (image animation), TI2V (text + image control)",
            "prompt": "Describe your video in detail. Include camera movements, lighting, style, and specific actions.",
            "resolution": "Higher resolutions look better but take longer to generate and use more VRAM.",
            "steps": "More steps = higher quality but longer generation time. 50 is balanced.",
            "duration": "Video length in seconds. Longer videos require significantly more processing time.",
            "fps": "Frames per second. Higher FPS = smoother motion but longer generation time.",
            "lora": "LoRA models add specific styles or effects. Use sparingly to avoid conflicts.",
            "image_upload_area": "Click to browse or drag & drop images. Supports PNG, JPG, JPEG, WebP formats.",
            "clear_image": "Remove the uploaded image and reset the upload area.",
            "image_preview": "Click to view full-size image. Hover for detailed information.",
            "validation_status": "Shows if your uploaded image meets technical requirements.",
            "compatibility_check": "Verifies that start and end images work well together."
        }
        
        # Error messages and recovery suggestions
        self.error_help = {
            "invalid_format": {
                "message": "Unsupported image format",
                "suggestions": [
                    "Convert image to PNG, JPG, JPEG, or WebP format",
                    "Use online converters or image editing software",
                    "Ensure file extension matches actual format"
                ]
            },
            "too_small": {
                "message": "Image dimensions too small",
                "suggestions": [
                    "Use images at least 256x256 pixels",
                    "Upscale image using AI tools or image editors",
                    "Choose a higher resolution source image"
                ]
            },
            "aspect_mismatch": {
                "message": "Start and end images have different aspect ratios",
                "suggestions": [
                    "Crop images to match aspect ratios",
                    "Use consistent image dimensions",
                    "Consider using only start image if end image doesn't match"
                ]
            },
            "file_too_large": {
                "message": "Image file size too large",
                "suggestions": [
                    "Compress image to under 50MB",
                    "Reduce image resolution if very large",
                    "Use image compression tools"
                ]
            }
        }
    
    def get_model_help_text(self, model_type: str, mobile: bool = False) -> str:
        """Get comprehensive help text for model type"""
        if model_type not in self.model_help:
            return ""
        
        help_content = self.model_help[model_type]
        
        if mobile and help_content.mobile_content:
            return help_content.mobile_content
        
        return help_content.content
    
    def get_image_help_text(self, model_type: str, mobile: bool = False) -> str:
        """Get context-sensitive image upload help text"""
        if model_type == "t2v-A14B":
            return ""  # No images needed for T2V
        
        base_help = """
**📸 Image Upload Guidelines**

Upload high-quality images for best video generation results.
        """
        
        if model_type == "i2v-A14B":
            specific_help = """
**For I2V Generation:**
- **Start image:** Required - will be animated into video
- **End image:** Optional - defines animation target
- Use clear, well-lit photos for best animation results
            """
        elif model_type == "ti2v-5B":
            specific_help = """
**For TI2V Generation:**
- **Start image:** Required - provides visual reference
- **End image:** Optional - creates smooth transitions
- Combine with detailed text prompts for precise control
            """
        else:
            specific_help = ""
        
        mobile_help = """
**📸 Image Upload**
Required for I2V/TI2V modes.
PNG/JPG formats, min 256x256px.
        """ if not mobile else base_help + specific_help
        
        return mobile_help if mobile else base_help + specific_help
    
    def get_tooltip_text(self, element: str) -> str:
        """Get tooltip text for UI element"""
        # Check general tooltips first
        if element in self.tooltips:
            return self.tooltips[element]
        
        # Check image-specific tooltips
        if element in self.image_help:
            return self.image_help[element].tooltip
        
        return ""
    
    def get_image_upload_tooltip(self, image_type: str) -> str:
        """Get specific tooltip for image upload areas"""
        if image_type not in self.image_help:
            return ""
        
        return self.image_help[image_type].tooltip
    
    def get_requirements_list(self, context: str) -> List[str]:
        """Get requirements list for specific context"""
        if context in self.model_help:
            return self.model_help[context].requirements or []
        elif context in self.image_help:
            return self.image_help[context].requirements or []
        return []
    
    def get_examples_list(self, context: str) -> List[str]:
        """Get examples list for specific context"""
        if context in self.model_help:
            return self.model_help[context].examples or []
        return []
    
    def get_error_help(self, error_type: str) -> Dict[str, Any]:
        """Get error message and recovery suggestions"""
        return self.error_help.get(error_type, {
            "message": "An error occurred",
            "suggestions": ["Please try again or contact support"]
        })
    
    def format_help_html(self, content: str, title: str = "", css_class: str = "help-content") -> str:
        """Format help content as HTML with responsive design"""
        html = f'<div class="{css_class}">'
        
        if title:
            html += f'<h4 class="help-title">{title}</h4>'
        
        # Convert markdown-style formatting to HTML
        content = content.replace("**", "<strong>").replace("**", "</strong>")
        content = content.replace("*", "<em>").replace("*", "</em>")
        content = content.replace("\n", "<br>")
        
        html += f'<div class="help-text">{content}</div>'
        html += '</div>'
        
        return html
    
    def get_responsive_help_css(self) -> str:
        """Get CSS for responsive help text display"""
        return """
        .help-content {
            background: #f8f9fa;
            border-left: 4px solid #007bff;
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
            font-size: 0.9em;
            line-height: 1.4;
        }
        
        .help-title {
            color: #007bff;
            margin: 0 0 10px 0;
            font-size: 1.1em;
            font-weight: bold;
        }
        
        .help-text {
            color: #495057;
        }
        
        .help-content strong {
            color: #212529;
            font-weight: 600;
        }
        
        .help-content em {
            color: #6c757d;
            font-style: italic;
        }
        
        .image-help {
            background: #e8f5e8;
            border-left-color: #28a745;
        }
        
        .error-help {
            background: #f8d7da;
            border-left-color: #dc3545;
        }
        
        .warning-help {
            background: #fff3cd;
            border-left-color: #ffc107;
        }
        
        /* Mobile responsive styles */
        @media (max-width: 768px) {
            .help-content {
                padding: 10px;
                font-size: 0.8em;
                margin: 5px 0;
            }
            
            .help-title {
                font-size: 1em;
                margin-bottom: 8px;
            }
            
            .help-text br {
                display: none;
            }
            
            .help-text {
                line-height: 1.3;
            }
        }
        
        /* Tooltip styles */
        .help-tooltip {
            position: relative;
            display: inline-block;
            cursor: help;
        }
        
        .help-tooltip .tooltip-text {
            visibility: hidden;
            width: 250px;
            background-color: #333;
            color: #fff;
            text-align: left;
            border-radius: 6px;
            padding: 10px;
            position: absolute;
            z-index: 1000;
            bottom: 125%;
            left: 50%;
            margin-left: -125px;
            opacity: 0;
            transition: opacity 0.3s;
            font-size: 0.8em;
            line-height: 1.3;
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        }
        
        .help-tooltip .tooltip-text::after {
            content: "";
            position: absolute;
            top: 100%;
            left: 50%;
            margin-left: -5px;
            border-width: 5px;
            border-style: solid;
            border-color: #333 transparent transparent transparent;
        }
        
        .help-tooltip:hover .tooltip-text {
            visibility: visible;
            opacity: 1;
        }
        
        /* Mobile tooltip adjustments */
        @media (max-width: 768px) {
            .help-tooltip .tooltip-text {
                width: 200px;
                margin-left: -100px;
                font-size: 0.7em;
                padding: 8px;
            }
        }
        """
    
    def create_tooltip_html(self, text: str, tooltip: str) -> str:
        """Create HTML with tooltip functionality"""
        return f'''
        <span class="help-tooltip">
            {text}
            <span class="tooltip-text">{tooltip}</span>
        </span>
        '''
    
    def get_context_sensitive_help(self, model_type: str, has_start_image: bool = False, 
                                 has_end_image: bool = False, mobile: bool = False) -> str:
        """Get context-sensitive help that adapts to current state"""
        help_parts = []
        
        # Model-specific help
        model_help = self.get_model_help_text(model_type, mobile)
        if model_help:
            help_parts.append(model_help)
        
        # Image-specific help if relevant
        if model_type in ["i2v-A14B", "ti2v-5B"]:
            image_help = self.get_image_help_text(model_type, mobile)
            if image_help:
                help_parts.append(image_help)
            
            # Status-specific guidance
            if not has_start_image:
                help_parts.append("⚠️ **Next step:** Upload a start image to begin")
            elif has_start_image and not has_end_image:
                help_parts.append("💡 **Optional:** Upload an end image for better control")
            elif has_start_image and has_end_image:
                help_parts.append("✅ **Ready:** Both images uploaded - you can generate now")
        
        return "\n\n".join(help_parts)


# Global instance
_help_system = None

def get_help_system(config: Dict[str, Any] = None) -> HelpTextSystem:
    """Get global help system instance"""
    global _help_system
    if _help_system is None:
        _help_system = HelpTextSystem(config)
    return _help_system


# Convenience functions for UI integration
def get_model_help_text(model_type: str, mobile: bool = False) -> str:
    """Get model help text"""
    return get_help_system().get_model_help_text(model_type, mobile)


def get_image_help_text(model_type: str, mobile: bool = False) -> str:
    """Get image help text"""
    return get_help_system().get_image_help_text(model_type, mobile)


def get_tooltip_text(element: str) -> str:
    """Get tooltip text"""
    return get_help_system().get_tooltip_text(element)


def get_context_sensitive_help(model_type: str, has_start_image: bool = False, 
                             has_end_image: bool = False, mobile: bool = False) -> str:
    """Get context-sensitive help"""
    return get_help_system().get_context_sensitive_help(
        model_type, has_start_image, has_end_image, mobile
    )
