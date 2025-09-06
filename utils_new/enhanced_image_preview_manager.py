"""
Enhanced Image Preview and Management System for Wan2.2
Provides thumbnail display, clear/remove buttons, image replacement, and hover tooltips
"""

import logging
import base64
import io
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, Union, Callable
from datetime import datetime
from pathlib import Path
import json

logger = logging.getLogger(__name__)

try:
    from PIL import Image, ImageOps
    import gradio as gr
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL or Gradio not available - image preview will be limited")

@dataclass
class ImagePreviewData:
    """Data structure for image preview information"""
    image: Optional[Image.Image] = None
    thumbnail_data: Optional[str] = None  # Base64 encoded thumbnail
    filename: str = ""
    format: str = ""
    dimensions: Tuple[int, int] = (0, 0)
    file_size_mb: float = 0.0
    aspect_ratio: float = 0.0
    upload_timestamp: Optional[datetime] = None
    is_valid: bool = False
    validation_message: str = ""
    
    @property
    def aspect_ratio_string(self) -> str:
        """Human-readable aspect ratio"""
        if self.aspect_ratio == 0:
            return "Unknown"
        
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
            "file_size_mb": self.file_size_mb,
            "aspect_ratio": round(self.aspect_ratio, 3),
            "aspect_ratio_string": self.aspect_ratio_string,
            "upload_timestamp": self.upload_timestamp.isoformat() if self.upload_timestamp else None,
            "is_valid": self.is_valid,
            "validation_message": self.validation_message,
            "has_thumbnail": self.thumbnail_data is not None
        }

class EnhancedImagePreviewManager:
    """Enhanced image preview and management system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Configuration parameters
        self.thumbnail_size = self.config.get("thumbnail_size", (150, 150))
        self.max_thumbnail_size = self.config.get("max_thumbnail_size", (300, 300))
        self.enable_hover_tooltips = self.config.get("enable_hover_tooltips", True)
        self.enable_image_replacement = self.config.get("enable_image_replacement", True)
        
        # Current image data
        self.start_image_data = ImagePreviewData()
        self.end_image_data = ImagePreviewData()
        
        # Callbacks for UI updates
        self.update_callbacks: List[Callable] = []
    
    def register_update_callback(self, callback: Callable):
        """Register callback for UI updates"""
        self.update_callbacks.append(callback)
    
    def _trigger_ui_update(self):
        """Trigger all registered UI update callbacks"""
        for callback in self.update_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"UI update callback failed: {e}")
    
    def process_image_upload(self, image: Optional[Image.Image], image_type: str = "start") -> Tuple[str, str, bool]:
        """
        Process image upload and generate preview data
        
        Args:
            image: PIL Image object or None
            image_type: "start" or "end"
            
        Returns:
            Tuple of (preview_html, tooltip_data, visibility)
        """
        if not PIL_AVAILABLE:
            return self._create_error_preview("PIL not available"), "", False
        
        # Get the appropriate data container
        image_data = self.start_image_data if image_type == "start" else self.end_image_data
        
        if image is None:
            # Clear the image data
            self._clear_image_data(image_type)
            return self._create_empty_preview(image_type), "", False
        
        try:
            # Extract image metadata
            self._extract_image_metadata(image, image_data)
            
            # Generate thumbnail
            image_data.thumbnail_data = self._generate_thumbnail(image)
            
            # Validate image (basic validation for preview)
            image_data.is_valid = self._basic_image_validation(image)
            image_data.validation_message = "Valid image" if image_data.is_valid else "Image validation failed"
            
            # Generate preview HTML
            preview_html = self._create_image_preview_html(image_data, image_type)
            
            # Generate tooltip data
            tooltip_data = self._create_tooltip_data(image_data)
            
            # Trigger UI update
            self._trigger_ui_update()
            
            return preview_html, tooltip_data, True
            
        except Exception as e:
            logger.error(f"Failed to process {image_type} image upload: {e}")
            error_preview = self._create_error_preview(f"Failed to process image: {str(e)}")
            return error_preview, "", True
    
    def clear_image(self, image_type: str) -> Tuple[str, str, bool]:
        """
        Clear uploaded image and reset preview
        
        Args:
            image_type: "start" or "end"
            
        Returns:
            Tuple of (preview_html, tooltip_data, visibility)
        """
        self._clear_image_data(image_type)
        empty_preview = self._create_empty_preview(image_type)
        self._trigger_ui_update()
        return empty_preview, "", False
    
    def get_image_summary(self) -> str:
        """Get HTML summary of uploaded images"""
        summary_parts = []
        
        if self.start_image_data.image is not None:
            summary_parts.append(f"📷 Start: {self.start_image_data.dimensions[0]}×{self.start_image_data.dimensions[1]} ({self.start_image_data.aspect_ratio_string})")
        
        if self.end_image_data.image is not None:
            summary_parts.append(f"🎯 End: {self.end_image_data.dimensions[0]}×{self.end_image_data.dimensions[1]} ({self.end_image_data.aspect_ratio_string})")
        
        if not summary_parts:
            return "<div style='color: #666; font-style: italic;'>No images uploaded</div>"
        
        return "<div style='background: #f8f9fa; padding: 10px; border-radius: 6px; margin: 5px 0;'>" + " | ".join(summary_parts) + "</div>"
    
    def get_compatibility_status(self) -> str:
        """Get compatibility status between start and end images"""
        if self.start_image_data.image is None or self.end_image_data.image is None:
            return ""
        
        issues = []
        
        # Check dimensions
        if self.start_image_data.dimensions != self.end_image_data.dimensions:
            issues.append("⚠️ Different dimensions")
        
        # Check aspect ratios
        if abs(self.start_image_data.aspect_ratio - self.end_image_data.aspect_ratio) > 0.05:
            issues.append("⚠️ Different aspect ratios")
        
        if issues:
            return f"<div style='color: #856404; background: #fff3cd; padding: 8px; border-radius: 4px; margin: 5px 0;'>{' | '.join(issues)}</div>"
        else:
            return "<div style='color: #155724; background: #d4edda; padding: 8px; border-radius: 4px; margin: 5px 0;'>✅ Images are compatible</div>"
    
    def _extract_image_metadata(self, image: Image.Image, image_data: ImagePreviewData):
        """Extract metadata from PIL Image"""
        image_data.image = image
        image_data.filename = getattr(image, 'filename', 'uploaded_image')
        image_data.format = getattr(image, 'format', 'PNG')
        image_data.dimensions = image.size
        image_data.aspect_ratio = image.size[0] / image.size[1] if image.size[1] > 0 else 0
        image_data.upload_timestamp = datetime.now()
        
        # Estimate file size (rough approximation)
        width, height = image.size
        estimated_size = width * height * 3  # RGB assumption
        if hasattr(image, 'mode'):
            if image.mode == 'RGBA':
                estimated_size = width * height * 4
            elif image.mode == 'L':
                estimated_size = width * height
        
        image_data.file_size_mb = estimated_size / (1024 * 1024)
    
    def _generate_thumbnail(self, image: Image.Image) -> Optional[str]:
        """Generate base64-encoded thumbnail"""
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
    
    def _basic_image_validation(self, image: Image.Image) -> bool:
        """Basic image validation for preview"""
        try:
            width, height = image.size
            return width >= 256 and height >= 256
        except Exception:
            return False
    
    def _clear_image_data(self, image_type: str):
        """Clear image data for specified type"""
        if image_type == "start":
            self.start_image_data = ImagePreviewData()
        else:
            self.end_image_data = ImagePreviewData()
    
    def _create_image_preview_html(self, image_data: ImagePreviewData, image_type: str) -> str:
        """Create HTML for image preview with thumbnail and controls"""
        if not image_data.thumbnail_data:
            return self._create_error_preview("No thumbnail available")
        
        # Status indicator
        status_color = "#28a745" if image_data.is_valid else "#dc3545"
        status_icon = "✅" if image_data.is_valid else "❌"
        
        # Image type label
        type_label = "Start Frame" if image_type == "start" else "End Frame"
        type_icon = "📷" if image_type == "start" else "🎯"
        
        html = f"""
        <div class="image-preview-container" style="
            border: 2px solid {status_color};
            border-radius: 12px;
            padding: 15px;
            margin: 10px 0;
            background: linear-gradient(135deg, {status_color}10, {status_color}05);
            position: relative;
            transition: all 0.3s ease;
        " onmouseover="this.style.transform='scale(1.02)'" onmouseout="this.style.transform='scale(1)'">
            
            <!-- Header with type and status -->
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <div style="display: flex; align-items: center;">
                    <span style="font-size: 1.2em; margin-right: 8px;">{type_icon}</span>
                    <strong style="color: {status_color}; font-size: 1.1em;">{type_label}</strong>
                </div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="color: {status_color}; font-size: 1.1em;">{status_icon}</span>
                    <button onclick="clearImage('{image_type}')" style="
                        background: #dc3545;
                        color: white;
                        border: none;
                        border-radius: 6px;
                        padding: 4px 8px;
                        cursor: pointer;
                        font-size: 0.8em;
                        transition: background 0.2s;
                    " onmouseover="this.style.background='#c82333'" onmouseout="this.style.background='#dc3545'">
                        🗑️ Clear
                    </button>
                </div>
            </div>
            
            <!-- Thumbnail and info -->
            <div style="display: flex; gap: 15px; align-items: flex-start;">
                
                <!-- Thumbnail -->
                <div style="flex-shrink: 0;">
                    <img src="{image_data.thumbnail_data}" 
                         alt="{type_label} thumbnail"
                         style="
                             border-radius: 8px;
                             box-shadow: 0 4px 8px rgba(0,0,0,0.2);
                             max-width: {self.thumbnail_size[0]}px;
                             max-height: {self.thumbnail_size[1]}px;
                             cursor: pointer;
                         "
                         onclick="showLargePreview('{image_type}', '{image_data.thumbnail_data}')"
                         title="Click to view larger preview"
                    />
                </div>
                
                <!-- Image information -->
                <div style="flex-grow: 1; min-width: 0;">
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 8px; font-size: 0.9em;">
                        <div><strong>Size:</strong> {image_data.dimensions[0]}×{image_data.dimensions[1]}</div>
                        <div><strong>Format:</strong> {image_data.format}</div>
                        <div><strong>File Size:</strong> {image_data.file_size_mb:.2f} MB</div>
                        <div><strong>Aspect:</strong> {image_data.aspect_ratio_string}</div>
                    </div>
                    
                    <!-- Upload timestamp -->
                    <div style="margin-top: 8px; font-size: 0.8em; color: #666;">
                        <strong>Uploaded:</strong> {image_data.upload_timestamp.strftime('%H:%M:%S') if image_data.upload_timestamp else 'Unknown'}
                    </div>
                    
                    <!-- Validation status -->
                    <div style="margin-top: 8px; font-size: 0.9em; color: {status_color};">
                        <strong>Status:</strong> {image_data.validation_message}
                    </div>
                </div>
            </div>
            
            <!-- Hover tooltip data (hidden) -->
            <div class="tooltip-data" style="display: none;" data-tooltip='{json.dumps(image_data.to_dict())}'></div>
        </div>
        """
        
        return html
    
    def _create_empty_preview(self, image_type: str) -> str:
        """Create HTML for empty image preview"""
        type_label = "Start Frame" if image_type == "start" else "End Frame"
        type_icon = "📷" if image_type == "start" else "🎯"
        
        return f"""
        <div class="image-preview-empty" style="
            border: 2px dashed #dee2e6;
            border-radius: 12px;
            padding: 30px;
            margin: 10px 0;
            text-align: center;
            background: #f8f9fa;
            color: #6c757d;
            transition: all 0.3s ease;
        " onmouseover="this.style.background='#e9ecef'" onmouseout="this.style.background='#f8f9fa'">
            <div style="font-size: 2em; margin-bottom: 10px;">{type_icon}</div>
            <div style="font-size: 1.1em; font-weight: bold; margin-bottom: 5px;">{type_label}</div>
            <div style="font-size: 0.9em;">Click to upload an image</div>
        </div>
        """
    
    def _create_error_preview(self, error_message: str) -> str:
        """Create HTML for error preview"""
        return f"""
        <div class="image-preview-error" style="
            border: 2px solid #dc3545;
            border-radius: 12px;
            padding: 20px;
            margin: 10px 0;
            text-align: center;
            background: linear-gradient(135deg, #dc354515, #dc354505);
            color: #721c24;
        ">
            <div style="font-size: 1.5em; margin-bottom: 10px;">❌</div>
            <div style="font-weight: bold; margin-bottom: 5px;">Error</div>
            <div style="font-size: 0.9em;">{error_message}</div>
        </div>
        """
    
    def _create_tooltip_data(self, image_data: ImagePreviewData) -> str:
        """Create tooltip data for hover functionality"""
        if not self.enable_hover_tooltips:
            return ""
        
        tooltip_info = {
            "filename": image_data.filename,
            "dimensions": f"{image_data.dimensions[0]}×{image_data.dimensions[1]}",
            "format": image_data.format,
            "file_size": f"{image_data.file_size_mb:.2f} MB",
            "aspect_ratio": image_data.aspect_ratio_string,
            "upload_time": image_data.upload_timestamp.strftime('%Y-%m-%d %H:%M:%S') if image_data.upload_timestamp else 'Unknown',
            "validation_status": image_data.validation_message
        }
        
        return json.dumps(tooltip_info)

def create_image_preview_components() -> Dict[str, Any]:
    """Create Gradio components for enhanced image preview"""
    components = {}
    
    # Start image preview
    components['start_image_preview'] = gr.HTML(
        value="",
        visible=False,
        elem_classes=["image-preview-display"]
    )
    
    # End image preview  
    components['end_image_preview'] = gr.HTML(
        value="",
        visible=False,
        elem_classes=["image-preview-display"]
    )
    
    # Image summary display
    components['image_summary'] = gr.HTML(
        value="",
        visible=False,
        elem_classes=["image-summary-display"]
    )
    
    # Compatibility status display
    components['compatibility_status'] = gr.HTML(
        value="",
        visible=False,
        elem_classes=["compatibility-status-display"]
    )
    
    # Hidden components for clear functionality
    components['clear_start_btn'] = gr.Button(
        "Clear Start Image",
        visible=False,
        elem_id="clear_start_image_btn"
    )
    
    components['clear_end_btn'] = gr.Button(
        "Clear End Image", 
        visible=False,
        elem_id="clear_end_image_btn"
    )
    
    return components

def get_preview_manager(config: Optional[Dict[str, Any]] = None) -> EnhancedImagePreviewManager:
    """Get configured image preview manager instance"""
    return EnhancedImagePreviewManager(config)

# JavaScript for enhanced interactivity
PREVIEW_JAVASCRIPT = """
<script>
// Clear image function
function clearImage(imageType) {
    // Trigger the appropriate clear button
    const clearBtn = document.getElementById('clear_' + imageType + '_image_btn');
    if (clearBtn) {
        clearBtn.click();
    }
}

// Show large preview function
function showLargePreview(imageType, thumbnailData) {
    // Create modal overlay
    const modal = document.createElement('div');
    modal.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0,0,0,0.8);
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 10000;
        cursor: pointer;
    `;
    
    // Create large image
    const img = document.createElement('img');
    img.src = thumbnailData;
    img.style.cssText = `
        max-width: 90%;
        max-height: 90%;
        border-radius: 8px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.5);
    `;
    
    modal.appendChild(img);
    document.body.appendChild(modal);
    
    // Close on click
    modal.addEventListener('click', () => {
        document.body.removeChild(modal);
    });
    
    // Close on escape key
    const escapeHandler = (e) => {
        if (e.key === 'Escape') {
            document.body.removeChild(modal);
            document.removeEventListener('keydown', escapeHandler);
        }
    };
    document.addEventListener('keydown', escapeHandler);
}

// Enhanced tooltip functionality
document.addEventListener('DOMContentLoaded', function() {
    // Add hover tooltips to image previews
    const previewContainers = document.querySelectorAll('.image-preview-container');
    
    previewContainers.forEach(container => {
        const tooltipData = container.querySelector('.tooltip-data');
        if (tooltipData) {
            const data = JSON.parse(tooltipData.dataset.tooltip);
            
            container.addEventListener('mouseenter', function(e) {
                showTooltip(e, data);
            });
            
            container.addEventListener('mouseleave', function() {
                hideTooltip();
            });
        }
    });
});

let tooltipElement = null;

function showTooltip(event, data) {
    hideTooltip(); // Remove any existing tooltip
    
    tooltipElement = document.createElement('div');
    tooltipElement.style.cssText = `
        position: absolute;
        background: #333;
        color: white;
        padding: 10px;
        border-radius: 6px;
        font-size: 0.8em;
        z-index: 1000;
        max-width: 250px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        pointer-events: none;
    `;
    
    const tooltipContent = `
        <div><strong>File:</strong> ${data.filename}</div>
        <div><strong>Size:</strong> ${data.dimensions}</div>
        <div><strong>Format:</strong> ${data.format}</div>
        <div><strong>File Size:</strong> ${data.file_size}</div>
        <div><strong>Aspect:</strong> ${data.aspect_ratio}</div>
        <div><strong>Uploaded:</strong> ${data.upload_time}</div>
        <div><strong>Status:</strong> ${data.validation_status}</div>
    `;
    
    tooltipElement.innerHTML = tooltipContent;
    document.body.appendChild(tooltipElement);
    
    // Position tooltip
    const rect = event.target.getBoundingClientRect();
    tooltipElement.style.left = (rect.right + 10) + 'px';
    tooltipElement.style.top = rect.top + 'px';
    
    // Adjust if tooltip goes off screen
    const tooltipRect = tooltipElement.getBoundingClientRect();
    if (tooltipRect.right > window.innerWidth) {
        tooltipElement.style.left = (rect.left - tooltipRect.width - 10) + 'px';
    }
    if (tooltipRect.bottom > window.innerHeight) {
        tooltipElement.style.top = (rect.bottom - tooltipRect.height) + 'px';
    }
}

function hideTooltip() {
    if (tooltipElement) {
        document.body.removeChild(tooltipElement);
        tooltipElement = null;
    }
}
</script>
"""