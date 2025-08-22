"""
Resolution Management System for Wan2.2 UI
Handles model-specific resolution options and validation
"""

import logging
from typing import Dict, List, Optional, Tuple
import gradio as gr

logger = logging.getLogger(__name__)

class ResolutionManager:
    """Manages resolution options and validation for different model types"""
    
    # Resolution mapping as specified in requirements 10.1, 10.2, 10.3
    RESOLUTION_MAP = {
        't2v-A14B': ['854x480', '480x854', '1280x720', '1280x704', '1920x1080'],
        'i2v-A14B': ['854x480', '480x854', '1280x720', '1280x704', '1920x1080'],
        'ti2v-5B': ['854x480', '480x854', '1280x720', '1280x704', '1920x1080', '1024x1024']
    }
    
    # Default resolution for each model type
    DEFAULT_RESOLUTION = {
        't2v-A14B': '1280x720',
        'i2v-A14B': '1280x720', 
        'ti2v-5B': '1280x720'
    }
    
    # Resolution descriptions for user guidance
    RESOLUTION_INFO = {
        't2v-A14B': "T2V-A14B supports 480p (landscape/portrait) to 1080p resolution (2-12 min generation time)",
        'i2v-A14B': "I2V-A14B supports 480p (landscape/portrait) to 1080p resolution (2-12 min generation time)",
        'ti2v-5B': "TI2V-5B supports 480p (landscape/portrait) to 1080p plus square format (2-17 min generation time)"
    }
    
    def __init__(self):
        """Initialize the resolution manager"""
        logger.info("Resolution manager initialized")
    
    def get_resolution_options(self, model_type: str) -> List[str]:
        """
        Get supported resolution options for a specific model type
        
        Args:
            model_type: The model type (t2v-A14B, i2v-A14B, ti2v-5B)
            
        Returns:
            List of supported resolution strings
        """
        if model_type not in self.RESOLUTION_MAP:
            logger.warning(f"Unknown model type: {model_type}, using default options")
            return self.RESOLUTION_MAP['t2v-A14B']
        
        return self.RESOLUTION_MAP[model_type]
    
    def get_default_resolution(self, model_type: str) -> str:
        """
        Get the default resolution for a specific model type
        
        Args:
            model_type: The model type
            
        Returns:
            Default resolution string
        """
        return self.DEFAULT_RESOLUTION.get(model_type, '1280x720')
    
    def get_resolution_info(self, model_type: str) -> str:
        """
        Get descriptive information about resolution options for a model type
        
        Args:
            model_type: The model type
            
        Returns:
            Information string about resolution options
        """
        return self.RESOLUTION_INFO.get(model_type, "Resolution options available")
    
    def update_resolution_dropdown(self, model_type: str, current_resolution: Optional[str] = None) -> gr.update:
        """
        Update resolution dropdown with model-specific options
        
        Args:
            model_type: The selected model type
            current_resolution: Currently selected resolution (if any)
            
        Returns:
            Gradio update object for the dropdown
        """
        try:
            # Get supported resolutions for this model
            resolution_choices = self.get_resolution_options(model_type)
            resolution_info = self.get_resolution_info(model_type)
            
            # Determine the value to select
            if current_resolution and current_resolution in resolution_choices:
                # Keep current selection if it's supported
                selected_value = current_resolution
            else:
                # Use default resolution for this model type
                selected_value = self.get_default_resolution(model_type)
                if current_resolution and current_resolution not in resolution_choices:
                    logger.info(f"Resolution {current_resolution} not supported by {model_type}, switching to {selected_value}")
            
            return gr.update(
                choices=resolution_choices,
                value=selected_value,
                info=resolution_info
            )
            
        except Exception as e:
            logger.error(f"Error updating resolution dropdown: {e}")
            # Return safe fallback
            return gr.update(
                choices=['1280x720'],
                value='1280x720',
                info="Error loading resolution options"
            )
    
    def validate_resolution_compatibility(self, resolution: str, model_type: str) -> Tuple[bool, str]:
        """
        Validate if a resolution is compatible with a model type
        
        Args:
            resolution: Resolution string to validate
            model_type: Model type to check against
            
        Returns:
            Tuple of (is_valid, message)
        """
        try:
            supported_resolutions = self.get_resolution_options(model_type)
            
            if resolution in supported_resolutions:
                return True, f"✅ {resolution} is supported by {model_type}"
            else:
                closest = self.find_closest_supported_resolution(resolution, model_type)
                return False, f"❌ {resolution} not supported by {model_type}. Closest supported: {closest}"
                
        except Exception as e:
            logger.error(f"Error validating resolution compatibility: {e}")
            return False, "Error validating resolution compatibility"
    
    def find_closest_supported_resolution(self, resolution: str, model_type: str) -> str:
        """
        Find the closest supported resolution for a model type
        
        Args:
            resolution: Target resolution
            model_type: Model type
            
        Returns:
            Closest supported resolution string
        """
        try:
            supported_resolutions = self.get_resolution_options(model_type)
            
            if not resolution or 'x' not in resolution:
                return self.get_default_resolution(model_type)
            
            # Parse target resolution
            try:
                target_width, target_height = map(int, resolution.split('x'))
                target_pixels = target_width * target_height
            except ValueError:
                return self.get_default_resolution(model_type)
            
            # Find closest by pixel count
            closest_resolution = supported_resolutions[0]
            closest_diff = float('inf')
            
            for supported_res in supported_resolutions:
                try:
                    width, height = map(int, supported_res.split('x'))
                    pixels = width * height
                    diff = abs(pixels - target_pixels)
                    
                    if diff < closest_diff:
                        closest_diff = diff
                        closest_resolution = supported_res
                except ValueError:
                    continue
            
            return closest_resolution
            
        except Exception as e:
            logger.error(f"Error finding closest resolution: {e}")
            return self.get_default_resolution(model_type)
    
    def get_resolution_dimensions(self, resolution: str) -> Tuple[int, int]:
        """
        Parse resolution string to get width and height
        
        Args:
            resolution: Resolution string (e.g., "1280x720")
            
        Returns:
            Tuple of (width, height)
        """
        try:
            if 'x' not in resolution:
                raise ValueError(f"Invalid resolution format: {resolution}")
            
            width, height = map(int, resolution.split('x'))
            return width, height
            
        except Exception as e:
            logger.error(f"Error parsing resolution {resolution}: {e}")
            return 1280, 720  # Safe fallback
    
    def format_resolution_display(self, resolution: str) -> str:
        """
        Format resolution for display with additional information
        
        Args:
            resolution: Resolution string
            
        Returns:
            Formatted display string
        """
        try:
            width, height = self.get_resolution_dimensions(resolution)
            aspect_ratio = width / height
            
            # Determine quality label
            pixels = width * height
            if pixels >= 2073600:  # 1920x1080
                quality = "Full HD"
            elif pixels >= 921600:  # 1280x720
                quality = "HD"
            elif pixels >= 409920:  # 854x480 or 480x854
                if width > height:
                    quality = "480p Landscape"
                else:
                    quality = "480p Portrait"
            else:
                quality = "SD"
            
            return f"{resolution} ({quality}, {aspect_ratio:.2f}:1)"
            
        except Exception as e:
            logger.error(f"Error formatting resolution display: {e}")
            return resolution


# Global instance
_resolution_manager = None

def get_resolution_manager() -> ResolutionManager:
    """Get the global resolution manager instance"""
    global _resolution_manager
    if _resolution_manager is None:
        _resolution_manager = ResolutionManager()
    return _resolution_manager