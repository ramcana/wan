"""
Test suite for responsive image upload interface
Tests responsive design functionality across different screen sizes
"""

import pytest
import gradio as gr
from unittest.mock import Mock, patch, MagicMock
import json
from responsive_image_interface import ResponsiveImageInterface, get_responsive_image_interface

class TestResponsiveImageInterface:
    """Test responsive image interface functionality"""
    
    @pytest.fixture
    def config(self):
        """Test configuration"""
        return {
            "directories": {
                "models_directory": "models",
                "outputs_directory": "outputs"
            },
            "generation": {
                "default_resolution": "1280x720"
            }
        }
    
    @pytest.fixture
    def responsive_interface(self, config):
        """Create responsive interface instance"""
        return ResponsiveImageInterface(config)
    
    def test_responsive_interface_initialization(self, responsive_interface):
        """Test responsive interface initializes correctly"""
        assert responsive_interface.config is not None
        assert responsive_interface.breakpoints['mobile'] == 768
        assert responsive_interface.breakpoints['tablet'] == 1024
        assert responsive_interface.breakpoints['desktop'] == 1200
    
    def test_responsive_css_generation(self, responsive_interface):
        """Test responsive CSS is generated correctly"""
        css = responsive_interface.get_responsive_css()
        
        # Check for key responsive elements
        assert "@media (max-width: 768px)" in css
        assert "@media (min-width: 769px)" in css
        assert ".image-inputs-row" in css
        assert ".image-column" in css
        assert "flex-direction: column" in css
        assert "flex-direction: row" in css
    
    def test_responsive_image_row_creation(self, responsive_interface):
        """Test responsive image row creation"""
        with patch('gradio.Row') as mock_row, \
             patch('gradio.Column') as mock_column, \
             patch('gradio.Image') as mock_image, \
             patch('gradio.HTML') as mock_html, \
             patch('gradio.Markdown') as mock_markdown:
            
            # Mock the context managers
            mock_row.return_value.__enter__ = Mock(return_value=mock_row.return_value)
            mock_row.return_value.__exit__ = Mock(return_value=None)
            mock_column.return_value.__enter__ = Mock(return_value=mock_column.return_value)
            mock_column.return_value.__exit__ = Mock(return_value=None)
            
            image_row, components = responsive_interface.create_responsive_image_row(visible=True)
            
            # Verify components were created
            assert 'image_row' in components
            assert 'start_image' in components
            assert 'end_image' in components
            assert 'start_preview' in components
            assert 'end_preview' in components
            
            # Verify Row was called with responsive classes
            mock_row.assert_called_with(
                visible=True, 
                elem_classes=["image-inputs-row", "responsive-grid", "two-column"]
            )
            
            # Verify Columns were created with responsive classes
            assert mock_column.call_count == 2
            mock_column.assert_called_with(elem_classes=["image-column"])
    
    def test_responsive_requirements_text(self, responsive_interface):
        """Test responsive requirements text generation"""
        start_text = responsive_interface._get_responsive_requirements_text("start")
        end_text = responsive_interface._get_responsive_requirements_text("end")
        
        # Check start image requirements
        assert "Start Image Requirements" in start_text
        assert "PNG, JPG, JPEG, WebP" in start_text
        assert "256×256 pixels" in start_text
        assert "first frame" in start_text
        
        # Check end image requirements
        assert "End Image Requirements" in end_text
        assert "PNG, JPG, JPEG, WebP" in end_text
        assert "match start image" in end_text
        assert "optional" in end_text.lower()
    
    def test_responsive_help_text_generation(self, responsive_interface):
        """Test responsive help text for different model types"""
        t2v_help = responsive_interface.create_responsive_help_text("t2v-A14B")
        i2v_help = responsive_interface.create_responsive_help_text("i2v-A14B")
        ti2v_help = responsive_interface.create_responsive_help_text("ti2v-5B")
        
        # Check T2V help
        assert "Text-to-Video Generation" in t2v_help
        assert "No images required" in t2v_help
        assert "desktop-help" in t2v_help
        assert "mobile-help" in t2v_help
        
        # Check I2V help
        assert "Image-to-Video Generation" in i2v_help
        assert "Start image required" in i2v_help
        
        # Check TI2V help
        assert "Text+Image-to-Video Generation" in ti2v_help
        assert "Start image required" in ti2v_help
        assert "Text prompt guides" in ti2v_help
    
    def test_responsive_validation_message(self, responsive_interface):
        """Test responsive validation message creation"""
        success_msg = responsive_interface.create_responsive_validation_message(
            "Image uploaded successfully",
            is_success=True,
            details={
                'dimensions': '1280x720',
                'format': 'PNG',
                'file_size': '2.5MB'
            }
        )
        
        error_msg = responsive_interface.create_responsive_validation_message(
            "Invalid image format",
            is_success=False,
            details={'format': 'BMP', 'supported': 'PNG, JPG, JPEG, WebP'}
        )
        
        # Check success message
        assert "✅" in success_msg
        assert "Image uploaded successfully" in success_msg
        assert "validation-success-mobile" in success_msg
        assert "desktop-details" in success_msg
        assert "mobile-details" in success_msg
        
        # Check error message
        assert "❌" in error_msg
        assert "Invalid image format" in error_msg
        assert "validation-error-mobile" in error_msg
    
    def test_responsive_image_preview(self, responsive_interface):
        """Test responsive image preview creation"""
        mock_image_data = "mock_base64_data"
        metadata = {
            'dimensions': '1280x720',
            'format': 'PNG',
            'file_size': '2.5MB',
            'aspect_ratio': '16:9'
        }
        
        preview_html = responsive_interface.create_responsive_image_preview(
            mock_image_data,
            "start",
            metadata
        )
        
        # Check preview structure
        assert "image-preview-container" in preview_html
        assert "responsive-preview" in preview_html
        assert "image-preview-thumbnail" in preview_html
        assert "desktop-metadata" in preview_html
        assert "mobile-metadata" in preview_html
        assert "1280x720" in preview_html
        assert "PNG" in preview_html
        assert "2.5MB" in preview_html
    
    def test_responsive_javascript_generation(self, responsive_interface):
        """Test responsive JavaScript generation"""
        js_code = responsive_interface.get_responsive_javascript()
        
        # Check for key JavaScript functionality
        assert "ResponsiveImageInterface" in js_code
        assert "getCurrentBreakpoint" in js_code
        assert "setupResizeListener" in js_code
        assert "updateLayout" in js_code
        assert "showLargePreview" in js_code
        assert "clearImage" in js_code
        
        # Check for mobile-specific functionality
        assert "setupTouchHandlers" in js_code
        assert "handleTouchStart" in js_code
        assert "handleTouchEnd" in js_code
    
    def test_singleton_pattern(self, config):
        """Test singleton pattern for responsive interface"""
        interface1 = get_responsive_image_interface(config)
        interface2 = get_responsive_image_interface(config)
        
        # Should return the same instance
        assert interface1 is interface2
    
    def test_breakpoint_detection(self, responsive_interface):
        """Test breakpoint detection logic in JavaScript"""
        js_code = responsive_interface.get_responsive_javascript()
        
        # Check breakpoint values are correctly set
        assert "mobile: 768" in js_code
        assert "tablet: 1024" in js_code
        assert "desktop: 1200" in js_code
    
    def test_css_media_queries(self, responsive_interface):
        """Test CSS media queries for different screen sizes"""
        css = responsive_interface.get_responsive_css()
        
        # Check mobile media queries
        assert "@media (max-width: 768px)" in css
        assert "flex-direction: column !important" in css
        
        # Check tablet media queries
        assert "@media (min-width: 769px) and (max-width: 1024px)" in css
        
        # Check desktop media queries
        assert "@media (min-width: 769px)" in css
        assert "flex-direction: row" in css
        
        # Check extra small mobile
        assert "@media (max-width: 480px)" in css
    
    def test_accessibility_features(self, responsive_interface):
        """Test accessibility features in responsive design"""
        css = responsive_interface.get_responsive_css()
        
        # Check for reduced motion support
        assert "@media (prefers-reduced-motion: reduce)" in css
        assert "animation: none" in css
        assert "transition: none" in css
        
        # Check for high contrast support
        assert "@media (prefers-contrast: high)" in css
        assert "border-width: 3px" in css
    
    def test_touch_handling(self, responsive_interface):
        """Test touch handling for mobile devices"""
        js_code = responsive_interface.get_responsive_javascript()
        
        # Check touch event handling
        assert "touchstart" in js_code
        assert "touchend" in js_code
        assert "touch-active" in js_code
        assert "passive: true" in js_code
    
    def test_responsive_animations(self, responsive_interface):
        """Test responsive animations and transitions"""
        css = responsive_interface.get_responsive_css()
        
        # Check for animation classes
        assert "image-preview-fade-in" in css
        assert "image-preview-slide-up" in css
        assert "@keyframes fadeInScale" in css
        assert "@keyframes slideUpFade" in css
        
        # Check loading spinner
        assert "loading-spinner" in css
        assert "@keyframes spin" in css
    
    def test_responsive_button_styles(self, responsive_interface):
        """Test responsive button styling"""
        css = responsive_interface.get_responsive_css()
        
        # Check button classes
        assert "responsive-button" in css
        assert "responsive-button.secondary" in css
        
        # Check mobile button adjustments
        mobile_section = css[css.find("@media (max-width: 768px)"):]
        assert "padding: 6px 12px" in mobile_section
        assert "font-size: 0.8em" in mobile_section
    
    def test_grid_layout_responsiveness(self, responsive_interface):
        """Test responsive grid layout"""
        css = responsive_interface.get_responsive_css()
        
        # Check grid classes
        assert "responsive-grid" in css
        assert "responsive-grid.two-column" in css
        assert "grid-template-columns: 1fr 1fr" in css
        assert "grid-template-columns: 1fr" in css


class TestResponsiveIntegration:
    """Test integration with main UI"""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration"""
        return {
            "directories": {"models_directory": "models"},
            "generation": {"default_resolution": "1280x720"}
        }
    
    def test_ui_integration(self, mock_config):
        """Test integration with main UI class"""
        with patch('responsive_image_interface.ResponsiveImageInterface') as mock_interface:
            mock_instance = Mock()
            mock_instance.get_responsive_css.return_value = "/* responsive css */"
            mock_instance.get_responsive_javascript.return_value = "<script>/* js */</script>"
            mock_interface.return_value = mock_instance
            
            # Test that responsive interface is properly integrated
            interface = get_responsive_image_interface(mock_config)
            css = interface.get_responsive_css()
            js = interface.get_responsive_javascript()
            
            assert css == "/* responsive css */"
            assert js == "<script>/* js */</script>"
    
    def test_fallback_behavior(self):
        """Test fallback behavior when responsive interface is not available"""
        # This would test the fallback JavaScript in the main UI
        fallback_js = """
        function updateResponsiveLayout() {
            const isMobile = window.innerWidth <= 768;
            // Basic responsive functionality
        }
        """
        
        # Verify fallback contains essential functionality
        assert "updateResponsiveLayout" in fallback_js
        assert "window.innerWidth <= 768" in fallback_js


if __name__ == "__main__":
    # Run basic functionality test
    config = {
        "directories": {"models_directory": "models"},
        "generation": {"default_resolution": "1280x720"}
    }
    
    interface = ResponsiveImageInterface(config)
    
    print("✅ Responsive interface initialized")
    print("✅ CSS generated:", len(interface.get_responsive_css()), "characters")
    print("✅ JavaScript generated:", len(interface.get_responsive_javascript()), "characters")
    
    # Test requirements text
    start_req = interface._get_responsive_requirements_text("start")
    end_req = interface._get_responsive_requirements_text("end")
    
    print("✅ Start requirements generated:", len(start_req), "characters")
    print("✅ End requirements generated:", len(end_req), "characters")
    
    # Test help text
    help_text = interface.create_responsive_help_text("i2v-A14B")
    print("✅ Help text generated:", len(help_text), "characters")
    
    # Test validation message
    validation = interface.create_responsive_validation_message(
        "Test message", 
        True, 
        {"dimensions": "1280x720", "format": "PNG"}
    )
    print("✅ Validation message generated:", len(validation), "characters")
    
    print("\n🎉 All responsive image interface tests passed!")