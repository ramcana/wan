"""
Integration test for Enhanced Image Validation with UI
Tests the integration between the enhanced validation system and the UI components
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from PIL import Image, ImageDraw

# Import UI class
from ui import Wan22UI

class TestUIImageValidationIntegration:
    """Test integration between enhanced validation and UI"""
    
    def setup_method(self):
        """Set up test fixtures"""
        # Mock config to avoid file dependencies
        mock_config = {
            "directories": {
                "models_directory": "models",
                "loras_directory": "loras",
                "outputs_directory": "outputs"
            },
            "optimization": {
                "default_quantization": "bf16",
                "enable_offload": True,
                "vae_tile_size": 256,
                "max_vram_usage_gb": 12
            },
            "generation": {
                "default_resolution": "1280x720",
                "default_steps": 50,
                "max_prompt_length": 500
            }
        }
        
        # Create test image
        self.test_image = Image.new('RGB', (1280, 720), (100, 150, 200))
        draw = ImageDraw.Draw(self.test_image)
        draw.rectangle([10, 10, 1270, 710], outline=(255, 255, 255), width=2)
        
        # Mock UI dependencies
        with patch('ui.get_model_manager'), \
             patch('ui.VRAMOptimizer'), \
             patch('ui.get_performance_profiler'), \
             patch('ui.start_performance_monitoring'), \
             patch.object(Wan22UI, '_load_config', return_value=mock_config), \
             patch.object(Wan22UI, '_create_interface'), \
             patch.object(Wan22UI, '_start_auto_refresh'), \
             patch.object(Wan22UI, '_perform_startup_checks'):
            
            self.ui = Wan22UI()
    
    def create_test_image(self, width, height, color=(255, 255, 255)):
        """Create a test image with specified dimensions"""
        image = Image.new('RGB', (width, height), color)
        draw = ImageDraw.Draw(image)
        draw.rectangle([5, 5, width-5, height-5], outline=(0, 0, 0), width=1)
        return image
    
    @patch('ui.validate_start_image')
    def test_validate_start_image_integration(self, mock_validate):
        """Test start image validation integration"""
        from enhanced_image_validation import ValidationFeedback, ImageMetadata
        from datetime import datetime
        
        # Mock validation result
        mock_metadata = ImageMetadata(
            filename="test.jpg",
            format="JPEG",
            dimensions=(1280, 720),
            file_size_bytes=1000000,
            aspect_ratio=16/9,
            color_mode="RGB",
            has_transparency=False,
            upload_timestamp=datetime.now(),
            thumbnail_data="data:image/png;base64,test"
        )
        
        mock_feedback = ValidationFeedback(
            is_valid=True,
            severity="success",
            title="Start Image Validated Successfully",
            message="Image meets all requirements",
            metadata=mock_metadata
        )
        
        mock_validate.return_value = mock_feedback
        
        # Test the UI method
        with patch.object(self.ui, '_show_notification', return_value="notification"):
            result = self.ui._validate_start_image_upload(self.test_image, "i2v-A14B")
            
            # Verify the enhanced validation was called
            mock_validate.assert_called_once_with(self.test_image, "i2v-A14B")
            
            # Verify result structure
            assert len(result) == 3  # validation_html, notification, gr.update
            validation_html, notification, update = result
            
            # Check that HTML contains success elements
            assert "Start Image Validated Successfully" in validation_html
            assert "✅" in validation_html
            assert "1280×720" in validation_html
    
    @patch('ui.validate_end_image')
    def test_validate_end_image_integration(self, mock_validate):
        """Test end image validation integration"""
        from enhanced_image_validation import ValidationFeedback
        
        # Mock validation result with warning
        mock_feedback = ValidationFeedback(
            is_valid=True,
            severity="warning",
            title="End Image Uploaded with Warnings",
            message="Image has some quality concerns",
            details=["Image appears overexposed"],
            suggestions=["Consider reducing brightness"]
        )
        
        mock_validate.return_value = mock_feedback
        
        # Test the UI method
        with patch.object(self.ui, '_show_notification', return_value="notification"):
            result = self.ui._validate_end_image_upload(self.test_image, "ti2v-5B")
            
            # Verify the enhanced validation was called
            mock_validate.assert_called_once_with(self.test_image, "ti2v-5B")
            
            # Verify result structure
            validation_html, notification, update = result
            
            # Check that HTML contains warning elements
            assert "End Image Uploaded with Warnings" in validation_html
            assert "⚠️" in validation_html
            assert "overexposed" in validation_html
    
    @patch('ui.validate_image_pair')
    def test_validate_image_compatibility_integration(self, mock_validate):
        """Test image compatibility validation integration"""
        from enhanced_image_validation import ValidationFeedback
        
        # Mock compatibility result
        mock_feedback = ValidationFeedback(
            is_valid=True,
            severity="warning",
            title="Image Compatibility Issues",
            message="Images have some compatibility concerns",
            details=["Dimension mismatch: Start 1280×720 vs End 1920×1080"],
            suggestions=["Resize images to match dimensions"]
        )
        
        mock_validate.return_value = mock_feedback
        
        start_image = self.create_test_image(1280, 720)
        end_image = self.create_test_image(1920, 1080)
        
        # Test the UI method
        result = self.ui._validate_image_compatibility(start_image, end_image)
        
        # Verify the enhanced validation was called
        mock_validate.assert_called_once_with(start_image, end_image)
        
        # Verify result structure
        validation_html, update = result
        
        # Check that HTML contains compatibility warning
        assert "Image Compatibility Issues" in validation_html
        assert "Dimension mismatch" in validation_html
        assert "Resize images" in validation_html
    
    def test_fallback_validation_start_image(self):
        """Test fallback validation when enhanced system not available"""
        # Test with valid image
        with patch.object(self.ui, '_show_notification', return_value="notification"):
            result = self.ui._validate_start_image_upload_basic(self.test_image)
            
            validation_html, notification, update = result
            
            # Should use basic validation
            assert "uploaded successfully" in validation_html.lower()
            assert "1280x720" in validation_html
    
    def test_fallback_validation_end_image(self):
        """Test fallback validation for end image"""
        # Test with small image (should fail)
        small_image = self.create_test_image(100, 100)
        
        with patch.object(self.ui, '_show_notification', return_value="notification"):
            result = self.ui._validate_end_image_upload_basic(small_image)
            
            validation_html, notification, update = result
            
            # Should show error for small image
            assert "too small" in validation_html.lower()
            assert "100x100" in validation_html
    
    def test_fallback_compatibility_validation(self):
        """Test fallback compatibility validation"""
        start_image = self.create_test_image(1280, 720)
        end_image = self.create_test_image(1920, 1080)  # Different size
        
        result = self.ui._validate_image_compatibility_basic(start_image, end_image)
        
        validation_html, update = result
        
        # Should detect dimension mismatch
        assert "dimension mismatch" in validation_html.lower()
        assert "1280x720" in validation_html
        assert "1920x1080" in validation_html
    
    def test_none_image_handling(self):
        """Test handling of None images"""
        # Test start image
        with patch.object(self.ui, '_show_notification', return_value="notification"):
            result = self.ui._validate_start_image_upload_basic(None)
            
            validation_html, notification, update = result
            assert validation_html == ""
            assert notification == ""
        
        # Test end image
        with patch.object(self.ui, '_show_notification', return_value="notification"):
            result = self.ui._validate_end_image_upload_basic(None)
            
            validation_html, notification, update = result
            assert validation_html == ""
            assert notification == ""
        
        # Test compatibility
        result = self.ui._validate_image_compatibility(None, None)
        validation_html, update = result
        assert validation_html == ""
    
    @patch('enhanced_image_validation.PIL_AVAILABLE', False)
    def test_no_pil_handling(self):
        """Test behavior when PIL is not available"""
        with patch.object(self.ui, '_show_notification', return_value="notification"):
            result = self.ui._validate_start_image_upload(self.test_image, "i2v-A14B")
            
            # Should fall back to basic validation
            validation_html, notification, update = result
            assert "uploaded successfully" in validation_html.lower()
    
    def test_error_handling_in_validation(self):
        """Test error handling during validation"""
        # Create a mock image that will cause an error
        mock_image = Mock()
        mock_image.size = Mock(side_effect=Exception("Test error"))
        
        with patch.object(self.ui, '_show_notification', return_value="notification"):
            result = self.ui._validate_start_image_upload_basic(mock_image)
            
            validation_html, notification, update = result
            
            # Should handle error gracefully
            assert "failed to process" in validation_html.lower()
            assert "test error" in validation_html.lower()

def test_enhanced_validation_import():
    """Test that enhanced validation can be imported"""
    try:
        from enhanced_image_validation import (
            EnhancedImageValidator,
            validate_start_image,
            validate_end_image,
            validate_image_pair
        )
        assert True  # Import successful
    except ImportError:
        pytest.fail("Enhanced image validation module could not be imported")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])