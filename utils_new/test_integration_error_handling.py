"""
Integration Test for Comprehensive Image Error Handling
Tests integration between error handling system and enhanced image validation
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from enhanced_image_validation import EnhancedImageValidator, ValidationFeedback, ImageMetadata
from image_error_handler import ImageErrorHandler, ImageErrorType, ErrorContext

class TestErrorHandlingIntegration:
    """Test integration between error handling and validation systems"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.validator = EnhancedImageValidator()
        self.handler = ImageErrorHandler()
    
    @patch('enhanced_image_validation.PIL_AVAILABLE', True)
    def test_format_error_integration(self):
        """Test format error integration with validation system"""
        # Create mock image with unsupported format
        mock_image = Mock()
        mock_image.size = (1280, 720)
        mock_image.format = "TIFF"
        mock_image.mode = "RGB"
        mock_image.filename = "test.tiff"
        mock_image.copy.return_value = mock_image
        
        # Mock metadata extraction
        with patch.object(self.validator, '_extract_metadata') as mock_metadata:
            mock_metadata.return_value = ImageMetadata(
                filename="test.tiff",
                format="TIFF",
                dimensions=(1280, 720),
                file_size_bytes=1000000,
                aspect_ratio=16/9,
                color_mode="RGB",
                has_transparency=False,
                upload_timestamp=datetime.now()
            )
            
            result = self.validator.validate_image_upload(mock_image, "start", "i2v-A14B")
            
            # Verify error handling
            assert result.is_valid is False
            assert result.severity == "error"
            assert "Invalid" in result.title
            assert len(result.suggestions) > 0
            assert any("convert" in suggestion.lower() for suggestion in result.suggestions)
    
    @patch('enhanced_image_validation.PIL_AVAILABLE', True)
    def test_dimension_error_integration(self):
        """Test dimension error integration with validation system"""
        mock_image = Mock()
        mock_image.size = (100, 100)  # Too small
        mock_image.format = "JPEG"
        mock_image.mode = "RGB"
        mock_image.filename = "small.jpg"
        mock_image.copy.return_value = mock_image
        
        with patch.object(self.validator, '_extract_metadata') as mock_metadata:
            mock_metadata.return_value = ImageMetadata(
                filename="small.jpg",
                format="JPEG",
                dimensions=(100, 100),
                file_size_bytes=10000,
                aspect_ratio=1.0,
                color_mode="RGB",
                has_transparency=False,
                upload_timestamp=datetime.now()
            )
            
            result = self.validator.validate_image_upload(mock_image, "start", "i2v-A14B")
            
            # Verify error handling
            assert result.is_valid is False
            assert result.severity == "error"
            assert "Invalid" in result.title
            assert len(result.suggestions) > 0
            assert any("resize" in suggestion.lower() or "upscale" in suggestion.lower() 
                      for suggestion in result.suggestions)
    
    @patch('enhanced_image_validation.PIL_AVAILABLE', True)
    def test_file_size_error_integration(self):
        """Test file size error integration with validation system"""
        mock_image = Mock()
        mock_image.size = (2000, 2000)
        mock_image.format = "JPEG"
        mock_image.mode = "RGB"
        mock_image.filename = "large.jpg"
        mock_image.copy.return_value = mock_image
        
        with patch.object(self.validator, '_extract_metadata') as mock_metadata:
            mock_metadata.return_value = ImageMetadata(
                filename="large.jpg",
                format="JPEG",
                dimensions=(2000, 2000),
                file_size_bytes=100 * 1024 * 1024,  # 100MB
                aspect_ratio=1.0,
                color_mode="RGB",
                has_transparency=False,
                upload_timestamp=datetime.now()
            )
            
            result = self.validator.validate_image_upload(mock_image, "start", "i2v-A14B")
            
            # Verify error handling
            assert result.is_valid is False
            assert result.severity == "error"
            assert "Invalid" in result.title
            assert len(result.suggestions) > 0
            assert any("compress" in suggestion.lower() or "reduce" in suggestion.lower() 
                      for suggestion in result.suggestions)
    
    @patch('enhanced_image_validation.PIL_AVAILABLE', True)
    @patch('enhanced_image_validation.NUMPY_AVAILABLE', True)
    @patch('numpy.array')
    @patch('numpy.mean')
    def test_quality_warning_integration(self, mock_mean, mock_array):
        """Test quality warning integration with validation system"""
        mock_image = Mock()
        mock_image.size = (1280, 720)
        mock_image.format = "JPEG"
        mock_image.mode = "RGB"
        mock_image.filename = "dark.jpg"
        mock_image.copy.return_value = mock_image
        
        # Mock numpy analysis for dark image
        mock_array.return_value = Mock()
        mock_mean.return_value = 10  # Very dark
        
        with patch.object(self.validator, '_extract_metadata') as mock_metadata:
            mock_metadata.return_value = ImageMetadata(
                filename="dark.jpg",
                format="JPEG",
                dimensions=(1280, 720),
                file_size_bytes=1000000,
                aspect_ratio=16/9,
                color_mode="RGB",
                has_transparency=False,
                upload_timestamp=datetime.now()
            )
            
            result = self.validator.validate_image_upload(mock_image, "start", "i2v-A14B")
            
            # Verify warning handling
            assert result.is_valid is True
            assert result.severity == "warning"
            assert "Warning" in result.title
            assert len(result.suggestions) > 0
    
    @patch('enhanced_image_validation.PIL_AVAILABLE', True)
    def test_compatibility_error_integration(self):
        """Test compatibility error integration"""
        start_image = Mock()
        start_image.size = (1280, 720)
        start_image.format = "JPEG"
        start_image.mode = "RGB"
        
        end_image = Mock()
        end_image.size = (1920, 1080)
        end_image.format = "PNG"
        end_image.mode = "RGBA"
        
        with patch.object(self.validator, '_extract_metadata') as mock_metadata:
            mock_metadata.side_effect = [
                ImageMetadata(
                    filename="start.jpg", format="JPEG", dimensions=(1280, 720),
                    file_size_bytes=1000000, aspect_ratio=16/9, color_mode="RGB",
                    has_transparency=False, upload_timestamp=datetime.now()
                ),
                ImageMetadata(
                    filename="end.png", format="PNG", dimensions=(1920, 1080),
                    file_size_bytes=2000000, aspect_ratio=16/9, color_mode="RGBA",
                    has_transparency=True, upload_timestamp=datetime.now()
                )
            ]
            
            result = self.validator.validate_image_compatibility(start_image, end_image)
            
            # Verify compatibility warning
            assert result.is_valid is True
            assert result.severity == "warning"
            assert "Compatibility" in result.title
            assert len(result.details) > 0
            assert len(result.suggestions) > 0
    
    @patch('enhanced_image_validation.PIL_AVAILABLE', False)
    def test_system_error_integration(self):
        """Test system error integration (PIL not available)"""
        mock_image = Mock()
        
        result = self.validator.validate_image_upload(mock_image, "start", "i2v-A14B")
        
        # Verify system error handling
        assert result.is_valid is False
        assert result.severity == "error"
        assert "Library" in result.title or "PIL" in result.message
        assert len(result.suggestions) > 0
        assert any("install" in suggestion.lower() for suggestion in result.suggestions)
    
    @patch('enhanced_image_validation.PIL_AVAILABLE', True)
    def test_processing_error_integration(self):
        """Test processing error integration"""
        # Create mock image that causes exception during processing
        mock_image = Mock()
        mock_image.size = (1280, 720)
        mock_image.format = "JPEG"
        mock_image.mode = "RGB"
        mock_image.filename = "error.jpg"
        
        # Mock metadata extraction to raise exception
        with patch.object(self.validator, '_extract_metadata') as mock_metadata:
            mock_metadata.side_effect = RuntimeError("Processing failed")
            
            result = self.validator.validate_image_upload(mock_image, "start", "i2v-A14B")
            
            # Verify processing error handling
            assert result.is_valid is False
            assert result.severity == "error"
            assert len(result.suggestions) > 0
    
    @patch('enhanced_image_validation.PIL_AVAILABLE', True)
    def test_multiple_errors_integration(self):
        """Test handling multiple errors in single validation"""
        mock_image = Mock()
        mock_image.size = (50, 50)  # Too small
        mock_image.format = "TIFF"  # Unsupported
        mock_image.mode = "RGB"
        mock_image.filename = "bad.tiff"
        mock_image.copy.return_value = mock_image
        
        with patch.object(self.validator, '_extract_metadata') as mock_metadata:
            mock_metadata.return_value = ImageMetadata(
                filename="bad.tiff",
                format="TIFF",
                dimensions=(50, 50),
                file_size_bytes=100 * 1024 * 1024,  # 100MB - too large
                aspect_ratio=1.0,
                color_mode="RGB",
                has_transparency=False,
                upload_timestamp=datetime.now()
            )
            
            result = self.validator.validate_image_upload(mock_image, "start", "i2v-A14B")
            
            # Verify multiple error handling
            assert result.is_valid is False
            assert result.severity == "error"
            assert "Invalid" in result.title
            assert len(result.details) > 0  # Should have multiple error details
            assert len(result.suggestions) > 0  # Should have multiple suggestions
    
    def test_error_to_feedback_conversion(self):
        """Test conversion from ImageError to ValidationFeedback"""
        context = ErrorContext(image_type="start", model_type="i2v-A14B")
        error = self.handler.handle_format_error("TIFF", context)
        
        feedback = self.validator._convert_error_to_feedback(error)
        
        assert isinstance(feedback, ValidationFeedback)
        assert feedback.is_valid is False
        assert feedback.severity == "error"
        assert feedback.title == error.title
        assert feedback.message == error.message
        assert len(feedback.suggestions) > 0
    
    def test_comprehensive_feedback_creation(self):
        """Test comprehensive feedback creation from multiple errors"""
        context = ErrorContext(image_type="start", model_type="i2v-A14B")
        
        # Create multiple errors
        format_error = self.handler.handle_format_error("TIFF", context)
        dimension_error = self.handler.handle_dimension_error(
            (100, 100), (256, 256), (4096, 4096), context
        )
        quality_error = self.handler.handle_quality_error("brightness_low", 10.0, context)
        
        errors = [format_error, dimension_error, quality_error]
        
        metadata = Mock(
            dimensions=(100, 100),
            format="TIFF",
            file_size_mb=1.0,
            aspect_ratio_string="1:1 (Square)"
        )
        
        feedback = self.validator._create_comprehensive_feedback(
            errors, metadata, "start", "i2v-A14B"
        )
        
        # Should be invalid due to blocking errors (format and dimension)
        assert feedback.is_valid is False
        assert feedback.severity == "error"
        assert "Invalid" in feedback.title
        assert len(feedback.details) >= 2  # At least format and dimension errors
        assert len(feedback.suggestions) > 0
        assert feedback.metadata == metadata
    
    def test_successful_validation_integration(self):
        """Test successful validation with comprehensive error handling"""
        mock_image = Mock()
        mock_image.size = (1280, 720)
        mock_image.format = "JPEG"
        mock_image.mode = "RGB"
        mock_image.filename = "good.jpg"
        mock_image.copy.return_value = mock_image
        
        with patch.object(self.validator, '_extract_metadata') as mock_metadata:
            with patch.object(self.validator, '_generate_thumbnail') as mock_thumbnail:
                mock_metadata.return_value = ImageMetadata(
                    filename="good.jpg",
                    format="JPEG",
                    dimensions=(1280, 720),
                    file_size_bytes=1000000,
                    aspect_ratio=16/9,
                    color_mode="RGB",
                    has_transparency=False,
                    upload_timestamp=datetime.now()
                )
                mock_thumbnail.return_value = "data:image/png;base64,test"
                
                result = self.validator.validate_image_upload(mock_image, "start", "i2v-A14B")
                
                # Verify successful validation
                assert result.is_valid is True
                assert result.severity == "success"
                assert "Successfully" in result.title
                assert len(result.details) > 0  # Should have image info
                assert result.metadata is not None
                assert result.metadata.thumbnail_data == "data:image/png;base64,test"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
