"""
Test suite for Enhanced Image Validation System
Tests comprehensive validation, feedback generation, and thumbnail creation
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Import the module to test
from enhanced_image_validation import (
    EnhancedImageValidator,
    ValidationFeedback,
    ImageMetadata,
    validate_start_image,
    validate_end_image,
    validate_image_pair,
    get_image_validator
)

class TestImageMetadata:
    """Test ImageMetadata class functionality"""
    
    def test_metadata_creation(self):
        """Test basic metadata creation"""
        metadata = ImageMetadata(
            filename="test.jpg",
            format="JPEG",
            dimensions=(1280, 720),
            file_size_bytes=1024000,
            aspect_ratio=16/9,
            color_mode="RGB",
            has_transparency=False,
            upload_timestamp=datetime.now()
        )
        
        assert metadata.filename == "test.jpg"
        assert metadata.format == "JPEG"
        assert metadata.dimensions == (1280, 720)
        assert metadata.file_size_mb == pytest.approx(0.976, rel=1e-2)
        assert metadata.aspect_ratio_string == "16:9 (Widescreen)"
    
    def test_aspect_ratio_detection(self):
        """Test aspect ratio string detection"""
        test_cases = [
            ((1920, 1080), "16:9 (Widescreen)"),
            ((1024, 1024), "1:1 (Square)"),
            ((1600, 1200), "4:3 (Standard)"),
            ((1080, 1920), "9:16 (Portrait)"),
            ((800, 600), "4:3 (Standard)"),
            ((1000, 500), "1000:500 (Custom)")
        ]
        
        for dimensions, expected in test_cases:
            metadata = ImageMetadata(
                filename="test.jpg",
                format="JPEG", 
                dimensions=dimensions,
                file_size_bytes=1000000,
                aspect_ratio=dimensions[0]/dimensions[1],
                color_mode="RGB",
                has_transparency=False,
                upload_timestamp=datetime.now()
            )
            assert metadata.aspect_ratio_string == expected
    
    def test_metadata_serialization(self):
        """Test metadata to_dict conversion"""
        metadata = ImageMetadata(
            filename="test.png",
            format="PNG",
            dimensions=(512, 512),
            file_size_bytes=500000,
            aspect_ratio=1.0,
            color_mode="RGBA",
            has_transparency=True,
            upload_timestamp=datetime(2024, 1, 1, 12, 0, 0)
        )
        
        result = metadata.to_dict()
        
        assert result["filename"] == "test.png"
        assert result["format"] == "PNG"
        assert result["dimensions"] == (512, 512)
        assert result["file_size_mb"] == 0.48
        assert result["aspect_ratio_string"] == "1:1 (Square)"
        assert result["has_transparency"] is True

class TestValidationFeedback:
    """Test ValidationFeedback class functionality"""
    
    def test_feedback_creation(self):
        """Test basic feedback creation"""
        feedback = ValidationFeedback(
            is_valid=True,
            severity="success",
            title="Test Success",
            message="Test message",
            details=["Detail 1", "Detail 2"],
            suggestions=["Suggestion 1"]
        )
        
        assert feedback.is_valid is True
        assert feedback.severity == "success"
        assert feedback.title == "Test Success"
        assert len(feedback.details) == 2
        assert len(feedback.suggestions) == 1
    
    def test_html_generation_success(self):
        """Test HTML generation for success feedback"""
        feedback = ValidationFeedback(
            is_valid=True,
            severity="success",
            title="Image Valid",
            message="Image uploaded successfully",
            details=["1280x720 resolution", "JPEG format"],
            suggestions=["Consider PNG for better quality"]
        )
        
        html = feedback.to_html()
        
        assert "✅" in html
        assert "Image Valid" in html
        assert "Image uploaded successfully" in html
        assert "1280x720 resolution" in html
        assert "Consider PNG for better quality" in html
        assert "#28a745" in html  # Success color
    
    def test_html_generation_error(self):
        """Test HTML generation for error feedback"""
        feedback = ValidationFeedback(
            is_valid=False,
            severity="error",
            title="Invalid Image",
            message="Image validation failed",
            details=["Too small: 100x100"],
            suggestions=["Use larger image"]
        )
        
        html = feedback.to_html()
        
        assert "❌" in html
        assert "Invalid Image" in html
        assert "Too small: 100x100" in html
        assert "Use larger image" in html
        assert "#dc3545" in html  # Error color
    
    def test_html_with_metadata(self):
        """Test HTML generation with metadata"""
        metadata = ImageMetadata(
            filename="test.jpg",
            format="JPEG",
            dimensions=(1920, 1080),
            file_size_bytes=2000000,
            aspect_ratio=16/9,
            color_mode="RGB",
            has_transparency=False,
            upload_timestamp=datetime.now()
        )
        
        feedback = ValidationFeedback(
            is_valid=True,
            severity="success",
            title="Success",
            message="Valid image",
            metadata=metadata
        )
        
        html = feedback.to_html()
        
        assert "1920×1080" in html
        assert "JPEG" in html
        assert "1.91 MB" in html
        assert "16:9 (Widescreen)" in html
        assert "RGB" in html

@patch('enhanced_image_validation.PIL_AVAILABLE', True)
class TestEnhancedImageValidator:
    """Test EnhancedImageValidator class functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.validator = EnhancedImageValidator()
        
        # Mock PIL Image
        self.mock_image = Mock()
        self.mock_image.size = (1280, 720)
        self.mock_image.format = "JPEG"
        self.mock_image.mode = "RGB"
        self.mock_image.filename = "test.jpg"
    
    def test_validator_initialization(self):
        """Test validator initialization with default config"""
        validator = EnhancedImageValidator()
        
        assert validator.supported_formats == ["JPEG", "PNG", "WEBP", "BMP"]
        assert validator.max_file_size_mb == 50
        assert validator.min_dimensions == (256, 256)
        assert validator.max_dimensions == (4096, 4096)
    
    def test_validator_custom_config(self):
        """Test validator initialization with custom config"""
        config = {
            "supported_formats": ["PNG", "JPEG"],
            "max_file_size_mb": 25,
            "min_dimensions": (512, 512)
        }
        
        validator = EnhancedImageValidator(config)
        
        assert validator.supported_formats == ["PNG", "JPEG"]
        assert validator.max_file_size_mb == 25
        assert validator.min_dimensions == (512, 512)
    
    def test_validate_none_image(self):
        """Test validation of None image"""
        result = self.validator.validate_image_upload(None, "start", "i2v-A14B")
        
        assert result.is_valid is True
        assert result.severity == "success"
        assert "No Image Uploaded" in result.title
        assert "required for I2V/TI2V" in result.message
    
    def test_validate_none_end_image(self):
        """Test validation of None end image"""
        result = self.validator.validate_image_upload(None, "end", "i2v-A14B")
        
        assert result.is_valid is True
        assert result.severity == "success"
        assert "optional" in result.message
    
    @patch('enhanced_image_validation.EnhancedImageValidator._extract_metadata')
    @patch('enhanced_image_validation.EnhancedImageValidator._generate_thumbnail')
    def test_validate_valid_image(self, mock_thumbnail, mock_metadata):
        """Test validation of valid image"""
        # Mock metadata
        mock_metadata.return_value = ImageMetadata(
            filename="test.jpg",
            format="JPEG",
            dimensions=(1280, 720),
            file_size_bytes=1000000,
            aspect_ratio=16/9,
            color_mode="RGB",
            has_transparency=False,
            upload_timestamp=datetime.now()
        )
        mock_thumbnail.return_value = "data:image/png;base64,test"
        
        result = self.validator.validate_image_upload(self.mock_image, "start", "i2v-A14B")
        
        assert result.is_valid is True
        assert result.severity == "success"
        assert "Validated Successfully" in result.title
        assert result.metadata is not None
        assert result.metadata.thumbnail_data == "data:image/png;base64,test"
    
    @patch('enhanced_image_validation.EnhancedImageValidator._extract_metadata')
    def test_validate_small_image(self, mock_metadata):
        """Test validation of too small image"""
        # Mock metadata for small image
        mock_metadata.return_value = ImageMetadata(
            filename="small.jpg",
            format="JPEG",
            dimensions=(100, 100),  # Too small
            file_size_bytes=10000,
            aspect_ratio=1.0,
            color_mode="RGB",
            has_transparency=False,
            upload_timestamp=datetime.now()
        )
        
        result = self.validator.validate_image_upload(self.mock_image, "start", "i2v-A14B")
        
        assert result.is_valid is False
        assert result.severity == "error"
        assert "Invalid" in result.title
        assert any("100" in detail for detail in result.details)  # Check for dimension values
        assert any("resize" in suggestion.lower() or "upscale" in suggestion.lower() for suggestion in result.suggestions)
    
    @patch('enhanced_image_validation.EnhancedImageValidator._extract_metadata')
    def test_validate_large_file(self, mock_metadata):
        """Test validation of too large file"""
        # Mock metadata for large file
        mock_metadata.return_value = ImageMetadata(
            filename="large.jpg",
            format="JPEG",
            dimensions=(2000, 2000),
            file_size_bytes=100 * 1024 * 1024,  # 100MB - too large
            aspect_ratio=1.0,
            color_mode="RGB",
            has_transparency=False,
            upload_timestamp=datetime.now()
        )
        
        result = self.validator.validate_image_upload(self.mock_image, "start", "i2v-A14B")
        
        assert result.is_valid is False
        assert result.severity == "error"
        assert any("100" in detail for detail in result.details)  # Check for file size values
        assert any("compress" in suggestion.lower() or "reduce" in suggestion.lower() for suggestion in result.suggestions)
    
    @patch('enhanced_image_validation.EnhancedImageValidator._extract_metadata')
    def test_validate_unsupported_format(self, mock_metadata):
        """Test validation of unsupported format"""
        # Mock metadata for unsupported format
        mock_metadata.return_value = ImageMetadata(
            filename="test.tiff",
            format="TIFF",  # Not in supported formats
            dimensions=(1280, 720),
            file_size_bytes=1000000,
            aspect_ratio=16/9,
            color_mode="RGB",
            has_transparency=False,
            upload_timestamp=datetime.now()
        )
        
        result = self.validator.validate_image_upload(self.mock_image, "start", "i2v-A14B")
        
        assert result.is_valid is False
        assert result.severity == "error"
        assert any("TIFF" in detail for detail in result.details)  # Check for format name
        assert any("convert" in suggestion.lower() for suggestion in result.suggestions)
    
    def test_validate_image_compatibility_matching(self):
        """Test compatibility validation with matching images"""
        # Create matching mock images
        start_image = Mock()
        start_image.size = (1280, 720)
        start_image.format = "JPEG"
        start_image.mode = "RGB"
        
        end_image = Mock()
        end_image.size = (1280, 720)
        end_image.format = "JPEG"
        end_image.mode = "RGB"
        
        with patch.object(self.validator, '_extract_metadata') as mock_metadata:
            mock_metadata.side_effect = [
                ImageMetadata(
                    filename="start.jpg", format="JPEG", dimensions=(1280, 720),
                    file_size_bytes=1000000, aspect_ratio=16/9, color_mode="RGB",
                    has_transparency=False, upload_timestamp=datetime.now()
                ),
                ImageMetadata(
                    filename="end.jpg", format="JPEG", dimensions=(1280, 720),
                    file_size_bytes=1000000, aspect_ratio=16/9, color_mode="RGB",
                    has_transparency=False, upload_timestamp=datetime.now()
                )
            ]
            
            result = self.validator.validate_image_compatibility(start_image, end_image)
            
            assert result.is_valid is True
            assert result.severity == "success"
            assert "Compatible" in result.title
    
    def test_validate_image_compatibility_mismatched(self):
        """Test compatibility validation with mismatched images"""
        start_image = Mock()
        end_image = Mock()
        
        with patch.object(self.validator, '_extract_metadata') as mock_metadata:
            mock_metadata.side_effect = [
                ImageMetadata(
                    filename="start.jpg", format="JPEG", dimensions=(1280, 720),
                    file_size_bytes=1000000, aspect_ratio=16/9, color_mode="RGB",
                    has_transparency=False, upload_timestamp=datetime.now()
                ),
                ImageMetadata(
                    filename="end.jpg", format="PNG", dimensions=(1920, 1080),
                    file_size_bytes=2000000, aspect_ratio=16/9, color_mode="RGBA",
                    has_transparency=True, upload_timestamp=datetime.now()
                )
            ]
            
            result = self.validator.validate_image_compatibility(start_image, end_image)
            
            assert result.is_valid is True
            assert result.severity == "warning"
            assert "Compatibility" in result.title
            assert any("1280" in detail and "1920" in detail for detail in result.details)  # Check for dimension values
            assert any("RGB" in detail and "RGBA" in detail for detail in result.details)  # Check for color mode values
    
    @patch('enhanced_image_validation.NUMPY_AVAILABLE', True)
    @patch('numpy.array')
    @patch('numpy.mean')
    @patch('numpy.std')
    def test_quality_analysis_with_numpy(self, mock_std, mock_mean, mock_array):
        """Test image quality analysis with numpy available"""
        mock_img_array = Mock()
        mock_img_array.shape = (100, 100, 3)  # Color image shape
        mock_array.return_value = mock_img_array
        mock_mean.return_value = 10  # Very dark
        mock_std.return_value = 5   # Low contrast
        
        issues, suggestions = self.validator._analyze_image_quality(self.mock_image)
        
        assert any("very dark" in issue.lower() for issue in issues)
        assert any("low contrast" in issue.lower() for issue in issues)
        assert any("brighten" in suggestion.lower() for suggestion in suggestions)
        assert any("contrast" in suggestion.lower() for suggestion in suggestions)
    
    def test_model_specific_validation(self):
        """Test model-specific validation requirements"""
        metadata = ImageMetadata(
            filename="test.jpg", format="JPEG", dimensions=(400, 400),
            file_size_bytes=1000000, aspect_ratio=1.0, color_mode="RGB",
            has_transparency=False, upload_timestamp=datetime.now()
        )
        
        issues, suggestions = self.validator._validate_for_model(metadata, "i2v-A14B", "start")
        
        # Should warn about size being below recommended minimum
        assert any("below recommended size" in issue.lower() for issue in issues)
        assert any("512" in suggestion for suggestion in suggestions)
    
    @patch('enhanced_image_validation.Image')
    @patch('base64.b64encode')
    def test_thumbnail_generation(self, mock_b64encode, mock_pil):
        """Test thumbnail generation"""
        # Mock PIL operations
        mock_thumbnail = Mock()
        mock_image_copy = Mock()
        mock_image_copy.thumbnail = Mock()
        mock_image_copy.save = Mock()
        self.mock_image.copy.return_value = mock_image_copy
        
        mock_b64encode.return_value = b"encoded_data"
        
        result = self.validator._generate_thumbnail(self.mock_image)
        
        assert result == "data:image/png;base64,encoded_data"
        mock_image_copy.thumbnail.assert_called_once()
        mock_image_copy.save.assert_called_once()

class TestConvenienceFunctions:
    """Test convenience functions for UI integration"""
    
    @patch('enhanced_image_validation.EnhancedImageValidator')
    def test_validate_start_image(self, mock_validator_class):
        """Test validate_start_image convenience function"""
        mock_validator = Mock()
        mock_validator_class.return_value = mock_validator
        mock_validator.validate_image_upload.return_value = ValidationFeedback(
            is_valid=True, severity="success", title="Test", message="Test"
        )
        
        mock_image = Mock()
        result = validate_start_image(mock_image, "i2v-A14B")
        
        mock_validator.validate_image_upload.assert_called_once_with(mock_image, "start", "i2v-A14B")
        assert result.is_valid is True
    
    @patch('enhanced_image_validation.EnhancedImageValidator')
    def test_validate_end_image(self, mock_validator_class):
        """Test validate_end_image convenience function"""
        mock_validator = Mock()
        mock_validator_class.return_value = mock_validator
        mock_validator.validate_image_upload.return_value = ValidationFeedback(
            is_valid=True, severity="success", title="Test", message="Test"
        )
        
        mock_image = Mock()
        result = validate_end_image(mock_image, "ti2v-5B")
        
        mock_validator.validate_image_upload.assert_called_once_with(mock_image, "end", "ti2v-5B")
        assert result.is_valid is True
    
    @patch('enhanced_image_validation.EnhancedImageValidator')
    def test_validate_image_pair(self, mock_validator_class):
        """Test validate_image_pair convenience function"""
        mock_validator = Mock()
        mock_validator_class.return_value = mock_validator
        mock_validator.validate_image_compatibility.return_value = ValidationFeedback(
            is_valid=True, severity="success", title="Test", message="Test"
        )
        
        start_image = Mock()
        end_image = Mock()
        result = validate_image_pair(start_image, end_image)
        
        mock_validator.validate_image_compatibility.assert_called_once_with(start_image, end_image)
        assert result.is_valid is True
    
    def test_get_image_validator(self):
        """Test get_image_validator convenience function"""
        config = {"max_file_size_mb": 25}
        validator = get_image_validator(config)
        
        assert isinstance(validator, EnhancedImageValidator)
        assert validator.max_file_size_mb == 25

@patch('enhanced_image_validation.PIL_AVAILABLE', False)
class TestWithoutPIL:
    """Test behavior when PIL is not available"""
    
    def test_validator_without_pil(self):
        """Test validator behavior when PIL is not available"""
        validator = EnhancedImageValidator()
        mock_image = Mock()
        
        result = validator.validate_image_upload(mock_image, "start", "i2v-A14B")
        
        assert result.is_valid is False
        assert result.severity == "error"
        assert "Library" in result.title or "PIL" in result.message
        assert "Pillow" in result.message
    
    def test_compatibility_without_pil(self):
        """Test compatibility validation without PIL"""
        validator = EnhancedImageValidator()
        
        result = validator.validate_image_compatibility(Mock(), Mock())
        
        assert result.is_valid is False
        assert result.severity == "error"
        assert "Library" in result.title or "PIL" in result.message

if __name__ == "__main__":
    pytest.main([__file__, "-v"])