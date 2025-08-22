"""
Unit Tests for Image Validation Functions
Focused testing of individual validation functions and components
"""

import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from datetime import datetime
from typing import Dict, Any

# Test imports
try:
    from PIL import Image, ImageDraw
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Import modules under test
from enhanced_image_validation import (
    EnhancedImageValidator, ValidationFeedback, ImageMetadata
)


class TestImageMetadata(unittest.TestCase):
    """Test ImageMetadata class functionality"""
    
    def test_image_metadata_creation(self):
        """Test ImageMetadata object creation"""
        metadata = ImageMetadata(
            filename="test.png",
            format="PNG",
            dimensions=(800, 600),
            file_size_bytes=1024000,
            aspect_ratio=800/600,
            color_mode="RGB",
            has_transparency=False,
            upload_timestamp=datetime.now()
        )
        
        self.assertEqual(metadata.filename, "test.png")
        self.assertEqual(metadata.format, "PNG")
        self.assertEqual(metadata.dimensions, (800, 600))
        self.assertEqual(metadata.file_size_bytes, 1024000)
        self.assertAlmostEqual(metadata.aspect_ratio, 800/600, places=3)
        self.assertEqual(metadata.color_mode, "RGB")
        self.assertFalse(metadata.has_transparency)
    
    def test_file_size_mb_property(self):
        """Test file size MB calculation"""
        metadata = ImageMetadata(
            filename="test.png",
            format="PNG",
            dimensions=(800, 600),
            file_size_bytes=2097152,  # 2 MB
            aspect_ratio=800/600,
            color_mode="RGB",
            has_transparency=False,
            upload_timestamp=datetime.now()
        )
        
        self.assertAlmostEqual(metadata.file_size_mb, 2.0, places=1)
    
    def test_aspect_ratio_string_common_ratios(self):
        """Test aspect ratio string for common ratios"""
        test_cases = [
            ((1920, 1080), "16:9 (Widescreen)"),
            ((1024, 768), "4:3 (Standard)"),
            ((512, 512), "1:1 (Square)"),
            ((1080, 1920), "9:16 (Portrait)"),
            ((800, 533), "3:2 (Photo)"),
            ((2560, 1080), "21:9 (Ultrawide)")
        ]
        
        for dimensions, expected_string in test_cases:
            with self.subTest(dimensions=dimensions):
                metadata = ImageMetadata(
                    filename="test.png",
                    format="PNG",
                    dimensions=dimensions,
                    file_size_bytes=1000000,
                    aspect_ratio=dimensions[0]/dimensions[1],
                    color_mode="RGB",
                    has_transparency=False,
                    upload_timestamp=datetime.now()
                )
                
                self.assertEqual(metadata.aspect_ratio_string, expected_string)
    
    def test_aspect_ratio_string_custom_ratio(self):
        """Test aspect ratio string for custom ratios"""
        metadata = ImageMetadata(
            filename="test.png",
            format="PNG",
            dimensions=(1337, 420),
            file_size_bytes=1000000,
            aspect_ratio=1337/420,
            color_mode="RGB",
            has_transparency=False,
            upload_timestamp=datetime.now()
        )
        
        self.assertIn("Custom", metadata.aspect_ratio_string)
        self.assertIn("1337:420", metadata.aspect_ratio_string)
    
    def test_to_dict_serialization(self):
        """Test metadata serialization to dictionary"""
        metadata = ImageMetadata(
            filename="test.png",
            format="PNG",
            dimensions=(800, 600),
            file_size_bytes=1024000,
            aspect_ratio=800/600,
            color_mode="RGB",
            has_transparency=False,
            upload_timestamp=datetime.now(),
            thumbnail_data="base64_data_here"
        )
        
        metadata_dict = metadata.to_dict()
        
        self.assertIsInstance(metadata_dict, dict)
        self.assertEqual(metadata_dict['filename'], "test.png")
        self.assertEqual(metadata_dict['format'], "PNG")
        self.assertEqual(metadata_dict['dimensions'], (800, 600))
        self.assertIn('file_size_mb', metadata_dict)
        self.assertIn('aspect_ratio_string', metadata_dict)
        self.assertEqual(metadata_dict['thumbnail_data'], "base64_data_here")


class TestValidationFeedback(unittest.TestCase):
    """Test ValidationFeedback class functionality"""
    
    def test_validation_feedback_creation(self):
        """Test ValidationFeedback object creation"""
        feedback = ValidationFeedback(
            is_valid=True,
            severity="success",
            title="Test Title",
            message="Test message",
            details=["Detail 1", "Detail 2"],
            suggestions=["Suggestion 1", "Suggestion 2"]
        )
        
        self.assertTrue(feedback.is_valid)
        self.assertEqual(feedback.severity, "success")
        self.assertEqual(feedback.title, "Test Title")
        self.assertEqual(feedback.message, "Test message")
        self.assertEqual(len(feedback.details), 2)
        self.assertEqual(len(feedback.suggestions), 2)
    
    def test_validation_feedback_html_success(self):
        """Test HTML generation for success feedback"""
        feedback = ValidationFeedback(
            is_valid=True,
            severity="success",
            title="Success Title",
            message="Success message"
        )
        
        html = feedback.to_html()
        
        self.assertIsInstance(html, str)
        self.assertIn("Success Title", html)
        self.assertIn("Success message", html)
        self.assertIn("✅", html)  # Success icon
        self.assertIn("#28a745", html)  # Success color
    
    def test_validation_feedback_html_error(self):
        """Test HTML generation for error feedback"""
        feedback = ValidationFeedback(
            is_valid=False,
            severity="error",
            title="Error Title",
            message="Error message",
            details=["Error detail"],
            suggestions=["Fix suggestion"]
        )
        
        html = feedback.to_html()
        
        self.assertIsInstance(html, str)
        self.assertIn("Error Title", html)
        self.assertIn("Error message", html)
        self.assertIn("Error detail", html)
        self.assertIn("Fix suggestion", html)
        self.assertIn("❌", html)  # Error icon
        self.assertIn("#dc3545", html)  # Error color
    
    def test_validation_feedback_html_warning(self):
        """Test HTML generation for warning feedback"""
        feedback = ValidationFeedback(
            is_valid=True,
            severity="warning",
            title="Warning Title",
            message="Warning message"
        )
        
        html = feedback.to_html()
        
        self.assertIsInstance(html, str)
        self.assertIn("Warning Title", html)
        self.assertIn("Warning message", html)
        self.assertIn("⚠️", html)  # Warning icon
        self.assertIn("#ffc107", html)  # Warning color
    
    def test_validation_feedback_html_with_metadata(self):
        """Test HTML generation with metadata"""
        metadata = ImageMetadata(
            filename="test.png",
            format="PNG",
            dimensions=(800, 600),
            file_size_bytes=1024000,
            aspect_ratio=800/600,
            color_mode="RGB",
            has_transparency=False,
            upload_timestamp=datetime.now()
        )
        
        feedback = ValidationFeedback(
            is_valid=True,
            severity="success",
            title="Success with Metadata",
            message="Success message",
            metadata=metadata
        )
        
        html = feedback.to_html()
        
        self.assertIn("800×600", html)
        self.assertIn("PNG", html)
        self.assertIn("RGB", html)
        self.assertIn("📊 Image Information", html)


class TestEnhancedImageValidatorConfiguration(unittest.TestCase):
    """Test EnhancedImageValidator configuration and initialization"""
    
    def test_default_configuration(self):
        """Test validator with default configuration"""
        validator = EnhancedImageValidator()
        
        self.assertEqual(validator.supported_formats, ["JPEG", "PNG", "WEBP", "BMP"])
        self.assertEqual(validator.max_file_size_mb, 50)
        self.assertEqual(validator.min_dimensions, (256, 256))
        self.assertEqual(validator.max_dimensions, (4096, 4096))
        self.assertEqual(validator.thumbnail_size, (150, 150))
    
    def test_custom_configuration(self):
        """Test validator with custom configuration"""
        config = {
            "supported_formats": ["PNG", "JPEG"],
            "max_file_size_mb": 25,
            "min_dimensions": (512, 512),
            "max_dimensions": (2048, 2048),
            "thumbnail_size": (200, 200)
        }
        
        validator = EnhancedImageValidator(config)
        
        self.assertEqual(validator.supported_formats, ["PNG", "JPEG"])
        self.assertEqual(validator.max_file_size_mb, 25)
        self.assertEqual(validator.min_dimensions, (512, 512))
        self.assertEqual(validator.max_dimensions, (2048, 2048))
        self.assertEqual(validator.thumbnail_size, (200, 200))
    
    def test_model_requirements_configuration(self):
        """Test model-specific requirements configuration"""
        validator = EnhancedImageValidator()
        
        # Check that model requirements are properly configured
        self.assertIn("t2v-A14B", validator.model_requirements)
        self.assertIn("i2v-A14B", validator.model_requirements)
        self.assertIn("ti2v-5B", validator.model_requirements)
        
        # Check specific requirements
        i2v_req = validator.model_requirements["i2v-A14B"]
        self.assertIn("preferred_aspect_ratios", i2v_req)
        self.assertIn("recommended_min_size", i2v_req)
        self.assertIn("notes", i2v_req)


class TestImageValidationMethods(unittest.TestCase):
    """Test individual validation methods"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.validator = EnhancedImageValidator()
    
    @pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not available")
    def test_extract_metadata_rgb_image(self):
        """Test metadata extraction from RGB image"""
        image = Image.new('RGB', (800, 600), color='red')
        image.format = 'PNG'
        
        metadata = self.validator._extract_metadata(image)
        
        self.assertEqual(metadata.dimensions, (800, 600))
        self.assertEqual(metadata.format, 'PNG')
        self.assertEqual(metadata.color_mode, 'RGB')
        self.assertAlmostEqual(metadata.aspect_ratio, 800/600, places=3)
        self.assertFalse(metadata.has_transparency)
    
    @pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not available")
    def test_extract_metadata_rgba_image(self):
        """Test metadata extraction from RGBA image"""
        image = Image.new('RGBA', (512, 512), color=(255, 0, 0, 128))
        image.format = 'PNG'
        
        metadata = self.validator._extract_metadata(image)
        
        self.assertEqual(metadata.dimensions, (512, 512))
        self.assertEqual(metadata.format, 'PNG')
        self.assertEqual(metadata.color_mode, 'RGBA')
        self.assertEqual(metadata.aspect_ratio, 1.0)
        self.assertTrue(metadata.has_transparency)
    
    @pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not available")
    def test_extract_metadata_grayscale_image(self):
        """Test metadata extraction from grayscale image"""
        image = Image.new('L', (640, 480), color=128)
        image.format = 'JPEG'
        
        metadata = self.validator._extract_metadata(image)
        
        self.assertEqual(metadata.dimensions, (640, 480))
        self.assertEqual(metadata.format, 'JPEG')
        self.assertEqual(metadata.color_mode, 'L')
        self.assertAlmostEqual(metadata.aspect_ratio, 640/480, places=3)
        self.assertFalse(metadata.has_transparency)
    
    @pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not available")
    def test_extract_metadata_no_format(self):
        """Test metadata extraction from image without format"""
        image = Image.new('RGB', (400, 300), color='blue')
        # Don't set format - should default to PNG
        
        metadata = self.validator._extract_metadata(image)
        
        self.assertEqual(metadata.dimensions, (400, 300))
        self.assertEqual(metadata.format, 'PNG')  # Should default to PNG
        self.assertEqual(metadata.color_mode, 'RGB')
    
    @pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not available")
    def test_validate_format_supported(self):
        """Test format validation with supported format"""
        image = Image.new('RGB', (512, 512), color='green')
        image.format = 'PNG'
        metadata = self.validator._extract_metadata(image)
        
        is_valid, error_msg, suggestions = self.validator._validate_format(image, metadata)
        
        self.assertTrue(is_valid)
        self.assertEqual(error_msg, "")
        self.assertEqual(suggestions, [])
    
    @pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not available")
    def test_validate_format_unsupported(self):
        """Test format validation with unsupported format"""
        image = Image.new('RGB', (512, 512), color='yellow')
        image.format = 'TIFF'  # Not in default supported formats
        metadata = self.validator._extract_metadata(image)
        
        is_valid, error_msg, suggestions = self.validator._validate_format(image, metadata)
        
        self.assertFalse(is_valid)
        self.assertIn("Unsupported format", error_msg)
        self.assertGreater(len(suggestions), 0)
        self.assertTrue(any("Convert to" in suggestion for suggestion in suggestions))
    
    @pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not available")
    def test_validate_dimensions_valid(self):
        """Test dimension validation with valid dimensions"""
        image = Image.new('RGB', (512, 512), color='purple')
        metadata = self.validator._extract_metadata(image)
        
        is_valid, error_msg, suggestions = self.validator._validate_dimensions(metadata, "i2v-A14B")
        
        self.assertTrue(is_valid)
        self.assertEqual(error_msg, "")
        self.assertEqual(suggestions, [])
    
    @pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not available")
    def test_validate_dimensions_too_small(self):
        """Test dimension validation with too small dimensions"""
        image = Image.new('RGB', (100, 100), color='orange')
        metadata = self.validator._extract_metadata(image)
        
        is_valid, error_msg, suggestions = self.validator._validate_dimensions(metadata, "i2v-A14B")
        
        self.assertFalse(is_valid)
        self.assertIn("too small", error_msg)
        self.assertGreater(len(suggestions), 0)
        self.assertTrue(any("Resize" in suggestion for suggestion in suggestions))
    
    @pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not available")
    def test_validate_dimensions_too_large(self):
        """Test dimension validation with too large dimensions"""
        image = Image.new('RGB', (5000, 5000), color='cyan')
        metadata = self.validator._extract_metadata(image)
        
        is_valid, error_msg, suggestions = self.validator._validate_dimensions(metadata, "i2v-A14B")
        
        self.assertFalse(is_valid)
        self.assertIn("too large", error_msg)
        self.assertGreater(len(suggestions), 0)
        self.assertTrue(any("Resize" in suggestion for suggestion in suggestions))
    
    @pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not available")
    def test_validate_file_size_valid(self):
        """Test file size validation with valid size"""
        # Create metadata with acceptable file size
        metadata = ImageMetadata(
            filename="test.png",
            format="PNG",
            dimensions=(512, 512),
            file_size_bytes=1024000,  # 1 MB
            aspect_ratio=1.0,
            color_mode="RGB",
            has_transparency=False,
            upload_timestamp=datetime.now()
        )
        
        is_valid, error_msg, suggestions = self.validator._validate_file_size(metadata)
        
        self.assertTrue(is_valid)
        self.assertEqual(error_msg, "")
        self.assertEqual(suggestions, [])
    
    @pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not available")
    def test_validate_file_size_too_large(self):
        """Test file size validation with too large size"""
        # Create metadata with excessive file size
        metadata = ImageMetadata(
            filename="test.png",
            format="PNG",
            dimensions=(512, 512),
            file_size_bytes=100 * 1024 * 1024,  # 100 MB
            aspect_ratio=1.0,
            color_mode="RGB",
            has_transparency=False,
            upload_timestamp=datetime.now()
        )
        
        is_valid, error_msg, suggestions = self.validator._validate_file_size(metadata)
        
        self.assertFalse(is_valid)
        self.assertIn("too large", error_msg)
        self.assertGreater(len(suggestions), 0)
        self.assertTrue(any("Compress" in suggestion for suggestion in suggestions))
    
    @pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not available")
    def test_generate_thumbnail(self):
        """Test thumbnail generation"""
        image = Image.new('RGB', (1000, 1000), color='magenta')
        
        thumbnail_data = self.validator._generate_thumbnail(image)
        
        self.assertIsNotNone(thumbnail_data)
        self.assertIsInstance(thumbnail_data, str)
        
        # Should be base64 encoded
        import base64
        try:
            decoded = base64.b64decode(thumbnail_data)
            self.assertGreater(len(decoded), 0)
        except Exception:
            self.fail("Thumbnail data is not valid base64")
    
    @pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not available")
    def test_analyze_image_quality_normal_image(self):
        """Test image quality analysis with normal image"""
        image = Image.new('RGB', (512, 512), color=(128, 128, 128))
        
        issues, suggestions = self.validator._analyze_image_quality(image)
        
        # Normal image should have minimal issues
        self.assertIsInstance(issues, list)
        self.assertIsInstance(suggestions, list)
    
    @pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not available")
    def test_analyze_image_quality_grayscale_image(self):
        """Test image quality analysis with grayscale image"""
        image = Image.new('L', (512, 512), color=128)
        
        issues, suggestions = self.validator._analyze_image_quality(image)
        
        # Grayscale image should be flagged
        self.assertGreater(len(issues), 0)
        self.assertTrue(any("grayscale" in issue.lower() for issue in issues))
        self.assertGreater(len(suggestions), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)