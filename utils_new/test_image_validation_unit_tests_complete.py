"""
Complete Unit Tests for Image Validation Functions
Comprehensive testing of individual validation functions and components
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
try:
    from enhanced_image_validation import (
        EnhancedImageValidator, ValidationFeedback, ImageMetadata
    )
    IMAGE_VALIDATION_AVAILABLE = True
except ImportError:
    IMAGE_VALIDATION_AVAILABLE = False


class TestImageMetadataComplete(unittest.TestCase):
    """Complete tests for ImageMetadata class functionality"""
    
    def test_image_metadata_creation_complete(self):
        """Test complete ImageMetadata object creation with all fields"""
        metadata = ImageMetadata(
            filename="test_image.png",
            format="PNG",
            dimensions=(1920, 1080),
            file_size_bytes=2048000,
            aspect_ratio=1920/1080,
            color_mode="RGB",
            has_transparency=False,
            upload_timestamp=datetime.now(),
            thumbnail_data="base64_thumbnail_data"
        )
        
        self.assertEqual(metadata.filename, "test_image.png")
        self.assertEqual(metadata.format, "PNG")
        self.assertEqual(metadata.dimensions, (1920, 1080))
        self.assertEqual(metadata.file_size_bytes, 2048000)
        self.assertAlmostEqual(metadata.aspect_ratio, 1920/1080, places=3)
        self.assertEqual(metadata.color_mode, "RGB")
        self.assertFalse(metadata.has_transparency)
        self.assertEqual(metadata.thumbnail_data, "base64_thumbnail_data")
    
    def test_file_size_mb_property_various_sizes(self):
        """Test file size MB calculation with various sizes"""
        test_cases = [
            (1024, 0.001),      # 1 KB
            (1048576, 1.0),     # 1 MB
            (10485760, 10.0),   # 10 MB
            (52428800, 50.0),   # 50 MB
            (104857600, 100.0)  # 100 MB
        ]
        
        for file_size_bytes, expected_mb in test_cases:
            with self.subTest(file_size_bytes=file_size_bytes):
                metadata = ImageMetadata(
                    filename="test.png",
                    format="PNG",
                    dimensions=(800, 600),
                    file_size_bytes=file_size_bytes,
                    aspect_ratio=800/600,
                    color_mode="RGB",
                    has_transparency=False,
                    upload_timestamp=datetime.now()
                )
                
                self.assertAlmostEqual(metadata.file_size_mb, expected_mb, places=2)
    
    def test_aspect_ratio_string_all_common_ratios(self):
        """Test aspect ratio string for all common ratios"""
        test_cases = [
            ((1920, 1080), "16:9 (Widescreen)"),
            ((1024, 768), "4:3 (Standard)"),
            ((512, 512), "1:1 (Square)"),
            ((1080, 1920), "9:16 (Portrait)"),
            ((1800, 1200), "3:2 (Photo)"),
            ((2560, 1080), "21:9 (Ultrawide)"),
            ((1600, 900), "16:9 (Widescreen)"),  # Another 16:9 variant
            ((800, 600), "4:3 (Standard)"),     # Another 4:3 variant
            ((1337, 420), "Custom")             # Custom ratio
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
                
                if "Custom" in expected_string:
                    self.assertIn("Custom", metadata.aspect_ratio_string)
                    self.assertIn(f"{dimensions[0]}:{dimensions[1]}", metadata.aspect_ratio_string)
                else:
                    self.assertEqual(metadata.aspect_ratio_string, expected_string)
    
    def test_to_dict_serialization_complete(self):
        """Test complete metadata serialization to dictionary"""
        timestamp = datetime.now()
        metadata = ImageMetadata(
            filename="test_complete.png",
            format="PNG",
            dimensions=(1280, 720),
            file_size_bytes=1536000,  # 1.5 MB
            aspect_ratio=1280/720,
            color_mode="RGBA",
            has_transparency=True,
            upload_timestamp=timestamp,
            thumbnail_data="base64_encoded_thumbnail"
        )
        
        metadata_dict = metadata.to_dict()
        
        # Verify all fields are present
        expected_fields = [
            'filename', 'format', 'dimensions', 'file_size_bytes',
            'file_size_mb', 'aspect_ratio', 'aspect_ratio_string',
            'color_mode', 'has_transparency', 'upload_timestamp',
            'thumbnail_data'
        ]
        
        for field in expected_fields:
            self.assertIn(field, metadata_dict, f"Field {field} missing from serialization")
        
        # Verify specific values
        self.assertEqual(metadata_dict['filename'], "test_complete.png")
        self.assertEqual(metadata_dict['format'], "PNG")
        self.assertEqual(metadata_dict['dimensions'], (1280, 720))
        self.assertEqual(metadata_dict['file_size_mb'], 1.5)
        self.assertEqual(metadata_dict['aspect_ratio_string'], "16:9 (Widescreen)")
        self.assertEqual(metadata_dict['color_mode'], "RGBA")
        self.assertTrue(metadata_dict['has_transparency'])
        self.assertEqual(metadata_dict['upload_timestamp'], timestamp.isoformat())
        self.assertEqual(metadata_dict['thumbnail_data'], "base64_encoded_thumbnail")


class TestValidationFeedbackComplete(unittest.TestCase):
    """Complete tests for ValidationFeedback class functionality"""
    
    def test_validation_feedback_all_severities(self):
        """Test ValidationFeedback with all severity levels"""
        severities = ["success", "warning", "error", "info"]
        
        for severity in severities:
            with self.subTest(severity=severity):
                feedback = ValidationFeedback(
                    is_valid=(severity != "error"),
                    severity=severity,
                    title=f"{severity.title()} Title",
                    message=f"Test {severity} message",
                    details=[f"{severity} detail 1", f"{severity} detail 2"],
                    suggestions=[f"{severity} suggestion 1", f"{severity} suggestion 2"]
                )
                
                self.assertEqual(feedback.severity, severity)
                self.assertEqual(feedback.title, f"{severity.title()} Title")
                self.assertEqual(feedback.message, f"Test {severity} message")
                self.assertEqual(len(feedback.details), 2)
                self.assertEqual(len(feedback.suggestions), 2)
    
    def test_validation_feedback_html_all_severities(self):
        """Test HTML generation for all severity levels"""
        severity_icons = {
            "success": "✅",
            "warning": "⚠️",
            "error": "❌",
            "info": "ℹ️"
        }
        
        severity_colors = {
            "success": "#28a745",
            "warning": "#ffc107",
            "error": "#dc3545",
            "info": "#6c757d"
        }
        
        for severity in severity_icons.keys():
            with self.subTest(severity=severity):
                feedback = ValidationFeedback(
                    is_valid=(severity != "error"),
                    severity=severity,
                    title=f"{severity.title()} Test",
                    message=f"Test {severity} message"
                )
                
                html = feedback.to_html()
                
                self.assertIsInstance(html, str)
                self.assertIn(f"{severity.title()} Test", html)
                self.assertIn(f"Test {severity} message", html)
                
                # Check for appropriate icon (if defined)
                if severity in severity_icons:
                    expected_icon = severity_icons[severity]
                    self.assertIn(expected_icon, html)
                
                # Check for appropriate color
                expected_color = severity_colors.get(severity, "#6c757d")
                self.assertIn(expected_color, html)
    
    def test_validation_feedback_html_with_complete_metadata(self):
        """Test HTML generation with complete metadata"""
        metadata = ImageMetadata(
            filename="complete_test.jpg",
            format="JPEG",
            dimensions=(1920, 1080),
            file_size_bytes=3145728,  # 3 MB
            aspect_ratio=1920/1080,
            color_mode="RGB",
            has_transparency=False,
            upload_timestamp=datetime.now(),
            thumbnail_data="base64_thumbnail"
        )
        
        feedback = ValidationFeedback(
            is_valid=True,
            severity="success",
            title="Complete Validation Success",
            message="Image passed all validation checks",
            details=["Format is supported", "Dimensions are acceptable", "File size is within limits"],
            suggestions=["Image is ready for generation"],
            metadata=metadata
        )
        
        html = feedback.to_html()
        
        # Check for metadata display
        self.assertIn("📊 Image Information", html)
        self.assertIn("1920×1080", html)
        self.assertIn("JPEG", html)
        self.assertIn("RGB", html)
        self.assertIn("3.00 MB", html)
        self.assertIn("16:9 (Widescreen)", html)
        self.assertIn("No", html)  # Transparency: No
        
        # Check for details and suggestions
        self.assertIn("Format is supported", html)
        self.assertIn("Image is ready for generation", html)


class TestEnhancedImageValidatorConfiguration(unittest.TestCase):
    """Complete tests for EnhancedImageValidator configuration"""
    
    def test_default_configuration_complete(self):
        """Test validator with complete default configuration"""
        if not IMAGE_VALIDATION_AVAILABLE:
            self.skipTest("Enhanced image validation not available")
        
        validator = EnhancedImageValidator()
        
        # Test all default configuration values
        self.assertEqual(validator.supported_formats, ["JPEG", "PNG", "WEBP", "BMP"])
        self.assertEqual(validator.max_file_size_mb, 50)
        self.assertEqual(validator.min_dimensions, (256, 256))
        self.assertEqual(validator.max_dimensions, (4096, 4096))
        self.assertEqual(validator.thumbnail_size, (150, 150))
        
        # Test quality thresholds
        self.assertIn("min_brightness", validator.quality_thresholds)
        self.assertIn("max_brightness", validator.quality_thresholds)
        self.assertIn("min_contrast", validator.quality_thresholds)
        self.assertIn("max_blur_threshold", validator.quality_thresholds)
        
        # Test model requirements
        self.assertIn("t2v-A14B", validator.model_requirements)
        self.assertIn("i2v-A14B", validator.model_requirements)
        self.assertIn("ti2v-5B", validator.model_requirements)
    
    def test_custom_configuration_complete(self):
        """Test validator with complete custom configuration"""
        if not IMAGE_VALIDATION_AVAILABLE:
            self.skipTest("Enhanced image validation not available")
        
        custom_config = {
            "supported_formats": ["PNG", "JPEG", "WEBP"],
            "max_file_size_mb": 25,
            "min_dimensions": (512, 512),
            "max_dimensions": (2048, 2048),
            "thumbnail_size": (200, 200)
        }
        
        validator = EnhancedImageValidator(custom_config)
        
        self.assertEqual(validator.supported_formats, ["PNG", "JPEG", "WEBP"])
        self.assertEqual(validator.max_file_size_mb, 25)
        self.assertEqual(validator.min_dimensions, (512, 512))
        self.assertEqual(validator.max_dimensions, (2048, 2048))
        self.assertEqual(validator.thumbnail_size, (200, 200))
    
    def test_model_requirements_complete_structure(self):
        """Test complete model requirements structure"""
        if not IMAGE_VALIDATION_AVAILABLE:
            self.skipTest("Enhanced image validation not available")
        
        validator = EnhancedImageValidator()
        
        required_models = ["t2v-A14B", "i2v-A14B", "ti2v-5B"]
        required_fields = ["preferred_aspect_ratios", "recommended_min_size", "notes"]
        
        for model in required_models:
            with self.subTest(model=model):
                self.assertIn(model, validator.model_requirements)
                
                model_req = validator.model_requirements[model]
                for field in required_fields:
                    self.assertIn(field, model_req, f"Field {field} missing for model {model}")
                
                # Verify field types
                self.assertIsInstance(model_req["preferred_aspect_ratios"], list)
                self.assertIsInstance(model_req["recommended_min_size"], tuple)
                self.assertIsInstance(model_req["notes"], str)


class TestImageValidationMethodsComplete(unittest.TestCase):
    """Complete tests for individual validation methods"""
    
    def setUp(self):
        """Set up test fixtures"""
        if not IMAGE_VALIDATION_AVAILABLE:
            self.skipTest("Enhanced image validation not available")
        
        self.validator = EnhancedImageValidator()
    
    @pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not available")
    def test_extract_metadata_all_color_modes(self):
        """Test metadata extraction from all color modes"""
        color_modes = [
            ('RGB', (255, 0, 0)),
            ('RGBA', (255, 0, 0, 128)),
            ('L', 128),  # Grayscale
            ('P', 0)     # Palette mode
        ]
        
        for mode, color in color_modes:
            with self.subTest(mode=mode):
                if mode == 'P':
                    # Create palette image
                    image = Image.new('P', (400, 300))
                    image.putpalette([i for i in range(256)] * 3)  # Simple palette
                else:
                    image = Image.new(mode, (400, 300), color=color)
                
                image.format = 'PNG'
                
                metadata = self.validator._extract_metadata(image)
                
                self.assertEqual(metadata.dimensions, (400, 300))
                self.assertEqual(metadata.format, 'PNG')
                self.assertEqual(metadata.color_mode, mode)
                self.assertAlmostEqual(metadata.aspect_ratio, 400/300, places=3)
                
                # Check transparency detection
                if mode == 'RGBA':
                    self.assertTrue(metadata.has_transparency)
                else:
                    self.assertFalse(metadata.has_transparency)
    
    @pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not available")
    def test_validate_format_all_supported_formats(self):
        """Test format validation with all supported formats"""
        if not hasattr(self.validator, '_validate_format'):
            self.skipTest("Legacy validation method not available")
        
        supported_formats = ["JPEG", "PNG", "WEBP", "BMP"]
        
        for format_name in supported_formats:
            with self.subTest(format=format_name):
                image = Image.new('RGB', (512, 512), color='blue')
                image.format = format_name
                metadata = self.validator._extract_metadata(image)
                
                is_valid, error_msg, suggestions = self.validator._validate_format(image, metadata)
                
                self.assertTrue(is_valid, f"Format {format_name} should be valid")
                self.assertEqual(error_msg, "")
                self.assertEqual(suggestions, [])
    
    @pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not available")
    def test_validate_format_all_unsupported_formats(self):
        """Test format validation with unsupported formats"""
        if not hasattr(self.validator, '_validate_format'):
            self.skipTest("Legacy validation method not available")
        
        unsupported_formats = ["TIFF", "GIF", "ICO", "PSD"]
        
        for format_name in unsupported_formats:
            with self.subTest(format=format_name):
                image = Image.new('RGB', (512, 512), color='red')
                image.format = format_name
                metadata = self.validator._extract_metadata(image)
                
                is_valid, error_msg, suggestions = self.validator._validate_format(image, metadata)
                
                self.assertFalse(is_valid, f"Format {format_name} should be invalid")
                self.assertIn("Unsupported format", error_msg)
                self.assertGreater(len(suggestions), 0)
                self.assertTrue(any("Convert to" in suggestion for suggestion in suggestions))
    
    @pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not available")
    def test_validate_dimensions_boundary_conditions(self):
        """Test dimension validation with boundary conditions"""
        if not hasattr(self.validator, '_validate_dimensions'):
            self.skipTest("Legacy validation method not available")
        
        # Test cases: (width, height, should_be_valid, description)
        test_cases = [
            (255, 255, False, "Just below minimum"),
            (256, 256, True, "Exactly minimum"),
            (257, 257, True, "Just above minimum"),
            (4095, 4095, True, "Just below maximum"),
            (4096, 4096, True, "Exactly maximum"),
            (4097, 4097, False, "Just above maximum"),
            (256, 255, False, "Width valid, height invalid"),
            (255, 256, False, "Height valid, width invalid"),
            (0, 0, False, "Zero dimensions"),
            (-1, -1, False, "Negative dimensions")
        ]
        
        for width, height, should_be_valid, description in test_cases:
            with self.subTest(dimensions=(width, height), description=description):
                metadata = ImageMetadata(
                    filename="test.png",
                    format="PNG",
                    dimensions=(width, height),
                    file_size_bytes=1000000,
                    aspect_ratio=width/height if height > 0 else 1.0,
                    color_mode="RGB",
                    has_transparency=False,
                    upload_timestamp=datetime.now()
                )
                
                is_valid, error_msg, suggestions = self.validator._validate_dimensions(metadata, "i2v-A14B")
                
                if should_be_valid:
                    self.assertTrue(is_valid, f"Dimensions {width}x{height} should be valid: {description}")
                    self.assertEqual(error_msg, "")
                    self.assertEqual(suggestions, [])
                else:
                    self.assertFalse(is_valid, f"Dimensions {width}x{height} should be invalid: {description}")
                    self.assertNotEqual(error_msg, "")
                    self.assertGreater(len(suggestions), 0)
    
    @pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not available")
    def test_validate_file_size_boundary_conditions(self):
        """Test file size validation with boundary conditions"""
        if not hasattr(self.validator, '_validate_file_size'):
            self.skipTest("Legacy validation method not available")
        
        max_size_mb = self.validator.max_file_size_mb
        max_size_bytes = max_size_mb * 1024 * 1024
        
        # Test cases: (file_size_bytes, should_be_valid, description)
        test_cases = [
            (max_size_bytes - 1, True, "Just below maximum"),
            (max_size_bytes, True, "Exactly maximum"),
            (max_size_bytes + 1, False, "Just above maximum"),
            (0, True, "Zero size"),
            (1024, True, "1 KB"),
            (1048576, True, "1 MB"),
            (max_size_bytes * 2, False, "Double maximum")
        ]
        
        for file_size_bytes, should_be_valid, description in test_cases:
            with self.subTest(file_size_bytes=file_size_bytes, description=description):
                metadata = ImageMetadata(
                    filename="test.png",
                    format="PNG",
                    dimensions=(512, 512),
                    file_size_bytes=file_size_bytes,
                    aspect_ratio=1.0,
                    color_mode="RGB",
                    has_transparency=False,
                    upload_timestamp=datetime.now()
                )
                
                is_valid, error_msg, suggestions = self.validator._validate_file_size(metadata)
                
                if should_be_valid:
                    self.assertTrue(is_valid, f"File size {file_size_bytes} bytes should be valid: {description}")
                    self.assertEqual(error_msg, "")
                    self.assertEqual(suggestions, [])
                else:
                    self.assertFalse(is_valid, f"File size {file_size_bytes} bytes should be invalid: {description}")
                    self.assertIn("too large", error_msg)
                    self.assertGreater(len(suggestions), 0)
    
    @pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not available")
    def test_generate_thumbnail_various_sizes(self):
        """Test thumbnail generation with various image sizes"""
        test_sizes = [
            (100, 100),    # Small square
            (1000, 1000),  # Large square
            (1920, 1080),  # Widescreen
            (1080, 1920),  # Portrait
            (3000, 2000),  # Large landscape
            (50, 200),     # Narrow vertical
            (200, 50)      # Wide horizontal
        ]
        
        for width, height in test_sizes:
            with self.subTest(size=(width, height)):
                image = Image.new('RGB', (width, height), color='purple')
                
                thumbnail_data = self.validator._generate_thumbnail(image)
                
                self.assertIsNotNone(thumbnail_data)
                self.assertIsInstance(thumbnail_data, str)
                
                # Should be base64 encoded
                import base64
                try:
                    decoded = base64.b64decode(thumbnail_data)
                    self.assertGreater(len(decoded), 0)
                    
                    # Verify it's a valid image by trying to load it
                    thumbnail_image = Image.open(io.BytesIO(decoded))
                    self.assertIsNotNone(thumbnail_image)
                    
                    # Verify thumbnail is within size limits
                    thumb_width, thumb_height = thumbnail_image.size
                    max_thumb_size = max(self.validator.thumbnail_size)
                    self.assertLessEqual(max(thumb_width, thumb_height), max_thumb_size)
                    
                except Exception as e:
                    self.fail(f"Thumbnail data is not valid: {e}")


if __name__ == '__main__':
    unittest.main(verbosity=2)