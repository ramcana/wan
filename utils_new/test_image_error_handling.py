"""
Comprehensive Test Suite for Image Error Handling System
Tests error handling, recovery suggestions, and user-friendly error messages
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from PIL import Image
import io

# Import the modules to test
from image_error_handler import (
    ImageErrorHandler, ImageError, ImageErrorType, ErrorContext, RecoveryAction,
    create_format_error, create_dimension_error, create_file_size_error,
    create_system_error, get_error_handler
)

from enhanced_image_validation import (
    EnhancedImageValidator, ValidationFeedback
)

class TestImageErrorType:
    """Test ImageErrorType enumeration"""
    
    def test_error_type_values(self):
        """Test that all error types have correct values"""
        assert ImageErrorType.UNSUPPORTED_FORMAT.value == "unsupported_format"
        assert ImageErrorType.TOO_SMALL.value == "too_small"
        assert ImageErrorType.FILE_TOO_LARGE.value == "file_too_large"
        assert ImageErrorType.PIL_NOT_AVAILABLE.value == "pil_not_available"
        assert ImageErrorType.CORRUPTED_FILE.value == "corrupted_file"

class TestErrorContext:
    """Test ErrorContext class"""
    
    def test_error_context_creation(self):
        """Test basic error context creation"""
        context = ErrorContext(
            image_type="start",
            model_type="i2v-A14B",
            file_path="test.jpg",
            file_size=1024000,
            dimensions=(1280, 720),
            format="JPEG",
            operation="validation",
            user_action="upload"
        )
        
        assert context.image_type == "start"
        assert context.model_type == "i2v-A14B"
        assert context.file_path == "test.jpg"
        assert context.file_size == 1024000
        assert context.dimensions == (1280, 720)
        assert context.format == "JPEG"
        assert context.operation == "validation"
        assert context.user_action == "upload"
    
    def test_error_context_defaults(self):
        """Test error context with default values"""
        context = ErrorContext()
        
        assert context.image_type == "unknown"
        assert context.model_type == "unknown"
        assert context.file_path is None
        assert context.operation == "validation"
        assert context.user_action == "upload"

class TestRecoveryAction:
    """Test RecoveryAction class"""
    
    def test_recovery_action_creation(self):
        """Test recovery action creation"""
        action = RecoveryAction(
            action_type="convert",
            title="Convert Image Format",
            description="Convert your image to a supported format",
            instructions=["Open image editor", "Save as PNG"],
            tools_needed=["Image editor"],
            estimated_time="2-3 minutes",
            difficulty="easy"
        )
        
        assert action.action_type == "convert"
        assert action.title == "Convert Image Format"
        assert action.description == "Convert your image to a supported format"
        assert len(action.instructions) == 2
        assert action.tools_needed == ["Image editor"]
        assert action.estimated_time == "2-3 minutes"
        assert action.difficulty == "easy"

class TestImageError:
    """Test ImageError class"""
    
    def test_image_error_creation(self):
        """Test basic image error creation"""
        context = ErrorContext(image_type="start", model_type="i2v-A14B")
        
        error = ImageError(
            error_type=ImageErrorType.UNSUPPORTED_FORMAT,
            severity="error",
            title="Unsupported Format",
            message="TIFF format is not supported",
            technical_details="PIL cannot process TIFF",
            context=context,
            recovery_actions=[],
            prevention_tips=["Use PNG or JPEG format"]
        )
        
        assert error.error_type == ImageErrorType.UNSUPPORTED_FORMAT
        assert error.severity == "error"
        assert error.title == "Unsupported Format"
        assert error.message == "TIFF format is not supported"
        assert error.technical_details == "PIL cannot process TIFF"
        assert error.context == context
        assert error.prevention_tips == ["Use PNG or JPEG format"]
    
    def test_error_to_user_friendly_dict(self):
        """Test conversion to user-friendly dictionary"""
        context = ErrorContext(image_type="start", model_type="i2v-A14B")
        action = RecoveryAction(
            action_type="convert",
            title="Convert Format",
            description="Convert to PNG",
            instructions=["Use image editor"],
            difficulty="easy"
        )
        
        error = ImageError(
            error_type=ImageErrorType.UNSUPPORTED_FORMAT,
            severity="error",
            title="Unsupported Format",
            message="Format not supported",
            context=context,
            recovery_actions=[action],
            prevention_tips=["Use supported formats"]
        )
        
        result = error.to_user_friendly_dict()
        
        assert result["error_type"] == "unsupported_format"
        assert result["severity"] == "error"
        assert result["title"] == "Unsupported Format"
        assert result["message"] == "Format not supported"
        assert len(result["recovery_actions"]) == 1
        assert result["recovery_actions"][0]["type"] == "convert"
        assert result["recovery_actions"][0]["title"] == "Convert Format"
        assert result["prevention_tips"] == ["Use supported formats"]
        assert result["context"]["image_type"] == "start"
        assert result["context"]["model_type"] == "i2v-A14B"

class TestImageErrorHandler:
    """Test ImageErrorHandler class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.handler = ImageErrorHandler()
        self.context = ErrorContext(
            image_type="start",
            model_type="i2v-A14B",
            operation="validation"
        )
    
    def test_handler_initialization(self):
        """Test error handler initialization"""
        handler = ImageErrorHandler()
        
        assert handler.config == {}
        assert isinstance(handler.error_registry, dict)
        assert isinstance(handler.recovery_registry, dict)
        assert ImageErrorType.UNSUPPORTED_FORMAT in handler.error_registry
        assert ImageErrorType.UNSUPPORTED_FORMAT in handler.recovery_registry
    
    def test_handler_custom_config(self):
        """Test error handler with custom config"""
        config = {"custom_setting": "value"}
        handler = ImageErrorHandler(config)
        
        assert handler.config == config
    
    def test_handle_format_error_unsupported(self):
        """Test handling unsupported format error"""
        error = self.handler.handle_format_error("TIFF", self.context)
        
        assert error.error_type == ImageErrorType.UNSUPPORTED_FORMAT
        assert error.severity == "error"
        assert "TIFF" in error.message
        assert "Unsupported" in error.title
        assert len(error.recovery_actions) > 0
        assert any("convert" in action.action_type.lower() for action in error.recovery_actions)
    
    def test_handle_format_error_supported(self):
        """Test handling supported format (should return corrupted file error)"""
        error = self.handler.handle_format_error("JPEG", self.context)
        
        assert error.error_type == ImageErrorType.CORRUPTED_FILE
        assert error.severity == "error"
    
    def test_handle_dimension_error_too_small(self):
        """Test handling too small dimension error"""
        error = self.handler.handle_dimension_error(
            (100, 100), (256, 256), (4096, 4096), self.context
        )
        
        assert error.error_type == ImageErrorType.TOO_SMALL
        assert error.severity == "error"
        assert "100" in error.message
        assert "256" in error.message
        assert len(error.recovery_actions) > 0
        assert any("resize" in action.action_type.lower() for action in error.recovery_actions)
    
    def test_handle_dimension_error_too_large(self):
        """Test handling too large dimension error"""
        error = self.handler.handle_dimension_error(
            (5000, 5000), (256, 256), (4096, 4096), self.context
        )
        
        assert error.error_type == ImageErrorType.TOO_LARGE
        assert error.severity == "error"
        assert "5000" in error.message
        assert "4096" in error.message
    
    def test_handle_file_size_error(self):
        """Test handling file size error"""
        error = self.handler.handle_file_size_error(100.0, 50.0, self.context)
        
        assert error.error_type == ImageErrorType.FILE_TOO_LARGE
        assert error.severity == "error"
        assert "100" in error.message
        assert "50" in error.message
        assert len(error.recovery_actions) > 0
    
    def test_handle_quality_error_dark(self):
        """Test handling dark image quality error"""
        error = self.handler.handle_quality_error("brightness_low", 10.0, self.context)
        
        assert error.error_type == ImageErrorType.TOO_DARK
        assert error.severity == "warning"
        assert "10.0" in error.message
    
    def test_handle_quality_error_bright(self):
        """Test handling bright image quality error"""
        error = self.handler.handle_quality_error("brightness_high", 250.0, self.context)
        
        assert error.error_type == ImageErrorType.TOO_BRIGHT
        assert error.severity == "warning"
        assert "250.0" in error.message
    
    def test_handle_quality_error_low_contrast(self):
        """Test handling low contrast quality error"""
        error = self.handler.handle_quality_error("contrast_low", 5.0, self.context)
        
        assert error.error_type == ImageErrorType.LOW_CONTRAST
        assert error.severity == "warning"
        assert "5.0" in error.message
    
    def test_handle_compatibility_error(self):
        """Test handling image compatibility error"""
        error = self.handler.handle_compatibility_error(
            (1280, 720), (1920, 1080), self.context
        )
        
        assert error.error_type == ImageErrorType.INCOMPATIBLE_IMAGES
        assert error.severity == "warning"
        assert "1280" in error.message
        assert "1920" in error.message
    
    def test_handle_system_error_pil(self):
        """Test handling PIL not available error"""
        error = self.handler.handle_system_error(
            ImageErrorType.PIL_NOT_AVAILABLE, context=self.context
        )
        
        assert error.error_type == ImageErrorType.PIL_NOT_AVAILABLE
        assert error.severity == "error"
        assert "PIL" in error.message or "Pillow" in error.message
    
    def test_handle_system_error_with_exception(self):
        """Test handling system error with exception"""
        exception = ValueError("Test exception")
        error = self.handler.handle_system_error(
            ImageErrorType.UNKNOWN_ERROR, exception=exception, context=self.context
        )
        
        assert error.error_type == ImageErrorType.UNKNOWN_ERROR
        assert error.severity == "error"
        assert "ValueError: Test exception" in error.technical_details
    
    def test_handle_processing_error(self):
        """Test handling processing error"""
        exception = RuntimeError("Processing failed")
        error = self.handler.handle_processing_error("validation", exception, self.context)
        
        assert error.error_type == ImageErrorType.UNKNOWN_ERROR
        assert error.severity == "error"
        assert "RuntimeError: Processing failed" in error.technical_details
    
    def test_classify_exception_pil(self):
        """Test exception classification for PIL errors"""
        exception = ImportError("No module named 'PIL'")
        error_type = self.handler._classify_exception(exception)
        
        assert error_type == ImageErrorType.PIL_NOT_AVAILABLE
    
    def test_classify_exception_numpy(self):
        """Test exception classification for NumPy errors"""
        exception = ImportError("No module named 'numpy'")
        error_type = self.handler._classify_exception(exception)
        
        assert error_type == ImageErrorType.NUMPY_NOT_AVAILABLE
    
    def test_classify_exception_permission(self):
        """Test exception classification for permission errors"""
        exception = PermissionError("Access denied")
        error_type = self.handler._classify_exception(exception)
        
        assert error_type == ImageErrorType.PERMISSION_DENIED
    
    def test_classify_exception_memory(self):
        """Test exception classification for memory errors"""
        exception = MemoryError("Out of memory")
        error_type = self.handler._classify_exception(exception)
        
        assert error_type == ImageErrorType.MEMORY_INSUFFICIENT
    
    def test_classify_exception_corrupted(self):
        """Test exception classification for corrupted file errors"""
        exception = ValueError("Image file is corrupted")
        error_type = self.handler._classify_exception(exception)
        
        assert error_type == ImageErrorType.CORRUPTED_FILE
    
    def test_classify_exception_unknown(self):
        """Test exception classification for unknown errors"""
        exception = RuntimeError("Unknown error")
        error_type = self.handler._classify_exception(exception)
        
        assert error_type == ImageErrorType.UNKNOWN_ERROR
    
    def test_create_validation_summary_success(self):
        """Test creating validation summary with no errors"""
        summary = self.handler.create_validation_summary([])
        
        assert summary["status"] == "success"
        assert summary["message"] == "All validations passed"
        assert summary["errors"] == []
        assert summary["warnings"] == []
    
    def test_create_validation_summary_errors(self):
        """Test creating validation summary with errors"""
        error1 = ImageError(
            error_type=ImageErrorType.TOO_SMALL,
            severity="error",
            title="Too Small",
            message="Image too small",
            context=self.context
        )
        error2 = ImageError(
            error_type=ImageErrorType.TOO_DARK,
            severity="warning",
            title="Too Dark",
            message="Image too dark",
            context=self.context
        )
        
        summary = self.handler.create_validation_summary([error1, error2])
        
        assert summary["status"] == "error"
        assert "1 error(s) and 1 warning(s)" in summary["message"]
        assert len(summary["errors"]) == 1
        assert len(summary["warnings"]) == 1
    
    def test_create_validation_summary_warnings_only(self):
        """Test creating validation summary with warnings only"""
        warning = ImageError(
            error_type=ImageErrorType.TOO_DARK,
            severity="warning",
            title="Too Dark",
            message="Image too dark",
            context=self.context
        )
        
        summary = self.handler.create_validation_summary([warning])
        
        assert summary["status"] == "warning"
        assert "1 warning(s)" in summary["message"]
        assert len(summary["errors"]) == 0
        assert len(summary["warnings"]) == 1

class TestConvenienceFunctions:
    """Test convenience functions"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.context = ErrorContext(image_type="start", model_type="i2v-A14B")
    
    def test_create_format_error(self):
        """Test create_format_error convenience function"""
        error = create_format_error("TIFF", self.context)
        
        assert error.error_type == ImageErrorType.UNSUPPORTED_FORMAT
        assert error.severity == "error"
        assert "TIFF" in error.message
    
    def test_create_dimension_error(self):
        """Test create_dimension_error convenience function"""
        error = create_dimension_error(
            (100, 100), (256, 256), (4096, 4096), self.context
        )
        
        assert error.error_type == ImageErrorType.TOO_SMALL
        assert error.severity == "error"
    
    def test_create_file_size_error(self):
        """Test create_file_size_error convenience function"""
        error = create_file_size_error(100.0, 50.0, self.context)
        
        assert error.error_type == ImageErrorType.FILE_TOO_LARGE
        assert error.severity == "error"
    
    def test_create_system_error(self):
        """Test create_system_error convenience function"""
        error = create_system_error(ImageErrorType.PIL_NOT_AVAILABLE, context=self.context)
        
        assert error.error_type == ImageErrorType.PIL_NOT_AVAILABLE
        assert error.severity == "error"
    
    def test_get_error_handler(self):
        """Test get_error_handler convenience function"""
        config = {"test": "value"}
        handler = get_error_handler(config)
        
        assert isinstance(handler, ImageErrorHandler)
        assert handler.config == config

@patch('enhanced_image_validation.PIL_AVAILABLE', True)
class TestEnhancedImageValidatorErrorHandling:
    """Test enhanced image validator with comprehensive error handling"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.validator = EnhancedImageValidator()
        
        # Mock PIL Image
        self.mock_image = Mock()
        self.mock_image.size = (1280, 720)
        self.mock_image.format = "JPEG"
        self.mock_image.mode = "RGB"
        self.mock_image.filename = "test.jpg"
        self.mock_image.copy.return_value = self.mock_image
    
    @patch('enhanced_image_validation.EnhancedImageValidator._extract_metadata')
    def test_validate_with_format_error(self, mock_metadata):
        """Test validation with format error"""
        # Mock metadata for unsupported format
        mock_metadata.return_value = Mock(
            filename="test.tiff",
            format="TIFF",
            dimensions=(1280, 720),
            file_size_bytes=1000000,
            file_size_mb=1.0,
            aspect_ratio=16/9,
            color_mode="RGB",
            has_transparency=False,
            upload_timestamp=datetime.now()
        )
        
        result = self.validator.validate_image_upload(self.mock_image, "start", "i2v-A14B")
        
        assert result.is_valid is False
        assert result.severity == "error"
        assert "format" in result.title.lower() or "invalid" in result.title.lower()
    
    @patch('enhanced_image_validation.EnhancedImageValidator._extract_metadata')
    def test_validate_with_dimension_error(self, mock_metadata):
        """Test validation with dimension error"""
        # Mock metadata for too small image
        mock_metadata.return_value = Mock(
            filename="small.jpg",
            format="JPEG",
            dimensions=(100, 100),
            file_size_bytes=10000,
            file_size_mb=0.01,
            aspect_ratio=1.0,
            color_mode="RGB",
            has_transparency=False,
            upload_timestamp=datetime.now()
        )
        
        result = self.validator.validate_image_upload(self.mock_image, "start", "i2v-A14B")
        
        assert result.is_valid is False
        assert result.severity == "error"
        assert "small" in result.title.lower() or "invalid" in result.title.lower()
    
    @patch('enhanced_image_validation.EnhancedImageValidator._extract_metadata')
    def test_validate_with_file_size_error(self, mock_metadata):
        """Test validation with file size error"""
        # Mock metadata for too large file
        mock_metadata.return_value = Mock(
            filename="large.jpg",
            format="JPEG",
            dimensions=(2000, 2000),
            file_size_bytes=100 * 1024 * 1024,
            file_size_mb=100.0,
            aspect_ratio=1.0,
            color_mode="RGB",
            has_transparency=False,
            upload_timestamp=datetime.now()
        )
        
        result = self.validator.validate_image_upload(self.mock_image, "start", "i2v-A14B")
        
        assert result.is_valid is False
        assert result.severity == "error"
        assert "large" in result.title.lower() or "invalid" in result.title.lower()
    
    @patch('enhanced_image_validation.EnhancedImageValidator._extract_metadata')
    @patch('enhanced_image_validation.NUMPY_AVAILABLE', True)
    @patch('numpy.array')
    @patch('numpy.mean')
    def test_validate_with_quality_warning(self, mock_mean, mock_array, mock_metadata):
        """Test validation with quality warning"""
        # Mock metadata for valid image
        mock_metadata.return_value = Mock(
            filename="dark.jpg",
            format="JPEG",
            dimensions=(1280, 720),
            file_size_bytes=1000000,
            file_size_mb=1.0,
            aspect_ratio=16/9,
            color_mode="RGB",
            has_transparency=False,
            upload_timestamp=datetime.now()
        )
        
        # Mock numpy analysis for dark image
        mock_array.return_value = Mock()
        mock_mean.return_value = 10  # Very dark
        
        result = self.validator.validate_image_upload(self.mock_image, "start", "i2v-A14B")
        
        assert result.is_valid is True
        assert result.severity == "warning"
        assert "warning" in result.title.lower()
    
    def test_validate_processing_error(self):
        """Test validation with processing error"""
        # Mock image that causes exception during metadata extraction
        mock_image = Mock()
        mock_image.size = None  # This should cause an error
        
        result = self.validator.validate_image_upload(mock_image, "start", "i2v-A14B")
        
        assert result.is_valid is False
        assert result.severity == "error"
    
    @patch('enhanced_image_validation.EnhancedImageValidator._extract_metadata')
    def test_validate_compatibility_with_errors(self, mock_metadata):
        """Test compatibility validation with errors"""
        # Mock different metadata for start and end images
        mock_metadata.side_effect = [
            Mock(
                filename="start.jpg", format="JPEG", dimensions=(1280, 720),
                file_size_bytes=1000000, aspect_ratio=16/9, color_mode="RGB",
                has_transparency=False, upload_timestamp=datetime.now()
            ),
            Mock(
                filename="end.jpg", format="PNG", dimensions=(1920, 1080),
                file_size_bytes=2000000, aspect_ratio=16/9, color_mode="RGBA",
                has_transparency=True, upload_timestamp=datetime.now()
            )
        ]
        
        start_image = Mock()
        end_image = Mock()
        
        result = self.validator.validate_image_compatibility(start_image, end_image)
        
        assert result.is_valid is True
        assert result.severity == "warning"
        assert "compatibility" in result.title.lower()

class TestErrorHandlingIntegration:
    """Test integration between error handling and validation systems"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.validator = EnhancedImageValidator()
        self.handler = ImageErrorHandler()
    
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
    
    def test_comprehensive_feedback_creation_success(self):
        """Test comprehensive feedback creation with no errors"""
        metadata = Mock(
            dimensions=(1280, 720),
            format="JPEG",
            file_size_mb=1.0,
            aspect_ratio_string="16:9 (Widescreen)"
        )
        
        feedback = self.validator._create_comprehensive_feedback(
            [], metadata, "start", "i2v-A14B"
        )
        
        assert feedback.is_valid is True
        assert feedback.severity == "success"
        assert "successfully" in feedback.title.lower()
        assert len(feedback.details) > 0
    
    def test_comprehensive_feedback_creation_with_errors(self):
        """Test comprehensive feedback creation with errors"""
        context = ErrorContext(image_type="start", model_type="i2v-A14B")
        error = self.handler.handle_format_error("TIFF", context)
        
        metadata = Mock(
            dimensions=(1280, 720),
            format="TIFF",
            file_size_mb=1.0,
            aspect_ratio_string="16:9 (Widescreen)"
        )
        
        feedback = self.validator._create_comprehensive_feedback(
            [error], metadata, "start", "i2v-A14B"
        )
        
        assert feedback.is_valid is False
        assert feedback.severity == "error"
        assert "invalid" in feedback.title.lower()
        assert len(feedback.details) > 0
        assert len(feedback.suggestions) > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])