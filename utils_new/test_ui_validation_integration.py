"""
Integration Tests for UI Validation and Feedback System

Tests the integration between UI components, validation, error handling,
and progress feedback systems.
"""

import pytest
import gradio as gr
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import numpy as np

from ui_validation import UIValidationManager, UIValidationState
from ui_event_handlers import UIEventHandlers
from input_validation import ValidationResult, ValidationSeverity, ValidationIssue
from error_handler import UserFriendlyError, ErrorCategory, ErrorSeverity as ErrorSev

class TestUIValidationIntegration:
    """Test UI validation integration"""
    
    @pytest.fixture
    def validation_manager(self):
        """Create validation manager for testing"""
        config = {
            "max_prompt_length": 500,
            "enable_realtime_validation": True,
            "validation_delay_ms": 100
        }
        return UIValidationManager(config)
    
    @pytest.fixture
    def event_handlers(self):
        """Create event handlers for testing"""
        config = {
            "max_prompt_length": 500,
            "enable_realtime_validation": True
        }
        return UIEventHandlers(config)
    
    @pytest.fixture
    def mock_ui_components(self):
        """Create mock UI components"""
        return {
            'model_type': Mock(),
            'prompt_input': Mock(),
            'char_count': Mock(),
            'image_input': Mock(),
            'resolution': Mock(),
            'steps': Mock(),
            'notification_area': Mock(),
            'clear_notification_btn': Mock(),
            'validation_summary': Mock(),
            'progress_indicator': Mock()
        }
    
    def test_prompt_validation_realtime(self, validation_manager):
        """Test real-time prompt validation"""
        # Test empty prompt
        validation_html, is_valid, char_count = validation_manager.validate_prompt_realtime("", "t2v-A14B")
        assert char_count == "0/500"
        assert is_valid
        assert validation_html == ""
        
        # Test valid prompt
        prompt = "A beautiful sunset over the ocean with gentle waves"
        validation_html, is_valid, char_count = validation_manager.validate_prompt_realtime(prompt, "t2v-A14B")
        assert char_count == f"{len(prompt)}/500"
        assert is_valid
        
        # Test prompt too long
        long_prompt = "A" * 600
        validation_html, is_valid, char_count = validation_manager.validate_prompt_realtime(long_prompt, "t2v-A14B")
        assert char_count == "600/500"
        assert not is_valid
        assert "too long" in validation_html.lower()
    
    def test_image_validation_realtime(self, validation_manager):
        """Test real-time image validation"""
        # Test no image
        validation_html, is_valid = validation_manager.validate_image_realtime(None, "i2v-A14B")
        assert is_valid
        assert validation_html == ""
        
        # Test with valid PIL image
        test_image = Image.new('RGB', (1280, 720), color='red')
        validation_html, is_valid = validation_manager.validate_image_realtime(test_image, "i2v-A14B")
        assert is_valid
        
        # Test with very small image
        small_image = Image.new('RGB', (100, 100), color='red')
        validation_html, is_valid = validation_manager.validate_image_realtime(small_image, "i2v-A14B")
        assert not is_valid
        assert "too low" in validation_html.lower()
    
    def test_parameter_validation(self, validation_manager):
        """Test parameter validation"""
        # Test valid parameters
        params = {
            'model_type': 't2v-A14B',
            'resolution': '1280x720',
            'steps': 50
        }
        validation_html, is_valid = validation_manager.validate_generation_params(params, "t2v-A14B")
        assert is_valid
        
        # Test invalid parameters
        invalid_params = {
            'model_type': 'invalid-model',
            'resolution': '100x100',
            'steps': 200
        }
        validation_html, is_valid = validation_manager.validate_generation_params(invalid_params, "t2v-A14B")
        assert not is_valid
        assert "unknown model" in validation_html.lower() or "too high" in validation_html.lower()
    
    def test_comprehensive_validation_summary(self, validation_manager):
        """Test comprehensive validation summary"""
        # Set up validation states with errors and warnings
        validation_manager.validation_states = {
            'prompt': UIValidationState(
                is_valid=False,
                errors=["Prompt too long"],
                warnings=["Contains repetitive content"],
                suggestions=["Shorten the prompt"]
            ),
            'image': UIValidationState(
                is_valid=True,
                errors=[],
                warnings=["Low contrast image"],
                suggestions=["Use higher contrast image"]
            )
        }
        
        summary_html, all_valid = validation_manager.create_comprehensive_validation_summary()
        assert not all_valid
        assert "prompt too long" in summary_html.lower()
        assert "low contrast" in summary_html.lower()
        assert "issues found" in summary_html.lower()
    
    def test_progress_indicator_creation(self, validation_manager):
        """Test progress indicator creation"""
        # Test validation stage
        progress_html = validation_manager.create_progress_indicator("validation", 0.3, "Validating inputs...")
        assert "validation" in progress_html.lower()
        assert "30.0%" in progress_html
        assert "validating inputs" in progress_html.lower()
        
        # Test generation stage
        progress_html = validation_manager.create_progress_indicator("generation", 0.7, "Generating video...")
        assert "generation" in progress_html.lower()
        assert "70.0%" in progress_html
        assert "generating video" in progress_html.lower()
    
    def test_error_display_creation(self, validation_manager):
        """Test error display creation"""
        # Test with UserFriendlyError
        user_error = UserFriendlyError(
            category=ErrorCategory.INPUT_VALIDATION,
            severity=ErrorSev.MEDIUM,
            title="Validation Error",
            message="Invalid input provided",
            recovery_suggestions=["Check your inputs", "Try different values"],
            recovery_actions=[],
            technical_details="Field: prompt, Value: invalid"
        )
        
        error_html, show_display = validation_manager.create_error_display_with_recovery(user_error, "test")
        assert show_display
        assert "validation error" in error_html.lower()
        assert "invalid input provided" in error_html.lower()
        assert "check your inputs" in error_html.lower()
        
        # Test with regular exception
        regular_error = ValueError("Invalid value provided")
        error_html, show_display = validation_manager.create_error_display_with_recovery(regular_error, "test")
        assert show_display
        assert "invalid value provided" in error_html.lower()

class TestUIEventHandlersIntegration:
    """Test UI event handlers integration"""
    
    @pytest.fixture
    def event_handlers(self):
        """Create event handlers for testing"""
        return UIEventHandlers()
    
    @pytest.fixture
    def mock_components(self):
        """Create mock UI components"""
        components = {}
        component_names = [
            'model_type', 'prompt_input', 'char_count', 'image_input',
            'resolution', 'steps', 'notification_area', 'clear_notification_btn',
            'validation_summary', 'progress_indicator', 'generate_btn', 'queue_btn'
        ]
        
        for name in component_names:
            mock_component = Mock()
            mock_component.change = Mock()
            mock_component.click = Mock()
            components[name] = mock_component
        
        return components
    
    def test_component_registration(self, event_handlers, mock_components):
        """Test UI component registration"""
        event_handlers.register_components(mock_components)
        assert len(event_handlers.ui_components) == len(mock_components)
        assert 'model_type' in event_handlers.ui_components
    
    def test_model_type_change_handler(self, event_handlers):
        """Test model type change handler"""
        # Test T2V model (no image input)
        image_update, help_text, compatibility_html = event_handlers.handle_model_type_change("t2v-A14B")
        assert "text-to-video" in help_text.lower()
        assert "compatible" in compatibility_html.lower()
        
        # Test I2V model (requires image input)
        image_update, help_text, compatibility_html = event_handlers.handle_model_type_change("i2v-A14B")
        assert "image-to-video" in help_text.lower()
        assert "upload an image" in help_text.lower()
    
    def test_prompt_change_handler(self, event_handlers):
        """Test prompt change handler"""
        # Test valid prompt
        char_count, validation_html, show_validation = event_handlers.handle_prompt_change(
            "A beautiful sunset", "t2v-A14B"
        )
        assert "17/500" in char_count
        
        # Test empty prompt
        char_count, validation_html, show_validation = event_handlers.handle_prompt_change("", "t2v-A14B")
        assert "0/500" in char_count
        assert not show_validation
        
        # Test long prompt
        long_prompt = "A" * 600
        char_count, validation_html, show_validation = event_handlers.handle_prompt_change(long_prompt, "t2v-A14B")
        assert "600/500" in char_count
        assert show_validation
    
    def test_image_change_handler(self, event_handlers):
        """Test image change handler"""
        # Test no image
        validation_html, show_validation = event_handlers.handle_image_change(None, "i2v-A14B")
        assert not show_validation
        
        # Test with valid image
        test_image = Image.new('RGB', (1280, 720), color='blue')
        validation_html, show_validation = event_handlers.handle_image_change(test_image, "i2v-A14B")
        # Should not show validation errors for valid image
        if show_validation:
            assert "error" not in validation_html.lower()
    
    def test_parameter_change_handler(self, event_handlers):
        """Test parameter change handler"""
        # Test valid parameters
        validation_html, show_validation = event_handlers.handle_parameter_change(
            "t2v-A14B", "1280x720", 50
        )
        # Should not show validation errors for valid parameters
        if show_validation:
            assert "error" not in validation_html.lower()
        
        # Test invalid parameters
        validation_html, show_validation = event_handlers.handle_parameter_change(
            "invalid-model", "invalid-resolution", 200
        )
        assert show_validation
        assert "error" in validation_html.lower() or "invalid" in validation_html.lower()
    
    @patch('ui_event_handlers.generate_video')
    def test_generate_video_handler(self, mock_generate, event_handlers):
        """Test video generation handler"""
        # Mock successful generation
        mock_generate.return_value = {
            'success': True,
            'video_path': '/path/to/video.mp4'
        }
        
        status, video_update, notification_html, show_notification = event_handlers.handle_generate_video(
            "t2v-A14B", "A beautiful sunset", None, "1280x720", 50, "", 1.0
        )
        
        assert "completed successfully" in status.lower()
        assert show_notification
        assert "successfully" in notification_html.lower()
        
        # Mock failed generation
        mock_generate.return_value = {
            'success': False,
            'error': 'Generation failed'
        }
        
        status, video_update, notification_html, show_notification = event_handlers.handle_generate_video(
            "t2v-A14B", "A beautiful sunset", None, "1280x720", 50, "", 1.0
        )
        
        assert "failed" in status.lower()
        assert show_notification
    
    def test_queue_generation_handler(self, event_handlers):
        """Test queue generation handler"""
        notification_html, show_notification = event_handlers.handle_queue_generation(
            "t2v-A14B", "A beautiful sunset", None, "1280x720", 50, "", 1.0
        )
        
        assert show_notification
        assert "queue" in notification_html.lower()
    
    def test_validation_request_comprehensive(self, event_handlers):
        """Test comprehensive validation of generation request"""
        # Test valid request
        is_valid, message = event_handlers._validate_generation_request(
            "t2v-A14B", "A beautiful sunset over the ocean", None, "1280x720", 50, "", 1.0
        )
        assert is_valid
        assert "validation passed" in message.lower()
        
        # Test invalid request (empty prompt)
        is_valid, message = event_handlers._validate_generation_request(
            "t2v-A14B", "", None, "1280x720", 50, "", 1.0
        )
        assert not is_valid
        assert "validation failed" in message.lower()

class TestUIIntegrationEndToEnd:
    """End-to-end integration tests"""
    
    def test_full_validation_workflow(self):
        """Test complete validation workflow from input to display"""
        validation_manager = UIValidationManager()
        
        # Simulate user input sequence
        # 1. User selects model type
        model_type = "t2v-A14B"
        
        # 2. User enters prompt
        prompt = "A serene lake with gentle waves reflecting the golden sunset"
        validation_html, is_valid, char_count = validation_manager.validate_prompt_realtime(prompt, model_type)
        assert is_valid
        assert len(prompt) < 500
        
        # 3. User sets parameters
        params = {
            'model_type': model_type,
            'resolution': '1280x720',
            'steps': 50
        }
        param_validation_html, param_valid = validation_manager.validate_generation_params(params, model_type)
        assert param_valid
        
        # 4. Get comprehensive summary
        summary_html, all_valid = validation_manager.create_comprehensive_validation_summary()
        assert all_valid
    
    def test_error_recovery_workflow(self):
        """Test error recovery workflow"""
        validation_manager = UIValidationManager()
        
        # Simulate validation error
        long_prompt = "A" * 600  # Too long
        validation_html, is_valid, char_count = validation_manager.validate_prompt_realtime(long_prompt, "t2v-A14B")
        assert not is_valid
        assert "too long" in validation_html.lower()
        
        # Simulate user fixing the error
        fixed_prompt = "A beautiful sunset over the ocean"
        validation_html, is_valid, char_count = validation_manager.validate_prompt_realtime(fixed_prompt, "t2v-A14B")
        assert is_valid
        
        # Verify comprehensive validation now passes
        summary_html, all_valid = validation_manager.create_comprehensive_validation_summary()
        assert all_valid
    
    def test_progress_tracking_workflow(self):
        """Test progress tracking workflow"""
        validation_manager = UIValidationManager()
        
        # Simulate generation progress stages
        stages = [
            ("validation", 0.1, "Validating inputs..."),
            ("model_loading", 0.2, "Loading model..."),
            ("generation", 0.5, "Generating video..."),
            ("post_processing", 0.8, "Processing output..."),
            ("saving", 0.9, "Saving video..."),
            ("complete", 1.0, "Generation complete!")
        ]
        
        for stage, progress, message in stages:
            progress_html = validation_manager.create_progress_indicator(stage, progress, message)
            assert stage.replace('_', ' ').title() in progress_html
            assert f"{progress * 100:.1f}%" in progress_html
            assert message in progress_html

if __name__ == "__main__":
    pytest.main([__file__, "-v"])