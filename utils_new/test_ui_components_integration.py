"""
UI Components Integration Tests

Tests the integration between UI components, validation feedback,
error handling, and user interactions.
"""

import pytest
import gradio as gr
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from PIL import Image

# Mock Gradio components for testing
class MockGradioComponent:
    def __init__(self, value=None, visible=True):
        self.value = value
        self.visible = visible
        self.change_handlers = []
        self.click_handlers = []
    
    def change(self, fn, inputs, outputs):
        self.change_handlers.append((fn, inputs, outputs))
        return self
    
    def click(self, fn, inputs, outputs):
        self.click_handlers.append((fn, inputs, outputs))
        return self
    
    def update(self, value=None, visible=None, **kwargs):
        if value is not None:
            self.value = value
        if visible is not None:
            self.visible = visible
        return self

class TestUIComponentsIntegration:
    """Test UI components integration with validation and feedback"""
    
    @pytest.fixture
    def mock_ui_instance(self):
        """Create mock UI instance with components"""
        from ui import Wan22UI
        
        # Mock the UI instance
        ui_instance = Mock(spec=Wan22UI)
        ui_instance.config = {
            "max_prompt_length": 500,
            "enable_realtime_validation": True
        }
        
        # Create mock components
        ui_instance.generation_components = {
            'model_type': MockGradioComponent(value="t2v-A14B"),
            'prompt_input': MockGradioComponent(value=""),
            'char_count': MockGradioComponent(value="0/500"),
            'image_input': MockGradioComponent(value=None, visible=False),
            'resolution': MockGradioComponent(value="1280x720"),
            'steps': MockGradioComponent(value=50),
            'notification_area': MockGradioComponent(value="", visible=False),
            'clear_notification_btn': MockGradioComponent(visible=False),
            'validation_summary': MockGradioComponent(value="", visible=False),
            'progress_indicator': MockGradioComponent(value="", visible=False),
            'generate_btn': MockGradioComponent(),
            'queue_btn': MockGradioComponent(),
            'output_video': MockGradioComponent(visible=False)
        }
        
        return ui_instance
    
    def test_model_type_change_integration(self, mock_ui_instance):
        """Test model type change affects other components"""
        from ui_event_handlers import UIEventHandlers
        
        event_handlers = UIEventHandlers()
        event_handlers.register_components(mock_ui_instance.generation_components)
        
        # Test changing to I2V model
        image_update, help_text, compatibility_html = event_handlers.handle_model_type_change("i2v-A14B")
        
        # Verify image input becomes visible
        assert help_text is not None
        assert "image-to-video" in help_text.lower()
        assert compatibility_html is not None
    
    def test_prompt_validation_feedback_integration(self, mock_ui_instance):
        """Test prompt validation provides real-time feedback"""
        from ui_event_handlers import UIEventHandlers
        
        event_handlers = UIEventHandlers()
        event_handlers.register_components(mock_ui_instance.generation_components)
        
        # Test valid prompt
        char_count, validation_html, show_validation = event_handlers.handle_prompt_change(
            "A beautiful sunset over the ocean", "t2v-A14B"
        )
        
        assert "34/500" in char_count
        # Valid prompt should not show validation errors
        if show_validation:
            assert "error" not in validation_html.lower()
        
        # Test invalid prompt (too long)
        long_prompt = "A" * 600
        char_count, validation_html, show_validation = event_handlers.handle_prompt_change(
            long_prompt, "t2v-A14B"
        )
        
        assert "600/500" in char_count
        assert show_validation
        assert "too long" in validation_html.lower()
    
    def test_image_validation_feedback_integration(self, mock_ui_instance):
        """Test image validation provides feedback"""
        from ui_event_handlers import UIEventHandlers
        
        event_handlers = UIEventHandlers()
        event_handlers.register_components(mock_ui_instance.generation_components)
        
        # Test valid image
        valid_image = Image.new('RGB', (1280, 720), color='red')
        validation_html, show_validation = event_handlers.handle_image_change(valid_image, "i2v-A14B")
        
        # Valid image should not show errors
        if show_validation:
            assert "error" not in validation_html.lower()
        
        # Test invalid image (too small)
        small_image = Image.new('RGB', (100, 100), color='red')
        validation_html, show_validation = event_handlers.handle_image_change(small_image, "i2v-A14B")
        
        assert show_validation
        assert "too low" in validation_html.lower()
    
    def test_parameter_validation_integration(self, mock_ui_instance):
        """Test parameter validation integration"""
        from ui_event_handlers import UIEventHandlers
        
        event_handlers = UIEventHandlers()
        event_handlers.register_components(mock_ui_instance.generation_components)
        
        # Test valid parameters
        validation_html, show_validation = event_handlers.handle_parameter_change(
            "t2v-A14B", "1280x720", 50
        )
        
        # Valid parameters should not show errors
        if show_validation:
            assert "error" not in validation_html.lower()
        
        # Test invalid parameters
        validation_html, show_validation = event_handlers.handle_parameter_change(
            "invalid-model", "invalid-resolution", 200
        )
        
        assert show_validation
        assert ("invalid" in validation_html.lower() or 
                "unknown" in validation_html.lower() or 
                "too high" in validation_html.lower())
    
    def test_notification_system_integration(self, mock_ui_instance):
        """Test notification system integration"""
        from ui_event_handlers import UIEventHandlers
        
        event_handlers = UIEventHandlers()
        event_handlers.register_components(mock_ui_instance.generation_components)
        
        # Test clearing notifications
        notification_html, show_btn = event_handlers.handle_clear_notification()
        assert notification_html == ""
        assert not show_btn
    
    @patch('ui_event_handlers.generate_video')
    def test_generation_workflow_integration(self, mock_generate, mock_ui_instance):
        """Test complete generation workflow integration"""
        from ui_event_handlers import UIEventHandlers
        
        event_handlers = UIEventHandlers()
        event_handlers.register_components(mock_ui_instance.generation_components)
        
        # Mock successful generation
        mock_generate.return_value = {
            'success': True,
            'video_path': '/path/to/video.mp4'
        }
        
        # Test generation
        status, video_update, notification_html, show_notification = event_handlers.handle_generate_video(
            "t2v-A14B", 
            "A beautiful sunset over the ocean",
            None,
            "1280x720",
            50,
            "",
            1.0
        )
        
        assert "completed successfully" in status.lower()
        assert show_notification
        assert "successfully" in notification_html.lower()
        
        # Verify generate_video was called with correct parameters
        mock_generate.assert_called_once()
        call_args = mock_generate.call_args
        assert call_args[1]['model_type'] == "t2v-A14B"
        assert call_args[1]['prompt'] == "A beautiful sunset over the ocean"
        assert call_args[1]['resolution'] == "1280x720"
        assert call_args[1]['steps'] == 50
    
    def test_error_handling_integration(self, mock_ui_instance):
        """Test error handling integration"""
        from ui_event_handlers import UIEventHandlers
        from ui_validation import UIValidationManager
        
        event_handlers = UIEventHandlers()
        validation_manager = UIValidationManager()
        
        # Test error display creation
        test_error = ValueError("Test error message")
        error_html, show_error = validation_manager.create_error_display_with_recovery(
            test_error, "test_context"
        )
        
        assert show_error
        assert "test error message" in error_html.lower()
        assert "error" in error_html.lower()
    
    def test_progress_indicator_integration(self, mock_ui_instance):
        """Test progress indicator integration"""
        from ui_validation import UIValidationManager
        
        validation_manager = UIValidationManager()
        
        # Test different progress stages
        stages = [
            ("validation", 0.1, "Validating inputs..."),
            ("model_loading", 0.3, "Loading model..."),
            ("generation", 0.7, "Generating video..."),
            ("complete", 1.0, "Generation complete!")
        ]
        
        for stage, progress, message in stages:
            progress_html = validation_manager.create_progress_indicator(stage, progress, message)
            
            assert stage.replace('_', ' ').title() in progress_html
            assert f"{progress * 100:.1f}%" in progress_html
            assert message in progress_html
            assert "progress-container" in progress_html
    
    def test_lora_management_integration(self, mock_ui_instance):
        """Test LoRA management integration"""
        from ui_event_handlers import UIEventHandlers
        
        event_handlers = UIEventHandlers()
        event_handlers.register_components(mock_ui_instance.generation_components)
        
        # Test adding LoRA
        status_html, controls_html, memory_html, notification_html, show_notification = event_handlers.handle_add_lora(
            "test_lora.safetensors"
        )
        
        assert show_notification
        assert "test_lora" in notification_html.lower()
        
        # Test clearing LoRAs
        status_html, controls_html, memory_html, notification_html, show_notification = event_handlers.handle_clear_all_loras()
        
        assert show_notification
        assert "cleared" in notification_html.lower()
    
    def test_comprehensive_validation_summary_integration(self, mock_ui_instance):
        """Test comprehensive validation summary integration"""
        from ui_validation import UIValidationManager, UIValidationState
        
        validation_manager = UIValidationManager()
        
        # Set up mixed validation states
        validation_manager.validation_states = {
            'prompt': UIValidationState(
                is_valid=True,
                errors=[],
                warnings=["Consider adding motion terms"],
                suggestions=["Add words like 'flowing', 'moving'"]
            ),
            'image': UIValidationState(
                is_valid=False,
                errors=["Image resolution too low"],
                warnings=[],
                suggestions=["Use higher resolution image"]
            ),
            'params': UIValidationState(
                is_valid=True,
                errors=[],
                warnings=[],
                suggestions=[]
            )
        }
        
        summary_html, all_valid = validation_manager.create_comprehensive_validation_summary()
        
        assert not all_valid  # Should be false due to image error
        assert "image resolution too low" in summary_html.lower()
        assert "consider adding motion terms" in summary_html.lower()
        assert "issues found" in summary_html.lower()

class TestUIResponsiveness:
    """Test UI responsiveness and real-time feedback"""
    
    def test_real_time_character_counting(self):
        """Test real-time character counting"""
        from ui_validation import UIValidationManager
        
        validation_manager = UIValidationManager()
        
        test_prompts = [
            "",
            "Short",
            "A medium length prompt for testing",
            "A" * 500,  # Exactly at limit
            "A" * 600   # Over limit
        ]
        
        for prompt in test_prompts:
            validation_html, is_valid, char_count = validation_manager.validate_prompt_realtime(
                prompt, "t2v-A14B"
            )
            
            expected_count = f"{len(prompt)}/500"
            assert char_count == expected_count
            
            if len(prompt) > 500:
                assert not is_valid
                assert "too long" in validation_html.lower()
    
    def test_debounced_validation(self):
        """Test debounced validation behavior"""
        from ui_event_handlers import UIEventHandlers
        import time

        event_handlers = UIEventHandlers()
        
        # Simulate rapid typing
        prompts = ["A", "A b", "A be", "A bea", "A beau", "A beautiful sunset"]
        
        for prompt in prompts:
            char_count, validation_html, show_validation = event_handlers.handle_prompt_change(
                prompt, "t2v-A14B"
            )
            
            # Character count should update immediately
            assert f"{len(prompt)}/500" in char_count
        
        # Final validation should be for complete prompt
        final_char_count, final_validation_html, final_show_validation = event_handlers.handle_prompt_change(
            "A beautiful sunset", "t2v-A14B"
        )
        
        assert "18/500" in final_char_count
    
    def test_progressive_validation_feedback(self):
        """Test progressive validation feedback"""
        from ui_validation import UIValidationManager
        
        validation_manager = UIValidationManager()
        
        # Test progression from invalid to valid
        invalid_prompt = "A" * 600
        validation_html, is_valid, char_count = validation_manager.validate_prompt_realtime(
            invalid_prompt, "t2v-A14B"
        )
        assert not is_valid
        assert "too long" in validation_html.lower()
        
        # Fix the prompt
        valid_prompt = "A beautiful sunset over the ocean"
        validation_html, is_valid, char_count = validation_manager.validate_prompt_realtime(
            valid_prompt, "t2v-A14B"
        )
        assert is_valid
        # Should not show error messages for valid prompt
        if validation_html:
            assert "error" not in validation_html.lower()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
