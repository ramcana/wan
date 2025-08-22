"""
Test Enhanced UI Event Handlers
Tests for the comprehensive event handling system
"""

import pytest
import gradio as gr
import threading
import time
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import numpy as np

from ui_event_handlers_enhanced import EnhancedUIEventHandlers, get_enhanced_event_handlers
from input_validation import ValidationResult

class TestEnhancedUIEventHandlers:
    """Test enhanced UI event handlers"""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration"""
        return {
            "progress_update_interval": 1.0,
            "enable_system_monitoring": True,
            "validation": {
                "max_prompt_length": 500,
                "min_image_size": (256, 256)
            }
        }
    
    @pytest.fixture
    def mock_components(self):
        """Mock UI components"""
        return {
            'model_type': Mock(),
            'prompt_input': Mock(),
            'image_input': Mock(),
            'end_image_input': Mock(),
            'resolution': Mock(),
            'steps': Mock(),
            'lora_path': Mock(),
            'lora_strength': Mock(),
            'generate_btn': Mock(),
            'queue_btn': Mock(),
            'enhance_btn': Mock(),
            'clear_start_btn': Mock(),
            'clear_end_btn': Mock(),
            'clear_notification_btn': Mock(),
            'image_inputs_row': Mock(),
            'image_help_text': Mock(),
            'model_help_text': Mock(),
            'lora_compatibility_display': Mock(),
            'start_image_preview': Mock(),
            'end_image_preview': Mock(),
            'start_image_validation': Mock(),
            'end_image_validation': Mock(),
            'image_compatibility_status': Mock(),
            'char_count': Mock(),
            'prompt_validation': Mock(),
            'parameter_validation': Mock(),
            'generation_status': Mock(),
            'progress_display': Mock(),
            'output_video': Mock(),
            'notification_area': Mock(),
            'enhanced_prompt_display': Mock()
        }
    
    @pytest.fixture
    def mock_image(self):
        """Create a mock PIL Image"""
        # Create a simple test image
        img_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        return Image.fromarray(img_array)
    
    @pytest.fixture
    def event_handlers(self, mock_config):
        """Create event handlers instance"""
        with patch('ui_event_handlers_enhanced.get_validation_manager'), \
             patch('ui_event_handlers_enhanced.get_image_validator'), \
             patch('ui_event_handlers_enhanced.get_preview_manager'), \
             patch('ui_event_handlers_enhanced.get_resolution_manager'), \
             patch('ui_event_handlers_enhanced.get_progress_tracker'), \
             patch('ui_event_handlers_enhanced.get_help_system'):
            
            handlers = EnhancedUIEventHandlers(mock_config)
            return handlers
    
    def test_initialization(self, event_handlers, mock_config):
        """Test event handlers initialization"""
        assert event_handlers.config == mock_config
        assert event_handlers.validation_delay == 0.5
        assert not event_handlers.generation_in_progress
        assert event_handlers.current_task_id is None
        assert len(event_handlers.ui_components) == 0
    
    def test_register_components(self, event_handlers, mock_components):
        """Test component registration"""
        event_handlers.register_components(mock_components)
        
        assert event_handlers.ui_components == mock_components
        assert len(event_handlers.ui_components) == len(mock_components)
    
    def test_setup_all_event_handlers(self, event_handlers, mock_components):
        """Test setting up all event handlers"""
        event_handlers.register_components(mock_components)
        
        # Mock the setup methods to avoid actual Gradio calls
        with patch.object(event_handlers, 'setup_model_type_events'), \
             patch.object(event_handlers, 'setup_image_upload_events'), \
             patch.object(event_handlers, 'setup_validation_events'), \
             patch.object(event_handlers, 'setup_generation_events'), \
             patch.object(event_handlers, 'setup_progress_events'), \
             patch.object(event_handlers, 'setup_utility_events'):
            
            event_handlers.setup_all_event_handlers()
            
            # Verify all setup methods were called
            event_handlers.setup_model_type_events.assert_called_once()
            event_handlers.setup_image_upload_events.assert_called_once()
            event_handlers.setup_validation_events.assert_called_once()
            event_handlers.setup_generation_events.assert_called_once()
            event_handlers.setup_progress_events.assert_called_once()
            event_handlers.setup_utility_events.assert_called_once()
    
    def test_handle_model_type_change_t2v(self, event_handlers):
        """Test model type change to T2V"""
        with patch.object(event_handlers.help_text_system, 'get_image_help_text') as mock_image_help, \
             patch.object(event_handlers.help_text_system, 'get_model_help_text') as mock_model_help, \
             patch.object(event_handlers.resolution_manager, 'update_resolution_dropdown') as mock_resolution:
            
            mock_image_help.return_value = ""
            mock_model_help.return_value = "T2V help text"
            mock_resolution.return_value = gr.update(choices=["1280x720"])
            
            result = event_handlers.handle_model_type_change("t2v-A14B")
            
            # Should hide image inputs for T2V
            assert len(result) == 7
            # Verify image inputs are hidden
            image_inputs_update = result[0]
            assert hasattr(image_inputs_update, 'visible') or 'visible' in str(image_inputs_update)
    
    def test_handle_model_type_change_i2v(self, event_handlers):
        """Test model type change to I2V"""
        with patch.object(event_handlers.help_text_system, 'get_image_help_text') as mock_image_help, \
             patch.object(event_handlers.help_text_system, 'get_model_help_text') as mock_model_help, \
             patch.object(event_handlers.resolution_manager, 'update_resolution_dropdown') as mock_resolution:
            
            mock_image_help.return_value = "I2V image help"
            mock_model_help.return_value = "I2V help text"
            mock_resolution.return_value = gr.update(choices=["1280x720"])
            
            result = event_handlers.handle_model_type_change("i2v-A14B")
            
            # Should show image inputs for I2V
            assert len(result) == 7
            # Verify success notification contains model type
            notification_html = result[5]
            assert "i2v-A14B" in notification_html
            assert "Image inputs are now visible" in notification_html
    
    def test_handle_start_image_upload_valid(self, event_handlers, mock_image):
        """Test valid start image upload"""
        # Mock validation result
        mock_validation_result = Mock()
        mock_validation_result.is_valid = True
        mock_validation_result.message = "Valid image"
        
        with patch.object(event_handlers.image_validator, 'validate_image', return_value=mock_validation_result), \
             patch.object(event_handlers.preview_manager, 'create_image_preview', return_value="<preview>"):
            
            result = event_handlers.handle_start_image_upload(mock_image, "i2v-A14B")
            
            assert len(result) == 5
            preview_html, validation_html, clear_btn_visible, notification_html, show_notification = result
            
            assert preview_html == "<preview>"
            assert clear_btn_visible is True
            assert "successfully" in notification_html
            assert show_notification is True
    
    def test_handle_start_image_upload_invalid(self, event_handlers, mock_image):
        """Test invalid start image upload"""
        # Mock validation result
        mock_validation_result = Mock()
        mock_validation_result.is_valid = False
        mock_validation_result.message = "Invalid image format"
        
        with patch.object(event_handlers.image_validator, 'validate_image', return_value=mock_validation_result), \
             patch.object(event_handlers.preview_manager, 'create_image_preview', return_value="<preview>"):
            
            result = event_handlers.handle_start_image_upload(mock_image, "i2v-A14B")
            
            assert len(result) == 5
            preview_html, validation_html, clear_btn_visible, notification_html, show_notification = result
            
            assert clear_btn_visible is False
            assert "validation failed" in notification_html
            assert "Invalid image format" in notification_html
    
    def test_handle_end_image_upload_with_compatibility(self, event_handlers, mock_image):
        """Test end image upload with compatibility check"""
        # Mock validation results
        mock_validation_result = Mock()
        mock_validation_result.is_valid = True
        mock_validation_result.message = "Valid image"
        
        mock_compatibility_result = Mock()
        mock_compatibility_result.is_valid = True
        mock_compatibility_result.message = "Compatible images"
        
        with patch.object(event_handlers.image_validator, 'validate_image', return_value=mock_validation_result), \
             patch.object(event_handlers.image_validator, 'validate_image_compatibility', return_value=mock_compatibility_result), \
             patch.object(event_handlers.preview_manager, 'create_image_preview', return_value="<preview>"):
            
            result = event_handlers.handle_end_image_upload(mock_image, "i2v-A14B", mock_image)
            
            assert len(result) == 6
            preview_html, validation_html, clear_btn_visible, compatibility_html, notification_html, show_notification = result
            
            assert preview_html == "<preview>"
            assert clear_btn_visible is True
            assert "compatible" in compatibility_html.lower()
            assert "successfully" in notification_html
    
    def test_handle_clear_start_image(self, event_handlers):
        """Test clearing start image"""
        result = event_handlers.handle_clear_start_image()
        
        assert len(result) == 7
        image_update, preview_html, validation_html, clear_btn_visible, compatibility_html, notification_html, show_notification = result
        
        assert preview_html == ""
        assert validation_html == ""
        assert clear_btn_visible is False
        assert compatibility_html == ""
        assert "cleared" in notification_html
        assert show_notification is True
    
    def test_handle_prompt_change_valid(self, event_handlers):
        """Test prompt change with valid prompt"""
        mock_validation_result = Mock()
        mock_validation_result.is_valid = True
        mock_validation_result.message = "Valid prompt"
        
        with patch.object(event_handlers.validation_manager.prompt_validator, 'validate_prompt', return_value=mock_validation_result):
            
            result = event_handlers.handle_prompt_change("A beautiful sunset", "t2v-A14B")
            
            assert len(result) == 4
            char_count, validation_html, notification_html, show_notification = result
            
            assert char_count == "17/500"
            assert notification_html == ""
            assert show_notification is False
    
    def test_handle_prompt_change_invalid(self, event_handlers):
        """Test prompt change with invalid prompt"""
        mock_validation_result = Mock()
        mock_validation_result.is_valid = False
        mock_validation_result.message = "Prompt too short"
        
        with patch.object(event_handlers.validation_manager.prompt_validator, 'validate_prompt', return_value=mock_validation_result):
            
            result = event_handlers.handle_prompt_change("Hi", "t2v-A14B")
            
            assert len(result) == 4
            char_count, validation_html, notification_html, show_notification = result
            
            assert char_count == "2/500"
            assert "Prompt too short" in notification_html
            assert show_notification is True
    
    def test_handle_generate_video_validation_failure(self, event_handlers):
        """Test video generation with validation failure"""
        with patch.object(event_handlers, '_validate_generation_request', return_value=(False, "Validation failed")):
            
            result = event_handlers.handle_generate_video(
                "t2v-A14B", "test prompt", None, None, "1280x720", 50, "", 1.0
            )
            
            assert len(result) == 5
            status, progress_html, video_update, notification_html, show_notification = result
            
            assert status == "❌ Validation Failed"
            assert progress_html == ""
            assert "Validation failed" in notification_html
            assert show_notification is True
    
    def test_handle_generate_video_already_in_progress(self, event_handlers):
        """Test video generation when already in progress"""
        event_handlers.generation_in_progress = True
        
        result = event_handlers.handle_generate_video(
            "t2v-A14B", "test prompt", None, None, "1280x720", 50, "", 1.0
        )
        
        assert len(result) == 5
        status, progress_html, video_update, notification_html, show_notification = result
        
        assert "already in progress" in status
        assert "already running" in notification_html
        assert show_notification is True
    
    def test_handle_queue_generation_success(self, event_handlers):
        """Test successful queue generation"""
        with patch.object(event_handlers, '_validate_generation_request', return_value=(True, "Validation passed")):
            
            result = event_handlers.handle_queue_generation(
                "t2v-A14B", "test prompt", None, None, "1280x720", 50, "", 1.0
            )
            
            assert len(result) == 2
            notification_html, show_notification = result
            
            assert "added to queue" in notification_html
            assert show_notification is True
    
    def test_handle_prompt_enhancement_success(self, event_handlers):
        """Test successful prompt enhancement"""
        with patch('ui_event_handlers_enhanced.enhance_prompt', return_value="Enhanced test prompt"):
            
            result = event_handlers.handle_prompt_enhancement("test prompt", "t2v-A14B")
            
            assert len(result) == 3
            enhanced_update, notification_html, show_notification = result
            
            assert "enhanced successfully" in notification_html
            assert show_notification is True
    
    def test_handle_prompt_enhancement_no_change(self, event_handlers):
        """Test prompt enhancement with no change"""
        with patch('ui_event_handlers_enhanced.enhance_prompt', return_value="test prompt"):
            
            result = event_handlers.handle_prompt_enhancement("test prompt", "t2v-A14B")
            
            assert len(result) == 3
            enhanced_update, notification_html, show_notification = result
            
            assert "already well-optimized" in notification_html
            assert show_notification is True
    
    def test_validate_generation_request_success(self, event_handlers):
        """Test successful generation request validation"""
        # Mock all validators to return valid results
        mock_prompt_result = Mock()
        mock_prompt_result.is_valid = True
        
        mock_param_result = Mock()
        mock_param_result.is_valid = True
        
        with patch.object(event_handlers.validation_manager.prompt_validator, 'validate_prompt', return_value=mock_prompt_result), \
             patch.object(event_handlers.validation_manager.config_validator, 'validate_generation_params', return_value=mock_param_result):
            
            success, message = event_handlers._validate_generation_request(
                "t2v-A14B", "test prompt", None, None, "1280x720", 50, "", 1.0
            )
            
            assert success is True
            assert message == "Validation passed"
    
    def test_validate_generation_request_prompt_failure(self, event_handlers):
        """Test generation request validation with prompt failure"""
        mock_prompt_result = Mock()
        mock_prompt_result.is_valid = False
        mock_prompt_result.message = "Invalid prompt"
        
        with patch.object(event_handlers.validation_manager.prompt_validator, 'validate_prompt', return_value=mock_prompt_result):
            
            success, message = event_handlers._validate_generation_request(
                "t2v-A14B", "test prompt", None, None, "1280x720", 50, "", 1.0
            )
            
            assert success is False
            assert "Invalid prompt" in message
    
    def test_validate_generation_request_missing_image(self, event_handlers):
        """Test generation request validation with missing required image"""
        success, message = event_handlers._validate_generation_request(
            "i2v-A14B", "test prompt", None, None, "1280x720", 50, "", 1.0
        )
        
        assert success is False
        assert "Start image is required" in message
    
    def test_create_validation_display_valid(self, event_handlers):
        """Test creating validation display for valid result"""
        mock_result = Mock()
        mock_result.is_valid = True
        
        html = event_handlers._create_validation_display(mock_result, "test")
        
        assert "✅" in html
        assert "validation passed" in html
        assert "#28a745" in html  # Success color
    
    def test_create_validation_display_invalid(self, event_handlers):
        """Test creating validation display for invalid result"""
        mock_result = Mock()
        mock_result.is_valid = False
        mock_result.message = "Test error"
        
        html = event_handlers._create_validation_display(mock_result, "test")
        
        assert "❌" in html
        assert "Test error" in html
        assert "#dc3545" in html  # Error color
    
    def test_create_error_display(self, event_handlers):
        """Test creating error display"""
        error = Exception("Test error message")
        
        html = event_handlers._create_error_display(error, "test_context")
        
        assert "❌" in html
        assert "Test error message" in html
        assert "test context" in html
        assert "#dc3545" in html  # Error color
    
    def test_get_lora_compatibility_display(self, event_handlers):
        """Test getting LoRA compatibility display"""
        html = event_handlers._get_lora_compatibility_display("t2v-A14B")
        
        assert "LoRA Compatibility" in html
        assert "t2v-A14B" in html
        assert "style" in html
        assert "character" in html
        assert "concept" in html
    
    def test_get_enhanced_event_handlers_singleton(self, mock_config):
        """Test that get_enhanced_event_handlers returns singleton"""
        with patch('ui_event_handlers_enhanced.get_validation_manager'), \
             patch('ui_event_handlers_enhanced.get_image_validator'), \
             patch('ui_event_handlers_enhanced.get_preview_manager'), \
             patch('ui_event_handlers_enhanced.get_resolution_manager'), \
             patch('ui_event_handlers_enhanced.get_progress_tracker'), \
             patch('ui_event_handlers_enhanced.get_help_system'):
            
            handlers1 = get_enhanced_event_handlers(mock_config)
            handlers2 = get_enhanced_event_handlers()
            
            assert handlers1 is handlers2

class TestEventHandlerIntegration:
    """Integration tests for event handlers"""
    
    def test_full_workflow_simulation(self):
        """Test a complete workflow simulation"""
        with patch('ui_event_handlers_enhanced.get_validation_manager'), \
             patch('ui_event_handlers_enhanced.get_image_validator'), \
             patch('ui_event_handlers_enhanced.get_preview_manager'), \
             patch('ui_event_handlers_enhanced.get_resolution_manager'), \
             patch('ui_event_handlers_enhanced.get_progress_tracker'), \
             patch('ui_event_handlers_enhanced.get_help_system'):
            
            handlers = EnhancedUIEventHandlers()
            
            # Simulate model type change
            with patch.object(handlers.help_text_system, 'get_image_help_text', return_value="Help"), \
                 patch.object(handlers.help_text_system, 'get_model_help_text', return_value="Model help"), \
                 patch.object(handlers.resolution_manager, 'update_resolution_dropdown', return_value=gr.update()):
                
                result = handlers.handle_model_type_change("i2v-A14B")
                assert len(result) == 7
            
            # Simulate prompt change
            mock_validation_result = Mock()
            mock_validation_result.is_valid = True
            
            with patch.object(handlers.validation_manager.prompt_validator, 'validate_prompt', return_value=mock_validation_result):
                result = handlers.handle_prompt_change("Test prompt", "i2v-A14B")
                assert len(result) == 4
            
            # Simulate image upload
            mock_image = Mock()
            mock_validation_result = Mock()
            mock_validation_result.is_valid = True
            mock_validation_result.message = "Valid"
            
            with patch.object(handlers.image_validator, 'validate_image', return_value=mock_validation_result), \
                 patch.object(handlers.preview_manager, 'create_image_preview', return_value="<preview>"):
                
                result = handlers.handle_start_image_upload(mock_image, "i2v-A14B")
                assert len(result) == 5
                assert result[2] is True  # clear button visible
    
    def test_error_handling_robustness(self):
        """Test error handling robustness"""
        with patch('ui_event_handlers_enhanced.get_validation_manager'), \
             patch('ui_event_handlers_enhanced.get_image_validator'), \
             patch('ui_event_handlers_enhanced.get_preview_manager'), \
             patch('ui_event_handlers_enhanced.get_resolution_manager'), \
             patch('ui_event_handlers_enhanced.get_progress_tracker'), \
             patch('ui_event_handlers_enhanced.get_help_system'):
            
            handlers = EnhancedUIEventHandlers()
            
            # Test error in model type change
            with patch.object(handlers.help_text_system, 'get_image_help_text', side_effect=Exception("Test error")):
                result = handlers.handle_model_type_change("i2v-A14B")
                assert len(result) == 7
                assert "Error updating model type" in result[3]
            
            # Test error in image upload
            with patch.object(handlers.image_validator, 'validate_image', side_effect=Exception("Validation error")):
                result = handlers.handle_start_image_upload(Mock(), "i2v-A14B")
                assert len(result) == 5
                assert result[4] is True  # show notification
                assert "Error" in result[3]

if __name__ == "__main__":
    pytest.main([__file__, "-v"])