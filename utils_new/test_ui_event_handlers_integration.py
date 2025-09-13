"""
Test UI Event Handlers Integration
Tests the comprehensive event handler system for proper integration
"""

import pytest
import gradio as gr
import logging
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_enhanced_event_handlers_import():
    """Test that enhanced event handlers can be imported"""
    try:
        from ui_event_handlers_enhanced import EnhancedUIEventHandlers, get_enhanced_event_handlers
        assert EnhancedUIEventHandlers is not None
        assert get_enhanced_event_handlers is not None
        logger.info("✅ Enhanced event handlers import successful")
    except ImportError as e:
        pytest.fail(f"Failed to import enhanced event handlers: {e}")

def test_event_handlers_integration_import():
    """Test that event handlers integration module can be imported"""
    try:
        from ui_event_handlers import get_event_handlers, setup_event_handlers
        assert get_event_handlers is not None
        assert setup_event_handlers is not None
        logger.info("✅ Event handlers integration import successful")
    except ImportError as e:
        pytest.fail(f"Failed to import event handlers integration: {e}")

def test_enhanced_event_handlers_initialization():
    """Test enhanced event handlers initialization"""
    try:
        from ui_event_handlers_enhanced import EnhancedUIEventHandlers
        
        config = {
            "test": True,
            "validation": {"enabled": True}
        }
        
        # Mock the required managers
        with patch('ui_event_handlers_enhanced.get_validation_manager') as mock_validation, \
             patch('ui_event_handlers_enhanced.get_image_validator') as mock_image_validator, \
             patch('ui_event_handlers_enhanced.get_preview_manager') as mock_preview, \
             patch('ui_event_handlers_enhanced.get_resolution_manager') as mock_resolution, \
             patch('ui_event_handlers_enhanced.get_progress_tracker') as mock_progress, \
             patch('ui_event_handlers_enhanced.get_help_system') as mock_help:
            
            # Set up mocks
            mock_validation.return_value = Mock()
            mock_image_validator.return_value = Mock()
            mock_preview.return_value = Mock()
            mock_resolution.return_value = Mock()
            mock_progress_tracker = Mock()
            mock_progress_tracker.add_update_callback = Mock()
            mock_progress.return_value = mock_progress_tracker
            mock_help.return_value = Mock()
            
            handlers = EnhancedUIEventHandlers(config)
            
            assert handlers is not None
            assert handlers.config == config
            assert handlers.ui_components == {}
            assert handlers.generation_in_progress == False
            assert handlers.registered_handlers == []
            
            logger.info("✅ Enhanced event handlers initialization successful")
            
    except Exception as e:
        pytest.fail(f"Enhanced event handlers initialization failed: {e}")

def test_component_registration():
    """Test UI component registration"""
    try:
        from ui_event_handlers_enhanced import EnhancedUIEventHandlers
        
        # Mock the required managers
        with patch('ui_event_handlers_enhanced.get_validation_manager') as mock_validation, \
             patch('ui_event_handlers_enhanced.get_image_validator') as mock_image_validator, \
             patch('ui_event_handlers_enhanced.get_preview_manager') as mock_preview, \
             patch('ui_event_handlers_enhanced.get_resolution_manager') as mock_resolution, \
             patch('ui_event_handlers_enhanced.get_progress_tracker') as mock_progress, \
             patch('ui_event_handlers_enhanced.get_help_system') as mock_help:
            
            # Set up mocks
            mock_validation_manager = Mock()
            mock_validation_manager.register_ui_components = Mock()
            mock_validation.return_value = mock_validation_manager
            
            mock_image_validator.return_value = Mock()
            
            mock_preview_manager = Mock()
            mock_preview_manager.register_ui_components = Mock()
            mock_preview.return_value = mock_preview_manager
            
            mock_resolution.return_value = Mock()
            
            mock_progress_tracker = Mock()
            mock_progress_tracker.add_update_callback = Mock()
            mock_progress.return_value = mock_progress_tracker
            
            mock_help.return_value = Mock()
            
            handlers = EnhancedUIEventHandlers()
            
            # Create mock UI components
            mock_components = {
                'model_type': Mock(),
                'prompt_input': Mock(),
                'image_input': Mock(),
                'generate_btn': Mock(),
                'notification_area': Mock(),
                'clear_notification_btn': Mock()
            }
            
            # Register components
            handlers.register_components(mock_components)
            
            assert handlers.ui_components == mock_components
            assert len(handlers.ui_components) == 6
            
            # Verify that managers were called to register components
            mock_validation_manager.register_ui_components.assert_called_once_with(mock_components)
            mock_preview_manager.register_ui_components.assert_called_once_with(mock_components)
            
            logger.info("✅ Component registration successful")
            
    except Exception as e:
        pytest.fail(f"Component registration failed: {e}")

def test_model_type_change_handler():
    """Test model type change event handler"""
    try:
        from ui_event_handlers_enhanced import EnhancedUIEventHandlers
        from input_validation import ValidationResult
        
        # Mock the required managers
        with patch('ui_event_handlers_enhanced.get_validation_manager') as mock_validation, \
             patch('ui_event_handlers_enhanced.get_image_validator') as mock_image_validator, \
             patch('ui_event_handlers_enhanced.get_preview_manager') as mock_preview, \
             patch('ui_event_handlers_enhanced.get_resolution_manager') as mock_resolution, \
             patch('ui_event_handlers_enhanced.get_progress_tracker') as mock_progress, \
             patch('ui_event_handlers_enhanced.get_help_system') as mock_help:
            
            # Set up mocks
            mock_validation.return_value = Mock()
            mock_image_validator.return_value = Mock()
            mock_preview.return_value = Mock()
            
            mock_resolution_manager = Mock()
            mock_resolution_manager.update_resolution_dropdown.return_value = gr.update(choices=["1280x720"])
            mock_resolution.return_value = mock_resolution_manager
            
            mock_progress_tracker = Mock()
            mock_progress_tracker.add_update_callback = Mock()
            mock_progress.return_value = mock_progress_tracker
            
            mock_help_system = Mock()
            mock_help_system.get_image_help_text.return_value = "Test image help"
            mock_help_system.get_model_help_text.return_value = "Test model help"
            mock_help_system.get_image_requirements_text.return_value = "Test requirements"
            mock_help.return_value = mock_help_system
            
            handlers = EnhancedUIEventHandlers()
            
            # Test model type change
            result = handlers.handle_model_type_change("i2v-A14B")
            
            assert len(result) == 11  # Should return 11 values
            assert result[6] == True  # Notification should be visible
            
            # Test that help system methods were called
            mock_help_system.get_image_help_text.assert_called_with("i2v-A14B")
            mock_help_system.get_model_help_text.assert_called_with("i2v-A14B")
            mock_resolution_manager.update_resolution_dropdown.assert_called_with("i2v-A14B")
            
            logger.info("✅ Model type change handler test successful")
            
    except Exception as e:
        pytest.fail(f"Model type change handler test failed: {e}")

def test_image_upload_handler():
    """Test image upload event handler"""
    try:
        from ui_event_handlers_enhanced import EnhancedUIEventHandlers
        from input_validation import ValidationResult
        from PIL import Image
        import numpy as np

        # Mock the required managers
        with patch('ui_event_handlers_enhanced.get_validation_manager') as mock_validation, \
             patch('ui_event_handlers_enhanced.get_image_validator') as mock_image_validator, \
             patch('ui_event_handlers_enhanced.get_preview_manager') as mock_preview, \
             patch('ui_event_handlers_enhanced.get_resolution_manager') as mock_resolution, \
             patch('ui_event_handlers_enhanced.get_progress_tracker') as mock_progress, \
             patch('ui_event_handlers_enhanced.get_help_system') as mock_help:
            
            # Set up mocks
            mock_validation.return_value = Mock()
            
            mock_image_validator_instance = Mock()
            # Create a mock validation result
            mock_validation_result = type('ValidationResult', (), {
                'is_valid': True,
                'message': "Image is valid",
                'details': {"width": 512, "height": 512}
            })()
            mock_image_validator_instance.validate_image.return_value = mock_validation_result
            mock_image_validator.return_value = mock_image_validator_instance
            
            mock_preview_manager = Mock()
            mock_preview_manager.create_image_preview.return_value = "<div>Preview HTML</div>"
            mock_preview.return_value = mock_preview_manager
            
            mock_resolution.return_value = Mock()
            
            mock_progress_tracker = Mock()
            mock_progress_tracker.add_update_callback = Mock()
            mock_progress.return_value = mock_progress_tracker
            
            mock_help.return_value = Mock()
            
            handlers = EnhancedUIEventHandlers()
            
            # Create a mock image
            mock_image = Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8))
            
            # Test image upload
            result = handlers.handle_start_image_upload(mock_image, "i2v-A14B")
            
            assert len(result) == 9  # Should return 9 values
            assert result[2] == True  # Clear button should be visible
            assert result[4] == True  # Notification should be visible
            
            # Verify that validator was called
            mock_image_validator_instance.validate_image.assert_called_with(mock_image, "start", "i2v-A14B")
            mock_preview_manager.create_image_preview.assert_called_with(mock_image, "start", mock_validation_result)
            
            logger.info("✅ Image upload handler test successful")
            
    except Exception as e:
        pytest.fail(f"Image upload handler test failed: {e}")

def test_generation_handler():
    """Test generation event handler"""
    try:
        from ui_event_handlers_enhanced import EnhancedUIEventHandlers
        
        # Mock the required managers
        with patch('ui_event_handlers_enhanced.get_validation_manager') as mock_validation, \
             patch('ui_event_handlers_enhanced.get_image_validator') as mock_image_validator, \
             patch('ui_event_handlers_enhanced.get_preview_manager') as mock_preview, \
             patch('ui_event_handlers_enhanced.get_resolution_manager') as mock_resolution, \
             patch('ui_event_handlers_enhanced.get_progress_tracker') as mock_progress, \
             patch('ui_event_handlers_enhanced.get_help_system') as mock_help:
            
            # Set up mocks
            mock_validation.return_value = Mock()
            mock_image_validator.return_value = Mock()
            mock_preview.return_value = Mock()
            mock_resolution.return_value = Mock()
            
            mock_progress_tracker = Mock()
            mock_progress_tracker.add_update_callback = Mock()
            mock_progress_tracker.start_progress_tracking = Mock()
            mock_progress_tracker.get_progress_html = Mock(return_value="<div>Progress</div>")
            mock_progress.return_value = mock_progress_tracker
            
            mock_help.return_value = Mock()
            
            handlers = EnhancedUIEventHandlers()
            
            # Test generation with valid parameters
            result = handlers.handle_generate_video(
                model_type="t2v-A14B",
                prompt="Test prompt",
                start_image=None,
                end_image=None,
                resolution="1280x720",
                steps=50,
                lora_path="",
                lora_strength=1.0
            )
            
            assert len(result) == 8  # Should return 8 values
            assert "Generation Started" in result[0]
            assert result[4] == True  # Notification should be visible
            
            # Verify that progress tracking was started
            mock_progress_tracker.start_progress_tracking.assert_called_once()
            
            logger.info("✅ Generation handler test successful")
            
    except Exception as e:
        pytest.fail(f"Generation handler test failed: {e}")

def test_event_handler_cleanup():
    """Test event handler cleanup functionality"""
    try:
        from ui_event_handlers_enhanced import EnhancedUIEventHandlers
        
        # Mock the required managers
        with patch('ui_event_handlers_enhanced.get_validation_manager') as mock_validation, \
             patch('ui_event_handlers_enhanced.get_image_validator') as mock_image_validator, \
             patch('ui_event_handlers_enhanced.get_preview_manager') as mock_preview, \
             patch('ui_event_handlers_enhanced.get_resolution_manager') as mock_resolution, \
             patch('ui_event_handlers_enhanced.get_progress_tracker') as mock_progress, \
             patch('ui_event_handlers_enhanced.get_help_system') as mock_help:
            
            # Set up mocks
            mock_validation.return_value = Mock()
            mock_image_validator.return_value = Mock()
            mock_preview.return_value = Mock()
            mock_resolution.return_value = Mock()
            
            mock_progress_tracker = Mock()
            mock_progress_tracker.add_update_callback = Mock()
            mock_progress.return_value = mock_progress_tracker
            
            mock_help.return_value = Mock()
            
            handlers = EnhancedUIEventHandlers()
            
            # Add some mock handlers
            mock_component = Mock()
            mock_handler = Mock()
            handlers._register_handler(mock_component, "click", mock_handler)
            
            assert len(handlers.registered_handlers) == 1
            
            # Test cleanup
            handlers.cleanup_handlers()
            
            assert len(handlers.registered_handlers) == 0
            
            logger.info("✅ Event handler cleanup test successful")
            
    except Exception as e:
        pytest.fail(f"Event handler cleanup test failed: {e}")

def test_integration_with_main_ui():
    """Test integration with main UI module"""
    try:
        from ui_event_handlers import get_event_handlers, setup_event_handlers
        
        config = {"test": True}
        
        # Mock UI components
        mock_components = {
            'model_type': Mock(),
            'prompt_input': Mock(),
            'image_input': Mock(),
            'generate_btn': Mock(),
            'notification_area': Mock(),
            'clear_notification_btn': Mock()
        }
        
        # Test getting event handlers
        handlers = get_event_handlers(config)
        
        if handlers:
            # Test setup
            result = setup_event_handlers(mock_components, config)
            assert result is not None
            logger.info("✅ Integration with main UI successful")
        else:
            logger.warning("⚠️ Enhanced event handlers not available - this is expected in some environments")
            
    except Exception as e:
        pytest.fail(f"Integration with main UI failed: {e}")

if __name__ == "__main__":
    """Run all tests"""
    print("🧪 Running UI Event Handlers Integration Tests")
    print("=" * 60)
    
    tests = [
        test_enhanced_event_handlers_import,
        test_event_handlers_integration_import,
        test_enhanced_event_handlers_initialization,
        test_component_registration,
        test_model_type_change_handler,
        test_image_upload_handler,
        test_generation_handler,
        test_event_handler_cleanup,
        test_integration_with_main_ui
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            print(f"\n🔍 Running {test.__name__}...")
            test()
            print(f"✅ {test.__name__} PASSED")
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} FAILED: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Event handler integration is working correctly.")
    else:
        print(f"⚠️ {failed} tests failed. Please review the implementation.")
