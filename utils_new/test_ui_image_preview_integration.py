"""
Integration test for Enhanced Image Preview in UI
Tests the integration between the preview manager and the UI components
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

def test_ui_image_preview_integration():
    """Test the integration of image preview with UI components"""
    try:
        # Mock Gradio components to avoid import issues
        with patch('gradio.HTML') as mock_html, \
             patch('gradio.Button') as mock_button, \
             patch('gradio.update') as mock_update:
            
            # Import the UI class
            from ui import Wan22UI
            
            # Create a mock config
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
                }
            }
            
            # Mock the config loading
            with patch.object(Wan22UI, '_load_config', return_value=mock_config), \
                 patch.object(Wan22UI, '_create_interface', return_value=Mock()), \
                 patch.object(Wan22UI, '_start_auto_refresh'), \
                 patch.object(Wan22UI, '_perform_startup_checks'), \
                 patch('ui.get_model_manager', return_value=Mock()), \
                 patch('ui.VRAMOptimizer', return_value=Mock()), \
                 patch('ui.get_performance_profiler', return_value=Mock()), \
                 patch('ui.start_performance_monitoring'):
                
                # Create UI instance
                ui = Wan22UI()
                
                # Check that image preview manager was initialized
                assert hasattr(ui, 'image_preview_manager')
                
        print("✅ UI image preview integration test passed")
        return True
        
    except Exception as e:
        print(f"❌ UI image preview integration test failed: {e}")
        return False

def test_enhanced_image_handlers():
    """Test the enhanced image upload handlers"""
    try:
        # Mock PIL Image
        mock_image = Mock()
        mock_image.size = (1280, 720)
        mock_image.format = "JPEG"
        mock_image.mode = "RGB"
        mock_image.filename = "test.jpg"
        
        # Mock the UI class methods
        with patch('ui.Wan22UI') as MockUI:
            mock_ui = MockUI.return_value
            
            # Mock the image preview manager
            mock_preview_manager = Mock()
            mock_preview_manager.process_image_upload.return_value = (
                "<div>Preview HTML</div>",  # preview_html
                "tooltip_data",  # tooltip_data
                True  # visible
            )
            mock_preview_manager.get_image_summary.return_value = "<div>Summary</div>"
            mock_preview_manager.get_compatibility_status.return_value = "<div>Compatible</div>"
            
            mock_ui.image_preview_manager = mock_preview_manager
            
            # Import and test the handler methods
            from ui import Wan22UI
            
            # Create a real instance for testing the methods
            with patch.object(Wan22UI, '__init__', lambda x: None):
                ui_instance = Wan22UI()
                ui_instance.image_preview_manager = mock_preview_manager
                
                # Mock the validation methods
                ui_instance._validate_start_image_upload = Mock(return_value=(
                    "<div>Validation</div>", "notification", Mock()
                ))
                ui_instance._show_notification = Mock(return_value="notification")
                
                # Test the enhanced start image handler
                result = ui_instance._handle_start_image_upload(mock_image, "i2v-A14B")
                
                # Verify the handler was called correctly
                mock_preview_manager.process_image_upload.assert_called_once_with(mock_image, "start")
                mock_preview_manager.get_image_summary.assert_called_once()
                mock_preview_manager.get_compatibility_status.assert_called_once()
                
                # Check that result has the expected structure
                assert len(result) == 6  # Should return 6 components
                
        print("✅ Enhanced image handlers test passed")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced image handlers test failed: {e}")
        return False

def test_clear_image_handlers():
    """Test the clear image functionality"""
    try:
        # Mock the UI class methods
        with patch('ui.Wan22UI') as MockUI:
            mock_ui = MockUI.return_value
            
            # Mock the image preview manager
            mock_preview_manager = Mock()
            mock_preview_manager.clear_image.return_value = (
                "<div>Empty Preview</div>",  # preview_html
                "",  # tooltip_data
                False  # visible
            )
            mock_preview_manager.get_image_summary.return_value = ""
            mock_preview_manager.get_compatibility_status.return_value = ""
            
            mock_ui.image_preview_manager = mock_preview_manager
            
            # Import and test the handler methods
            from ui import Wan22UI
            
            # Create a real instance for testing the methods
            with patch.object(Wan22UI, '__init__', lambda x: None):
                ui_instance = Wan22UI()
                ui_instance.image_preview_manager = mock_preview_manager
                
                # Test the clear start image handler
                result = ui_instance._clear_start_image()
                
                # Verify the handler was called correctly
                mock_preview_manager.clear_image.assert_called_once_with("start")
                mock_preview_manager.get_image_summary.assert_called_once()
                mock_preview_manager.get_compatibility_status.assert_called_once()
                
                # Check that result has the expected structure
                assert len(result) == 5  # Should return 5 components
                
        print("✅ Clear image handlers test passed")
        return True
        
    except Exception as e:
        print(f"❌ Clear image handlers test failed: {e}")
        return False

def test_model_type_change_integration():
    """Test that model type changes properly update image preview visibility"""
    try:
        from ui import Wan22UI
        
        # Mock the resolution manager
        mock_resolution_manager = Mock()
        mock_resolution_update = Mock()
        mock_resolution_manager.update_resolution_dropdown.return_value = mock_resolution_update
        
        with patch('ui.get_resolution_manager', return_value=mock_resolution_manager), \
             patch.object(Wan22UI, '__init__', lambda x: None):
            
            ui_instance = Wan22UI()
            ui_instance.current_model_type = "t2v-A14B"
            ui_instance._get_model_help_text = Mock(return_value="Help text")
            
            # Test model type change to I2V (should show images)
            result = ui_instance._on_model_type_change("i2v-A14B")
            
            # Should return 5 components now (including image_status_row)
            assert len(result) == 5
            
            # First two should be visibility updates (both True for I2V)
            assert result[0].visible == True  # image_inputs_row
            assert result[1].visible == True  # image_status_row
            
            # Test model type change to T2V (should hide images)
            result = ui_instance._on_model_type_change("t2v-A14B")
            
            # First two should be visibility updates (both False for T2V)
            assert result[0].visible == False  # image_inputs_row
            assert result[1].visible == False  # image_status_row
            
        print("✅ Model type change integration test passed")
        return True
        
    except Exception as e:
        print(f"❌ Model type change integration test failed: {e}")
        return False

def test_css_and_javascript_integration():
    """Test that CSS and JavaScript are properly integrated"""
    try:
        # Read the UI file to check for CSS and JavaScript
        with open('ui.py', 'r', encoding='utf-8') as f:
            ui_content = f.read()
        
        # Check for enhanced CSS classes
        css_classes = [
            "image-preview-display",
            "image-preview-container",
            "image-preview-empty",
            "image-summary-display",
            "compatibility-status-display"
        ]
        
        for css_class in css_classes:
            assert css_class in ui_content, f"Missing CSS class: {css_class}"
        
        # Check for JavaScript functions
        js_functions = [
            "clearImage",
            "showLargePreview",
            "showTooltip",
            "hideTooltip"
        ]
        
        for js_function in js_functions:
            assert js_function in ui_content, f"Missing JavaScript function: {js_function}"
        
        # Check for responsive design
        assert "@media (max-width: 768px)" in ui_content
        assert "Mobile responsive image previews" in ui_content
        
        print("✅ CSS and JavaScript integration test passed")
        return True
        
    except Exception as e:
        print(f"❌ CSS and JavaScript integration test failed: {e}")
        return False

def test_component_registration():
    """Test that all new components are properly registered"""
    try:
        # Read the UI file to check component registration
        with open('ui.py', 'r', encoding='utf-8') as f:
            ui_content = f.read()
        
        # Check that new components are in the generation_components dictionary
        new_components = [
            "'start_image_preview'",
            "'end_image_preview'",
            "'image_summary'",
            "'compatibility_status'",
            "'clear_start_btn'",
            "'clear_end_btn'",
            "'image_status_row'"
        ]
        
        for component in new_components:
            assert component in ui_content, f"Missing component registration: {component}"
        
        # Check that event handlers are properly connected
        event_handlers = [
            "_handle_start_image_upload",
            "_handle_end_image_upload",
            "_clear_start_image",
            "_clear_end_image"
        ]
        
        for handler in event_handlers:
            assert handler in ui_content, f"Missing event handler: {handler}"
        
        print("✅ Component registration test passed")
        return True
        
    except Exception as e:
        print(f"❌ Component registration test failed: {e}")
        return False

def run_integration_tests():
    """Run all integration tests"""
    print("🧪 Running Enhanced Image Preview UI Integration Tests")
    print("=" * 70)
    
    tests = [
        test_ui_image_preview_integration,
        test_enhanced_image_handlers,
        test_clear_image_handlers,
        test_model_type_change_integration,
        test_css_and_javascript_integration,
        test_component_registration
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            failed += 1
        print()
    
    print("=" * 70)
    print(f"📊 Integration Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All integration tests passed!")
        return True
    else:
        print(f"⚠️ {failed} tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = run_integration_tests()
    exit(0 if success else 1)