"""
Test UI integration with resolution manager
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestUIResolutionIntegration(unittest.TestCase):
    """Test UI integration with resolution manager"""
    
    def test_resolution_manager_import_in_ui(self):
        """Test that resolution manager can be imported in UI context"""
        try:
            from resolution_manager import get_resolution_manager
            manager = get_resolution_manager()
            self.assertIsNotNone(manager)
        except ImportError as e:
            self.fail(f"Failed to import resolution manager: {e}")
    
    @patch('gradio.update')
    def test_ui_model_type_change_handler_integration(self, mock_gr_update):
        """Test that UI model type change handler works with resolution manager"""
        mock_gr_update.return_value = MagicMock()
        
        # Mock the UI class structure
        class MockUI:
            def __init__(self):
                self.current_model_type = "t2v-A14B"
            
            def _get_model_help_text(self, model_type):
                return f"Help text for {model_type}"
            
            def _on_model_type_change(self, model_type: str):
                """Simulate the updated UI method"""
                try:
                    from resolution_manager import get_resolution_manager
                    resolution_manager = get_resolution_manager()
                    
                    show_images = model_type in ["i2v-A14B", "ti2v-5B"]
                    resolution_update = resolution_manager.update_resolution_dropdown(model_type)
                    
                    image_help = "Test image help" if show_images else ""
                    self.current_model_type = model_type
                    
                    return (
                        {"visible": show_images},  # image_inputs_row visibility
                        {"value": image_help, "visible": show_images},  # image_help_text
                        resolution_update,  # resolution dropdown
                        self._get_model_help_text(model_type)  # model help text
                    )
                except Exception as e:
                    return None, None, None, f"Error: {e}"
        
        # Test the integration
        ui = MockUI()
        
        # Test t2v-A14B
        result = ui._on_model_type_change("t2v-A14B")
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 4)
        self.assertFalse(result[0]["visible"])  # Images should be hidden
        
        # Test ti2v-5B
        result = ui._on_model_type_change("ti2v-5B")
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 4)
        self.assertTrue(result[0]["visible"])  # Images should be visible
        
        # Verify gr.update was called
        self.assertTrue(mock_gr_update.called)
    
    def test_ui_event_handlers_integration(self):
        """Test that UI event handlers can use resolution manager"""
        try:
            # Test importing in ui_event_handlers context
            from resolution_manager import get_resolution_manager
            
            # Simulate the event handler update
            resolution_manager = get_resolution_manager()
            
            # Test each model type
            for model_type in ["t2v-A14B", "i2v-A14B", "ti2v-5B"]:
                with patch('gradio.update') as mock_update:
                    mock_update.return_value = MagicMock()
                    
                    result = resolution_manager.update_resolution_dropdown(model_type)
                    
                    # Verify the update was called
                    mock_update.assert_called_once()
                    
                    # Verify the call had the right structure
                    call_args = mock_update.call_args[1]
                    self.assertIn('choices', call_args)
                    self.assertIn('value', call_args)
                    self.assertIn('info', call_args)
                    
                    # Verify choices are correct for each model
                    if model_type == "ti2v-5B":
                        self.assertEqual(len(call_args['choices']), 4)
                        self.assertIn("1024x1024", call_args['choices'])
                    else:
                        self.assertEqual(len(call_args['choices']), 3)
                        self.assertNotIn("1024x1024", call_args['choices'])
        
        except Exception as e:
            self.fail(f"UI event handlers integration failed: {e}")


if __name__ == '__main__':
    unittest.main(verbosity=2)