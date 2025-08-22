"""
Integration tests for LoRA management tab UI components
Tests the complete LoRA management workflow including upload, selection, and file management
"""

import pytest
import tempfile
import shutil
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import gradio as gr

# Import the UI class
from ui import Wan22UI

class TestLoRAUIIntegration:
    """Test suite for LoRA UI integration"""
    
    @pytest.fixture
    def temp_config(self):
        """Create temporary configuration for testing"""
        temp_dir = tempfile.mkdtemp()
        config = {
            "directories": {
                "models_directory": os.path.join(temp_dir, "models"),
                "loras_directory": os.path.join(temp_dir, "loras"),
                "outputs_directory": os.path.join(temp_dir, "outputs")
            },
            "optimization": {
                "default_quantization": "bf16",
                "enable_offload": True,
                "vae_tile_size": 256,
                "max_vram_usage_gb": 12
            },
            "generation": {
                "default_resolution": "1280x720",
                "default_steps": 50,
                "max_prompt_length": 500
            }
        }
        
        # Create directories
        for dir_path in config["directories"].values():
            os.makedirs(dir_path, exist_ok=True)
        
        yield config
        
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def mock_lora_files(self, temp_config):
        """Create mock LoRA files for testing"""
        loras_dir = Path(temp_config["directories"]["loras_directory"])
        
        # Create mock LoRA files
        mock_files = [
            "cinematic_style.safetensors",
            "anime_character.pt",
            "realistic_portrait.ckpt"
        ]
        
        for filename in mock_files:
            file_path = loras_dir / filename
            # Create a small mock file with some content
            with open(file_path, 'wb') as f:
                f.write(b"mock_lora_data" * 1000)  # ~13KB file
        
        return mock_files
    
    @pytest.fixture
    def ui_instance(self, temp_config):
        """Create UI instance with temporary config"""
        # Write config to temporary file
        config_file = os.path.join(temp_config["directories"]["models_directory"], "test_config.json")
        with open(config_file, 'w') as f:
            json.dump(temp_config, f)
        
        # Mock dependencies to avoid actual model loading
        with patch('ui.get_model_manager'), \
             patch('ui.VRAMOptimizer'), \
             patch('ui.get_performance_profiler'), \
             patch('ui.start_performance_monitoring'), \
             patch('lora_ui_state.LoRAManager'):
            
            ui = Wan22UI(config_file)
            return ui
    
    def test_lora_tab_creation(self, ui_instance):
        """Test that LoRA tab is created with all required components"""
        # Check that lora_components dictionary exists
        assert hasattr(ui_instance, 'lora_components')
        assert isinstance(ui_instance.lora_components, dict)
        
        # Check required components exist
        required_components = [
            'lora_file_upload',
            'upload_btn',
            'upload_status',
            'refresh_loras_btn',
            'sort_loras',
            'auto_refresh_loras',
            'lora_library_display',
            'selection_summary',
            'clear_selection_btn',
            'memory_usage_display',
            'strength_controls_container',
            'selected_lora_for_action',
            'delete_lora_btn',
            'rename_lora_btn'
        ]
        
        for component in required_components:
            assert component in ui_instance.lora_components, f"Missing component: {component}"
    
    def test_lora_library_html_generation(self, ui_instance, mock_lora_files):
        """Test LoRA library HTML generation"""
        # Mock the LoRA UI state to return test data
        mock_display_data = {
            "selected_loras": [
                {
                    "name": "cinematic_style",
                    "size_formatted": "13.0 KB",
                    "strength": 0.8,
                    "is_valid": True
                }
            ],
            "available_loras": [
                {
                    "name": "anime_character",
                    "size_formatted": "13.0 KB"
                },
                {
                    "name": "realistic_portrait", 
                    "size_formatted": "13.0 KB"
                }
            ]
        }
        
        if ui_instance.lora_ui_state:
            ui_instance.lora_ui_state.get_display_data = Mock(return_value=mock_display_data)
        
        # Generate HTML
        html = ui_instance._get_lora_library_html()
        
        # Check that HTML contains expected elements
        assert "lora-grid" in html
        assert "cinematic_style" in html
        assert "anime_character" in html
        assert "realistic_portrait" in html
        assert "13.0 KB" in html
    
    def test_selection_summary_html_generation(self, ui_instance):
        """Test selection summary HTML generation"""
        # Mock display data with selections
        mock_display_data = {
            "selection_status": {
                "count": 2,
                "max_count": 5,
                "is_valid": True
            },
            "selected_loras": [
                {
                    "name": "cinematic_style",
                    "size_formatted": "13.0 KB",
                    "strength": 0.8,
                    "strength_percent": 80,
                    "is_valid": True
                },
                {
                    "name": "anime_character",
                    "size_formatted": "13.0 KB", 
                    "strength": 1.2,
                    "strength_percent": 120,
                    "is_valid": True
                }
            ]
        }
        
        if ui_instance.lora_ui_state:
            ui_instance.lora_ui_state.get_display_data = Mock(return_value=mock_display_data)
        
        # Generate HTML
        html = ui_instance._get_selection_summary_html()
        
        # Check content
        assert "Selected: 2/5" in html
        assert "cinematic_style" in html
        assert "anime_character" in html
        assert "0.8" in html
        assert "1.2" in html
    
    def test_memory_usage_html_generation(self, ui_instance):
        """Test memory usage HTML generation"""
        # Mock memory estimate
        mock_memory_estimate = {
            "total_mb": 156.0,
            "total_gb": 0.152,
            "individual_mb": {
                "cinematic_style": 78.0,
                "anime_character": 78.0
            },
            "estimated_load_time_seconds": 1.56
        }
        
        if ui_instance.lora_ui_state:
            ui_instance.lora_ui_state.estimate_memory_impact = Mock(return_value=mock_memory_estimate)
        
        # Generate HTML
        html = ui_instance._get_memory_usage_html()
        
        # Check content
        assert "156.0MB" in html
        assert "0.15GB" in html or "0.152GB" in html
        assert "1.56" in html or "1.6" in html
        assert "cinematic_style" in html
        assert "anime_character" in html
    
    def test_strength_controls_html_generation(self, ui_instance):
        """Test strength controls HTML generation"""
        # Mock display data with selections
        mock_display_data = {
            "selected_loras": [
                {
                    "name": "cinematic_style",
                    "strength": 0.8
                },
                {
                    "name": "anime_character",
                    "strength": 1.2
                }
            ]
        }
        
        if ui_instance.lora_ui_state:
            ui_instance.lora_ui_state.get_display_data = Mock(return_value=mock_display_data)
        
        # Generate HTML
        html = ui_instance._get_strength_controls_html()
        
        # Check content
        assert "cinematic_style" in html
        assert "anime_character" in html
        assert "0.8" in html
        assert "1.2" in html
        assert "input type=\"range\"" in html
    
    def test_available_lora_names(self, ui_instance):
        """Test getting available LoRA names for dropdown"""
        # Mock display data
        mock_display_data = {
            "selected_loras": [
                {"name": "cinematic_style"}
            ],
            "available_loras": [
                {"name": "anime_character"},
                {"name": "realistic_portrait"}
            ]
        }
        
        if ui_instance.lora_ui_state:
            ui_instance.lora_ui_state.get_display_data = Mock(return_value=mock_display_data)
        
        # Get names
        names = ui_instance._get_available_lora_names()
        
        # Check results
        assert isinstance(names, list)
        assert "cinematic_style" in names
        assert "anime_character" in names
        assert "realistic_portrait" in names
        assert len(names) == 3
    
    @patch('ui.LoRAUploadHandler')
    def test_handle_lora_upload_success(self, mock_upload_handler_class, ui_instance):
        """Test successful LoRA file upload"""
        # Mock upload handler
        mock_handler = Mock()
        mock_upload_handler_class.return_value = mock_handler
        
        # Mock successful upload result
        mock_handler.process_upload.return_value = {
            "success": True,
            "filename": "test_lora.safetensors",
            "size_mb": 144.5
        }
        
        # Mock file object
        mock_file = Mock()
        mock_file.name = "test_lora.safetensors"
        
        # Mock file reading
        with patch('builtins.open', mock_open(read_data=b"mock_lora_data")):
            # Test upload
            result = ui_instance._handle_lora_upload(mock_file)
            
            # Check results
            assert len(result) == 3  # notification, library_html, dropdown_update
            notification, library_html, dropdown_update = result
            
            # Check notification contains success message
            assert "✅" in notification
            assert "test_lora.safetensors" in notification
            assert "144.5MB" in notification
    
    @patch('ui.LoRAUploadHandler')
    def test_handle_lora_upload_failure(self, mock_upload_handler_class, ui_instance):
        """Test failed LoRA file upload"""
        # Mock upload handler
        mock_handler = Mock()
        mock_upload_handler_class.return_value = mock_handler
        
        # Mock failed upload result
        mock_handler.process_upload.return_value = {
            "success": False,
            "error": "Invalid file format",
            "filename": "invalid_file.txt"
        }
        
        # Mock file object
        mock_file = Mock()
        mock_file.name = "invalid_file.txt"
        
        # Mock file reading
        with patch('builtins.open', mock_open(read_data=b"invalid_data")):
            # Test upload
            result = ui_instance._handle_lora_upload(mock_file)
            
            # Check results
            notification, library_html, dropdown_update = result
            
            # Check notification contains error message
            assert "❌" in notification
            assert "Invalid file format" in notification
    
    def test_refresh_lora_library(self, ui_instance):
        """Test LoRA library refresh functionality"""
        # Mock LoRA UI state refresh
        if ui_instance.lora_ui_state:
            ui_instance.lora_ui_state.refresh_state = Mock()
        
        # Test refresh
        result = ui_instance._refresh_lora_library()
        
        # Check that all displays are returned
        assert len(result) == 5  # library, selection, memory, strength, dropdown
        
        # Check that refresh was called
        if ui_instance.lora_ui_state:
            ui_instance.lora_ui_state.refresh_state.assert_called_once()
    
    def test_clear_lora_selection(self, ui_instance):
        """Test clearing LoRA selection"""
        # Mock LoRA UI state
        if ui_instance.lora_ui_state:
            ui_instance.lora_ui_state.clear_selection = Mock(return_value=(True, "Selection cleared"))
        
        # Test clear selection
        result = ui_instance._clear_lora_selection()
        
        # Check results
        assert len(result) == 4  # selection, memory, strength, library
        
        # Check that clear_selection was called
        if ui_instance.lora_ui_state:
            ui_instance.lora_ui_state.clear_selection.assert_called_once()
    
    def test_apply_lora_preset(self, ui_instance):
        """Test applying LoRA presets"""
        # Mock LoRA UI state and manager
        if ui_instance.lora_ui_state:
            ui_instance.lora_ui_state.clear_selection = Mock()
            ui_instance.lora_ui_state.update_selection = Mock(return_value=(True, "Updated"))
            ui_instance.lora_ui_state.lora_manager = Mock()
            ui_instance.lora_ui_state.lora_manager.list_available_loras = Mock(return_value={
                "cinematic_lora": {},
                "film_grain": {},
                "lighting_enhance": {}
            })
        
        # Test preset application
        result = ui_instance._apply_lora_preset("cinematic")
        
        # Check results
        assert len(result) == 4  # selection, memory, strength, library
        
        # Check that methods were called
        if ui_instance.lora_ui_state:
            ui_instance.lora_ui_state.clear_selection.assert_called_once()
    
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.unlink')
    def test_delete_lora_file(self, mock_unlink, mock_exists, ui_instance):
        """Test LoRA file deletion"""
        # Mock file exists
        mock_exists.return_value = True
        
        # Mock LoRA UI state
        if ui_instance.lora_ui_state:
            ui_instance.lora_ui_state.remove_selection = Mock()
            ui_instance.lora_ui_state.refresh_state = Mock()
        
        # Test deletion
        result = ui_instance._delete_lora_file("test_lora")
        
        # Check results
        assert len(result) == 3  # notification, library, dropdown
        notification, library_html, dropdown_update = result
        
        # Check success notification
        assert "✅" in notification
        assert "test_lora" in notification
        
        # Check that file was deleted
        mock_unlink.assert_called_once()
    
    def test_show_rename_dialog(self, ui_instance):
        """Test showing rename dialog"""
        # Test with valid LoRA name
        result = ui_instance._show_rename_dialog("test_lora")
        
        # Check results
        assert len(result) == 2  # dialog visibility, name field
        dialog_update, name_value = result
        
        # Check that dialog is shown and name is set
        assert name_value == "test_lora"
    
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.rename')
    def test_confirm_rename_lora(self, mock_rename, mock_exists, ui_instance):
        """Test confirming LoRA file rename"""
        # Mock file operations
        def exists_side_effect(path):
            # Old file exists, new file doesn't
            return "old_lora" in str(path)
        
        mock_exists.side_effect = exists_side_effect
        
        # Mock LoRA UI state
        if ui_instance.lora_ui_state:
            ui_instance.lora_ui_state.selected_loras = {"old_lora": Mock(strength=0.8)}
            ui_instance.lora_ui_state.remove_selection = Mock()
            ui_instance.lora_ui_state.update_selection = Mock()
            ui_instance.lora_ui_state.refresh_state = Mock()
        
        # Test rename
        result = ui_instance._confirm_rename_lora("old_lora", "new_lora")
        
        # Check results
        assert len(result) == 4  # notification, dialog, library, dropdown
        notification, dialog_update, library_html, dropdown_update = result
        
        # Check success notification
        assert "✅" in notification
        assert "old_lora" in notification
        assert "new_lora" in notification
        
        # Check that file was renamed
        mock_rename.assert_called_once()
    
    def test_cancel_rename_dialog(self, ui_instance):
        """Test canceling rename dialog"""
        result = ui_instance._cancel_rename_dialog()
        
        # Check results
        assert len(result) == 2  # dialog visibility, name field
        dialog_update, name_value = result
        
        # Check that dialog is hidden and name is cleared
        assert name_value == ""

def mock_open(read_data=b""):
    """Helper function to mock file opening"""
    from unittest.mock import mock_open as original_mock_open
    return original_mock_open(read_data=read_data)

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])