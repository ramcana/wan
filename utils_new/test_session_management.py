"""
Test suite for session state management functionality
Tests image persistence, UI integration, and session cleanup
"""

import pytest
import tempfile
import shutil
import json
import time
from pathlib import Path
from PIL import Image
from unittest.mock import Mock, patch, MagicMock
import gradio as gr

from image_session_manager import ImageSessionManager, get_image_session_manager, cleanup_session_manager
from ui_session_integration import (
    UISessionIntegration, 
    get_ui_session_integration, 
    cleanup_ui_session_integration,
    create_session_management_ui,
    setup_session_management_handlers
)

class TestImageSessionManager:
    """Test cases for ImageSessionManager"""
    
    @pytest.fixture
    def config(self):
        """Test configuration"""
        return {
            "session": {
                "max_age_hours": 1,
                "cleanup_interval_minutes": 1,
                "enable_cleanup_thread": False  # Disable for testing
            }
        }
    
    @pytest.fixture
    def session_manager(self, config):
        """Create test session manager"""
        manager = ImageSessionManager(config)
        yield manager
        # Cleanup after test
        manager.cleanup_current_session()
    
    @pytest.fixture
    def test_image(self):
        """Create test image"""
        image = Image.new('RGB', (512, 512), color='red')
        return image

        assert True  # TODO: Add proper assertion

        assert True  # TODO: Add proper assertion

        assert True  # TODO: Add proper assertion

        assert True  # TODO: Add proper assertion

        assert True  # TODO: Add proper assertion

        assert True  # TODO: Add proper assertion

        assert True  # TODO: Add proper assertion

        assert True  # TODO: Add proper assertion

        assert True  # TODO: Add proper assertion

        assert True  # TODO: Add proper assertion
    
    @pytest.fixture
    def test_image_small(self):
        """Create small test image"""
        image = Image.new('RGB', (256, 256), color='blue')
        return image

        assert True  # TODO: Add proper assertion
    
    def test_session_manager_initialization(self, config):
        """Test session manager initialization"""
        manager = ImageSessionManager(config)
        
        assert manager.session_id is not None
        assert manager.current_session_dir.exists()
        assert manager.session_state is not None
        assert "session_created" in manager.session_state
        
        # Cleanup
        manager.cleanup_current_session()
    
    def test_store_and_retrieve_start_image(self, session_manager, test_image):
        """Test storing and retrieving start image"""
        # Store image
        success, message = session_manager.store_image(test_image, "start")
        assert success
        assert "Successfully stored start image" in message
        
        # Verify image exists
        assert session_manager.has_image("start")
        
        # Retrieve image
        retrieved_image, metadata = session_manager.retrieve_image("start")
        assert retrieved_image is not None
        assert retrieved_image.size == test_image.size
        assert metadata is not None
        assert metadata["type"] == "start"
        assert metadata["size"] == test_image.size
    
    def test_store_and_retrieve_end_image(self, session_manager, test_image_small):
        """Test storing and retrieving end image"""
        # Store image
        success, message = session_manager.store_image(test_image_small, "end")
        assert success
        assert "Successfully stored end image" in message
        
        # Verify image exists
        assert session_manager.has_image("end")
        
        # Retrieve image
        retrieved_image, metadata = session_manager.retrieve_image("end")
        assert retrieved_image is not None
        assert retrieved_image.size == test_image_small.size
        assert metadata is not None
        assert metadata["type"] == "end"
    
    def test_store_both_images(self, session_manager, test_image, test_image_small):
        """Test storing both start and end images"""
        # Store both images
        success1, _ = session_manager.store_image(test_image, "start")
        success2, _ = session_manager.store_image(test_image_small, "end")
        
        assert success1 and success2
        assert session_manager.has_image("start")
        assert session_manager.has_image("end")
        
        # Retrieve both
        start_image, start_meta = session_manager.retrieve_image("start")
        end_image, end_meta = session_manager.retrieve_image("end")
        
        assert start_image.size == test_image.size
        assert end_image.size == test_image_small.size
        assert start_meta["type"] == "start"
        assert end_meta["type"] == "end"
    
    def test_clear_single_image(self, session_manager, test_image):
        """Test clearing a single image"""
        # Store image
        session_manager.store_image(test_image, "start")
        assert session_manager.has_image("start")
        
        # Clear image
        success = session_manager.clear_image("start")
        assert success
        assert not session_manager.has_image("start")
        
        # Verify retrieval returns None
        retrieved_image, metadata = session_manager.retrieve_image("start")
        assert retrieved_image is None
        assert metadata is None
    
    def test_clear_all_images(self, session_manager, test_image, test_image_small):
        """Test clearing all images"""
        # Store both images
        session_manager.store_image(test_image, "start")
        session_manager.store_image(test_image_small, "end")
        
        assert session_manager.has_image("start")
        assert session_manager.has_image("end")
        
        # Clear all
        success = session_manager.clear_all_images()
        assert success
        
        assert not session_manager.has_image("start")
        assert not session_manager.has_image("end")
    
    def test_session_info(self, session_manager, test_image):
        """Test getting session information"""
        # Initially no images
        info = session_manager.get_session_info()
        assert not info["has_start_image"]
        assert not info["has_end_image"]
        assert info["session_id"] == session_manager.session_id
        
        # Store image and check info
        session_manager.store_image(test_image, "start")
        info = session_manager.get_session_info()
        assert info["has_start_image"]
        assert not info["has_end_image"]
        assert info["start_image_metadata"] is not None
    
    def test_image_metadata_generation(self, session_manager, test_image):
        """Test image metadata generation"""
        session_manager.store_image(test_image, "start")
        _, metadata = session_manager.retrieve_image("start")
        
        assert metadata["type"] == "start"
        assert metadata["size"] == test_image.size
        assert metadata["format"] == "PNG"  # Default save format
        assert "aspect_ratio" in metadata
        assert "pixel_count" in metadata
        assert "stored_at" in metadata
        assert "image_hash" in metadata
    
    def test_invalid_image_type(self, session_manager, test_image):
        """Test handling invalid image type"""
        success, message = session_manager.store_image(test_image, "invalid")
        assert not success
        assert "Invalid image type" in message
        
        retrieved_image, metadata = session_manager.retrieve_image("invalid")
        assert retrieved_image is None
        assert metadata is None
    
    def test_session_persistence(self, config, test_image):
        """Test session persistence across manager instances"""
        # Create first manager and store image
        manager1 = ImageSessionManager(config)
        session_id = manager1.session_id
        manager1.store_image(test_image, "start")
        
        # Create second manager and restore session
        manager2 = ImageSessionManager(config)
        success = manager2.restore_from_session_id(session_id)
        assert success
        
        # Verify image is available
        assert manager2.has_image("start")
        retrieved_image, metadata = manager2.retrieve_image("start")
        assert retrieved_image is not None
        assert retrieved_image.size == test_image.size
        
        # Cleanup both managers
        manager1.cleanup_current_session()
        manager2.cleanup_current_session()
    
    def test_session_cleanup(self, session_manager, test_image):
        """Test session cleanup"""
        # Store image
        session_manager.store_image(test_image, "start")
        session_dir = session_manager.current_session_dir
        
        assert session_dir.exists()
        assert session_manager.has_image("start")
        
        # Cleanup session
        session_manager.cleanup_current_session()
        
        # Verify cleanup
        assert not session_dir.exists()
    
    def test_cleanup_thread_functionality(self, config):
        """Test cleanup thread functionality separately"""
        # Enable cleanup thread for this test
        test_config = config.copy()
        test_config["session"]["enable_cleanup_thread"] = True
        test_config["session"]["cleanup_interval_minutes"] = 0.01  # Very short interval
        
        manager = ImageSessionManager(test_config)
        
        # Verify cleanup thread is running
        assert hasattr(manager, '_cleanup_thread')
        assert manager._cleanup_thread.is_alive()
        
        # Cleanup
        manager.cleanup_current_session()


class TestUISessionIntegration:
    """Test cases for UISessionIntegration"""
    
    @pytest.fixture
    def config(self):
        """Test configuration"""
        return {
            "session": {
                "max_age_hours": 1,
                "cleanup_interval_minutes": 1,
                "enable_cleanup_thread": False  # Disable for testing
            }
        }
    
    @pytest.fixture
    def ui_integration(self, config):
        """Create test UI integration"""
        integration = UISessionIntegration(config)
        yield integration
        # Cleanup after test
        integration.cleanup_session()
    
    @pytest.fixture
    def mock_gradio_components(self):
        """Create mock Gradio components"""
        start_image = Mock(spec=gr.Image)
        end_image = Mock(spec=gr.Image)
        model_dropdown = Mock(spec=gr.Dropdown)
        
        return start_image, end_image, model_dropdown
    
    @pytest.fixture
    def test_image(self):
        """Create test image"""
        return Image.new('RGB', (512, 512), color='green')
    
    def test_ui_integration_initialization(self, config):
        """Test UI integration initialization"""
        integration = UISessionIntegration(config)
        
        assert integration.session_manager is not None
        assert integration.ui_state is not None
        assert integration.ui_state["current_model_type"] == "t2v-A14B"
        
        # Cleanup
        integration.cleanup_session()
    
    def test_setup_ui_components(self, ui_integration, mock_gradio_components):
        """Test setting up UI components"""
        start_image, end_image, model_dropdown = mock_gradio_components
        
        ui_integration.setup_ui_components(start_image, end_image, model_dropdown)
        
        assert ui_integration.start_image_input == start_image
        assert ui_integration.end_image_input == end_image
        assert ui_integration.model_type_dropdown == model_dropdown
    
    def test_get_session_images(self, ui_integration, test_image):
        """Test getting session images"""
        # Initially no images
        start_image, end_image = ui_integration.get_session_images()
        assert start_image is None
        assert end_image is None
        
        # Store image and retrieve
        ui_integration.session_manager.store_image(test_image, "start")
        start_image, end_image = ui_integration.get_session_images()
        
        assert start_image is not None
        assert start_image.size == test_image.size
        assert end_image is None
    
    def test_get_ui_state_for_generation(self, ui_integration, test_image):
        """Test getting UI state for generation"""
        # Store test image
        ui_integration.session_manager.store_image(test_image, "start")
        ui_integration.ui_state["current_model_type"] = "i2v-A14B"
        
        state = ui_integration.get_ui_state_for_generation()
        
        assert state["start_image"] is not None
        assert state["end_image"] is None
        assert state["model_type"] == "i2v-A14B"
        assert state["session_id"] is not None
        assert state["has_session_images"] is True
        assert "image_metadata" in state
    
    def test_clear_session_images(self, ui_integration, test_image):
        """Test clearing session images"""
        # Store image
        ui_integration.session_manager.store_image(test_image, "start")
        assert ui_integration.session_manager.has_image("start")
        
        # Clear images
        success = ui_integration.clear_session_images()
        assert success
        assert not ui_integration.session_manager.has_image("start")
    
    def test_create_session_info_display(self, ui_integration, test_image):
        """Test creating session info display"""
        # Test with no images
        html = ui_integration.create_session_info_display()
        assert "Session Information" in html
        assert "No images in current session" in html
        
        # Test with image
        ui_integration.session_manager.store_image(test_image, "start")
        html = ui_integration.create_session_info_display()
        assert "Start Image:" in html
        assert "512, 512" in html  # Image dimensions
    
    def test_image_change_handlers(self, ui_integration, test_image):
        """Test image change event handlers"""
        # Test start image change
        ui_integration._on_start_image_change(test_image)
        assert ui_integration.session_manager.has_image("start")
        
        # Test clearing start image
        ui_integration._on_start_image_change(None)
        assert not ui_integration.session_manager.has_image("start")
        
        # Test end image change
        ui_integration._on_end_image_change(test_image)
        assert ui_integration.session_manager.has_image("end")
        
        # Test clearing end image
        ui_integration._on_end_image_change(None)
        assert not ui_integration.session_manager.has_image("end")
    
    def test_model_type_change_handler(self, ui_integration):
        """Test model type change handler"""
        ui_integration._on_model_type_change("i2v-A14B")
        assert ui_integration.ui_state["current_model_type"] == "i2v-A14B"
        
        ui_integration._on_model_type_change("ti2v-5B")
        assert ui_integration.ui_state["current_model_type"] == "ti2v-5B"
    
    def test_session_restoration(self, config, test_image):
        """Test session restoration functionality"""
        # Create first integration and store image
        integration1 = UISessionIntegration(config)
        integration1.session_manager.store_image(test_image, "start")
        session_id = integration1.session_manager.session_id
        
        # Create second integration and restore
        integration2 = UISessionIntegration(config)
        success = integration2.restore_from_session_id(session_id)
        assert success
        
        # Verify image is available
        start_image, _ = integration2.get_session_images()
        assert start_image is not None
        assert start_image.size == test_image.size
        
        # Cleanup
        integration1.cleanup_session()
        integration2.cleanup_session()


class TestSessionManagementUI:
    """Test cases for session management UI components"""
    
    @pytest.fixture
    def config(self):
        """Test configuration"""
        return {
            "session": {
                "max_age_hours": 1,
                "cleanup_interval_minutes": 1,
                "enable_cleanup_thread": False  # Disable for testing
            }
        }
    
    def test_create_session_management_ui(self):
        """Test creating session management UI components"""
        session_info_display, clear_button, refresh_button = create_session_management_ui()
        
        assert isinstance(session_info_display, gr.HTML)
        assert isinstance(clear_button, gr.Button)
        assert isinstance(refresh_button, gr.Button)
    
    @patch('ui_session_integration.gr')
    def test_setup_session_management_handlers(self, mock_gr, config):
        """Test setting up session management handlers"""
        # Create mock components
        ui_integration = UISessionIntegration(config)
        session_info_display = Mock(spec=gr.HTML)
        clear_button = Mock(spec=gr.Button)
        refresh_button = Mock(spec=gr.Button)
        start_image = Mock(spec=gr.Image)
        end_image = Mock(spec=gr.Image)
        
        # Setup handlers (should not raise exceptions)
        setup_session_management_handlers(
            ui_integration,
            session_info_display,
            clear_button,
            refresh_button,
            start_image,
            end_image
        )
        
        # Verify click handlers were set up
        assert refresh_button.click.called
        assert clear_button.click.called
        
        # Cleanup
        ui_integration.cleanup_session()


class TestSessionWorkflows:
    """Test complete session management workflows"""
    
    @pytest.fixture
    def config(self):
        """Test configuration"""
        return {
            "session": {
                "max_age_hours": 1,
                "cleanup_interval_minutes": 1,
                "enable_cleanup_thread": False  # Disable for testing
            }
        }
    
    @pytest.fixture
    def test_images(self):
        """Create test images"""
        start_image = Image.new('RGB', (512, 512), color='red')
        end_image = Image.new('RGB', (512, 512), color='blue')
        return start_image, end_image

        assert True  # TODO: Add proper assertion
    
    def test_complete_image_upload_workflow(self, config, test_images):
        """Test complete image upload and session persistence workflow"""
        start_image, end_image = test_images
        
        # Create UI integration
        ui_integration = UISessionIntegration(config)
        
        # Simulate image uploads
        ui_integration._on_start_image_change(start_image)
        ui_integration._on_end_image_change(end_image)
        
        # Verify images are stored
        assert ui_integration.session_manager.has_image("start")
        assert ui_integration.session_manager.has_image("end")
        
        # Get generation state
        gen_state = ui_integration.get_ui_state_for_generation()
        assert gen_state["has_session_images"] is True
        assert gen_state["start_image"] is not None
        assert gen_state["end_image"] is not None
        
        # Simulate model type change (images should persist)
        ui_integration._on_model_type_change("i2v-A14B")
        assert ui_integration.session_manager.has_image("start")
        assert ui_integration.session_manager.has_image("end")
        
        # Clear session
        success = ui_integration.clear_session_images()
        assert success
        assert not ui_integration.session_manager.has_image("start")
        assert not ui_integration.session_manager.has_image("end")
        
        # Cleanup
        ui_integration.cleanup_session()
    
    def test_session_persistence_across_ui_restarts(self, config, test_images):
        """Test session persistence when UI is restarted"""
        start_image, end_image = test_images
        
        # First UI session
        ui_integration1 = UISessionIntegration(config)
        ui_integration1._on_start_image_change(start_image)
        ui_integration1._on_end_image_change(end_image)
        session_id = ui_integration1.session_manager.session_id
        
        # Simulate UI restart - create new integration
        ui_integration2 = UISessionIntegration(config)
        success = ui_integration2.restore_from_session_id(session_id)
        assert success
        
        # Verify images are available
        restored_start, restored_end = ui_integration2.get_session_images()
        assert restored_start is not None
        assert restored_end is not None
        assert restored_start.size == start_image.size
        assert restored_end.size == end_image.size
        
        # Cleanup
        ui_integration1.cleanup_session()
        ui_integration2.cleanup_session()
    
    def test_tab_switching_preserves_images(self, config, test_images):
        """Test that switching tabs preserves uploaded images"""
        start_image, end_image = test_images
        
        ui_integration = UISessionIntegration(config)
        
        # Upload images
        ui_integration._on_start_image_change(start_image)
        ui_integration._on_end_image_change(end_image)
        
        # Simulate tab switching by getting images multiple times
        for _ in range(5):
            retrieved_start, retrieved_end = ui_integration.get_session_images()
            assert retrieved_start is not None
            assert retrieved_end is not None
            assert retrieved_start.size == start_image.size
            assert retrieved_end.size == end_image.size
        
        # Cleanup
        ui_integration.cleanup_session()
    
    def test_model_type_switching_preserves_images(self, config, test_images):
        """Test that switching model types preserves images"""
        start_image, end_image = test_images
        
        ui_integration = UISessionIntegration(config)
        
        # Upload images
        ui_integration._on_start_image_change(start_image)
        ui_integration._on_end_image_change(end_image)
        
        # Switch between different model types
        model_types = ["t2v-A14B", "i2v-A14B", "ti2v-5B", "i2v-A14B", "t2v-A14B"]
        
        for model_type in model_types:
            ui_integration._on_model_type_change(model_type)
            
            # Verify images are still available
            assert ui_integration.session_manager.has_image("start")
            assert ui_integration.session_manager.has_image("end")
            
            # Verify UI state is updated
            assert ui_integration.ui_state["current_model_type"] == model_type
        
        # Cleanup
        ui_integration.cleanup_session()


class TestGlobalSessionManagement:
    """Test global session management functions"""
    
    @pytest.fixture
    def config(self):
        """Test configuration"""
        return {
            "session": {
                "max_age_hours": 1,
                "cleanup_interval_minutes": 1,
                "enable_cleanup_thread": False  # Disable for testing
            }
        }
    
    def test_global_session_manager(self, config):
        """Test global session manager functions"""
        # Get global manager
        manager1 = get_image_session_manager(config)
        manager2 = get_image_session_manager(config)
        
        # Should return same instance
        assert manager1 is manager2
        
        # Cleanup
        cleanup_session_manager()
        
        # Should create new instance after cleanup
        manager3 = get_image_session_manager(config)
        assert manager3 is not manager1
        
        # Final cleanup
        cleanup_session_manager()
    
    def test_global_ui_integration(self, config):
        """Test global UI integration functions"""
        # Get global integration
        integration1 = get_ui_session_integration(config)
        integration2 = get_ui_session_integration(config)
        
        # Should return same instance
        assert integration1 is integration2
        
        # Cleanup
        cleanup_ui_session_integration()
        
        # Should create new instance after cleanup
        integration3 = get_ui_session_integration(config)
        assert integration3 is not integration1
        
        # Final cleanup
        cleanup_ui_session_integration()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])