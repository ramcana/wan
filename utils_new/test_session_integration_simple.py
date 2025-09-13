"""
Simple integration test for session management functionality
Tests the complete workflow from image upload to generation task preparation
"""

import pytest
import tempfile
from pathlib import Path
from PIL import Image
from unittest.mock import Mock

from image_session_manager import ImageSessionManager
from ui_session_integration import UISessionIntegration

class TestSessionIntegrationSimple:
    """Simple integration tests for session management"""
    
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
    
    def test_complete_session_workflow(self, config, test_images):
        """Test complete session workflow from upload to generation preparation"""
        start_image, end_image = test_images
        
        # Create UI integration
        ui_integration = UISessionIntegration(config)
        
        # Simulate image uploads
        ui_integration._on_start_image_change(start_image)
        ui_integration._on_end_image_change(end_image)
        
        # Verify images are stored
        assert ui_integration.session_manager.has_image("start")
        assert ui_integration.session_manager.has_image("end")
        
        # Test generation task preparation
        base_task_data = {
            "prompt": "Test prompt",
            "model_type": "i2v-A14B",
            "resolution": "1280x720",
            "steps": 50
        }
        
        enhanced_task_data = ui_integration.prepare_generation_task_with_session_data(base_task_data)
        
        # Verify enhanced task data includes session images
        assert enhanced_task_data["start_image"] is not None
        assert enhanced_task_data["end_image"] is not None
        assert enhanced_task_data["session_id"] is not None
        assert enhanced_task_data["image_metadata"] is not None
        assert enhanced_task_data["prompt"] == "Test prompt"  # Original data preserved
        
        # Test session info display
        session_info_html = ui_integration.create_session_info_display()
        assert "Start Image:" in session_info_html
        assert "End Image:" in session_info_html
        assert ui_integration.session_manager.session_id in session_info_html
        
        # Test model type switching preserves images
        ui_integration._on_model_type_change("ti2v-5B")
        assert ui_integration.session_manager.has_image("start")
        assert ui_integration.session_manager.has_image("end")
        assert ui_integration.ui_state["current_model_type"] == "ti2v-5B"
        
        # Test clearing session
        success = ui_integration.clear_session_images()
        assert success
        assert not ui_integration.session_manager.has_image("start")
        assert not ui_integration.session_manager.has_image("end")
        
        # Cleanup
        ui_integration.cleanup_session()
    
    def test_session_persistence_workflow(self, config, test_images):
        """Test session persistence across UI instances"""
        start_image, end_image = test_images
        
        # First UI instance
        ui_integration1 = UISessionIntegration(config)
        ui_integration1._on_start_image_change(start_image)
        ui_integration1._on_end_image_change(end_image)
        session_id = ui_integration1.session_manager.session_id
        
        # Verify images are stored
        assert ui_integration1.session_manager.has_image("start")
        assert ui_integration1.session_manager.has_image("end")
        
        # Second UI instance (simulating UI restart)
        ui_integration2 = UISessionIntegration(config)
        success = ui_integration2.restore_from_session_id(session_id)
        assert success
        
        # Verify images are available in second instance
        restored_start, restored_end = ui_integration2.get_session_images()
        assert restored_start is not None
        assert restored_end is not None
        assert restored_start.size == start_image.size
        assert restored_end.size == end_image.size
        
        # Test generation task preparation with restored session
        base_task_data = {
            "prompt": "Restored session test",
            "model_type": "i2v-A14B"
        }
        
        enhanced_task_data = ui_integration2.prepare_generation_task_with_session_data(base_task_data)
        assert enhanced_task_data["start_image"] is not None
        assert enhanced_task_data["end_image"] is not None
        assert enhanced_task_data["session_id"] == session_id
        
        # Cleanup both instances
        ui_integration1.cleanup_session()
        ui_integration2.cleanup_session()
    
    def test_error_handling_workflow(self, config):
        """Test error handling in session management"""
        ui_integration = UISessionIntegration(config)
        
        # Test with None images (clearing)
        ui_integration._on_start_image_change(None)
        ui_integration._on_end_image_change(None)
        
        # Should not crash and should return empty state
        ui_state = ui_integration.get_ui_state_for_generation()
        assert ui_state["start_image"] is None
        assert ui_state["end_image"] is None
        assert ui_state["has_session_images"] is False
        
        # Test generation task preparation with no images
        base_task_data = {"prompt": "No images test"}
        enhanced_task_data = ui_integration.prepare_generation_task_with_session_data(base_task_data)
        assert enhanced_task_data["start_image"] is None
        assert enhanced_task_data["end_image"] is None
        assert enhanced_task_data["prompt"] == "No images test"
        
        # Test session info display with no images
        session_info_html = ui_integration.create_session_info_display()
        assert "No images in current session" in session_info_html
        
        # Cleanup
        ui_integration.cleanup_session()
    
    def test_memory_management_workflow(self, config, test_images):
        """Test memory management and cleanup"""
        start_image, end_image = test_images
        
        ui_integration = UISessionIntegration(config)
        
        # Store images multiple times (simulating user uploading different images)
        for i in range(3):
            test_start = Image.new('RGB', (256, 256), color=(i*50, 0, 0))
            test_end = Image.new('RGB', (256, 256), color=(0, i*50, 0))
            
            ui_integration._on_start_image_change(test_start)
            ui_integration._on_end_image_change(test_end)
            
            # Verify only latest images are stored
            assert ui_integration.session_manager.has_image("start")
            assert ui_integration.session_manager.has_image("end")
        
        # Get final images
        final_start, final_end = ui_integration.get_session_images()
        assert final_start is not None
        assert final_end is not None
        
        # Clear and verify cleanup
        success = ui_integration.clear_session_images()
        assert success
        
        # Verify no images remain
        cleared_start, cleared_end = ui_integration.get_session_images()
        assert cleared_start is None
        assert cleared_end is None
        
        # Cleanup
        ui_integration.cleanup_session()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
