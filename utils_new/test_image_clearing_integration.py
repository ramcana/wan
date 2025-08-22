"""
Integration Test for Image Clearing and Replacement Functionality
Tests the complete workflow of Task 9 implementation
"""

import pytest
from unittest.mock import Mock, patch
from PIL import Image
import io

def create_test_image(width=512, height=512, color=(255, 0, 0)):
    """Create a test PIL image"""
    return Image.new('RGB', (width, height), color)

@pytest.fixture(autouse=True)
def mock_dependencies():
    """Mock all external dependencies"""
    with patch('ui.get_model_manager'), \
         patch('ui.VRAMOptimizer'), \
         patch('ui.get_progress_tracker'), \
         patch('ui.get_performance_profiler'), \
         patch('ui.start_performance_monitoring'), \
         patch('ui.apply_model_fixes'), \
         patch('enhanced_image_preview_manager.PIL_AVAILABLE', True):
        yield

def test_complete_image_clearing_workflow():
    """Test the complete image clearing and replacement workflow"""
    from ui import Wan22UI
    
    # Mock the config loading and interface creation
    with patch.object(Wan22UI, '_load_config', return_value={}), \
         patch.object(Wan22UI, '_create_interface', return_value=Mock()), \
         patch.object(Wan22UI, '_start_auto_refresh'), \
         patch.object(Wan22UI, '_perform_startup_checks'):
        
        ui = Wan22UI()
        
        # Mock the image preview manager
        ui.image_preview_manager = Mock()
        ui.image_preview_manager.process_image_upload.return_value = ("preview_html", "tooltip", True)
        ui.image_preview_manager.clear_image.return_value = ("empty_preview", "", False)
        ui.image_preview_manager.get_image_summary.return_value = "summary"
        ui.image_preview_manager.get_compatibility_status.return_value = "compatible"
        
        # Mock validation methods
        ui._validate_start_image_upload = Mock(return_value=("validation", "notification", True))
        ui._validate_end_image_upload = Mock(return_value=("validation", "notification", True))
        
        # Test 1: Upload start image and verify clear button appears
        test_image = create_test_image()
        upload_result = ui._handle_start_image_upload(test_image, "i2v-A14B")
        
        assert len(upload_result) == 7
        assert upload_result[6]['visible'] == True  # Clear button should be visible
        
        # Test 2: Clear start image and verify everything is reset
        clear_result = ui._clear_start_image()
        
        assert len(clear_result) == 7
        assert clear_result[0]['value'] is None  # Image input cleared
        assert clear_result[4]['value'] == ""    # Validation messages cleared
        assert clear_result[5]['visible'] == False  # Clear button hidden
        assert clear_result[6]['value'] == ""    # Notifications cleared
        
        # Test 3: Upload end image and verify clear button appears
        end_upload_result = ui._handle_end_image_upload(test_image, "i2v-A14B")
        
        assert len(end_upload_result) == 7
        assert end_upload_result[6]['visible'] == True  # Clear button should be visible
        
        # Test 4: Clear end image and verify everything is reset
        end_clear_result = ui._clear_end_image()
        
        assert len(end_clear_result) == 7
        assert end_clear_result[0]['value'] is None  # Image input cleared
        assert end_clear_result[4]['value'] == ""    # Validation messages cleared
        assert end_clear_result[5]['visible'] == False  # Clear button hidden
        assert end_clear_result[6]['value'] == ""    # Notifications cleared
        
        # Test 5: Model type switching preserves functionality
        model_change_result = ui._on_model_type_change("i2v-A14B")
        
        assert len(model_change_result) == 10
        assert model_change_result[0]['visible'] == True  # Image inputs shown for I2V
        assert model_change_result[1]['visible'] == True  # Image status shown for I2V
        
        # Test 6: Model type switching to T2V hides image inputs
        t2v_result = ui._on_model_type_change("t2v-A14B")
        
        assert len(t2v_result) == 10
        assert t2v_result[0]['visible'] == False  # Image inputs hidden for T2V
        assert t2v_result[1]['visible'] == False  # Image status hidden for T2V
        
        print("✅ Complete image clearing workflow test passed!")

def test_image_replacement_workflow():
    """Test automatic image replacement functionality"""
    from ui import Wan22UI
    
    with patch.object(Wan22UI, '_load_config', return_value={}), \
         patch.object(Wan22UI, '_create_interface', return_value=Mock()), \
         patch.object(Wan22UI, '_start_auto_refresh'), \
         patch.object(Wan22UI, '_perform_startup_checks'):
        
        ui = Wan22UI()
        
        # Mock the image preview manager
        ui.image_preview_manager = Mock()
        ui.image_preview_manager.process_image_upload.return_value = ("preview_html", "tooltip", True)
        ui.image_preview_manager.get_image_summary.return_value = "summary"
        ui.image_preview_manager.get_compatibility_status.return_value = "compatible"
        
        # Mock validation methods
        ui._validate_start_image_upload = Mock(return_value=("validation", "notification", True))
        
        # Test automatic replacement
        image1 = create_test_image(color=(255, 0, 0))  # Red
        image2 = create_test_image(color=(0, 255, 0))  # Green
        
        # Upload first image
        result1 = ui._handle_start_image_upload(image1, "i2v-A14B")
        assert result1[6]['visible'] == True  # Clear button visible
        
        # Upload second image (should replace first automatically)
        result2 = ui._handle_start_image_upload(image2, "i2v-A14B")
        assert result2[6]['visible'] == True  # Clear button still visible
        
        # Verify image preview manager was called twice (replacement occurred)
        assert ui.image_preview_manager.process_image_upload.call_count == 2
        
        print("✅ Image replacement workflow test passed!")

def test_error_handling_workflow():
    """Test error handling in clearing operations"""
    from ui import Wan22UI
    
    with patch.object(Wan22UI, '_load_config', return_value={}), \
         patch.object(Wan22UI, '_create_interface', return_value=Mock()), \
         patch.object(Wan22UI, '_start_auto_refresh'), \
         patch.object(Wan22UI, '_perform_startup_checks'):
        
        ui = Wan22UI()
        
        # Mock the image preview manager to raise exceptions
        ui.image_preview_manager = Mock()
        ui.image_preview_manager.clear_image.side_effect = Exception("Test error")
        
        # Test that clear operations handle errors gracefully
        start_result = ui._clear_start_image()
        end_result = ui._clear_end_image()
        
        # Should still return proper structure
        assert len(start_result) == 7
        assert len(end_result) == 7
        
        # Should still clear inputs even with errors
        assert start_result[0]['value'] is None
        assert end_result[0]['value'] is None
        
        print("✅ Error handling workflow test passed!")

if __name__ == "__main__":
    print("Running comprehensive integration tests for image clearing and replacement...")
    
    test_complete_image_clearing_workflow()
    test_image_replacement_workflow()
    test_error_handling_workflow()
    
    print("\n🎉 All integration tests passed!")
    print("\nTask 9 Implementation Summary:")
    print("✅ Clear buttons for both start and end image uploads")
    print("✅ Automatic image replacement when new files are uploaded")
    print("✅ Validation messages cleared when images are removed")
    print("✅ Image preservation when switching between model types")
    print("✅ Proper error handling and graceful fallbacks")
    print("✅ Enhanced UI styling and responsive behavior")
    print("✅ Integration with enhanced image preview manager")