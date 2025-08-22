"""
Test script for Enhanced Image Preview and Management System
Tests thumbnail generation, clear/remove functionality, and hover tooltips
"""

import pytest
import tempfile
import os
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

# Test the enhanced image preview manager
def test_enhanced_image_preview_manager():
    """Test the enhanced image preview manager functionality"""
    try:
        from enhanced_image_preview_manager import EnhancedImagePreviewManager, ImagePreviewData
        
        # Test initialization
        config = {
            "thumbnail_size": (150, 150),
            "enable_hover_tooltips": True,
            "enable_image_replacement": True
        }
        
        manager = EnhancedImagePreviewManager(config)
        
        # Test initial state
        assert manager.start_image_data.image is None
        assert manager.end_image_data.image is None
        
        print("✅ Enhanced image preview manager initialization test passed")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Enhanced image preview manager test failed: {e}")
        return False

def test_image_preview_data():
    """Test the ImagePreviewData dataclass"""
    try:
        from enhanced_image_preview_manager import ImagePreviewData
        
        # Test creation
        data = ImagePreviewData(
            filename="test.jpg",
            format="JPEG",
            dimensions=(1280, 720),
            file_size_mb=2.5,
            aspect_ratio=16/9,
            upload_timestamp=datetime.now()
        )
        
        # Test aspect ratio string
        assert "16:9" in data.aspect_ratio_string
        
        # Test dictionary conversion
        data_dict = data.to_dict()
        assert data_dict["filename"] == "test.jpg"
        assert data_dict["format"] == "JPEG"
        assert data_dict["dimensions"] == (1280, 720)
        
        print("✅ ImagePreviewData test passed")
        return True
        
    except Exception as e:
        print(f"❌ ImagePreviewData test failed: {e}")
        return False

def test_mock_image_processing():
    """Test image processing with mock PIL Image"""
    try:
        from enhanced_image_preview_manager import EnhancedImagePreviewManager
        
        # Create mock image
        mock_image = Mock()
        mock_image.size = (1280, 720)
        mock_image.format = "JPEG"
        mock_image.mode = "RGB"
        mock_image.filename = "test_image.jpg"
        
        # Mock PIL operations
        with patch('enhanced_image_preview_manager.PIL_AVAILABLE', True):
            manager = EnhancedImagePreviewManager()
            
            # Test processing without actual PIL
            preview_html, tooltip_data, visible = manager.process_image_upload(None, "start")
            
            # Should handle None image gracefully
            assert not visible
            assert "No Image" in preview_html or "empty" in preview_html.lower()
            
        print("✅ Mock image processing test passed")
        return True
        
    except Exception as e:
        print(f"❌ Mock image processing test failed: {e}")
        return False

def test_clear_functionality():
    """Test image clearing functionality"""
    try:
        from enhanced_image_preview_manager import EnhancedImagePreviewManager, ImagePreviewData
        
        manager = EnhancedImagePreviewManager()
        
        # Set some mock data
        manager.start_image_data = ImagePreviewData(
            filename="test.jpg",
            format="JPEG",
            dimensions=(1280, 720),
            file_size_mb=2.5,
            aspect_ratio=16/9,
            upload_timestamp=datetime.now()
        )
        
        # Test clearing
        preview_html, tooltip_data, visible = manager.clear_image("start")
        
        # Should reset the data
        assert manager.start_image_data.image is None
        assert manager.start_image_data.filename == ""
        
        print("✅ Clear functionality test passed")
        return True
        
    except Exception as e:
        print(f"❌ Clear functionality test failed: {e}")
        return False

def test_image_summary():
    """Test image summary generation"""
    try:
        from enhanced_image_preview_manager import EnhancedImagePreviewManager, ImagePreviewData
        
        manager = EnhancedImagePreviewManager()
        
        # Test with no images
        summary = manager.get_image_summary()
        assert "No images" in summary
        
        # Test with start image
        mock_image = Mock()
        mock_image.size = (1280, 720)
        
        manager.start_image_data = ImagePreviewData(
            image=mock_image,
            filename="start.jpg",
            format="JPEG",
            dimensions=(1280, 720),
            file_size_mb=2.5,
            aspect_ratio=16/9,
            upload_timestamp=datetime.now()
        )
        
        summary = manager.get_image_summary()
        assert "Start:" in summary
        assert "1280×720" in summary
        
        print("✅ Image summary test passed")
        return True
        
    except Exception as e:
        print(f"❌ Image summary test failed: {e}")
        return False

def test_compatibility_status():
    """Test image compatibility checking"""
    try:
        from enhanced_image_preview_manager import EnhancedImagePreviewManager, ImagePreviewData
        
        manager = EnhancedImagePreviewManager()
        
        # Test with no images
        status = manager.get_compatibility_status()
        assert status == ""
        
        # Test with matching images
        mock_image1 = Mock()
        mock_image1.size = (1280, 720)
        mock_image2 = Mock()
        mock_image2.size = (1280, 720)
        
        manager.start_image_data = ImagePreviewData(
            image=mock_image1,
            dimensions=(1280, 720),
            aspect_ratio=16/9
        )
        
        manager.end_image_data = ImagePreviewData(
            image=mock_image2,
            dimensions=(1280, 720),
            aspect_ratio=16/9
        )
        
        status = manager.get_compatibility_status()
        assert "compatible" in status.lower()
        
        # Test with mismatched dimensions
        manager.end_image_data.dimensions = (1920, 1080)
        manager.end_image_data.aspect_ratio = 16/9
        
        status = manager.get_compatibility_status()
        assert "different dimensions" in status.lower() or "⚠️" in status
        
        print("✅ Compatibility status test passed")
        return True
        
    except Exception as e:
        print(f"❌ Compatibility status test failed: {e}")
        return False

def test_ui_integration():
    """Test UI integration components"""
    try:
        from enhanced_image_preview_manager import create_image_preview_components
        
        components = create_image_preview_components()
        
        # Check that all required components are created
        required_components = [
            'start_image_preview',
            'end_image_preview',
            'image_summary',
            'compatibility_status',
            'clear_start_btn',
            'clear_end_btn'
        ]
        
        for component_name in required_components:
            assert component_name in components, f"Missing component: {component_name}"
        
        print("✅ UI integration test passed")
        return True
        
    except Exception as e:
        print(f"❌ UI integration test failed: {e}")
        return False

def test_javascript_functionality():
    """Test JavaScript functionality (basic validation)"""
    try:
        # The JavaScript is embedded in the UI file, so we'll check for its presence there
        with open('ui.py', 'r', encoding='utf-8') as f:
            ui_content = f.read()
        
        # Check that JavaScript contains required functions
        required_functions = [
            "clearImage",
            "showLargePreview",
            "showTooltip",
            "hideTooltip"
        ]
        
        for func_name in required_functions:
            assert func_name in ui_content, f"Missing JavaScript function: {func_name}"
        
        # Check for event listeners
        assert "addEventListener" in ui_content
        assert "MutationObserver" in ui_content
        
        print("✅ JavaScript functionality test passed")
        return True
        
    except Exception as e:
        print(f"❌ JavaScript functionality test failed: {e}")
        return False

def run_all_tests():
    """Run all enhanced image preview tests"""
    print("🧪 Running Enhanced Image Preview and Management Tests")
    print("=" * 60)
    
    tests = [
        test_enhanced_image_preview_manager,
        test_image_preview_data,
        test_mock_image_processing,
        test_clear_functionality,
        test_image_summary,
        test_compatibility_status,
        test_ui_integration,
        test_javascript_functionality
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
    
    print("=" * 60)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All enhanced image preview tests passed!")
        return True
    else:
        print(f"⚠️ {failed} tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)