#!/usr/bin/env python3
"""
Test Help Text System Integration
Tests the comprehensive help text and guidance system integration with the UI
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_help_text_system_import():
    """Test that help text system can be imported"""
    try:
        from help_text_system import HelpTextSystem, get_help_system
        assert HelpTextSystem is not None
        assert get_help_system is not None
        print("✅ Help text system imports successfully")
    except ImportError as e:
        pytest.fail(f"Failed to import help text system: {e}")

def test_help_text_system_initialization():
    """Test help text system initialization"""
    try:
        from help_text_system import HelpTextSystem
        
        help_system = HelpTextSystem()
        assert help_system is not None
        assert hasattr(help_system, 'model_help')
        assert hasattr(help_system, 'image_help')
        assert hasattr(help_system, 'tooltips')
        assert hasattr(help_system, 'error_help')
        
        print("✅ Help text system initializes correctly")
    except Exception as e:
        pytest.fail(f"Failed to initialize help text system: {e}")

def test_model_help_content():
    """Test model-specific help content"""
    try:
        from help_text_system import HelpTextSystem
        
        help_system = HelpTextSystem()
        
        # Test all model types
        model_types = ["t2v-A14B", "i2v-A14B", "ti2v-5B"]
        
        for model_type in model_types:
            help_text = help_system.get_model_help_text(model_type)
            assert help_text is not None
            assert len(help_text) > 0
            # Check for model type indicators in the help text
            model_indicators = [model_type, model_type.upper(), model_type.replace("-", ""), "T2V", "I2V", "TI2V"]
            assert any(indicator in help_text for indicator in model_indicators), f"No model indicator found in help text for {model_type}"
            
            # Test mobile version
            mobile_help = help_system.get_model_help_text(model_type, mobile=True)
            assert mobile_help is not None
            
            print(f"✅ Model help text for {model_type}: {len(help_text)} chars")
        
        print("✅ All model help content is available")
    except Exception as e:
        pytest.fail(f"Failed to get model help content: {e}")

def test_image_help_content():
    """Test image-specific help content"""
    try:
        from help_text_system import HelpTextSystem
        
        help_system = HelpTextSystem()
        
        # Test image help for different model types
        model_types = ["t2v-A14B", "i2v-A14B", "ti2v-5B"]
        
        for model_type in model_types:
            help_text = help_system.get_image_help_text(model_type)
            
            if model_type == "t2v-A14B":
                # T2V should have no image help
                assert help_text == ""
            else:
                # I2V and TI2V should have image help
                assert len(help_text) > 0
                assert "image" in help_text.lower()
            
            print(f"✅ Image help text for {model_type}: {len(help_text)} chars")
        
        print("✅ All image help content is appropriate")
    except Exception as e:
        pytest.fail(f"Failed to get image help content: {e}")

def test_tooltip_content():
    """Test tooltip content for UI elements"""
    try:
        from help_text_system import HelpTextSystem
        
        help_system = HelpTextSystem()
        
        # Test common UI element tooltips
        elements = [
            "model_type", "prompt", "resolution", "steps", 
            "image_upload_area", "start_image", "end_image",
            "clear_image", "image_preview"
        ]
        
        for element in elements:
            tooltip = help_system.get_tooltip_text(element)
            assert tooltip is not None
            if len(tooltip) == 0:
                print(f"⚠️ Empty tooltip for {element} - this may be expected")
            else:
                print(f"✅ Tooltip for {element}: {tooltip[:50]}...")
        
        # Ensure at least some tooltips are present
        non_empty_tooltips = [help_system.get_tooltip_text(e) for e in elements if len(help_system.get_tooltip_text(e)) > 0]
        assert len(non_empty_tooltips) >= 6, f"Expected at least 6 non-empty tooltips, got {len(non_empty_tooltips)}"
        
        print("✅ All tooltip content is available")
    except Exception as e:
        pytest.fail(f"Failed to get tooltip content: {e}")

def test_error_help_content():
    """Test error help content with recovery suggestions"""
    try:
        from help_text_system import HelpTextSystem
        
        help_system = HelpTextSystem()
        
        # Test common error types
        error_types = [
            "invalid_format", "too_small", "aspect_mismatch", "file_too_large"
        ]
        
        for error_type in error_types:
            error_help = help_system.get_error_help(error_type)
            assert error_help is not None
            assert "message" in error_help
            assert "suggestions" in error_help
            assert len(error_help["suggestions"]) > 0
            
            print(f"✅ Error help for {error_type}: {error_help['message']}")
        
        print("✅ All error help content is available")
    except Exception as e:
        pytest.fail(f"Failed to get error help content: {e}")

def test_context_sensitive_help():
    """Test context-sensitive help that adapts to current state"""
    try:
        from help_text_system import HelpTextSystem
        
        help_system = HelpTextSystem()
        
        # Test different contexts
        contexts = [
            ("t2v-A14B", False, False),
            ("i2v-A14B", False, False),
            ("i2v-A14B", True, False),
            ("ti2v-5B", True, True)
        ]
        
        for model_type, has_start, has_end in contexts:
            help_text = help_system.get_context_sensitive_help(
                model_type, has_start, has_end
            )
            assert help_text is not None
            assert len(help_text) > 0
            
            # Test mobile version
            mobile_help = help_system.get_context_sensitive_help(
                model_type, has_start, has_end, mobile=True
            )
            assert mobile_help is not None
            
            print(f"✅ Context help for {model_type} (start:{has_start}, end:{has_end}): {len(help_text)} chars")
        
        print("✅ Context-sensitive help works correctly")
    except Exception as e:
        pytest.fail(f"Failed to get context-sensitive help: {e}")

def test_responsive_css():
    """Test responsive CSS generation"""
    try:
        from help_text_system import HelpTextSystem
        
        help_system = HelpTextSystem()
        
        css = help_system.get_responsive_help_css()
        assert css is not None
        assert len(css) > 0
        assert ".help-content" in css
        assert "@media" in css
        assert "mobile" in css.lower()
        
        print(f"✅ Responsive CSS generated: {len(css)} chars")
    except Exception as e:
        pytest.fail(f"Failed to generate responsive CSS: {e}")

def test_html_formatting():
    """Test HTML formatting for help content"""
    try:
        from help_text_system import HelpTextSystem
        
        help_system = HelpTextSystem()
        
        test_content = "**Bold text** and *italic text* with\nnew lines"
        html = help_system.format_help_html(test_content, "Test Title")
        
        assert html is not None
        assert "<div class=" in html
        assert "<strong>" in html
        assert "<em>" in html
        assert "<br>" in html
        assert "Test Title" in html
        
        print(f"✅ HTML formatting works: {len(html)} chars")
    except Exception as e:
        pytest.fail(f"Failed to format HTML: {e}")

def test_tooltip_html_creation():
    """Test tooltip HTML creation"""
    try:
        from help_text_system import HelpTextSystem
        
        help_system = HelpTextSystem()
        
        tooltip_html = help_system.create_tooltip_html("Hover text", "Tooltip content")
        
        assert tooltip_html is not None
        assert "help-tooltip" in tooltip_html
        assert "tooltip-text" in tooltip_html
        assert "Hover text" in tooltip_html
        assert "Tooltip content" in tooltip_html
        
        print(f"✅ Tooltip HTML creation works: {len(tooltip_html)} chars")
    except Exception as e:
        pytest.fail(f"Failed to create tooltip HTML: {e}")

def test_ui_integration_methods():
    """Test UI integration methods without full UI instantiation"""
    try:
        # Test the help text functions directly
        from help_text_system import get_model_help_text, get_image_help_text, get_tooltip_text
        
        # Test model help text
        model_help = get_model_help_text("i2v-A14B")
        assert model_help is not None
        assert len(model_help) > 0
        
        # Test image help text
        image_help = get_image_help_text("i2v-A14B")
        assert image_help is not None
        assert len(image_help) > 0
        
        # Test tooltip text
        tooltip = get_tooltip_text("start_image")
        assert tooltip is not None
        assert len(tooltip) > 0
        
        # Test context-sensitive help
        from help_text_system import get_context_sensitive_help
        context_help = get_context_sensitive_help("i2v-A14B", True, False)
        assert context_help is not None
        assert len(context_help) > 0
        
        print("✅ UI integration methods work correctly")
    except Exception as e:
        pytest.fail(f"Failed UI integration test: {e}")

def test_requirements_compliance():
    """Test compliance with requirements 4.1-4.4"""
    try:
        from help_text_system import HelpTextSystem
        
        help_system = HelpTextSystem()
        
        # Requirement 4.1: Display help text explaining image requirements
        for model_type in ["i2v-A14B", "ti2v-5B"]:
            help_text = help_system.get_image_help_text(model_type)
            # Check for image-related keywords
            image_keywords = ["image", "upload", "png", "jpg", "format", "requirement", "size", "pixel"]
            assert any(keyword in help_text.lower() for keyword in image_keywords), f"No image requirements found in help text for {model_type}"
            print(f"✅ Requirement 4.1: Help text for {model_type} explains requirements")
        
        # Requirement 4.2: Tooltips with format and size requirements
        start_tooltip = help_system.get_tooltip_text("start_image")
        end_tooltip = help_system.get_tooltip_text("end_image")
        assert "256x256" in start_tooltip or "format" in start_tooltip.lower()
        assert "aspect ratio" in end_tooltip.lower() or "format" in end_tooltip.lower()
        print("✅ Requirement 4.2: Tooltips include format and size requirements")
        
        # Requirement 4.3: Specific guidance for validation errors
        for error_type in ["invalid_format", "too_small", "aspect_mismatch"]:
            error_help = help_system.get_error_help(error_type)
            assert len(error_help["suggestions"]) > 0
            print(f"✅ Requirement 4.3: Specific guidance for {error_type}")
        
        # Requirement 4.4: Display aspect ratio and resolution information
        # This is tested through the image requirements text
        start_req = help_system.image_help["start_image"].content
        assert "aspect ratio" in start_req.lower() or "resolution" in start_req.lower()
        print("✅ Requirement 4.4: Information about aspect ratio and resolution")
        
        print("✅ All requirements 4.1-4.4 are satisfied")
    except Exception as e:
        pytest.fail(f"Requirements compliance test failed: {e}")

if __name__ == "__main__":
    print("🧪 Testing Help Text System Integration...")
    
    # Run all tests
    test_functions = [
        test_help_text_system_import,
        test_help_text_system_initialization,
        test_model_help_content,
        test_image_help_content,
        test_tooltip_content,
        test_error_help_content,
        test_context_sensitive_help,
        test_responsive_css,
        test_html_formatting,
        test_tooltip_html_creation,
        test_ui_integration_methods,
        test_requirements_compliance
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            print(f"\n🔍 Running {test_func.__name__}...")
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} failed: {e}")
            failed += 1
    
    print(f"\n📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All help text integration tests passed!")
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
        sys.exit(1)