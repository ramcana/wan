"""
Validation script for Enhanced Image Preview Implementation
Validates that all required components and functionality are properly implemented
"""

import os
import sys
from pathlib import Path

def validate_enhanced_image_preview_manager():
    """Validate the enhanced image preview manager module"""
    print("🔍 Validating Enhanced Image Preview Manager...")
    
    try:
        from enhanced_image_preview_manager import (
            EnhancedImagePreviewManager,
            ImagePreviewData,
            create_image_preview_components,
            get_preview_manager
        )
        
        # Test basic functionality
        manager = get_preview_manager()
        components = create_image_preview_components()
        
        # Validate required components
        required_components = [
            'start_image_preview',
            'end_image_preview', 
            'image_summary',
            'compatibility_status',
            'clear_start_btn',
            'clear_end_btn'
        ]
        
        for component in required_components:
            if component not in components:
                print(f"❌ Missing component: {component}")
                return False
        
        print("✅ Enhanced Image Preview Manager validation passed")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False

def validate_ui_integration():
    """Validate UI integration"""
    print("🔍 Validating UI Integration...")
    
    try:
        # Read UI file
        with open('ui.py', 'r', encoding='utf-8') as f:
            ui_content = f.read()
        
        # Check for required UI components
        ui_components = [
            'start_image_preview',
            'end_image_preview',
            'image_summary',
            'compatibility_status',
            'image_status_row',
            'clear_start_btn',
            'clear_end_btn'
        ]
        
        for component in ui_components:
            if f"'{component}'" not in ui_content:
                print(f"❌ Missing UI component: {component}")
                return False
        
        # Check for enhanced handlers
        handlers = [
            '_handle_start_image_upload',
            '_handle_end_image_upload',
            '_clear_start_image',
            '_clear_end_image'
        ]
        
        for handler in handlers:
            if f"def {handler}" not in ui_content:
                print(f"❌ Missing handler: {handler}")
                return False
        
        # Check for image preview manager initialization
        if 'image_preview_manager' not in ui_content:
            print("❌ Missing image preview manager initialization")
            return False
        
        print("✅ UI Integration validation passed")
        return True
        
    except Exception as e:
        print(f"❌ UI validation error: {e}")
        return False

def validate_css_and_javascript():
    """Validate CSS and JavaScript implementation"""
    print("🔍 Validating CSS and JavaScript...")
    
    try:
        with open('ui.py', 'r', encoding='utf-8') as f:
            ui_content = f.read()
        
        # Check for CSS classes
        css_classes = [
            'image-preview-display',
            'image-preview-container',
            'image-preview-empty',
            'image-summary-display',
            'compatibility-status-display'
        ]
        
        for css_class in css_classes:
            if css_class not in ui_content:
                print(f"❌ Missing CSS class: {css_class}")
                return False
        
        # Check for JavaScript functions
        js_functions = [
            'clearImage',
            'showLargePreview',
            'showTooltip',
            'hideTooltip'
        ]
        
        for js_function in js_functions:
            if f"function {js_function}" not in ui_content:
                print(f"❌ Missing JavaScript function: {js_function}")
                return False
        
        # Check for event listeners
        if 'addEventListener' not in ui_content:
            print("❌ Missing event listeners")
            return False
        
        if 'MutationObserver' not in ui_content:
            print("❌ Missing MutationObserver")
            return False
        
        print("✅ CSS and JavaScript validation passed")
        return True
        
    except Exception as e:
        print(f"❌ CSS/JS validation error: {e}")
        return False

def validate_event_handlers():
    """Validate event handler connections"""
    print("🔍 Validating Event Handlers...")
    
    try:
        with open('ui.py', 'r', encoding='utf-8') as f:
            ui_content = f.read()
        
        # Check for upload event handlers
        upload_handlers = [
            "image_input'].upload(",
            "end_image_input'].upload(",
            "_handle_start_image_upload",
            "_handle_end_image_upload"
        ]
        
        for handler in upload_handlers:
            if handler not in ui_content:
                print(f"❌ Missing upload handler: {handler}")
                return False
        
        # Check for clear button handlers
        clear_handlers = [
            "clear_start_btn'].click(",
            "clear_end_btn'].click(",
            "_clear_start_image",
            "_clear_end_image"
        ]
        
        for handler in clear_handlers:
            if handler not in ui_content:
                print(f"❌ Missing clear handler: {handler}")
                return False
        
        # Check for model type change handler updates
        if 'image_status_row' not in ui_content:
            print("❌ Missing image_status_row in model type handler")
            return False
        
        print("✅ Event Handlers validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Event handlers validation error: {e}")
        return False

def validate_requirements_coverage():
    """Validate that all requirements are covered"""
    print("🔍 Validating Requirements Coverage...")
    
    # Requirements from task 4:
    # - Add thumbnail display functionality for uploaded start and end images
    # - Create clear/remove buttons for each uploaded image  
    # - Implement image replacement functionality when new files are uploaded
    # - Add hover tooltips showing image dimensions and file information
    
    requirements_coverage = {
        "thumbnail_display": False,
        "clear_remove_buttons": False,
        "image_replacement": False,
        "hover_tooltips": False
    }
    
    try:
        # Check enhanced_image_preview_manager.py
        with open('enhanced_image_preview_manager.py', 'r', encoding='utf-8') as f:
            manager_content = f.read()
        
        # Check UI file
        with open('ui.py', 'r', encoding='utf-8') as f:
            ui_content = f.read()
        
        # Thumbnail display functionality
        if '_generate_thumbnail' in manager_content and 'thumbnail_data' in manager_content:
            requirements_coverage["thumbnail_display"] = True
        
        # Clear/remove buttons
        if 'clear_start_btn' in ui_content and 'clear_end_btn' in ui_content:
            requirements_coverage["clear_remove_buttons"] = True
        
        # Image replacement functionality
        if 'process_image_upload' in manager_content and 'enable_image_replacement' in manager_content:
            requirements_coverage["image_replacement"] = True
        
        # Hover tooltips
        if 'showTooltip' in ui_content and 'tooltip-data' in manager_content:
            requirements_coverage["hover_tooltips"] = True
        
        # Report coverage
        covered_count = sum(requirements_coverage.values())
        total = len(requirements_coverage)
        
        print(f"📊 Requirements Coverage: {covered_count}/{total}")
        
        for req, is_covered in requirements_coverage.items():
            status = "✅" if is_covered else "❌"
            print(f"  {status} {req.replace('_', ' ').title()}")
        
        if covered_count == total:
            print("✅ All requirements covered")
            return True
        else:
            print(f"❌ {total - covered_count} requirements not fully covered")
            return False
        
    except Exception as e:
        print(f"❌ Requirements validation error: {e}")
        return False

def validate_task_completion():
    """Validate overall task completion"""
    print("\n" + "="*60)
    print("📋 TASK 4 COMPLETION VALIDATION")
    print("="*60)
    
    validations = [
        validate_enhanced_image_preview_manager,
        validate_ui_integration,
        validate_css_and_javascript,
        validate_event_handlers,
        validate_requirements_coverage
    ]
    
    passed = 0
    failed = 0
    
    for validation in validations:
        try:
            if validation():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Validation {validation.__name__} failed: {e}")
            failed += 1
        print()
    
    print("="*60)
    print(f"📊 Validation Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 TASK 4 IMPLEMENTATION COMPLETE!")
        print("\n✅ All sub-tasks implemented:")
        print("  • Thumbnail display functionality")
        print("  • Clear/remove buttons for each image")
        print("  • Image replacement functionality")
        print("  • Hover tooltips with image information")
        print("\n🚀 Enhanced Image Preview and Management is ready!")
        return True
    else:
        print(f"⚠️ Task incomplete: {failed} validations failed")
        return False

if __name__ == "__main__":
    success = validate_task_completion()
    exit(0 if success else 1)
