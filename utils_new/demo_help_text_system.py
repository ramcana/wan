"""
Demo script for the comprehensive help text and guidance system
Showcases all the features and functionality of the help system
"""

import sys
import os

# Add the current directory to the path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from help_text_system import (
        HelpTextSystem,
        get_help_system,
        get_model_help_text,
        get_image_help_text,
        get_tooltip_text,
        get_context_sensitive_help
    )
except ImportError as e:
    print(f"❌ Error: Could not import help_text_system: {e}")
    sys.exit(1)


def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print('='*60)


def print_subsection(title: str):
    """Print a formatted subsection header"""
    print(f"\n📋 {title}")
    print('-'*40)


def demo_model_help_text():
    """Demonstrate model-specific help text"""
    print_section("Model-Specific Help Text")
    
    models = ["t2v-A14B", "i2v-A14B", "ti2v-5B"]
    
    for model in models:
        print_subsection(f"{model} Help Text")
        
        # Desktop version
        desktop_help = get_model_help_text(model, mobile=False)
        print("🖥️ Desktop Version:")
        print(desktop_help)
        
        # Mobile version
        mobile_help = get_model_help_text(model, mobile=True)
        print(f"\n📱 Mobile Version:")
        print(mobile_help)
        
        print(f"\n📊 Stats: Desktop: {len(desktop_help)} chars, Mobile: {len(mobile_help)} chars")


def demo_image_help_text():
    """Demonstrate image upload help text"""
    print_section("Image Upload Help Text")
    
    models = ["t2v-A14B", "i2v-A14B", "ti2v-5B"]
    
    for model in models:
        print_subsection(f"{model} Image Help")
        
        image_help = get_image_help_text(model)
        if image_help:
            print(image_help)
        else:
            print("ℹ️ No image help needed for this model type")
        
        print(f"📊 Length: {len(image_help)} characters")


def demo_tooltip_system():
    """Demonstrate tooltip functionality"""
    print_section("Tooltip System")
    
    elements = [
        "model_type", "prompt", "resolution", "steps", 
        "duration", "fps", "image_upload_area", "clear_image"
    ]
    
    for element in elements:
        tooltip = get_tooltip_text(element)
        print(f"🏷️ {element}: {tooltip}")
    
    print_subsection("Image-Specific Tooltips")
    
    help_system = get_help_system()
    for image_type in ["start_image", "end_image"]:
        tooltip = help_system.get_image_upload_tooltip(image_type)
        print(f"📸 {image_type}: {tooltip}")


def demo_context_sensitive_help():
    """Demonstrate context-sensitive help"""
    print_section("Context-Sensitive Help")
    
    models = ["t2v-A14B", "i2v-A14B", "ti2v-5B"]
    states = [
        (False, False, "No images uploaded"),
        (True, False, "Start image only"),
        (True, True, "Both images uploaded")
    ]
    
    for model in models:
        print_subsection(f"{model} Context Help")
        
        for has_start, has_end, description in states:
            print(f"\n🔄 State: {description}")
            context_help = get_context_sensitive_help(model, has_start, has_end)
            
            # Show first 200 characters
            preview = context_help[:200] + "..." if len(context_help) > 200 else context_help
            print(preview)
            print(f"📊 Full length: {len(context_help)} characters")


def demo_requirements_and_examples():
    """Demonstrate requirements and examples"""
    print_section("Requirements and Examples")
    
    help_system = get_help_system()
    
    contexts = ["t2v-A14B", "i2v-A14B", "ti2v-5B", "start_image", "end_image"]
    
    for context in contexts:
        print_subsection(f"{context} Requirements")
        
        requirements = help_system.get_requirements_list(context)
        if requirements:
            for i, req in enumerate(requirements, 1):
                print(f"   {i}. {req}")
        else:
            print("   ℹ️ No specific requirements")
        
        # Show examples for model types
        if context in ["t2v-A14B", "i2v-A14B", "ti2v-5B"]:
            examples = help_system.get_examples_list(context)
            if examples:
                print(f"\n💡 Examples:")
                for i, example in enumerate(examples, 1):
                    print(f"   {i}. {example}")


def demo_error_help():
    """Demonstrate error help system"""
    print_section("Error Help System")
    
    help_system = get_help_system()
    error_types = ["invalid_format", "too_small", "aspect_mismatch", "file_too_large"]
    
    for error_type in error_types:
        print_subsection(f"{error_type.replace('_', ' ').title()} Error")
        
        error_help = help_system.get_error_help(error_type)
        print(f"❌ Message: {error_help['message']}")
        print(f"💡 Suggestions:")
        for i, suggestion in enumerate(error_help['suggestions'], 1):
            print(f"   {i}. {suggestion}")


def demo_html_formatting():
    """Demonstrate HTML formatting capabilities"""
    print_section("HTML Formatting")
    
    help_system = get_help_system()
    
    # Test content with markdown-style formatting
    test_content = """
**Bold text** and *italic text* with
multiple lines and formatting.

This demonstrates the HTML conversion.
    """
    
    print_subsection("Original Content")
    print(repr(test_content))
    
    print_subsection("Formatted HTML")
    html = help_system.format_help_html(test_content, "Test Title", "custom-class")
    print(html)
    
    print_subsection("Tooltip HTML")
    tooltip_html = help_system.create_tooltip_html("Hover me", "This is a tooltip")
    print(tooltip_html)


def demo_responsive_css():
    """Demonstrate responsive CSS generation"""
    print_section("Responsive CSS")
    
    help_system = get_help_system()
    css = help_system.get_responsive_help_css()
    
    print(f"📊 CSS Length: {len(css)} characters")
    print(f"📱 Contains responsive queries: {'@media' in css}")
    print(f"🎨 Contains tooltip styles: {'.help-tooltip' in css}")
    print(f"📋 Contains help content styles: {'.help-content' in css}")
    
    # Show first few lines
    print_subsection("CSS Preview (first 500 characters)")
    print(css[:500] + "...")


def demo_integration_functions():
    """Demonstrate integration with UI"""
    print_section("UI Integration Functions")
    
    print_subsection("Global Function Tests")
    
    # Test all global functions
    functions = [
        ("get_model_help_text", lambda: get_model_help_text("t2v-A14B")),
        ("get_image_help_text", lambda: get_image_help_text("i2v-A14B")),
        ("get_tooltip_text", lambda: get_tooltip_text("model_type")),
        ("get_context_sensitive_help", lambda: get_context_sensitive_help("ti2v-5B", True, False))
    ]
    
    for func_name, func in functions:
        try:
            result = func()
            print(f"✅ {func_name}: {len(result)} characters")
        except Exception as e:
            print(f"❌ {func_name}: Error - {e}")
    
    print_subsection("Singleton Pattern Test")
    system1 = get_help_system()
    system2 = get_help_system()
    print(f"✅ Singleton working: {system1 is system2}")


def demo_performance_stats():
    """Show performance statistics"""
    print_section("Performance Statistics")
    
    help_system = get_help_system()
    
    # Count total content
    total_chars = 0
    total_items = 0
    
    # Model help
    for model in ["t2v-A14B", "i2v-A14B", "ti2v-5B"]:
        help_text = help_system.get_model_help_text(model)
        total_chars += len(help_text)
        total_items += 1
    
    # Image help
    for model in ["i2v-A14B", "ti2v-5B"]:
        help_text = help_system.get_image_help_text(model)
        total_chars += len(help_text)
        total_items += 1
    
    # Tooltips
    elements = ["model_type", "prompt", "resolution", "steps", "duration", "fps"]
    for element in elements:
        tooltip = help_system.get_tooltip_text(element)
        total_chars += len(tooltip)
        total_items += 1
    
    print(f"📊 Total help content: {total_chars:,} characters")
    print(f"📋 Total help items: {total_items}")
    print(f"📈 Average item length: {total_chars // total_items} characters")
    
    # CSS stats
    css = help_system.get_responsive_help_css()
    print(f"🎨 CSS size: {len(css):,} characters")
    
    # Memory usage estimate (rough)
    estimated_memory = (total_chars + len(css)) * 2  # Rough estimate
    print(f"💾 Estimated memory usage: ~{estimated_memory / 1024:.1f} KB")


def main():
    """Run all demonstrations"""
    print("🎬 Comprehensive Help Text System Demo")
    print("This demo showcases all features of the help text and guidance system")
    
    try:
        demo_model_help_text()
        demo_image_help_text()
        demo_tooltip_system()
        demo_context_sensitive_help()
        demo_requirements_and_examples()
        demo_error_help()
        demo_html_formatting()
        demo_responsive_css()
        demo_integration_functions()
        demo_performance_stats()
        
        print_section("Demo Complete")
        print("🎉 All help system features demonstrated successfully!")
        print("✅ The comprehensive help text system is ready for use in the UI")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
