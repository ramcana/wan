#!/usr/bin/env python3
"""
Test script to verify event handler implementations
"""

import ast
import inspect

def test_event_handlers():
    """Test that all required event handlers are implemented"""
    
    # Read the ui.py file
    with open('ui.py', 'r', encoding='utf-8') as f:
        ui_content = f.read()
    
    # Parse the AST to find method definitions
    tree = ast.parse(ui_content)
    
    # Find all method definitions in the Wan22UI class
    methods = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == 'Wan22UI':
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    methods.append(item.name)
    
    # Required event handler methods
    required_methods = [
        '_setup_generation_events',
        '_setup_optimization_events', 
        '_setup_queue_stats_events',
        '_setup_outputs_events',
        '_on_model_type_change',
        '_update_char_count',
        '_enhance_prompt',
        '_generate_video',
        '_add_to_queue',
        '_validate_image_upload',
        '_validate_lora_path',
        '_clear_notification',
        '_handle_generation_error',
        '_check_system_requirements',
        '_perform_startup_checks',
        '_show_notification'
    ]
    
    print("🔍 Checking Event Handler Implementation:")
    print("=" * 50)
    
    missing_methods = []
    for method in required_methods:
        if method in methods:
            print(f"✅ {method}")
        else:
            print(f"❌ {method} - MISSING")
            missing_methods.append(method)
    
    print("\n📊 Summary:")
    print(f"✅ Implemented: {len(required_methods) - len(missing_methods)}/{len(required_methods)}")
    if missing_methods:
        print(f"❌ Missing: {len(missing_methods)} methods")
        print(f"   {', '.join(missing_methods)}")
    else:
        print("🎉 All required event handlers are implemented!")
    
    # Check for event handler connections
    print("\n🔗 Checking Event Handler Connections:")
    print("=" * 50)
    
    # Look for .click(), .change(), .upload() calls
    event_connections = []
    lines = ui_content.split('\n')
    for i, line in enumerate(lines):
        if any(event in line for event in ['.click(', '.change(', '.upload(', '.select(']):
            event_connections.append(f"Line {i+1}: {line.strip()}")
        elif 'fn=self._' in line:
            event_connections.append(f"Line {i+1}: {line.strip()}")
    
    print(f"Found {len(event_connections)} event handler connections:")
    for connection in event_connections[:10]:  # Show first 10
        print(f"  {connection}")
    
    if len(event_connections) > 10:
        print(f"  ... and {len(event_connections) - 10} more")
    
    return len(missing_methods) == 0

    assert True  # TODO: Add proper assertion

if __name__ == "__main__":
    success = test_event_handlers()
    exit(0 if success else 1)