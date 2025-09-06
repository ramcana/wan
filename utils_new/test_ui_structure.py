"""
Test script to verify UI structure without heavy dependencies
"""

import ast
import sys

def test_ui_structure():
    """Test that the UI file has correct structure"""
    
    try:
        # Parse the UI file
        with open('ui.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse AST to check syntax
        tree = ast.parse(content)
        
        # Check for main class
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        wan22ui_class = None
        
        for cls in classes:
            if cls.name == 'Wan22UI':
                wan22ui_class = cls
                break
        
        if not wan22ui_class:
            print("❌ Wan22UI class not found")
            return False
        
        # Check for required methods
        methods = [node.name for node in ast.walk(wan22ui_class) if isinstance(node, ast.FunctionDef)]
        
        required_methods = [
            '__init__',
            '_create_interface',
            '_create_generation_tab',
            '_create_optimization_tab', 
            '_create_queue_stats_tab',
            '_create_outputs_tab',
            '_setup_generation_events',
            '_setup_optimization_events',
            '_setup_queue_stats_events',
            '_setup_outputs_events',
            'launch'
        ]
        
        missing_methods = []
        for method in required_methods:
            if method not in methods:
                missing_methods.append(method)
        
        if missing_methods:
            print(f"❌ Missing methods: {missing_methods}")
            return False
        
        print("✅ UI structure validation passed")
        print(f"✅ Found Wan22UI class with {len(methods)} methods")
        print("✅ All required methods present")
        
        # Check for component storage
        component_stores = [
            'generation_components',
            'optimization_components', 
            'queue_stats_components',
            'outputs_components'
        ]
        
        content_lower = content.lower()
        for store in component_stores:
            if store.lower() in content_lower:
                print(f"✅ Found {store}")
            else:
                print(f"⚠️ {store} not found in content")
        
        return True
        
    except SyntaxError as e:
        print(f"❌ Syntax error in UI file: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing UI structure: {e}")
        return False

    assert True  # TODO: Add proper assertion

if __name__ == "__main__":
    success = test_ui_structure()
    sys.exit(0 if success else 1)