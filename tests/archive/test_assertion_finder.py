#!/usr/bin/env python3
"""
Simple test to find test functions without assertions.
"""

import ast
from pathlib import Path

def main():
    print("🚀 Starting simple test assertion finder...")
    
    project_root = Path.cwd()
    print(f"📁 Working directory: {project_root}")
    
    # Find Python files
    python_files = list(project_root.rglob('*.py'))
    print(f"📄 Found {len(python_files)} Python files")
    
    # Filter to test files
    test_files = [f for f in python_files if 'test' in str(f).lower()]
    print(f"🧪 Found {len(test_files)} test files")
    
    # Show first few test files
    for i, test_file in enumerate(test_files[:5]):
        print(f"  {i+1}. {test_file.relative_to(project_root)}")
    
    if len(test_files) > 5:
        print(f"  ... and {len(test_files) - 5} more")
    
    # Analyze first test file
    if test_files:
        first_test_file = test_files[0]
        print(f"\n🔍 Analyzing: {first_test_file.relative_to(project_root)}")
        
        try:
            with open(first_test_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            print(f"📝 File size: {len(content)} characters")
            
            try:
                tree = ast.parse(content)
                print("✅ File parsed successfully")
                
                # Find test functions
                test_functions = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                        test_functions.append(node.name)
                
                print(f"🧪 Found {len(test_functions)} test functions:")
                for func in test_functions[:3]:
                    print(f"  - {func}")
                
            except SyntaxError as e:
                print(f"❌ Syntax error: {e}")
                
        except Exception as e:
            print(f"❌ Error reading file: {e}")

if __name__ == "__main__":
    main()
