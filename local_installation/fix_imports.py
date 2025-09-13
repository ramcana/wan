#!/usr/bin/env python3
"""
Quick script to fix relative imports in the scripts directory.
This converts all relative imports (from .module) to absolute imports (from module).
"""

import os
import re
from pathlib import Path

def fix_relative_imports(file_path):
    """Fix relative imports in a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Pattern to match relative imports like "from .module import something"
        pattern = r'from \.(\w+) import'
        replacement = r'from \1 import'
        
        # Replace relative imports with absolute imports
        new_content = re.sub(pattern, replacement, content)
        
        # Check if any changes were made
        if new_content != content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"✅ Fixed imports in: {file_path}")
            return True
        else:
            print(f"⚪ No changes needed: {file_path}")
            return False
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return False

def main():
    """Fix all relative imports in the scripts directory."""
    print("🔧 Fixing relative imports in scripts directory...")
    
    scripts_dir = Path(__file__).parent / "scripts"
    
    if not scripts_dir.exists():
        print(f"❌ Scripts directory not found: {scripts_dir}")
        return
    
    fixed_count = 0
    total_count = 0
    
    # Process all Python files in the scripts directory
    for py_file in scripts_dir.glob("*.py"):
        if py_file.name == "__init__.py":
            continue
            
        total_count += 1
        if fix_relative_imports(py_file):
            fixed_count += 1
    
    print(f"\n📊 Summary:")
    print(f"   Total files processed: {total_count}")
    print(f"   Files with fixes: {fixed_count}")
    print(f"   Files unchanged: {total_count - fixed_count}")
    
    if fixed_count > 0:
        print(f"\n✅ Import fixes completed! You can now run install.bat")
    else:
        print(f"\n⚪ No import fixes needed.")

if __name__ == "__main__":
    main()
