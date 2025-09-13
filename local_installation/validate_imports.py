#!/usr/bin/env python3
"""Validate all installation script imports."""

import sys
import importlib.util
from pathlib import Path
import os

def test_module_import(module_path):
    """Test if a module can be imported successfully."""
    try:
        # Change to scripts directory for relative imports
        original_cwd = os.getcwd()
        scripts_dir = Path("scripts")
        os.chdir(scripts_dir)
        
        # Get just the filename for the module in the scripts directory
        module_name = module_path.name
        spec = importlib.util.spec_from_file_location("test_module", module_name)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        os.chdir(original_cwd)
        return True, None
    except Exception as e:
        os.chdir(original_cwd)
        return False, str(e)

def main():
    scripts_dir = Path("scripts")
    failed_imports = []
    successful_imports = []
    
    print("Validating all installation script imports...")
    print("=" * 50)
    
    for py_file in scripts_dir.glob("*.py"):
        if py_file.name.startswith("__"):
            continue
            
        success, error = test_module_import(py_file)
        if not success:
            failed_imports.append((py_file.name, error))
            print(f"FAIL {py_file.name}: {error}")
        else:
            successful_imports.append(py_file.name)
            print(f"OK   {py_file.name}: imports successfully")
    
    print("=" * 50)
    print(f"Results: {len(successful_imports)} successful, {len(failed_imports)} failed")
    
    if failed_imports:
        print(f"\n{len(failed_imports)} modules failed to import:")
        for filename, error in failed_imports:
            print(f"   - {filename}: {error}")
        sys.exit(1)
    else:
        print(f"\nAll {len(successful_imports)} modules imported successfully!")
        sys.exit(0)

if __name__ == "__main__":
    main()
