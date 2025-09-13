#!/usr/bin/env python3
"""
Fix relative imports in health-checker module
"""

import os
import re
from pathlib import Path

def fix_imports_in_file(file_path: Path):
    """Fix relative imports in a single file"""
    if not file_path.exists():
        return
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace relative imports with absolute imports
    patterns = [
        (r'from \.health_checker import', 'from health_checker import'),
        (r'from \.health_models import', 'from health_models import'),
        (r'from \.health_reporter import', 'from health_reporter import'),
        (r'from \.health_notifier import', 'from health_notifier import'),
        (r'from \.recommendation_engine import', 'from recommendation_engine import'),
        (r'from \.health_analytics import', 'from health_analytics import'),
        (r'from \.dashboard_server import', 'from dashboard_server import'),
    ]
    
    modified = False
    for pattern, replacement in patterns:
        if re.search(pattern, content):
            content = re.sub(pattern, replacement, content)
            modified = True
    
    if modified:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Fixed imports in {file_path}")

def main():
    """Fix all relative imports in health-checker module"""
    health_checker_dir = Path("tools/health-checker")
    
    if not health_checker_dir.exists():
        print("Health-checker directory not found")
        return
    
    # Fix imports in all Python files
    for py_file in health_checker_dir.glob("*.py"):
        if py_file.name != "__init__.py":  # Skip __init__.py
            fix_imports_in_file(py_file)
    
    print("Import fixing completed")

if __name__ == "__main__":
    main()
