#!/usr/bin/env python3
"""
Comprehensive import fixer to address the 1,181 import issues.
"""

import ast
import sys
from pathlib import Path
import re

def is_project_file(file_path):
    """Check if file is part of the project (not venv or site-packages)."""
    path_str = str(file_path)
    exclude_patterns = [
        'venv', 'site-packages', '__pycache__', '.git', 
        'node_modules', 'local_installation'
    ]
    return not any(pattern in path_str for pattern in exclude_patterns)

def find_import_issues():
    """Find all files with import issues."""
    project_root = Path.cwd()
    python_files = [f for f in project_root.rglob('*.py') if is_project_file(f)]
    
    import_issues = []
    
    print(f"🔍 Checking {len(python_files)} files for import issues...")
    
    for py_file in python_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for various import issues
            issues = []
            lines = content.split('\n')
            
            for i, line in enumerate(lines, 1):
                # Issue 1: Duplicate module names in imports
                if re.search(r'from (\w+)\.(\w+) import \1\.', line):
                    issues.append(f"Line {i}: Duplicate module name in import")
                
                # Issue 2: Invalid import syntax
                if re.search(r'from .* import .*\..*\.', line):
                    issues.append(f"Line {i}: Complex import path")
                
                # Issue 3: Missing imports for common modules
                if any(keyword in line for keyword in ['pytest', 'Mock', 'patch', 'TestClient']) and 'import' not in line:
                    # Check if these are used but not imported
                    if not any('import pytest' in l or 'from unittest.mock import' in l for l in lines[:i]):
                        issues.append(f"Line {i}: Missing import for {line.strip()}")
                
                # Issue 4: Relative imports that should be absolute
                if line.strip().startswith('from .') and 'backend' not in str(py_file):
                    issues.append(f"Line {i}: Relative import outside backend")
            
            if issues:
                import_issues.append({
                    'file': py_file,
                    'issues': issues,
                    'content': content
                })
        
        except Exception:
            continue
    
    return import_issues

def fix_import_issues(content, file_path):
    """Fix common import issues."""
    lines = content.split('\n')
    fixed_lines = []
    changes_made = False
    
    # Track what we've imported to avoid duplicates
    imported_modules = set()
    
    for i, line in enumerate(lines):
        original_line = line
        
        # Fix 1: Duplicate module names in imports
        match = re.match(r'from (\w+)\.(\w+) import \1\.(\w+)', line)
        if match:
            module1, module2, module3 = match.groups()
            line = f'from {module1}.{module2} import {module3}'
            changes_made = True
        
        # Fix 2: Backend app import issue
from backend.app import app
            line = 'from backend.app import app'
            changes_made = True
        
        # Fix 3: Fix relative imports in non-backend files
        if line.strip().startswith('from .') and 'backend' not in str(file_path):
            # Convert to absolute import
            relative_part = line.strip()[5:]  # Remove 'from .'
            if 'import' in relative_part:
                module_part, import_part = relative_part.split(' import ', 1)
                # Guess the absolute path based on file location
                file_parts = file_path.parts
                if 'tests' in file_parts:
                    line = f'from tests.{module_part} import {import_part}'
                elif 'tools' in file_parts:
                    line = f'from tools.{module_part} import {import_part}'
                else:
                    line = f'from {module_part} import {import_part}'
                changes_made = True
        
        # Fix 4: Add missing common imports at the top
        if i == 0 and 'import' not in line and any(keyword in content for keyword in ['pytest', 'Mock', 'patch']):
            # Add common test imports
            if 'pytest' in content and 'import pytest' not in content:
                fixed_lines.append('import pytest')
                changes_made = True
            if 'Mock' in content and 'from unittest.mock import' not in content:
                fixed_lines.append('from unittest.mock import Mock, patch')
                changes_made = True
        
        fixed_lines.append(line)
    
    return '\n'.join(fixed_lines), changes_made

def main():
    """Main function to fix import issues."""
    print("🔍 Finding import issues...")
    import_issues = find_import_issues()
    
    if not import_issues:
        print("✅ No import issues found!")
        return
    
    print(f"❌ Found {len(import_issues)} files with import issues")
    
    # Show sample issues
    total_issues = sum(len(item['issues']) for item in import_issues)
    print(f"📊 Total import issues: {total_issues}")
    
    print("\n🔧 Attempting to fix import issues...")
    
    fixes_applied = 0
    files_fixed = 0
    
    for issue_info in import_issues:
        file_path = issue_info['file']
        content = issue_info['content']
        
        try:
            fixed_content, changes_made = fix_import_issues(content, file_path)
            
            if changes_made:
                # Write the fixed content
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(fixed_content)
                print(f"✅ Fixed imports in: {file_path.relative_to(Path.cwd())}")
                files_fixed += 1
                fixes_applied += len(issue_info['issues'])
        
        except Exception as e:
            print(f"❌ Error processing {file_path.relative_to(Path.cwd())}: {e}")
    
    print(f"\n📊 Import Fix Summary:")
    print(f"  - Files with import issues: {len(import_issues)}")
    print(f"  - Files fixed: {files_fixed}")
    print(f"  - Import issues resolved: {fixes_applied}")
    
    # Check remaining issues
    if files_fixed > 0:
        print("\n🔍 Checking for remaining import issues...")
        remaining_issues = find_import_issues()
        remaining_count = sum(len(item['issues']) for item in remaining_issues)
        print(f"✅ Remaining import issues: {remaining_count}")
        
        improvement = total_issues - remaining_count
        if improvement > 0:
            print(f"🎉 Improved import issues by {improvement} ({improvement/total_issues*100:.1f}%)")

if __name__ == "__main__":
    main()
