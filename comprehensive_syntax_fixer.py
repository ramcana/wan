#!/usr/bin/env python3
"""
Comprehensive syntax error detection and fixing script.
Scans all Python files and fixes common syntax issues.
"""

import ast
import sys
from pathlib import Path
import re

def find_syntax_errors():
    """Find all Python files with syntax errors."""
    project_root = Path.cwd()
    syntax_errors = []
    
    # Get all Python files
    python_files = list(project_root.rglob('*.py'))
    
    print(f"Checking {len(python_files)} Python files for syntax errors...")
    
    for py_file in python_files:
        # Skip virtual environment and cache files
        if any(part in str(py_file) for part in ['.venv', '__pycache__', '.git', 'node_modules']):
            continue
            
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            try:
                ast.parse(content)
            except SyntaxError as e:
                syntax_errors.append({
                    'file': py_file,
                    'error': str(e),
                    'line': e.lineno,
                    'content': content
                })
        except Exception as e:
            print(f"Error reading {py_file}: {e}")
    
    return syntax_errors

def fix_common_syntax_issues(content, file_path):
    """Fix common syntax issues in Python code."""
    lines = content.split('\n')
    fixed_lines = []
    changes_made = False
    
    for i, line in enumerate(lines):
        original_line = line
        
        # Fix 1: Missing indentation after imports
        if (i > 0 and 
            lines[i-1].strip().startswith(('import ', 'from ')) and
            line.strip() and 
            not line.startswith((' ', '\t')) and
            not line.strip().startswith(('#', 'def ', 'class ', 'if ', 'try:', 'except', 'finally:', 'import ', 'from '))):
            
            # Find appropriate indentation level
            for j in range(i-1, -1, -1):
                if (lines[j].strip() and 
                    not lines[j].strip().startswith(('import ', 'from ')) and
                    (lines[j].startswith(' ') or lines[j].startswith('\t'))):
                    indent = len(lines[j]) - len(lines[j].lstrip())
                    line = ' ' * indent + line.strip()
                    changes_made = True
                    break
        
        # Fix 2: Print statements missing indentation
        elif (line.strip().startswith('print(') and 
              i > 0 and 
              lines[i-1].strip() and 
              not line.startswith((' ', '\t')) and
              (lines[i-1].startswith(' ') or lines[i-1].startswith('\t'))):
            
            prev_indent = len(lines[i-1]) - len(lines[i-1].lstrip())
            line = ' ' * prev_indent + line.strip()
            changes_made = True
        
        # Fix 3: Assert statements missing indentation
        elif (line.strip().startswith('assert ') and 
              i > 0 and 
              lines[i-1].strip() and 
              not line.startswith((' ', '\t')) and
              (lines[i-1].startswith(' ') or lines[i-1].startswith('\t'))):
            
            prev_indent = len(lines[i-1]) - len(lines[i-1].lstrip())
            line = ' ' * prev_indent + line.strip()
            changes_made = True
        
        # Fix 4: Function calls missing indentation
        elif (re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*\(', line.strip()) and 
              i > 0 and 
              lines[i-1].strip() and 
              not line.startswith((' ', '\t')) and
              (lines[i-1].startswith(' ') or lines[i-1].startswith('\t')) and
              not lines[i-1].strip().startswith(('def ', 'class '))):
            
            prev_indent = len(lines[i-1]) - len(lines[i-1].lstrip())
            line = ' ' * prev_indent + line.strip()
            changes_made = True
        
        # Fix 5: Variable assignments missing indentation
        elif ('=' in line and 
              not line.strip().startswith(('#', 'def ', 'class ')) and
              i > 0 and 
              lines[i-1].strip() and 
              not line.startswith((' ', '\t')) and
              (lines[i-1].startswith(' ') or lines[i-1].startswith('\t'))):
            
            prev_indent = len(lines[i-1]) - len(lines[i-1].lstrip())
            line = ' ' * prev_indent + line.strip()
            changes_made = True
        
        # Fix 6: Return statements missing indentation
        elif (line.strip().startswith('return ') and 
              i > 0 and 
              lines[i-1].strip() and 
              not line.startswith((' ', '\t')) and
              (lines[i-1].startswith(' ') or lines[i-1].startswith('\t'))):
            
            prev_indent = len(lines[i-1]) - len(lines[i-1].lstrip())
            line = ' ' * prev_indent + line.strip()
            changes_made = True
        
        fixed_lines.append(line)
    
    return '\n'.join(fixed_lines), changes_made

def main():
    """Main function to find and fix syntax errors."""
    print("🔍 Finding syntax errors...")
    syntax_errors = find_syntax_errors()
    
    if not syntax_errors:
        print("✅ No syntax errors found!")
        return
    
    print(f"❌ Found {len(syntax_errors)} files with syntax errors:")
    for error in syntax_errors[:10]:  # Show first 10
        print(f"  - {error['file']}: {error['error']}")
    
    if len(syntax_errors) > 10:
        print(f"  ... and {len(syntax_errors) - 10} more")
    
    print("\n🔧 Attempting to fix syntax errors...")
    
    fixes_applied = 0
    for error_info in syntax_errors:
        file_path = error_info['file']
        content = error_info['content']
        
        try:
            fixed_content, changes_made = fix_common_syntax_issues(content, file_path)
            
            if changes_made:
                # Verify the fix worked
                try:
                    ast.parse(fixed_content)
                    # Write the fixed content
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(fixed_content)
                    print(f"✅ Fixed: {file_path}")
                    fixes_applied += 1
                except SyntaxError:
                    print(f"❌ Could not fix: {file_path}")
            else:
                print(f"⚠️  No automatic fix available: {file_path}")
        
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
    
    print(f"\n📊 Summary:")
    print(f"  - Files with syntax errors: {len(syntax_errors)}")
    print(f"  - Files fixed: {fixes_applied}")
    print(f"  - Files remaining: {len(syntax_errors) - fixes_applied}")
    
    # Check remaining errors
    if fixes_applied > 0:
        print("\n🔍 Checking for remaining syntax errors...")
        remaining_errors = find_syntax_errors()
        print(f"✅ Remaining syntax errors: {len(remaining_errors)}")

if __name__ == "__main__":
    main()