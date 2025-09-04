#!/usr/bin/env python3
"""
Fix syntax errors specifically in project files (not venv/site-packages).
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

def find_project_syntax_errors():
    """Find syntax errors in project files only."""
    project_root = Path.cwd()
    syntax_errors = []
    
    # Get all Python files
    python_files = [f for f in project_root.rglob('*.py') if is_project_file(f)]
    
    print(f"Checking {len(python_files)} project Python files for syntax errors...")
    
    for py_file in python_files:
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

def fix_advanced_syntax_issues(content, file_path):
    """Fix more advanced syntax issues."""
    lines = content.split('\n')
    fixed_lines = []
    changes_made = False
    
    i = 0
    while i < len(lines):
        line = lines[i]
        original_line = line
        
        # Fix incomplete try blocks
        if line.strip() == 'try:' and i + 1 < len(lines):
            # Look for the next non-empty line
            next_line_idx = i + 1
            while next_line_idx < len(lines) and not lines[next_line_idx].strip():
                next_line_idx += 1
            
            if (next_line_idx < len(lines) and 
                not lines[next_line_idx].strip().startswith(('except', 'finally')) and
                not lines[next_line_idx].startswith('    ')):
                # Add pass statement
                fixed_lines.append(line)
                fixed_lines.append('    pass')
                changes_made = True
                i += 1
                continue
        
        # Fix missing colons in function definitions
        if (re.match(r'^\s*def\s+\w+\([^)]*\)\s*$', line) and 
            not line.rstrip().endswith(':')):
            line = line.rstrip() + ':'
            changes_made = True
        
        # Fix missing colons in class definitions
        if (re.match(r'^\s*class\s+\w+.*\s*$', line) and 
            not line.rstrip().endswith(':') and
            '(' in line and ')' in line):
            line = line.rstrip() + ':'
            changes_made = True
        
        # Fix missing colons in if statements
        if (re.match(r'^\s*if\s+.*\s*$', line) and 
            not line.rstrip().endswith(':') and
            not line.strip().endswith('\\')):
            line = line.rstrip() + ':'
            changes_made = True
        
        # Fix missing indentation for statements after function definitions
        if (i > 0 and 
            lines[i-1].strip().endswith(':') and
            line.strip() and 
            not line.startswith(' ') and 
            not line.startswith('\t') and
            not line.strip().startswith(('#', 'def ', 'class ', '@'))):
            
            # Add 4 spaces indentation
            line = '    ' + line.strip()
            changes_made = True
        
        # Fix unmatched parentheses by adding missing closing paren
        if line.count('(') > line.count(')'):
            missing_parens = line.count('(') - line.count(')')
            line = line.rstrip() + ')' * missing_parens
            changes_made = True
        
        # Fix unmatched brackets
        if line.count('[') > line.count(']'):
            missing_brackets = line.count('[') - line.count(']')
            line = line.rstrip() + ']' * missing_brackets
            changes_made = True
        
        # Fix unmatched braces
        if line.count('{') > line.count('}'):
            missing_braces = line.count('{') - line.count('}')
            line = line.rstrip() + '}' * missing_braces
            changes_made = True
        
        fixed_lines.append(line)
        i += 1
    
    return '\n'.join(fixed_lines), changes_made

def main():
    """Main function to fix project syntax errors."""
    print("🔍 Finding syntax errors in project files...")
    syntax_errors = find_project_syntax_errors()
    
    if not syntax_errors:
        print("✅ No syntax errors found in project files!")
        return
    
    print(f"❌ Found {len(syntax_errors)} project files with syntax errors:")
    for error in syntax_errors:
        print(f"  - {error['file'].relative_to(Path.cwd())}: {error['error']}")
    
    print("\n🔧 Attempting to fix syntax errors...")
    
    fixes_applied = 0
    for error_info in syntax_errors:
        file_path = error_info['file']
        content = error_info['content']
        
        try:
            # Try basic fixes first
            from comprehensive_syntax_fixer import fix_common_syntax_issues
            fixed_content, changes_made = fix_common_syntax_issues(content, file_path)
            
            # If basic fixes didn't work, try advanced fixes
            if not changes_made:
                fixed_content, changes_made = fix_advanced_syntax_issues(content, file_path)
            
            if changes_made:
                # Verify the fix worked
                try:
                    ast.parse(fixed_content)
                    # Write the fixed content
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(fixed_content)
                    print(f"✅ Fixed: {file_path.relative_to(Path.cwd())}")
                    fixes_applied += 1
                except SyntaxError as e:
                    print(f"❌ Could not fix: {file_path.relative_to(Path.cwd())} - {e}")
            else:
                print(f"⚠️  No automatic fix available: {file_path.relative_to(Path.cwd())}")
        
        except Exception as e:
            print(f"❌ Error processing {file_path.relative_to(Path.cwd())}: {e}")
    
    print(f"\n📊 Summary:")
    print(f"  - Project files with syntax errors: {len(syntax_errors)}")
    print(f"  - Files fixed: {fixes_applied}")
    print(f"  - Files remaining: {len(syntax_errors) - fixes_applied}")
    
    # Check remaining errors
    if fixes_applied > 0:
        print("\n🔍 Checking for remaining project syntax errors...")
        remaining_errors = find_project_syntax_errors()
        print(f"✅ Remaining project syntax errors: {len(remaining_errors)}")
        
        if remaining_errors:
            print("\nRemaining errors:")
            for error in remaining_errors[:5]:
                print(f"  - {error['file'].relative_to(Path.cwd())}: {error['error']}")

if __name__ == "__main__":
    main()