#!/usr/bin/env python3
"""
Fix indentation issues introduced by the import fixer.
"""

import ast
from pathlib import Path

def fix_indentation_issues():
    """Fix indentation issues in test files."""
    
    project_root = Path.cwd()
    fixes_applied = 0
    
    # Files we know have indentation issues
    problem_files = [
        'tests/test_backend.py',
        'tools/test-auditor/test_auditor.py'
    ]
    
    for file_path in problem_files:
        test_file = project_root / file_path
        if not test_file.exists():
            continue
            
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            fixed_lines = []
            in_function = False
            function_indent = 0
            
            for i, line in enumerate(lines):
                # Detect function definitions to track expected indentation
                if line.strip().startswith('def '):
                    in_function = True
                    function_indent = len(line) - len(line.lstrip())
                    fixed_lines.append(line)
                    continue
                
                # If we're in a function and find a line that should be indented
                if in_function and line.strip() and not line.startswith(' '):
                    # Check if this line should be indented (like print statements)
                    if (line.strip().startswith('print(') or 
                        line.strip().startswith('return ') or
                        line.strip().startswith('import ') or
                        line.strip().startswith('from ')):
                        # Add proper indentation
                        line = ' ' * (function_indent + 4) + line.strip() + '\n'
                
                # Reset function tracking on empty lines or class/function definitions
                if not line.strip() or line.strip().startswith(('def ', 'class ', '@')):
                    in_function = False
                
                fixed_lines.append(line)
            
            # Write back if changes were made
            new_content = ''.join(fixed_lines)
            with open(test_file, 'r', encoding='utf-8') as f:
                old_content = f.read()
            
            if new_content != old_content:
                with open(test_file, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                print(f"Fixed indentation in: {file_path}")
                fixes_applied += 1
        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    print(f"\nFixed indentation in {fixes_applied} files")
    return fixes_applied

if __name__ == "__main__":
    fix_indentation_issues()