#!/usr/bin/env python3
"""
Fix syntax errors introduced by our import fixer.
Focus on indentation issues with print statements.
"""

import ast
from pathlib import Path

def fix_syntax_errors():
    """Fix syntax errors in test files."""
    
    project_root = Path.cwd()
    fixes_applied = 0
    
    # Files with known syntax errors from the audit
    problem_files = [
        'backend/test_cuda_detection.py',
        'backend/test_real_ai_ready.py',
        'backend/tests/test_advanced_system_features.py'
    ]
    
    # Also scan for files with common patterns
    test_files = []
    for pattern in ['test_*.py', '*_test.py']:
        test_files.extend(project_root.rglob(pattern))
    
    # Add first 20 test files to check
    problem_files.extend([str(f.relative_to(project_root)) for f in test_files[:20]])
    
    for file_path in problem_files:
        test_file = project_root / file_path
        if not test_file.exists():
            continue
            
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if file has syntax errors by trying to parse it
            try:
                ast.parse(content)
                continue  # No syntax error, skip
            except SyntaxError:
                pass  # Has syntax error, continue to fix
            
            lines = content.split('\n')
            fixed_lines = []
            
            for i, line in enumerate(lines):
                # Look for print statements that lost indentation
                if (line.startswith('print(') and 
                    i > 0 and 
                    lines[i-1].strip() and 
                    not lines[i-1].startswith('print(') and
                    (lines[i-1].startswith('    ') or lines[i-1].startswith('\t'))):
                    # Add indentation to match previous line
                    prev_line = lines[i-1]
                    indent = len(prev_line) - len(prev_line.lstrip())
                    line = ' ' * indent + line
                
                # Look for other statements that lost indentation after imports
                elif (i > 0 and 
                      lines[i-1].strip().startswith(('import ', 'from ')) and
                      line.strip() and 
                      not line.startswith(' ') and
                      not line.startswith('\t') and
                      not line.strip().startswith(('#', 'def ', 'class ', 'if ', 'try:', 'except', 'finally'))):
                    # This line probably needs indentation
                    # Look back to find the right indentation level
                    for j in range(i-1, -1, -1):
                        if lines[j].strip() and not lines[j].strip().startswith(('import ', 'from ')):
                            if lines[j].startswith('    ') or lines[j].startswith('\t'):
                                indent = len(lines[j]) - len(lines[j].lstrip())
                                line = ' ' * indent + line.strip()
                                break
                            break
                
                fixed_lines.append(line)
            
            new_content = '\n'.join(fixed_lines)
            
            # Verify the fix worked
            try:
                ast.parse(new_content)
                # Only write if we fixed the syntax error
                if new_content != content:
                    with open(test_file, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    print(f"Fixed syntax errors in: {file_path}")
                    fixes_applied += 1
            except SyntaxError:
                # Fix didn't work, skip
                pass
        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    print(f"\nFixed syntax errors in {fixes_applied} files")
    return fixes_applied

if __name__ == "__main__":
    fix_syntax_errors()