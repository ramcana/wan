#!/usr/bin/env python3
"""
Fix remaining syntax errors by targeting files with known issues.
"""

import ast
import json
from pathlib import Path

def fix_remaining_syntax_errors():
    """Fix syntax errors in files identified by the audit."""
    
    project_root = Path.cwd()
    fixes_applied = 0
    
    # Read the audit report to get files with syntax errors
    audit_report_path = project_root / 'test_audit_report.json'
    syntax_error_files = []
    
    if audit_report_path.exists():
        try:
            with open(audit_report_path, 'r') as f:
                audit_data = json.load(f)
            
            # Extract files with syntax errors from critical issues
            for issue in audit_data.get('critical_issues', []):
                if 'Syntax error' in issue.get('description', ''):
                    file_path = issue.get('test_file', '')
                    if file_path and file_path not in syntax_error_files:
                        syntax_error_files.append(file_path)
        except Exception as e:
            print(f"Could not read audit report: {e}")
    
    # Add some known problematic files
    syntax_error_files.extend([
        'backend/test_cuda_detection.py',
        'backend/test_real_ai_ready.py',
        'backend/tests/test_advanced_system_features.py'
    ])
    
    print(f"Found {len(syntax_error_files)} files with potential syntax errors")
    
    for file_path in syntax_error_files[:50]:  # Process first 50
        test_file = Path(file_path)
        if not test_file.exists():
            continue
            
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if file has syntax errors
            try:
                ast.parse(content)
                continue  # No syntax error, skip
            except SyntaxError as e:
                print(f"Fixing syntax error in {file_path}: {e.msg}")
            
            # Fix common indentation issues
            lines = content.split('\n')
            fixed_lines = []
            
            for i, line in enumerate(lines):
                original_line = line
                
                # Fix print statements that lost indentation
                if line.startswith('print(') and i > 0:
                    prev_line = lines[i-1]
                    if prev_line.strip() and (prev_line.startswith('    ') or prev_line.startswith('\t')):
                        # Match indentation of previous line
                        indent = len(prev_line) - len(prev_line.lstrip())
                        line = ' ' * indent + line
                
                # Fix return statements that lost indentation
                elif line.startswith('return ') and i > 0:
                    prev_line = lines[i-1]
                    if prev_line.strip() and (prev_line.startswith('    ') or prev_line.startswith('\t')):
                        indent = len(prev_line) - len(prev_line.lstrip())
                        line = ' ' * indent + line
                
                # Fix other statements after imports
                elif (line.strip() and 
                      not line.startswith(' ') and 
                      not line.startswith('\t') and
                      i > 0 and
                      lines[i-1].strip().startswith(('import ', 'from '))):
                    # Look for the right indentation context
                    for j in range(i-1, max(0, i-10), -1):
                        if (lines[j].strip() and 
                            not lines[j].strip().startswith(('import ', 'from ')) and
                            (lines[j].startswith('    ') or lines[j].startswith('\t'))):
                            indent = len(lines[j]) - len(lines[j].lstrip())
                            line = ' ' * indent + line.strip()
                            break
                
                fixed_lines.append(line)
            
            new_content = '\n'.join(fixed_lines)
            
            # Verify the fix worked
            try:
                ast.parse(new_content)
                if new_content != content:
                    with open(test_file, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    print(f"✅ Fixed syntax errors in: {file_path}")
                    fixes_applied += 1
                else:
                    print(f"⚠️  No changes needed for: {file_path}")
            except SyntaxError as e:
                print(f"❌ Could not fix syntax error in {file_path}: {e.msg}")
        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    print(f"\nFixed syntax errors in {fixes_applied} files")
    return fixes_applied

if __name__ == "__main__":
    fix_remaining_syntax_errors()