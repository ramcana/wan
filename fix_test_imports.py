#!/usr/bin/env python3
"""
Quick fix for common test import issues found by the audit.
"""

import os
import re
from pathlib import Path

def fix_common_import_issues():
    """Fix the most common import issues found in the audit."""
    
    project_root = Path.cwd()
    fixes_applied = 0
    
    # Common import fixes
    import_fixes = {
        # Backend app imports
        r'from app import app': 'from backend.app import app',
        r'import app': 'import backend.app as app',
        
        # Tools imports (fix hyphenated directory names)
        r'from tools\.test_auditor': 'from tools.test-auditor',
        r'import tools\.test_auditor': 'import tools.test-auditor',
        r'from tools\.test_runner': 'from tools.test-runner',
        r'import tools\.test_runner': 'import tools.test-runner',
        
        # Backend service imports
        r'from services\.generation_service': 'from backend.services.generation_service',
        r'import services\.generation_service': 'import backend.services.generation_service',
        
        # Core imports
        r'from core\.': 'from backend.core.',
        r'import core\.': 'import backend.core.',
        
        # Mock imports
        r'from mock_startup_manager': 'from tests.utils.mock_startup_manager',
        r'import mock_startup_manager': 'import tests.utils.mock_startup_manager',
    }
    
    # Find all Python test files
    test_files = []
    for pattern in ['test_*.py', '*_test.py']:
        test_files.extend(project_root.rglob(pattern))
    
    print(f"Found {len(test_files)} test files to check...")
    
    for test_file in test_files:
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Apply import fixes
            for old_pattern, new_import in import_fixes.items():
                content = re.sub(old_pattern, new_import, content)
            
            # Fix common syntax issues
            # Fix indentation issues (common problem from audit)
            lines = content.split('\n')
            fixed_lines = []
            
            for i, line in enumerate(lines):
                # Fix unexpected indentation after imports
                if i > 0 and lines[i-1].strip().startswith('import') and line.startswith('    ') and not line.strip().startswith('#'):
                    # Check if this line should be dedented
                    if not any(keyword in line for keyword in ['def ', 'class ', 'if ', 'for ', 'while ', 'try:', 'except', 'with ']):
                        line = line.lstrip()
                
                fixed_lines.append(line)
            
            content = '\n'.join(fixed_lines)
            
            # Only write if changes were made
            if content != original_content:
                with open(test_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                rel_path = test_file.relative_to(project_root)
                print(f"Fixed imports in: {rel_path}")
                fixes_applied += 1
        
        except Exception as e:
            rel_path = test_file.relative_to(project_root)
            print(f"Error processing {rel_path}: {e}")
    
    print(f"\nApplied fixes to {fixes_applied} files")
    return fixes_applied

if __name__ == "__main__":
    fix_common_import_issues()