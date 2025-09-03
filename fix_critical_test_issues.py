#!/usr/bin/env python3
"""
Fix critical test issues found by the audit.
Focus on the most common and safe fixes.
"""

import os
import re
from pathlib import Path

def fix_critical_issues():
    """Fix the most critical and safe issues."""
    
    project_root = Path.cwd()
    fixes_applied = 0
    
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
            
            # Fix specific problematic imports that we know are wrong
            # Fix the double backend.app import we created
            content = re.sub(r'from backend\.app import backend\.app as app', 'from backend.app import app', content)
            
            # Fix simple app imports in backend tests
            if 'backend' in str(test_file):
                content = re.sub(r'from app import app', 'from backend.app import app', content)
            
            # Fix tools imports with hyphens (these are directory names)
            content = re.sub(r'from tools\.test_auditor', 'from tools.test-auditor', content)
            content = re.sub(r'from tools\.test_runner', 'from tools.test-runner', content)
            
            # Only write if changes were made
            if content != original_content:
                with open(test_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                rel_path = test_file.relative_to(project_root)
                print(f"Fixed critical issues in: {rel_path}")
                fixes_applied += 1
        
        except Exception as e:
            rel_path = test_file.relative_to(project_root)
            print(f"Error processing {rel_path}: {e}")
    
    print(f"\nApplied critical fixes to {fixes_applied} files")
    return fixes_applied

if __name__ == "__main__":
    fix_critical_issues()