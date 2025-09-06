#!/usr/bin/env python3
"""
Test Cleanup Tool - Declutter the test codebase
Identifies and removes redundant, outdated, or unnecessary test files and code.
"""

import ast
import re
from pathlib import Path
from typing import List, Dict, Set, Tuple
import json

class TestCleanupAnalyzer:
    def __init__(self):
        self.project_root = Path.cwd()
        self.test_files = []
        self.duplicate_tests = []
        self.empty_tests = []
        self.outdated_tests = []
        self.redundant_files = []
        self.cleanup_candidates = []
        
    def find_all_test_files(self):
        """Find all test files in the project."""
        test_patterns = [
            'test_*.py',
            '*_test.py',
            'tests/**/*.py'
        ]
        
        test_files = set()
        for pattern in test_patterns:
            test_files.update(self.project_root.rglob(pattern))
        
        # Filter out venv and other non-project files
        self.test_files = [
            f for f in test_files 
            if not any(exclude in str(f) for exclude in [
                'venv', 'site-packages', '__pycache__', '.git',
                'node_modules', 'local_installation'
            ])
        ]
        
        print(f"📁 Found {len(self.test_files)} test files")
        return self.test_files
    
    def analyze_empty_tests(self):
        """Find test files that are empty or have no real test functions."""
        empty_tests = []
        
        for test_file in self.test_files:
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                
                # Check if file is empty or only has imports/comments
                if not content:
                    empty_tests.append({
                        'file': test_file,
                        'reason': 'completely_empty',
                        'size': 0
                    })
                    continue
                
                # Parse and check for actual test functions
                try:
                    tree = ast.parse(content)
                    test_functions = [
                        node for node in ast.walk(tree)
                        if isinstance(node, ast.FunctionDef) and node.name.startswith('test_')
                    ]
                    
                    if not test_functions:
                        # Check if it's just imports and comments
                        lines = [line.strip() for line in content.split('\n') if line.strip()]
                        code_lines = [
                            line for line in lines 
                            if not (line.startswith('#') or 
                                   line.startswith('import ') or 
                                   line.startswith('from ') or
                                   line.startswith('"""') or
                                   line.startswith("'''"))
                        ]
                        
                        if len(code_lines) <= 2:  # Very minimal content
                            empty_tests.append({
                                'file': test_file,
                                'reason': 'no_test_functions',
                                'size': len(content),
                                'lines': len(lines)
                            })
                
                except SyntaxError:
                    # File has syntax errors, might be corrupted
                    empty_tests.append({
                        'file': test_file,
                        'reason': 'syntax_error',
                        'size': len(content)
                    })
            
            except Exception as e:
                print(f"Error analyzing {test_file}: {e}")
        
        self.empty_tests = empty_tests
        return empty_tests
    
    def analyze_duplicate_tests(self):
        """Find duplicate or very similar test files."""
        duplicates = []
        file_signatures = {}
        
        for test_file in self.test_files:
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Create a signature based on function names and structure
                try:
                    tree = ast.parse(content)
                    functions = [
                        node.name for node in ast.walk(tree)
                        if isinstance(node, ast.FunctionDef)
                    ]
                    
                    # Create signature from sorted function names
                    signature = tuple(sorted(functions))
                    
                    if signature in file_signatures and len(functions) > 0:
                        duplicates.append({
                            'original': file_signatures[signature],
                            'duplicate': test_file,
                            'functions': functions,
                            'similarity': 'identical_structure'
                        })
                    else:
                        file_signatures[signature] = test_file
                
                except SyntaxError:
                    continue
            
            except Exception:
                continue
        
        self.duplicate_tests = duplicates
        return duplicates
    
    def analyze_outdated_tests(self):
        """Find tests that appear to be outdated or obsolete."""
        outdated = []
        
        outdated_patterns = [
            r'test.*old.*\.py$',
            r'test.*deprecated.*\.py$',
            r'test.*legacy.*\.py$',
            r'test.*backup.*\.py$',
            r'test.*temp.*\.py$',
            r'test.*tmp.*\.py$',
            r'.*_old_test\.py$',
            r'.*_backup_test\.py$',
        ]
        
        for test_file in self.test_files:
            file_name = test_file.name.lower()
            file_path = str(test_file).lower()
            
            # Check filename patterns
            for pattern in outdated_patterns:
                if re.search(pattern, file_path):
                    outdated.append({
                        'file': test_file,
                        'reason': 'outdated_filename_pattern',
                        'pattern': pattern
                    })
                    break
            
            # Check file content for outdated markers
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                outdated_markers = [
                    'TODO: Remove this test',
                    'DEPRECATED',
                    'This test is obsolete',
                    'Legacy test',
                    'Old implementation'
                ]
                
                for marker in outdated_markers:
                    if marker.lower() in content.lower():
                        outdated.append({
                            'file': test_file,
                            'reason': 'contains_outdated_marker',
                            'marker': marker
                        })
                        break
            
            except Exception:
                continue
        
        self.outdated_tests = outdated
        return outdated
    
    def analyze_redundant_files(self):
        """Find files that might be redundant (multiple versions of same test)."""
        redundant = []
        
        # Group files by base name (without suffixes like _simple, _basic, etc.)
        base_groups = {}
        
        for test_file in self.test_files:
            # Extract base name
            name = test_file.stem
            base_name = re.sub(r'_(simple|basic|unit|integration|comprehensive|complete|full|minimal|advanced)$', '', name)
            
            if base_name not in base_groups:
                base_groups[base_name] = []
            base_groups[base_name].append(test_file)
        
        # Find groups with multiple files
        for base_name, files in base_groups.items():
            if len(files) > 1:
                # Sort by modification time or file size to identify the "main" version
                files_with_info = []
                for f in files:
                    try:
                        stat = f.stat()
                        with open(f, 'r', encoding='utf-8') as file:
                            content = file.read()
                        
                        files_with_info.append({
                            'file': f,
                            'size': len(content),
                            'mtime': stat.st_mtime,
                            'lines': len(content.split('\n'))
                        })
                    except Exception:
                        continue
                
                if len(files_with_info) > 1:
                    # Sort by size (larger files are likely more complete)
                    files_with_info.sort(key=lambda x: x['size'], reverse=True)
                    
                    redundant.append({
                        'base_name': base_name,
                        'files': files_with_info,
                        'recommended_keep': files_with_info[0]['file'],
                        'candidates_for_removal': [f['file'] for f in files_with_info[1:]]
                    })
        
        self.redundant_files = redundant
        return redundant
    
    def generate_cleanup_plan(self):
        """Generate a comprehensive cleanup plan."""
        cleanup_plan = {
            'empty_files': [],
            'duplicate_files': [],
            'outdated_files': [],
            'redundant_files': [],
            'safe_to_remove': [],
            'review_required': []
        }
        
        # Categorize empty files
        for empty in self.empty_tests:
            if empty['reason'] in ['completely_empty', 'syntax_error']:
                cleanup_plan['safe_to_remove'].append({
                    'file': empty['file'],
                    'reason': f"Empty file: {empty['reason']}",
                    'confidence': 'high'
                })
            else:
                cleanup_plan['review_required'].append({
                    'file': empty['file'],
                    'reason': f"No test functions: {empty['reason']}",
                    'confidence': 'medium'
                })
        
        # Categorize duplicates
        for dup in self.duplicate_tests:
            cleanup_plan['review_required'].append({
                'file': dup['duplicate'],
                'reason': f"Duplicate of {dup['original'].name}",
                'confidence': 'medium'
            })
        
        # Categorize outdated files
        for outdated in self.outdated_tests:
            if 'filename_pattern' in outdated['reason']:
                cleanup_plan['safe_to_remove'].append({
                    'file': outdated['file'],
                    'reason': f"Outdated filename pattern: {outdated.get('pattern', '')}",
                    'confidence': 'high'
                })
            else:
                cleanup_plan['review_required'].append({
                    'file': outdated['file'],
                    'reason': f"Contains outdated marker: {outdated.get('marker', '')}",
                    'confidence': 'medium'
                })
        
        # Categorize redundant files
        for redundant in self.redundant_files:
            for candidate in redundant['candidates_for_removal']:
                cleanup_plan['review_required'].append({
                    'file': candidate,
                    'reason': f"Redundant version of {redundant['base_name']} (keep {redundant['recommended_keep'].name})",
                    'confidence': 'low'
                })
        
        return cleanup_plan
    
    def execute_safe_cleanup(self, cleanup_plan, dry_run=True):
        """Execute the cleanup plan for files marked as safe to remove."""
        removed_files = []
        
        safe_files = cleanup_plan['safe_to_remove']
        
        print(f"\n🧹 {'DRY RUN: Would remove' if dry_run else 'Removing'} {len(safe_files)} safe files...")
        
        for item in safe_files:
            file_path = item['file']
            reason = item['reason']
            
            try:
                if not dry_run:
                    file_path.unlink()
                
                removed_files.append({
                    'file': str(file_path.relative_to(self.project_root)),
                    'reason': reason
                })
                
                print(f"{'Would remove' if dry_run else 'Removed'}: {file_path.relative_to(self.project_root)} - {reason}")
            
            except Exception as e:
                print(f"Error {'simulating removal of' if dry_run else 'removing'} {file_path}: {e}")
        
        return removed_files
    
    def generate_report(self):
        """Generate a comprehensive cleanup report."""
        report = {
            'summary': {
                'total_test_files': len(self.test_files),
                'empty_tests': len(self.empty_tests),
                'duplicate_tests': len(self.duplicate_tests),
                'outdated_tests': len(self.outdated_tests),
                'redundant_groups': len(self.redundant_files)
            },
            'details': {
                'empty_tests': self.empty_tests,
                'duplicate_tests': self.duplicate_tests,
                'outdated_tests': self.outdated_tests,
                'redundant_files': self.redundant_files
            }
        }
        
        return report

def main():
    print("🧹 Starting Test Codebase Cleanup Analysis...")
    
    analyzer = TestCleanupAnalyzer()
    
    # Step 1: Find all test files
    analyzer.find_all_test_files()
    
    # Step 2: Analyze different types of cleanup candidates
    print("\n🔍 Analyzing empty tests...")
    empty_tests = analyzer.analyze_empty_tests()
    print(f"Found {len(empty_tests)} empty or minimal test files")
    
    print("\n🔍 Analyzing duplicate tests...")
    duplicate_tests = analyzer.analyze_duplicate_tests()
    print(f"Found {len(duplicate_tests)} potential duplicate test files")
    
    print("\n🔍 Analyzing outdated tests...")
    outdated_tests = analyzer.analyze_outdated_tests()
    print(f"Found {len(outdated_tests)} potentially outdated test files")
    
    print("\n🔍 Analyzing redundant tests...")
    redundant_tests = analyzer.analyze_redundant_files()
    print(f"Found {len(redundant_tests)} groups of potentially redundant test files")
    
    # Step 3: Generate cleanup plan
    print("\n📋 Generating cleanup plan...")
    cleanup_plan = analyzer.generate_cleanup_plan()
    
    # Step 4: Show summary
    print(f"\n📊 Cleanup Summary:")
    print(f"  - Safe to remove: {len(cleanup_plan['safe_to_remove'])} files")
    print(f"  - Require review: {len(cleanup_plan['review_required'])} files")
    
    # Step 5: Show details
    if cleanup_plan['safe_to_remove']:
        print(f"\n✅ Files safe to remove:")
        for item in cleanup_plan['safe_to_remove'][:10]:
            rel_path = item['file'].relative_to(analyzer.project_root)
            print(f"  - {rel_path} ({item['reason']})")
        if len(cleanup_plan['safe_to_remove']) > 10:
            print(f"  ... and {len(cleanup_plan['safe_to_remove']) - 10} more")
    
    if cleanup_plan['review_required']:
        print(f"\n⚠️  Files requiring review:")
        for item in cleanup_plan['review_required'][:10]:
            rel_path = item['file'].relative_to(analyzer.project_root)
            print(f"  - {rel_path} ({item['reason']})")
        if len(cleanup_plan['review_required']) > 10:
            print(f"  ... and {len(cleanup_plan['review_required']) - 10} more")
    
    # Step 6: Execute safe cleanup (dry run first)
    print(f"\n🧹 Executing safe cleanup (DRY RUN)...")
    removed_files = analyzer.execute_safe_cleanup(cleanup_plan, dry_run=True)
    
    if removed_files:
        print(f"\n💡 To actually remove these files, run with dry_run=False")
        
        # Ask user if they want to proceed
        response = input(f"\nDo you want to remove {len(removed_files)} safe files? (y/N): ")
        if response.lower() == 'y':
            print(f"\n🗑️  Removing files...")
            actual_removed = analyzer.execute_safe_cleanup(cleanup_plan, dry_run=False)
            print(f"✅ Successfully removed {len(actual_removed)} files")
    
    # Step 7: Generate detailed report
    report = analyzer.generate_report()
    
    with open('test_cleanup_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n📄 Detailed report saved to: test_cleanup_report.json")
    
    return cleanup_plan

if __name__ == "__main__":
    main()