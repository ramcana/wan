"""
Test suite for dead code analysis functionality
"""

import os
import tempfile
from pathlib import Path

from dead_code_analyzer import DeadCodeAnalyzer


def test_dead_code_analyzer():
    """Test basic dead code analysis functionality"""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test Python file with dead code
        test_file = temp_path / "sample_module.py"
        test_content = '''
import os
import sys
import unused_module

def used_function():
    """This function is used"""
    return "used"

def dead_function():
    """This function is never called"""
    return "dead"

class UsedClass:
    def method(self):
        return "used"

class DeadClass:
    def dead_method(self):
        return "dead"

def main():
    used = used_function()
    obj = UsedClass()
    print(used, obj.method())

if __name__ == "__main__":
    main()
'''
        test_file.write_text(test_content)
        
        # Initialize analyzer
        analyzer = DeadCodeAnalyzer(str(temp_path))
        
        # Run analysis
        report = analyzer.analyze_dead_code(include_tests=False)
        
        # Verify results
        print(f"Files analyzed: {report.total_files_analyzed}")
        print(f"Dead functions: {len(report.dead_functions)}")
        print(f"Dead classes: {len(report.dead_classes)}")
        print(f"Unused imports: {len(report.unused_imports)}")
        print(f"Dead files: {len(report.dead_files)}")
        
        # Should find dead function and class
        assert len(report.dead_functions) >= 1, "Should find at least 1 dead function"
        assert len(report.dead_classes) >= 1, "Should find at least 1 dead class"
        assert len(report.unused_imports) >= 1, "Should find at least 1 unused import"
        
        # Check specific findings
        dead_func_names = [f.name for f in report.dead_functions]
        assert "dead_function" in dead_func_names, "Should find dead_function"
        
        dead_class_names = [c.name for c in report.dead_classes]
        assert "DeadClass" in dead_class_names, "Should find DeadClass"
        
        unused_import_names = [i.import_name for i in report.unused_imports]
        assert "unused_module" in unused_import_names, "Should find unused_module import"
        
        print("✓ Dead code analyzer test passed")


def test_unused_imports():
    """Test unused import detection"""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test file with various import types
        test_file = temp_path / "imports_sample.py"
        test_content = '''
import os
import sys
import unused_import
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict, Optional, Unused

def test_function():
    # Use some imports
    path = Path(".")
    data = defaultdict(list)
    items = Counter([1, 2, 3])
    result: List[str] = []
    return path, data, items, result
'''
        test_file.write_text(test_content)
        
        analyzer = DeadCodeAnalyzer(str(temp_path))
        report = analyzer.analyze_dead_code()
        
        print(f"Unused imports found: {len(report.unused_imports)}")
        for imp in report.unused_imports:
            print(f"  - {imp.import_name} from {imp.module_name}")
        
        # Should find unused imports
        unused_names = [i.import_name for i in report.unused_imports]
        assert "unused_import" in unused_names, "Should find unused_import"
        assert "Unused" in unused_names, "Should find unused type import"
        
        print("✓ Unused imports test passed")


def test_dead_files():
    """Test dead file detection"""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create main file that imports another
        main_file = temp_path / "main.py"
        main_content = '''
from used_module import some_function

def main():
    result = some_function()
    print(result)

if __name__ == "__main__":
    main()
'''
        main_file.write_text(main_content)
        
        # Create used module
        used_file = temp_path / "used_module.py"
        used_content = '''
def some_function():
    return "Hello from used module"
'''
        used_file.write_text(used_content)
        
        # Create dead file (never imported)
        dead_file = temp_path / "dead_module.py"
        dead_content = '''
def dead_function():
    return "This file is never imported"

class DeadClass:
    pass
'''
        dead_file.write_text(dead_content)
        
        analyzer = DeadCodeAnalyzer(str(temp_path))
        report = analyzer.analyze_dead_code()
        
        print(f"Dead files found: {len(report.dead_files)}")
        for file in report.dead_files:
            print(f"  - {file.file_path}")
        
        # Should find the dead file
        dead_file_paths = [f.file_path for f in report.dead_files]
        assert str(dead_file) in dead_file_paths, "Should find dead_module.py"
        assert str(used_file) not in dead_file_paths, "Should not mark used_module.py as dead"
        assert str(main_file) not in dead_file_paths, "Should not mark main.py as dead"
        
        print("✓ Dead files test passed")


def test_backup_functionality():
    """Test backup creation for dead code removal"""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test file
        test_file = temp_path / "backup_sample.py"
        test_content = '''
import unused_import

def dead_function():
    return "dead"

def main():
    print("main")
'''
        test_file.write_text(test_content)
        
        analyzer = DeadCodeAnalyzer(str(temp_path))
        
        # Create backup
        files_to_backup = [str(test_file)]
        backup_path = analyzer.create_backup(files_to_backup)
        
        print(f"Backup created at: {backup_path}")
        
        # Verify backup exists
        backup_dir = Path(backup_path)
        assert backup_dir.exists(), "Backup directory should exist"
        
        # Verify manifest exists
        manifest_path = backup_dir / "manifest.json"
        assert manifest_path.exists(), "Backup manifest should exist"
        
        # Verify file was backed up
        backed_up_file = backup_dir / test_file.relative_to(temp_path)
        assert backed_up_file.exists(), "File should be backed up"
        assert backed_up_file.read_text() == test_content, "Backup content should match original"
        
        print("✓ Backup functionality test passed")


if __name__ == "__main__":
    print("Running dead code analyzer tests...")
    
    try:
        test_dead_code_analyzer()
        test_unused_imports()
        test_dead_files()
        test_backup_functionality()
        print("\n✅ All dead code analyzer tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise