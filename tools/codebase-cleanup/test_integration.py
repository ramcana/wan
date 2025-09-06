"""
Integration test for all codebase cleanup tools
"""

import tempfile
from pathlib import Path

from duplicate_detector import DuplicateDetector
from dead_code_analyzer import DeadCodeAnalyzer
from naming_standardizer import NamingStandardizer


def test_full_cleanup_workflow():
    """Test the complete cleanup workflow with all tools"""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create a realistic project structure with various issues
        create_test_project(temp_path)
        
        print(f"Created test project in: {temp_path}")
        
        # 1. Duplicate Detection
        print("\n1. Running duplicate detection...")
        duplicate_detector = DuplicateDetector(str(temp_path))
        duplicate_report = duplicate_detector.scan_for_duplicates()
        
        print(f"   Found {len(duplicate_report.duplicate_files)} duplicate files")
        print(f"   Potential savings: {duplicate_report.potential_savings / 1024:.1f} KB")
        
        # 2. Dead Code Analysis
        print("\n2. Running dead code analysis...")
        dead_code_analyzer = DeadCodeAnalyzer(str(temp_path))
        dead_code_report = dead_code_analyzer.analyze_dead_code()
        
        print(f"   Found {len(dead_code_report.dead_functions)} dead functions")
        print(f"   Found {len(dead_code_report.unused_imports)} unused imports")
        print(f"   Found {len(dead_code_report.dead_files)} dead files")
        
        # 3. Naming Standardization
        print("\n3. Running naming analysis...")
        naming_standardizer = NamingStandardizer(str(temp_path))
        naming_report = naming_standardizer.analyze_naming_conventions()
        
        print(f"   Found {len(naming_report.violations)} naming violations")
        print(f"   Found {len(naming_report.organization_suggestions)} organization suggestions")
        
        # Verify we found the expected issues
        assert len(duplicate_report.duplicate_files) >= 2, "Should find duplicate files"
        assert len(dead_code_report.dead_functions) >= 1, "Should find dead functions"
        assert len(dead_code_report.unused_imports) >= 1, "Should find unused imports"
        assert len(naming_report.violations) >= 1, "Should find naming violations"
        
        print("\n✅ Full cleanup workflow test passed!")


def create_test_project(base_path: Path):
    """Create a test project with various cleanup issues"""
    
    # Create directory structure
    (base_path / "src").mkdir()
    (base_path / "tests").mkdir()
    (base_path / "utils").mkdir()
    (base_path / "BadDirectoryName").mkdir()
    
    # Create duplicate files
    duplicate_content = '''
def common_function():
    """This function appears in multiple files"""
    return "common"

class CommonClass:
    def method(self):
        return "common"
'''
    
    (base_path / "src" / "module1.py").write_text(duplicate_content)
    (base_path / "src" / "module2.py").write_text(duplicate_content)  # Exact duplicate
    
    # Create file with dead code and naming issues
    problematic_file = base_path / "src" / "BadFileName.py"
    problematic_content = '''
import os
import sys
import unused_module
from typing import List, Dict, Optional, Unused

class badClassName:
    """A class with bad naming"""
    
    def BadMethodName(self):
        """Method that's never called"""
        return "bad"
    
    def good_method(self):
        """Method that is used"""
        return "good"

def DeadFunction():
    """Function that's never called"""
    BadVariableName = "dead"
    return BadVariableName

def used_function():
    """Function that is called"""
    good_variable = "used"
    return good_variable

def main():
    obj = badClassName()
    result = obj.good_method()
    used_result = used_function()
    print(result, used_result)

if __name__ == "__main__":
    main()
'''
    problematic_file.write_text(problematic_content)
    
    # Create dead file (never imported)
    dead_file = base_path / "utils" / "dead_utility.py"
    dead_content = '''
def dead_utility_function():
    """This utility is never used"""
    return "dead utility"

class DeadUtilityClass:
    def dead_method(self):
        return "dead"
'''
    dead_file.write_text(dead_content)
    
    # Create JavaScript file with naming issues
    js_file = base_path / "src" / "bad-js-file.js"
    js_content = '''
class bad_class_name {
    constructor() {
        this.bad_variable_name = "bad";
    }
    
    Bad_Method_Name() {
        return this.bad_variable_name;
    }
}

function Bad_Function_Name() {
    return "bad function";
}

// Good examples
class GoodClassName {
    constructor() {
        this.goodVariable = "good";
    }
    
    goodMethod() {
        return this.goodVariable;
    }
}

function goodFunctionName() {
    return "good function";
}
'''
    js_file.write_text(js_content)
    
    # Create scattered test files (organization issue)
    (base_path / "test_scattered1.py").write_text("# Scattered test file 1")
    (base_path / "src" / "test_scattered2.py").write_text("# Scattered test file 2")
    (base_path / "utils" / "test_scattered3.py").write_text("# Scattered test file 3")
    
    # Create config files in wrong places
    (base_path / "config1.json").write_text('{"setting": "value1"}')
    (base_path / "src" / "config2.json").write_text('{"setting": "value2"}')
    
    print("Created test project with the following issues:")
    print("  - Duplicate files (module1.py, module2.py)")
    print("  - Dead code (DeadFunction, dead_utility.py)")
    print("  - Unused imports (unused_module, Unused)")
    print("  - Naming violations (badClassName, BadMethodName, etc.)")
    print("  - Organization issues (scattered test files, configs)")


def test_cli_integration():
    """Test CLI integration with all tools"""
    
    print("\n=== CLI Integration Test ===")
    
    # This would test the CLI in a real scenario
    # For now, just verify imports work
    try:
        from cli import main, handle_duplicates_command, handle_dead_code_command, handle_naming_command
        print("✅ CLI imports successful")
        
        # Test argument parsing (without actually running)
        import argparse
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest='command')
        
        from cli import setup_duplicate_parser, setup_dead_code_parser, setup_naming_parser
        setup_duplicate_parser(subparsers)
        setup_dead_code_parser(subparsers)
        setup_naming_parser(subparsers)
        
        # Test parsing some commands
        args = parser.parse_args(['duplicates', '.', '--output', 'test.json'])
        assert args.command == 'duplicates'
        assert args.path == '.'
        assert args.output == 'test.json'
        
        print("✅ CLI argument parsing test passed")
        
    except Exception as e:
        print(f"❌ CLI integration test failed: {e}")
        raise


if __name__ == "__main__":
    print("Running codebase cleanup integration tests...")
    
    try:
        test_full_cleanup_workflow()
        test_cli_integration()
        print("\n🎉 All integration tests passed!")
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        raise