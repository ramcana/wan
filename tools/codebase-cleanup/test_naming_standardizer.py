"""
Test suite for naming standardization functionality
"""

import tempfile
from pathlib import Path

from naming_standardizer import NamingStandardizer


def test_naming_standardizer():
    """Test basic naming standardization functionality"""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test Python file with naming violations
        test_file = temp_path / "BadFileName.py"
        test_content = '''
class badClassName:
    def BadMethodName(self):
        return "bad"

def BadFunctionName():
    BadVariableName = "bad"
    return BadVariableName

class GoodClassName:
    def good_method_name(self):
        return "good"

def good_function_name():
    good_variable_name = "good"
    return good_variable_name
'''
        test_file.write_text(test_content)
        
        # Create test JavaScript file
        js_file = temp_path / "bad-js-file.js"
        js_content = '''
class bad_class_name {
    constructor() {}
}

function Bad_Function_Name() {
    return "bad";
}

class GoodClassName {
    constructor() {}
}

function goodFunctionName() {
    return "good";
}
'''
        js_file.write_text(js_content)
        
        # Initialize standardizer
        standardizer = NamingStandardizer(str(temp_path))
        
        # Run analysis
        report = standardizer.analyze_naming_conventions()
        
        # Verify results
        print(f"Files analyzed: {report.total_files_analyzed}")
        print(f"Naming violations: {len(report.violations)}")
        print(f"Inconsistent patterns: {len(report.inconsistent_patterns)}")
        print(f"Organization suggestions: {len(report.organization_suggestions)}")
        
        # Should find violations
        assert len(report.violations) > 0, "Should find naming violations"
        
        # Check specific violations
        violation_names = [v.name for v in report.violations]
        assert "badClassName" in violation_names, "Should find badClassName violation"
        assert "BadMethodName" in violation_names, "Should find BadMethodName violation"
        assert "BadFunctionName" in violation_names, "Should find BadFunctionName violation"
        
        # Check violation types
        class_violations = [v for v in report.violations if v.element_type == 'class']
        function_violations = [v for v in report.violations if v.element_type == 'function']
        file_violations = [v for v in report.violations if v.element_type == 'file']
        
        assert len(class_violations) > 0, "Should find class naming violations"
        assert len(function_violations) > 0, "Should find function naming violations"
        assert len(file_violations) > 0, "Should find file naming violations"
        
        print("✓ Naming standardizer test passed")


def test_naming_convention_detection():
    """Test naming convention detection"""
    
    standardizer = NamingStandardizer(".")
    
    # Test convention detection
    test_cases = [
        ("snake_case_name", "snake_case"),
        ("camelCaseName", "camelCase"),
        ("PascalCaseName", "PascalCase"),
        ("kebab-case-name", "kebab-case"),
        ("UPPER_CASE_NAME", "UPPER_CASE"),
        ("badName123", "unknown")
    ]
    
    for name, expected_convention in test_cases:
        detected = standardizer._detect_convention(name)
        print(f"{name} -> {detected} (expected: {expected_convention})")
        
        if expected_convention != "unknown":
            assert detected == expected_convention, f"Expected {expected_convention}, got {detected}"
    
    print("✓ Convention detection test passed")


def test_naming_conversion():
    """Test naming convention conversion"""
    
    standardizer = NamingStandardizer(".")
    
    test_cases = [
        ("BadFunctionName", "snake_case", "bad_function_name"),
        ("bad_function_name", "camelCase", "badFunctionName"),
        ("bad_function_name", "PascalCase", "BadFunctionName"),
        ("BadFunctionName", "kebab-case", "bad-function-name"),
        ("camelCaseName", "snake_case", "camel_case_name"),
        ("kebab-case-name", "PascalCase", "KebabCaseName")
    ]
    
    for original, target_convention, expected in test_cases:
        converted = standardizer._convert_to_convention(original, target_convention)
        print(f"{original} -> {converted} (expected: {expected})")
        assert converted == expected, f"Expected {expected}, got {converted}"
    
    print("✓ Naming conversion test passed")


def test_file_purpose_detection():
    """Test file purpose detection"""
    
    standardizer = NamingStandardizer(".")
    
    test_cases = [
        ("test_module.py", "test"),
        ("demo_script.py", "demo"),
        ("example_usage.py", "example"),
        ("temp_file.py", "temporary"),
        ("backup_data.json", "backup"),
        ("old_version.py", "deprecated"),
        ("config.json", "configuration"),
        ("README.md", "documentation"),
        ("setup.sh", "script"),
        ("utils.py", "utility"),
        ("normal_file.py", "unknown")
    ]
    
    for filename, expected_purpose in test_cases:
        detected = standardizer._detect_file_purpose(filename)
        print(f"{filename} -> {detected} (expected: {expected_purpose})")
        assert detected == expected_purpose, f"Expected {expected_purpose}, got {detected}"
    
    print("✓ File purpose detection test passed")


def test_organization_suggestions():
    """Test file organization suggestions"""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create scattered test files
        (temp_path / "test_module1.py").write_text("# test file 1")
        (temp_path / "subdir").mkdir()
        (temp_path / "subdir" / "test_module2.py").write_text("# test file 2")
        (temp_path / "another_dir").mkdir()
        (temp_path / "another_dir" / "test_module3.py").write_text("# test file 3")
        
        # Create scattered demo files
        (temp_path / "demo_script1.py").write_text("# demo file 1")
        (temp_path / "subdir" / "demo_script2.py").write_text("# demo file 2")
        
        standardizer = NamingStandardizer(str(temp_path))
        report = standardizer.analyze_naming_conventions()
        
        print(f"Organization suggestions: {len(report.organization_suggestions)}")
        for suggestion in report.organization_suggestions:
            print(f"  {suggestion.current_path} -> {suggestion.suggested_path}")
            print(f"    Reason: {suggestion.reason}")
        
        # Should suggest consolidating test files
        test_suggestions = [s for s in report.organization_suggestions if 'test' in s.reason.lower()]
        assert len(test_suggestions) > 0, "Should suggest consolidating test files"
        
        print("✓ Organization suggestions test passed")


def test_backup_functionality():
    """Test backup creation for naming fixes"""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test file
        test_file = temp_path / "test_backup.py"
        test_content = '''
class BadClassName:
    def bad_method(self):
        return "test"
'''
        test_file.write_text(test_content)
        
        standardizer = NamingStandardizer(str(temp_path))
        
        # Create backup
        files_to_backup = [str(test_file)]
        backup_path = standardizer.create_backup(files_to_backup)
        
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
    print("Running naming standardizer tests...")
    
    try:
        test_naming_standardizer()
        test_naming_convention_detection()
        test_naming_conversion()
        test_file_purpose_detection()
        test_organization_suggestions()
        test_backup_functionality()
        print("\n✅ All naming standardizer tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise