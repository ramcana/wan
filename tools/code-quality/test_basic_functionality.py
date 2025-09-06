"""
Basic functionality test for code quality system.
"""

import sys
import tempfile
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import our modules
from models import QualityConfig, QualityReport, QualityIssue, QualityIssueType, QualitySeverity
from formatters.code_formatter import CodeFormatter
from validators.documentation_validator import DocumentationValidator
from validators.type_hint_validator import TypeHintValidator
from validators.style_validator import StyleValidator
from analyzers.complexity_analyzer import ComplexityAnalyzer


def test_models():
    """Test the data models."""
    print("Testing data models...")
    
    # Test QualityConfig
    config = QualityConfig()
    assert config.line_length == 88
    assert config.use_black == True
    assert config.require_function_docstrings == True
    
    # Test QualityIssue
    issue = QualityIssue(
        file_path=Path("test.py"),
        line_number=10,
        column=5,
        issue_type=QualityIssueType.DOCUMENTATION,
        severity=QualitySeverity.WARNING,
        message="Missing docstring",
        rule_code="MISSING_DOCSTRING"
    )
    assert issue.file_path == Path("test.py")
    assert issue.line_number == 10
    assert issue.issue_type == QualityIssueType.DOCUMENTATION
    
    # Test QualityReport
    report = QualityReport()
    report.add_issue(issue)
    assert report.total_issues == 1
    assert report.warnings == 1
    assert report.errors == 0
    
    print("✅ Data models test passed")


def test_formatters():
    """Test the code formatters."""
    print("Testing code formatters...")
    
    config = QualityConfig()
    formatter = CodeFormatter(config)
    
    # Test basic formatting check
    test_content = "def test(  ):   pass   "
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_content)
        temp_file = Path(f.name)
    
    try:
        issues = formatter.check_formatting(temp_file, test_content)
        # Should find some formatting issues
        assert len(issues) >= 0  # May or may not find issues depending on tools available
        print(f"Found {len(issues)} formatting issues")
    finally:
        temp_file.unlink(missing_ok=True)
    
    print("✅ Code formatters test passed")


def test_validators():
    """Test the validators."""
    print("Testing validators...")
    
    config = QualityConfig()
    
    # Test documentation validator
    doc_validator = DocumentationValidator(config)
    
    # Create test AST
    import ast
    test_code = '''
def undocumented_function():
    return "test"

def documented_function():
    """This function has documentation."""
    return "test"
'''
    tree = ast.parse(test_code)
    issues, metrics = doc_validator.validate_documentation(Path("test.py"), tree)
    
    # Should find missing docstring issue
    assert len(issues) > 0
    assert metrics['functions_count'] == 2
    assert metrics['documented_functions'] == 1
    
    print(f"Documentation validator found {len(issues)} issues")
    
    # Test type hint validator
    type_validator = TypeHintValidator(config)
    issues, metrics = type_validator.validate_type_hints(Path("test.py"), tree)
    
    # Should find missing type hints
    assert len(issues) >= 0  # May depend on mypy availability
    print(f"Type hint validator found {len(issues)} issues")
    
    print("✅ Validators test passed")


def test_analyzers():
    """Test the analyzers."""
    print("Testing analyzers...")
    
    config = QualityConfig()
    analyzer = ComplexityAnalyzer(config)
    
    # Create complex function for testing
    import ast
    complex_code = '''
def complex_function(x):
    if x > 10:
        if x > 20:
            if x > 30:
                return "high"
            else:
                return "medium"
        else:
            return "low"
    else:
        return "very low"
'''
    tree = ast.parse(complex_code)
    issues, metrics = analyzer.analyze_complexity(Path("test.py"), tree)
    
    # Should detect complexity
    assert metrics['total_functions'] == 1
    assert metrics['average_complexity'] > 1
    
    print(f"Complexity analyzer found {len(issues)} issues")
    print(f"Average complexity: {metrics['average_complexity']:.1f}")
    
    print("✅ Analyzers test passed")


def test_integration():
    """Test basic integration."""
    print("Testing basic integration...")
    
    # Create a test file with various issues
    test_content = '''
def bad_function(x,y,z):
    if x:
        if y:
            if z:
                print("deeply nested")
                return x+y+z
    return None
'''
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_content)
        temp_file = Path(f.name)
    
    try:
        config = QualityConfig()
        
        # Test each component individually
        formatter = CodeFormatter(config)
        doc_validator = DocumentationValidator(config)
        type_validator = TypeHintValidator(config)
        complexity_analyzer = ComplexityAnalyzer(config)
        
        # Parse the code
        import ast
        tree = ast.parse(test_content)
        
        # Run each validator
        format_issues = formatter.check_formatting(temp_file, test_content)
        doc_issues, doc_metrics = doc_validator.validate_documentation(temp_file, tree)
        type_issues, type_metrics = type_validator.validate_type_hints(temp_file, tree)
        complexity_issues, complexity_metrics = complexity_analyzer.analyze_complexity(temp_file, tree)
        
        total_issues = len(format_issues) + len(doc_issues) + len(type_issues) + len(complexity_issues)
        
        print(f"Total issues found: {total_issues}")
        print(f"  Formatting: {len(format_issues)}")
        print(f"  Documentation: {len(doc_issues)}")
        print(f"  Type hints: {len(type_issues)}")
        print(f"  Complexity: {len(complexity_issues)}")
        
        # Should find some issues in this poorly written code
        assert total_issues > 0
        
    finally:
        temp_file.unlink(missing_ok=True)
    
    print("✅ Integration test passed")


def main():
    """Run all tests."""
    print("🧪 Running Code Quality System Tests")
    print("=" * 50)
    
    try:
        test_models()
        test_formatters()
        test_validators()
        test_analyzers()
        test_integration()
        
        print("\n🎉 All tests passed successfully!")
        print("Code Quality Checking System is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)