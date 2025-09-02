"""
Simple test to verify the code review system works
"""

import os
import sys
import tempfile
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Now import the modules
from tools.code_review.code_reviewer import CodeReviewer, ReviewSeverity, IssueCategory
from tools.code_review.refactoring_engine import RefactoringEngine, RefactoringType


def test_basic_functionality():
    """Test basic functionality of the code review system"""
    print("Testing Code Review System...")
    
    # Create a test file with some issues
    test_code = '''
def complex_function(a, b, c, d, e):
    if a > 0:
        if b > 0:
            if c > 0:
                if d > 0:
                    if e > 0:
                        result = a + b + c + d + e
                        for i in range(10):
                            for j in range(10):
                                result += i * j
                        return result
    return 0

def function_without_docstring():
    return "no docs"

def dangerous_function():
    user_input = "print('hello')"
    eval(user_input)
    return "dangerous"
'''
    
    # Write test code to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_code)
        temp_file = f.name
    
    try:
        # Test Code Reviewer
        print("1. Testing Code Reviewer...")
        reviewer = CodeReviewer()
        issues = reviewer.review_file(temp_file)
        
        print(f"   Found {len(issues)} issues")
        for issue in issues[:3]:  # Show first 3 issues
            print(f"   - {issue.severity.value}: {issue.message}")
        
        assert len(issues) > 0, "Should find issues in test code"
        print("   ✅ Code Reviewer working")
        
        # Test Refactoring Engine
        print("2. Testing Refactoring Engine...")
        engine = RefactoringEngine()
        suggestions = engine.analyze_file(temp_file)
        
        print(f"   Found {len(suggestions)} refactoring suggestions")
        for suggestion in suggestions[:2]:  # Show first 2 suggestions
            print(f"   - Priority {suggestion.priority}: {suggestion.title}")
        
        print("   ✅ Refactoring Engine working")
        
        print("\n🎉 Basic functionality test passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise
    finally:
        # Clean up
        os.unlink(temp_file)


def test_cli_import():
    """Test that CLI can be imported"""
    print("3. Testing CLI import...")
    try:
        from tools.code_review.cli import main
        print("   ✅ CLI import successful")
    except Exception as e:
        print(f"   ❌ CLI import failed: {e}")
        raise


if __name__ == "__main__":
    test_basic_functionality()
    test_cli_import()
    print("\n✅ All tests completed successfully!")